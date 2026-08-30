"""Properties of the forward-return forecast layer.

Each test pins a specific defect or a specific claim made in
``KalmanForecast``'s docstring, in the idiom the risk-book tests established:
synthetic inputs, a seeded RNG, and assertions on **mathematical identities**
rather than on golden values.

The claims under test, and why each matters:

* The terminal log return has mean ``mu_log`` and, conditional on a posterior draw,
  standard deviation ``sigma_log``. If either drifts, every exported ``er_*`` column
  silently changes scale.
* Per-name marginals are invariant to ``factor_share``. This is the property that
  lets the factor structure be switched on without re-scaling ``er_sd`` — and
  therefore ``exp_vol``, which ``compute_cvar_aware_book`` asserts against ``er_sd``
  on every call.
* ``pooled_returns`` is shape- and pooling-compatible with ``ScreenDraws``, so a
  ``ForecastDraws`` is a drop-in ``return_draws``.
* ISIN labels are carried, never implied by position. A positional join between the
  screen and the draws attributed every risk column to the wrong name once already;
  the length-only guard could not see it.
* The step grid covers the horizon exactly once, with no stub step.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from probabilistic_ml_model.pymc_models.KalmanForecast import (
    ForecastConfig,
    ForecastDraws,
    ForecastInputs,
    simulate_forecast,
    summarize_forecast,
)

SEED = 20260825


def _make_inputs(
        n_isin: int = 40,
        n_samples: int = 300,
        *,
        constant: bool = False,
        seed: int = SEED,
) -> ForecastInputs:
    """Synthetic posterior inputs.

    ``constant=True`` pins ``mu_log`` and ``sigma_log`` to a single value across
    samples, which isolates the *conditional* moments: with parameter uncertainty
    removed the terminal spread must equal ``sigma_log`` exactly.
    """
    rng = np.random.default_rng(seed)
    if constant:
        mu = np.full((n_isin, n_samples), 0.15)
        sigma = np.full((n_isin, n_samples), 0.30)
    else:
        mu = rng.normal(0.15, 0.10, size=(n_isin, n_samples))
        sigma = np.abs(rng.normal(0.30, 0.05, size=(n_isin, n_samples)))
    return ForecastInputs(
        isins=np.array([f"TEST{i:08d}" for i in range(n_isin)]),
        mu_log=mu,
        sigma_log=sigma,
        nu=np.full(n_samples, 11.0),
        group_index={
            "sector": rng.integers(0, 5, n_isin),
            "trading_region": rng.integers(0, 3, n_isin),
        },
        ou_length_scale_days=81.2,
    )


# ---------------------------------------------------------------------------
# The step grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("horizon", "step", "expect_steps"),
    [(365, 91, 4), (365, 120, 3), (365, 365, 1), (365, 30, 12), (180, 91, 2)],
)
def test_step_grid_covers_the_horizon_exactly_once(horizon, step, expect_steps):
    """No stub step, no overshoot, and the fractions are a partition of the horizon.

    ``ceil(365 / 91)`` is 5, whose final step is one day long: it carries almost no
    variance, breaks the intended like-for-like with the AR simulator's four periods,
    and makes the grid read as finer than it is.
    """
    cfg = ForecastConfig(horizon_days=horizon, step_days=step)
    assert cfg.n_steps == expect_steps
    assert cfg.time_days[-1] == pytest.approx(float(horizon))
    assert np.all(np.diff(cfg.time_days) > 0)
    assert np.all(cfg.step_fractions > 0)
    assert cfg.step_fractions.sum() == pytest.approx(1.0, rel=1e-12)


def test_default_grid_matches_the_ar_simulators_period_count():
    """The default is four steps because ``mc_horizon = 4``; the contrast is like-for-like."""
    assert ForecastConfig().n_steps == 4


# ---------------------------------------------------------------------------
# Terminal moments
# ---------------------------------------------------------------------------


def test_terminal_log_mean_is_the_decision_latent():
    """``E[sum_t log r] == mu_log``: the drift is the model's own latent, undistorted."""
    inputs = _make_inputs(constant=True, n_samples=1)
    draws = simulate_forecast(inputs, ForecastConfig(n_scenarios=40_000))
    terminal_log = np.log1p(draws.terminal)
    assert terminal_log.mean() == pytest.approx(0.15, abs=5e-3)


def test_conditional_terminal_sd_is_sigma_isin():
    """``sd[sum_t log r | s] == sigma_log`` — the fitted per-name scale, exactly.

    Held conditional on one posterior draw. Marginally the terminal spread is WIDER,
    because it also carries the posterior spread of ``mu_log``; that is intended and
    is what makes parameter uncertainty reach the decision.
    """
    inputs = _make_inputs(constant=True, n_samples=1)
    draws = simulate_forecast(inputs, ForecastConfig(n_scenarios=60_000))
    terminal_log = np.log1p(draws.terminal)
    assert terminal_log.std(axis=1).mean() == pytest.approx(0.30, abs=5e-3)


def test_marginal_terminal_sd_exceeds_the_conditional_one():
    """Parameter uncertainty widens the terminal distribution rather than vanishing."""
    conditional = simulate_forecast(
        _make_inputs(constant=True, n_samples=1), ForecastConfig(n_scenarios=20_000)
    )
    marginal = simulate_forecast(
        _make_inputs(constant=False), ForecastConfig(n_scenarios=20_000)
    )
    assert (
        np.log1p(marginal.terminal).std(axis=1).mean()
        > np.log1p(conditional.terminal).std(axis=1).mean()
    )


def test_terminal_is_the_compounded_path():
    """``terminal`` is the product of the per-step gross returns, not their sum."""
    draws = simulate_forecast(_make_inputs(n_isin=6), ForecastConfig(n_scenarios=200))
    compounded = np.prod(1.0 + draws.paths, axis=2) - 1.0
    np.testing.assert_allclose(draws.terminal, compounded, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# The invariance the factor structure rests on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("share", [0.0, 0.35, 0.9])
def test_per_name_marginals_are_invariant_to_factor_share(share):
    """The common factor changes WHICH names move together, not how far each moves.

    Without this, switching the factor structure on would re-scale ``er_sd`` for
    every name, and with it ``exp_vol``, ``expected_sharpe_ratio``, ``tail_risk`` and
    the sized book — a silent revaluation of the whole export.
    """
    inputs = _make_inputs(constant=True, n_samples=1)
    draws = simulate_forecast(
        inputs, ForecastConfig(n_scenarios=60_000, factor_share=share)
    )
    assert np.log1p(draws.terminal).std(axis=1).mean() == pytest.approx(0.30, abs=8e-3)


def test_factor_share_raises_cross_sectional_correlation():
    """The joint distribution is what the factor share is FOR.

    Independent shocks make diversification free: pooling names averages their
    idiosyncratic risk to nothing, which is why a long book can report a positive
    expected shortfall. The correlation must rise monotonically with the share.
    """
    inputs = _make_inputs(n_isin=60)

    def avg_corr(share: float) -> float:
        draws = simulate_forecast(
            inputs, ForecastConfig(n_scenarios=20_000, factor_share=share)
        )
        corr = np.corrcoef(draws.terminal)
        n = corr.shape[0]
        return float((corr.sum() - np.trace(corr)) / (n * n - n))

    low, mid, high = avg_corr(0.0), avg_corr(0.35), avg_corr(0.9)
    assert low < mid < high
    assert low == pytest.approx(0.0, abs=0.02)


def test_zero_factor_share_reproduces_independent_shocks():
    """``factor_share=0`` is the AR simulator's cross-sectional assumption exactly."""
    draws = simulate_forecast(
        _make_inputs(n_isin=50), ForecastConfig(n_scenarios=20_000, factor_share=0.0)
    )
    assert draws.factor_draws is None
    assert draws.factor_share == 0.0


# ---------------------------------------------------------------------------
# Interoperation with the risk book
# ---------------------------------------------------------------------------


def test_pooled_returns_matches_the_screendraws_contract():
    """Shape AND pooling, so a ForecastDraws is a drop-in ``return_draws``."""
    cfg = ForecastConfig(n_scenarios=500)
    draws = simulate_forecast(_make_inputs(n_isin=12), cfg)
    assert draws.pooled_returns.shape == (12, cfg.n_scenarios * cfg.n_steps)
    np.testing.assert_allclose(
        draws.pooled_returns,
        draws.paths.reshape(12, -1),
        rtol=0,
        atol=0,
    )


def test_exp_vol_equals_er_sd_on_the_forecast_draws():
    """The identity ``compute_cvar_aware_book`` asserts on every call.

    Both are the standard deviation of the same pooled array. It holds only as long
    as the array feeding ``summarize_mc_returns`` is the array feeding the risk
    columns; a permutation between them broke it once, correlating -0.007 while the
    SORTED values still matched to 1e-9.
    """
    draws = simulate_forecast(_make_inputs(n_isin=20), ForecastConfig(n_scenarios=400))
    summary = summarize_forecast(draws)
    np.testing.assert_allclose(
        summary["er_sd"].to_numpy(),
        draws.pooled_returns.std(axis=1),
        rtol=1e-12,
    )
    assert list(summary["isin"]) == list(draws.isins)


def test_summarize_terminal_differs_from_pooled_and_says_so():
    """Pooling per-step marginals is not the horizon distribution.

    Both are legitimate; the column names do not distinguish them, which is why the
    ``terminal`` flag exists and why a frame built with it must record that.
    """
    draws = simulate_forecast(_make_inputs(n_isin=15), ForecastConfig(n_scenarios=4000))
    pooled = summarize_forecast(draws, terminal=False)
    terminal = summarize_forecast(draws, terminal=True)
    assert (terminal["er_sd"] > pooled["er_sd"]).all()


# ---------------------------------------------------------------------------
# Identity and reproducibility
# ---------------------------------------------------------------------------


def test_isins_follow_a_permuted_panel():
    """Labels travel with the rows they describe.

    Permuting the input must permute the output identically. The class of bug this
    pins attributed every risk column to the wrong name while every length check and
    twenty-two gates still passed.
    """
    n = 30
    # Each name gets a DISTINCT drift and a negligible scale, so its terminal mean
    # identifies it to far better than Monte-Carlo error. Comparing two noisy
    # simulations name-by-name would only test the RNG.
    mu = np.linspace(0.01, 0.30, n).reshape(n, 1)
    inputs = ForecastInputs(
        isins=np.array([f"TEST{i:08d}" for i in range(n)]),
        mu_log=mu,
        sigma_log=np.full((n, 1), 1e-4),
        nu=np.array([11.0]),
        group_index={"sector": np.arange(n) % 5},
        ou_length_scale_days=81.2,
    )
    cfg = ForecastConfig(n_scenarios=200)
    base = simulate_forecast(inputs, cfg)

    order = np.random.default_rng(7).permutation(n)
    permuted = simulate_forecast(
        ForecastInputs(
            isins=inputs.isins[order],
            mu_log=inputs.mu_log[order],
            sigma_log=inputs.sigma_log[order],
            nu=inputs.nu,
            group_index={k: v[order] for k, v in inputs.group_index.items()},
            ou_length_scale_days=inputs.ou_length_scale_days,
        ),
        cfg,
    )
    assert list(permuted.isins) == list(base.isins[order])
    base_mean = dict(zip(base.isins, base.terminal.mean(axis=1)))
    perm_mean = dict(zip(permuted.isins, permuted.terminal.mean(axis=1)))
    for isin, want in base_mean.items():
        assert perm_mean[isin] == pytest.approx(want, abs=1e-3)


def test_same_seed_reproduces_and_a_different_seed_does_not():
    """Determinism is a property the AR simulator has and this must not lose."""
    inputs = _make_inputs(n_isin=10)
    cfg = ForecastConfig(n_scenarios=200)
    a = simulate_forecast(inputs, cfg)
    b = simulate_forecast(inputs, cfg)
    c = simulate_forecast(inputs, replace(cfg, random_seed=cfg.random_seed + 1))
    np.testing.assert_array_equal(a.terminal, b.terminal)
    assert not np.array_equal(a.terminal, c.terminal)


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "nonesuch"},
        {"horizon_days": 0},
        {"step_days": 0},
        {"step_days": 400},          # exceeds the horizon
        {"n_scenarios": 1},
        {"factor_share": 1.0},       # no idiosyncratic risk left
        {"factor_share": -0.1},
        {"n_market_factors": 2},     # unidentified without a loadings model
        {"uplift_clip": (-1.5, 5.0)},
        {"uplift_clip": (5.0, -0.95)},
    ],
)
def test_invalid_config_is_rejected_at_construction(kwargs):
    with pytest.raises(ValueError):
        ForecastConfig(**kwargs)


def test_unbuilt_backends_name_their_blocker():
    """A NotImplementedError must say WHY, not merely that.

    ``statespace`` is blocked by a dependency conflict — pymc-extras pins pymc<6.3
    and pytensor<3.3, below the installed versions — and a reader hitting the error
    needs that rather than a bare stub message.
    """
    from probabilistic_ml_model.pymc_models import KalmanForecast as kf

    inputs = _make_inputs(n_isin=4, n_samples=10)
    for backend in ("pymc_forecast", "statespace"):
        with pytest.raises((NotImplementedError, ValueError)):
            kf.simulate_forecast(inputs, ForecastConfig(backend=backend))


def test_forecast_draws_rejects_mismatched_labels():
    """The dataclass refuses to hold draws it cannot label."""
    paths = np.zeros((3, 5, 2))
    with pytest.raises(ValueError):
        ForecastDraws(
            isins=np.array(["A", "B"]),
            paths=paths,
            terminal=np.zeros((3, 5)),
            time_days=np.array([1.0, 2.0]),
        )
