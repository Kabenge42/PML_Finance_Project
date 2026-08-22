"""Tests for the Max-and-Smooth screening backend.

Three things need to be true for the screener to be worth trusting:

1. **The Max step is arithmetically right.** Against a simulated panel whose
   truth is known, ``kappa_level`` must recover ``sqrt(w_L)`` -- the permanent
   level's SHARE of within-name variance, not ``sigma_level`` itself -- and the
   drift coefficients must come back inside their posterior uncertainty.
2. **It discriminates.** On a panel simulated WITH a permanent level the
   ``level_off`` arm must score worse than the baseline; on a panel simulated
   without one it must not.
3. **It refuses what it cannot see.** An arm that changes the data-level
   covariance must raise, because the pseudo-observations were conditioned on it.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
    KalmanModelConfig,
    _simulate_panel,
)
from probabilistic_ml_model.pymc_models._max_and_smooth import (
    COVARIANCE_FIELDS,
    assert_arm_is_screenable,
    build_pseudo_model,
    gaussian_likelihood_approximation,
)

BETA_TRUE = np.array([0.30, -0.20, 0.10, 0.05])
TAU = (1.45, 1.20, 1.05)
ELL = 105.0


class _PosteriorShim:
    """Minimal stand-in whose posterior means ARE the simulation truth.

    Isolates the Max-and-Smooth arithmetic from how well the full model fits.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self.posterior = ds


def _truth_posterior(cfg, n, s_level, s_state, s_obs):
    days = cfg.time_grid_days
    T = len(days)
    tau = np.array([*TAU, 1.0])
    tot2 = s_level ** 2 + s_state ** 2 + s_obs ** 2
    w_L, w_S, w_O = s_level ** 2 / tot2, s_state ** 2 / tot2, s_obs ** 2 / tot2
    K = np.exp(-np.abs(days[:, None] - days[None, :]) / ELL)
    A = (w_L * np.ones((T, T)) + w_S * K + w_O * np.eye(T)) * np.outer(tau, tau)
    sig = np.sqrt(tot2)

    def _v(a):
        return np.broadcast_to(np.asarray(a), (1, 1) + np.shape(a))

    return _PosteriorShim(
        xr.Dataset(
            {
                "within_name_cov": (("chain", "draw", "time", "time_b"), _v(A)),
                "sigma_isin": (("chain", "draw", "isin"), _v(np.full(n, sig))),
                "alpha_time": (("chain", "draw", "time"), _v(np.zeros(T))),
                "sigma_time": (("chain", "draw", "time"), _v(tau)),
                "sigma_level": (("chain", "draw"), np.array([[s_level]])),
                "sigma_total": (("chain", "draw"), np.array([[sig]])),
            }
        )
    )


def _case(n_isin=1200, s_level=0.55, s_state=0.75, s_obs=0.35, seed=11):
    panel, _ = _simulate_panel(
        n_isin=n_isin, rho_slope_true=0.0, vol_delta_true=0.0,
        signal_exponent_true=0.0, sigma_level_true=s_level,
        sigma_state_true=s_state, sigma_obs_true=s_obs, ell_true=ELL,
        tau_true=TAU, seed=seed,
    )
    cfg = KalmanModelConfig(
        lookbacks=("1y", "3m", "1w"), likelihood="normal",
        group_effects=(), enable_signal_scaling=False,
    )
    idata = _truth_posterior(cfg, len(panel.isins), s_level, s_state, s_obs)
    return panel, cfg, idata


def _fit(pseudo, cfg, seed=7):
    import pymc as pm

    with build_pseudo_model(pseudo, cfg):
        return pm.sample(draws=400, tune=400, chains=2, cores=1,
                         target_accept=0.9, random_seed=seed, progressbar=False)


@pytest.mark.slow
def test_max_step_keeps_every_name_and_reports_one_pattern():
    panel, cfg, idata = _case()
    ps = gaussian_likelihood_approximation(panel, idata, cfg)
    assert len(ps) == len(panel.isins)
    assert ps.diagnostics["n_dropped"] == 0
    # A fully observed simulated panel has exactly one missingness pattern, which
    # is what makes the Max step a single 4x4 solve rather than 1200 of them.
    assert ps.diagnostics["n_missingness_patterns"] == 1
    assert np.all(np.isfinite(ps.m_hat))
    assert np.all(ps.var_obs > 0)


@pytest.mark.slow
def test_smooth_recovers_beta_and_the_level_share():
    panel, cfg, idata = _case()
    ps = gaussian_likelihood_approximation(panel, idata, cfg)
    post = _fit(ps, cfg)

    b = post.posterior["beta"].values.reshape(-1, len(BETA_TRUE))
    z = (b.mean(axis=0) - BETA_TRUE) / b.std(axis=0)
    assert np.all(np.abs(z) < 3.0), f"beta z-scores {z}"

    # kappa estimates sqrt(w_L) -- the level's SHARE of within-name variance --
    # NOT sigma_level. Comparing it to 0.55 is the easy mistake: the pseudo
    # observation already carries sigma_i and the loading c in `level_scale`.
    w_l = ps.diagnostics["w_level"]
    kap = post.posterior["kappa_level"].values.reshape(-1)
    z_k = (kap.mean() - np.sqrt(w_l)) / kap.std()
    assert abs(z_k) < 3.0, f"kappa {kap.mean():.4f} vs sqrt(w_L) {np.sqrt(w_l):.4f}"


@pytest.mark.slow
def test_level_off_is_penalised_only_when_a_level_exists():
    """The screener has to be able to tell the two panels apart."""
    import arviz as az

    def _contrast(s_level):
        panel, cfg, idata = _case(s_level=s_level, seed=23)
        ps = gaussian_likelihood_approximation(panel, idata, cfg)
        arms = {
            "baseline": cfg,
            "level_off": KalmanModelConfig(
                **{**cfg.__dict__, "enable_isin_level": False}
            ),
        }
        fits = {}
        for name, arm_cfg in arms.items():
            post = _fit(ps, arm_cfg, seed=31)
            import pymc as pm
            with build_pseudo_model(ps, arm_cfg) as m:
                pm.compute_log_likelihood(post, model=m, progressbar=False)
            fits[name] = post
        cmp = az.compare(fits)
        return cmp.index[0]

    # Not 0.0: _simulate_panel takes a logit of the level SHARE, so an exactly
    # zero level is a math-domain error. 0.02 against a structured sd of ~0.93 is
    # a level share of 0.0007 -- nothing for the free parameter to buy.
    assert _contrast(0.55) == "baseline", "a real level must beat dropping it"
    assert _contrast(0.02) == "level_off", "no level: the free arm must not win"


def test_covariance_arms_are_refused():
    base = KalmanModelConfig()
    for fname, value in (
        ("likelihood", "normal"),
        ("rho_scale_buckets", 5),
        ("time_scale_applies_to", "observation"),
    ):
        arm = KalmanModelConfig(**{**base.__dict__, fname: value})
        with pytest.raises(ValueError, match=fname):
            assert_arm_is_screenable(base, arm, fname)


def test_latent_arms_are_admitted():
    base = KalmanModelConfig()
    for fname, value in (
        ("enable_isin_level", False),
        ("group_effects", ("trading_region", "sector", "country", "industry")),
        ("beta_prior_scale", 2.0),
    ):
        arm = KalmanModelConfig(**{**base.__dict__, fname: value})
        assert_arm_is_screenable(base, arm, fname)  # must not raise


def test_level_off_is_not_accidentally_a_covariance_field():
    """The whole partition rests on this, so state it as a test."""
    assert "enable_isin_level" not in COVARIANCE_FIELDS
    assert "group_effects" not in COVARIANCE_FIELDS
    # ...while the legs the Max step actually froze are all in there.
    for f in ("likelihood", "rho_scale_buckets", "enable_ou_state", "lookbacks"):
        assert f in COVARIANCE_FIELDS
