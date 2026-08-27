"""Properties of the probabilistic decision layer.

Wherever an analytic answer exists, the test asserts against **that** rather than
against a recorded output — a golden value would pin whatever the implementation did
on the day it was written, including its mistakes. So the Kelly solver is checked
against the closed-form ``(W*p - L*q) / (W*L)``, the volatility drag against half the
variance, and the risk measures against their definitional ordering.

Two tests encode findings rather than requirements:

``test_downside_deviation_stays_sensitive_where_tail_risk_goes_flat``
    pins recommendation 03 of the post-run analysis. The shipped ``tail_risk`` is
    ``max(-cvar05, k*er_sd, MIN_TAIL_RISK)``; for every name the book selects the
    first leg is negative, so the relative volatility floor becomes the entire
    denominator and the ratio stops responding to the tail at all. A replacement has
    to keep responding there, and this is the test that would fail if a future edit
    reintroduced the flat region.

``test_independent_draws_make_diversification_free``
    pins the reason the forecast layer grew a factor block. With cross-sectionally
    independent shocks a 25-name book's tail nearly vanishes, which is how a *long*
    book comes to report a positive expected shortfall. The test asserts the
    direction, not a magnitude.
"""
from __future__ import annotations

import numpy as np
import pytest

from probabilistic_ml_model.pymc_models.PortfolioOptimizationModel import (
    DEFAULT_KELLY_MULTIPLIER,
    LinearPositionLoss,
    downside_deviation,
    ergodicity_report,
    expected_loss,
    fractional_kelly,
    generative_expected_shortfall,
    generative_tail_risk,
    generative_var,
    kelly_fraction_from_draws,
    kelly_report,
    mean_variance_frontier,
    minimize_expected_loss,
    optimize_portfolio,
    terminal_wealth_curve,
)

SEED = 20260825


def _binary_draws(p: float, win: float, loss: float, n: int = 400_000, seed: int = SEED):
    """``n`` i.i.d. draws of a two-outcome bet, for checking against closed forms."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < p, win, -abs(loss))


def _joint_draws(n_isin: int, n_scen: int, *, rho: float, seed: int = SEED):
    """``(n_isin, n_scen)`` returns with a single common factor of correlation ``rho``.

    Marginals are identical for every ``rho``, exactly as the forecast layer's factor
    split guarantees, so any change in a portfolio statistic is attributable to the
    cross-section alone.
    """
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n_scen)
    idio = rng.standard_normal((n_isin, n_scen))
    z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    mu = rng.uniform(0.10, 0.30, size=(n_isin, 1))
    return mu + 0.25 * z


# ---------------------------------------------------------------------------
# A. The decision framework
# ---------------------------------------------------------------------------


def test_expected_loss_of_a_linear_position_is_minus_expected_profit():
    draws = np.array([0.10, -0.05, 0.02])
    loss = LinearPositionLoss(position_value=100_000.0)
    assert expected_loss(loss, draws, 1.0) == pytest.approx(-100_000.0 * draws.mean())


def test_minimize_expected_loss_picks_the_minimiser_and_shows_its_work():
    """The full table is part of the contract: an argmin alone cannot be audited."""
    draws = _binary_draws(0.76, 0.05, 0.15)
    chosen, table = minimize_expected_loss(
        LinearPositionLoss(100_000.0), draws, [0.0, 0.25, 0.5, 1.0]
    )
    assert chosen == 1.0  # positive edge, so the largest position minimises loss
    assert list(table.columns) == ["action", "expected_loss"]
    assert len(table) == 4
    assert table["expected_loss"].is_monotonic_increasing


def test_expected_loss_survives_non_finite_draws():
    """A name with unusable draws drops out of a ranking; it must not abort one."""
    loss = LinearPositionLoss(1.0)
    assert expected_loss(loss, np.array([0.1, np.nan, 0.3]), 1.0) == pytest.approx(-0.2)
    assert np.isnan(expected_loss(loss, np.array([np.nan, np.inf]), 1.0))


def test_minimize_expected_loss_rejects_an_empty_action_space():
    with pytest.raises(ValueError):
        minimize_expected_loss(LinearPositionLoss(1.0), np.array([0.1]), [])


# ---------------------------------------------------------------------------
# B. Generative risk measures
# ---------------------------------------------------------------------------


def test_risk_measures_are_ordered_by_construction():
    """``GVaR >= GES >= GTR``. Each is strictly more conservative than the last."""
    rng = np.random.default_rng(SEED)
    draws = rng.standard_t(5, size=200_000) * 0.2 + 0.08
    var = generative_var(draws)
    es = generative_expected_shortfall(draws)
    gtr = generative_tail_risk(draws)
    assert var >= es >= gtr
    assert gtr == pytest.approx(draws.min())


def test_generative_var_reads_the_quantile_not_a_z_score():
    """On a skewed distribution the empirical quantile and the Gaussian one differ.

    This is the whole point of computing risk from the predictive draws: a parametric
    VaR would report the same number for any distribution with these two moments.
    """
    rng = np.random.default_rng(SEED)
    draws = rng.standard_t(3, size=400_000) * 0.1
    empirical = generative_var(draws, prob=0.99)
    gaussian = -2.3263478740408408 * draws.std()
    assert empirical < gaussian  # fat left tail: the real quantile is worse


@pytest.mark.parametrize("prob", [0.0, 1.0, -0.1, 1.5])
def test_generative_var_rejects_an_impossible_confidence(prob):
    with pytest.raises(ValueError):
        generative_var(np.array([0.1, -0.2]), prob=prob)


def test_risk_measures_return_nan_on_empty_input():
    empty = np.array([np.nan, np.inf])
    assert np.isnan(generative_var(empty))
    assert np.isnan(generative_expected_shortfall(empty))
    assert np.isnan(generative_tail_risk(empty))
    assert np.isnan(downside_deviation(empty))


def test_downside_deviation_stays_sensitive_where_tail_risk_goes_flat():
    """Recommendation 03, as a property.

    Shift a return distribution steadily upward. Once its 5th percentile crosses
    zero, the shipped ``tail_risk`` is pinned to ``k * er_sd`` and stops moving with
    the tail — the state 25 of 25 book names are in. ``downside_deviation`` must keep
    falling across exactly that region, and must reach zero when no draw is negative.
    """
    rng = np.random.default_rng(SEED)
    base = rng.normal(0.20, 0.15, size=200_000)

    shifts = [0.0, 0.05, 0.10, 0.20]
    devs = [downside_deviation(base + s) for s in shifts]
    p05 = [float(np.quantile(base + s, 0.05)) for s in shifts]

    assert p05[0] < 0 < p05[1]                     # the floor starts binding here
    assert all(a > b for a, b in zip(devs, devs[1:]))  # strictly monotone throughout
    assert downside_deviation(np.array([0.1, 0.2, 0.3])) == 0.0


def test_downside_deviation_divides_by_every_draw_not_only_the_bad_ones():
    """A distribution with two bad draws in ten thousand is not as risky as one with two thousand.

    Dividing only by the count below the threshold would make the two identical.
    """
    rare = np.concatenate([np.full(9_998, 0.10), np.full(2, -0.50)])
    common = np.concatenate([np.full(8_000, 0.10), np.full(2_000, -0.50)])
    assert downside_deviation(rare) < downside_deviation(common)


# ---------------------------------------------------------------------------
# C. Ergodicity
# ---------------------------------------------------------------------------


def test_volatility_drag_is_about_half_the_variance():
    """The classical approximation, recovered from the draws rather than assumed."""
    rng = np.random.default_rng(SEED)
    report = ergodicity_report(rng.normal(0.08, 0.20, size=500_000))
    assert report["time_average"] < report["ensemble_average"]
    assert report["volatility_drag"] == pytest.approx(report["half_variance"], rel=0.10)


def test_ergodicity_report_counts_ruin_rather_than_averaging_through_it():
    """A path through total loss has no growth rate; it must not be averaged in."""
    draws = np.concatenate([np.full(900, 0.10), np.full(100, -1.0)])
    report = ergodicity_report(draws)
    assert report["prob_ruin"] == pytest.approx(0.10)
    assert np.isfinite(report["time_average"])
    assert report["time_average"] == pytest.approx(0.10)


def test_terminal_wealth_peaks_at_the_kelly_fraction():
    """The curve IS the argument for Kelly; the peak is the solver's answer."""
    draws = _binary_draws(0.55, 1.0, 1.0, n=2_000)
    f_star = kelly_fraction_from_draws(draws)
    curve = terminal_wealth_curve(draws, fractions=np.arange(0.0, 1.0, 0.01))
    peak = float(curve.loc[curve["terminal_wealth"].idxmax(), "fraction"])
    assert peak == pytest.approx(f_star, abs=0.02)


def test_overbetting_past_kelly_destroys_wealth():
    """Beyond the optimum the growth rate falls, and it keeps falling to ruin.

    The asymmetry is the reason :func:`fractional_kelly` refuses to scale up.
    """
    draws = _binary_draws(0.55, 1.0, 1.0, n=2_000)
    f_star = kelly_fraction_from_draws(draws)
    curve = terminal_wealth_curve(draws, fractions=np.array([f_star, 3.0 * f_star]))
    assert curve.loc[1, "terminal_wealth"] < 1e-3 * curve.loc[0, "terminal_wealth"]


def test_full_allocation_to_a_double_or_nothing_is_ruin():
    """``f = 1`` on a bet that can lose everything takes terminal wealth to zero."""
    draws = _binary_draws(0.55, 1.0, 1.0, n=500)
    curve = terminal_wealth_curve(draws, fractions=np.array([1.0]))
    assert curve.loc[0, "terminal_wealth"] == 0.0
    assert curve.loc[0, "log_growth"] == float("-inf")


# ---------------------------------------------------------------------------
# D. Capital allocation
# ---------------------------------------------------------------------------


def test_kelly_recovers_edge_over_odds_on_a_double_or_nothing():
    """``f* = p - q`` when the whole stake is at risk at even money.

    The target is computed from the REALISED win rate of the sample, not from the
    nominal 0.55. Against the nominal value this test would be measuring the RNG:
    ``f*`` moves about 2 units per unit of ``p``, so the binomial error on ``p``
    dominates any tolerance tight enough to catch a real solver defect.
    """
    draws = _binary_draws(0.55, 1.0, 1.0)
    p_hat = float((draws > 0).mean())
    solved = kelly_fraction_from_draws(draws)
    assert solved == pytest.approx(p_hat - (1.0 - p_hat), rel=1e-6)


def test_kelly_recovers_the_general_two_outcome_formula():
    """``f* = (W*p - L*q) / (W*L)`` when only part of the stake is at risk.

    The general form is the one the popular literature omits, and it is what the
    dashboard's ``(p*b - q) / b`` approximates after manufacturing an odds ratio.
    Here ``df*/dp`` is about 27, so the target again comes from the realised win
    rate rather than the nominal one.
    """
    win, loss = 0.05, 0.15
    draws = _binary_draws(0.76, win, loss)
    p_hat = float((draws > 0).mean())
    analytic = (win * p_hat - loss * (1.0 - p_hat)) / (win * loss)
    solved = kelly_fraction_from_draws(draws, max_fraction=10.0)
    assert solved == pytest.approx(analytic, rel=1e-6)


def test_kelly_is_zero_without_an_edge():
    """A non-positive expectation is sized at zero, never shorted from here.

    ``g(0) = E[r]``, so this falls out of the solver rather than being special-cased
    on top of it. Sizing the other side of a negative-edge bet is a different
    decision with different constraints.
    """
    assert kelly_fraction_from_draws(_binary_draws(0.40, 1.0, 1.0)) == 0.0
    assert kelly_fraction_from_draws(_binary_draws(0.50, 1.0, 1.0)) == pytest.approx(
        0.0, abs=5e-3
    )


def test_kelly_caps_when_no_scenario_loses():
    """An unbounded log-optimal fraction is a statement about the simulation."""
    assert kelly_fraction_from_draws(np.array([0.1, 0.2, 0.3])) == pytest.approx(1.0)


def test_fractional_kelly_scales_down_and_refuses_to_scale_up():
    assert fractional_kelly(0.10, multiplier=DEFAULT_KELLY_MULTIPLIER) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        fractional_kelly(0.10, multiplier=1.5)
    with pytest.raises(ValueError):
        fractional_kelly(0.10, multiplier=0.0)


def test_optimized_weights_are_a_capped_simplex():
    draws = _joint_draws(80, 4_000, rho=0.30)
    isins = np.array([f"TEST{i:08d}" for i in range(80)])
    book = optimize_portfolio(draws, isins, k_book=25, cap=0.10)
    w = book.weights
    assert len(w) == 25
    assert w.sum() == pytest.approx(1.0, rel=1e-9)
    assert (w >= 0).all()
    assert w.max() <= 0.10 + 1e-9
    assert set(w.index).issubset(set(isins))


def test_optimizer_beats_equal_weight_on_its_own_objective():
    """A minimum any optimiser must clear: it must not lose to the starting point."""
    from probabilistic_ml_model.pymc_models.PortfolioOptimizationModel import _log_growth

    draws = _joint_draws(60, 4_000, rho=0.30)
    isins = np.array([f"TEST{i:08d}" for i in range(60)])
    book = optimize_portfolio(draws, isins, k_book=20, cap=0.10)
    rows = [int(np.where(isins == i)[0][0]) for i in book.weights.index]
    held = draws[rows]
    equal = np.ones(len(rows)) / len(rows)
    assert book.summary["log_growth"] >= _log_growth(equal, held) - 1e-12


def test_independent_draws_make_diversification_free():
    """The finding that motivated the forecast layer's factor block.

    Same marginals, different cross-section. With independent shocks a 25-name book
    averages its idiosyncratic risk to nearly nothing, so the portfolio tail
    collapses and a LONG book reports a positive 1-in-100 outcome. That is an
    artefact of the generator, not a property of the names.
    """
    n, scen = 40, 8_000
    isins = np.array([f"TEST{i:08d}" for i in range(n)])
    indep = _joint_draws(n, scen, rho=0.0)
    joint = _joint_draws(n, scen, rho=0.35)

    # Marginals agree, so any difference below is cross-sectional.
    np.testing.assert_allclose(indep.std(axis=1), joint.std(axis=1), rtol=0.05)

    w = np.ones(25) / 25
    var_indep = generative_var(w @ indep[:25])
    var_joint = generative_var(w @ joint[:25])
    assert var_indep > var_joint            # independence understates the tail
    assert (w @ indep[:25]).std() < (w @ joint[:25]).std()


def test_portfolio_vol_prices_correlation_unlike_a_weighted_sum():
    """``Portfolio.summary['port_vol']`` is the sd of the portfolio return vector.

    ``RiskBook.summary['port_vol']`` is a weighted SUM of per-name volatilities,
    which assumes perfect correlation. The two agree only when everything moves
    together, and the difference is the diversification the draws contain.
    """
    draws = _joint_draws(50, 6_000, rho=0.30)
    isins = np.array([f"TEST{i:08d}" for i in range(50)])
    book = optimize_portfolio(draws, isins, k_book=25, cap=0.10)
    rows = [int(np.where(isins == i)[0][0]) for i in book.weights.index]
    weighted_sum = float((book.weights.to_numpy() * draws[rows].std(axis=1)).sum())
    assert book.summary["port_vol"] < weighted_sum


def test_empty_book_is_returned_rather_than_raised():
    """Every name loss-making: the routine reports an empty book, it does not abort."""
    draws = np.full((10, 500), -0.10)
    isins = np.array([f"TEST{i:08d}" for i in range(10)])
    book = optimize_portfolio(draws, isins, k_book=5, cap=0.20)
    assert book.summary["n_book"] == 0.0
    assert book.weights.empty
    assert np.isnan(book.summary["port_expected"])


def test_optimize_portfolio_requires_labels_that_match():
    draws = _joint_draws(10, 200, rho=0.2)
    with pytest.raises(ValueError):
        optimize_portfolio(draws, np.array(["A", "B"]), k_book=5)
    with pytest.raises(ValueError):
        optimize_portfolio(draws[0], np.array(["A"]), k_book=1)


def test_mean_variance_frontier_returns_a_labelled_max_sharpe_book():
    """The contrast arm still has to be correct to be worth contrasting against."""
    draws = _joint_draws(30, 3_000, rho=0.30)
    isins = np.array([f"TEST{i:08d}" for i in range(30)])
    mv = mean_variance_frontier(draws, isins, n_portfolios=2_000)
    w = mv["max_sharpe_weights"]
    assert w.sum() == pytest.approx(1.0, rel=1e-9)
    assert list(w.index) == list(isins)
    assert mv["max_sharpe"] == pytest.approx(
        (mv["max_sharpe_return"]) / mv["max_sharpe_vol"], rel=1e-9
    )
    assert np.isfinite(mv["sharpe"]).any()


# ---------------------------------------------------------------------------
# The export contract: nothing this module emits may be an infinity, and no
# quantity may leave under two names.
#
# Both were measured on run `6efb530d5881`, whose analytics write was refused by
# a blocking `export_finite` after a clean fit -- 0 divergences, R-hat 1.0023,
# min bulk ESS 1489 -- because of one column. These tests pin the encoding, not
# the values.
# ---------------------------------------------------------------------------


def _no_losing_scenario_draws(n=12, s=400, seed=0):
    """Draws where every scenario is a gain: the unbounded-Kelly case."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(0.08, 0.15, size=(n, s))) + 1e-3


def test_kelly_reports_no_finite_bound_as_null_plus_a_flag_not_as_inf():
    """`inf` is honest and unexportable; NULL + a boolean says the same thing.

    ``kelly_max_feasible`` carried ``+inf`` for 626 of 6,513 names on run
    ``6efb530d5881`` and was the whole of that run's blocking export failure.
    """
    winners = _no_losing_scenario_draws(n=1)[0]
    rep = kelly_report(winners)
    assert np.isnan(rep["kelly_max_feasible"])
    assert rep["kelly_unbounded"] == 1.0

    losers = np.array([0.20, -0.25, 0.10, -0.05])
    rep2 = kelly_report(losers)
    assert rep2["kelly_unbounded"] == 0.0
    # 1 + f*r > 0 for the worst draw: f < 1/0.25 = 4.
    assert rep2["kelly_max_feasible"] == pytest.approx(4.0)


def test_analytics_frame_carries_no_infinity_under_the_export_gate_semantics():
    """Exactly the check ``export_analytics``' finiteness gate runs.

    NaN is treated as 0.0 there — a NaN is a SQL NULL and always passed — so this
    asserts the absence of infinities specifically.
    """
    draws = _no_losing_scenario_draws()
    book = optimize_portfolio(draws, [f"X{i}" for i in range(draws.shape[0])],
                              k_book=5, cap=0.5)
    num = book.analytics.select_dtypes(include=[np.number])
    assert np.isfinite(num.to_numpy(dtype="float64", na_value=0.0)).all()
    # And the flag accounts for every NULL it created.
    assert (book.analytics["kelly_unbounded"]
            == book.analytics["kelly_max_feasible"].isna()).all()


def test_analytics_frame_carries_each_quantity_under_one_name():
    """`export_duplicate_content`'s test, run at the source that produced it.

    ``rank_denominator`` was a verbatim copy of ``downside_dev`` and
    ``tail_risk_admitted`` equalled ``tail_risk`` whenever the floor bound on
    nothing, which is the default.
    """
    rng = np.random.default_rng(3)
    draws = rng.normal(0.05, 0.2, size=(40, 800))
    book = optimize_portfolio(draws, [f"X{i}" for i in range(40)], k_book=8, cap=0.3)
    a = book.analytics
    assert "rank_denominator" not in a.columns
    assert not [c for c in a.columns if c.endswith("_admitted")]
    numeric = a.select_dtypes(include=[np.number])
    assert not numeric.T.duplicated(keep=False).any()
    # The metadata the copied column used to carry, as one fact.
    assert book.summary["rank_denominator_col"] == "downside_dev"


def test_floor_exclusions_are_reported_as_booleans_not_as_a_masked_copy():
    """The boolean carries what the masked float carried: which names were cut.

    Booleans also sit outside ``select_dtypes([np.number])``, so a flag can never
    itself trip the finiteness or duplicate-content gates.
    """
    rng = np.random.default_rng(4)
    draws = rng.normal(0.05, 0.2, size=(30, 600))
    # Two names with essentially no modelled downside: below the absolute floor.
    draws[0] = np.abs(draws[0]) + 0.5
    draws[1] = np.abs(draws[1]) + 0.5
    book = optimize_portfolio(draws, [f"X{i}" for i in range(30)], k_book=5, cap=0.5)
    a = book.analytics
    assert a["downside_dev_floored"].dtype == bool
    assert a["tail_risk_floored"].dtype == bool
    # A floored name is exactly a name whose ranking ratio could not be formed.
    floored = a["downside_dev_floored"].to_numpy()
    assert floored[:2].all()
    assert a.loc[floored, "reward_to_downside"].isna().all()


def test_terminal_wealth_is_null_where_it_is_unrepresentable_not_inf():
    """Compounding 2,000 winning bets is a real number; float64 cannot hold it.

    ``exp(0.42 * 2000)`` is about ``10**367`` against a ``~1.8e308`` ceiling, so
    the old curve returned ``+inf`` for the fractions past that crossing -- 21 of
    the 100 defaults, and 63 of them on run ``486df52e7014``'s own book. An
    infinity is not a wealth, and the ranking of two infinities is undefined.
    """
    draws = _no_losing_scenario_draws(n=1, s=2000, seed=1)[0] + 0.4
    curve = terminal_wealth_curve(draws)

    tw = curve["terminal_wealth"].to_numpy()
    assert not np.isinf(tw).any(), "an infinity reached the curve"
    assert curve["terminal_wealth_overflow"].any(), "this fixture must overflow"
    # The flag accounts for every NULL, and only for those.
    assert (curve["terminal_wealth_overflow"] == curve["terminal_wealth"].isna()).all()
    # The log columns carry the answer and stay finite throughout.
    assert np.isfinite(curve["log_terminal_wealth"]).all()
    assert np.isfinite(curve["log_growth"]).all()
    # And the peak -- the point of the curve -- survives the change of column.
    assert (curve.loc[curve["log_terminal_wealth"].idxmax(), "fraction"]
            == curve.loc[curve["log_growth"].idxmax(), "fraction"])


def test_terminal_wealth_is_unchanged_where_it_is_representable():
    """The fix must not move a curve that never overflowed."""
    rng = np.random.default_rng(7)
    draws = rng.normal(0.02, 0.10, size=200)
    curve = terminal_wealth_curve(draws, start_capital=100_000.0)
    assert not curve["terminal_wealth_overflow"].any()
    assert np.isfinite(curve["terminal_wealth"]).all()
    # exp(log wealth) is the wealth, to floating-point tolerance.
    np.testing.assert_allclose(
        curve["terminal_wealth"].to_numpy(),
        np.exp(curve["log_terminal_wealth"].to_numpy()),
        rtol=1e-9,
    )


def test_ruin_is_still_recorded_as_ruin_not_as_an_overflow():
    """A wiped-out path has zero wealth and -inf growth; that is not an overflow."""
    draws = np.array([0.5, -0.9, 0.2, 0.4])
    curve = terminal_wealth_curve(draws, fractions=np.array([0.5, 2.0]))
    ruined = curve[curve["fraction"] == 2.0].iloc[0]
    assert ruined["terminal_wealth"] == 0.0
    assert ruined["log_growth"] == float("-inf")
    assert not bool(ruined["terminal_wealth_overflow"])
