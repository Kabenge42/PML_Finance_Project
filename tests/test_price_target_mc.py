"""Tests for the price-target notebook helpers (``_price_target_mc.py``).

Covers:
  * Schema-aligned data preparation (Task 1 of the notebook spec).
  * AR(1) Monte-Carlo lagged risk-adjusted returns (Task 4).
  * Per-ISIN MC summary frame (Task 4 output).

The PyMC model itself (Tasks 2/3/5) requires a sampler and lives in
``pymc_price_target.ipynb``; here we test the deterministic pieces it relies
on. The helpers are pure-NumPy/pandas so they import without PyMC available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probabilistic_ml_model.pymc_models._price_target_mc import (
    PriceTargetInputs,
    prepare_price_target_inputs,
    simulate_lagged_risk_adjusted_returns,
    summarize_mc_returns,
)


# ---------------------------------------------------------------------------
# prepare_price_target_inputs
# ---------------------------------------------------------------------------
def _sample_pt_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "isin": ["A", "B", "C", "D", "E"],
            "sector": ["X", "Y", "X", None, "Y"],
            "close": [100.0, 50.0, 0.0, 25.0, 200.0],
            "price_target_avg": [120.0, 45.0, 10.0, 30.0, 220.0],
            "price_target_stddev": [5.0, 2.5, 1.0, 1.5, 0.0],
            "price_target_num": [10, 4, 2, 3, 1],
            "num_buys_ratings": [6, 1, 1, 2, 1],
        }
    )


class TestPreparePriceTargetInputs:
    def test_returns_dataclass_with_correct_shapes(self):
        prepped = prepare_price_target_inputs(_sample_pt_df())

        assert isinstance(prepped, PriceTargetInputs)
        # B (sector Y, close 50) and E (sector Y, close 200) survive;
        # A also survives. C dropped (close==0), D dropped (sector NaN).
        assert list(prepped.isins) == ["A", "B", "E"]
        assert prepped.target_vs_price_pct.shape == (3,)
        assert prepped.conviction_ratio.shape == (3,)
        assert prepped.dispersion_cv.shape == (3,)
        assert prepped.n_analysts.shape == (3,)

    def test_target_vs_price_pct_formula(self):
        prepped = prepare_price_target_inputs(_sample_pt_df())
        # (price_target_avg - close) / close
        expected = np.array([(120 - 100) / 100, (45 - 50) / 50, (220 - 200) / 200])
        np.testing.assert_allclose(prepped.target_vs_price_pct, expected)

    def test_conviction_and_dispersion(self):
        prepped = prepare_price_target_inputs(_sample_pt_df())
        # num_buys_ratings / max(price_target_num, 1)
        np.testing.assert_allclose(prepped.conviction_ratio, [6 / 10, 1 / 4, 1 / 1])
        # std / |avg|; row E has std=0 → dispersion 0.
        np.testing.assert_allclose(
            prepped.dispersion_cv, [5 / 120, 2.5 / 45, 0.0], rtol=1e-12
        )

    def test_n_analysts_floored_at_one(self):
        df = _sample_pt_df().assign(price_target_num=[10, 0, 2, 3, 0])
        prepped = prepare_price_target_inputs(df)
        # row B (was 0) and row E (was 0) are floored to 1
        assert (prepped.n_analysts >= 1).all()
        assert prepped.n_analysts.tolist() == [10.0, 1.0, 1.0]

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="missing required columns"):
            prepare_price_target_inputs(_sample_pt_df().drop(columns=["close"]))

    def test_idempotent_on_already_prepared_frame(self):
        once = prepare_price_target_inputs(_sample_pt_df())
        twice = prepare_price_target_inputs(once.frame)
        np.testing.assert_allclose(once.target_vs_price_pct, twice.target_vs_price_pct)


# ---------------------------------------------------------------------------
# simulate_lagged_risk_adjusted_returns
# ---------------------------------------------------------------------------
class TestSimulateLaggedRiskAdjustedReturns:
    @pytest.fixture
    def draws(self):
        rng = np.random.default_rng(0)
        mu = rng.normal(0.05, 0.02, size=(4, 50))
        sigma = np.abs(rng.normal(0.1, 0.01, size=(4, 50))) + 1e-3
        nu = rng.gamma(2.0, 10.0, size=50) + 2.0
        return mu, sigma, nu

    def test_output_shape(self, draws):
        mu, sigma, nu = draws
        mc = simulate_lagged_risk_adjusted_returns(mu, sigma, nu, horizon=4)
        assert mc.shape == (4, 50, 4)

    def test_seed_reproducible(self, draws):
        mu, sigma, nu = draws
        a = simulate_lagged_risk_adjusted_returns(mu, sigma, nu, random_seed=7)
        b = simulate_lagged_risk_adjusted_returns(mu, sigma, nu, random_seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, draws):
        mu, sigma, nu = draws
        a = simulate_lagged_risk_adjusted_returns(mu, sigma, nu, random_seed=1)
        b = simulate_lagged_risk_adjusted_returns(mu, sigma, nu, random_seed=2)
        assert not np.array_equal(a, b)

    def test_zero_sigma_collapses_to_ar1_drift(self):
        # With sigma=0 the recursion is deterministic and the per-horizon
        # values stay equal to mu (since rho*mu + (1-rho)*mu == mu).
        mu = np.full((2, 5), 0.07)
        sigma = np.zeros_like(mu)
        mc = simulate_lagged_risk_adjusted_returns(
            mu, sigma, nu_draws=10.0, horizon=3, rho=0.5
        )
        np.testing.assert_allclose(mc, 0.07)

    def test_scalar_nu_accepted(self, draws):
        mu, sigma, _ = draws
        mc = simulate_lagged_risk_adjusted_returns(mu, sigma, nu_draws=15.0)
        assert mc.shape == (mu.shape[0], mu.shape[1], 4)

    def test_invalid_shapes_raise(self):
        mu = np.zeros((3, 4))
        with pytest.raises(ValueError, match="mu_draws must be 2-D"):
            simulate_lagged_risk_adjusted_returns(mu.ravel(), mu, 5.0)
        with pytest.raises(ValueError, match="sigma_draws must match"):
            simulate_lagged_risk_adjusted_returns(mu, np.zeros((3, 5)), 5.0)
        with pytest.raises(ValueError, match="horizon"):
            simulate_lagged_risk_adjusted_returns(mu, mu, 5.0, horizon=0)
        with pytest.raises(ValueError, match="rho"):
            simulate_lagged_risk_adjusted_returns(mu, mu, 5.0, rho=1.0)
        with pytest.raises(ValueError, match="nu_draws"):
            simulate_lagged_risk_adjusted_returns(mu, mu, np.zeros(7))


# ---------------------------------------------------------------------------
# summarize_mc_returns
# ---------------------------------------------------------------------------
class TestSummarizeMcReturns:
    def test_columns_and_length(self):
        rng = np.random.default_rng(0)
        mc = rng.normal(0.05, 0.1, size=(3, 100, 4))
        isins = np.array(["A", "B", "C"])
        out = summarize_mc_returns(mc, isins)

        assert list(out.columns) == [
            "isin", "er_mean", "er_sd", "er_p05", "er_p50", "er_p95", "prob_pos"
        ]
        assert (out["er_sd"] >= 0.0).all()
        assert len(out) == 3
        assert list(out["isin"]) == ["A", "B", "C"]

    def test_quantile_ordering(self):
        rng = np.random.default_rng(0)
        mc = rng.normal(0.0, 0.2, size=(2, 200, 3))
        out = summarize_mc_returns(mc, np.array(["a", "b"]))
        assert (out["er_p05"] <= out["er_p50"]).all()
        assert (out["er_p50"] <= out["er_p95"]).all()
        # prob_pos ∈ [0, 1]
        assert ((out["prob_pos"] >= 0.0) & (out["prob_pos"] <= 1.0)).all()

    def test_prob_pos_perfectly_positive(self):
        mc = np.ones((2, 10, 3))
        out = summarize_mc_returns(mc, np.array(["a", "b"]))
        np.testing.assert_array_equal(out["prob_pos"].to_numpy(), [1.0, 1.0])
        np.testing.assert_allclose(out["er_mean"].to_numpy(), [1.0, 1.0])
        np.testing.assert_allclose(out["er_sd"].to_numpy(), [0.0, 0.0])

    def test_custom_quantiles_label_columns(self):
        mc = np.zeros((1, 5, 2))
        out = summarize_mc_returns(mc, np.array(["x"]), quantiles=(0.1, 0.5, 0.9))
        assert {"er_p10", "er_p50", "er_p90"}.issubset(out.columns)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="mc must be 3-D"):
            summarize_mc_returns(np.zeros((3, 4)), np.array(["a", "b", "c"]))
        with pytest.raises(ValueError, match="isins length"):
            summarize_mc_returns(np.zeros((3, 4, 2)), np.array(["a"]))
        with pytest.raises(ValueError, match="quantiles"):
            summarize_mc_returns(
                np.zeros((1, 4, 2)), np.array(["a"]), quantiles=(0.1, 0.9)
            )


# ---------------------------------------------------------------------------
# End-to-end smoke: data prep → MC sim → summary
# ---------------------------------------------------------------------------
def test_end_to_end_pipeline_smoke():
    prepped = prepare_price_target_inputs(_sample_pt_df())
    n = prepped.target_vs_price_pct.size
    # Pretend posterior draws: broadcast observed target as mu, dispersion as sigma.
    n_samples = 64
    mu = np.tile(prepped.target_vs_price_pct[:, None], (1, n_samples))
    sigma = np.tile(
        (prepped.dispersion_cv + 0.01)[:, None], (1, n_samples)
    )
    nu = np.full(n_samples, 12.0)

    mc = simulate_lagged_risk_adjusted_returns(
        mu, sigma, nu, horizon=4, rho=0.85, random_seed=42
    )
    summary = summarize_mc_returns(mc, prepped.isins)

    assert mc.shape == (n, n_samples, 4)
    assert len(summary) == n
    assert (summary["er_p05"] <= summary["er_p95"]).all()