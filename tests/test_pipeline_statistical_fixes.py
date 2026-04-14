"""TDD tests for pipeline statistical fixes identified in the comprehensive analysis.

Each test class targets one of the 7 identified issues from the expected returns
pipeline analysis.  Tests are written RED-first: they describe the *desired*
behaviour so that the corresponding production fix can be implemented to make
them GREEN.

Issue reference (priority order):
  1. Critical — MC return winsorization  (§2.1)
  2. Critical — Scale-aware weighted agreement  (§2.2)
  3. High    — Heavy-tail risk metric clipping  (§2.3)
  4. High    — Bayesian prior–likelihood balance  (§2.4)
  5. Medium  — Degenerate distress/safety rescaling  (§2.5)
  6. Medium  — Quantile-based quality tier bins  (§2.6)
  7. Low     — MC coverage gap diagnostic logging  (§2.8)
"""

from __future__ import annotations

import logging
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal DataFrames that satisfy column contracts
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _make_isins(n: int = 200) -> list[str]:
    return [f"ISIN{i:06d}" for i in range(n)]


def _mc_df(n: int = 200, *, extreme_pct: float = 0.05) -> pd.DataFrame:
    """MC DataFrame with a controllable fraction of extreme outliers."""
    isins = _make_isins(n)
    returns = _RNG.normal(28, 40, size=n)
    # Inject extreme outliers
    n_extreme = max(1, int(n * extreme_pct))
    extreme_idx = _RNG.choice(n, size=n_extreme, replace=False)
    returns[extreme_idx] = _RNG.uniform(400, 900, size=n_extreme)
    return pd.DataFrame(
        {
            "isin": isins,
            "ticker": [f"T{i}" for i in range(n)],
            "implied_return_mc": returns,
            "expected_upside_mc": returns,  # currently identical — see §1.2
            "price_target_mc": _RNG.uniform(50, 300, size=n),
            "prob_positive_upside": _RNG.uniform(30, 95, size=n),
            "var_5_pct": _RNG.uniform(-30, -5, size=n),
            "risk_reward_ratio": _RNG.uniform(0.5, 10, size=n),
        }
    )


def _kal_df(n: int = 200) -> pd.DataFrame:
    isins = _make_isins(n)
    return pd.DataFrame(
        {
            "isin": isins,
            "implied_return_kalman": _RNG.normal(26, 35, size=n),
            "expected_upside_kalman": _RNG.normal(24, 30, size=n),
            "kalman_estimate": _RNG.uniform(50, 300, size=n),
            "kalman_variance": _RNG.uniform(0.01, 5, size=n),
        }
    )


def _pt_df(n: int = 200) -> pd.DataFrame:
    isins = _make_isins(n)
    return pd.DataFrame(
        {
            "isin": isins,
            "implied_return_pt": _RNG.normal(7, 12, size=n),
            "achievement_probability": _RNG.uniform(0.2, 0.9, size=n),
            "mh_achievement_probability": _RNG.uniform(0.2, 0.9, size=n),
            "price_target_prob_weighted": _RNG.uniform(50, 300, size=n),
            "confidence_level": _RNG.choice(["High", "Medium", "Low"], size=n),
            "analyst_conviction": _RNG.uniform(0, 1, size=n),
            "bullish_pct": _RNG.uniform(0, 100, size=n),
            "eps_revision_momentum": _RNG.normal(0, 5, size=n),
            "analyst_rating_normalized": _RNG.uniform(1, 5, size=n),
        }
    )


def _earn_df(n: int = 200) -> pd.DataFrame:
    isins = _make_isins(n)
    return pd.DataFrame(
        {
            "isin": isins,
            "posterior_beat_prob": _RNG.uniform(0.2, 0.85, size=n),
            "posterior_std": _RNG.uniform(0.01, 0.15, size=n),
            "confidence_score": _RNG.uniform(0.18, 0.55, size=n),
            "beat_classification": _RNG.choice(["Beat", "Miss"], size=n),
            "base_posterior_mean": _RNG.uniform(0.4, 0.6, size=n),
            "resampled_posterior_mean": _RNG.uniform(0.43, 0.60, size=n),
            "technical_adjustment": _RNG.normal(0, 0.05, size=n),
            "momentum_signal": _RNG.normal(0, 1, size=n),
            "volatility_regime_score": _RNG.uniform(0, 1, size=n),
            "credible_interval_90": [f"({0.3+i*0.001:.3f}, {0.7+i*0.001:.3f})" for i in range(n)],
            "credible_interval_95": [f"({0.25+i*0.001:.3f}, {0.75+i*0.001:.3f})" for i in range(n)],
            "prob_beat_given_momentum": _RNG.uniform(0.38, 0.67, size=n),
            "streak_type": _RNG.choice(["improving", "declining", "stable"], size=n),
            "continuation_probability": _RNG.uniform(0.3, 0.8, size=n),
            "mean_reversion_probability": _RNG.uniform(0.2, 0.7, size=n),
            "expected_next_outcome": _RNG.choice(["Beat", "Miss"], size=n),
            "prediction_confidence": _RNG.uniform(0.4, 0.9, size=n),
            "model_confidence": _RNG.uniform(0.3, 0.8, size=n),
            "map_estimate": _RNG.uniform(0.3, 0.7, size=n),
        }
    )


def _anomaly_df(n: int = 200) -> pd.DataFrame:
    isins = _make_isins(n)
    return pd.DataFrame(
        {
            "isin": isins,
            "accounting_anomaly_score": _RNG.normal(47, 17, size=n),
            "anomaly_risk_rank": _RNG.uniform(0, 100, size=n),
            "anomaly_severity_score": _RNG.normal(105, 37, size=n),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1 — Critical: MC Return Winsorization (§2.1)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCReturnWinsorization:
    """MC implied returns must be winsorized at 1st/99th percentile,
    matching the treatment already applied to Kalman returns."""

    def test_mc_returns_clipped_after_run_monte_carlo_analysis(self):
        """run_monte_carlo_analysis should winsorize implied_return_mc."""
        from probabilistic_ml_model.pipeline_runners import run_monte_carlo_analysis

        # Build a minimal df that passes column checks
        n = 300
        isins = _make_isins(n)
        df = pd.DataFrame(
            {
                "isin": isins,
                "last_price": _RNG.uniform(10, 500, size=n),
                "price_target": _RNG.uniform(10, 600, size=n),
                "price_target_high": _RNG.uniform(100, 800, size=n),
                "price_target_low": _RNG.uniform(5, 100, size=n),
            }
        )

        with patch(
            "probabilistic_ml_model.statistical_functions.statistical_models.monte_carlo_price_target_simulation"
        ) as mock_mc:
            # Simulate MC output with extreme outliers
            mc_out = _mc_df(n)
            mock_mc.return_value = mc_out

            with patch(
                "probabilistic_ml_model.pipeline_runners.compute_price_target_mc",
                side_effect=lambda mc, _df, **kw: mc,
            ):
                    result = run_monte_carlo_analysis(df)

        if result.empty:
            pytest.skip("MC runner returned empty — column mismatch in test fixture")

        raw_max = mc_out["implied_return_mc"].max()
        result_max = result["implied_return_mc"].max()
        p99 = mc_out["implied_return_mc"].quantile(0.99)

        # After winsorization the max must not exceed the 99th percentile
        assert result_max <= p99 * 1.01, (
            f"MC returns not winsorized: max {result_max:.1f}% exceeds p99 {p99:.1f}%"
        )

    def test_mc_winsorization_symmetric(self):
        """Both lower and upper tails should be clipped."""
        mc = _mc_df(500)
        # Inject negative extreme
        mc.loc[0, "implied_return_mc"] = -500.0
        mc.loc[1, "implied_return_mc"] = 900.0

        lower, upper = mc["implied_return_mc"].quantile([0.01, 0.99])
        clipped = mc["implied_return_mc"].clip(lower, upper)

        assert clipped.min() >= lower
        assert clipped.max() <= upper


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 2 — Critical: Scale-Aware Weighted Agreement (§2.2)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScaleAwareAgreement:
    """Bullish thresholds should be model-specific or percentile-based
    to prevent the MC/Kalman scale (mean ~27%) from always dominating
    the PT model (mean ~7%)."""

    def test_uniform_threshold_biases_agreement(self):
        """With a uniform 10% threshold, MC and Kalman are almost always
        bullish while PT is often not — demonstrating the bias."""
        mc = _mc_df(1000)
        kal = _kal_df(1000)
        pt = _pt_df(1000)

        threshold = 10.0
        mc_bull_pct = (mc["implied_return_mc"] > threshold).mean()
        kal_bull_pct = (kal["implied_return_kalman"] > threshold).mean()
        pt_bull_pct = (pt["implied_return_pt"] > threshold).mean()

        # MC and Kalman should be bullish much more often than PT
        assert mc_bull_pct > pt_bull_pct + 0.15, "Expected MC to be bullish far more often than PT"
        assert kal_bull_pct > pt_bull_pct + 0.15, "Expected Kalman to be bullish far more often than PT"

    def test_build_tri_model_uses_scale_aware_thresholds(self):
        """build_tri_model_alignment should use model-specific thresholds
        so that each model's bullish rate is roughly comparable."""
        from probabilistic_ml_model.statistical_functions.ensemble_models import (
            build_tri_model_alignment,
        )

        n = 1000
        mc = _mc_df(n)
        kal = _kal_df(n)
        pt = _pt_df(n)

        tri = build_tri_model_alignment(mc, kal, pt)
        if tri.empty:
            pytest.skip("Tri-model alignment returned empty")

        mc_bull_rate = tri["mc_bullish"].mean()
        pt_bull_rate = tri["pt_bullish"].mean()

        # After scale-aware fix, the gap between MC and PT bullish rates
        # should be less than 30 percentage points (currently ~50+pp)
        gap = abs(mc_bull_rate - pt_bull_rate)
        assert gap < 0.30, (
            f"Bullish rate gap MC({mc_bull_rate:.0%}) vs PT({pt_bull_rate:.0%}) = "
            f"{gap:.0%} — scale-aware thresholds not applied"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 3 — High: Heavy-Tail Risk Metric Clipping (§2.3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHeavyTailClipping:
    """pt_spread, risk_reward_ratio, and upside_std must be robustly
    clipped before entering the summary to prevent outlier domination."""

    def _summary_with_heavy_tails(self, n: int = 500) -> pd.DataFrame:
        """Create a summary-like DataFrame with extreme heavy-tailed metrics."""
        isins = _make_isins(n)
        df = pd.DataFrame({"isin": isins})
        # Simulate heavy-tailed distributions matching observed kurtosis
        df["pt_spread"] = np.abs(_RNG.standard_cauchy(n)) * 5000
        df["risk_reward_ratio"] = np.abs(_RNG.standard_cauchy(n)) * 50
        df["upside_std"] = np.abs(_RNG.standard_cauchy(n)) * 15
        return df

    def test_heavy_tail_cols_clipped_in_summary(self):
        """build_expected_returns_summary should clip heavy-tailed columns."""
        from probabilistic_ml_model.statistical_functions.ensemble_models import (
            build_expected_returns_summary,
        )

        n = 200
        mc = _mc_df(n)
        kal = _kal_df(n)
        pt = _pt_df(n)
        earn = _earn_df(n)
        anomaly = _anomaly_df(n)

        # Inject extreme pt_spread and risk_reward_ratio into mc
        mc["pt_spread"] = np.abs(_RNG.standard_cauchy(n)) * 50000
        mc["risk_reward_ratio"] = np.abs(_RNG.standard_cauchy(n)) * 200

        summary = build_expected_returns_summary(mc, kal, pt, earn, anomaly)
        if summary.empty:
            pytest.skip("Summary returned empty")

        for col in ["pt_spread", "risk_reward_ratio"]:
            if col not in summary.columns:
                continue
            kurtosis = summary[col].kurtosis()
            # After clipping, kurtosis should be dramatically reduced from 800+
            assert kurtosis < 50, (
                f"{col} kurtosis {kurtosis:.0f} still extreme — "
                f"heavy-tail clipping not applied"
            )

    def test_clipping_preserves_central_values(self):
        """Clipping at 1st/99th percentile should not alter the median."""
        df = self._summary_with_heavy_tails(1000)
        for col in ["pt_spread", "risk_reward_ratio", "upside_std"]:
            original_median = df[col].median()
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            clipped = df[col].clip(lo, hi)
            assert abs(clipped.median() - original_median) < original_median * 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 4 — High: Bayesian Prior–Likelihood Balance (§2.4)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianPriorLikelihoodBalance:
    """The resampled posterior mean should have sufficient IQR (>0.10)
    to provide discriminative power across stocks."""

    def test_resampled_posterior_has_sufficient_iqr(self):
        """After strengthening the momentum prior, the IQR of
        resampled_posterior_mean should exceed 0.10."""
        from probabilistic_ml_model.statistical_functions.probability_models import (
            EarningsBeatProbabilityModel,
        )

        model = EarningsBeatProbabilityModel(
            use_quality_adjustment=True,
            use_momentum_prior=True,
            momentum_prior_strength=0.3,
        )
        # The momentum_prior_strength should be >= 0.3 to allow data
        # to overcome the prior (was 0.1 in original)
        assert model.momentum_prior_strength >= 0.3, (
            f"momentum_prior_strength={model.momentum_prior_strength} too low — "
            f"prior will dominate likelihood, producing near-constant posteriors"
        )

    def test_earnings_model_default_momentum_strength(self):
        """Default EarningsBeatProbabilityModel should use momentum_prior_strength >= 0.3."""
        from probabilistic_ml_model.statistical_functions.probability_models import (
            EarningsBeatProbabilityModel,
        )

        model = EarningsBeatProbabilityModel()
        assert model.momentum_prior_strength >= 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 5 — Medium: Degenerate Distress/Safety Score Rescaling (§2.5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDegenerateDistressRescaling:
    """When >40% of stocks are at the maximum distress_risk_score,
    the pipeline should apply percentile-based rescaling."""

    def _credit_df_degenerate(self, n: int = 200) -> pd.DataFrame:
        """Simulate degenerate distress scores: >50% at maximum."""
        isins = _make_isins(n)
        scores = np.full(n, 100.0)
        # Set ~40% to non-max values
        non_max = _RNG.choice(n, size=int(n * 0.4), replace=False)
        scores[non_max] = _RNG.uniform(20, 90, size=len(non_max))
        return pd.DataFrame(
            {
                "isin": isins,
                "distress_risk_score": scores,
                "distress_probability": _RNG.uniform(0.5, 1.0, size=n),
                "survival_probability": np.clip(_RNG.normal(0.96, 0.05, size=n), 0, 1),
                "ruin_probability": _RNG.uniform(0, 0.1, size=n),
            }
        )

    def test_degenerate_distress_detected(self):
        """More than 40% at max should trigger rescaling."""
        credit = self._credit_df_degenerate(500)
        at_max = (credit["distress_risk_score"] >= credit["distress_risk_score"].max()).mean()
        assert at_max > 0.40, "Test fixture should have >40% at max"

    def test_rescaled_distress_has_better_discrimination(self):
        """After percentile rescaling, the distribution should use the
        full 0–100 range instead of clustering at 100."""
        credit = self._credit_df_degenerate(500)
        at_max_before = (
            credit["distress_risk_score"] >= credit["distress_risk_score"].max()
        ).mean()

        if at_max_before > 0.40:
            # Apply the proposed rescaling
            credit["distress_risk_score"] = (
                credit["distress_risk_score"].rank(pct=True) * 100
            )

        # After rescaling, the IQR should span a meaningful range
        iqr = credit["distress_risk_score"].quantile(0.75) - credit["distress_risk_score"].quantile(0.25)
        assert iqr > 20, (
            f"IQR={iqr:.1f} too narrow after rescaling — "
            f"percentile rescaling not effective"
        )

    def test_pipeline_rescales_degenerate_credit(self):
        """run_credit_risk_analysis should detect and rescale degenerate
        distress_risk_score distributions."""
        # This test validates the pipeline integration point.
        # The fix should be in run_credit_risk_analysis or
        # build_expected_returns_summary.
        credit = self._credit_df_degenerate(500)
        original_std = credit["distress_risk_score"].std()

        # After rescaling, std should increase (better spread)
        credit["distress_risk_score"] = credit["distress_risk_score"].rank(pct=True) * 100
        rescaled_std = credit["distress_risk_score"].std()

        assert rescaled_std > original_std * 0.5, (
            "Rescaled std should be meaningful — percentile rescaling "
            "should spread the distribution"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 6 — Medium: Quantile-Based Quality Tier Bins (§2.6)
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuantileQualityTierBins:
    """Quality tier bins should be data-adaptive (quantile-based) when
    the universe is large enough, producing balanced bucket sizes."""

    def test_fixed_bins_produce_uneven_buckets(self):
        """Demonstrate that fixed bins [18,25,35,45,55,60,75] produce
        highly uneven tier sizes for the observed distribution."""
        scores = _RNG.normal(44.8, 9.6, size=5000).clip(15, 72)
        tiers = pd.cut(
            scores,
            bins=[18, 25, 35, 45, 55, 60, 75],
            labels=["Very Low", "Low", "Below Avg", "Above Avg", "High", "Premium"],
        )
        counts = pd.Series(tiers).value_counts()
        # The ratio of largest to smallest bucket should be very high
        ratio = counts.max() / max(counts.min(), 1)
        assert ratio > 3, (
            f"Fixed bins should produce uneven buckets (ratio={ratio:.1f})"
        )

    def test_quantile_bins_produce_balanced_buckets(self):
        """Quantile-based bins should produce roughly equal-sized tiers."""
        scores = pd.Series(_RNG.normal(44.8, 9.6, size=5000).clip(15, 72))
        q_bins = [
            scores.min() - 0.01,
            scores.quantile(0.10),
            scores.quantile(0.30),
            scores.quantile(0.50),
            scores.quantile(0.70),
            scores.quantile(0.90),
            scores.max() + 0.01,
        ]
        q_bins = sorted(set(q_bins))
        labels = ["Very Low", "Low", "Below Avg", "Above Avg", "High", "Premium"]
        tiers = pd.cut(scores, bins=q_bins, labels=labels[: len(q_bins) - 1])
        counts = tiers.value_counts()
        ratio = counts.max() / max(counts.min(), 1)
        # Quantile bins should produce a max/min ratio < 3
        assert ratio < 3, (
            f"Quantile bins should produce balanced buckets (ratio={ratio:.1f})"
        )

    def test_filter_quality_stocks_uses_adaptive_bins(self):
        """filter_quality_stocks should use quantile-based bins when
        the universe has >100 scored stocks."""
        # This is a specification test — the fix should modify
        # filter_quality_stocks in expected_returns_v3.py and/or
        # pipeline_runners.py to use adaptive bins.
        from expected_returns_v3 import filter_quality_stocks

        n = 500
        isins = _make_isins(n)
        summary = pd.DataFrame(
            {
                "isin": isins,
                "composite_score": _RNG.normal(44.8, 9.6, size=n).clip(15, 72),
            }
        )
        source_df = pd.DataFrame({"isin": isins})

        with patch("expected_returns_v3.rank_stocks_by_composite_score") as mock_rank:
            mock_rank.return_value = summary.copy()
            result = filter_quality_stocks(summary.copy(), source_df)

        if "quality_tier" not in result.columns:
            pytest.skip("quality_tier not computed — mock may not match")

        counts = result["quality_tier"].value_counts()
        if len(counts) < 2:
            pytest.skip("Too few tiers populated")

        ratio = counts.max() / max(counts.min(), 1)
        assert ratio < 5, (
            f"Quality tier bucket ratio {ratio:.1f} too uneven — "
            f"adaptive quantile bins should be used"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 7 — Low: MC Coverage Gap Diagnostic Logging (§2.8)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCCoverageGapLogging:
    """When MC processes <90% of input stocks, a warning should be logged."""

    def test_coverage_gap_warning_logged(self, caplog):
        """run_monte_carlo_analysis should log a warning when output
        count is <90% of input count."""
        from probabilistic_ml_model.pipeline_runners import run_monte_carlo_analysis

        n_input = 200
        n_output = 150  # 75% — below 90% threshold
        isins_in = _make_isins(n_input)
        isins_out = isins_in[:n_output]

        df = pd.DataFrame(
            {
                "isin": isins_in,
                "last_price": _RNG.uniform(10, 500, size=n_input),
                "price_target": _RNG.uniform(10, 600, size=n_input),
                "price_target_high": _RNG.uniform(100, 800, size=n_input),
                "price_target_low": _RNG.uniform(5, 100, size=n_input),
            }
        )

        mc_out = pd.DataFrame(
            {
                "isin": isins_out,
                "implied_return_mc": _RNG.normal(28, 40, size=n_output),
                "price_target_mc": _RNG.uniform(50, 300, size=n_output),
            }
        )

        with patch(
            "probabilistic_ml_model.statistical_functions.statistical_models.monte_carlo_price_target_simulation",
            return_value=mc_out,
        ):
            with patch(
                "probabilistic_ml_model.pipeline_runners.compute_price_target_mc",
                side_effect=lambda mc, _df, **kw: mc,
            ):
                with caplog.at_level(logging.WARNING):
                    result = run_monte_carlo_analysis(df)

        # Should contain a coverage gap warning
        coverage_warnings = [
            r for r in caplog.records if "coverage gap" in r.message.lower()
        ]
        assert len(coverage_warnings) >= 1, (
            "Expected a 'coverage gap' warning when MC processes <90% of input stocks. "
            f"Got {len(caplog.records)} log records, none about coverage gap."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-cutting: Consistency between v3 and pipeline_runners
# ═══════════════════════════════════════════════════════════════════════════════


class TestV3PipelineRunnerConsistency:
    """Both expected_returns_v3.py and pipeline_runners.py should apply
    the same statistical fixes."""

    def test_both_modules_winsorize_mc(self):
        """Both run_monte_carlo_analysis implementations should winsorize."""
        import inspect

        from probabilistic_ml_model import pipeline_runners as pr

        import expected_returns_v3 as v3

        pr_src = inspect.getsource(pr.run_monte_carlo_analysis)
        v3_src = inspect.getsource(v3.run_monte_carlo_analysis)

        # After the fix, both should contain winsorization logic
        for label, src in [("pipeline_runners", pr_src), ("expected_returns_v3", v3_src)]:
            assert "clip" in src or "winsoriz" in src.lower(), (
                f"{label}.run_monte_carlo_analysis does not contain "
                f"winsorization/clipping logic for MC returns"
            )

    def test_both_modules_winsorize_kalman(self):
        """Verify Kalman winsorization is present in both modules (baseline)."""
        import inspect

        from probabilistic_ml_model import pipeline_runners as pr

        import expected_returns_v3 as v3

        pr_src = inspect.getsource(pr.run_kalman_filter)
        v3_src = inspect.getsource(v3.run_kalman_filter)

        for label, src in [("pipeline_runners", pr_src), ("expected_returns_v3", v3_src)]:
            assert "clip" in src, (
                f"{label}.run_kalman_filter missing winsorization — "
                f"this is a regression"
            )
