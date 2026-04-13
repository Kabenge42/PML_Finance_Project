"""Tests for _safe_get helper and raw-value output in probability models.

Verifies that model output records preserve actual data values (or NaN when
the source column is missing/null) rather than the calculation defaults.
Also tests _compute_analyst_conviction fallback derivation.
"""

import math

import numpy as np
import pandas as pd
import pytest

from probabilistic_ml_model.statistical_functions.probability_models import (
    _safe_get,
    _compute_analyst_conviction,
    CreditRiskProbabilityModel,
    DividendCutProbabilityModel,
    EarningsBeatProbabilityModel,
    PriceTargetAchievementModel,
)
from analytics.probability_analytics import (
    _safe_get as _safe_get_legacy,
    _compute_analyst_conviction as _compute_analyst_conviction_legacy,
    CreditRiskProbabilityModel as LegacyCreditRiskModel,
    DividendCutProbabilityModel as LegacyDividendCutModel,
    EarningsBeatProbabilityModel as LegacyEarningsBeatModel,
    PriceTargetAchievementModel as LegacyPriceTargetModel,
)


# ─── _safe_get unit tests ────────────────────────────────────────────────────


class TestSafeGet:
    def test_column_present_with_value(self):
        row = pd.Series({"analyst_conviction": 85.0})
        calc, raw = _safe_get(row, "analyst_conviction", 50)
        assert calc == 85.0
        assert raw == 85.0

    def test_column_missing_returns_default_and_nan(self):
        row = pd.Series({"other_col": 1.0})
        calc, raw = _safe_get(row, "analyst_conviction", 50)
        assert calc == 50
        assert math.isnan(raw)

    def test_column_present_with_nan(self):
        row = pd.Series({"analyst_conviction": np.nan})
        calc, raw = _safe_get(row, "analyst_conviction", 50)
        assert calc == 50
        assert math.isnan(raw)

    def test_column_present_with_none(self):
        row = pd.Series({"analyst_conviction": None})
        calc, raw = _safe_get(row, "analyst_conviction", 50)
        assert calc == 50
        assert math.isnan(raw)

    def test_column_present_with_zero(self):
        row = pd.Series({"score": 0})
        calc, raw = _safe_get(row, "score", 50)
        assert calc == 0
        assert raw == 0

    def test_column_present_with_negative(self):
        row = pd.Series({"momentum": -3.5})
        calc, raw = _safe_get(row, "momentum", 0)
        assert calc == -3.5
        assert raw == -3.5


# ─── PriceTargetAchievementModel output tests ────────────────────────────────


class TestPriceTargetOutputValues:
    """Verify PriceTargetAchievementModel outputs raw data, not defaults."""

    @pytest.fixture()
    def model(self):
        return PriceTargetAchievementModel(use_mcmc=False)

    def test_actual_values_preserved_in_output(self, model):
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "expected_upside_pt": [25.0],
                "price_target_spread_pct": [18.0],
                "analyst_conviction": [89.5],
                "eps_revision_momentum": [3.2],
                "analyst_rating_normalized": [78.0],
                "analyst_bullish_pct": [72.0],
            }
        )
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert row["analyst_conviction"] == 89.5
        assert row["eps_revision_momentum"] == 3.2
        assert row["analyst_rating_normalized"] == 78.0
        assert row["bullish_pct"] == 72.0
        assert row["expected_upside_pt"] == 25.0
        assert row["price_target_spread_pct"] == 18.0

    def test_missing_columns_produce_nan_not_defaults(self, model):
        """Core bug reproduction: missing columns must yield NaN, not 50."""
        df = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "expected_upside_pt": [15.0],
            }
        )
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        # These columns were absent from input — output must be NaN, NOT 50
        assert pd.isna(row["analyst_conviction"]), (
            f"Expected NaN for missing analyst_conviction, got {row['analyst_conviction']}"
        )
        assert pd.isna(row["eps_revision_momentum"])
        assert pd.isna(row["analyst_rating_normalized"])
        assert pd.isna(row["bullish_pct"])
        assert pd.isna(row["price_target_spread_pct"])
        # expected_upside_pt WAS provided
        assert row["expected_upside_pt"] == 15.0

    def test_nan_values_produce_nan_in_output(self, model):
        df = pd.DataFrame(
            {
                "ticker": ["GOOG"],
                "expected_upside_pt": [10.0],
                "analyst_conviction": [np.nan],
                "eps_revision_momentum": [np.nan],
            }
        )
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert pd.isna(row["analyst_conviction"])
        assert pd.isna(row["eps_revision_momentum"])

    def test_calculation_still_uses_defaults_for_missing(self, model):
        """Probabilities should still be computed (not crash) when columns are missing."""
        df = pd.DataFrame({"ticker": ["X"], "expected_upside_pt": [20.0]})
        result = model.analyze_dataframe(df)
        assert len(result) == 1
        assert 0.05 <= result.iloc[0]["achievement_probability"] <= 0.90


# ─── CreditRiskProbabilityModel output tests ─────────────────────────────────


class TestCreditRiskOutputValues:
    @pytest.fixture()
    def model(self):
        return CreditRiskProbabilityModel(use_mcmc=False)

    def test_actual_values_preserved(self, model):
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "altman_z_score": [4.2],
                "liquidity_stress_score": [22.0],
                "beta_stability_score": [80.0],
                "distress_risk_score": [15.0],
            }
        )
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert row["altman_z_score"] == 4.2
        assert row["liquidity_stress_score"] == 22.0
        assert row["beta_stability_score"] == 80.0
        assert row["distress_risk_score"] == 15.0

    def test_missing_columns_produce_nan(self, model):
        df = pd.DataFrame({"ticker": ["TSLA"]})
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert pd.isna(row["altman_z_score"])
        assert pd.isna(row["liquidity_stress_score"])
        assert pd.isna(row["beta_stability_score"])
        assert pd.isna(row["distress_risk_score"])
        assert pd.isna(row["balance_sheet_strength"])
        # Computed fields should still be present
        assert 0.01 <= row["distress_probability"] <= 0.99


# ─── DividendCutProbabilityModel output tests ────────────────────────────────


class TestDividendCutOutputValues:
    @pytest.fixture()
    def model(self):
        return DividendCutProbabilityModel(use_mcmc=False)

    def test_actual_values_preserved(self, model):
        df = pd.DataFrame(
            {
                "ticker": ["JNJ"],
                "fcf_dividend_coverage": [3.5],
                "dividend_payout_ratio": [40.0],
                "dividend_streak": [25],
                "dividend_consistency": [0.95],
            }
        )
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert row["fcf_dividend_coverage"] == 3.5
        assert row["payout_ratio"] == 40.0
        assert row["dividend_streak"] == 25
        assert row["dividend_consistency"] == 0.95

    def test_missing_columns_produce_nan(self, model):
        df = pd.DataFrame({"ticker": ["ABC"]})
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert pd.isna(row["fcf_dividend_coverage"])
        assert pd.isna(row["payout_ratio"])
        assert pd.isna(row["dividend_streak"])
        assert pd.isna(row["dividend_consistency"])
        assert pd.isna(row["yield_vs_5y_avg"])
        assert 0.03 <= row["dividend_cut_probability"] <= 0.95


# ═════════════════════════════════════════════════════════════════════════════
# Legacy analytics module — same _safe_get refactor must apply
# ═════════════════════════════════════════════════════════════════════════════


class TestLegacySafeGet:
    def test_column_present_with_value(self):
        row = pd.Series({"analyst_conviction": 85.0})
        calc, raw = _safe_get_legacy(row, "analyst_conviction", 50)
        assert calc == 85.0
        assert raw == 85.0

    def test_column_missing_returns_default_and_nan(self):
        row = pd.Series({"other_col": 1.0})
        calc, raw = _safe_get_legacy(row, "analyst_conviction", 50)
        assert calc == 50
        assert math.isnan(raw)

    def test_column_present_with_nan(self):
        row = pd.Series({"analyst_conviction": np.nan})
        calc, raw = _safe_get_legacy(row, "analyst_conviction", 50)
        assert calc == 50
        assert math.isnan(raw)


class TestLegacyPriceTargetOutputValues:
    @pytest.fixture()
    def model(self):
        return LegacyPriceTargetModel(use_mcmc=False)

    def test_actual_values_preserved(self, model):
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "expected_upside_pt": [25.0],
                "analyst_conviction": [89.5],
                "eps_revision_momentum": [3.2],
                "analyst_rating_normalized": [78.0],
                "analyst_bullish_pct": [72.0],
            }
        )
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert row["analyst_conviction"] == 89.5
        assert row["eps_revision_momentum"] == 3.2
        assert row["analyst_rating_normalized"] == 78.0
        assert row["bullish_pct"] == 72.0

    def test_missing_columns_produce_nan_not_defaults(self, model):
        df = pd.DataFrame({"ticker": ["MSFT"], "expected_upside_pt": [15.0]})
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert pd.isna(row["analyst_conviction"])
        assert pd.isna(row["eps_revision_momentum"])
        assert pd.isna(row["analyst_rating_normalized"])
        assert pd.isna(row["bullish_pct"])


class TestLegacyCreditRiskOutputValues:
    @pytest.fixture()
    def model(self):
        return LegacyCreditRiskModel(use_mcmc=False)

    def test_actual_values_preserved(self, model):
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "altman_z_score": [4.2],
                "liquidity_stress_score": [22.0],
                "beta_stability_score": [80.0],
                "distress_risk_score": [15.0],
            }
        )
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert row["altman_z_score"] == 4.2
        assert row["liquidity_stress_score"] == 22.0
        assert row["beta_stability_score"] == 80.0
        assert row["distress_risk_score"] == 15.0

    def test_missing_columns_produce_nan(self, model):
        df = pd.DataFrame({"ticker": ["TSLA"]})
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert pd.isna(row["altman_z_score"])
        assert pd.isna(row["liquidity_stress_score"])
        assert pd.isna(row["beta_stability_score"])
        assert pd.isna(row["distress_risk_score"])


class TestLegacyDividendCutOutputValues:
    @pytest.fixture()
    def model(self):
        return LegacyDividendCutModel(use_mcmc=False)

    def test_actual_values_preserved(self, model):
        df = pd.DataFrame(
            {
                "ticker": ["JNJ"],
                "fcf_dividend_coverage": [3.5],
                "dividend_payout_ratio": [40.0],
                "dividend_streak": [25],
                "dividend_consistency": [0.95],
            }
        )
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert row["fcf_dividend_coverage"] == 3.5
        assert row["payout_ratio"] == 40.0
        assert row["dividend_streak"] == 25
        assert row["dividend_consistency"] == 0.95

    def test_missing_columns_produce_nan(self, model):
        df = pd.DataFrame({"ticker": ["ABC"]})
        result = model.analyze_dataframe(df)
        row = result.iloc[0]
        assert pd.isna(row["fcf_dividend_coverage"])
        assert pd.isna(row["payout_ratio"])
        assert pd.isna(row["dividend_streak"])
        assert pd.isna(row["dividend_consistency"])


# ═════════════════════════════════════════════════════════════════════════════
# _compute_analyst_conviction tests
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeAnalystConviction:
    """Test the _compute_analyst_conviction helper (both modules)."""

    def test_derives_from_bullish_bearish(self):
        row = pd.Series({"analyst_bullish_pct": 80.0, "analyst_bearish_pct": 10.0})
        assert _compute_analyst_conviction(row) == 70.0

    def test_returns_nan_when_bullish_missing(self):
        row = pd.Series({"analyst_bearish_pct": 10.0})
        assert pd.isna(_compute_analyst_conviction(row))

    def test_returns_nan_when_bearish_missing(self):
        row = pd.Series({"analyst_bullish_pct": 80.0})
        assert pd.isna(_compute_analyst_conviction(row))

    def test_returns_nan_when_both_missing(self):
        row = pd.Series({"other": 1.0})
        assert pd.isna(_compute_analyst_conviction(row))

    def test_returns_nan_when_bullish_nan(self):
        row = pd.Series({"analyst_bullish_pct": np.nan, "analyst_bearish_pct": 10.0})
        assert pd.isna(_compute_analyst_conviction(row))

    def test_absolute_value(self):
        row = pd.Series({"analyst_bullish_pct": 20.0, "analyst_bearish_pct": 60.0})
        assert _compute_analyst_conviction(row) == 40.0

    def test_legacy_module_matches(self):
        row = pd.Series({"analyst_bullish_pct": 75.0, "analyst_bearish_pct": 5.0})
        assert _compute_analyst_conviction_legacy(row) == 70.0

    def test_legacy_returns_nan_when_missing(self):
        row = pd.Series({"other": 1.0})
        assert pd.isna(_compute_analyst_conviction_legacy(row))


class TestConvictionFallbackInPriceTarget:
    """Verify PT model derives analyst_conviction from bullish/bearish when missing."""

    @pytest.fixture()
    def model(self):
        return PriceTargetAchievementModel(use_mcmc=False)

    def test_fallback_computes_conviction(self, model):
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "expected_upside_pt": [15.0],
            "analyst_bullish_pct": [80.0],
            "analyst_bearish_pct": [10.0],
        })
        result = model.analyze_dataframe(df)
        assert result.iloc[0]["analyst_conviction"] == 70.0

    def test_raw_value_preferred_over_fallback(self, model):
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "expected_upside_pt": [15.0],
            "analyst_conviction": [85.0],
            "analyst_bullish_pct": [80.0],
            "analyst_bearish_pct": [10.0],
        })
        result = model.analyze_dataframe(df)
        assert result.iloc[0]["analyst_conviction"] == 85.0

    def test_nan_when_no_source_data(self, model):
        df = pd.DataFrame({"ticker": ["MSFT"], "expected_upside_pt": [15.0]})
        result = model.analyze_dataframe(df)
        assert pd.isna(result.iloc[0]["analyst_conviction"])


class TestConvictionFallbackInPriceTargetLegacy:
    """Verify legacy PT model derives analyst_conviction from bullish/bearish."""

    @pytest.fixture()
    def model(self):
        return LegacyPriceTargetModel(use_mcmc=False)

    def test_fallback_computes_conviction(self, model):
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "expected_upside_pt": [15.0],
            "analyst_bullish_pct": [80.0],
            "analyst_bearish_pct": [10.0],
        })
        result = model.analyze_dataframe(df)
        assert result.iloc[0]["analyst_conviction"] == 70.0

    def test_raw_value_preferred_over_fallback(self, model):
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "expected_upside_pt": [15.0],
            "analyst_conviction": [85.0],
            "analyst_bullish_pct": [80.0],
            "analyst_bearish_pct": [10.0],
        })
        result = model.analyze_dataframe(df)
        assert result.iloc[0]["analyst_conviction"] == 85.0


class TestConvictionInEarningsBeat:
    """Verify earnings beat model passes through analyst_conviction with fallback."""

    @pytest.fixture()
    def model(self):
        return EarningsBeatProbabilityModel()

    def test_conviction_passthrough(self, model):
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "name": ["Apple"],
            "sector": ["Technology"],
            "eps_trajectory_score": [75.0],
            "analyst_conviction": [88.0],
        })
        result = model.analyze_dataframe_enhanced(df)
        assert not result.empty
        assert "analyst_conviction" in result.columns
        assert result.iloc[0]["analyst_conviction"] == 88.0

    def test_conviction_fallback_from_bullish_bearish(self, model):
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "name": ["Apple"],
            "sector": ["Technology"],
            "eps_trajectory_score": [75.0],
            "analyst_bullish_pct": [70.0],
            "analyst_bearish_pct": [5.0],
        })
        result = model.analyze_dataframe_enhanced(df)
        assert not result.empty
        assert "analyst_conviction" in result.columns
        assert result.iloc[0]["analyst_conviction"] == 65.0


class TestConvictionInEarningsBeatLegacy:
    """Verify legacy earnings beat model passes through analyst_conviction."""

    @pytest.fixture()
    def model(self):
        return LegacyEarningsBeatModel()

    def test_conviction_passthrough(self, model):
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "name": ["Apple"],
            "sector": ["Technology"],
            "eps_trajectory_score": [75.0],
            "analyst_conviction": [88.0],
        })
        result = model.analyze_dataframe_enhanced(df)
        assert not result.empty
        assert "analyst_conviction" in result.columns
        assert result.iloc[0]["analyst_conviction"] == 88.0

    def test_conviction_fallback_from_bullish_bearish(self, model):
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "name": ["Apple"],
            "sector": ["Technology"],
            "eps_trajectory_score": [75.0],
            "analyst_bullish_pct": [70.0],
            "analyst_bearish_pct": [5.0],
        })
        result = model.analyze_dataframe_enhanced(df)
        assert not result.empty
        assert "analyst_conviction" in result.columns
        assert result.iloc[0]["analyst_conviction"] == 65.0
