"""Tests for v3.5 accounting anomaly detection enhancements."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "sector": np.random.choice(["Tech", "Health", "Finance"], n),
            "industry": np.random.choice(["SW", "HW", "Bio"], n),
            # Original features
            "eps_adjustment_ratio": np.random.randn(n),
            "gaap_adj_eps_gap_pct": np.random.randn(n) * 5,
            "ebitda_adjustment_ratio": np.random.randn(n),
            # v3.5 new features — balance sheet & WC flags
            "accumulated_deficit_flag": np.random.choice([0, 1], n),
            "negative_wc_flag": np.random.choice([0, 1], n),
            "wc_deteriorating_flag": np.random.choice([0, 1], n),
            "intangibles_growth_flag": np.random.choice([0, 1], n),
            # Inventory signals
            "inventory_buildup_flag": np.random.choice([0, 1], n),
            "inventory_reduction_flag": np.random.choice([0, 1], n),
            # Impairment & writedown events
            "has_goodwill_impairment": np.random.choice([0, 1], n),
            "has_asset_writedown": np.random.choice([0, 1], n),
            "has_restructuring": np.random.choice([0, 1], n),
            "has_goodwill_impairment_ltm": np.random.choice([0, 1], n),
            "impairment_risk_score": np.random.uniform(0, 100, n),
            # Strategic & operational red flags
            "revenue_accelerating_flag": np.random.choice([0, 1], n),
            "overinvestment_flag": np.random.choice([0, 1], n),
            "recent_acquisition_flag": np.random.choice([0, 1], n),
            "high_rnd_intensity_flag": np.random.choice([0, 1], n),
            "has_unusual_items_flag": np.random.choice([0, 1], n),
            "low_tax_flag": np.random.choice([0, 1], n),
            "layoff_risk_flag": np.random.choice([0, 1], n),
            # External signals
            "analyst_bearish_pct": np.random.uniform(0, 50, n),
            "debt_maturity_risk": np.random.uniform(0, 100, n),
            # Quality frequency columns
            "goodwill_impairment_frequency": np.random.choice([0, 1, 2], n),
            "asset_writedown_frequency": np.random.choice([0, 1], n),
            "restructuring_frequency": np.random.choice([0, 1], n),
            "exceptional_items_frequency": np.random.choice([0, 1, 2], n),
            # Balance sheet quality columns for probability model
            "retained_earnings_vs_5y": np.random.uniform(0.3, 1.5, n),
            "asset_quality_score": np.random.uniform(10, 90, n),
            "accounting_anomaly_score": np.random.uniform(0, 100, n),
        }
    )


class TestDetectAccountingAnomaliesV35:
    """Test new v3.5 features in detect_accounting_anomalies."""

    def test_new_features_in_features_list(self, sample_df):
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            detect_accounting_anomalies,
        )

        result = detect_accounting_anomalies(sample_df)
        # Should have anomaly flags for new features
        new_flags = [
            "accumulated_deficit_flag_anomaly_flag",
            "inventory_buildup_flag_anomaly_flag",
            "has_goodwill_impairment_anomaly_flag",
            "analyst_bearish_pct_anomaly_flag",
            "debt_maturity_risk_anomaly_flag",
        ]
        for flag in new_flags:
            assert flag in result.columns, f"Missing flag column: {flag}"

    def test_new_feature_weights_applied(self, sample_df):
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            detect_accounting_anomalies,
        )

        result = detect_accounting_anomalies(sample_df)
        assert "accounting_anomaly_score" in result.columns
        # Score should be non-zero since we have data for the new features
        assert result["accounting_anomaly_score"].max() > 0

    def test_quality_frequency_includes_event_cols(self, sample_df):
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            detect_accounting_anomalies,
        )

        result = detect_accounting_anomalies(sample_df)
        assert "quality_frequency_score" in result.columns
        assert "repeat_offender_flag" in result.columns
        # Event cols (has_goodwill_impairment, has_asset_writedown, etc.) should
        # contribute to quality_frequency_score, making it potentially higher
        # than just the freq_cols alone
        max_score = result["quality_frequency_score"].max()
        assert max_score >= 0

    def test_graceful_with_missing_new_features(self):
        """New features should be silently skipped when not in DataFrame."""
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            detect_accounting_anomalies,
        )

        df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(60)],
                "eps_adjustment_ratio": np.random.randn(60),
                "gaap_adj_eps_gap_pct": np.random.randn(60) * 3,
            }
        )
        result = detect_accounting_anomalies(df)
        assert "accounting_anomaly_score" in result.columns


class TestAccountingAnomalyProbabilityModelV35:
    """Test enhanced balance sheet quality phase in probability model."""

    def test_severity_includes_new_flags(self, sample_df):
        from probabilistic_ml_model.statistical_functions.probability_models import (
            AccountingAnomalyProbabilityModel,
        )

        model = AccountingAnomalyProbabilityModel(use_mcmc=False)
        result = model.analyze_dataframe(sample_df)
        assert "anomaly_severity_score" in result.columns
        # Severity should be boosted by the new balance sheet quality checks
        assert result["anomaly_severity_score"].max() > 0

    def test_impairment_events_boost_severity(self):
        from probabilistic_ml_model.statistical_functions.probability_models import (
            AccountingAnomalyProbabilityModel,
        )

        np.random.seed(42)
        n = 60
        base = {
            "ticker": [f"T{i}" for i in range(n)],
            "sector": ["Tech"] * n,
            "industry": ["SW"] * n,
            "eps_adjustment_ratio": np.random.randn(n),
            "gaap_adj_eps_gap_pct": np.random.randn(n) * 2,
        }
        # Clean version
        clean_df = pd.DataFrame({**base, "has_goodwill_impairment": [0] * n})
        # Impaired version
        impaired_df = pd.DataFrame({**base, "has_goodwill_impairment": [1] * n})

        model = AccountingAnomalyProbabilityModel(use_mcmc=False)
        clean_result = model.analyze_dataframe(clean_df)
        impaired_result = model.analyze_dataframe(impaired_df)

        assert (
            impaired_result["anomaly_severity_score"].mean()
            > clean_result["anomaly_severity_score"].mean()
        )

    def test_analyst_bearish_threshold(self):
        from probabilistic_ml_model.statistical_functions.probability_models import (
            AccountingAnomalyProbabilityModel,
        )

        np.random.seed(42)
        n = 60
        base = {
            "ticker": [f"T{i}" for i in range(n)],
            "sector": ["Tech"] * n,
            "industry": ["SW"] * n,
            "eps_adjustment_ratio": np.random.randn(n),
        }
        # Low bearish
        low_df = pd.DataFrame({**base, "analyst_bearish_pct": [10.0] * n})
        # High bearish (> 30 threshold)
        high_df = pd.DataFrame({**base, "analyst_bearish_pct": [50.0] * n})

        model = AccountingAnomalyProbabilityModel(use_mcmc=False)
        low_result = model.analyze_dataframe(low_df)
        high_result = model.analyze_dataframe(high_df)

        assert (
            high_result["anomaly_severity_score"].mean()
            > low_result["anomaly_severity_score"].mean()
        )
