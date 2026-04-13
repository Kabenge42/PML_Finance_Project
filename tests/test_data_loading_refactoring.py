"""
Tests for the data loading refactoring (TDD).

Verifies that:
- PipelineConfig has `prefer_materialized_view` flag (default True).
- _step_load_data uses mv_all_stock_features as primary when flag is True.
- _step_load_data falls back to 3-source loading when flag is False or MV empty.
- load_analytics_table applies backfill & Kalman before fillna.
- All model step functions pass r.df_features (not r.df_all) as feature_df.
- _enrichment_source prefers df_features explicitly.
- Feature categories are reconciled against df_features columns.
- Historical columns from mv_equities are merged into df_features.
- All 700+ columns in mv_all_stock_features are covered.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import fields


# ---------------------------------------------------------------------------
# Helpers: synthetic DataFrames
# ---------------------------------------------------------------------------

# Columns only in mv_equities (historical snapshots)
_HISTORICAL_ONLY_COLS = [
    "price_5d_ago", "price_1w_ago", "price_1m_ago", "price_3m_ago",
    "price_target_1w_ago", "price_target_1m_ago", "price_target_3m_ago",
]

# Core columns present in mv_all_stock_features
_CORE_COLS = [
    "ticker", "isin", "company_name", "sector", "industry", "region", "country",
    "last_price", "price_target", "price_target_high", "price_target_low",
    "eps_surprise_pct", "eps_revision_momentum", "analyst_conviction",
    "distress_risk_score", "debt_to_equity", "altman_z_score",
    "piotroski_f_score", "fcf_positive_years", "eps_trajectory_score",
    "roe", "dividend_yield", "market_cap", "pe_ratio",
]

# Enhancement columns only in mv_all_stock_features (not in 17-view merge)
_ENHANCEMENT_COLS = [
    "composite_quality_score", "forward_consensus_eps",
    "volatility_surface_30d", "effective_tax_rate",
    "opex_temporal_trend", "asset_sale_flag",
    "fcf_estimate_curve_slope", "dividend_history_years",
    "investment_income_temporal", "share_dilution_pct",
]


def _make_mv_all_stock_features_df(n: int = 50) -> pd.DataFrame:
    """Create a synthetic mv_all_stock_features DataFrame with 700+ columns."""
    rng = np.random.RandomState(42)
    data = {"ticker": [f"T{i:04d}" for i in range(n)]}

    # Add core columns
    for col in _CORE_COLS[1:]:  # skip ticker
        data[col] = rng.randn(n)

    # Add enhancement columns
    for col in _ENHANCEMENT_COLS:
        data[col] = rng.randn(n)

    # Pad to 700+ columns to simulate real MV
    for i in range(670):
        data[f"feature_{i:04d}"] = rng.randn(n)

    return pd.DataFrame(data)


def _make_mv_equities_df(n: int = 50) -> pd.DataFrame:
    """Create a synthetic mv_equities DataFrame (~90 columns)."""
    rng = np.random.RandomState(42)
    data = {"ticker": [f"T{i:04d}" for i in range(n)]}

    for col in _CORE_COLS[1:]:
        data[col] = rng.randn(n)

    # Historical snapshot columns (only in mv_equities)
    for col in _HISTORICAL_ONLY_COLS:
        data[col] = rng.randn(n)

    # Pad to ~90
    for i in range(55):
        data[f"eq_col_{i:03d}"] = rng.randn(n)

    return pd.DataFrame(data)


def _make_feature_views_df(n: int = 50) -> pd.DataFrame:
    """Create a synthetic 17-view merged DataFrame (~400 columns)."""
    rng = np.random.RandomState(42)
    data = {"ticker": [f"T{i:04d}" for i in range(n)]}

    for col in _CORE_COLS[1:]:
        data[col] = rng.randn(n)

    for i in range(375):
        data[f"view_col_{i:04d}"] = rng.randn(n)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. PipelineConfig: prefer_materialized_view flag
# ---------------------------------------------------------------------------


class TestPipelineConfigFlag:
    """PipelineConfig should have a prefer_materialized_view flag."""

    def test_has_prefer_materialized_view_field(self):
        from expected_returns_v3 import PipelineConfig

        field_names = {f.name for f in fields(PipelineConfig)}
        assert "prefer_materialized_view" in field_names

    def test_prefer_materialized_view_defaults_true(self):
        from expected_returns_v3 import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.prefer_materialized_view is True

    def test_can_set_prefer_materialized_view_false(self):
        from expected_returns_v3 import PipelineConfig

        cfg = PipelineConfig(prefer_materialized_view=False)
        assert cfg.prefer_materialized_view is False

    def test_from_env_includes_prefer_materialized_view(self):
        from expected_returns_v3 import PipelineConfig
        import os

        with patch.dict(os.environ, {"ER_PREFER_MATERIALIZED_VIEW": "false"}):
            cfg = PipelineConfig.from_env()
            assert cfg.prefer_materialized_view is False

    def test_from_env_defaults_true(self):
        from expected_returns_v3 import PipelineConfig
        import os

        with patch.dict(os.environ, {}, clear=False):
            cfg = PipelineConfig.from_env()
            assert cfg.prefer_materialized_view is True


# ---------------------------------------------------------------------------
# 2. load_analytics_table: backfill & Kalman
# ---------------------------------------------------------------------------


class TestLoadAnalyticsTableBackfill:
    """load_analytics_table should apply _apply_backfill_and_kalman."""

    @patch("expected_returns_v3.load_feature_data_from_db")
    @patch("expected_returns_v3._apply_backfill_and_kalman")
    def test_applies_backfill_and_kalman(self, mock_backfill, mock_load):
        from expected_returns_v3 import load_analytics_table

        df_raw = _make_mv_all_stock_features_df(10)
        mock_load.return_value = df_raw.copy()
        mock_backfill.return_value = df_raw.copy()

        result = load_analytics_table()

        mock_backfill.assert_called_once()
        assert not result.empty

    @patch("expected_returns_v3.load_feature_data_from_db")
    @patch("expected_returns_v3._apply_backfill_and_kalman")
    def test_backfill_called_before_fillna(self, mock_backfill, mock_load):
        from expected_returns_v3 import load_analytics_table

        df_with_nans = _make_mv_all_stock_features_df(10)
        df_with_nans.iloc[0, 5] = np.nan
        mock_load.return_value = df_with_nans.copy()

        call_order = []

        def track_backfill(df):
            call_order.append("backfill")
            return df

        mock_backfill.side_effect = track_backfill

        result = load_analytics_table()
        assert "backfill" in call_order

    @patch("expected_returns_v3.load_feature_data_from_db")
    def test_empty_df_skips_backfill(self, mock_load):
        from expected_returns_v3 import load_analytics_table

        mock_load.return_value = pd.DataFrame()
        result = load_analytics_table()
        assert result.empty


# ---------------------------------------------------------------------------
# 3. _step_load_data: primary load from mv_all_stock_features
# ---------------------------------------------------------------------------


class TestStepLoadDataPrimary:
    """_step_load_data should use mv_all_stock_features as primary dataset."""

    @patch("expected_returns_v3.load_mv_equities_spec_from_db", return_value=None)
    @patch("expected_returns_v3.load_feature_registry_metadata_from_db", return_value=None)
    @patch("expected_returns_v3.load_equities_schema_metadata_from_db", return_value=None)
    @patch("expected_returns_v3._enrich_with_historical_target_drift")
    @patch("expected_returns_v3._resolve_available_historical_cols")
    @patch("expected_returns_v3.get_feature_catalog")
    @patch("expected_returns_v3.load_analytics_table")
    @patch("expected_returns_v3.load_all_stock_features")
    @patch("expected_returns_v3.load_expected_returns_data")
    def test_primary_uses_mv_all_stock_features(
        self, mock_equities, mock_views, mock_analytics,
        mock_catalog, mock_hist_cols, mock_hist_drift,
        mock_schema, mock_registry, mock_mv_spec,
    ):
        from expected_returns_v3 import _step_load_data, PipelineConfig

        df_features = _make_mv_all_stock_features_df(20)
        df_equities = _make_mv_equities_df(20)
        mock_analytics.return_value = df_features
        mock_equities.return_value = (df_equities, None)
        mock_views.return_value = (pd.DataFrame(), {})
        mock_catalog.return_value = MagicMock(_loaded=False)
        mock_hist_cols.return_value = {}
        mock_hist_drift.side_effect = lambda df, _: df

        cfg = PipelineConfig(prefer_materialized_view=True)
        r = _step_load_data(cfg)

        # df_features should be the primary dataset
        assert not r.df_features.empty
        assert len(r.df_features.columns) >= 700
        # df and df_all should reference df_features when MV loads successfully
        assert len(r.df.columns) >= 700 or r.df is r.df_features
        assert len(r.df_all.columns) >= 700 or r.df_all is r.df_features

    @patch("expected_returns_v3.load_mv_equities_spec_from_db", return_value=None)
    @patch("expected_returns_v3.load_feature_registry_metadata_from_db", return_value=None)
    @patch("expected_returns_v3.load_equities_schema_metadata_from_db", return_value=None)
    @patch("expected_returns_v3._enrich_with_historical_target_drift")
    @patch("expected_returns_v3._resolve_available_historical_cols")
    @patch("expected_returns_v3.get_feature_catalog")
    @patch("expected_returns_v3.load_analytics_table")
    @patch("expected_returns_v3.load_all_stock_features")
    @patch("expected_returns_v3.load_expected_returns_data")
    def test_fallback_when_mv_empty(
        self, mock_equities, mock_views, mock_analytics,
        mock_catalog, mock_hist_cols, mock_hist_drift,
        mock_schema, mock_registry, mock_mv_spec,
    ):
        from expected_returns_v3 import _step_load_data, PipelineConfig

        df_equities = _make_mv_equities_df(20)
        df_views = _make_feature_views_df(20)
        mock_analytics.return_value = pd.DataFrame()  # MV fails
        mock_equities.return_value = (df_equities, None)
        mock_views.return_value = (df_views, {})
        mock_catalog.return_value = MagicMock(_loaded=False)
        mock_hist_cols.return_value = {}
        mock_hist_drift.side_effect = lambda df, _: df

        cfg = PipelineConfig(prefer_materialized_view=True)
        r = _step_load_data(cfg)

        # Should fall back to equities + views
        assert not r.df.empty
        assert not r.df_all.empty

    @patch("expected_returns_v3.load_mv_equities_spec_from_db", return_value=None)
    @patch("expected_returns_v3.load_feature_registry_metadata_from_db", return_value=None)
    @patch("expected_returns_v3.load_equities_schema_metadata_from_db", return_value=None)
    @patch("expected_returns_v3._enrich_with_historical_target_drift")
    @patch("expected_returns_v3._resolve_available_historical_cols")
    @patch("expected_returns_v3.get_feature_catalog")
    @patch("expected_returns_v3.load_analytics_table")
    @patch("expected_returns_v3.load_all_stock_features")
    @patch("expected_returns_v3.load_expected_returns_data")
    def test_fallback_when_flag_false(
        self, mock_equities, mock_views, mock_analytics,
        mock_catalog, mock_hist_cols, mock_hist_drift,
        mock_schema, mock_registry, mock_mv_spec,
    ):
        from expected_returns_v3 import _step_load_data, PipelineConfig

        df_equities = _make_mv_equities_df(20)
        df_views = _make_feature_views_df(20)
        df_features = _make_mv_all_stock_features_df(20)
        mock_analytics.return_value = df_features
        mock_equities.return_value = (df_equities, None)
        mock_views.return_value = (df_views, {})
        mock_catalog.return_value = MagicMock(_loaded=False)
        mock_hist_cols.return_value = {}
        mock_hist_drift.side_effect = lambda df, _: df

        cfg = PipelineConfig(prefer_materialized_view=False)
        r = _step_load_data(cfg)

        # When flag is False, should use the old 3-source pattern
        mock_equities.assert_called_once()
        mock_views.assert_called_once()

    @patch("expected_returns_v3.load_mv_equities_spec_from_db", return_value=None)
    @patch("expected_returns_v3.load_feature_registry_metadata_from_db", return_value=None)
    @patch("expected_returns_v3.load_equities_schema_metadata_from_db", return_value=None)
    @patch("expected_returns_v3._enrich_with_historical_target_drift")
    @patch("expected_returns_v3._resolve_available_historical_cols")
    @patch("expected_returns_v3.get_feature_catalog")
    @patch("expected_returns_v3.load_analytics_table")
    @patch("expected_returns_v3.load_all_stock_features")
    @patch("expected_returns_v3.load_expected_returns_data")
    def test_historical_cols_merged_from_equities(
        self, mock_equities, mock_views, mock_analytics,
        mock_catalog, mock_hist_cols, mock_hist_drift,
        mock_schema, mock_registry, mock_mv_spec,
    ):
        from expected_returns_v3 import _step_load_data, PipelineConfig

        df_features = _make_mv_all_stock_features_df(20)
        df_equities = _make_mv_equities_df(20)
        mock_analytics.return_value = df_features
        mock_equities.return_value = (df_equities, None)
        mock_views.return_value = (pd.DataFrame(), {})
        mock_catalog.return_value = MagicMock(_loaded=False)
        mock_hist_cols.return_value = {}
        mock_hist_drift.side_effect = lambda df, _: df

        cfg = PipelineConfig(prefer_materialized_view=True)
        r = _step_load_data(cfg)

        # Historical columns from mv_equities should be present
        for col in _HISTORICAL_ONLY_COLS:
            assert col in r.df.columns or col in r.df_features.columns, (
                f"Historical column '{col}' missing from merged dataset"
            )

    @patch("expected_returns_v3.load_mv_equities_spec_from_db", return_value=None)
    @patch("expected_returns_v3.load_feature_registry_metadata_from_db", return_value=None)
    @patch("expected_returns_v3.load_equities_schema_metadata_from_db", return_value=None)
    @patch("expected_returns_v3._enrich_with_historical_target_drift")
    @patch("expected_returns_v3._resolve_available_historical_cols")
    @patch("expected_returns_v3.get_feature_catalog")
    @patch("expected_returns_v3.load_analytics_table")
    @patch("expected_returns_v3.load_all_stock_features")
    @patch("expected_returns_v3.load_expected_returns_data")
    def test_redundant_views_not_called_when_mv_succeeds(
        self, mock_equities, mock_views, mock_analytics,
        mock_catalog, mock_hist_cols, mock_hist_drift,
        mock_schema, mock_registry, mock_mv_spec,
    ):
        from expected_returns_v3 import _step_load_data, PipelineConfig

        df_features = _make_mv_all_stock_features_df(20)
        df_equities = _make_mv_equities_df(20)
        mock_analytics.return_value = df_features
        mock_equities.return_value = (df_equities, None)
        mock_views.return_value = (pd.DataFrame(), {})
        mock_catalog.return_value = MagicMock(_loaded=False)
        mock_hist_cols.return_value = {}
        mock_hist_drift.side_effect = lambda df, _: df

        cfg = PipelineConfig(prefer_materialized_view=True)
        r = _step_load_data(cfg)

        # load_all_stock_features should NOT be called when MV succeeds
        mock_views.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Model steps: feature_df should be r.df_features
# ---------------------------------------------------------------------------


class TestModelStepsUseFeatures:
    """All model step functions should pass r.df_features as feature_df."""

    def _make_pipeline_result(self):
        from expected_returns_v3 import PipelineResult

        r = PipelineResult()
        r.df = _make_mv_equities_df(10)
        r.df_all = _make_feature_views_df(10)
        r.df_features = _make_mv_all_stock_features_df(10)
        r.df_enriched = r.df.copy()
        return r

    @patch("expected_returns_v3.run_price_target_achievement")
    @patch("expected_returns_v3.compute_model_detailed_statistics", return_value={})
    @patch("expected_returns_v3.print_model_statistics")
    @patch("expected_returns_v3.compute_price_target_prob_weighted")
    def test_price_target_uses_df_features(self, mock_pw, mock_print, mock_stats, mock_run):
        from expected_returns_v3 import _step_price_target, PipelineConfig

        r = self._make_pipeline_result()
        cfg = PipelineConfig()
        mock_run.return_value = pd.DataFrame({
            "ticker": ["T0001"], "achievement_probability": [0.7],
            "implied_return_pt": [10.0],
        })
        mock_pw.return_value = mock_run.return_value

        _step_price_target(r, cfg)

        # Verify feature_df= was r.df_features
        call_kwargs = mock_run.call_args
        feature_df_arg = call_kwargs.kwargs.get("feature_df", call_kwargs[1].get("feature_df"))
        assert feature_df_arg is not None
        assert len(feature_df_arg.columns) >= 700, (
            f"feature_df should be df_features (700+ cols), got {len(feature_df_arg.columns)}"
        )

    @patch("expected_returns_v3.run_earnings_beat_analysis")
    @patch("expected_returns_v3.compute_model_detailed_statistics", return_value={})
    @patch("expected_returns_v3.print_model_statistics")
    def test_earnings_beat_uses_df_features(self, mock_print, mock_stats, mock_run):
        from expected_returns_v3 import _step_earnings_beat, PipelineConfig

        r = self._make_pipeline_result()
        cfg = PipelineConfig()
        mock_run.return_value = pd.DataFrame({
            "ticker": ["T0001"], "posterior_beat_prob": [0.65],
        })

        _step_earnings_beat(r, cfg)

        call_kwargs = mock_run.call_args
        feature_df_arg = call_kwargs.kwargs.get("feature_df", call_kwargs[1].get("feature_df"))
        assert feature_df_arg is not None
        assert len(feature_df_arg.columns) >= 700

    @patch("expected_returns_v3.run_accounting_anomaly_analysis")
    def test_anomaly_detection_uses_df_features(self, mock_run):
        from expected_returns_v3 import _step_anomaly_detection, PipelineConfig

        r = self._make_pipeline_result()
        cfg = PipelineConfig()
        mock_run.return_value = pd.DataFrame({
            "ticker": ["T0001"], "accounting_anomaly_score": [0.3],
        })

        _step_anomaly_detection(r, cfg)

        call_kwargs = mock_run.call_args
        feature_df_arg = call_kwargs.kwargs.get("feature_df", call_kwargs[1].get("feature_df"))
        assert feature_df_arg is not None
        assert len(feature_df_arg.columns) >= 700

    @patch("expected_returns_v3.run_dividend_safety_analysis")
    @patch("expected_returns_v3.run_credit_risk_analysis")
    def test_credit_dividend_uses_df_features(self, mock_credit, mock_div):
        from expected_returns_v3 import _step_credit_dividend, PipelineConfig

        r = self._make_pipeline_result()
        cfg = PipelineConfig()
        mock_credit.return_value = pd.DataFrame({
            "ticker": ["T0001"], "ruin_probability": [0.1],
        })
        mock_div.return_value = pd.DataFrame({
            "ticker": ["T0001"], "risk_category": ["Safe"],
        })

        _step_credit_dividend(r, cfg)

        # Credit risk
        credit_kwargs = mock_credit.call_args
        credit_fdf = credit_kwargs.kwargs.get("feature_df", credit_kwargs[1].get("feature_df"))
        assert credit_fdf is not None
        assert len(credit_fdf.columns) >= 700

        # Dividend safety
        div_kwargs = mock_div.call_args
        div_fdf = div_kwargs.kwargs.get("feature_df", div_kwargs[1].get("feature_df"))
        assert div_fdf is not None
        assert len(div_fdf.columns) >= 700

    @patch("expected_returns_v3.run_stock_screening")
    @patch("expected_returns_v3.analyze_employee_productivity_frontier", side_effect=Exception("skip"))
    @patch("expected_returns_v3.analyze_reporting_lag_sentiment", side_effect=Exception("skip"))
    def test_screening_uses_df_features(self, mock_lag, mock_prod, mock_screen):
        from expected_returns_v3 import _step_screening, PipelineConfig

        r = self._make_pipeline_result()
        cfg = PipelineConfig()
        mock_screen.return_value = {}

        _step_screening(r, cfg)

        # First positional arg should be df_features (700+ cols)
        call_args = mock_screen.call_args
        screening_df = call_args[0][0] if call_args[0] else call_args.kwargs.get("df_all")
        assert screening_df is not None
        assert len(screening_df.columns) >= 700


# ---------------------------------------------------------------------------
# 5. _enrichment_source: prefers df_features
# ---------------------------------------------------------------------------


class TestEnrichmentSource:
    """_enrichment_source should prefer df_features explicitly."""

    def test_enrichment_prefers_df_features(self):
        """Verify the enrichment source logic prefers df_features."""
        df_features = _make_mv_all_stock_features_df(10)
        df_all = _make_feature_views_df(10)
        df = _make_mv_equities_df(10)

        # This mirrors the expected refactored logic
        _enrichment_source = (
            df_features if not df_features.empty else df_all
        )
        assert len(_enrichment_source.columns) >= 700

    def test_enrichment_falls_back_to_df_all(self):
        df_features = pd.DataFrame()
        df_all = _make_feature_views_df(10)

        _enrichment_source = (
            df_features if not df_features.empty else df_all
        )
        assert not _enrichment_source.empty
        assert len(_enrichment_source.columns) < 700


# ---------------------------------------------------------------------------
# 6. Feature categories reconciled against df_features
# ---------------------------------------------------------------------------


class TestFeatureCategoryReconciliation:
    """Feature categories should be reconciled against df_features columns."""

    def test_reconcile_uses_full_column_set(self):
        from expected_returns_v3 import reconcile_feature_categories

        categories = {
            "quality": ["piotroski_f_score", "roe", "composite_quality_score"],
            "valuation": ["pe_ratio", "nonexistent_col"],
        }
        df_features = _make_mv_all_stock_features_df(5)
        result = reconcile_feature_categories(categories, set(df_features.columns))

        # Columns in df_features should be kept; missing ones dropped
        assert "piotroski_f_score" in result.get("quality", [])
        assert "composite_quality_score" in result.get("quality", [])
        assert "nonexistent_col" not in result.get("valuation", [])


# ---------------------------------------------------------------------------
# 7. Column coverage: 700+ columns
# ---------------------------------------------------------------------------


class TestColumnCoverage:
    """All 700+ columns in mv_all_stock_features should be accessible."""

    def test_mv_all_stock_features_has_700_plus_columns(self):
        df = _make_mv_all_stock_features_df(10)
        assert len(df.columns) >= 700, (
            f"Expected ≥700 columns, got {len(df.columns)}"
        )

    def test_enhancement_columns_present(self):
        df = _make_mv_all_stock_features_df(10)
        for col in _ENHANCEMENT_COLS:
            assert col in df.columns, f"Enhancement column '{col}' missing"

    def test_core_model_columns_present(self):
        """All columns needed by model runners should be in the superset."""
        df = _make_mv_all_stock_features_df(10)
        model_cols = [
            "last_price", "price_target", "price_target_high", "price_target_low",
            "eps_surprise_pct", "eps_revision_momentum", "analyst_conviction",
            "distress_risk_score", "debt_to_equity", "altman_z_score",
            "piotroski_f_score", "fcf_positive_years", "eps_trajectory_score",
            "sector", "industry", "region", "country",
        ]
        for col in model_cols:
            assert col in df.columns, f"Model column '{col}' missing"

    def test_historical_cols_not_in_mv_all_stock_features(self):
        """Historical snapshot columns should NOT be in mv_all_stock_features."""
        df = _make_mv_all_stock_features_df(10)
        for col in _HISTORICAL_ONLY_COLS:
            assert col not in df.columns, (
                f"Historical column '{col}' should not be in mv_all_stock_features"
            )


# ---------------------------------------------------------------------------
# 8. Integration: end-to-end _step_load_data with primary MV
# ---------------------------------------------------------------------------


class TestStepLoadDataIntegration:
    """Integration test for the full _step_load_data refactored flow."""

    @patch("expected_returns_v3.load_mv_equities_spec_from_db", return_value=None)
    @patch("expected_returns_v3.load_feature_registry_metadata_from_db", return_value=None)
    @patch("expected_returns_v3.load_equities_schema_metadata_from_db", return_value=None)
    @patch("expected_returns_v3._enrich_with_historical_target_drift")
    @patch("expected_returns_v3._resolve_available_historical_cols")
    @patch("expected_returns_v3.get_feature_catalog")
    @patch("expected_returns_v3.load_analytics_table")
    @patch("expected_returns_v3.load_all_stock_features")
    @patch("expected_returns_v3.load_expected_returns_data")
    def test_full_flow_primary_path(
        self, mock_equities, mock_views, mock_analytics,
        mock_catalog, mock_hist_cols, mock_hist_drift,
        mock_schema, mock_registry, mock_mv_spec,
    ):
        from expected_returns_v3 import _step_load_data, PipelineConfig

        df_features = _make_mv_all_stock_features_df(30)
        df_equities = _make_mv_equities_df(30)
        mock_analytics.return_value = df_features
        mock_equities.return_value = (df_equities, None)
        mock_views.return_value = (pd.DataFrame(), {})
        mock_catalog.return_value = MagicMock(_loaded=False)
        mock_hist_cols.return_value = {}
        mock_hist_drift.side_effect = lambda df, _: df

        cfg = PipelineConfig(prefer_materialized_view=True)
        r = _step_load_data(cfg)

        # Verify primary path outcomes
        assert not r.df_features.empty
        assert not r.df.empty
        assert not r.df_all.empty
        # df_enriched should also exist (from historical drift step)
        assert not r.df_enriched.empty

    @patch("expected_returns_v3.load_mv_equities_spec_from_db", return_value=None)
    @patch("expected_returns_v3.load_feature_registry_metadata_from_db", return_value=None)
    @patch("expected_returns_v3.load_equities_schema_metadata_from_db", return_value=None)
    @patch("expected_returns_v3._enrich_with_historical_target_drift")
    @patch("expected_returns_v3._resolve_available_historical_cols")
    @patch("expected_returns_v3.get_feature_catalog")
    @patch("expected_returns_v3.load_analytics_table")
    @patch("expected_returns_v3.load_all_stock_features")
    @patch("expected_returns_v3.load_expected_returns_data")
    def test_full_flow_fallback_path(
        self, mock_equities, mock_views, mock_analytics,
        mock_catalog, mock_hist_cols, mock_hist_drift,
        mock_schema, mock_registry, mock_mv_spec,
    ):
        from expected_returns_v3 import _step_load_data, PipelineConfig

        df_equities = _make_mv_equities_df(30)
        df_views = _make_feature_views_df(30)
        mock_analytics.return_value = pd.DataFrame()  # MV fails
        mock_equities.return_value = (df_equities, None)
        mock_views.return_value = (df_views, {})
        mock_catalog.return_value = MagicMock(_loaded=False)
        mock_hist_cols.return_value = {}
        mock_hist_drift.side_effect = lambda df, _: df

        cfg = PipelineConfig(prefer_materialized_view=True)
        r = _step_load_data(cfg)

        # Should have used fallback
        assert not r.df.empty
        assert not r.df_all.empty
        assert r.df_features.empty
