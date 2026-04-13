"""
Tests for the visualization ↔ FeatureViewCatalog integration.

Validates:
- VIZ_REQUIREMENTS registry completeness and structure
- Column alias resolution via resolve_column_from_catalog
- columns_for_viz fallback and catalog-driven modes
- _shared.py MV_COLUMN_ALIASES derivation from catalog
- resolve_column delegation chain (catalog → legacy extras)
- Visualization modules importing catalog-derived column lists
"""

from __future__ import annotations

import pandas as pd
import pytest

from probabilistic_ml_model.data_utils.feature_catalog import (
    FeatureViewCatalog,
    VisualizationFeatureRequirement,
    VIZ_REQUIREMENTS,
    columns_for_viz,
    get_column_aliases,
    resolve_column_from_catalog,
)


# ═══════════════════════════════════════════════════════════════════════════════
# VIZ_REQUIREMENTS registry tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVizRequirementsRegistry:
    """Tests for the VIZ_REQUIREMENTS registry structure."""

    def test_all_expected_keys_present(self):
        expected = {
            "quality_risk",
            "valuation",
            "growth_analysis",
            "earnings_quality",
            "expected_returns",
            "probability",
        }
        assert expected == set(VIZ_REQUIREMENTS.keys())

    def test_all_entries_are_visualization_feature_requirement(self):
        for key, req in VIZ_REQUIREMENTS.items():
            assert isinstance(req, VisualizationFeatureRequirement), (
                f"{key} is not a VisualizationFeatureRequirement"
            )

    def test_all_entries_have_viz_name(self):
        for key, req in VIZ_REQUIREMENTS.items():
            assert req.viz_name, f"{key} has empty viz_name"

    def test_all_entries_have_required_categories(self):
        for key, req in VIZ_REQUIREMENTS.items():
            assert len(req.required_categories) > 0, (
                f"{key} has no required_categories"
            )

    def test_all_entries_have_fallback_columns(self):
        for key, req in VIZ_REQUIREMENTS.items():
            assert len(req.fallback_columns) > 0, (
                f"{key} has no fallback_columns"
            )

    def test_quality_risk_fallbacks(self):
        req = VIZ_REQUIREMENTS["quality_risk"]
        assert "piotroski_f_score" in req.fallback_columns
        assert "altman_z_score" in req.fallback_columns
        assert "distress_risk_score" in req.fallback_columns

    def test_valuation_fallbacks(self):
        req = VIZ_REQUIREMENTS["valuation"]
        assert "p_e_ratio" in req.fallback_columns
        assert "ev_ebitda_ratio" in req.fallback_columns

    def test_growth_analysis_fallbacks(self):
        req = VIZ_REQUIREMENTS["growth_analysis"]
        assert "revenue_growth_yoy" in req.fallback_columns
        assert "eps_yoy_growth" in req.fallback_columns

    def test_earnings_quality_fallbacks(self):
        req = VIZ_REQUIREMENTS["earnings_quality"]
        assert "eps_surprise_pct" in req.fallback_columns
        assert "earnings_quality_composite" in req.fallback_columns

    def test_expected_returns_fallbacks(self):
        req = VIZ_REQUIREMENTS["expected_returns"]
        assert "implied_return_mc" in req.fallback_columns
        assert "implied_return_pt" in req.fallback_columns

    def test_probability_fallbacks(self):
        req = VIZ_REQUIREMENTS["probability"]
        assert "posterior_beat_prob" in req.fallback_columns
        assert "ruin_probability" in req.fallback_columns


# ═══════════════════════════════════════════════════════════════════════════════
# Column alias resolution tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestColumnAliasResolution:
    """Tests for resolve_column_from_catalog and get_column_aliases."""

    def test_get_column_aliases_returns_dict(self):
        aliases = get_column_aliases()
        assert isinstance(aliases, dict)
        assert len(aliases) > 0

    def test_growth_aliases_present(self):
        aliases = get_column_aliases()
        assert "revenue_growth_yoy" in aliases
        assert "revenue_yoy_growth" in aliases["revenue_growth_yoy"]

    def test_resolve_direct_column(self):
        df = pd.DataFrame({"piotroski_f_score": [7]})
        assert resolve_column_from_catalog(df, "piotroski_f_score") == "piotroski_f_score"

    def test_resolve_via_alias(self):
        df = pd.DataFrame({"f_score": [7]})
        assert resolve_column_from_catalog(df, "piotroski_f_score") == "f_score"

    def test_resolve_revenue_alias(self):
        df = pd.DataFrame({"revenue_yoy_growth": [10.5]})
        assert resolve_column_from_catalog(df, "revenue_growth_yoy") == "revenue_yoy_growth"

    def test_resolve_missing_returns_none(self):
        df = pd.DataFrame({"unrelated_col": [1]})
        assert resolve_column_from_catalog(df, "piotroski_f_score") is None

    def test_resolve_unknown_logical_name(self):
        df = pd.DataFrame({"col": [1]})
        assert resolve_column_from_catalog(df, "nonexistent_metric_xyz") is None

    def test_resolve_dividend_yield_alias(self):
        df = pd.DataFrame({"valuation_dividend_yield": [2.5]})
        assert resolve_column_from_catalog(df, "dividend_yield") == "valuation_dividend_yield"

    def test_resolve_eps_beat_count_alias(self):
        df = pd.DataFrame({"eps_positive_years": [5]})
        assert resolve_column_from_catalog(df, "eps_beat_count") == "eps_positive_years"

    def test_resolve_earnings_quality_alias(self):
        df = pd.DataFrame({"earnings_quality_score": [75]})
        assert resolve_column_from_catalog(df, "earnings_quality_composite") == "earnings_quality_score"


# ═══════════════════════════════════════════════════════════════════════════════
# columns_for_viz tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestColumnsForViz:
    """Tests for columns_for_viz fallback and catalog-driven modes."""

    def test_fallback_quality_risk(self):
        cols = columns_for_viz("quality_risk")
        assert "piotroski_f_score" in cols
        assert "altman_z_score" in cols

    def test_fallback_valuation(self):
        cols = columns_for_viz("valuation")
        assert "p_e_ratio" in cols
        assert len(cols) >= 5

    def test_fallback_growth_analysis(self):
        cols = columns_for_viz("growth_analysis")
        assert "revenue_growth_yoy" in cols

    def test_fallback_earnings_quality(self):
        cols = columns_for_viz("earnings_quality")
        assert "eps_surprise_pct" in cols

    def test_fallback_expected_returns(self):
        cols = columns_for_viz("expected_returns")
        assert "implied_return_mc" in cols

    def test_fallback_probability(self):
        cols = columns_for_viz("probability")
        assert "posterior_beat_prob" in cols

    def test_unknown_key_returns_empty(self):
        assert columns_for_viz("nonexistent_viz") == []

    def test_catalog_driven_mode(self):
        catalog = FeatureViewCatalog()
        catalog.category_columns = {
            "Quality & Risk": ["custom_quality_col", "custom_risk_col"],
        }
        catalog._loaded = True
        cols = columns_for_viz("quality_risk", catalog=catalog)
        assert cols == ["custom_quality_col", "custom_risk_col"]

    def test_catalog_driven_fallback_on_empty(self):
        """When catalog is loaded but has no matching categories, use fallbacks."""
        catalog = FeatureViewCatalog()
        catalog.category_columns = {"Unrelated Category": ["col_a"]}
        catalog._loaded = True
        cols = columns_for_viz("quality_risk", catalog=catalog)
        assert "piotroski_f_score" in cols

    def test_catalog_deduplication(self):
        catalog = FeatureViewCatalog()
        catalog.category_columns = {
            "Quality & Risk": ["col_a", "col_b", "col_a"],
        }
        catalog._loaded = True
        cols = columns_for_viz("quality_risk", catalog=catalog)
        assert cols == ["col_a", "col_b"]


# ═══════════════════════════════════════════════════════════════════════════════
# _shared.py integration tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSharedModuleIntegration:
    """Tests for _shared.py MV_COLUMN_ALIASES and resolve_column."""

    def test_mv_column_aliases_populated(self):
        from probabilistic_ml_model.visualizations._shared import MV_COLUMN_ALIASES
        assert isinstance(MV_COLUMN_ALIASES, dict)
        assert len(MV_COLUMN_ALIASES) >= 15

    def test_mv_aliases_include_catalog_entries(self):
        from probabilistic_ml_model.visualizations._shared import MV_COLUMN_ALIASES
        assert "revenue_growth_yoy" in MV_COLUMN_ALIASES
        assert "piotroski_f_score" in MV_COLUMN_ALIASES
        assert "dividend_yield" in MV_COLUMN_ALIASES

    def test_mv_aliases_include_legacy_extras(self):
        from probabilistic_ml_model.visualizations._shared import MV_COLUMN_ALIASES
        assert "inventory_turnover" in MV_COLUMN_ALIASES
        assert "beneish_m_score" in MV_COLUMN_ALIASES
        assert "quality_composite_score" in MV_COLUMN_ALIASES

    def test_resolve_column_uses_catalog(self):
        from probabilistic_ml_model.visualizations._shared import resolve_column
        df = pd.DataFrame({"f_score": [7]})
        assert resolve_column(df, "piotroski_f_score") == "f_score"

    def test_resolve_column_uses_legacy_extras(self):
        from probabilistic_ml_model.visualizations._shared import resolve_column
        df = pd.DataFrame({"inventory_turnover_itf": [3.5]})
        assert resolve_column(df, "inventory_turnover") == "inventory_turnover_itf"

    def test_resolve_column_direct_match(self):
        from probabilistic_ml_model.visualizations._shared import resolve_column
        df = pd.DataFrame({"altman_z_score": [2.5]})
        assert resolve_column(df, "altman_z_score") == "altman_z_score"

    def test_resolve_column_missing_returns_none(self):
        from probabilistic_ml_model.visualizations._shared import resolve_column
        df = pd.DataFrame({"unrelated": [1]})
        assert resolve_column(df, "nonexistent_metric") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization module import tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualizationModuleImports:
    """Verify that refactored visualization modules import successfully."""

    def test_import_valuation(self):
        from probabilistic_ml_model.visualizations.valuation import DEFAULT_VALUATION_METRICS
        assert len(DEFAULT_VALUATION_METRICS) == 5
        assert "p_e_ratio" in DEFAULT_VALUATION_METRICS

    def test_import_quality_risk(self):
        from probabilistic_ml_model.visualizations.quality_risk import _QUALITY_RISK_COLS
        assert len(_QUALITY_RISK_COLS) >= 5
        assert "piotroski_f_score" in _QUALITY_RISK_COLS

    def test_import_growth_analysis(self):
        from probabilistic_ml_model.visualizations import growth_analysis
        assert hasattr(growth_analysis, "create_growth_waterfall_chart")

    def test_import_earnings_quality(self):
        from probabilistic_ml_model.visualizations import earnings_quality
        assert hasattr(earnings_quality, "create_earnings_quality_decomposition")

    def test_import_expected_returns_viz(self):
        from probabilistic_ml_model.visualizations import expected_returns_viz
        assert hasattr(expected_returns_viz, "create_model_dispersion_dashboard")

    def test_import_probability_viz(self):
        from probabilistic_ml_model.visualizations.probability_viz import _PROBABILITY_VIZ_COLS
        assert len(_PROBABILITY_VIZ_COLS) >= 4

    def test_import_shared_exports(self):
        from probabilistic_ml_model.visualizations._shared import (
            PLOTLY_TEMPLATE,
            COLORS,
            MV_COLUMN_ALIASES,
            resolve_column,
            create_no_data_figure,
        )
        assert PLOTLY_TEMPLATE == "plotly_dark"
        assert len(COLORS) == 8

    def test_init_exports_shared(self):
        from probabilistic_ml_model.visualizations import MV_COLUMN_ALIASES, resolve_column
        assert isinstance(MV_COLUMN_ALIASES, dict)
        assert callable(resolve_column)
