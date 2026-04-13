"""
Unit tests for the schema-driven FeatureViewCatalog module.

Tests cover:
- ModelFeatureRequirement declarations and MODEL_REQUIREMENTS registry
- FeatureViewCatalog.columns_for_model (fallback and DB-loaded modes)
- auto_enrich_for_model enrichment logic
- Singleton get_feature_catalog / reset_feature_catalog lifecycle
"""

from __future__ import annotations

import pandas as pd
import pytest

from probabilistic_ml_model.data_utils.feature_catalog import (
    MODEL_REQUIREMENTS,
    FeatureViewCatalog,
    ModelFeatureRequirement,
    auto_enrich_for_model,
    get_feature_catalog,
    reset_feature_catalog,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ModelFeatureRequirement & MODEL_REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelFeatureRequirement:
    def test_frozen_dataclass(self):
        req = ModelFeatureRequirement(model_name="Test", fallback_columns=("a", "b"))
        assert req.model_name == "Test"
        assert req.fallback_columns == ("a", "b")
        with pytest.raises(AttributeError):
            req.model_name = "Changed"  # type: ignore[misc]

    def test_defaults(self):
        req = ModelFeatureRequirement(model_name="X")
        assert req.required_categories == ()
        assert req.optional_categories == ()
        assert req.fallback_columns == ()


class TestModelRequirementsRegistry:
    def test_all_expected_models_present(self):
        expected_keys = {
            "price_target_achievement",
            "credit_risk",
            "dividend_safety",
            "earnings_beat",
            "accounting_anomaly",
        }
        assert expected_keys == set(MODEL_REQUIREMENTS.keys())

    def test_all_models_have_fallback_columns(self):
        for key, req in MODEL_REQUIREMENTS.items():
            assert len(req.fallback_columns) > 0, f"{key} has no fallback columns"

    def test_all_models_have_model_name(self):
        for key, req in MODEL_REQUIREMENTS.items():
            assert req.model_name, f"{key} has empty model_name"

    def test_all_models_have_required_categories(self):
        for key, req in MODEL_REQUIREMENTS.items():
            assert len(req.required_categories) > 0, f"{key} has no required categories"


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureViewCatalog
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureViewCatalog:
    def test_initial_state(self):
        catalog = FeatureViewCatalog()
        assert catalog._loaded is False
        assert catalog.category_columns == {}
        assert catalog.view_columns == {}

    def test_columns_for_model_fallback_when_not_loaded(self):
        catalog = FeatureViewCatalog()
        cols = catalog.columns_for_model("price_target_achievement")
        expected = list(MODEL_REQUIREMENTS["price_target_achievement"].fallback_columns)
        assert cols == expected

    def test_columns_for_model_unknown_key(self):
        catalog = FeatureViewCatalog()
        assert catalog.columns_for_model("nonexistent_model") == []

    def test_columns_for_model_from_categories(self):
        catalog = FeatureViewCatalog()
        catalog.category_columns = {
            "Analyst Sentiment": ["expected_upside_pt", "analyst_conviction"],
            "Price Target Dynamics": ["pt_momentum_1m"],
            "Quality & Risk": ["beta_1y"],
        }
        catalog._loaded = True

        cols = catalog.columns_for_model("price_target_achievement")
        assert "expected_upside_pt" in cols
        assert "analyst_conviction" in cols
        assert "pt_momentum_1m" in cols
        assert "beta_1y" in cols

    def test_columns_for_model_deduplication(self):
        catalog = FeatureViewCatalog()
        catalog.category_columns = {
            "Analyst Sentiment": ["col_a", "col_b"],
            "Price Target Dynamics": ["col_b", "col_c"],
        }
        catalog._loaded = True

        cols = catalog.columns_for_model("price_target_achievement")
        assert cols.count("col_b") == 1

    def test_columns_for_model_falls_back_if_categories_empty(self):
        catalog = FeatureViewCatalog()
        catalog.category_columns = {}
        catalog._loaded = True

        cols = catalog.columns_for_model("price_target_achievement")
        expected = list(MODEL_REQUIREMENTS["price_target_achievement"].fallback_columns)
        assert cols == expected

    def test_load_from_db_no_sqlalchemy(self, monkeypatch):
        import probabilistic_ml_model.data_utils.feature_catalog as fc_mod

        monkeypatch.setattr(fc_mod, "create_engine", None)
        catalog = FeatureViewCatalog()
        catalog.load_from_db(db_url="postgresql://fake")
        assert catalog._loaded is False

    def test_load_from_db_no_db_url(self, monkeypatch):
        monkeypatch.delenv("DB_URL", raising=False)
        catalog = FeatureViewCatalog()
        catalog.load_from_db(db_url=None)
        assert catalog._loaded is False


# ═══════════════════════════════════════════════════════════════════════════════
# auto_enrich_for_model
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoEnrichForModel:
    @pytest.fixture()
    def catalog(self):
        cat = FeatureViewCatalog()
        cat._loaded = False  # will use fallback columns
        return cat

    @pytest.fixture()
    def target_df(self):
        return pd.DataFrame({
            "isin": ["US0378331005", "US5949181045", "US02079K3059"],
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "last_price": [150.0, 300.0, 140.0],
        })

    @pytest.fixture()
    def source_df(self):
        return pd.DataFrame({
            "isin": ["US0378331005", "US5949181045", "US02079K3059"],
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "expected_upside_pt": [10.0, 15.0, 20.0],
            "analyst_conviction": [0.8, 0.9, 0.7],
            "beta_1y": [1.1, 0.9, 1.2],
        })

    def test_enriches_missing_columns(self, target_df, source_df, catalog):
        result = auto_enrich_for_model(target_df, source_df, "price_target_achievement", catalog)
        assert "expected_upside_pt" in result.columns
        assert "analyst_conviction" in result.columns
        assert "beta_1y" in result.columns
        assert len(result) == 3

    def test_does_not_duplicate_existing_columns(self, source_df, catalog):
        target = source_df.copy()
        result = auto_enrich_for_model(target, source_df, "price_target_achievement", catalog)
        assert list(result.columns).count("expected_upside_pt") == 1

    def test_returns_target_when_no_isin_col(self, catalog):
        target = pd.DataFrame({"ticker": ["AAPL"], "val": [1]})
        source = pd.DataFrame({"ticker": ["AAPL"], "val": [3]})
        result = auto_enrich_for_model(target, source, "price_target_achievement", catalog)
        assert result is target

    def test_returns_target_when_source_is_none(self, target_df, catalog):
        result = auto_enrich_for_model(target_df, None, "price_target_achievement", catalog)
        assert result is target_df

    def test_returns_target_when_source_is_empty(self, target_df, catalog):
        empty = pd.DataFrame()
        result = auto_enrich_for_model(target_df, empty, "price_target_achievement", catalog)
        assert result is target_df

    def test_returns_target_when_no_isin_col_both(self, catalog):
        target = pd.DataFrame({"val": [1, 2]})
        source = pd.DataFrame({"val": [3, 4]})
        result = auto_enrich_for_model(target, source, "price_target_achievement", catalog)
        assert result is target

    def test_returns_target_when_no_missing_cols(self, catalog):
        df = pd.DataFrame({
            "isin": ["US0378331005"],
            "ticker": ["AAPL"],
            "expected_upside_pt": [10.0],
        })
        source = pd.DataFrame({
            "isin": ["US0378331005"],
            "ticker": ["AAPL"],
            "expected_upside_pt": [10.0],
        })
        result = auto_enrich_for_model(df, source, "price_target_achievement", catalog)
        assert list(result.columns) == list(df.columns)

    def test_left_join_preserves_all_target_rows(self, catalog):
        target = pd.DataFrame({
            "isin": ["US0378331005", "US5949181045", "US88160R1014"],
            "ticker": ["AAPL", "MSFT", "TSLA"],
            "last_price": [150.0, 300.0, 200.0],
        })
        source = pd.DataFrame({
            "isin": ["US0378331005", "US5949181045"],
            "ticker": ["AAPL", "MSFT"],
            "accounting_quality_score": [80.0, 90.0],
        })
        result = auto_enrich_for_model(target, source, "earnings_beat", catalog)
        assert len(result) == 3
        assert result.loc[result["ticker"] == "TSLA", "accounting_quality_score"].isna().all()

    def test_works_with_db_loaded_catalog(self):
        catalog = FeatureViewCatalog()
        catalog.category_columns = {
            "Analyst Sentiment": ["custom_col_a"],
            "Price Target Dynamics": ["custom_col_b"],
        }
        catalog._loaded = True

        target = pd.DataFrame({"isin": ["XX000000001"], "ticker": ["X"], "price": [100]})
        source = pd.DataFrame({"isin": ["XX000000001"], "ticker": ["X"], "custom_col_a": [1], "custom_col_b": [2]})

        result = auto_enrich_for_model(target, source, "price_target_achievement", catalog)
        assert "custom_col_a" in result.columns
        assert "custom_col_b" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class TestSingleton:
    def setup_method(self):
        reset_feature_catalog()

    def teardown_method(self):
        reset_feature_catalog()

    def test_get_returns_same_instance(self, monkeypatch):
        monkeypatch.delenv("DB_URL", raising=False)
        cat1 = get_feature_catalog()
        cat2 = get_feature_catalog()
        assert cat1 is cat2

    def test_reset_clears_instance(self, monkeypatch):
        monkeypatch.delenv("DB_URL", raising=False)
        cat1 = get_feature_catalog()
        reset_feature_catalog()
        cat2 = get_feature_catalog()
        assert cat1 is not cat2

    def test_singleton_uses_fallback_without_db(self, monkeypatch):
        monkeypatch.delenv("DB_URL", raising=False)
        cat = get_feature_catalog()
        assert cat._loaded is False
        cols = cat.columns_for_model("credit_risk")
        assert len(cols) > 0
