"""
Tests for the catalog consolidation refactoring.

Verifies that:
- feature_catalog.py is the single source of truth for FEATURE_VIEW_REGISTRY,
  DEFAULT_IDENTIFIER_COLUMNS, and IDENTIFIER_COLUMNS_SET.
- inference_schema.py imports these from feature_catalog (no local definitions).
- data_utils.py imports these from feature_catalog (no local definitions).
- __init__.py re-exports the canonical symbols.
- All modules agree on the same registry values.
"""

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Canonical Registry Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureViewRegistry:
    """FEATURE_VIEW_REGISTRY is the single source of truth."""

    def test_catalog_has_17_views(self):
        from probabilistic_ml_model.data_utils.feature_catalog import FEATURE_VIEW_REGISTRY

        assert len(FEATURE_VIEW_REGISTRY) == 17

    def test_all_views_start_with_vw_features(self):
        from probabilistic_ml_model.data_utils.feature_catalog import FEATURE_VIEW_REGISTRY

        for view_name in FEATURE_VIEW_REGISTRY:
            assert view_name.startswith("vw_features_"), view_name

    def test_all_categories_are_nonempty_strings(self):
        from probabilistic_ml_model.data_utils.feature_catalog import FEATURE_VIEW_REGISTRY

        for view_name, category in FEATURE_VIEW_REGISTRY.items():
            assert isinstance(category, str) and len(category) > 0, (view_name, category)

    def test_inference_schema_uses_same_registry(self):
        from probabilistic_ml_model.data_utils.feature_catalog import (
            FEATURE_VIEW_REGISTRY as catalog_reg,
        )
        from probabilistic_ml_model.data_utils.inference_schema import (
            FEATURE_VIEW_REGISTRY as schema_reg,
        )

        assert catalog_reg is schema_reg

    def test_init_reexports_registry(self):
        from probabilistic_ml_model.data_utils import FEATURE_VIEW_REGISTRY

        assert len(FEATURE_VIEW_REGISTRY) == 17

    def test_vw_features_views_matches_registry_keys(self):
        from probabilistic_ml_model.data_utils.feature_catalog import (
            FEATURE_VIEW_REGISTRY,
            VW_FEATURES_VIEWS,
        )

        assert VW_FEATURES_VIEWS == list(FEATURE_VIEW_REGISTRY.keys())

    def test_data_utils_vw_features_views_matches(self):
        from probabilistic_ml_model.data_utils.data_utils import VW_FEATURES_VIEWS
        from probabilistic_ml_model.data_utils.feature_catalog import (
            VW_FEATURES_VIEWS as catalog_views,
        )

        assert VW_FEATURES_VIEWS is catalog_views


class TestIdentifierColumns:
    """DEFAULT_IDENTIFIER_COLUMNS is the single source of truth."""

    def test_catalog_has_identifier_columns(self):
        from probabilistic_ml_model.data_utils.feature_catalog import DEFAULT_IDENTIFIER_COLUMNS

        assert len(DEFAULT_IDENTIFIER_COLUMNS) > 20
        assert "ticker" in DEFAULT_IDENTIFIER_COLUMNS
        assert "isin" in DEFAULT_IDENTIFIER_COLUMNS
        assert "sector" in DEFAULT_IDENTIFIER_COLUMNS

    def test_frozenset_matches_list(self):
        from probabilistic_ml_model.data_utils.feature_catalog import (
            DEFAULT_IDENTIFIER_COLUMNS,
            IDENTIFIER_COLUMNS_SET,
        )

        assert IDENTIFIER_COLUMNS_SET == frozenset(DEFAULT_IDENTIFIER_COLUMNS)

    def test_data_utils_uses_catalog_defaults(self):
        """data_utils._DEFAULT_IDENTIFIER_COLS should be the catalog's list."""
        from probabilistic_ml_model.data_utils.data_utils import _DEFAULT_IDENTIFIER_COLS
        from probabilistic_ml_model.data_utils.feature_catalog import DEFAULT_IDENTIFIER_COLUMNS

        assert _DEFAULT_IDENTIFIER_COLS is DEFAULT_IDENTIFIER_COLUMNS

    def test_inference_schema_uses_catalog_set(self):
        """inference_schema._IDENTIFIER_COLS should be the catalog's frozenset."""
        from probabilistic_ml_model.data_utils.inference_schema import _IDENTIFIER_COLS
        from probabilistic_ml_model.data_utils.feature_catalog import IDENTIFIER_COLUMNS_SET

        assert _IDENTIFIER_COLS is IDENTIFIER_COLUMNS_SET

    def test_init_reexports_identifier_columns(self):
        from probabilistic_ml_model.data_utils import (
            DEFAULT_IDENTIFIER_COLUMNS,
            IDENTIFIER_COLUMNS_SET,
        )

        assert isinstance(DEFAULT_IDENTIFIER_COLUMNS, list)
        assert isinstance(IDENTIFIER_COLUMNS_SET, frozenset)

    def test_identifier_set_contains_core_columns(self):
        from probabilistic_ml_model.data_utils.feature_catalog import IDENTIFIER_COLUMNS_SET

        core = {"ticker", "isin", "name", "sector", "industry", "country", "exchange", "region"}
        assert core.issubset(IDENTIFIER_COLUMNS_SET)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Module Consistency Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossModuleConsistency:
    """Verify no module defines its own copy of consolidated registries."""

    def test_inference_schema_no_local_registry_definition(self):
        """FEATURE_VIEW_REGISTRY in inference_schema should be imported, not defined locally."""
        import inspect
        import probabilistic_ml_model.data_utils.inference_schema as mod

        source = inspect.getsource(mod)
        # Should NOT contain a dict literal defining the registry
        assert "FEATURE_VIEW_REGISTRY: dict[str, str] = {" not in source

    def test_inference_schema_no_local_identifier_cols(self):
        """_IDENTIFIER_COLS in inference_schema should be imported, not a local frozenset."""
        import inspect
        import probabilistic_ml_model.data_utils.inference_schema as mod

        source = inspect.getsource(mod)
        assert '_IDENTIFIER_COLS: frozenset[str] = frozenset(' not in source

    def test_data_utils_no_local_identifier_list(self):
        """_DEFAULT_IDENTIFIER_COLS in data_utils should be imported, not a local list."""
        import inspect
        import probabilistic_ml_model.data_utils.data_utils as mod

        source = inspect.getsource(mod)
        assert '_DEFAULT_IDENTIFIER_COLS = [' not in source

    def test_data_utils_no_local_vw_features_list(self):
        """VW_FEATURES_VIEWS in data_utils should be imported, not a local list."""
        import inspect
        import probabilistic_ml_model.data_utils.data_utils as mod

        source = inspect.getsource(mod)
        assert 'VW_FEATURES_VIEWS = [' not in source


# ═══════════════════════════════════════════════════════════════════════════════
# Import Smoke Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestImportSmoke:
    """Verify all modules import cleanly after refactoring."""

    def test_import_feature_catalog(self):
        import probabilistic_ml_model.data_utils.feature_catalog

    def test_import_inference_schema(self):
        import probabilistic_ml_model.data_utils.inference_schema

    def test_import_data_utils(self):
        import probabilistic_ml_model.data_utils.data_utils

    def test_import_init(self):
        import probabilistic_ml_model.data_utils

    def test_import_expected_returns_v3(self):
        import expected_returns_v3

    def test_import_expected_returns_v4(self):
        import expected_returns_v4


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog Singleton & v4 Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestCatalogIntegration:
    """Verify catalog singleton works across modules."""

    def setup_method(self):
        from probabilistic_ml_model.data_utils.feature_catalog import reset_feature_catalog

        reset_feature_catalog()

    def teardown_method(self):
        from probabilistic_ml_model.data_utils.feature_catalog import reset_feature_catalog

        reset_feature_catalog()

    def test_get_feature_catalog_returns_instance(self):
        from probabilistic_ml_model.data_utils.feature_catalog import (
            FeatureViewCatalog,
            get_feature_catalog,
        )

        catalog = get_feature_catalog()
        assert isinstance(catalog, FeatureViewCatalog)

    def test_catalog_singleton_is_same_instance(self):
        from probabilistic_ml_model.data_utils.feature_catalog import get_feature_catalog

        c1 = get_feature_catalog()
        c2 = get_feature_catalog()
        assert c1 is c2

    def test_reset_clears_singleton(self):
        from probabilistic_ml_model.data_utils.feature_catalog import (
            get_feature_catalog,
            reset_feature_catalog,
        )

        c1 = get_feature_catalog()
        reset_feature_catalog()
        c2 = get_feature_catalog()
        assert c1 is not c2
