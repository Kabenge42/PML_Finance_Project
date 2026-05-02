"""Tests for the multi-level hierarchical shrinkage infrastructure shared by
all PyMC models in ``probabilistic_ml_model.pymc_models``.

These tests cover the pure-NumPy helpers (``build_hierarchy_indices``,
``HIERARCHICAL_CATEGORY_COLS``, ``PARENT_MAP``, ``_resolve_prior_sigma``) that
do not depend on PyMC, plus the backward-compatibility shim for the legacy
``sectors=`` parameter on the public ``fit(...)`` signature of each model.

PyMC-dependent fits are smoke-gated: skipped when ``pymc`` is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Constants & helpers (no PyMC required)
# ---------------------------------------------------------------------------
class TestHierarchyConstants:
    def test_canonical_columns_match_statistical_models(self):
        from probabilistic_ml_model.pymc_models._hierarchy import (
            HIERARCHICAL_CATEGORY_COLS,
            PARENT_MAP,
        )

        # Constants from the canonical statistical_models definition
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            _HIERARCHICAL_CATEGORY_COLS,
        )

        assert tuple(HIERARCHICAL_CATEGORY_COLS) == tuple(_HIERARCHICAL_CATEGORY_COLS)

        # Parent-map invariants
        assert PARENT_MAP["region"] is None
        assert PARENT_MAP["country"] == "region"
        assert PARENT_MAP["exchange"] == "country"
        assert PARENT_MAP["sector"] == "exchange"
        assert PARENT_MAP["industry"] == "sector"
        assert PARENT_MAP["style_class"] is None
        assert PARENT_MAP["size_class"] == "style_class"
        assert PARENT_MAP["unit"] is None
        assert PARENT_MAP["trading_country"] is None

        # Every column must be keyed in PARENT_MAP
        assert set(PARENT_MAP) == set(HIERARCHICAL_CATEGORY_COLS)


class TestBuildHierarchyIndices:
    @pytest.fixture
    def categories_df(self):
        isins = np.array(["A", "B", "C", "D"])
        return pd.DataFrame(
            {
                "region": ["NA", "NA", "EU", "EU"],
                "country": ["US", "US", "DE", "FR"],
                "exchange": ["NYSE", "NASDAQ", "XETRA", "EURONEXT"],
                "sector": ["Tech", "Tech", "Tech", "Energy"],
                "industry": ["SW", "HW", "SW", "OilGas"],
            },
            index=isins,
        )

    def test_returns_one_entry_per_level(self, categories_df):
        from probabilistic_ml_model.pymc_models._hierarchy import build_hierarchy_indices

        isins = np.array(categories_df.index)
        out = build_hierarchy_indices(
            categories_df, isins, levels=["region", "country", "exchange", "sector", "industry"]
        )
        assert set(out) == {"region", "country", "exchange", "sector", "industry"}

    def test_idx_shape_and_dtype(self, categories_df):
        from probabilistic_ml_model.pymc_models._hierarchy import build_hierarchy_indices

        isins = np.array(categories_df.index)
        out = build_hierarchy_indices(categories_df, isins, levels=["region", "sector"])
        assert out["region"]["idx"].shape == (4,)
        assert out["region"]["idx"].dtype == np.int32
        assert set(out["region"]["labels"]) == {"NA", "EU"}

    def test_parent_of_child_consistency(self, categories_df):
        from probabilistic_ml_model.pymc_models._hierarchy import build_hierarchy_indices

        isins = np.array(categories_df.index)
        out = build_hierarchy_indices(
            categories_df, isins, levels=["region", "country", "exchange", "sector", "industry"]
        )

        # Top-level (region) has no parent
        assert out["region"]["parent_label"] is None
        assert out["region"]["parent_of"] is None

        # Country -> Region. Validate via dataframe groupby.
        country_to_region = (
            categories_df.groupby("country")["region"].first().reindex(out["country"]["labels"])
        )
        expected_parent_idx = np.array(
            [list(out["region"]["labels"]).index(r) for r in country_to_region],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(out["country"]["parent_of"], expected_parent_idx)

    def test_default_levels_uses_columns_present(self, categories_df):
        from probabilistic_ml_model.pymc_models._hierarchy import build_hierarchy_indices

        isins = np.array(categories_df.index)
        out = build_hierarchy_indices(categories_df, isins)
        # Should include each column present in categories_df, in canonical order
        from probabilistic_ml_model.pymc_models._hierarchy import HIERARCHICAL_CATEGORY_COLS

        expected = [c for c in HIERARCHICAL_CATEGORY_COLS if c in categories_df.columns]
        assert list(out.keys()) == expected


class TestResolvePriorSigma:
    def test_growth_calculation_type_tighter_than_default(self):
        from probabilistic_ml_model.pymc_models._hierarchy import _resolve_prior_sigma

        sigma_growth = _resolve_prior_sigma(data_type="pct", calculation_type="growth")
        sigma_default = _resolve_prior_sigma(data_type="ratio", calculation_type=None)
        assert sigma_growth < sigma_default

    def test_pct_smaller_than_ratio(self):
        from probabilistic_ml_model.pymc_models._hierarchy import _resolve_prior_sigma

        assert _resolve_prior_sigma(data_type="pct") < _resolve_prior_sigma(data_type="ratio")

    def test_returns_positive_float(self):
        from probabilistic_ml_model.pymc_models._hierarchy import _resolve_prior_sigma

        s = _resolve_prior_sigma()
        assert isinstance(s, float)
        assert s > 0


# ---------------------------------------------------------------------------
# Backward compatibility: legacy ``sectors=`` should map to single-level
# ``categories_df`` + ``hierarchy_levels=["sector"]``.  These tests don't run
# the actual sampler — they patch ``pm.sample`` to a no-op and inspect the
# resulting coords/Data containers.
# ---------------------------------------------------------------------------
pm = pytest.importorskip("pymc")


def _make_isins(n: int = 6) -> np.ndarray:
    return np.array([f"ISIN{i:03d}" for i in range(n)], dtype=object)


def _no_sample(monkeypatch):
    """Replace ``pm.sample`` with a stub that returns a minimal InferenceData
    so we can introspect the model graph without expensive sampling."""
    import arviz as az

    def fake_sample(*_args, **_kwargs):
        return az.from_dict({"x": np.zeros((1, 1))})

    monkeypatch.setattr(pm, "sample", fake_sample)


class TestEarningsBeatBackCompat:
    def test_legacy_sectors_param_still_accepted(self, monkeypatch):
        _no_sample(monkeypatch)
        from probabilistic_ml_model.pymc_models.EarningsBeatModel import EarningsBeatBayesian

        isins = _make_isins(4)
        sectors = np.array(["Tech", "Tech", "Energy", "Energy"])
        n_total = np.array([10, 12, 8, 9], dtype=np.int32)
        n_beats = np.array([6, 8, 3, 4], dtype=np.int32)

        m = EarningsBeatBayesian()
        m.fit(n_beats, n_total, isins, sectors=sectors, samples=2, tune=2, chains=1)
        assert m.model_ is not None
        # Should have a "sector" coord (single level fallback)
        assert "sector" in m.model_.coords

    def test_categories_df_multi_level_creates_coords(self, monkeypatch):
        _no_sample(monkeypatch)
        from probabilistic_ml_model.pymc_models.EarningsBeatModel import EarningsBeatBayesian

        isins = _make_isins(4)
        cats = pd.DataFrame(
            {
                "exchange": ["NYSE", "NYSE", "XETRA", "XETRA"],
                "sector": ["Tech", "Tech", "Energy", "Energy"],
                "industry": ["SW", "HW", "OilGas", "OilGas"],
            },
            index=isins,
        )
        n_total = np.array([10, 12, 8, 9], dtype=np.int32)
        n_beats = np.array([6, 8, 3, 4], dtype=np.int32)

        m = EarningsBeatBayesian()
        m.fit(
            n_beats,
            n_total,
            isins,
            categories_df=cats,
            hierarchy_levels=["exchange", "sector", "industry"],
            samples=2,
            tune=2,
            chains=1,
        )
        assert m.model_ is not None
        for level in ("exchange", "sector", "industry"):
            assert level in m.model_.coords


class TestCreditRiskBackCompat:
    def test_legacy_sectors_param_still_accepted(self, monkeypatch):
        _no_sample(monkeypatch)
        from probabilistic_ml_model.pymc_models.CreditRiskModel import CreditRiskBayesian

        isins = _make_isins(4)
        sectors = np.array(["Tech", "Tech", "Energy", "Energy"])
        z_scores = np.array([2.0, 1.5, 3.0, 0.9])
        de = np.array([0.5, 0.6, 0.4, 1.2])

        m = CreditRiskBayesian()
        m.fit(z_scores, de, isins, sectors=sectors, samples=2, tune=2, chains=1)
        assert "sector" in m.model_.coords
