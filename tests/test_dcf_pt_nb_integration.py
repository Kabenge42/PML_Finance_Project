"""
Tests for the PyMC DCF Price-Target notebook integration.

Covers (per DCF_PT_nb_integration.md):
- `DCF_PriceTargetModel.py` re-alignment to ``pml.mv_pymc_dcf_pt`` and the
  ``dcf_pt`` feature catalogue:
    * ``_DCF_COORD_COLS`` / ``_DCF_GROUP_EFFECTS`` class metadata,
    * catalogue-driven ``_resolve_dcf_feature_aliases``,
    * ``fit(...)`` registering every MV coord + ``<coord>_idx`` container and
      hierarchical partial-pooling group effects.
- ``pymc_dcf.ipynb`` §1/§2 header fixes and the self-contained §4/§5/§6 DCF
  model implementation.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "pymc_dcf.ipynb",
)


# ═══════════════════════════════════════════════════════════════════════════════
# DCF_PriceTargetModel.py — catalogue alignment & coord registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestDCFCoordMetadata:
    """Class-level coord/group-effect metadata aligned with the DCF MV."""

    def test_coord_cols(self):
        from probabilistic_ml_model.pymc_models.DCF_PriceTargetModel import DCFPriceTarget

        assert DCFPriceTarget._DCF_COORD_COLS == (
            "region",
            "country",
            "trading_country",
            "exchange",
            "unit",
            "style_class",
            "size_class",
            "sector",
            "industry",
        )
        # feat_next_earnings_status is NOT emitted by the DCF MV.
        assert "feat_next_earnings_status" not in DCFPriceTarget._DCF_COORD_COLS

    def test_group_effects(self):
        from probabilistic_ml_model.pymc_models.DCF_PriceTargetModel import DCFPriceTarget

        assert DCFPriceTarget._DCF_GROUP_EFFECTS == ("sector", "size_class", "style_class")


class TestResolveDcfFeatureAliases:
    """``_resolve_dcf_feature_aliases`` reads the PyMC feature catalogue."""

    def _clear_cache(self):
        from probabilistic_ml_model.pymc_models.DCF_PriceTargetModel import DCFPriceTarget

        DCFPriceTarget._resolve_dcf_feature_aliases.cache_clear()

    def test_queries_pymc_catalogue(self):
        from probabilistic_ml_model.pymc_models.DCF_PriceTargetModel import DCFPriceTarget

        self._clear_cache()
        captured = {}

        def fake_read_sql(sql, _engine):
            captured["sql"] = str(sql)
            return pd.DataFrame({"feature_alias": ["feat_fcf_ltm", "feat_fcf_fy1e"]})

        with patch("pandas.read_sql", side_effect=fake_read_sql):
            aliases = DCFPriceTarget._resolve_dcf_feature_aliases(
                "postgresql+psycopg2://u:p@localhost:5432/db"
            )

        self._clear_cache()
        assert aliases == ("feat_fcf_ltm", "feat_fcf_fy1e")
        sql = captured["sql"].lower()
        assert "vw_pymc_feature_catalogue" in sql
        assert "dcf_pt" in sql
        assert "mutable_predictor" in sql

    def test_defensive_on_error(self):
        from probabilistic_ml_model.pymc_models.DCF_PriceTargetModel import DCFPriceTarget

        self._clear_cache()
        with patch("pandas.read_sql", side_effect=RuntimeError("no db")):
            aliases = DCFPriceTarget._resolve_dcf_feature_aliases(
                "postgresql+psycopg2://u:p@localhost:5432/db"
            )
        self._clear_cache()
        assert aliases == tuple()


class TestDCFFitCoordRegistration:
    """``fit`` registers MV coords, ``<coord>_idx`` containers and group effects."""

    @pytest.fixture()
    def fitted_model(self):
        pytest.importorskip("pymc")
        from probabilistic_ml_model.pymc_models.DCF_PriceTargetModel import DCFPriceTarget

        isins = np.array(["I1", "I2", "I3", "I4"])
        cats = pd.DataFrame(
            {
                "isin": isins,
                "region": ["NA", "NA", "EU", "EU"],
                "country": ["US", "US", "DE", "DE"],
                "trading_country": ["US", "US", "DE", "DE"],
                "exchange": ["NMS", "NMS", "XETRA", "XETRA"],
                "unit": ["USD", "USD", "EUR", "EUR"],
                "style_class": ["Growth", "Value", "Growth", "Value"],
                "size_class": ["Large", "Mid", "Large", "Mid"],
                "sector": ["Tech", "Tech", "Fin", "Fin"],
                "industry": ["SW", "SW", "Bank", "Bank"],
            }
        )
        dcf = DCFPriceTarget()
        # Avoid any DB access for the auxiliary dcf_feature coord.
        with patch.object(
                DCFPriceTarget, "_resolve_dcf_feature_aliases", return_value=tuple()
        ):
            _, model = dcf.fit(
                historical_fcf=np.array([100.0, 110.0, 120.0]),
                price_target=np.array([150.0, 160.0, 170.0, 180.0]),
                isins=isins,
                categories_df=cats,
                samples=5,
                tune=5,
                chains=1,
                cores=1,
                random_seed=42,
                progressbar=False,
            )
        return model

    def test_all_coords_registered(self, fitted_model):
        from probabilistic_ml_model.pymc_models.DCF_PriceTargetModel import DCFPriceTarget

        names = set(fitted_model.named_vars)
        for col in DCFPriceTarget._DCF_COORD_COLS:
            assert col in fitted_model.coords, f"missing coord dim {col}"
            assert f"{col}_idx" in names, f"missing {col}_idx container"

    def test_group_effects_present(self, fitted_model):
        from probabilistic_ml_model.pymc_models.DCF_PriceTargetModel import DCFPriceTarget

        names = set(fitted_model.named_vars)
        for col in DCFPriceTarget._DCF_GROUP_EFFECTS:
            assert f"sigma_{col}" in names
            assert f"z_{col}" in names
            assert f"{col}_effect" in names

    def test_intrinsic_value_has_isin_dim(self, fitted_model):
        assert "intrinsic_value" in fitted_model.named_vars
        dims = fitted_model.named_vars_to_dims.get("intrinsic_value")
        assert dims == ("isin",)


# ═══════════════════════════════════════════════════════════════════════════════
# pymc_dcf.ipynb — content & structure
# ═══════════════════════════════════════════════════════════════════════════════


def _load_nb():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _sources(nb):
    return ["".join(c["source"]) for c in nb["cells"]]


class TestNotebookContent:
    """Header fixes and §4/§5/§6 implementation are present in the notebook."""

    def test_notebook_is_valid_json(self):
        nb = _load_nb()
        assert "cells" in nb and len(nb["cells"]) > 0

    def test_section1_header_dcf(self):
        text = "\n".join(_sources(_load_nb()))
        assert "DCFPriceTarget" in text
        # earnings-beat leftovers removed from the header
        assert "DCFBayesian" not in text
        assert "Beta-Binomial" not in text

    def test_section2_model_target_dcf_pt(self):
        nb = _load_nb()
        src5 = "".join(nb["cells"][5]["source"])
        assert "model_target = 'dcf_pt'" in src5
        assert "earnings_beat" not in src5

    def test_section2_observed_pt_fallback(self):
        text = "\n".join(_sources(_load_nb()))
        assert "observed_pt" in text
        # the Beta-Binomial response fallback loop must be gone
        assert "'n_total', 'n_beats'" not in text

    def test_no_next_earnings_status_coord(self):
        # The earnings-beat-only coord must be gone from the prose/markdown and
        # must never appear inside a DCF coord list (only the explanatory code
        # comment that documents its exclusion may mention it).
        nb = _load_nb()
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                assert "feat_next_earnings_status" not in src
            assert "DCF_COORD_COLS = [" not in src or (
                    "feat_next_earnings_status" not in src.split("DCF_COORD_COLS = [", 1)[1]
            )

    def test_section4_model_present(self):
        text = "\n".join(_sources(_load_nb()))
        assert "DCF_COORD_COLS" in text
        assert "CATEGORICAL_COORDS" in text
        assert "GROUP_EFFECTS" in text
        assert "pm.Model(coords=coords)" in text
        assert "dcf_model" in text
        assert "intrinsic_value_ps" in text
        assert "_idx" in text

    def test_section5_6_predictive(self):
        text = "\n".join(_sources(_load_nb()))
        assert "sample_prior_predictive" in text
        assert "sample_posterior_predictive" in text


class TestNotebookSection4Executes:
    """The self-contained §4 cells build a valid PyMC model on synthetic data."""

    def _section4_code(self):
        nb = _load_nb()
        parts = []
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            if src.lstrip().startswith(("# 4.0", "# 4.1", "# 4.2")):
                parts.append(src)
        return "\n".join(parts)

    def test_section4_builds_model(self):
        pm = pytest.importorskip("pymc")
        code = self._section4_code()
        assert code.strip(), "No §4 code cells found"
        # Drop graphviz rendering (no system binary required for the test).
        code = "\n".join(
            ln for ln in code.splitlines()
            if not ln.strip().startswith("pm.model_to_graphviz")
        )

        n = 6
        rng = np.random.default_rng(0)
        dcf_df = pd.DataFrame(
            {
                "isin": [f"I{i}" for i in range(n)],
                "observed_pt": rng.uniform(50, 200, n),
                "shrs_out": rng.uniform(1e6, 1e7, n),
                "feat_fcf_ltm": rng.uniform(1e6, 1e7, n),
                "feat_fcf_fy1e": rng.uniform(1e6, 1e7, n),
                "feat_fcf_fy2e": rng.uniform(1e6, 1e7, n),
                "feat_fcf_fy3e": rng.uniform(1e6, 1e7, n),
                "feat_fcf_fy4e": rng.uniform(1e6, 1e7, n),
                "feat_fcf_fy5e": rng.uniform(1e6, 1e7, n),
                "feat_fcf_growth_1y": rng.uniform(-0.1, 0.2, n),
                "feat_fcf_terminal_growth": rng.uniform(0.0, 0.03, n),
                "region": ["NA", "NA", "EU", "EU", "AS", "AS"],
                "country": ["US", "US", "DE", "DE", "JP", "JP"],
                "trading_country": ["US", "US", "DE", "DE", "JP", "JP"],
                "exchange": ["NMS", "NMS", "XE", "XE", "TSE", "TSE"],
                "unit": ["USD", "USD", "EUR", "EUR", "JPY", "JPY"],
                "style_class": ["G", "V", "G", "V", "G", "V"],
                "size_class": ["L", "M", "L", "M", "L", "M"],
                "sector": ["Tech", "Tech", "Fin", "Fin", "Ind", "Ind"],
                "industry": ["SW", "SW", "Bank", "Bank", "Mfg", "Mfg"],
            }
        )
        ns = {"pd": pd, "np": np, "pm": pm, "dcf_df": dcf_df, "RANDOM_SEED": 42}
        exec(code, ns)  # noqa: S102 - controlled notebook source

        model = ns["dcf_model"]
        names = set(model.named_vars)
        assert "intrinsic_value_ps" in names
        assert "price_obs" in names
        assert "wacc" in names
        for col in ("sector", "size_class", "style_class"):
            assert f"{col}_idx" in names
            assert f"{col}_effect" in names