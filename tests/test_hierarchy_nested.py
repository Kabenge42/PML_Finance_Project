"""Nested group effects, the derived group labels, and the leaf-only mean.

What these protect
------------------
Three failures in this area are SILENT -- the model still builds and still
samples, it is simply a different model than the one the arm's name claims:

1. A level configured to nest that quietly builds crossed. The arm then reduces
   to a coarser one and the comparison reports "no difference", which is
   indistinguishable from a real null result.
2. An intermediate level's effect added to the mean alongside its children's.
   A nested child already carries its ancestry, so the parent is then counted
   once per level of depth below it.
3. A parent map whose index space disagrees with the coord factorisation, which
   attributes every child to the wrong parent.

The fourth thing here is an identity check: ``group_parents=None`` must
reproduce the shipped crossed model exactly, because that equality is what makes
a nested arm a one-change contrast rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from probabilistic_ml_model.pymc_models._hierarchy import (
    DERIVED_GROUP_BUILDERS,
    HIERARCHICAL_CATEGORY_COLS,
    OECD_MEMBERS,
    PARENT_MAP,
    UNKNOWN_LABEL,
    attach_derived_group_labels,
    build_hierarchy_indices,
    oecd_bloc,
    order_levels,
    resolve_parent,
    style_box,
)

_REGIONS = (
    "Europe",
    "United States and Canada",
    "Asia / Pacific",
    "Latin America and Caribbean",
    "Africa / Middle East",
)


# --------------------------------------------------------------------------- #
# Derived labels                                                              #
# --------------------------------------------------------------------------- #
class TestOecdBloc:
    def test_membership_splits_within_a_region(self):
        assert oecd_bloc("JP", "Asia / Pacific") == "OECD_ASIA_PACIFIC"
        assert oecd_bloc("CN", "Asia / Pacific") == "NON_OECD_ASIA_PACIFIC"
        assert oecd_bloc("MX", "Latin America and Caribbean") == "OECD_LATAM"
        assert oecd_bloc("BR", "Latin America and Caribbean") == "NON_OECD_LATAM"

    def test_every_bloc_lies_inside_exactly_one_region(self):
        """The nesting is structural, and this is the property that says so.

        If a bloc could span two regions, ``region -> oecd_bloc -> country``
        would not be a tree and :func:`build_hierarchy_indices` would silently
        pick the modal parent for the straddling level.
        """
        seen: dict[str, str] = {}
        codes = sorted(OECD_MEMBERS) + ["CN", "BR", "ZA", "XX"]
        for region in _REGIONS:
            for code in codes:
                bloc = oecd_bloc(code, region)
                assert seen.setdefault(bloc, region) == region

    def test_missing_region_is_unknown_not_non_oecd(self):
        """A name with no region must not be defaulted into the non-OECD bloc."""
        assert oecd_bloc("US", "n/a") == UNKNOWN_LABEL
        assert oecd_bloc("US", None) == UNKNOWN_LABEL
        assert oecd_bloc(None, "Europe") == "NON_OECD_EUROPE"

    def test_na_is_a_category_not_a_missing_marker(self):
        """``NA`` is Namibia, and an abbreviation for North America.

        Treating it as a missing sentinel silently merges a real category into
        the unknown bucket -- which is why the sentinel set is narrow.
        """
        frame = pd.DataFrame(
            {
                "region": ["NA"],
                "country": ["NA"],
                "size_class": ["Large Cap"],
                "style_class": ["Value"],
            }
        )
        out = attach_derived_group_labels(frame)
        assert out["region"].iloc[0] == "NA"
        assert out["oecd_bloc"].iloc[0] == "NON_OECD_NA"


class TestStyleBox:
    def test_nine_cells(self):
        sizes = ["Large Cap", "Mid Cap", "Small Cap"]
        styles = ["Growth", "Core", "Value"]
        cells = {style_box(a, b) for a in sizes for b in styles}
        assert len(cells) == 9
        assert UNKNOWN_LABEL not in cells

    def test_a_half_known_cell_is_not_a_cell(self):
        assert style_box("n/a", "Value") == UNKNOWN_LABEL
        assert style_box("Small Cap", "") == UNKNOWN_LABEL


class TestAttachDerivedGroupLabels:
    def test_vendor_sentinel_becomes_unknown(self):
        """An ``n/a`` reaching ``pd.factorize`` becomes a fitted group level."""
        frame = pd.DataFrame(
            {
                "trading_region": ["Europe", "n/a", "", None],
                "sector": ["Health Care", "n/a", "Energy", "Financials"],
            }
        )
        out = attach_derived_group_labels(frame)
        assert list(out["trading_region"]) == [
            "Europe",
            UNKNOWN_LABEL,
            UNKNOWN_LABEL,
            UNKNOWN_LABEL,
        ]
        assert out["sector"].iloc[1] == UNKNOWN_LABEL

    def test_does_not_mutate_the_input(self):
        frame = pd.DataFrame(
            {"sector": ["n/a"], "region": ["Europe"], "country": ["DE"]}
        )
        before = frame.copy()
        attach_derived_group_labels(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_absent_sources_are_skipped_not_fatal(self):
        out = attach_derived_group_labels(pd.DataFrame({"sector": ["Energy"]}))
        assert "oecd_bloc" not in out.columns
        assert "style_box" not in out.columns

    def test_unknown_derived_level_is_refused(self):
        with pytest.raises(ValueError, match="not a derived group level"):
            attach_derived_group_labels(pd.DataFrame(), levels=["vibes"])


# --------------------------------------------------------------------------- #
# The map itself                                                              #
# --------------------------------------------------------------------------- #
class TestParentMap:
    def test_derived_levels_are_registered_everywhere(self):
        for level in DERIVED_GROUP_BUILDERS:
            assert level in HIERARCHICAL_CATEGORY_COLS
            assert level in PARENT_MAP

    def test_nearest_ancestor_when_the_direct_parent_is_absent(self):
        """Inserting ``oecd_bloc`` must not demote ``country`` for old callers.

        Six other models call ``build_hierarchy_indices(levels=["region",
        "country"])``. Without the walk-up they would silently get a FLAT
        country level.
        """
        assert PARENT_MAP["country"] == "oecd_bloc"
        assert resolve_parent("country", {"region"}) == "region"
        assert resolve_parent("country", {"region", "oecd_bloc"}) == "oecd_bloc"
        assert resolve_parent("country", set()) is None

    def test_order_levels_puts_parents_first(self):
        out = order_levels(["country", "industry", "region", "oecd_bloc", "sector"])
        assert out.index("region") < out.index("oecd_bloc") < out.index("country")
        assert out.index("sector") < out.index("industry")

    def test_order_levels_keeps_unknown_levels(self):
        assert order_levels(["sector", "vibes"]) == ["sector", "vibes"]


@pytest.fixture
def frame() -> pd.DataFrame:
    """A small panel frame with sparse and dense levels, and a missing cell."""
    rng = np.random.default_rng(0)
    n = 60
    region = rng.choice(_REGIONS[:3], size=n)
    country = np.array(
        [
            {
                "Europe": "DE",
                "United States and Canada": "US",
                "Asia / Pacific": "CN",
            }[r]
            for r in region
        ]
    )
    country[0] = "BG"  # a 1-name country: the sparse case that motivates nesting
    country[1] = "JP"
    out = pd.DataFrame(
        {
            "isin": [f"I{i:03d}" for i in range(n)],
            "region": region,
            "country": country,
            "trading_region": region,
            "sector": rng.choice(["Health Care", "Energy", "Financials"], size=n),
            "industry": rng.choice(["Biotech", "Oil", "Banks"], size=n),
            "style_class": rng.choice(["Growth", "Core", "Value"], size=n),
            "size_class": rng.choice(["Large Cap", "Mid Cap", "Small Cap"], size=n),
        }
    )
    out.loc[0, "region"] = "Europe"
    out.loc[1, "region"] = "Asia / Pacific"
    out.loc[2, "sector"] = "n/a"
    return attach_derived_group_labels(out)


class TestBuildHierarchyIndicesNesting:
    def test_parent_of_maps_each_child_to_its_real_parent(self, frame):
        isins = frame["isin"].to_numpy()
        meta = build_hierarchy_indices(
            frame.set_index("isin"),
            isins,
            levels=["country", "oecd_bloc", "region"],
        )
        for level, parent in (("oecd_bloc", "region"), ("country", "oecd_bloc")):
            assert meta[level]["parent_label"] == parent
            child_labels = meta[level]["labels"]
            parent_labels = list(meta[parent]["labels"])
            truth = frame.groupby(level)[parent].first()
            for i, lbl in enumerate(child_labels):
                assert meta[level]["parent_of"][i] == parent_labels.index(truth[lbl]), lbl

    def test_sparse_country_still_gets_a_parent(self, frame):
        """The 1-name country is the whole point of the chain."""
        isins = frame["isin"].to_numpy()
        meta = build_hierarchy_indices(
            frame.set_index("isin"),
            isins,
            levels=["region", "oecd_bloc", "country"],
        )
        bg = list(meta["country"]["labels"]).index("BG")
        parent = meta["oecd_bloc"]["labels"][meta["country"]["parent_of"][bg]]
        assert parent == "NON_OECD_EUROPE"


# --------------------------------------------------------------------------- #
# The model                                                                   #
# --------------------------------------------------------------------------- #
pm = pytest.importorskip("pymc")


def _coords_idx_parents(frame: pd.DataFrame, levels: list[str]):
    """Coord uniques, per-ISIN indices and parent maps, mirroring prepare_panel."""
    levels = order_levels(levels)
    coord_uniques: dict[str, np.ndarray] = {}
    coord_idx: dict[str, np.ndarray] = {}
    for col in levels:
        codes, uniques = pd.factorize(frame[col].astype(str), sort=True)
        coord_uniques[col] = np.asarray(uniques)
        coord_idx[col] = codes.astype("int32")
    meta = build_hierarchy_indices(
        frame.set_index(frame["isin"].astype(str))[levels],
        frame["isin"].astype(str).to_numpy(),
        levels=levels,
    )
    parent_of = {
        k: np.asarray(v["parent_of"], dtype="int32")
        for k, v in meta.items()
        if v["parent_of"] is not None
    }
    return coord_uniques, coord_idx, parent_of


class TestBuildGroupEffectTerms:
    def test_crossed_is_unchanged_when_group_parents_is_none(self, frame):
        """The shipped model must be reproduced variable-for-variable."""
        from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
            KalmanModelConfig,
            build_group_effect_terms,
        )

        levels = ["trading_region", "sector", "style_class", "size_class"]
        uniques, idx, _ = _coords_idx_parents(frame, levels)
        cfg = replace(KalmanModelConfig(), group_effects=tuple(levels))
        assert cfg.group_parents is None

        with pm.Model(coords=uniques) as model:
            _, leaves = build_group_effect_terms(cfg, idx, {})

        assert sorted(leaves) == sorted(levels)  # every level is a leaf
        names = {v.name for v in model.free_RVs}
        assert names == {f"{lv}_effect" for lv in levels}
        assert not any(n.endswith("_dev") for n in names)

    def test_nested_samples_a_deviation_and_derives_the_effect(self, frame):
        from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
            KalmanModelConfig,
            build_group_effect_terms,
        )

        levels = ["region", "oecd_bloc", "country", "sector"]
        uniques, idx, parent_of = _coords_idx_parents(frame, levels)
        cfg = replace(
            KalmanModelConfig(), group_effects=tuple(levels), group_parents={}
        )

        with pm.Model(coords=uniques) as model:
            _, leaves = build_group_effect_terms(cfg, idx, parent_of)

        # Only the chain's leaf and the crossed root enter the mean.
        assert set(leaves) == {"country", "sector"}
        free = {v.name for v in model.free_RVs}
        assert "oecd_bloc_dev" in free and "country_dev" in free
        assert "region_effect" in free and "sector_effect" in free
        # The intermediate effects still EXIST, as Deterministics, so they stay
        # reportable -- they are simply not added to the mean a second time.
        det = {v.name for v in model.deterministics}
        assert {"oecd_bloc_effect", "country_effect"} <= det

    def test_nesting_without_a_parent_map_refuses(self, frame):
        """Silently building crossed would make the arm BE a coarser one."""
        from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
            KalmanModelConfig,
            build_group_effect_terms,
        )

        levels = ["region", "oecd_bloc"]
        uniques, idx, _ = _coords_idx_parents(frame, levels)
        cfg = replace(
            KalmanModelConfig(), group_effects=tuple(levels), group_parents={}
        )
        with pm.Model(coords=uniques):
            with pytest.raises(ValueError, match="no coord_parent_of"):
                build_group_effect_terms(cfg, idx, {})

    def test_the_mean_counts_each_chain_exactly_once(self, frame):
        """A parent must not contribute once per level of depth below it.

        Checked numerically: the summed leaf contribution has to equal the
        leaf effect alone, because the leaf already carries its ancestry.
        """
        from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
            KalmanModelConfig,
            build_group_effect_terms,
        )

        levels = ["region", "oecd_bloc", "country"]
        uniques, idx, parent_of = _coords_idx_parents(frame, levels)
        cfg = replace(
            KalmanModelConfig(), group_effects=tuple(levels), group_parents={}
        )

        with pm.Model(coords=uniques) as model:
            effects, leaves = build_group_effect_terms(cfg, idx, parent_of)
            eta = sum(effects[lv][idx[lv]] for lv in leaves)
            eta_v, region_v, bloc_v, country_v, bloc_dev, country_dev = pm.draw(
                [
                    eta,
                    effects["region"],
                    effects["oecd_bloc"],
                    effects["country"],
                    model["oecd_bloc_dev"],
                    model["country_dev"],
                ],
                random_seed=7,
            )

        # The mean is the LEAF alone: region and oecd_bloc reach it only through
        # country. Were an intermediate added as well, eta would exceed this.
        np.testing.assert_allclose(eta_v, country_v[idx["country"]], rtol=1e-10)

        # ...and the leaf really is the accumulated chain, not a free level.
        np.testing.assert_allclose(
            bloc_v, region_v[parent_of["oecd_bloc"]] + bloc_dev, rtol=1e-10
        )
        np.testing.assert_allclose(
            country_v, bloc_v[parent_of["country"]] + country_dev, rtol=1e-10
        )
        assert not np.allclose(country_v, country_dev)

    def test_a_cycle_is_refused(self, frame):
        from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
            KalmanModelConfig,
            build_group_effect_terms,
        )

        levels = ["sector", "industry"]
        uniques, idx, parent_of = _coords_idx_parents(frame, levels)
        cfg = replace(
            KalmanModelConfig(),
            group_effects=tuple(levels),
            group_parents={"sector": "industry", "industry": "sector"},
        )
        with pm.Model(coords=uniques):
            with pytest.raises(ValueError, match="cycle"):
                build_group_effect_terms(cfg, idx, parent_of)


class TestComparisonArms:
    def test_new_arms_are_registered_and_screenable(self):
        import pymc_kalman_filter_pt_v2 as v2
        from probabilistic_ml_model.pymc_models._max_and_smooth import (
            assert_arm_is_screenable,
        )

        base = v2.KalmanModelConfig()
        for arm in (
            "hierarchy_nested",
            "hierarchy_nested_full",
            "hierarchy_geo",
            "hierarchy_styled",
        ):
            assert arm in v2.COMPARISON_ARMS
            assert_arm_is_screenable(base, v2.COMPARISON_ARMS[arm](base), arm)

    def test_baseline_and_hierarchy_fine_stay_crossed(self):
        import pymc_kalman_filter_pt_v2 as v2

        base = v2.KalmanModelConfig()
        assert v2.COMPARISON_ARMS["baseline"](base).group_parents is None
        assert v2.COMPARISON_ARMS["hierarchy_fine"](base).group_parents is None

    def test_nested_arms_resolve_a_forest(self):
        import pymc_kalman_filter_pt_v2 as v2
        from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
            group_effect_leaves,
            resolve_group_parents,
        )

        base = v2.KalmanModelConfig()
        for arm in ("hierarchy_nested", "hierarchy_nested_full", "hierarchy_styled"):
            cfg = v2.COMPARISON_ARMS[arm](base)
            levels = order_levels(list(cfg.group_effects))
            parents = resolve_group_parents(cfg, levels)
            leaves = group_effect_leaves(levels, parents)
            roots = [lv for lv in levels if lv not in parents]
            assert len(leaves) == len(roots), arm

    def test_coord_parent_of_survives_subsampling(self):
        """It is indexed by LEVEL, not by ISIN, so it must not be sliced."""
        import pymc_kalman_filter_pt_v2 as v2

        assert "coord_parent_of" in v2._PANEL_NON_ISIN_FIELDS
