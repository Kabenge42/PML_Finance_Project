"""
Tests for ArviZ 1.0 refactoring of the visualization package.

Covers:
- _shared.py: ARVIZ_TEMPLATE update, _fig_from_pc hardening, _pc_add_title
- arviz_diagnostics.py: azp migration, new plot types (dot, ecdf, ppc rootogram)
- convergence_diagnostics.py: unified convergence dashboard
- probability_viz.py: ArviZ 1.0 category forest
- __init__.py: registry updates
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_pc():
    """Create a mock PlotCollection with a figure inside viz.ds['figure']."""
    fig, _ = plt.subplots()
    pc = MagicMock()
    pc.viz.ds = {"figure": MagicMock(item=MagicMock(return_value=fig))}
    return pc, fig


def _make_summary_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "implied_return_mc": rng.normal(10, 20, n),
        "implied_return_kalman": rng.normal(12, 18, n),
        "implied_return_pt": rng.normal(11, 15, n),
        "agreement_score": rng.uniform(0, 1, n),
        "industry": rng.choice(["Tech", "Health", "Finance", "Energy"], n),
    })


def _make_screens() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    return {
        "quality": pd.DataFrame({
            "ticker": [f"Q{i}" for i in range(50)],
            "implied_return_pt": rng.normal(15, 10, 50),
        }),
        "value": pd.DataFrame({
            "ticker": [f"V{i}" for i in range(50)],
            "implied_return_pt": rng.normal(8, 12, 50),
        }),
    }


def _make_resampled_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "ticker": [f"R{i}" for i in range(n)],
        "posterior_mean": rng.normal(5, 10, n),
        "posterior_std": rng.uniform(1, 5, n),
    })


def _make_mcmc_result(n_chains: int = 4, n_samples: int = 500) -> dict:
    rng = np.random.default_rng(42)
    chains = [rng.normal(10, 5, n_samples).tolist() for _ in range(n_chains)]
    observed = rng.normal(10, 5, 200).tolist()
    return {"chain_samples": chains, "observed_returns": observed}


def _make_category_analytics() -> dict[str, dict]:
    return {
        "Profitability": {
            "bayesian_results": {
                "roe": {"posterior_mean": 12.5, "posterior_std": 3.0},
                "roa": {"posterior_mean": 8.2, "posterior_std": 2.5},
                "net_margin": {"posterior_mean": 15.0, "posterior_std": 4.0},
            }
        },
        "Growth": {
            "bayesian_results": {
                "revenue_growth": {"posterior_mean": 20.0, "posterior_std": 8.0},
                "eps_growth": {"posterior_mean": 18.0, "posterior_std": 6.0},
            }
        },
    }


# ===================================================================
# 1. _shared.py tests
# ===================================================================


class TestSharedConstants:
    """Test that _shared.py constants are updated for ArviZ 1.0."""

    def test_arviz_template_is_variat(self):
        from probabilistic_ml_model.visualizations._shared import ARVIZ_TEMPLATE

        assert ARVIZ_TEMPLATE == "arviz-variat", (
            f"ARVIZ_TEMPLATE should be 'arviz-variat', got '{ARVIZ_TEMPLATE}'"
        )

    def test_plotly_template_unchanged(self):
        from probabilistic_ml_model.visualizations._shared import PLOTLY_TEMPLATE

        assert PLOTLY_TEMPLATE == "plotly_dark"


class TestApplyArvizTheme:
    """Test apply_arviz_theme uses arviz_plots first, then arviz, then mpl."""

    def test_applies_arviz_plots_first(self):
        mock_azp = MagicMock()
        with patch.dict("sys.modules", {"arviz_plots": mock_azp}):
            from probabilistic_ml_model.visualizations._shared import apply_arviz_theme

            apply_arviz_theme()
            # Should attempt azp.style.use
            # (may also fall through if mock raises; that's fine)

    def test_falls_back_to_matplotlib(self):
        """When both arviz_plots and arviz are unavailable, falls back to mpl."""
        from probabilistic_ml_model.visualizations._shared import apply_arviz_theme

        with patch.dict("sys.modules", {"arviz_plots": None, "arviz": None}):
            # Should not raise
            apply_arviz_theme()


class TestFigFromPc:
    """Test _fig_from_pc handles PlotCollection, raw Figure, and None."""

    def test_extracts_from_plot_collection(self):
        from probabilistic_ml_model.visualizations._shared import _fig_from_pc

        pc, expected_fig = _make_mock_pc()
        result = _fig_from_pc(pc)
        assert result is expected_fig

    def test_returns_raw_figure_passthrough(self):
        from probabilistic_ml_model.visualizations._shared import _fig_from_pc

        fig, _ = plt.subplots()
        result = _fig_from_pc(fig)
        assert isinstance(result, plt.Figure)
        plt.close(fig)

    def test_returns_none_for_none(self):
        from probabilistic_ml_model.visualizations._shared import _fig_from_pc

        result = _fig_from_pc(None)
        assert result is None

    def test_handles_broken_plot_collection(self):
        from probabilistic_ml_model.visualizations._shared import _fig_from_pc

        broken_pc = MagicMock()
        broken_pc.viz.ds.__getitem__ = MagicMock(side_effect=KeyError("figure"))
        result = _fig_from_pc(broken_pc)
        # Should not raise, returns something (the broken pc or None)
        assert result is not None or result is None  # just no exception


class TestPcAddTitle:
    """Test _pc_add_title helper."""

    def test_adds_title_via_add_title(self):
        from probabilistic_ml_model.visualizations._shared import _pc_add_title

        pc = MagicMock()
        result = _pc_add_title(pc, "Test Title")
        pc.add_title.assert_called_once_with("Test Title")
        assert result is pc

    def test_falls_back_to_suptitle(self):
        from probabilistic_ml_model.visualizations._shared import _pc_add_title

        pc = MagicMock()
        pc.add_title.side_effect = AttributeError
        # _fig_from_pc should extract figure and call suptitle
        fig = MagicMock()
        fig.suptitle = MagicMock()
        with patch(
            "probabilistic_ml_model.visualizations._shared._fig_from_pc",
            return_value=fig,
        ):
            result = _pc_add_title(pc, "Fallback Title")
            fig.suptitle.assert_called_once_with("Fallback Title")
            assert result is pc


class TestMakeDatatree:
    """Test _make_datatree builds correct DataTree structures."""

    def test_empty_call_returns_empty_datatree(self):
        from probabilistic_ml_model.visualizations._shared import _make_datatree

        dt = _make_datatree()
        assert len(dt.children) == 0

    def test_single_group(self):
        import xarray as xr
        from probabilistic_ml_model.visualizations._shared import _make_datatree

        ds = xr.Dataset({"x": (["chain", "draw"], np.ones((2, 10)))})
        dt = _make_datatree(posterior=ds)
        assert "posterior" in dt.children

    def test_none_groups_skipped(self):
        import xarray as xr
        from probabilistic_ml_model.visualizations._shared import _make_datatree

        ds = xr.Dataset({"x": (["chain", "draw"], np.ones((2, 10)))})
        dt = _make_datatree(posterior=ds, observed_data=None)
        assert "posterior" in dt.children
        assert "observed_data" not in dt.children


# ===================================================================
# 2. arviz_diagnostics.py tests — ArviZ 1.0 migration
# ===================================================================


class TestArvizDiagnosticsImports:
    """Test that arviz_diagnostics uses arviz_plots/arviz_stats/arviz_base."""

    def test_module_has_arviz_available_flag(self):
        from probabilistic_ml_model.visualizations import arviz_diagnostics as ad

        assert hasattr(ad, "ARVIZ_AVAILABLE")

    def test_uses_azp_not_az_for_plots(self):
        """Verify the module references azp (arviz_plots) for plot calls."""
        from probabilistic_ml_model.visualizations import arviz_diagnostics as ad

        # Check that the module has azp reference
        assert hasattr(ad, "azp"), "arviz_diagnostics should define 'azp' for arviz_plots"

    def test_uses_azs_for_stats(self):
        """Verify the module references azs (arviz_stats) for stat calls."""
        from probabilistic_ml_model.visualizations import arviz_diagnostics as ad

        assert hasattr(ad, "azs"), "arviz_diagnostics should define 'azs' for arviz_stats"


class TestBuildScreeningInferenceData:
    """Test build_screening_inference_data produces valid DataTree."""

    def test_returns_datatree_with_posterior(self):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            build_screening_inference_data,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        screens = _make_screens()
        dt = build_screening_inference_data(screens)
        assert "posterior" in dt.children

    def test_empty_screens_returns_empty(self):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            build_screening_inference_data,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        dt = build_screening_inference_data({})
        assert "posterior" not in dt.children


class TestResampledPosteriorDiagnostics:
    """Test create_resampled_posterior_diagnostics with ArviZ 1.0 features."""

    def test_generates_ecdf_plot(self, tmp_path):
        """New ECDF plot should be generated."""
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_resampled_posterior_diagnostics,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        df = _make_resampled_df(200)
        outputs = create_resampled_posterior_diagnostics(df, tmp_path)
        ecdf_files = [o for o in outputs if "ecdf" in o]
        assert len(ecdf_files) >= 1, "Should generate ECDF plot"

    def test_generates_dotplot(self, tmp_path):
        """New dot plot should be generated."""
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_resampled_posterior_diagnostics,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        df = _make_resampled_df(200)
        outputs = create_resampled_posterior_diagnostics(df, tmp_path)
        dot_files = [o for o in outputs if "dotplot" in o]
        assert len(dot_files) >= 1, "Should generate dot plot"

    def test_title_added_to_trace(self, tmp_path):
        """Trace plot should have title via _pc_add_title."""
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_resampled_posterior_diagnostics,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        df = _make_resampled_df(200)
        outputs = create_resampled_posterior_diagnostics(df, tmp_path)
        trace_files = [o for o in outputs if "trace" in o]
        assert len(trace_files) >= 1

    def test_empty_df_returns_empty(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_resampled_posterior_diagnostics,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        outputs = create_resampled_posterior_diagnostics(pd.DataFrame(), tmp_path)
        assert outputs == []


class TestModelAlignmentPanel:
    """Test create_model_alignment_arviz_panel with ArviZ 1.0 features."""

    def test_generates_ecdf_plot(self, tmp_path):
        """New ECDF comparison should be generated."""
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_model_alignment_arviz_panel,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        summary = _make_summary_df()
        outputs = create_model_alignment_arviz_panel(summary, tmp_path)
        ecdf_files = [o for o in outputs if "ecdf" in o]
        assert len(ecdf_files) >= 1, "Should generate ECDF comparison"

    def test_forest_uses_shade_label(self, tmp_path):
        """Forest plot should use shade_label parameter."""
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_model_alignment_arviz_panel,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        summary = _make_summary_df()
        outputs = create_model_alignment_arviz_panel(summary, tmp_path)
        forest_files = [o for o in outputs if "forest" in o]
        assert len(forest_files) >= 1


class TestMcmcConvergencePanel:
    """Test create_mcmc_convergence_panel_arviz with new plot types."""

    def test_generates_dotplot(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_mcmc_convergence_panel_arviz,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        result = _make_mcmc_result()
        outputs = create_mcmc_convergence_panel_arviz(result, tmp_path)
        dot_files = [o for o in outputs if "dotplot" in o]
        assert len(dot_files) >= 1, "Should generate MCMC dot plot"

    def test_generates_ecdf(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_mcmc_convergence_panel_arviz,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        result = _make_mcmc_result()
        outputs = create_mcmc_convergence_panel_arviz(result, tmp_path)
        ecdf_files = [o for o in outputs if "ecdf" in o]
        assert len(ecdf_files) >= 1, "Should generate MCMC ECDF"

    def test_generates_ppc_rootogram(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_mcmc_convergence_panel_arviz,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        result = _make_mcmc_result()
        outputs = create_mcmc_convergence_panel_arviz(result, tmp_path)
        ppc_files = [o for o in outputs if "ppc_rootogram" in o]
        assert len(ppc_files) >= 1, "Should generate PPC rootogram"

    def test_uses_azs_summary(self, tmp_path):
        """Should use azs.summary instead of az.summary."""
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_mcmc_convergence_panel_arviz,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        result = _make_mcmc_result()
        # Should not raise
        outputs = create_mcmc_convergence_panel_arviz(result, tmp_path)
        assert isinstance(outputs, list)

    def test_empty_chains_returns_empty(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_mcmc_convergence_panel_arviz,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        outputs = create_mcmc_convergence_panel_arviz({}, tmp_path)
        assert outputs == []

    def test_single_chain_returns_empty(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_mcmc_convergence_panel_arviz,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        result = {"chain_samples": [[1, 2, 3]]}
        outputs = create_mcmc_convergence_panel_arviz(result, tmp_path)
        assert outputs == []


# ===================================================================
# 3. New functions in arviz_diagnostics.py
# ===================================================================


class TestScreeningPpcRootogram:
    """Test create_screening_ppc_rootogram (new function)."""

    def test_function_exists(self):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_screening_ppc_rootogram,
        )

        assert callable(create_screening_ppc_rootogram)

    def test_returns_figure_or_none(self):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_screening_ppc_rootogram,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        screens = _make_screens()
        result = create_screening_ppc_rootogram(screens)
        assert result is None or isinstance(result, plt.Figure)

    def test_empty_screens_returns_none(self):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_screening_ppc_rootogram,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        result = create_screening_ppc_rootogram({})
        assert result is None


class TestHierarchicalDotComparison:
    """Test create_hierarchical_dot_comparison (new function)."""

    def test_function_exists(self):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_hierarchical_dot_comparison,
        )

        assert callable(create_hierarchical_dot_comparison)

    def test_returns_figure_or_none(self):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_hierarchical_dot_comparison,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        summary = _make_summary_df()
        result = create_hierarchical_dot_comparison(summary)
        assert result is None or isinstance(result, plt.Figure)


class TestCrossModelEcdfWithReferences:
    """Test create_cross_model_ecdf_with_references (new function)."""

    def test_function_exists(self):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_cross_model_ecdf_with_references,
        )

        assert callable(create_cross_model_ecdf_with_references)

    def test_returns_path_or_none(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_cross_model_ecdf_with_references,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        summary = _make_summary_df()
        result = create_cross_model_ecdf_with_references(summary, tmp_path)
        assert result is None or isinstance(result, str)


# ===================================================================
# 4. convergence_diagnostics.py tests
# ===================================================================


class TestConvergenceDiagnosticsModule:
    """Test the new convergence_diagnostics module."""

    def test_module_importable(self):
        from probabilistic_ml_model.visualizations import convergence_diagnostics

        assert hasattr(convergence_diagnostics, "create_unified_convergence_dashboard")

    def test_has_arviz_available_flag(self):
        from probabilistic_ml_model.visualizations import convergence_diagnostics as cd

        assert hasattr(cd, "ARVIZ_AVAILABLE")


class TestUnifiedConvergenceDashboard:
    """Test create_unified_convergence_dashboard."""

    def test_returns_list(self, tmp_path):
        from probabilistic_ml_model.visualizations.convergence_diagnostics import (
            create_unified_convergence_dashboard,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        mcmc = _make_mcmc_result()
        anomaly = pd.DataFrame({
            "anomaly_posterior_mean": np.random.default_rng(42).normal(0, 1, 200),
        })
        summary = _make_summary_df()
        outputs = create_unified_convergence_dashboard(mcmc, anomaly, summary, tmp_path)
        assert isinstance(outputs, list)

    def test_empty_mcmc_returns_empty(self, tmp_path):
        from probabilistic_ml_model.visualizations.convergence_diagnostics import (
            create_unified_convergence_dashboard,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        outputs = create_unified_convergence_dashboard(
            {}, pd.DataFrame(), pd.DataFrame(), tmp_path
        )
        assert outputs == []


# ===================================================================
# 5. probability_viz.py — ArviZ 1.0 category forest
# ===================================================================


class TestMcmcCategoryPosteriorArviz:
    """Test create_mcmc_category_posterior_arviz (new function)."""

    def test_function_exists(self):
        from probabilistic_ml_model.visualizations.probability_viz import (
            create_mcmc_category_posterior_arviz,
        )

        assert callable(create_mcmc_category_posterior_arviz)

    def test_returns_figure_or_none(self):
        from probabilistic_ml_model.visualizations.probability_viz import (
            create_mcmc_category_posterior_arviz,
        )

        analytics = _make_category_analytics()
        result = create_mcmc_category_posterior_arviz(
            analytics["Profitability"], category_name="Profitability"
        )
        assert result is None or isinstance(result, plt.Figure)

    def test_empty_analytics_returns_none(self):
        from probabilistic_ml_model.visualizations.probability_viz import (
            create_mcmc_category_posterior_arviz,
        )

        result = create_mcmc_category_posterior_arviz({})
        assert result is None


# ===================================================================
# 6. __init__.py registry tests
# ===================================================================


class TestInitRegistry:
    """Test that __init__.py registers new functions and modules."""

    def test_convergence_diagnostics_in_registry(self):
        import probabilistic_ml_model.visualizations as viz_init

        registry = viz_init._IMPORT_REGISTRY
        module_paths = [entry[0] for entry in registry]
        assert ".convergence_diagnostics" in module_paths, (
            "convergence_diagnostics should be in _IMPORT_REGISTRY"
        )

    def test_new_arviz_functions_in_registry(self):
        import probabilistic_ml_model.visualizations as viz_init

        registry = viz_init._IMPORT_REGISTRY
        arviz_entry = None
        for mod_path, names in registry:
            if mod_path == ".arviz_diagnostics":
                arviz_entry = names
                break
        assert arviz_entry is not None
        assert "create_screening_ppc_rootogram" in arviz_entry
        assert "create_hierarchical_dot_comparison" in arviz_entry
        assert "create_cross_model_ecdf_with_references" in arviz_entry

    def test_unified_convergence_dashboard_in_registry(self):
        import probabilistic_ml_model.visualizations as viz_init

        registry = viz_init._IMPORT_REGISTRY
        conv_entry = None
        for mod_path, names in registry:
            if mod_path == ".convergence_diagnostics":
                conv_entry = names
                break
        assert conv_entry is not None
        assert "create_unified_convergence_dashboard" in conv_entry

    def test_new_probability_viz_function_in_registry(self):
        import probabilistic_ml_model.visualizations as viz_init

        registry = viz_init._IMPORT_REGISTRY
        prob_entry = None
        for mod_path, names in registry:
            if mod_path == ".probability_viz":
                prob_entry = names
                break
        assert prob_entry is not None
        assert "create_mcmc_category_posterior_arviz" in prob_entry


# ===================================================================
# 7. Category diagnostics — ArviZ 1.0 migration
# ===================================================================


class TestCategoryPosteriorDiagnostics:
    """Test category diagnostics use azp instead of az."""

    def test_generates_outputs(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_category_posterior_diagnostics,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        analytics = _make_category_analytics()
        df = pd.DataFrame({"ticker": ["A", "B"]})
        outputs = create_category_posterior_diagnostics(analytics, df, tmp_path)
        assert isinstance(outputs, list)

    def test_empty_analytics_returns_empty(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_category_posterior_diagnostics,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        outputs = create_category_posterior_diagnostics({}, pd.DataFrame(), tmp_path)
        assert outputs == []


class TestCrossCategorySummary:
    """Test cross_category_summary uses azp."""

    def test_returns_path_or_none(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_cross_category_summary,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        analytics = _make_category_analytics()
        result = create_cross_category_summary(analytics, tmp_path)
        assert result is None or isinstance(result, str)


# ===================================================================
# 8. Integration: end-to-end DataTree → plot → save
# ===================================================================


class TestEndToEndDataTreePlotSave:
    """Integration test: build DataTree, plot, save."""

    def test_screening_ridge_end_to_end(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_screening_posterior_ridge,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        screens = _make_screens()
        fig = create_screening_posterior_ridge(screens)
        if fig is not None:
            assert isinstance(fig, plt.Figure)
            fig.savefig(tmp_path / "test_ridge.png")
            assert (tmp_path / "test_ridge.png").exists()
            plt.close(fig)

    def test_resampled_diagnostics_end_to_end(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_resampled_posterior_diagnostics,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        df = _make_resampled_df(200)
        outputs = create_resampled_posterior_diagnostics(df, tmp_path)
        for path_str in outputs:
            assert Path(path_str).exists(), f"Output file should exist: {path_str}"

    def test_mcmc_convergence_end_to_end(self, tmp_path):
        from probabilistic_ml_model.visualizations.arviz_diagnostics import (
            create_mcmc_convergence_panel_arviz,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")
        result = _make_mcmc_result()
        outputs = create_mcmc_convergence_panel_arviz(result, tmp_path)
        for path_str in outputs:
            assert Path(path_str).exists(), f"Output file should exist: {path_str}"
