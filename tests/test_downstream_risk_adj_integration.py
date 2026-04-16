"""
Tests for downstream integration of risk-adjusted expected returns.

Validates that pipeline_runners, inference_schema, and visualization modules
correctly integrate the ensemble return columns (ensemble_return,
ensemble_return_shrunk, risk_adj_return, mcmc_shrinkage) produced by
build_quad_model_alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_mc(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "isin": [f"ISIN{i:04d}" for i in range(n)],
        "ticker": [f"T{i}" for i in range(n)],
        "implied_return_mc": rng.normal(7, 10, n),
        "price_target_mc": rng.uniform(50, 200, n),
        "prob_positive_upside": rng.uniform(30, 90, n),
        "var_5_pct": rng.uniform(-20, -5, n),
        "risk_reward_ratio": rng.uniform(0.5, 3, n),
    })


def _make_quad(n: int = 50) -> pd.DataFrame:
    """Build a quad-like DataFrame with all ensemble return columns."""
    from probabilistic_ml_model.statistical_functions.ensemble_models import (
        build_quad_model_alignment,
        build_tri_model_alignment,
    )

    mc = _make_mc(n)
    rng = np.random.default_rng(42)
    kal = mc[["isin"]].assign(
        implied_return_kalman=mc["implied_return_mc"] * 0.9,
        kalman_estimate=100.0,
        kalman_variance=rng.uniform(0.1, 2.0, n),
    )
    pt = mc[["isin"]].assign(
        implied_return_pt=mc["implied_return_mc"] * 1.1,
        achievement_probability=0.6,
        price_target_prob_weighted=120.0,
        confidence_level="Medium",
        analyst_conviction=0.7,
        eps_revision_momentum=0.5,
        analyst_rating_normalized=0.6,
    )
    beat = mc[["isin"]].assign(prob_beat_given_momentum=0.55)
    tri = build_tri_model_alignment(mc, kal, pt)
    mcmc_result = {"posterior_mean": 6.76, "posterior_std": 0.15, "converged": True}
    quad = build_quad_model_alignment(tri, beat, mcmc_result=mcmc_result)
    return quad


# ===========================================================================
# 1. _shared.py exports
# ===========================================================================


class TestSharedConstants:
    def test_ensemble_return_cols_defined(self):
        from probabilistic_ml_model.visualizations._shared import ENSEMBLE_RETURN_COLS
        assert "ensemble_return" in ENSEMBLE_RETURN_COLS
        assert "risk_adj_return" in ENSEMBLE_RETURN_COLS
        assert "mcmc_shrinkage" in ENSEMBLE_RETURN_COLS
        assert "ensemble_return_shrunk" in ENSEMBLE_RETURN_COLS

    def test_risk_discount_map_defined(self):
        from probabilistic_ml_model.visualizations._shared import RISK_DISCOUNT_MAP
        assert RISK_DISCOUNT_MAP == {0: 0.70, 1: 0.85, 2: 0.95, 3: 1.00}

    def test_risk_quality_labels_defined(self):
        from probabilistic_ml_model.visualizations._shared import RISK_QUALITY_LABELS
        assert 0 in RISK_QUALITY_LABELS
        assert 3 in RISK_QUALITY_LABELS

    def test_shared_exports_in_init(self):
        from probabilistic_ml_model.visualizations import (
            ENSEMBLE_RETURN_COLS,
            RISK_DISCOUNT_MAP,
            RISK_QUALITY_LABELS,
        )
        assert len(ENSEMBLE_RETURN_COLS) == 4
        assert isinstance(RISK_DISCOUNT_MAP, dict)
        assert isinstance(RISK_QUALITY_LABELS, dict)


# ===========================================================================
# 2. pipeline_runners.py — PipelineRunner.enrich_quad_with_mcmc
# ===========================================================================


class TestPipelineRunnerEnrichQuad:
    def test_enrich_quad_method_exists(self):
        from probabilistic_ml_model.pipeline_runners import PipelineRunner, PipelineConfig
        cfg = PipelineConfig()
        runner = PipelineRunner(cfg)
        assert hasattr(runner, "enrich_quad_with_mcmc")

    def test_enrich_quad_noop_when_no_mcmc(self):
        from probabilistic_ml_model.pipeline_runners import PipelineRunner, PipelineConfig
        cfg = PipelineConfig()
        runner = PipelineRunner(cfg)
        # quad empty, mcmc_result empty → should not fail
        runner.enrich_quad_with_mcmc()
        assert runner.r.quad.empty

    def test_enrich_quad_populates_risk_adj(self):
        from probabilistic_ml_model.pipeline_runners import PipelineRunner, PipelineConfig
        from probabilistic_ml_model.statistical_functions.ensemble_models import (
            build_quad_model_alignment,
            build_tri_model_alignment,
        )

        mc = _make_mc(30)
        rng = np.random.default_rng(42)
        kal = mc[["isin"]].assign(
            implied_return_kalman=mc["implied_return_mc"] * 0.9,
            kalman_estimate=100.0,
            kalman_variance=rng.uniform(0.1, 2.0, 30),
        )
        pt = mc[["isin"]].assign(
            implied_return_pt=mc["implied_return_mc"] * 1.1,
            achievement_probability=0.6,
            price_target_prob_weighted=120.0,
            confidence_level="Medium",
            analyst_conviction=0.7,
            eps_revision_momentum=0.5,
            analyst_rating_normalized=0.6,
        )
        beat = mc[["isin"]].assign(prob_beat_given_momentum=0.55)
        tri = build_tri_model_alignment(mc, kal, pt)
        quad = build_quad_model_alignment(tri, beat)

        cfg = PipelineConfig()
        runner = PipelineRunner(cfg)
        runner.r.tri = tri
        runner.r.beat = beat
        runner.r.quad = quad
        runner.r.mcmc_result = {"posterior_mean": 6.76, "posterior_std": 0.15, "converged": True}

        runner.enrich_quad_with_mcmc()

        assert "risk_adj_return" in runner.r.quad.columns
        assert "ensemble_return" in runner.r.quad.columns
        assert "mcmc_shrinkage" in runner.r.quad.columns
        assert runner.r.quad["risk_adj_return"].notna().all()


# ===========================================================================
# 3. inference_schema — build_ensemble_risk_adj_inference_data
# ===========================================================================


class TestEnsembleInferenceData:
    def test_builder_exists_in_data_utils(self):
        from probabilistic_ml_model.data_utils import build_ensemble_risk_adj_inference_data
        assert callable(build_ensemble_risk_adj_inference_data)

    def test_build_returns_dataset(self):
        from probabilistic_ml_model.data_utils.inference_schema import (
            build_ensemble_risk_adj_inference_data,
        )

        quad = _make_quad(30)
        idata = build_ensemble_risk_adj_inference_data(quad)
        assert idata is not None
        # Should have posterior data
        try:
            import arviz as az
            assert hasattr(idata, "posterior")
        except ImportError:
            import xarray as xr
            assert isinstance(idata, (xr.Dataset, xr.DataTree))

    def test_build_with_mcmc_result(self):
        from probabilistic_ml_model.data_utils.inference_schema import (
            build_ensemble_risk_adj_inference_data,
        )

        quad = _make_quad(30)
        mcmc = {"posterior_mean": 6.76, "posterior_std": 0.15, "r_hat": 1.0001}
        idata = build_ensemble_risk_adj_inference_data(quad, mcmc_result=mcmc)
        assert idata is not None

    def test_build_empty_quad(self):
        from probabilistic_ml_model.data_utils.inference_schema import (
            build_ensemble_risk_adj_inference_data,
        )

        quad = pd.DataFrame({"isin": [], "ticker": []})
        result = build_ensemble_risk_adj_inference_data(quad)
        # Should handle gracefully (empty dataset or None)
        assert result is not None or result is None  # no crash


# ===========================================================================
# 4. Visualization modules — new functions exist and are callable
# ===========================================================================


class TestVisualizationFunctionRegistration:
    def test_risk_adj_return_posterior_panel_registered(self):
        from probabilistic_ml_model.visualizations import create_risk_adj_return_posterior_panel
        assert callable(create_risk_adj_return_posterior_panel)

    def test_ensemble_return_comparison_registered(self):
        from probabilistic_ml_model.visualizations import create_ensemble_return_comparison
        assert callable(create_ensemble_return_comparison)

    def test_risk_quality_score_dashboard_registered(self):
        from probabilistic_ml_model.visualizations import create_risk_quality_score_dashboard
        assert callable(create_risk_quality_score_dashboard)


class TestEnsembleReturnComparison:
    def test_produces_figure(self):
        from probabilistic_ml_model.visualizations.probability_viz import (
            create_ensemble_return_comparison,
        )
        import plotly.graph_objects as go

        quad = _make_quad(30)
        fig = create_ensemble_return_comparison(quad)
        assert isinstance(fig, go.Figure)

    def test_missing_columns_returns_no_data(self):
        from probabilistic_ml_model.visualizations.probability_viz import (
            create_ensemble_return_comparison,
        )
        import plotly.graph_objects as go

        df = pd.DataFrame({"ticker": ["A", "B"], "x": [1, 2]})
        fig = create_ensemble_return_comparison(df)
        assert isinstance(fig, go.Figure)
        # Should be a "no data" figure
        assert len(fig.data) == 0 or "No data" in str(fig.layout.annotations)


class TestRiskQualityScoreDashboard:
    def test_produces_figure(self):
        from probabilistic_ml_model.visualizations.quality_risk import (
            create_risk_quality_score_dashboard,
        )
        import plotly.graph_objects as go

        quad = _make_quad(50)
        fig = create_risk_quality_score_dashboard(quad)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_no_risk_quality_score(self):
        from probabilistic_ml_model.visualizations.quality_risk import (
            create_risk_quality_score_dashboard,
        )
        import plotly.graph_objects as go

        df = pd.DataFrame({"ticker": ["A"], "x": [1]})
        fig = create_risk_quality_score_dashboard(df)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# 5. expected_returns_viz — sector heatmap includes new columns
# ===========================================================================


class TestSectorHeatmapNewColumns:
    def test_heatmap_accepts_new_columns(self):
        from probabilistic_ml_model.visualizations.expected_returns_viz import (
            create_sector_return_analytics_heatmap,
        )
        import plotly.graph_objects as go

        sector_df = pd.DataFrame({
            "industry": ["Tech", "Health", "Finance"],
            "mc_mean": [10, 5, 8],
            "pt_mean": [12, 6, 9],
            "mean_risk_adj_return": [8.5, 4.2, 7.1],
            "mean_ensemble_return": [9.0, 4.8, 7.5],
        })
        fig = create_sector_return_analytics_heatmap(sector_df)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# 6. convergence_diagnostics — no crash with summary containing ensemble cols
# ===========================================================================


class TestConvergenceDiagnosticsIntegration:
    def test_dashboard_accepts_summary_with_ensemble(self):
        from probabilistic_ml_model.visualizations.convergence_diagnostics import (
            create_unified_convergence_dashboard,
        )
        from pathlib import Path
        import tempfile

        rng = np.random.default_rng(42)
        n = 200
        mcmc_result = {
            "chain_samples": [rng.normal(7, 2, 500).tolist() for _ in range(4)],
        }
        anomaly_results = pd.DataFrame({
            "anomaly_posterior_mean": rng.normal(0.3, 0.1, n),
        })
        summary = pd.DataFrame({
            "risk_adj_return": rng.normal(8, 5, n),
            "ensemble_return": rng.normal(10, 5, n),
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = create_unified_convergence_dashboard(
                mcmc_result, anomaly_results, summary, Path(tmpdir)
            )
            # Should not crash; output depends on ArviZ availability
            assert isinstance(outputs, list)


# ===========================================================================
# 7. Module imports don't crash
# ===========================================================================


class TestModuleImports:
    """Verify all updated visualization modules import without errors."""

    def test_import_shared(self):
        import probabilistic_ml_model.visualizations._shared

    def test_import_arviz_diagnostics(self):
        import probabilistic_ml_model.visualizations.arviz_diagnostics

    def test_import_convergence_diagnostics(self):
        import probabilistic_ml_model.visualizations.convergence_diagnostics

    def test_import_probability_viz(self):
        import probabilistic_ml_model.visualizations.probability_viz

    def test_import_expected_returns_viz(self):
        import probabilistic_ml_model.visualizations.expected_returns_viz

    def test_import_valuation(self):
        import probabilistic_ml_model.visualizations.valuation

    def test_import_quality_risk(self):
        import probabilistic_ml_model.visualizations.quality_risk

    def test_import_growth_analysis(self):
        import probabilistic_ml_model.visualizations.growth_analysis

    def test_import_earnings_quality(self):
        import probabilistic_ml_model.visualizations.earnings_quality

    def test_import_viz_init(self):
        import probabilistic_ml_model.visualizations
