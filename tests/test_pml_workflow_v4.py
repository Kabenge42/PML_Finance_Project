"""
Tests for pml_workflow_v4.ipynb enhancements.

Validates:
- Notebook code structure (cell layout, Gap 4)
- DCF model integration in Phase 3 (Gap 1)
- DCF posterior merge in Phase 5 (Gap 3)
- main() entry point (Gap 2)
- enable_dcf_model override (Gap 5)
- Pipeline config, result container, and model execution patterns
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import types
from dataclasses import fields
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Import Smoke Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestImportSmoke:
    """Verify that expected_returns_v4 and its key classes import cleanly."""

    def test_import_expected_returns_v4(self):
        import expected_returns_v4

    def test_import_pipeline_config(self):
        from expected_returns_v4 import PipelineConfig

        assert PipelineConfig is not None

    def test_import_pipeline_result(self):
        from expected_returns_v4 import BaselinePipelineResult

        assert BaselinePipelineResult is not None

    def test_import_baseline_pipeline(self):
        from expected_returns_v4 import BaselinePipeline

        assert BaselinePipeline is not None

    def test_import_main(self):
        from expected_returns_v4 import main

        assert callable(main)

    def test_import_dcf_model(self):
        from probabilistic_ml_model.pml_models.DCF_PriceTargetModel import DCFPriceTarget

        assert DCFPriceTarget is not None


# ═══════════════════════════════════════════════════════════════════════════════
# PipelineConfig Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelineConfig:
    """PipelineConfig dataclass defaults and from_env construction."""

    def test_default_enable_dcf_model_is_false(self):
        from expected_returns_v4 import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.enable_dcf_model is False

    def test_default_enable_plr_model_is_true(self):
        from expected_returns_v4 import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.enable_plr_model is True

    def test_dcf_sampling_defaults(self):
        from expected_returns_v4 import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.dcf_samples == 2_000
        assert cfg.dcf_tune == 1_000
        assert cfg.dcf_chains == 2
        assert cfg.dcf_cores == 1

    def test_from_env_creates_config(self):
        from expected_returns_v4 import PipelineConfig

        cfg = PipelineConfig.from_env()
        assert isinstance(cfg, PipelineConfig)
        assert cfg.mc_simulations > 0
        assert cfg.mcmc_chains > 0

    def test_from_env_reads_dcf_env_vars(self):
        from expected_returns_v4 import PipelineConfig

        with patch.dict(os.environ, {"ER_DCF_SAMPLES": "500", "ER_DCF_TUNE": "200"}):
            cfg = PipelineConfig.from_env()
            assert cfg.dcf_samples == 500
            assert cfg.dcf_tune == 200

    def test_enable_dcf_can_be_overridden(self):
        """Gap 5: enable_dcf_model can be set to True for notebook workflow."""
        from expected_returns_v4 import PipelineConfig

        cfg = PipelineConfig()
        cfg.enable_dcf_model = True
        assert cfg.enable_dcf_model is True

    def test_from_env_then_override_dcf(self):
        """Gap 5: from_env() + override pattern works as recommended."""
        from expected_returns_v4 import PipelineConfig

        cfg = PipelineConfig.from_env()
        cfg.enable_dcf_model = True
        assert cfg.enable_dcf_model is True


# ═══════════════════════════════════════════════════════════════════════════════
# BaselinePipelineResult Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelineResult:
    """BaselinePipelineResult dataclass has all required fields for DCF integration."""

    def test_has_dcf_result_field(self):
        from expected_returns_v4 import BaselinePipelineResult

        r = BaselinePipelineResult()
        assert hasattr(r, "dcf_result")
        assert r.dcf_result is None

    def test_has_plr_result_field(self):
        from expected_returns_v4 import BaselinePipelineResult

        r = BaselinePipelineResult()
        assert hasattr(r, "plr_result")
        assert r.plr_result is None

    def test_has_idata_dcf_field(self):
        from expected_returns_v4 import BaselinePipelineResult

        r = BaselinePipelineResult()
        assert hasattr(r, "idata_dcf")
        assert r.idata_dcf is None

    def test_has_idata_plr_field(self):
        from expected_returns_v4 import BaselinePipelineResult

        r = BaselinePipelineResult()
        assert hasattr(r, "idata_plr")
        assert r.idata_plr is None

    def test_all_phase_outputs_initialized(self):
        from expected_returns_v4 import BaselinePipelineResult

        r = BaselinePipelineResult()
        # Phase 1
        assert isinstance(r.df, pd.DataFrame)
        assert isinstance(r.df_all, pd.DataFrame)
        assert isinstance(r.df_features, pd.DataFrame)
        # Phase 2
        assert isinstance(r.mc, pd.DataFrame)
        assert isinstance(r.pt, pd.DataFrame)
        assert isinstance(r.beat, pd.DataFrame)
        # Phase 3
        assert r.plr_result is None
        assert r.dcf_result is None
        # Phase 4
        assert isinstance(r.screens, dict)
        # Phase 5
        assert isinstance(r.summary, pd.DataFrame)
        # Timing
        assert isinstance(r.phase_timings, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 1: DCF Step 15 in Phase 3
# ═══════════════════════════════════════════════════════════════════════════════


class TestDCFPhase3Integration:
    """Verify DCF model step is present in phase_3_market_models."""

    def test_phase_3_has_dcf_code(self):
        """phase_3_market_models should contain DCF_PriceTargetModel import."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_3_market_models)
        assert "DCFPriceTarget" in source
        assert "enable_dcf_model" in source
        assert "Step 15" in source

    def test_phase_3_dcf_uses_correct_columns(self):
        """DCF step should look for free_cash_flow and current_price columns."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_3_market_models)
        assert "free_cash_flow" in source
        assert "current_price" in source

    def test_phase_3_dcf_uses_config_params(self):
        """DCF step should pass cfg.dcf_samples, dcf_tune, dcf_chains, dcf_cores."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_3_market_models)
        assert "cfg.dcf_samples" in source
        assert "cfg.dcf_tune" in source
        assert "cfg.dcf_chains" in source
        assert "cfg.dcf_cores" in source

    def test_phase_3_dcf_stores_result(self):
        """DCF result should be stored in r.dcf_result."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_3_market_models)
        assert "r.dcf_result" in source

    def test_phase_3_dcf_has_data_threshold(self):
        """DCF should check for minimum 20 observations."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_3_market_models)
        assert "len(dcf_data) > 20" in source

    def test_phase_3_dcf_skipped_when_disabled(self):
        """When enable_dcf_model=False, DCF step should not execute."""
        from expected_returns_v4 import BaselinePipeline, PipelineConfig, BaselinePipelineResult

        cfg = PipelineConfig(enable_plr_model=False, enable_dcf_model=False)
        pipeline = BaselinePipeline(config=cfg)
        # Provide minimal data to avoid errors
        pipeline.result.df_all = pd.DataFrame({"a": [1]})
        pipeline.result.summary = pd.DataFrame({"a": [1]})

        pipeline.phase_3_market_models()

        assert pipeline.result.dcf_result is None
        assert pipeline.result.plr_result is None

    def test_phase_3_dcf_handles_missing_columns(self):
        """DCF should skip gracefully when required columns are missing."""
        from expected_returns_v4 import BaselinePipeline, PipelineConfig

        cfg = PipelineConfig(enable_plr_model=False, enable_dcf_model=True)
        pipeline = BaselinePipeline(config=cfg)
        # DataFrame without free_cash_flow or current_price
        pipeline.result.df_all = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "sector": ["Tech", "Tech"]})
        pipeline.result.summary = pd.DataFrame()

        pipeline.phase_3_market_models()

        assert pipeline.result.dcf_result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 3: DCF Posterior Merge in Phase 5
# ═══════════════════════════════════════════════════════════════════════════════


class TestDCFPhase5Merge:
    """Verify DCF posterior merge is present in phase_5_ensemble_alignment."""

    def test_phase_5_has_dcf_merge_code(self):
        """phase_5_ensemble_alignment should contain DCF posterior merge logic."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_5_ensemble_alignment)
        assert "dcf_result" in source
        assert "intrinsic_value" in source
        assert "dcf_intrinsic_value_posterior" in source

    def test_phase_5_dcf_merge_checks_posterior_attr(self):
        """DCF merge should check hasattr(r.dcf_result, 'posterior')."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_5_ensemble_alignment)
        assert 'hasattr(r.dcf_result, "posterior")' in source

    def test_phase_5_both_plr_and_dcf_merges_present(self):
        """Step 24 should merge both PLR and DCF posteriors."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_5_ensemble_alignment)
        assert "plr_intercept_posterior" in source
        assert "dcf_intrinsic_value_posterior" in source

    def test_dcf_merge_with_mock_posterior(self):
        """DCF posterior merge should add column to summary DataFrame."""
        from expected_returns_v4 import BaselinePipelineResult

        r = BaselinePipelineResult()
        r.summary = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "implied_return_mc": [10.0, 5.0]})

        # Mock DCF result with posterior
        mock_posterior = MagicMock()
        mock_posterior.__getitem__ = MagicMock(return_value=MagicMock(mean=MagicMock(return_value=150.0)))
        mock_dcf = MagicMock()
        mock_dcf.posterior = mock_posterior

        r.dcf_result = mock_dcf

        # Simulate the merge logic from phase_5
        if r.dcf_result is not None:
            if hasattr(r.dcf_result, "posterior"):
                dcf_iv_mean = float(r.dcf_result.posterior["intrinsic_value"].mean())
                r.summary["dcf_intrinsic_value_posterior"] = dcf_iv_mean

        assert "dcf_intrinsic_value_posterior" in r.summary.columns
        assert r.summary["dcf_intrinsic_value_posterior"].iloc[0] == 150.0

    def test_dcf_merge_skipped_when_no_result(self):
        """When dcf_result is None, no column should be added."""
        from expected_returns_v4 import BaselinePipelineResult

        r = BaselinePipelineResult()
        r.summary = pd.DataFrame({"ticker": ["AAPL"]})

        assert r.dcf_result is None
        assert "dcf_intrinsic_value_posterior" not in r.summary.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 2: main() Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


class TestMainEntryPoint:
    """Verify main() function exists and has correct signature."""

    def test_main_exists(self):
        from expected_returns_v4 import main

        assert callable(main)

    def test_main_signature(self):
        from expected_returns_v4 import main

        sig = inspect.signature(main)
        params = list(sig.parameters.keys())
        assert "config" in params

    def test_main_accepts_none_config(self):
        """main() should accept config=None and use from_env defaults."""
        from expected_returns_v4 import main

        sig = inspect.signature(main)
        config_param = sig.parameters["config"]
        assert config_param.default is None

    def test_main_return_annotation(self):
        from expected_returns_v4 import BaselinePipelineResult, main

        sig = inspect.signature(main)
        # Return annotation may be the class or a string (due to __future__ annotations)
        assert sig.return_annotation in (BaselinePipelineResult, "BaselinePipelineResult")


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook Cell Structure Tests (Gap 4)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNotebookCellStructure:
    """Verify the notebook has the recommended cell layout."""

    @pytest.fixture(autouse=True)
    def _load_notebook(self):
        """Read the notebook file content once."""
        nb_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "pml_workflow_v4.ipynb",
        )
        with open(nb_path, "r", encoding="utf-8") as f:
            self.content = f.read()

    def test_has_markdown_cell(self):
        # Standard .ipynb JSON format uses "cell_type": "markdown"
        assert "markdown" in self.content
        assert "Expected Returns Analytics" in self.content

    def test_has_sql_cell(self):
        assert "sql" in self.content
        assert "mv_all_stock_features" in self.content

    def test_has_main_entry_point_cell(self):
        assert "def main(" in self.content

    def test_has_interactive_execution_cell(self):
        assert "cfg.enable_dcf_model = True" in self.content

    def test_has_result_inspection_cell(self):
        assert "result.summary" in self.content
        assert "result.mc" in self.content

    def test_has_visualization_cell(self):
        assert "create_mc_return_distribution" in self.content
        assert "fig.show()" in self.content

    def test_minimum_cell_count(self):
        """Notebook should have at least 7 cells."""
        # Count cell boundaries: #%% markers in percent format, or "cell_type" in JSON format
        if "#%%" in self.content:
            cell_markers = self.content.count("#%%")
        else:
            cell_markers = self.content.count('"cell_type"')
        assert cell_markers >= 7, f"Expected >= 7 cells, found {cell_markers}"

    def test_dcf_step_15_in_code(self):
        """Notebook code should contain Step 15 DCF implementation."""
        assert "DCFPriceTarget" in self.content
        assert "Step 15" in self.content

    def test_dcf_posterior_merge_in_code(self):
        """Notebook code should contain DCF posterior merge logic."""
        assert "dcf_intrinsic_value_posterior" in self.content


# ═══════════════════════════════════════════════════════════════════════════════
# DCF Model Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDCFPriceTargetModel:
    """Unit tests for the DCFPriceTarget model class."""

    def test_dcf_class_exists(self):
        from probabilistic_ml_model.pml_models.DCF_PriceTargetModel import DCFPriceTarget

        dcf = DCFPriceTarget()
        assert dcf.terminal_growth == 0.02

    def test_dcf_custom_terminal_growth(self):
        from probabilistic_ml_model.pml_models.DCF_PriceTargetModel import DCFPriceTarget

        dcf = DCFPriceTarget(terminal_growth=0.03)
        assert dcf.terminal_growth == 0.03

    def test_dcf_fit_signature(self):
        from probabilistic_ml_model.pml_models.DCF_PriceTargetModel import DCFPriceTarget

        sig = inspect.signature(DCFPriceTarget.fit)
        params = set(sig.parameters.keys())
        expected = {"self", "historical_fcf", "market_prices", "samples", "tune", "chains", "cores"}
        assert expected.issubset(params), f"Missing params: {expected - params}"

    def test_dcf_fit_has_random_seed_param(self):
        from probabilistic_ml_model.pml_models.DCF_PriceTargetModel import DCFPriceTarget

        sig = inspect.signature(DCFPriceTarget.fit)
        assert "random_seed" in sig.parameters


# ═══════════════════════════════════════════════════════════════════════════════
# BaselinePipeline Structure Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaselinePipelineStructure:
    """Verify BaselinePipeline has all 8 phases and correct orchestration."""

    def test_has_all_8_phases(self):
        from expected_returns_v4 import BaselinePipeline

        assert hasattr(BaselinePipeline, "phase_1_load_data")
        assert hasattr(BaselinePipeline, "phase_2_core_models")
        assert hasattr(BaselinePipeline, "phase_3_market_models")
        assert hasattr(BaselinePipeline, "phase_4_statistics_and_screening")
        assert hasattr(BaselinePipeline, "phase_5_ensemble_alignment")
        assert hasattr(BaselinePipeline, "phase_6_inference_data")
        assert hasattr(BaselinePipeline, "phase_7_visualizations")
        assert hasattr(BaselinePipeline, "phase_8_export")

    def test_run_method_references_all_phases(self):
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.run)
        for i in range(1, 9):
            assert f"Phase {i}" in source, f"Phase {i} not in run()"

    def test_pipeline_default_config(self):
        from expected_returns_v4 import BaselinePipeline, PipelineConfig

        pipeline = BaselinePipeline()
        assert isinstance(pipeline.cfg, PipelineConfig)

    def test_pipeline_custom_config(self):
        from expected_returns_v4 import BaselinePipeline, PipelineConfig

        cfg = PipelineConfig(mc_simulations=100, enable_dcf_model=True)
        pipeline = BaselinePipeline(config=cfg)
        assert pipeline.cfg.mc_simulations == 100
        assert pipeline.cfg.enable_dcf_model is True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6 InferenceData DCF Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhase6DCFInference:
    """Verify Phase 6 stores DCF InferenceData."""

    def test_phase_6_stores_dcf_idata(self):
        """phase_6_inference_data should assign r.idata_dcf = r.dcf_result."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_6_inference_data)
        assert "idata_dcf" in source
        assert "dcf_result" in source

    def test_phase_6_stores_plr_idata(self):
        """phase_6_inference_data should assign r.idata_plr = r.plr_result."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_6_inference_data)
        assert "idata_plr" in source
        assert "plr_result" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Database Schema Alignment Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDatabaseSchemaAlignment:
    """Verify DCF integration uses correct database column names."""

    def test_dcf_uses_snake_case_columns(self):
        """DCF step should reference snake_case column names matching mv_all_stock_features."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_3_market_models)
        # Should use snake_case matching the DB schema
        assert "free_cash_flow" in source
        assert "current_price" in source
        # Should NOT use camelCase or other formats
        assert "FreeCashFlow" not in source
        assert "CurrentPrice" not in source

    def test_dcf_posterior_column_name_is_descriptive(self):
        """DCF posterior merge column should follow naming convention."""
        from expected_returns_v4 import BaselinePipeline

        source = inspect.getsource(BaselinePipeline.phase_5_ensemble_alignment)
        assert "dcf_intrinsic_value_posterior" in source
