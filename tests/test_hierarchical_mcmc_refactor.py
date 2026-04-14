"""Tests for hierarchical MCMC import refactoring.

Validates that:
1. hierarchical_mcmc_by_sector and hierarchical_mcmc_multi_level are importable
   from pipeline_runners (wrapper functions).
2. PipelineRunner has run_parallel_mcmc and run_resampled_posterior methods.
3. The wrapper functions produce correct results on synthetic data.
4. expected_returns_v3 imports these functions from pipeline_runners, not
   directly from statistical_models.
"""

import inspect

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Import validation
# ---------------------------------------------------------------------------


class TestImportRefactoring:
    """Ensure hierarchical MCMC functions are importable from pipeline_runners."""

    def test_hierarchical_mcmc_by_sector_importable(self):
        from probabilistic_ml_model.pipeline_runners import hierarchical_mcmc_by_sector

        assert callable(hierarchical_mcmc_by_sector)

    def test_hierarchical_mcmc_multi_level_importable(self):
        from probabilistic_ml_model.pipeline_runners import hierarchical_mcmc_multi_level

        assert callable(hierarchical_mcmc_multi_level)

    def test_pipeline_runner_has_run_parallel_mcmc(self):
        from probabilistic_ml_model.pipeline_runners import PipelineRunner

        assert hasattr(PipelineRunner, "run_parallel_mcmc")
        assert callable(getattr(PipelineRunner, "run_parallel_mcmc"))

    def test_pipeline_runner_has_run_resampled_posterior(self):
        from probabilistic_ml_model.pipeline_runners import PipelineRunner

        assert hasattr(PipelineRunner, "run_resampled_posterior")
        assert callable(getattr(PipelineRunner, "run_resampled_posterior"))

    def test_v3_imports_from_pipeline_runners(self):
        """expected_returns_v3 should import hierarchical MCMC from pipeline_runners."""
        import expected_returns_v3 as v3

        # The functions should be accessible on the v3 module
        assert hasattr(v3, "hierarchical_mcmc_by_sector")
        assert hasattr(v3, "hierarchical_mcmc_multi_level")

        # Verify they originate from pipeline_runners (not statistical_models)
        src_file = inspect.getfile(v3.hierarchical_mcmc_by_sector)
        assert "pipeline_runners" in src_file, (
            f"hierarchical_mcmc_by_sector should come from pipeline_runners, "
            f"got {src_file}"
        )
        src_file_ml = inspect.getfile(v3.hierarchical_mcmc_multi_level)
        assert "pipeline_runners" in src_file_ml, (
            f"hierarchical_mcmc_multi_level should come from pipeline_runners, "
            f"got {src_file_ml}"
        )


# ---------------------------------------------------------------------------
# Wrapper function correctness
# ---------------------------------------------------------------------------


class TestHierarchicalMCMCBySector:
    """Verify the pipeline_runners wrapper delegates correctly."""

    @pytest.fixture()
    def sector_df(self):
        rng = np.random.default_rng(42)
        n = 200
        return pd.DataFrame(
            {
                "industry": rng.choice(["Tech", "Finance", "Health"], size=n),
                "roe": rng.normal(0.12, 0.05, size=n),
            }
        )

    def test_returns_dict_with_sector_keys(self, sector_df):
        from probabilistic_ml_model.pipeline_runners import hierarchical_mcmc_by_sector

        result = hierarchical_mcmc_by_sector(sector_df, "roe")
        # May be wrapped in {"sectors": ..., "inference_data": ...}
        sectors = result.get("sectors", result) if isinstance(result, dict) else result
        assert isinstance(sectors, dict)
        assert len(sectors) > 0

    def test_posterior_mean_present(self, sector_df):
        from probabilistic_ml_model.pipeline_runners import hierarchical_mcmc_by_sector

        result = hierarchical_mcmc_by_sector(sector_df, "roe")
        sectors = result.get("sectors", result) if isinstance(result, dict) else result
        for info in sectors.values():
            if isinstance(info, dict):
                assert "posterior_mean" in info
                assert "shrinkage" in info

    def test_signature_matches_statistical_models(self):
        from probabilistic_ml_model.pipeline_runners import hierarchical_mcmc_by_sector
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            hierarchical_mcmc_by_sector as original,
        )

        wrapper_params = set(inspect.signature(hierarchical_mcmc_by_sector).parameters)
        original_params = set(inspect.signature(original).parameters)
        assert wrapper_params == original_params


class TestHierarchicalMCMCMultiLevel:
    """Verify the pipeline_runners wrapper delegates correctly."""

    @pytest.fixture()
    def multi_df(self):
        rng = np.random.default_rng(42)
        n = 300
        return pd.DataFrame(
            {
                "industry": rng.choice(["Tech", "Finance", "Health"], size=n),
                "sector": rng.choice(["IT", "Banking", "Pharma"], size=n),
                "region": rng.choice(["NA", "EU"], size=n),
                "implied_return_pt": rng.normal(10, 5, size=n),
            }
        )

    def test_returns_global_and_levels(self, multi_df):
        from probabilistic_ml_model.pipeline_runners import hierarchical_mcmc_multi_level

        result = hierarchical_mcmc_multi_level(multi_df, "implied_return_pt")
        assert "global" in result
        assert "levels" in result
        assert result["global"]["n_obs"] > 0

    def test_levels_contain_posterior_stats(self, multi_df):
        from probabilistic_ml_model.pipeline_runners import hierarchical_mcmc_multi_level

        result = hierarchical_mcmc_multi_level(multi_df, "implied_return_pt")
        for level_name, groups in result["levels"].items():
            for grp, info in groups.items():
                assert "posterior_mean" in info
                assert "shrinkage" in info
                assert "n_obs" in info

    def test_signature_matches_statistical_models(self):
        from probabilistic_ml_model.pipeline_runners import hierarchical_mcmc_multi_level
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            hierarchical_mcmc_multi_level as original,
        )

        wrapper_params = set(inspect.signature(hierarchical_mcmc_multi_level).parameters)
        original_params = set(inspect.signature(original).parameters)
        assert wrapper_params == original_params
