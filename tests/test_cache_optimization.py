"""Tests for MCMC cache optimization — stripping large sample arrays."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from finance_ml.ml_workflow.v3.cache import (
    _strip_large_mcmc_keys,
    save_json,
    load_json,
    _json_default,
)


def _make_mcmc_result(n_chains=4, n_samples=100):
    """Create a realistic MCMC result dict with raw sample arrays."""
    chains = [np.random.default_rng(i).normal(5.0, 1.0, n_samples) for i in range(n_chains)]
    combined = np.concatenate(chains)
    return {
        "chains": chains,
        "combined_samples": combined,
        "inference_data": "fake_idata",
        "chain_means": [float(np.mean(c)) for c in chains],
        "chain_stds": [float(np.std(c)) for c in chains],
        "posterior_mean": float(np.mean(combined)),
        "posterior_std": float(np.std(combined)),
        "ci_95": [float(np.percentile(combined, 2.5)), float(np.percentile(combined, 97.5))],
        "r_hat": 1.001,
        "ess_bulk": 3200.0,
        "ess_tail": 2800.0,
        "converged": True,
        "student_t_mu": 5.01,
        "student_t_df": 8.5,
        "hierarchical": {"global": {"mean": 5.0, "std": 1.0, "n_obs": 400}},
    }


class TestStripLargeMcmcKeys:
    def test_strips_chains_and_combined_samples(self):
        mcmc = _make_mcmc_result()
        stripped = _strip_large_mcmc_keys(mcmc)
        assert "chains" not in stripped
        assert "combined_samples" not in stripped
        assert "inference_data" not in stripped

    def test_preserves_summary_keys(self):
        mcmc = _make_mcmc_result()
        stripped = _strip_large_mcmc_keys(mcmc)
        for key in [
            "posterior_mean", "posterior_std", "r_hat", "converged",
            "ci_95", "ess_bulk", "ess_tail", "chain_means", "chain_stds",
            "student_t_mu", "student_t_df", "hierarchical",
        ]:
            assert key in stripped, f"Missing key: {key}"

    def test_does_not_modify_original(self):
        mcmc = _make_mcmc_result()
        _strip_large_mcmc_keys(mcmc)
        assert "chains" in mcmc
        assert "combined_samples" in mcmc

    def test_category_analytics_unchanged(self):
        cat = {"Valuation": {"features_analyzed": 5, "mean": 0.5}}
        assert _strip_large_mcmc_keys(cat) is cat

    def test_non_dict_unchanged(self):
        assert _strip_large_mcmc_keys([1, 2, 3]) == [1, 2, 3]


class TestSaveJsonMcmc:
    def test_roundtrip_strips_large_keys(self, tmp_path):
        mcmc = _make_mcmc_result()
        p = tmp_path / "mcmc.json"
        save_json(p, mcmc)
        loaded = load_json(p)
        assert "chains" not in loaded
        assert "combined_samples" not in loaded
        assert loaded["posterior_mean"] == pytest.approx(mcmc["posterior_mean"], abs=1e-7)

    def test_file_size_small(self, tmp_path):
        mcmc = _make_mcmc_result(n_chains=8, n_samples=25000)
        p = tmp_path / "mcmc.json"
        save_json(p, mcmc)
        size_kb = p.stat().st_size / 1024
        # Without stripping this would be ~130 MB; with stripping < 10 KB
        assert size_kb < 50, f"Cache file too large: {size_kb:.1f} KB"

    def test_category_analytics_roundtrip(self, tmp_path):
        cat = {
            "Valuation": {
                "features_analyzed": 5,
                "mcmc_posterior": {"mean": 0.123456789012, "std": 0.05},
            }
        }
        p = tmp_path / "cat.json"
        save_json(p, cat)
        loaded = load_json(p)
        assert loaded["Valuation"]["features_analyzed"] == 5


class TestJsonDefault:
    def test_numpy_float_rounded(self):
        val = _json_default(np.float64(1.123456789012345))
        assert val == round(1.123456789012345, 8)

    def test_python_float_rounded(self):
        val = _json_default(1.123456789012345)
        assert val == round(1.123456789012345, 8)

    def test_numpy_int(self):
        assert _json_default(np.int64(42)) == 42

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert _json_default(arr) == [1.0, 2.0, 3.0]
