"""Tests for cache optimization — stripping large sample arrays, gzip, eviction."""

import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from finance_ml.ml_workflow.v3.cache import (
    _FLOAT_PRECISION,
    _RoundingEncoder,
    _evict_old_caches,
    _json_default,
    _strip_category_simulation_arrays,
    _strip_hierarchical_samples,
    _strip_large_mcmc_keys,
    load_json,
    save_json,
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
        "hierarchical": {
            "global": {"mean": 5.0, "std": 1.0, "n_obs": 400},
            "levels": {
                "region": {
                    "US": {
                        "raw_mean": 5.1,
                        "posterior_mean": 5.05,
                        "shrinkage": 0.1,
                        "n_obs": 200,
                        "prob_positive": 0.95,
                        "samples": [
                            float(x) for x in np.random.default_rng(0).normal(5.0, 1.0, 500)
                        ],
                    },
                    "EU": {
                        "raw_mean": 4.9,
                        "posterior_mean": 4.95,
                        "shrinkage": 0.15,
                        "n_obs": 150,
                        "prob_positive": 0.92,
                        "samples": [
                            float(x) for x in np.random.default_rng(1).normal(4.9, 1.0, 500)
                        ],
                    },
                },
            },
        },
    }


def _make_category_analytics():
    """Create a realistic category analytics dict with simulation arrays."""
    rng = np.random.default_rng(42)
    return {
        "Valuation": {
            "features_analyzed": 3,
            "distribution_fits": {
                "pe_ratio": {
                    "best_distribution": "norm",
                    "params": [15.0, 5.0],
                    "aic": 120.5,
                    "simulated_mean": 15.1,
                    "simulated_std": 5.2,
                    "simulations": [float(x) for x in rng.normal(15.0, 5.0, 10000)],
                },
                "pb_ratio": {
                    "best_distribution": "lognorm",
                    "params": [0.5, 0.0, 2.0],
                    "aic": 80.3,
                    "simulated_mean": 2.1,
                    "simulated_std": 1.1,
                    "simulations": [float(x) for x in rng.lognormal(0.5, 0.3, 10000)],
                },
            },
            "bayesian_results": {"pe_ratio": {"posterior_mean": 15.0, "ci_95_low": 10.0}},
            "summary_statistics": {"pe_ratio": {"mean": 15.0}},
        },
        "Growth": {
            "features_analyzed": 2,
            "distribution_fits": {
                "revenue_growth": {
                    "best_distribution": "norm",
                    "params": [0.1, 0.05],
                    "aic": 50.0,
                    "simulations": [float(x) for x in rng.normal(0.1, 0.05, 10000)],
                },
            },
            "bayesian_results": {},
        },
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


class TestStripCategorySimulationArrays:
    def test_strips_simulations(self):
        cat = _make_category_analytics()
        stripped = _strip_category_simulation_arrays(cat)
        for cat_name, cat_data in stripped.items():
            for feat, fit in cat_data.get("distribution_fits", {}).items():
                assert "simulations" not in fit, f"simulations found in {cat_name}/{feat}"

    def test_preserves_distribution_params(self):
        cat = _make_category_analytics()
        stripped = _strip_category_simulation_arrays(cat)
        val_fits = stripped["Valuation"]["distribution_fits"]
        assert val_fits["pe_ratio"]["best_distribution"] == "norm"
        assert val_fits["pe_ratio"]["params"] == [15.0, 5.0]
        assert val_fits["pe_ratio"]["aic"] == pytest.approx(120.5)

    def test_preserves_other_keys(self):
        cat = _make_category_analytics()
        stripped = _strip_category_simulation_arrays(cat)
        assert stripped["Valuation"]["features_analyzed"] == 3
        assert "bayesian_results" in stripped["Valuation"]

    def test_does_not_modify_original(self):
        cat = _make_category_analytics()
        _strip_category_simulation_arrays(cat)
        pe = cat["Valuation"]["distribution_fits"]["pe_ratio"]
        assert "simulations" in pe
        assert len(pe["simulations"]) == 10000

    def test_non_category_dict_unchanged(self):
        mcmc = {"posterior_mean": 5.0, "r_hat": 1.0}
        assert _strip_category_simulation_arrays(mcmc) is mcmc

    def test_non_dict_unchanged(self):
        assert _strip_category_simulation_arrays("hello") == "hello"


class TestStripHierarchicalSamples:
    def test_strips_samples_from_all_groups(self):
        mcmc = _make_mcmc_result()
        stripped = _strip_hierarchical_samples(mcmc)
        for level, groups in stripped["hierarchical"]["levels"].items():
            for grp, data in groups.items():
                assert "samples" not in data, f"samples found in {level}/{grp}"

    def test_preserves_summary_stats(self):
        mcmc = _make_mcmc_result()
        stripped = _strip_hierarchical_samples(mcmc)
        us = stripped["hierarchical"]["levels"]["region"]["US"]
        assert us["posterior_mean"] == pytest.approx(5.05)
        assert us["raw_mean"] == pytest.approx(5.1)
        assert us["shrinkage"] == pytest.approx(0.1)
        assert us["n_obs"] == 200

    def test_preserves_global(self):
        mcmc = _make_mcmc_result()
        stripped = _strip_hierarchical_samples(mcmc)
        assert stripped["hierarchical"]["global"]["mean"] == 5.0

    def test_does_not_modify_original(self):
        mcmc = _make_mcmc_result()
        _strip_hierarchical_samples(mcmc)
        us = mcmc["hierarchical"]["levels"]["region"]["US"]
        assert "samples" in us
        assert len(us["samples"]) == 500

    def test_no_hierarchical_key_unchanged(self):
        d = {"posterior_mean": 5.0}
        assert _strip_hierarchical_samples(d) is d


class TestSaveJsonGzip:
    def test_writes_gzip_file(self, tmp_path):
        mcmc = _make_mcmc_result()
        p = tmp_path / "mcmc.json.gz"
        save_json(p, mcmc)
        assert p.exists()
        # Verify it's valid gzip
        with gzip.open(p, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        assert "posterior_mean" in data

    def test_roundtrip_strips_large_keys(self, tmp_path):
        mcmc = _make_mcmc_result()
        p = tmp_path / "mcmc.json.gz"
        save_json(p, mcmc)
        loaded = load_json(p)
        assert "chains" not in loaded
        assert "combined_samples" not in loaded
        assert loaded["posterior_mean"] == pytest.approx(mcmc["posterior_mean"], abs=1e-5)

    def test_strips_hierarchical_samples(self, tmp_path):
        mcmc = _make_mcmc_result()
        p = tmp_path / "mcmc.json.gz"
        save_json(p, mcmc)
        loaded = load_json(p)
        for level, groups in loaded["hierarchical"]["levels"].items():
            for grp, data in groups.items():
                assert "samples" not in data

    def test_strips_category_simulations(self, tmp_path):
        cat = _make_category_analytics()
        p = tmp_path / "cat.json.gz"
        save_json(p, cat)
        loaded = load_json(p)
        for cat_name, cat_data in loaded.items():
            for feat, fit in cat_data.get("distribution_fits", {}).items():
                assert "simulations" not in fit

    def test_file_size_small_mcmc(self, tmp_path):
        mcmc = _make_mcmc_result(n_chains=8, n_samples=25000)
        p = tmp_path / "mcmc.json.gz"
        save_json(p, mcmc)
        size_kb = p.stat().st_size / 1024
        assert size_kb < 50, f"Cache file too large: {size_kb:.1f} KB"

    def test_file_size_small_category(self, tmp_path):
        cat = _make_category_analytics()
        p = tmp_path / "cat.json.gz"
        save_json(p, cat)
        size_kb = p.stat().st_size / 1024
        assert size_kb < 10, f"Cache file too large: {size_kb:.1f} KB"

    def test_adds_gz_extension_if_missing(self, tmp_path):
        cat = {"Valuation": {"features_analyzed": 5}}
        p = tmp_path / "test.json"
        save_json(p, cat)
        assert (tmp_path / "test.json.gz").exists()

    def test_category_analytics_roundtrip(self, tmp_path):
        cat = {
            "Valuation": {
                "features_analyzed": 5,
                "mcmc_posterior": {"mean": 0.123456789012, "std": 0.05},
            }
        }
        p = tmp_path / "cat.json.gz"
        save_json(p, cat)
        loaded = load_json(p)
        assert loaded["Valuation"]["features_analyzed"] == 5


class TestLoadJsonBackcompat:
    def test_loads_plain_json_fallback(self, tmp_path):
        """Old plain-JSON files should still be readable."""
        p = tmp_path / "old.json"
        p.write_text(json.dumps({"key": "value"}))
        loaded = load_json(p)
        assert loaded == {"key": "value"}

    def test_prefers_gz_over_plain(self, tmp_path):
        """If both .json and .json.gz exist, prefer .gz."""
        p_json = tmp_path / "data.json"
        p_gz = tmp_path / "data.json.gz"
        p_json.write_text(json.dumps({"source": "plain"}))
        with gzip.open(p_gz, "wb") as f:
            f.write(json.dumps({"source": "gzip"}).encode("utf-8"))
        loaded = load_json(p_json)
        assert loaded["source"] == "gzip"

    def test_returns_none_for_missing(self, tmp_path):
        p = tmp_path / "missing.json.gz"
        assert load_json(p) is None


class TestCacheEviction:
    def test_evicts_old_files(self, tmp_path):
        subdir = tmp_path / "mcmc_return"
        subdir.mkdir()
        import time

        for i in range(5):
            (subdir / f"file_{i}.json.gz").write_bytes(b"x")
            time.sleep(0.05)
        removed = _evict_old_caches(tmp_path, "mcmc_return", max_files=2)
        assert removed == 3
        remaining = list(subdir.iterdir())
        assert len(remaining) == 2

    def test_keeps_most_recent(self, tmp_path):
        subdir = tmp_path / "test_sub"
        subdir.mkdir()
        import time

        for i in range(4):
            (subdir / f"file_{i}.json.gz").write_bytes(b"x")
            time.sleep(0.05)
        _evict_old_caches(tmp_path, "test_sub", max_files=2)
        remaining = sorted(f.name for f in subdir.iterdir())
        assert "file_2.json.gz" in remaining
        assert "file_3.json.gz" in remaining

    def test_no_error_on_missing_dir(self, tmp_path):
        removed = _evict_old_caches(tmp_path, "nonexistent", max_files=2)
        assert removed == 0

    def test_save_json_triggers_eviction(self, tmp_path):
        subdir = tmp_path / "mcmc_return"
        subdir.mkdir()
        import time

        for i in range(4):
            (subdir / f"old_{i}.json.gz").write_bytes(b"x")
            time.sleep(0.05)
        p = subdir / "new.json.gz"
        save_json(p, {"posterior_mean": 5.0, "chains": [[1, 2]], "combined_samples": [1]})
        remaining = list(subdir.iterdir())
        assert len(remaining) <= 2


class TestRoundingEncoder:
    def test_rounds_native_floats_in_lists(self):
        data = {"values": [1.123456789012345, 2.987654321098765]}
        result = json.loads(json.dumps(data, cls=_RoundingEncoder))
        assert result["values"][0] == round(1.123456789012345, _FLOAT_PRECISION)
        assert result["values"][1] == round(2.987654321098765, _FLOAT_PRECISION)

    def test_rounds_nested_floats(self):
        data = {"a": {"b": 3.141592653589793}}
        result = json.loads(json.dumps(data, cls=_RoundingEncoder))
        assert result["a"]["b"] == round(3.141592653589793, _FLOAT_PRECISION)

    def test_preserves_integers(self):
        data = {"count": 42}
        result = json.loads(json.dumps(data, cls=_RoundingEncoder))
        assert result["count"] == 42

    def test_handles_numpy_types(self):
        data = {"val": np.float64(1.123456789012345), "arr": np.array([1.1, 2.2])}
        result = json.loads(json.dumps(data, cls=_RoundingEncoder))
        assert result["val"] == round(1.123456789012345, _FLOAT_PRECISION)


class TestJsonDefault:
    def test_numpy_float_rounded(self):
        val = _json_default(np.float64(1.123456789012345))
        assert val == round(1.123456789012345, _FLOAT_PRECISION)

    def test_python_float_rounded(self):
        val = _json_default(1.123456789012345)
        assert val == round(1.123456789012345, _FLOAT_PRECISION)

    def test_numpy_int(self):
        assert _json_default(np.int64(42)) == 42

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _json_default(arr)
        assert result == [1.0, 2.0, 3.0]
