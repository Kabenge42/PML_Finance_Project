"""Tests for fit_distributions_by_category division-by-zero handling."""

import warnings

import numpy as np
import pandas as pd
import pytest

from probabilistic_ml_model.statistical_functions.statistical_models import (
    fit_distributions_by_category,
)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


def _make_df(n: int, columns: dict[str, np.ndarray]) -> pd.DataFrame:
    """Helper to build a DataFrame from column arrays."""
    return pd.DataFrame(columns)


class TestFitDistributionsByCategory:
    """Ensure degenerate / edge-case data does not produce RuntimeWarnings."""

    def test_constant_column_skipped(self):
        """A column with zero variance should be silently skipped."""
        df = pd.DataFrame({"const": np.full(200, 5.0)})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = fit_distributions_by_category(df, "test", ["const"])
        assert result == {}

    def test_near_constant_column_skipped(self):
        """A column that becomes constant after quantile trimming."""
        arr = np.full(300, 7.0)
        arr[0] = 0.0  # single outlier keeps pre-trim variance > 0
        arr[1] = 100.0
        df = pd.DataFrame({"near_const": arr})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = fit_distributions_by_category(df, "test", ["near_const"])
        # Either skipped or fitted without warning — both are acceptable
        assert isinstance(result, dict)

    def test_all_nan_column_skipped(self):
        """All-NaN column should be skipped (len < 100 after dropna)."""
        df = pd.DataFrame({"nans": [np.nan] * 200})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = fit_distributions_by_category(df, "test", ["nans"])
        assert result == {}

    def test_normal_data_no_warnings(self, rng):
        """Well-behaved normal data should fit without any RuntimeWarning."""
        data = rng.normal(loc=10, scale=2, size=500)
        df = pd.DataFrame({"feature": data})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = fit_distributions_by_category(df, "Normal", ["feature"])
        assert "feature" in result
        assert result["feature"]["best_distribution"] in {
            "normal",
            "student_t",
            "skew_normal",
        }

    def test_mixed_degenerate_and_valid(self, rng):
        """Mix of a constant column and a valid column — only valid produces results."""
        df = pd.DataFrame(
            {
                "const": np.full(300, 42.0),
                "valid": rng.normal(5, 1, size=300),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = fit_distributions_by_category(
                df, "mixed", ["const", "valid"]
            )
        assert "const" not in result
        assert "valid" in result

    def test_binary_flag_column_skipped(self):
        """A binary 0/1 column (very low variance) should not cause warnings."""
        arr = np.zeros(300)
        arr[:10] = 1.0  # very few 1s
        df = pd.DataFrame({"flag": arr})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = fit_distributions_by_category(df, "flags", ["flag"])
        # May or may not fit, but must not warn
        assert isinstance(result, dict)
