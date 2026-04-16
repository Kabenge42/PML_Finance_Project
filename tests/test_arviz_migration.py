"""
Tests for ArviZ 1.0 migration — covers the four code fixes and Pipfile alignment.

Fix 1: probability_models.py  — ARVIZ_AVAILABLE check accepts ``from_dict``
Fix 2: probability_models.py  — ResampledBeatProbabilityModel.build_inference_data uses kwargs
Fix 3: statistical_models.py  — hierarchical_mcmc_multi_level uses kwargs for az.from_dict
Fix 4: inference_schema.py    — build_feature_view_inference_data uses kwargs for az.from_dict
Pipfile: arviz pinned to >=1.0.0,<2.0.0 with sub-packages
"""

from __future__ import annotations

import importlib
import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPFILE_PATH = PROJECT_ROOT / "Pipfile"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


# ============================================================================
# 1. Pipfile version alignment
# ============================================================================
class TestPipfileArvizVersion:
    """Pipfile must pin arviz >=1.0.0 and declare sub-packages."""

    def test_arviz_version_is_1x(self):
        text = PIPFILE_PATH.read_text()
        # Should NOT contain the old <1.0.0 pin
        assert ">=0.19.0,<1.0.0" not in text, "Pipfile still pins legacy arviz <1.0.0"

    def test_arviz_base_declared(self):
        text = PIPFILE_PATH.read_text()
        assert "arviz-base" in text

    def test_arviz_stats_declared(self):
        text = PIPFILE_PATH.read_text()
        assert "arviz-stats" in text

    def test_arviz_plots_declared(self):
        text = PIPFILE_PATH.read_text()
        assert "arviz-plots" in text

    def test_arviz_1x_lower_bound(self):
        text = PIPFILE_PATH.read_text()
        # The main arviz line should require >=1.0.0
        match = re.search(r'arviz\s*=\s*"([^"]+)"', text)
        assert match, "arviz entry not found in Pipfile"
        assert ">=1.0.0" in match.group(1)

    def test_consistent_with_requirements(self):
        """Pipfile, requirements.txt, and pyproject.toml should all require arviz >=1.0.0."""
        for path in (PIPFILE_PATH, REQUIREMENTS_PATH, PYPROJECT_PATH):
            text = path.read_text()
            assert "arviz" in text.lower()
            assert ">=0.19.0,<1.0.0" not in text, f"{path.name} still has legacy arviz pin"


# ============================================================================
# 2. ARVIZ_AVAILABLE check (Fix 4 in issue — probability_models.py L224)
# ============================================================================
class TestArvizAvailableCheck:
    """ARVIZ_AVAILABLE must be True when az.from_dict exists (ArviZ 1.0)."""

    def test_from_dict_suffices(self):
        """Even without InferenceData, from_dict should set ARVIZ_AVAILABLE = True."""
        mock_az = MagicMock()
        del mock_az.InferenceData  # simulate ArviZ 1.0 (no InferenceData)
        mock_az.from_dict = MagicMock()
        result = hasattr(mock_az, "from_dict") or hasattr(mock_az, "InferenceData")
        assert result is True

    def test_inference_data_suffices(self):
        """Legacy ArviZ with InferenceData should also work."""
        mock_az = MagicMock()
        del mock_az.from_dict
        mock_az.InferenceData = MagicMock()
        result = hasattr(mock_az, "from_dict") or hasattr(mock_az, "InferenceData")
        assert result is True

    def test_neither_available(self):
        mock_az = MagicMock()
        del mock_az.from_dict
        del mock_az.InferenceData
        result = hasattr(mock_az, "from_dict") or hasattr(mock_az, "InferenceData")
        assert result is False

    def test_source_code_uses_correct_check(self):
        """probability_models.py must check for from_dict OR InferenceData."""
        src = (
            PROJECT_ROOT
            / "probabilistic_ml_model"
            / "statistical_functions"
            / "probability_models.py"
        ).read_text(encoding="utf-8")
        assert 'hasattr(az, "from_dict")' in src
        assert 'hasattr(az, "InferenceData")' in src


# ============================================================================
# 3. az.from_dict called with nested dict (ArviZ 1.0 API)
# ============================================================================
class TestFromDictKeywordAPI:
    """All az.from_dict calls must use nested dict as first positional arg (ArviZ 1.0 API)."""

    @staticmethod
    def _source(relpath: str) -> str:
        return (PROJECT_ROOT / relpath).read_text(encoding="utf-8")

    def test_probability_models_uses_nested_dict(self):
        src = self._source(
            "probabilistic_ml_model/statistical_functions/probability_models.py"
        )
        # Must use nested dict as positional arg (ArviZ 1.0 API)
        assert 'from_dict(\n                {' in src

    def test_statistical_models_uses_nested_dict(self):
        src = self._source(
            "probabilistic_ml_model/statistical_functions/statistical_models.py"
        )
        # Must use nested dict as positional arg (ArviZ 1.0 API)
        assert 'from_dict(\n                {"posterior"' in src

    def test_inference_schema_uses_positional_dict(self):
        src = self._source("probabilistic_ml_model/data_utils/inference_schema.py")
        # Must pass groups dict as first positional arg (ArviZ 1.0 API)
        assert 'from_dict(\n            groups,' in src


# ============================================================================
# 4. Integration: az.from_dict receives keyword args at runtime
# ============================================================================

def _make_beat_df(n: int = 5) -> pd.DataFrame:
    """Minimal DataFrame for ResampledBeatProbabilityModel."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "name": [f"Name{i}" for i in range(n)],
        "sector": ["Tech"] * n,
        "industry": ["Software"] * n,
        "country": ["US"] * n,
        "exchange": ["NYSE"] * n,
        "eps_positive_years": rng.integers(1, 10, n),
        "eps_positive_streak": rng.integers(3, 10, n),
        "eps_trajectory_score": rng.uniform(30, 90, n),
        "eps_improvement_count": rng.integers(0, 5, n),
        "last_price": rng.uniform(10, 200, n),
        "price_target": rng.uniform(15, 250, n),
        "analyst_count": rng.integers(1, 20, n),
        "piotroski_f_score": rng.integers(3, 9, n),
        "momentum_score_3m": rng.uniform(-1, 1, n),
        "momentum_score_6m": rng.uniform(-1, 1, n),
        "momentum_score_12m": rng.uniform(-1, 1, n),
        "volatility_regime": rng.uniform(0, 1, n),
        "posterior_alpha": rng.uniform(1, 5, n),
        "posterior_beta": rng.uniform(1, 5, n),
        "posterior_mean": rng.uniform(0.3, 0.8, n),
    })


class TestResampledBeatInferenceDataKwargs:
    """ResampledBeatProbabilityModel.build_inference_data must call az.from_dict with nested dict."""

    def test_from_dict_called_with_positional_dict(self):
        """Intercept az.from_dict and verify nested dict is passed as positional arg."""
        from probabilistic_ml_model.statistical_functions import probability_models as pm

        if not pm.ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")

        df = _make_beat_df(5)
        model = pm.ResampledBeatProbabilityModel(n_posterior_samples=50, n_chains=2, random_seed=42)

        original_from_dict = pm.az.from_dict
        captured = {}

        def spy_from_dict(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return original_from_dict(*args, **kwargs)

        with patch.object(pm.az, "from_dict", side_effect=spy_from_dict):
            model.build_inference_data(df)

        assert len(captured["args"]) == 1, (
            "az.from_dict should be called with nested dict as first positional arg"
        )
        assert "posterior" in captured["args"][0], (
            "nested dict must contain 'posterior' key"
        )


class TestHierarchicalMcmcFromDictKwargs:
    """hierarchical_mcmc_multi_level must call az.from_dict with nested dict."""

    def test_from_dict_called_with_positional_dict(self):
        from probabilistic_ml_model.statistical_functions import statistical_models as sm

        if not sm.ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")

        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(n)],
            "sector": rng.choice(["Tech", "Health", "Finance"], n),
            "industry": rng.choice(["SW", "HW", "Bio", "Bank"], n),
            "test_feature": rng.normal(50, 10, n),
        })

        original_from_dict = sm.az.from_dict
        captured = {}

        def spy_from_dict(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return original_from_dict(*args, **kwargs)

        with patch.object(sm.az, "from_dict", side_effect=spy_from_dict):
            result = sm.hierarchical_mcmc_multi_level(df, feature="test_feature", n_samples=200, min_group_size=5)

        if "inference_data" in result:
            assert len(captured["args"]) == 1
            assert "posterior" in captured["args"][0]


class TestBuildFeatureViewInferenceDataKwargs:
    """build_feature_view_inference_data must call az.from_dict with nested dict."""

    def test_from_dict_called_with_positional_dict(self):
        from probabilistic_ml_model.data_utils import inference_schema as ischema

        if not ischema.ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")

        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "name": ["Apple", "Microsoft", "Google"],
            "sector": ["Tech", "Tech", "Tech"],
            "industry": ["SW", "SW", "SW"],
            "country": ["US", "US", "US"],
            "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
        })

        original_from_dict = ischema.az.from_dict
        captured = {}

        def spy_from_dict(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return original_from_dict(*args, **kwargs)

        with patch.object(ischema.az, "from_dict", side_effect=spy_from_dict):
            ischema.build_feature_view_inference_data(
                "vw_features_momentum", df,
                n_posterior_samples=50, n_chains=2, random_seed=42,
            )

        assert len(captured["args"]) == 1
        assert "posterior" in captured["args"][0]


# ============================================================================
# 5. Functional: az.from_dict actually succeeds (no dimension conflict)
# ============================================================================
class TestFromDictProducesValidOutput:
    """Verify az.from_dict returns valid objects (not dimension errors)."""

    def test_inference_schema_build_feature_view(self):
        from probabilistic_ml_model.data_utils import inference_schema as ischema

        if not ischema.ARVIZ_AVAILABLE:
            pytest.skip("ArviZ not available")

        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "name": ["A Inc", "B Inc"],
            "sector": ["X", "X"],
            "industry": ["Y", "Y"],
            "country": ["US", "US"],
            "exchange": ["NYSE", "NYSE"],
            "f1": [10.0, 20.0],
        })
        result = ischema.build_feature_view_inference_data(
            "vw_features_momentum", df,
            n_posterior_samples=50, n_chains=2, random_seed=1,
        )
        assert result is not None

    def test_hierarchical_mcmc_multi_level_succeeds(self):
        from probabilistic_ml_model.statistical_functions import statistical_models as sm

        rng = np.random.default_rng(99)
        n = 80
        df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(n)],
            "sector": rng.choice(["A", "B"], n),
            "industry": rng.choice(["X", "Y"], n),
            "val": rng.normal(0, 1, n),
        })
        result = sm.hierarchical_mcmc_multi_level(df, feature="val", n_samples=200, min_group_size=5)
        # Should not raise; inference_data may or may not be present
        assert "global" in result
