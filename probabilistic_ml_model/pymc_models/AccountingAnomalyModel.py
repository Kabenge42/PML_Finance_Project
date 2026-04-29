"""
Accounting Anomaly Bayesian Model — Multi-layer anomaly detection.

Models z-score components for Mahalanobis-style anomaly detection, producing
a sigmoid-based anomaly probability per stock.

Performance note
----------------
The previous implementation declared a latent ``z_scores`` matrix of shape
``(n_stocks, n_features)`` as an unobserved ``pm.Normal``. With typical inputs
(~2,000 stocks × ~30 features) this produced ~60,000 free parameters — one
per observation — which is unidentifiable and forces NUTS to evaluate a
massive gradient on every leapfrog step. Sampling four chains of 1,000 draws
took ~15 minutes and mixed poorly.

The refactored model standardizes the observed feature matrix once in NumPy
(it is *already* the empirical z-score matrix), keeps only a small number of
identifiable Bayesian parameters (per-feature scale ``feature_scale`` and a
learnable ``threshold``), and computes ``anomaly_prob`` deterministically
from the observed data. Sampling now runs in seconds while preserving the
same Bayesian semantics and ``constant_data`` contract.

Reference: AccountingAnomalyProbabilityModel in probability_models.py (line 237);
           build_accounting_anomaly_inference_data() in inference_schema.py (line 676).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional, TYPE_CHECKING

from arviz import InferenceData
from pymc.backends.base import MultiTrace

try:
    import arviz as az
except ImportError:
    try:
        import arviz_base as az
    except ImportError:
        az = None  # type: ignore[assignment]

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:
    pm = None  # type: ignore[assignment]
    pt = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import arviz as az_typing  # noqa: F401
    import pymc as pm_typing  # noqa: F401

from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs
from probabilistic_ml_model.data_utils.data_utils import load_feature_categories_from_db

logger = logging.getLogger(__name__)

# Canonical category names in public.calculated_features_registry whose
# feature_aliases drive the auxiliary "anomaly_feature" dim. Aligned with
# the accounting-quality / risk / unusual-items features feeding the
# anomaly-detection model.
_ANOMALY_CATEGORY_KEYS: tuple[str, ...] = (
    "Accounting Quality",
    "Quality & Risk",
    "Unusual Items",
)


class AccountingAnomalyBayesian:
    """Bayesian accounting anomaly detection model.

    Parameters
    ----------
    threshold : float
        Sigmoid threshold shift for anomaly probability.
    """

    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[az_typing.InferenceData] = None

    @staticmethod
    @lru_cache(maxsize=4)
    def _resolve_anomaly_feature_aliases(
        connection_string: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Resolve the anomaly feature aliases from calculated_features_registry.

        These serve as labelled coordinates along the ``anomaly_feature`` dim
        for the auxiliary ``pm.Data`` container of observed anomaly-feature
        values (accounting-quality + risk + unusual-items signals).
        """
        try:
            categories = load_feature_categories_from_db(connection_string)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load feature categories: %s", exc)
            return tuple()
        aliases: list[str] = []
        seen: set[str] = set()
        for key in _ANOMALY_CATEGORY_KEYS:
            for alias in categories.get(key, []):
                if alias not in seen:
                    seen.add(alias)
                    aliases.append(alias)
        return tuple(aliases)

    @staticmethod
    def _align_anomaly_features(
        anomaly_features_df: Optional[pd.DataFrame],
        isin: np.ndarray,
        feature_aliases: list[str],
    ) -> np.ndarray:
        """Align an (isin × anomaly_feature) matrix to the model dims.

        Missing columns/rows are filled with ``0.0`` so the container always
        has shape ``(n_isin, n_anomaly_feature)``.
        """
        n_isin = len(isin)
        n_feat = len(feature_aliases)
        if anomaly_features_df is None or n_feat == 0 or n_isin == 0:
            return np.zeros((n_isin, max(n_feat, 0)), dtype="float64")

        df = anomaly_features_df.copy()
        if "isin" in df.columns:
            df = df.drop_duplicates(subset="isin").set_index("isin")

        aligned = df.reindex(index=isin, columns=feature_aliases)
        return aligned.astype("float64").fillna(0.0).to_numpy()

    def fit(
        self,
        feature_values: np.ndarray,
        isins: np.ndarray,
        feature_names: list[str] | None = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[InferenceData | MultiTrace, Any]:
        """Fit anomaly model and return ``(InferenceData, Model)``."""
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install pymc + arviz to use AccountingAnomalyBayesian."
            )

        feature_values = np.asarray(feature_values, dtype="float64")
        if feature_values.ndim != 2 or feature_values.size == 0:
            raise ValueError("feature_values must be 2-D and non-empty.")
        n_stocks, n_features = feature_values.shape

        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(n_features)]

        # Standardize columns once (NaN-safe). The model below treats the
        # standardized matrix as the empirical z-score matrix, so we do not
        # need to introduce a latent (n_stocks, n_features) Normal — which
        # would be unidentifiable and dominate sampling cost.
        col_mean = np.nanmean(feature_values, axis=0)
        col_std = np.nanstd(feature_values, axis=0)
        col_std = np.where(col_std < 1e-9, 1.0, col_std)
        z_matrix = (feature_values - col_mean) / col_std
        z_matrix = np.nan_to_num(z_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize the aggregate so its prior std is O(1) regardless of
        # n_features (sum of n unit-variance terms has std sqrt(n)).
        norm = float(np.sqrt(max(n_features, 1)))

        coords = {
            "isin": np.asarray(isins),
            "feature": list(feature_names),
        }

        with pm.Model(coords=coords) as model:
            feat_data = pm.Data("feature_values", z_matrix, dims=("isin", "feature"))

            # Small, identifiable Bayesian layer:
            # - per-feature positive scale (importance weight)
            # - learnable threshold around the user-supplied prior mean
            feature_scale = pm.HalfNormal("feature_scale", sigma=1.0, dims="feature")
            threshold = pm.Normal("threshold", mu=self.threshold, sigma=1.0)

            weighted = feat_data * feature_scale  # (isin, feature)
            agg_score = pt.sum(weighted, axis=1) / norm

            pm.Deterministic("agg_score", agg_score, dims="isin")
            pm.Deterministic(
                "anomaly_prob",
                pm.math.sigmoid(agg_score - threshold),
                dims="isin",
            )

            obs_sigma = pm.HalfNormal("obs_sigma", sigma=1.0)

            # Likelihood: observed standardized features are noisy reflections
            # of their (scale-weighted) latent mean. This is identifiable
            # because feature_scale is shared across all stocks per feature.
            pm.Normal(
                "feature_obs",
                mu=weighted,
                sigma=obs_sigma,
                observed=feat_data,
                dims=("isin", "feature"),
            )

            scall: dict[str, Any] = dict(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True,
                compile_kwargs=get_pytensor_compile_kwargs(),
            )
            if nuts_sampler is not None:
                scall["nuts_sampler"] = nuts_sampler
            scall.setdefault("idata_kwargs", {"log_likelihood": False})
            scall.update(sample_kwargs)

            idata = pm.sample(**scall)

        # Some external NUTS samplers (e.g. nutpie) return an InferenceData
        # without the ``constant_data`` group that PyMC normally attaches for
        # ``pm.Data`` containers. Add it manually so downstream code can rely
        # on ``idata.constant_data`` regardless of the sampler used.
        if "constant_data" not in idata.groups():
            import xarray as xr

            constant_ds = xr.Dataset(
                {"feature_values": (("isin", "feature"), z_matrix)},
                coords={
                    "isin": np.asarray(isins),
                    "feature": list(feature_names),
                },
            )
            idata.add_groups({"constant_data": constant_ds})

        self.model_ = model
        self.idata_ = idata
        return idata, model
