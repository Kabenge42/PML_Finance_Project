"""
Accounting Anomaly Bayesian Model — Multi-layer anomaly detection.

Models z-score components for Mahalanobis-style anomaly detection, producing
a sigmoid-based anomaly probability per stock.

Reference: AccountingAnomalyProbabilityModel in probability_models.py (line 237);
           build_accounting_anomaly_inference_data() in inference_schema.py (line 676).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

try:
    import arviz as az
except ImportError:
    try:
        import arviz_base as az
    except ImportError:
        az = None  # type: ignore[assignment]

import numpy as np

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

logger = logging.getLogger(__name__)


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
    ) -> tuple[az_typing.InferenceData, pm_typing.Model]:
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

        coords = {
            "isin": np.asarray(isins),
            "feature": list(feature_names),
        }

        with pm.Model(coords=coords) as model:
            feat_data = pm.Data("feature_values", feature_values, dims=("isin", "feature"))

            z_scores = pm.Normal("z_scores", mu=0, sigma=1, dims=("isin", "feature"))

            agg_score = pt.sum(z_scores, axis=1)
            pm.Deterministic(
                "anomaly_prob",
                pm.math.sigmoid(agg_score - self.threshold),
                dims="isin",
            )

            obs_sigma = pm.HalfNormal("obs_sigma", sigma=1.0)

            pm.Normal(
                "feature_obs",
                mu=z_scores,
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

        self.model_ = model
        self.idata_ = idata
        return idata, model
