"""
Accounting Anomaly Bayesian Model — Multi-layer anomaly detection.

Models z-score components for Mahalanobis-style anomaly detection, producing
a sigmoid-based anomaly probability per stock.

Reference: AccountingAnomalyProbabilityModel in probability_analytics.py (line 237);
           build_accounting_anomaly_inference_data() in inference_schema.py (line 676).
"""

from __future__ import annotations

import logging

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger(__name__)


class AccountingAnomalyBayesian:
    """Bayesian accounting anomaly detection model.

    Places Normal priors on per-stock z-score components, then derives
    an anomaly probability via sigmoid of the aggregate score.

    Parameters
    ----------
    threshold : float
        Sigmoid threshold shift for anomaly probability.
    """

    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold

    def fit(
        self,
        feature_values: np.ndarray,
        tickers: np.ndarray,
        feature_names: list[str] | None = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Fit anomaly model and return InferenceData.

        Parameters
        ----------
        feature_values : array (n_stocks, n_features)
            Observed feature values (e.g. financial ratios).
        tickers : array of str
            Ticker symbols.
        feature_names : list of str, optional
            Names for each feature column.
        samples, tune, chains, target_accept, random_seed
            MCMC sampling parameters.

        Returns
        -------
        az.InferenceData
            Posterior contains ``z_scores``, ``anomaly_prob``.
        """
        feature_values = np.asarray(feature_values, dtype="float64")
        n_stocks, n_features = feature_values.shape

        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(n_features)]

        coords = {
            "ticker": tickers,
            "feature": feature_names,
        }

        with pm.Model(coords=coords) as model:
            z_scores = pm.Normal(
                "z_scores",
                mu=0,
                sigma=1,
                dims=("ticker", "feature"),
            )

            agg_score = pt.sum(z_scores, axis=1)
            anomaly_prob = pm.Deterministic(
                "anomaly_prob",
                pm.math.sigmoid(agg_score - self.threshold),
                dims="ticker",
            )

            pm.Normal(
                "feature_obs",
                mu=z_scores,
                sigma=1.0,
                observed=feature_values,
                dims=("ticker", "feature"),
            )

            idata = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
            )

        return idata
