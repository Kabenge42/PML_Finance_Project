"""
Probabilistic Linear Regression Model — Standard Bayesian regression.

Implements a Bayesian linear regression with Normal priors on intercept and
coefficients, HalfNormal on noise, and posterior predictive checks.

Reference: PyMC_overview.md linear regression example;
           inference_schema.py _build_posterior_samples_normal() (line 289).
"""

from __future__ import annotations
import os
import logging
from typing import Optional, Sequence

# Ensure C backend is disabled before any pytensor import
# (guards against cases where PYTENSOR_FLAGS wasn't set early enough)
if not os.environ.get("PYTENSOR_FLAGS"):
    os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64,cxx="
elif "cxx=" not in os.environ["PYTENSOR_FLAGS"]:
    os.environ["PYTENSOR_FLAGS"] += ",cxx="

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger(__name__)


class ProbabilisticLinearRegression:
    """Bayesian linear regression model using PyMC.

    Priors
    ------
    alpha ~ Normal(0, 10)
    betas ~ Normal(0, 10) per feature
    sigma ~ HalfNormal(1)
    """

    def __init__(self) -> None:
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        obs_names: Optional[np.ndarray] = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 1,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Fit Bayesian linear regression and return InferenceData.

        Parameters
        ----------
        X : array (n_obs, n_features)
            Feature matrix.
        y : array (n_obs,)
            Target vector.
        feature_names : list of str, optional
            Names for each feature (column of X).
        obs_names : array of str, optional
            Observation identifiers (e.g. tickers).
        samples, tune, chains, cores, target_accept, random_seed
            MCMC sampling parameters.  ``cores=1`` avoids multiprocessing
            overhead on Windows where each spawned worker must re-import
            PyTensor in Python-only mode (no C compiler).

        Returns
        -------
        az.InferenceData
        """
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64")
        n_obs, n_features = X.shape

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        if obs_names is None:
            obs_names = np.arange(n_obs)

        coords = {
            "feature": list(feature_names),
            "obs": obs_names,
        }

        with pm.Model(coords=coords) as model:
            X_data = pm.Data("X", X, dims=("obs", "feature"))

            alpha = pm.Normal("alpha", mu=0, sigma=10)
            betas = pm.Normal("betas", mu=0, sigma=10, dims="feature")
            sigma = pm.HalfNormal("sigma", sigma=1)

            mu = alpha + pt.dot(X_data, betas)

            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="obs")

            idata = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
            )

            idata.extend(pm.sample_posterior_predictive(idata))

        return idata
