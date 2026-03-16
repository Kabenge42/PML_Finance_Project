"""
Kalman Filter Price Target Model — GaussianRandomWalk state-space.

Uses PyMC GaussianRandomWalk for latent price state with an observation model
for noisy price targets.

Reference: run_kalman_filter() in expected_returns_v3.py (line 1519);
           probability_analytics.py Kalman references.
"""

from __future__ import annotations

import logging
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger(__name__)


class KalmanFilterPriceTarget:
    """Bayesian state-space model for price target filtering.

    Uses GaussianRandomWalk as latent state with HalfCauchy priors on
    state and observation noise.
    """

    def __init__(self) -> None:
        pass

    def fit(
        self,
        price_targets: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        ticker: str = "UNKNOWN",
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Fit Kalman-style state-space model and return InferenceData.

        Parameters
        ----------
        price_targets : array of float
            Observed (noisy) price targets over time.
        dates : DatetimeIndex, optional
            Time index for coordinates.
        ticker : str
            Ticker symbol for labelling.
        samples, tune, chains, target_accept, random_seed
            MCMC sampling parameters.

        Returns
        -------
        az.InferenceData
            Posterior contains filtered ``state``, ``sigma_state``, ``sigma_obs``.
        """
        price_targets = np.asarray(price_targets, dtype="float64")
        T = len(price_targets)

        if dates is None:
            time_coords = np.arange(T)
        else:
            time_coords = dates

        coords = {"time": time_coords}

        with pm.Model(coords=coords) as model:
            sigma_state = pm.HalfCauchy("sigma_state", beta=1.0)
            sigma_obs = pm.HalfCauchy("sigma_obs", beta=1.0)

            state = pm.GaussianRandomWalk(
                "state",
                sigma=sigma_state,
                init_dist=pm.Normal.dist(mu=price_targets[0], sigma=10.0),
                shape=T,
                dims="time",
            )

            pm.Normal(
                "obs",
                mu=state,
                sigma=sigma_obs,
                observed=price_targets,
                dims="time",
            )

            idata = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=False,
            )

        return idata
