"""
Price Target Achievement Model — Beta-Normal hybrid.

Models the probability of achieving analyst price targets using a Beta prior
on achievement probability and a Normal model for expected returns with
risk adjustment.

Reference: PriceTargetAchievementModel in probability_analytics.py (line 3057);
           _resolve_price_target_inputs() in inference_schema.py (line 746).
"""

from __future__ import annotations

import logging

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger(__name__)


class PriceTargetAchievement:
    """Bayesian price target achievement model.

    Parameters
    ----------
    prior_alpha : float
        Beta prior alpha for achievement probability.
    prior_beta : float
        Beta prior beta for achievement probability.
    risk_penalty : float
        Exponential risk penalty factor applied to expected return.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        risk_penalty: float = 0.1,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.risk_penalty = risk_penalty

    def fit(
        self,
        consensus_upside: np.ndarray,
        analyst_dispersion: np.ndarray,
        tickers: np.ndarray,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 1,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Fit price target achievement model and return InferenceData.

        Parameters
        ----------
        consensus_upside : array of float
            Consensus upside potential per stock.
        analyst_dispersion : array of float
            Analyst estimate dispersion (std) per stock.
        tickers : array of str
            Ticker symbols.
        samples, tune, chains, cores, target_accept, random_seed
            MCMC sampling parameters.

        Returns
        -------
        az.InferenceData
            Posterior contains ``achieve_prob``, ``expected_return``,
            ``risk_adj_return``.
        """
        consensus_upside = np.asarray(consensus_upside, dtype="float64")
        analyst_dispersion = np.asarray(analyst_dispersion, dtype="float64")

        coords = {"ticker": tickers}

        with pm.Model(coords=coords) as model:
            achieve_prob = pm.Beta(
                "achieve_prob",
                alpha=self.prior_alpha,
                beta=self.prior_beta,
                dims="ticker",
            )

            expected_return = pm.Normal(
                "expected_return",
                mu=consensus_upside,
                sigma=analyst_dispersion,
                dims="ticker",
            )

            risk_adj_return = pm.Deterministic(
                "risk_adj_return",
                expected_return * pt.exp(-self.risk_penalty * analyst_dispersion),
                dims="ticker",
            )

            sigma = pm.HalfNormal("sigma", sigma=0.1)

            pm.Normal(
                "upside_obs",
                mu=risk_adj_return,
                sigma=sigma,
                observed=consensus_upside,
                dims="ticker",
            )

            idata = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
            )

        return idata
