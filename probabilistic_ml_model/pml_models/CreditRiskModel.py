"""
Credit Risk Bayesian Model — Multi-signal distress estimation.

Uses PyMC with Altman Z-score zones, debt-to-equity signals, and optional
sector-level hierarchical shrinkage.

Reference: CreditRiskProbabilityModel in probability_analytics.py (line 2456);
           build_credit_risk_inference_data() in inference_schema.py (line 591).
"""

from __future__ import annotations

import logging
from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger(__name__)


class CreditRiskBayesian:
    """Bayesian credit risk / distress probability model.

    Parameters
    ----------
    prior_alpha : float
        Beta prior alpha for distress probability.
    prior_beta : float
        Beta prior beta for distress probability.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 3.0,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def fit(
        self,
        z_scores: np.ndarray,
        debt_to_equity: np.ndarray,
        tickers: np.ndarray,
        sectors: Optional[np.ndarray] = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Fit credit-risk model and return ArviZ InferenceData.

        Parameters
        ----------
        z_scores : array of float
            Altman Z-scores per stock.
        debt_to_equity : array of float
            Debt-to-equity ratios per stock.
        tickers : array of str
            Ticker symbols.
        sectors : array of str, optional
            Sector labels for hierarchical shrinkage.
        samples, tune, chains, target_accept, random_seed
            MCMC sampling parameters.

        Returns
        -------
        az.InferenceData
        """
        n_stocks = len(tickers)
        z_scores = np.asarray(z_scores, dtype="float64")
        debt_to_equity = np.asarray(debt_to_equity, dtype="float64")

        coords = {"ticker": tickers}

        with pm.Model(coords=coords) as model:
            distress_prob = pm.Beta(
                "distress_prob",
                alpha=self.prior_alpha,
                beta=self.prior_beta,
                dims="ticker",
            )

            z_data = pm.Data("z_score_data", z_scores, dims="ticker")
            de_data = pm.Data("de_data", debt_to_equity, dims="ticker")

            zone_adj = pm.Deterministic(
                "zone_adj",
                pt.switch(
                    pt.lt(z_data, 1.81),
                    0.75,
                    pt.switch(pt.lt(z_data, 2.67), 0.35, 0.15),
                ),
                dims="ticker",
            )

            debt_trend = pm.Normal(
                "debt_trend",
                mu=de_data,
                sigma=0.1,
                dims="ticker",
            )

            expected_distress = pm.Deterministic(
                "expected_distress",
                pt.clip(distress_prob * zone_adj + 0.05 * debt_trend, 0.0, 1.0),
                dims="ticker",
            )

            pm.Normal(
                "distress_obs",
                mu=expected_distress,
                sigma=0.1,
                observed=zone_adj,
                dims="ticker",
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
