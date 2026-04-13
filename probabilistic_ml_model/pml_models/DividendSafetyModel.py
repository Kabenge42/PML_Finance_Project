"""
Dividend Safety Bayesian Model — Cut probability with FCF coverage.

Models dividend cut probability with Beta priors and conditional risk
adjustments based on payout ratio thresholds.

Reference: DividendCutProbabilityModel in probability_models.py (line 2793).
"""

from __future__ import annotations

import logging

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

from probabilistic_ml_model.pml_models._pytensor_compat import get_pytensor_compile_kwargs

logger = logging.getLogger(__name__)


class DividendSafetyBayesian:
    """Bayesian dividend cut probability model.

    Parameters
    ----------
    prior_alpha : float
        Beta prior alpha for cut probability.
    prior_beta : float
        Beta prior beta for cut probability.
    high_payout_threshold : float
        Payout ratio above which risk adjustment is applied.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 5.0,
        high_payout_threshold: float = 0.9,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.high_payout_threshold = high_payout_threshold

    def fit(
        self,
        payout_ratios: np.ndarray,
        fcf_coverage: np.ndarray,
        tickers: np.ndarray,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Fit dividend safety model and return InferenceData.

        Parameters
        ----------
        payout_ratios : array of float
            Dividend payout ratios per stock.
        fcf_coverage : array of float
            FCF-to-dividend coverage ratios per stock.
        tickers : array of str
            Ticker symbols.
        samples, tune, chains, target_accept, random_seed
            MCMC sampling parameters.

        Returns
        -------
        az.InferenceData
            Posterior contains ``cut_prob``, ``risk_adj``.
        """
        payout_ratios = np.asarray(payout_ratios, dtype="float64")
        fcf_coverage = np.asarray(fcf_coverage, dtype="float64")

        coords = {"ticker": tickers}

        with pm.Model(coords=coords):
            payout_data = pm.Data("payout_data", payout_ratios, dims="ticker")

            cut_prob = pm.Beta(
                "cut_prob",
                alpha=self.prior_alpha,
                beta=self.prior_beta,
                dims="ticker",
            )

            risk_adj = pm.Deterministic(
                "risk_adj",
                pt.clip(
                    pt.switch(
                        pt.gt(payout_data, self.high_payout_threshold),
                        cut_prob * 1.3,
                        cut_prob,
                    ),
                    0.0,
                    1.0,
                ),
                dims="ticker",
            )

            expected_coverage = pm.Deterministic(
                "expected_coverage",
                pt.clip(1.0 / (risk_adj + 0.01), 0.0, 20.0),
                dims="ticker",
            )

            sigma = pm.HalfNormal("sigma", sigma=1.0)

            pm.Normal(
                "fcf_coverage_obs",
                mu=expected_coverage,
                sigma=sigma,
                observed=fcf_coverage,
                dims="ticker",
            )

            idata = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=False,
                compile_kwargs=get_pytensor_compile_kwargs(),
            )

        return idata
