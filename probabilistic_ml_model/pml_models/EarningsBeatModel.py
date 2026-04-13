"""
Earnings Beat Bayesian Model — Hierarchical Beta-Binomial.

Replaces scipy Beta with PyMC Beta-Binomial, supports optional sector-level
hierarchical structure and forward-adjustment deterministics.

Reference: EarningsBeatProbabilityModel in probability_models.py (line 1131);
           build_beat_probability_inference_data() in inference_schema.py (line 459).
"""

from __future__ import annotations

import logging
from typing import Optional

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
except ImportError:
    pm = None  # type: ignore[assignment]

from probabilistic_ml_model.pml_models._pytensor_compat import get_pytensor_compile_kwargs

logger = logging.getLogger(__name__)


class EarningsBeatBayesian:
    """Hierarchical Beta-Binomial earnings beat probability model.

    Parameters
    ----------
    prior_alpha : float
        Default Beta prior alpha for beat probability.
    prior_beta : float
        Default Beta prior beta for beat probability.
    """

    def __init__(
        self,
        prior_alpha: float = 1.5,
        prior_beta: float = 2.0,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def fit(
        self,
        n_beats: np.ndarray,
        n_total: np.ndarray,
        tickers: np.ndarray,
        sectors: Optional[np.ndarray] = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Fit the Beta-Binomial model and return ArviZ InferenceData.

        Parameters
        ----------
        n_beats : array of int
            Number of earnings beats per stock.
        n_total : array of int
            Total number of earnings reports per stock.
        tickers : array of str
            Ticker symbols.
        sectors : array of str, optional
            Sector labels — enables hierarchical sector-rate structure.
        samples, tune, chains, target_accept, random_seed
            MCMC sampling parameters.

        Returns
        -------
        az.InferenceData
        """
        n_beats = np.asarray(n_beats, dtype="int32")
        n_total = np.asarray(n_total, dtype="int32")

        coords = {"ticker": tickers}
        hierarchical = sectors is not None
        sector_idx = None

        if hierarchical:
            unique_sectors = np.unique(sectors)
            sector_idx = np.array([np.where(unique_sectors == s)[0][0] for s in sectors])
            coords["sector"] = unique_sectors

        with pm.Model(coords=coords) as model:
            if hierarchical:
                sector_rate = pm.Beta(
                    "sector_rate",
                    alpha=self.prior_alpha,
                    beta=self.prior_beta,
                    dims="sector",
                )
                kappa = pm.HalfNormal("kappa", sigma=10.0)
                beat_prob = pm.Beta(
                    "beat_prob",
                    alpha=sector_rate[sector_idx] * kappa,
                    beta=(1 - sector_rate[sector_idx]) * kappa,
                    dims="ticker",
                )
            else:
                beat_prob = pm.Beta(
                    "beat_prob",
                    alpha=self.prior_alpha,
                    beta=self.prior_beta,
                    dims="ticker",
                )

            pm.Binomial(
                "beats_obs",
                n=n_total,
                p=beat_prob,
                observed=n_beats,
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
