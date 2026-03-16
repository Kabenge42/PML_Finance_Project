"""
Monte Carlo Return Simulation — Bayesian posterior predictive sampling.

Models return distributions with learnable mean/variance priors and generates
forward simulations via posterior predictive checks.

Reference: run_monte_carlo_analysis() in expected_returns_v3.py (line 1361);
           build_monte_carlo_inference_data() in inference_schema.py (line 786).
"""

from __future__ import annotations

import logging

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger(__name__)


class MonteCarloReturnSimulation:
    """Bayesian Monte Carlo return simulation model.

    Uses learnable priors on return mean and volatility, then generates
    forward simulations via posterior predictive sampling.
    """

    def __init__(self) -> None:
        pass

    def fit(
        self,
        historical_means: np.ndarray,
        historical_stds: np.ndarray,
        tickers: np.ndarray,
        n_sims: int = 10_000,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        """Fit return distribution model and generate simulations.

        Parameters
        ----------
        historical_means : array of float
            Historical mean returns per stock.
        historical_stds : array of float
            Historical return standard deviations per stock.
        tickers : array of str
            Ticker symbols.
        n_sims : int
            Number of forward simulations per stock.
        samples, tune, chains, target_accept, random_seed
            MCMC sampling parameters.

        Returns
        -------
        az.InferenceData
            Contains posterior (mu_return, sigma_return, prob_positive) and
            posterior_predictive (sim_returns).
        """
        n_stocks = len(tickers)
        historical_means = np.asarray(historical_means, dtype="float64")
        historical_stds = np.asarray(historical_stds, dtype="float64")

        coords = {
            "ticker": tickers,
            "simulation": np.arange(n_sims),
        }

        with pm.Model(coords=coords) as model:
            mu_return = pm.Normal(
                "mu_return",
                mu=historical_means,
                sigma=0.05,
                dims="ticker",
            )
            sigma_return = pm.HalfNormal(
                "sigma_return",
                sigma=historical_stds,
                dims="ticker",
            )

            sim_returns = pm.Normal(
                "sim_returns",
                mu=mu_return[:, None],
                sigma=sigma_return[:, None],
                dims=("ticker", "simulation"),
            )

            prob_positive = pm.Deterministic(
                "prob_positive",
                pt.mean(pt.gt(sim_returns, 0.0).astype("float64"), axis=1),
                dims="ticker",
            )

            idata = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
            )

            idata.extend(
                pm.sample_posterior_predictive(
                    idata,
                    var_names=["sim_returns"],
                )
            )

        return idata
