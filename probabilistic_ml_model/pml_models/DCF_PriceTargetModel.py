"""
DCF Price Target Model — Discounted Cash Flow with Bayesian priors.

Places priors on FCF growth rate and WACC, computes projected FCFs and
terminal value as PyMC Deterministics, and fits against observed market prices.

Reference: compute_derived_price_target() in expected_returns_v3.py (line 2987).
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import arviz as az
except ImportError:
    try:
        import arviz_base as az
    except ImportError:
        az = None  # type: ignore[assignment]

try:
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:
    pm = None  # type: ignore[assignment]
    pt = None  # type: ignore[assignment]

from probabilistic_ml_model.pml_models._pytensor_compat import get_pytensor_compile_kwargs

logger = logging.getLogger(__name__)


class DCFPriceTarget:
    """Bayesian Discounted Cash Flow intrinsic-value model.

    Priors
    ------
    fcf_growth ~ Normal(historical_growth, 0.05)
    wacc ~ Normal(0.10, 0.02)
    terminal_growth fixed at 0.02
    """

    def __init__(self, terminal_growth: float = 0.02) -> None:
        self.terminal_growth = terminal_growth

    def fit(
        self,
        historical_fcf: np.ndarray,
        market_prices: np.ndarray,
        n_projection_years: int = 5,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 1,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ):
        """Fit DCF model and return InferenceData.

        Parameters
        ----------
        historical_fcf : array of float
            Historical free cash flow series (oldest → newest).
        market_prices : array of float
            Observed market prices to fit against.
        n_projection_years : int
            Number of years to project FCFs forward.
        samples, tune, chains, cores, target_accept, random_seed
            MCMC sampling parameters.  ``cores=1`` avoids multiprocessing
            overhead on Windows where each spawned worker must re-import
            PyTensor in Python-only mode (no C compiler).

        Returns
        -------
        az.InferenceData
            Posterior contains ``wacc``, ``fcf_growth``, ``intrinsic_value``.
        """
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install a compatible version of pymc and arviz "
                "(arviz<1.0 or patch pymc for arviz-base) to use DCFPriceTarget."
            )

        historical_fcf = np.asarray(historical_fcf, dtype="float64")
        market_prices = np.asarray(market_prices, dtype="float64")

        growth_rates = np.diff(historical_fcf) / np.abs(historical_fcf[:-1] + 1e-10)
        hist_growth = float(np.mean(growth_rates))
        last_fcf = float(historical_fcf[-1])

        t = np.arange(1, n_projection_years + 1, dtype="float64")

        with pm.Model():
            fcf_growth = pm.Normal("fcf_growth", mu=hist_growth, sigma=0.05)
            wacc = pm.Normal("wacc", mu=0.10, sigma=0.02)

            fcf_projected = last_fcf * (1 + fcf_growth) ** t
            discount_factors = (1 + wacc) ** t
            pv_fcfs = pt.sum(fcf_projected / discount_factors)

            fcf_terminal = last_fcf * (1 + fcf_growth) ** (n_projection_years + 1)
            terminal_value = fcf_terminal / (wacc - self.terminal_growth + 1e-6)
            terminal_pv = terminal_value / (1 + wacc) ** n_projection_years

            intrinsic_value = pm.Deterministic("intrinsic_value", pv_fcfs + terminal_pv)

            sigma = pm.HalfNormal("sigma", sigma=500.0)

            pm.Normal(
                "price_obs",
                mu=intrinsic_value,
                sigma=sigma,
                observed=market_prices,
            )

            idata = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=False,
                compile_kwargs=get_pytensor_compile_kwargs(),
            )

        return idata
