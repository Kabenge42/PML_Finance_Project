"""
DCF Price Target Model — Discounted Cash Flow with Bayesian priors.

Places priors on FCF growth rate and WACC, computes projected FCFs and
terminal value as PyMC Deterministics, and fits against observed market prices.

Reference: compute_derived_price_target() in expected_returns_v3.py (line 2987).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

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

if TYPE_CHECKING:
    import arviz as az_typing  # noqa: F401
    import pymc as pm_typing  # noqa: F401

from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs

logger = logging.getLogger(__name__)


class DCFPriceTarget:
    """Bayesian Discounted Cash Flow intrinsic-value model.

    Priors
    ------
    fcf_growth ~ Normal(historical_growth, 0.05)
    wacc ~ TruncatedNormal(0.10, 0.02, lower=terminal_growth+0.005, upper=0.30)
    terminal_growth fixed at 0.02

    The truncated WACC prior guarantees ``wacc > terminal_growth`` so the
    Gordon-growth terminal value remains finite.
    """

    def __init__(self, terminal_growth: float = 0.02) -> None:
        self.terminal_growth = terminal_growth
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[az_typing.InferenceData] = None

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
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[az_typing.InferenceData, pm_typing.Model]:
        """Fit DCF model and return ``(InferenceData, Model)``."""
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install a compatible version of pymc and arviz "
                "(arviz<1.0 or patch pymc for arviz-base) to use DCFPriceTarget."
            )

        hf = np.asarray(historical_fcf, dtype="float64")
        mp = np.asarray(market_prices, dtype="float64")
        if hf.size < 2 or mp.size == 0:
            raise ValueError("Need ≥2 historical_fcf and ≥1 market_prices.")

        growth_rates = np.diff(hf) / np.abs(hf[:-1] + 1e-10)
        hist_growth = float(np.mean(growth_rates))
        last_fcf = float(hf[-1])

        t = np.arange(1, n_projection_years + 1, dtype="float64")

        with pm.Model() as model:
            price_data = pm.Data("market_prices", mp)

            fcf_growth = pm.Normal("fcf_growth", mu=hist_growth, sigma=0.05)
            # WACC bounded strictly above terminal_growth → finite terminal value.
            wacc = pm.TruncatedNormal(
                "wacc",
                mu=0.10,
                sigma=0.02,
                lower=self.terminal_growth + 0.005,
                upper=0.30,
            )

            fcf_projected = last_fcf * (1 + fcf_growth) ** t
            discount_factors = (1 + wacc) ** t
            pv_fcfs = pt.sum(fcf_projected / discount_factors)

            fcf_terminal = last_fcf * (1 + fcf_growth) ** (n_projection_years + 1)
            terminal_value = fcf_terminal / (wacc - self.terminal_growth)
            terminal_pv = terminal_value / (1 + wacc) ** n_projection_years

            intrinsic_value = pm.Deterministic("intrinsic_value", pv_fcfs + terminal_pv)

            sigma = pm.HalfNormal("sigma", sigma=500.0)

            pm.Normal(
                "price_obs",
                mu=intrinsic_value,
                sigma=sigma,
                observed=price_data,
            )

            scall: dict[str, Any] = dict(
                draws=samples,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=False,
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
