"""
Price Target Achievement Model — Beta-Normal hybrid.

Models the probability of achieving analyst price targets using a Beta prior
on achievement probability and a Normal model for expected returns with
risk adjustment.

Reference: PriceTargetAchievementModel in probability_models.py (line 3057);
           _resolve_price_target_inputs() in inference_schema.py (line 746).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

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

if TYPE_CHECKING:
    import arviz as az_typing  # noqa: F401
    import pymc as pm_typing  # noqa: F401

from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs

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
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[az_typing.InferenceData] = None

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
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[az_typing.InferenceData, pm_typing.Model]:
        """Fit price target achievement model and return ``(InferenceData, Model)``.

        Notes
        -----
        ``consensus_upside`` is observed *once* via ``upside_obs``; the
        latent ``expected_return`` has a prior centred at 0 with a
        dispersion-aware sigma to avoid the previous degenerate likelihood
        where the same series acted as both prior mean and observation.
        """
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install pymc + arviz to use PriceTargetAchievement."
            )

        consensus_upside = np.asarray(consensus_upside, dtype="float64")
        analyst_dispersion = np.asarray(analyst_dispersion, dtype="float64")
        if consensus_upside.size == 0:
            raise ValueError("PriceTargetAchievement.fit received empty consensus_upside.")
        if consensus_upside.shape != analyst_dispersion.shape:
            raise ValueError("consensus_upside and analyst_dispersion must share shape.")

        coords = {"ticker": np.asarray(tickers)}

        with pm.Model(coords=coords) as model:
            cu_data = pm.Data("consensus_upside", consensus_upside, dims="ticker")
            ad_data = pm.Data("analyst_dispersion", analyst_dispersion, dims="ticker")

            pm.Beta(
                "achieve_prob",
                alpha=self.prior_alpha,
                beta=self.prior_beta,
                dims="ticker",
            )

            expected_return = pm.Normal(
                "expected_return",
                mu=0.0,
                sigma=pt.maximum(ad_data, 1e-3),
                dims="ticker",
            )

            risk_adj_return = pm.Deterministic(
                "risk_adj_return",
                expected_return * pt.exp(-self.risk_penalty * ad_data),
                dims="ticker",
            )

            sigma = pm.HalfNormal("sigma", sigma=0.5)

            pm.Normal(
                "upside_obs",
                mu=risk_adj_return,
                sigma=sigma,
                observed=cu_data,
                dims="ticker",
            )

            scall: dict[str, Any] = dict(
                draws=samples,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True,
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
