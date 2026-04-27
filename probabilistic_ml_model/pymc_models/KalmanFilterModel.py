"""
Kalman Filter Price Target Model — GaussianRandomWalk state-space.

Uses PyMC GaussianRandomWalk for latent price state with an observation model
for noisy price targets.

Reference: run_kalman_filter() in expected_returns_v3.py (line 1519);
           probability_models.py Kalman references.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from arviz import InferenceData
from pymc.backends.base import MultiTrace

try:
    import arviz as az
except ImportError:
    try:
        import arviz_base as az
    except ImportError:
        az = None  # type: ignore[assignment]
import numpy as np
import pandas as pd

try:
    import pymc as pm
except ImportError:
    pm = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import arviz as az_typing  # noqa: F401
    import pymc as pm_typing  # noqa: F401

from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs

logger = logging.getLogger(__name__)


class KalmanFilterPriceTarget:
    """Bayesian state-space model for price target filtering.

    Uses GaussianRandomWalk as latent state with HalfNormal priors on
    state and observation noise (scaled to data std).
    """

    def __init__(self) -> None:
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[az_typing.InferenceData] = None

    def fit(
        self,
        price_targets: np.ndarray,
        isin: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[InferenceData | MultiTrace, Any]:
        """Fit Kalman-style state-space model and return ``(InferenceData, Model)``."""
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install pymc + arviz to use KalmanFilterPriceTarget."
            )

        pt_arr = np.asarray(price_targets, dtype="float64")
        if pt_arr.size < 2:
            raise ValueError("price_target must have length ≥ 2.")

        T = len(pt_arr)
        scale = float(np.nanstd(pt_arr)) or 1.0
        time_coords = dates if dates is not None else np.arange(T, dtype=np.int64)
        coords: dict[str, Any] = {"time": time_coords}
        if isin is not None:
            coords["isin"] = [isin]

        with pm.Model(coords=coords) as model:
            obs_data = pm.Data("price_target", pt_arr, dims="time")

            sigma_state = pm.HalfNormal("sigma_state", sigma=scale)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=scale)

            state = pm.GaussianRandomWalk(
                "state",
                sigma=sigma_state,
                init_dist=pm.Normal.dist(mu=float(pt_arr[0]), sigma=scale),
                dims="time",
            )

            pm.Normal(
                "obs",
                mu=state,
                sigma=sigma_obs,
                observed=obs_data,
                dims="time",
            )

            scall: dict[str, Any] = dict(
                draws=samples,
                tune=tune,
                chains=chains,
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

        # Store model metadata for downstream consumers
        if isin is not None:
            try:
                model.name = f"KalmanFilter[{isin}]"
            except Exception:
                pass

        self.model_ = model
        self.idata_ = idata if isinstance(idata, InferenceData) else None
        return idata, model
