"""
Probabilistic Linear Regression Model — Standard Bayesian regression.

Implements a Bayesian linear regression with Normal priors on intercept and
coefficients, HalfNormal on noise, and posterior predictive checks.

Performance optimizations (v4.1)
--------------------------------
- **Z-score standardization** of X and y before sampling so that default
  priors are well-calibrated and the NUTS sampler explores efficiently.
- **Observation sub-sampling** (configurable ``max_obs``) prevents the
  sampler from running on thousands of rows in Python-only PyTensor mode.
- **Tighter default priors** (sigma ≈ N(0, 2) for betas on standardized
  data) avoid the wide, uninformative geometry that causes divergences.
- **max_treedepth = 8** caps individual leapfrog trajectories, preventing
  quasi-infinite loops on pathological posteriors.
- **Optional posterior-predictive** sampling — skipped by default in
  pipeline mode to halve wall-clock time.

Reference: PyMC_overview.md linear regression example;
           inference_schema.py _build_posterior_samples_normal() (line 289).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

# Ensure C backend is disabled before any pytensor import
# (guards against cases where PYTENSOR_FLAGS wasn't set early enough)
if not os.environ.get("PYTENSOR_FLAGS"):
    os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64,cxx="
elif "cxx=" not in os.environ["PYTENSOR_FLAGS"]:
    os.environ["PYTENSOR_FLAGS"] += ",cxx="

from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults tuned for the expected-returns pipeline running PyTensor in
# Python-only mode (no C compiler) on ~6 000 stocks.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_OBS = 500
_DEFAULT_SAMPLES = 500
_DEFAULT_TUNE = 500
_DEFAULT_CHAINS = 2
_DEFAULT_CORES = 1
_DEFAULT_TARGET_ACCEPT = 0.9
_DEFAULT_MAX_TREEDEPTH = 8


class ProbabilisticLinearRegression:
    """Bayesian linear regression model using PyMC.

    Priors (on standardized data)
    -----------------------------
    alpha ~ Normal(0, 1)
    betas ~ Normal(0, 2) per feature
    sigma ~ HalfNormal(2)

    The feature matrix and target are z-score standardized internally
    before sampling.  Posterior summaries are reported on the standardized
    scale; callers can back-transform if needed using ``self.X_mean_``,
    ``self.X_std_``, ``self.y_mean_``, and ``self.y_std_``.
    """

    def __init__(self) -> None:
        self.X_mean_: Optional[np.ndarray] = None
        self.X_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        obs_names: Optional[np.ndarray] = None,
        samples: int = _DEFAULT_SAMPLES,
        tune: int = _DEFAULT_TUNE,
        chains: int = _DEFAULT_CHAINS,
        cores: int = _DEFAULT_CORES,
        target_accept: float = _DEFAULT_TARGET_ACCEPT,
        random_seed: int = 42,
        max_obs: int = _DEFAULT_MAX_OBS,
        max_treedepth: int = _DEFAULT_MAX_TREEDEPTH,
        posterior_predictive: bool = False,
        progressbar: bool = True,
    ):
        """Fit Bayesian linear regression and return InferenceData.

        Parameters
        ----------
        X : array (n_obs, n_features)
            Feature matrix.
        y : array (n_obs,)
            Target vector.
        feature_names : list of str, optional
            Names for each feature (column of X).
        obs_names : array of str, optional
            Observation identifiers (e.g., tickers).
        samples, tune, chains, cores, target_accept, random_seed
            MCMC sampling parameters.  ``cores=1`` avoids multiprocessing
            overhead on Windows where each spawned worker must re-import
            PyTensor in Python-only mode (no C compiler).
        max_obs : int, default 500
            Maximum observations to sample on.  If the dataset is larger
            a deterministic random sub-sample is drawn so that MCMC
            completes in bounded time.
        max_treedepth : int, default 8
            NUTS ``max_treedepth`` — limits leapfrog steps per transition
            to 2^max_treedepth, preventing quasi-infinite loops.
        posterior_predictive : bool, default False
            Whether to run ``sample_posterior_predictive`` after MCMC.
            Disabled by default to save time in pipeline mode.
        progressbar : bool, default True
            Show PyMC progress bars during sampling.

        Returns
        -------
        az.InferenceData
        """
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install a compatible version of pymc and arviz "
                "(arviz<1.0 or patch pymc for arviz-base) to use ProbabilisticLinearRegression."
            )

        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64")

        # ── Sub-sample observations if dataset is too large ───────────
        n_obs, n_features = X.shape
        if n_obs > max_obs:
            logger.info(
                "PLR: sub-sampling %d → %d observations (random_seed=%d)",
                n_obs,
                max_obs,
                random_seed,
            )
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(n_obs, size=max_obs, replace=False)
            idx.sort()
            X = X[idx]
            y = y[idx]
            if obs_names is not None:
                obs_names = np.asarray(obs_names)[idx]
            n_obs = max_obs

        # ── Z-score standardisation ───────────────────────────────────
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0)
        self.X_std_[self.X_std_ < 1e-12] = 1.0  # avoid division by zero
        X_scaled = (X - self.X_mean_) / self.X_std_

        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std())
        if self.y_std_ < 1e-12:
            self.y_std_ = 1.0
        y_scaled = (y - self.y_mean_) / self.y_std_

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        if obs_names is None:
            obs_names = np.arange(n_obs)

        coords = {
            "feature": list(feature_names),
            "obs": obs_names,
        }

        logger.info(
            "PLR: fitting on %d obs × %d features "
            "(samples=%d, tune=%d, chains=%d, max_treedepth=%d)",
            n_obs,
            n_features,
            samples,
            tune,
            chains,
            max_treedepth,
        )

        # ── Build and sample the model ────────────────────────────────
        with pm.Model(coords=coords):
            X_data = pm.Data("X", X_scaled, dims=("obs", "feature"))

            alpha = pm.Normal("alpha", mu=0, sigma=1)
            betas = pm.Normal("betas", mu=0, sigma=2, dims="feature")
            sigma = pm.HalfNormal("sigma", sigma=2)

            mu = pt.add(alpha, pt.dot(X_data, betas))

            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled, dims="obs")

            nuts_kwargs: dict = {"max_treedepth": max_treedepth}

            _compile_kwargs = get_pytensor_compile_kwargs()

            idata = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                nuts_sampler="pymc",
                idata_kwargs={"posterior_predictive": {}},
                compile_kwargs=_compile_kwargs,
                **nuts_kwargs,
            )

            if posterior_predictive:
                idata.extend(
                    pm.sample_posterior_predictive(
                        idata, compile_kwargs=_compile_kwargs
                    )
                )

        logger.info("PLR: sampling complete — %d draws × %d chains", samples, chains)
        return idata
