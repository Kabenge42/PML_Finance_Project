"""
Monte Carlo Return Simulation — Bayesian posterior predictive sampling.

Models return distributions with learnable mean/variance priors and generates
forward simulations via posterior predictive checks.

Performance note: The MCMC model only samples ``mu_return`` and ``sigma_return``
(one parameter per ticker each).  Forward simulations (``sim_returns``) are
generated *post-hoc* via NumPy draws conditioned on the learned posteriors,
keeping the MCMC parameter space small and the sampler fast.
``prob_positive`` is computed analytically from the Normal CDF rather than
from Monte Carlo counts inside the sampler graph.

Schema alignment
----------------
The model exposes its inputs as ``pm.Data`` containers (``historical_means``,
``historical_stds``) so they appear in ``idata.constant_data`` and can be
swapped via ``pm.set_data({...})`` for out-of-sample prediction — matching
the pattern used by the other PyMC models in this package
(``EarningsBeatBayesian``, ``CreditRiskBayesian``, ``DividendSafetyBayesian``,
etc.).

Reference: monte_carlo_price_target_simulation() in statistical_models.py;
           run_monte_carlo_analysis() in expected_returns_v3.py (line 1361);
           build_monte_carlo_inference_data() in inference_schema.py (line 926).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, TYPE_CHECKING

try:
    import arviz as az
except ImportError:
    try:
        import arviz_base as az
    except ImportError:
        az = None  # type: ignore[assignment]

import numpy as np
from scipy.special import erf as _scipy_erf

try:
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:
    pm = None  # type: ignore[assignment]
    pt = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import arviz as az_typing  # noqa: F401
    import pymc as pm_typing  # noqa: F401

from probabilistic_ml_model._pymc_arviz_compat import InferenceLike
from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-aware defaults
# ---------------------------------------------------------------------------
_N_JOBS = int(os.environ.get("N_JOBS", "1"))
_CORES_DEFAULT = max(_N_JOBS, 1) if _N_JOBS > 0 else 1
_RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

# Minimum standard deviation floor to prevent degenerate HalfNormal priors.
# A value of 0.01 (1%) ensures the sampler always has a feasible region and
# avoids numerical issues with near-zero sigma in the Normal CDF computation.
_MIN_SIGMA_FLOOR = 0.01

# Weakly informative prior scale for mu and sigma.  Values are chosen so that
# stock returns in the range [-1, +1] (i.e. -100% to +100% as decimals) sit
# comfortably within the prior bulk, while still providing regularisation.
_MU_PRIOR_SIGMA = 0.5
_SIGMA_PRIOR_SIGMA = 0.5


def _sanitize_inputs(
    historical_means: np.ndarray,
    historical_stds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, NaN-fill, and clip inputs in place-safe fashion."""
    historical_means = np.asarray(historical_means, dtype="float64").copy()
    historical_stds = np.asarray(historical_stds, dtype="float64").copy()

    nan_mask_mu = ~np.isfinite(historical_means)
    if nan_mask_mu.any():
        logger.warning(
            "Replacing %d non-finite historical_means with 0.0", nan_mask_mu.sum()
        )
        historical_means[nan_mask_mu] = 0.0

    nan_mask_std = ~np.isfinite(historical_stds)
    if nan_mask_std.any():
        logger.warning(
            "Replacing %d non-finite historical_stds with %.4f",
            nan_mask_std.sum(),
            _MIN_SIGMA_FLOOR,
        )
        historical_stds[nan_mask_std] = _MIN_SIGMA_FLOOR

    # Clamp stds to a minimum floor so HalfNormal sigma is never zero/tiny.
    historical_stds = np.clip(historical_stds, _MIN_SIGMA_FLOOR, None)

    if len(historical_means) == 0:
        raise ValueError("historical_means is empty — nothing to sample.")
    if historical_means.shape != historical_stds.shape:
        raise ValueError("historical_means and historical_stds must share shape.")

    return historical_means, historical_stds


def _build_mc_model(
    historical_means: np.ndarray,
    historical_stds: np.ndarray,
    tickers: np.ndarray,
) -> "pm_typing.Model":
    """Build the Bayesian return-distribution model with `pm.Data` containers."""
    coords = {"ticker": np.asarray(tickers)}
    with pm.Model(coords=coords) as model:
        # pm.Data containers — exposed in idata.constant_data, swappable via
        # pm.set_data({"historical_means": ..., "historical_stds": ...}).
        means_data = pm.Data("historical_means", historical_means, dims="ticker")
        # `historical_stds` is currently informational (used post-hoc for the
        # NumPy posterior-predictive draws and the analytical fallback) but is
        # registered as a pm.Data container so it round-trips through
        # `idata.constant_data` and can be swapped via `pm.set_data`.
        pm.Data("historical_stds", historical_stds, dims="ticker")

        # Weakly informative priors
        mu_return = pm.Normal(
            "mu_return",
            mu=0.0,
            sigma=_MU_PRIOR_SIGMA,
            dims="ticker",
        )
        sigma_return = pm.HalfNormal(
            "sigma_return",
            sigma=_SIGMA_PRIOR_SIGMA,
            dims="ticker",
        )

        # Observed likelihood — the historical means serve as a single
        # observation per ticker drawn from Normal(mu, sigma).
        pm.Normal(
            "obs",
            mu=mu_return,
            sigma=sigma_return,
            observed=means_data,
            dims="ticker",
        )

        # Analytical P(return > 0) via Normal CDF — no simulation needed.
        # P(X > 0) = 1 - Φ(-μ/σ) = Φ(μ/σ)
        pm.Deterministic(
            "prob_positive",
            0.5 * (1.0 + pt.erf(mu_return / (sigma_return * pt.sqrt(2.0)))),
            dims="ticker",
        )
    return model


def fit(
    historical_means: np.ndarray,
    historical_stds: np.ndarray,
    tickers: np.ndarray,
    n_sims: int = 1_000,
    samples: int = 500,
    tune: int = 500,
    chains: int = 2,
    cores: int = _CORES_DEFAULT,
    target_accept: float = 0.90,
    random_seed: int = _RANDOM_SEED,
    nuts_sampler: Optional[str] = None,
    return_model: bool = False,
    **sample_kwargs: Any,
):
    """Fit return-distribution model and generate simulations.

    Parameters
    ----------
    historical_means : array of float
        Historical mean returns per stock (as decimals, e.g. 0.05 = 5%).
    historical_stds : array of float
        Historical return standard deviations per stock (as decimals).
    tickers : array of str
        Ticker / ISIN symbols.
    n_sims : int
        Number of forward simulations per stock (generated post-hoc via
        NumPy, **not** during MCMC).
    samples, tune, chains, cores, target_accept, random_seed
        MCMC sampling parameters.  ``cores`` defaults to the ``N_JOBS``
        environment variable (clamped ≥ 1).
    nuts_sampler : str, optional
        Alternate NUTS implementation (e.g. ``"nutpie"``).  Forwarded to
        ``pm.sample``.
    return_model : bool
        If True, return ``(InferenceData, pm.Model)`` to mirror the
        class-based API used by the other PyMC models in this package.
    **sample_kwargs
        Extra keyword arguments forwarded to ``pm.sample`` (e.g.
        ``idata_kwargs={"log_likelihood": False}``).

    Returns
    -------
    az.InferenceData
        Contains posterior (``mu_return``, ``sigma_return``,
        ``prob_positive``), constant_data (``historical_means``,
        ``historical_stds``) and posterior_predictive (``sim_returns``).
        If ``return_model=True``, returns a ``(idata, model)`` tuple.
    """
    if pm is None:
        raise ImportError(
            "PyMC is not available. Install pymc + arviz to use MonteCarloReturnSimulation."
        )

    historical_means, historical_stds = _sanitize_inputs(historical_means, historical_stds)

    logger.info(
        "MC fit: %d tickers, means=[%.4f, %.4f], stds=[%.4f, %.4f]",
        len(tickers),
        historical_means.min(),
        historical_means.max(),
        historical_stds.min(),
        historical_stds.max(),
    )

    # ------------------------------------------------------------------
    # Phase 1 — MCMC: infer mu_return & sigma_return
    # ------------------------------------------------------------------
    _compile_kwargs = get_pytensor_compile_kwargs()

    def _run_sample(n_draws: int, n_tune: int, n_chains: int, accept: float):
        """Build model and run sampler with the given settings."""
        model = _build_mc_model(historical_means, historical_stds, tickers)
        with model:
            scall: dict[str, Any] = dict(
                draws=n_draws,
                tune=n_tune,
                chains=n_chains,
                cores=cores,
                target_accept=accept,
                random_seed=random_seed,
                init="adapt_diag",
                progressbar=True,
                compile_kwargs=_compile_kwargs,
            )
            if nuts_sampler is not None:
                scall["nuts_sampler"] = nuts_sampler
            scall.setdefault("idata_kwargs", {"log_likelihood": False})
            scall.update(sample_kwargs)

            idata = pm.sample(**scall)
        return idata, model

    # Try with requested settings; on failure, retry with conservative settings.
    try:
        idata, model = _run_sample(samples, tune, chains, target_accept)
    except Exception as exc:
        logger.warning(
            "Primary MCMC sampling failed (%s: %s). Retrying with conservative "
            "settings (chains=1, draws=500, tune=500, target_accept=0.99).",
            type(exc).__name__,
            exc,
        )
        try:
            idata, model = _run_sample(
                n_draws=500, n_tune=500, n_chains=1, accept=0.99
            )
        except Exception as exc2:
            logger.error(
                "Conservative MCMC sampling also failed (%s: %s). "
                "Returning analytical approximation.",
                type(exc2).__name__,
                exc2,
            )
            idata = _analytical_fallback(
                historical_means, historical_stds, tickers, n_sims, random_seed
            )
            return (idata, None) if return_model else idata

    # ------------------------------------------------------------------
    # Phase 2 — Generate forward simulations from the posterior
    # Draw sim_returns directly from NumPy using posterior mu/sigma samples.
    # ------------------------------------------------------------------
    idata = _generate_posterior_predictive(idata, tickers, n_sims, random_seed)

    return (idata, model) if return_model else idata


def _generate_posterior_predictive(
    idata: "InferenceLike",
    tickers: np.ndarray,
    n_sims: int,
    random_seed: int,
) -> "InferenceLike":
    """Attach posterior-predictive sim_returns to *idata* using NumPy.

    For each posterior draw of (mu_return, sigma_return), generate
    ``n_sims`` Normal draws.  This is equivalent to
    ``pm.sample_posterior_predictive`` but avoids a second PyMC model.
    """
    import xarray as xr

    rng = np.random.default_rng(random_seed)

    mu_post = idata.posterior["mu_return"].values  # (chain, draw, ticker)
    sigma_post = idata.posterior["sigma_return"].values

    n_chains, n_draws, n_tickers = mu_post.shape

    # (chain, draw, ticker, simulation)
    sim_returns = rng.normal(
        loc=mu_post[..., np.newaxis],
        scale=np.clip(sigma_post[..., np.newaxis], _MIN_SIGMA_FLOOR, None),
        size=(n_chains, n_draws, n_tickers, n_sims),
    )

    pp_dataset = xr.Dataset(
        {
            "sim_returns": (
                ["chain", "draw", "ticker", "simulation"],
                sim_returns,
            ),
        },
        coords={
            "chain": idata.posterior.coords["chain"].values,
            "draw": idata.posterior.coords["draw"].values,
            "ticker": list(tickers),
            "simulation": np.arange(n_sims),
        },
    )

    if az is not None:
        # ArviZ 1.0 InferenceData / xarray.DataTree both support add_groups
        idata.add_groups({"posterior_predictive": pp_dataset})
    else:
        idata.posterior_predictive = pp_dataset

    return idata


def _analytical_fallback(
    historical_means: np.ndarray,
    historical_stds: np.ndarray,
    tickers: np.ndarray,
    n_sims: int,
    random_seed: int,
) -> "InferenceLike":
    """Build an InferenceData / DataTree object analytically when MCMC fails entirely.

    Uses the historical means/stds directly as point estimates, generating
    synthetic posterior draws from a narrow Normal centred on each estimate.
    This ensures downstream code always receives a valid InferenceData,
    including a ``constant_data`` group that mirrors the schema-aligned
    MCMC path.
    """
    import xarray as xr

    rng = np.random.default_rng(random_seed)
    n_tickers = len(tickers)
    n_draws = 500

    # Synthetic posterior (narrow spread around point estimates)
    mu_samples = rng.normal(
        loc=historical_means[None, :],
        scale=0.005,
        size=(1, n_draws, n_tickers),
    )  # (chain, draw, ticker)

    sigma_samples = np.abs(
        rng.normal(
            loc=historical_stds[None, :],
            scale=0.002,
            size=(1, n_draws, n_tickers),
        )
    )
    sigma_samples = np.clip(sigma_samples, _MIN_SIGMA_FLOOR, None)

    # Analytical prob_positive using scipy.special.erf (vectorized)
    prob_positive = 0.5 * (
        1.0 + _scipy_erf(mu_samples / (sigma_samples * np.sqrt(2.0)))
    )

    posterior = xr.Dataset(
        {
            "mu_return": (["chain", "draw", "ticker"], mu_samples),
            "sigma_return": (["chain", "draw", "ticker"], sigma_samples),
            "prob_positive": (["chain", "draw", "ticker"], prob_positive),
        },
        coords={
            "chain": [0],
            "draw": np.arange(n_draws),
            "ticker": list(tickers),
        },
    )

    # Schema-aligned constant_data group mirrors the pm.Data containers.
    constant_data = xr.Dataset(
        {
            "historical_means": (["ticker"], historical_means),
            "historical_stds": (["ticker"], historical_stds),
        },
        coords={"ticker": list(tickers)},
    )

    # Simulated returns
    sim_returns = rng.normal(
        loc=historical_means[None, None, :, None],
        scale=np.clip(historical_stds, _MIN_SIGMA_FLOOR, None)[None, None, :, None],
        size=(1, n_draws, n_tickers, n_sims),
    )

    posterior_predictive = xr.Dataset(
        {
            "sim_returns": (
                ["chain", "draw", "ticker", "simulation"],
                sim_returns,
            ),
        },
        coords={
            "chain": [0],
            "draw": np.arange(n_draws),
            "ticker": list(tickers),
            "simulation": np.arange(n_sims),
        },
    )

    if az is not None:
        # ArviZ 1.0 replaced ``az.InferenceData`` with ``xarray.DataTree``.
        # Build the equivalent DataTree directly to avoid the deprecation
        # warning emitted on ``arviz.InferenceData`` attribute access.
        idata = xr.DataTree.from_dict(
            {
                "posterior": xr.DataTree(posterior),
                "posterior_predictive": xr.DataTree(posterior_predictive),
                "constant_data": xr.DataTree(constant_data),
            }
        )
    else:
        idata = type("InferenceData", (), {
            "posterior": posterior,
            "posterior_predictive": posterior_predictive,
            "constant_data": constant_data,
            "extend": lambda self, other: None,
        })()

    logger.info(
        "Analytical fallback: %d tickers, %d synthetic draws", n_tickers, n_draws
    )
    return idata


class MonteCarloReturnSimulation:
    """Bayesian Monte Carlo return simulation model.

    Uses learnable priors on return mean and volatility, then generates
    forward simulations via posterior predictive sampling.

    The MCMC step only infers ``mu_return`` and ``sigma_return`` (2 × N_tickers
    parameters).  ``prob_positive`` is derived analytically from the Normal CDF,
    and ``sim_returns`` are produced in a lightweight post-hoc NumPy pass.

    Schema alignment
    ----------------
    Inputs are registered as ``pm.Data`` containers (``historical_means``,
    ``historical_stds``) so they appear in ``idata.constant_data`` and can be
    swapped via ``pm.set_data({...})`` for out-of-sample prediction —
    matching the API of the other PyMC models in this package.
    """

    def __init__(self) -> None:
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[InferenceLike] = None

    def fit(
        self,
        historical_means: np.ndarray,
        historical_stds: np.ndarray,
        tickers: np.ndarray,
        n_sims: int = 1_000,
        samples: int = 500,
        tune: int = 500,
        chains: int = 2,
        cores: int = _CORES_DEFAULT,
        target_accept: float = 0.90,
        random_seed: int = _RANDOM_SEED,
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> "tuple[InferenceLike, Optional[pm_typing.Model]]":
        """Fit the model and return ``(InferenceData, Model)``.

        Mirrors the ``fit`` signature of the other PyMC models in this
        package (``EarningsBeatBayesian``, ``CreditRiskBayesian``,
        ``DividendSafetyBayesian`` …).
        """
        idata, model = fit(
            historical_means=historical_means,
            historical_stds=historical_stds,
            tickers=tickers,
            n_sims=n_sims,
            samples=samples,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            nuts_sampler=nuts_sampler,
            return_model=True,
            **sample_kwargs,
        )
        self.idata_ = idata
        self.model_ = model
        return idata, model
