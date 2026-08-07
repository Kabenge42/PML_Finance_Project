"""
Shared Bayesian-workflow helpers for every PyMC model in
:mod:`probabilistic_ml_model.pymc_models`.

The project follows the PyMC *Bayesian workflow* (conceptual model building →
prior predictive → computational implementation → fitting & diagnostics →
model evaluation → model comparison → expansion/simplification → decision
analysis). Historically only two of those stages had shared code, and only
inside :mod:`~probabilistic_ml_model.pymc_models.KalmanFilterModel`; the
standalone workflow script ``pymc_kalman_filter_pt.py`` (§6 prior predictive,
§8 posterior predictive, §9 diagnostics) was the sole full reference
implementation. This module lifts those pieces to one place so every model can
reach the same stages without copying ~12 lines of sampler boilerplate:

* :func:`build_sample_kwargs` — the canonical ``pm.sample()`` keyword
  assembly (project compile kwargs, ``log_likelihood`` policy, caller
  overrides, the nutpie ``idata_kwargs`` strip and the ``chains < 2``
  diagnostic gate). Lifted from ``KalmanFilterPriceTarget._build_sample_kwargs``.
* :func:`log_sample_diagnostics` — divergence + bulk-ESS self-reporting.
  Lifted from ``KalmanFilterPriceTarget._log_sample_diagnostics``.
* :func:`prior_predictive_check` / :func:`posterior_predictive_check` — the
  two predictive stages, modelled on ``run_prior_predictive`` /
  ``run_posterior_predictive`` in ``pymc_kalman_filter_pt.py``.
* :func:`attach_log_likelihood` — makes ``az.loo`` / ``az.compare`` possible
  **after the fact**, which is the only route that survives the nutpie
  ``idata_kwargs`` strip (see :func:`build_sample_kwargs`).

Every helper degrades gracefully: PyMC/ArviZ are imported lazily and a missing
optional dependency, or a failure inside a best-effort diagnostic, logs and
returns rather than breaking a fit that already succeeded.

Notes
-----
``log_likelihood`` is **off by default** across the project. Computing and
storing the pointwise log-likelihood roughly doubles ``InferenceData`` size and
adds materially to wall-clock on a ~5k-ISIN cross-section, and the pipeline
paths never consume it. Model comparison is therefore opt-in, via either

* ``fit(..., idata_kwargs={"log_likelihood": True})`` — works for the
  ``pymc`` / ``numpyro`` / ``blackjax`` samplers, but is **silently discarded
  under nutpie** (the project default), which ignores ``idata_kwargs``; or
* :func:`attach_log_likelihood` after sampling — sampler-independent, and the
  recommended route.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import numpy as np

from probabilistic_ml_model.pymc_models._pytensor_compat import (
    get_pytensor_compile_kwargs,
)

try:
    import pymc as pm
except ImportError:  # pragma: no cover - optional dependency
    pm = None  # type: ignore[assignment]

try:
    import arviz as az
except ImportError:  # pragma: no cover - optional dependency
    az = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

#: Project-wide convergence gate. A minimum bulk-ESS below this is reported as
#: a warning by :func:`log_sample_diagnostics`. 400 is the usual rule of thumb
#: (~100 effective draws per chain at the project default ``chains=4``).
MIN_ESS_GATE = 400

#: Backwards-compatible private alias — ``KalmanFilterModel`` historically
#: exposed this name at module level.
_MIN_ESS_GATE = MIN_ESS_GATE

__all__ = [
    "MIN_ESS_GATE",
    "attach_log_likelihood",
    "build_sample_kwargs",
    "log_sample_diagnostics",
    "posterior_dataset",
    "posterior_predictive_check",
    "prior_predictive_check",
]


def posterior_dataset(idata: Any) -> Any:
    """Return the ``posterior`` group as a plain :class:`xarray.Dataset`.

    ArviZ 1.x stores inference results as :class:`xarray.DataTree`, whose group
    nodes do not support list-of-name selection (``posterior[keys]``) the way
    ``azs.rhat`` / ``azs.ess`` inputs require. Unwrap via ``.dataset`` on the
    DataTree node, falling back to ``.to_dataset()`` for legacy
    ``arviz.InferenceData`` groups.

    Parameters
    ----------
    idata
        Inference data carrying a ``posterior`` group.

    Returns
    -------
    xarray.Dataset
        The posterior group as a flat dataset.
    """
    posterior = idata.posterior
    return posterior.dataset if hasattr(posterior, "dataset") else posterior.to_dataset()


def build_sample_kwargs(
        *,
        samples: int,
        tune: int,
        chains: int,
        target_accept: float,
        random_seed: int,
        nuts_sampler: Optional[str] = None,
        sample_kwargs: Optional[dict[str, Any]] = None,
        cores: Optional[int] = None,
        model_name: str = "model",
        progressbar: bool = True,
) -> dict[str, Any]:
    """Assemble the keyword arguments for :func:`pymc.sample`.

    Applies the project defaults (compile kwargs, no log-likelihood), layers in
    ``nuts_sampler`` / ``cores`` and caller overrides, then strips
    ``idata_kwargs`` for nutpie (which ignores it and warns).

    Parameters
    ----------
    samples, tune, chains, target_accept, random_seed
        The standard NUTS budget forwarded from the model's ``fit()``.
    nuts_sampler
        Optional sampler name (``"nutpie"`` / ``"numpyro"`` / ``"blackjax"`` /
        ``"pymc"``). Omitted from the result when ``None`` so PyMC picks.
    sample_kwargs
        Caller overrides (a model's ``**sample_kwargs``). Applied **last**, so
        anything here wins over the project defaults — including
        ``idata_kwargs``.
    cores
        Chains to run in parallel. Omitted when ``None`` so PyMC decides.
    model_name
        Label used in the emitted warnings.
    progressbar
        Forwarded to :func:`pymc.sample`.

    Returns
    -------
    dict[str, Any]
        Keyword arguments ready to splat into ``pm.sample(**kwargs)``.

    Notes
    -----
    The layering order is load-bearing and must not be rearranged::

        defaults -> nuts_sampler/cores -> setdefault(idata_kwargs) ->
        update(sample_kwargs) -> nutpie strip

    ``setdefault`` before ``update`` is what lets a caller re-enable
    ``log_likelihood``; the nutpie strip afterwards is what silently defeats
    it under the project's default sampler. Use :func:`attach_log_likelihood`
    instead of fighting that ordering.

    Warns when the effective chain count (after ``sample_kwargs`` overrides) is
    below 2: r-hat and between-chain ESS are undefined for a single chain, so
    downstream ArviZ diagnostics come back NaN.
    """
    scall: dict[str, Any] = dict(
        draws=samples,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        progressbar=progressbar,
        compile_kwargs=get_pytensor_compile_kwargs(),
    )
    if nuts_sampler is not None:
        scall["nuts_sampler"] = nuts_sampler
    if cores is not None:
        scall["cores"] = cores
    scall.setdefault("idata_kwargs", {"log_likelihood": False})
    if sample_kwargs:
        scall.update(sample_kwargs)

    eff_chains = int(scall.get("chains", chains))
    if eff_chains < 2:
        logger.warning(
            "%s: sampling with chains=%d; r_hat and between-chain ESS "
            "diagnostics require >= 2 chains (4 recommended) and will be NaN. "
            "Single-chain fits are for fast tests only.",
            model_name,
            eff_chains,
        )

    # nutpie ignores idata_kwargs and emits a UserWarning; strip it to keep
    # logs clean while preserving behaviour for other samplers.
    if scall.get("nuts_sampler") == "nutpie":
        requested = scall.get("idata_kwargs") or {}
        if requested.get("log_likelihood"):
            logger.info(
                "%s: idata_kwargs['log_likelihood']=True is ignored by nutpie; "
                "call attach_log_likelihood(idata, model) after sampling to "
                "enable az.loo / az.compare.",
                model_name,
            )
        scall.pop("idata_kwargs", None)
    return scall


def log_sample_diagnostics(
        idata: Any,
        *,
        model_name: str = "model",
        tag: Optional[str] = None,
        min_ess: int = MIN_ESS_GATE,
) -> None:
    """Log divergences and minimum ESS so fit quality is self-reported.

    Inspects ``idata.sample_stats["diverging"]`` and, when available, the bulk
    effective sample size, emitting warnings rather than relying on console
    scraping of the sampler output.

    Parameters
    ----------
    idata
        The object returned by :func:`pymc.sample` (ArviZ ``InferenceData`` /
        ``xarray.DataTree`` or a ``MultiTrace``).
    model_name
        Model label used in the log messages.
    tag
        Optional extra label (e.g. an ISIN) appended as ``model_name[tag]``.
    min_ess
        Bulk-ESS gate below which a warning is emitted. Defaults to
        :data:`MIN_ESS_GATE`.

    Notes
    -----
    Best-effort by design: a ``MultiTrace`` without ``sample_stats``, or an
    ArviZ failure on an unusual group layout, returns silently instead of
    breaking a fit that already succeeded.
    """
    label = f"{model_name}[{tag}]" if tag is not None else model_name
    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is None or "diverging" not in getattr(sample_stats, "data_vars", {}):
        return
    try:
        n_div = int(sample_stats["diverging"].sum())
    except Exception:  # pragma: no cover - defensive
        return
    if n_div:
        logger.warning(
            "%s: %d divergences after tuning; consider a non-centred "
            "parameterization or a higher target_accept.",
            label,
            n_div,
        )
    if az is not None and hasattr(az, "ess"):
        try:
            min_observed = float(az.ess(idata).to_array().min())
        except Exception:  # pragma: no cover - defensive
            min_observed = float("nan")
        if np.isfinite(min_observed) and min_observed < min_ess:
            logger.warning(
                "%s: minimum ESS %.0f < %d (project convergence gate); "
                "increase tune/draws for reliable r-hat / ESS.",
                label,
                min_observed,
                min_ess,
            )


def prior_predictive_check(
        model: Any,
        *,
        var_names: Optional[Iterable[str]] = None,
        draws: int = 500,
        random_seed: int = 42,
) -> Any:
    """Sample the prior predictive distribution for ``model``.

    The *prior predictive* stage of the Bayesian workflow: draw from the model
    with no data attached and check that the implied observations are on a
    plausible scale. Callers are expected to de-standardise the result onto an
    interpretable scale (e.g. implied upside in %) and compare it against the
    empirical distribution — see ``run_prior_predictive`` in
    ``pymc_kalman_filter_pt.py`` for the reference treatment.

    Parameters
    ----------
    model
        A built :class:`pymc.Model`.
    var_names
        Variables to record. Names not present in ``model.named_vars`` are
        dropped, so callers may pass an optimistic superset covering several
        parameterizations.
    draws
        Number of prior draws.
    random_seed
        RNG seed.

    Returns
    -------
    arviz.InferenceData | xarray.DataTree
        The prior / prior_predictive groups.

    Raises
    ------
    ImportError
        If PyMC is not installed.
    """
    if pm is None:
        raise ImportError(
            "PyMC is not available. Install pymc to run prior predictive checks."
        )

    resolved: Optional[list[str]] = None
    if var_names is not None:
        named = getattr(model, "named_vars", {})
        resolved = [v for v in var_names if v in named]
        if not resolved:
            logger.warning(
                "prior_predictive_check: none of the requested var_names exist "
                "on the model; sampling all variables instead."
            )
            resolved = None

    with model:
        return pm.sample_prior_predictive(
            draws=draws,
            var_names=resolved,
            random_seed=random_seed,
            return_inferencedata=True,
            compile_kwargs=get_pytensor_compile_kwargs(),
        )


def posterior_predictive_check(
        model: Any,
        idata: Any,
        *,
        var_names: Optional[Iterable[str]] = None,
        random_seed: int = 42,
        extend: bool = True,
        progressbar: bool = True,
) -> Any:
    """Sample the posterior predictive distribution for ``model``.

    The *model evaluation* stage: replicate the observed data from the fitted
    posterior so calibration can be assessed. Pair this with at least one
    calibration statistic — an ECDF overlay, a t-stat check, an interval
    coverage table or a PIT ECDF (see ``run_posterior_predictive`` in
    ``pymc_kalman_filter_pt.py``).

    Parameters
    ----------
    model
        The fitted :class:`pymc.Model`.
    idata
        Inference data carrying the ``posterior`` group.
    var_names
        Optional subset of observed/deterministic variables to replicate.
    random_seed
        RNG seed.
    extend
        When ``True`` (default) the ``posterior_predictive`` group is added to
        ``idata`` in place and ``idata`` is returned.
    progressbar
        Forwarded to :func:`pymc.sample_posterior_predictive`.

    Returns
    -------
    arviz.InferenceData | xarray.DataTree
        ``idata`` extended in place when ``extend`` is ``True``, otherwise a
        fresh object holding only the ``posterior_predictive`` group.

    Raises
    ------
    ImportError
        If PyMC is not installed.
    """
    if pm is None:
        raise ImportError(
            "PyMC is not available. Install pymc to run posterior predictive checks."
        )

    kwargs: dict[str, Any] = dict(
        random_seed=random_seed,
        progressbar=progressbar,
        compile_kwargs=get_pytensor_compile_kwargs(),
    )
    if var_names is not None:
        kwargs["var_names"] = list(var_names)

    with model:
        return pm.sample_posterior_predictive(
            idata, extend_inferencedata=extend, **kwargs
        )


def _has_group(idata: Any, name: str) -> bool:
    """Return whether ``idata`` already carries the group ``name``.

    Handles both ArviZ ``InferenceData`` (``groups()`` is a method) and
    ``xarray.DataTree`` (group membership is ``in`` / ``children``), so the
    check works across the ArviZ 1.x transition.
    """
    groups = getattr(idata, "groups", None)
    if callable(groups):
        try:
            return name in groups()
        except Exception:  # pragma: no cover - defensive
            return False
    try:
        return name in idata
    except Exception:  # pragma: no cover - defensive
        return False


def attach_log_likelihood(idata: Any, model: Any) -> Any:
    """Compute and attach the pointwise ``log_likelihood`` group in place.

    The *model comparison* stage. Because the project samples with
    ``idata_kwargs={"log_likelihood": False}`` — and because the default
    sampler (nutpie) ignores ``idata_kwargs`` entirely — this post-hoc route is
    the only one that reliably enables :func:`arviz.loo`, :func:`arviz.waic`
    and :func:`arviz.compare` regardless of which sampler produced ``idata``.

    Parameters
    ----------
    idata
        Inference data carrying a ``posterior`` group.
    model
        The :class:`pymc.Model` that produced ``idata``.

    Returns
    -------
    arviz.InferenceData | xarray.DataTree
        ``idata``, extended with a ``log_likelihood`` group. Returned unchanged
        (with a warning) if the computation fails.

    Raises
    ------
    ImportError
        If PyMC is not installed.

    Examples
    --------
    >>> idata_a, model_a = ModelA().fit(...)          # doctest: +SKIP
    >>> idata_b, model_b = ModelB().fit(...)          # doctest: +SKIP
    >>> attach_log_likelihood(idata_a, model_a)       # doctest: +SKIP
    >>> attach_log_likelihood(idata_b, model_b)       # doctest: +SKIP
    >>> az.compare({"a": idata_a, "b": idata_b})      # doctest: +SKIP

    Notes
    -----
    Cost scales with ``chains x draws x n_observations``; on a full
    cross-section this can dominate the fit itself, which is why it is opt-in.
    """
    if pm is None:
        raise ImportError(
            "PyMC is not available. Install pymc to compute the log-likelihood."
        )
    if _has_group(idata, "log_likelihood"):
        return idata
    try:
        with model:
            pm.compute_log_likelihood(idata, extend_inferencedata=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "attach_log_likelihood: could not compute the log-likelihood "
            "group (%s); az.loo / az.compare remain unavailable for this fit.",
            exc,
        )
    return idata
