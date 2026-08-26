"""Forward-return forecast layer for the Kalman v2 price-target panel.

Why this module exists
----------------------
:mod:`probabilistic_ml_model.pymc_models.KalmanFilterModel_v2` is a **pure
cross-sectional smoother of a backward-looking trail**. It has no ``pm.set_data``
path, no ``predictions=True`` call and no ``forecast()`` method; the two functions
it exports whose names contain "forecast" —
:func:`~KalmanFilterModel_v2.forecast_error_variance` and
:func:`~KalmanFilterModel_v2.apply_forecast_error_shrinkage` — measure the standard
error of the analyst consensus *at the snapshot*, not anything over a horizon. The
builder docstring records forward prediction as deliberately out of scope.

Everything downstream of the fit nevertheless needs a forward distribution:
``er_mean`` / ``er_sd`` / ``er_p05``, ``cvar05``, ``exp_vol``, ``starr``, the sized
book and every GEIB card. Today that distribution comes from
:func:`~probabilistic_ml_model.pymc_models._price_target_mc.simulate_lagged_risk_adjusted_returns`,
an AR(1) recursion applied to posterior draws *after* the fit with ``rho = 0.85``
and ``horizon = 4`` as hand-set constants. Two consequences follow, and this module
addresses both.

**The horizon has no calendar meaning.** ``horizon=4`` is four unitless periods,
while the model's own state decay is calibrated in *real days*: ``_ou_correlation``
takes ``|t_i - t_j|`` in calendar days and the posterior reports
``ou_length_scale_days`` directly (81.2 days, a 56-day half-life, on run
``02a1f2a641ef``). A price target is a ~12-month quantity — ``PRICE_TARGET_HORIZON_YEARS``
in ``dashboards/geib/metrics.py`` says so. Here the horizon is days, the step grid
is days, and the forward decay **is** the fitted OU kernel rather than a second,
unrelated persistence parameter.

**The cross-section is incoherent.** In the AR simulator the shock is drawn
``sigma_draws * rng.standard_t(df=nu_row, size=(n_isin, n_samples))`` — independently
per ``(isin, sample)``. The only cross-sectional dependence is through shared posterior
parameter draws, which couples the *means* and leaves the *shocks* orthogonal. The risk
book then computes ``port_cvar`` as ``w @ held`` over those draws, which treats 25 names'
idiosyncratic risk as uncorrelated, while ``port_vol`` in the same summary is a weighted
*sum* of per-name volatilities, which assumes perfect correlation. Two portfolio risk
numbers on one book, assuming opposite things, neither of them modelling a market. This
module emits **joint** scenarios: a market factor and one factor per configured group
level are shared across names, so diversification has to be earned.

The factor split is variance-preserving by construction (see
:func:`_common_factor_shocks`), so turning it on changes the *joint* distribution and
leaves every per-name marginal — and therefore ``er_sd`` and ``exp_vol`` — where it was.
That is what allows the structure to be adopted without silently re-scaling every
exported risk column, and it is pinned by a test rather than asserted here.

What this module does NOT claim
-------------------------------
Nothing here validates the decision. Both the fitted model and any backtest over the
trail score against the analyst price-target series the model was fitted to; the v2 gate
report is a complete *internal* report and says nothing about realised returns. See
``scripts/score_panel_vintages.py`` — the only realised-return instrument in the repo —
and the caveats it carries (survivorship, no FX or total return, horizon mismatch, one
overlapping period). Scoring a forecast produced here means capturing a vintage now and
scoring it against a later one; nothing in this module can substitute for that wait.

Backends
--------
``native``
    Plain NumPy on the posterior draws. No optional dependency, deterministic under
    ``random_seed``, and the reference implementation of the scenario generator.
``pymc_forecast``
    A ``pymc_forecast.ForecastingModel`` subclass. The panel maps one-to-one onto that
    library's own hierarchical example (its ``origin`` axis is our ``isin`` axis).
    Requires the optional ``pymc-forecast`` dependency.
``statespace``
    A custom ``pymc_extras.statespace.PyMCStateSpace`` carrying the **factor block only**
    — ``k_endog`` of order ten, never the 6,500-wide observation vector, which is not a
    viable Kalman filter. Requires ``pymc-extras``, which at the time of writing pins
    ``pymc<6.3`` / ``pytensor<3.3`` and therefore cannot be installed alongside the
    project's current stack without downgrading both. The guard below reports that
    rather than failing obscurely.

Only ``native`` is implemented in this revision; the other two raise
:class:`NotImplementedError` with the reason.

See Also
--------
probabilistic_ml_model.pymc_models.PortfolioOptimizationModel
    The decision layer that consumes :class:`ForecastDraws`.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

# NOTE: pymc itself is deliberately NOT imported. The native backend reads posterior
# draws and simulates forward in NumPy, so this module is usable wherever the draws
# are — a scoring script, a notebook, a dashboard process — without paying for the
# PyMC import or its PyTensor compilation. The two backends that DO need a model
# graph guard their own dependency below.

try:  # optional: the `forecast` extra
    import pymc_forecast as _pmf
except ImportError:  # pragma: no cover - optional dependency
    _pmf = None  # type: ignore[assignment]

try:  # optional: pymc-extras, needed only for the statespace backend
    from pymc_extras.statespace.core.statespace import (  # noqa: F401
        PyMCStateSpace as _PyMCStateSpace,
    )
except ImportError:  # pragma: no cover - optional dependency
    _PyMCStateSpace = None  # type: ignore[assignment]

from probabilistic_ml_model._pymc_arviz_compat import InferenceLike

logger = logging.getLogger(__name__)

__all__ = [
    "FORECAST_BACKENDS",
    "ForecastConfig",
    "ForecastDraws",
    "ForecastInputs",
    "prepare_forecast_inputs",
    "simulate_forecast",
    "forecast_from_posterior",
    "summarize_forecast",
]

#: The backends :class:`ForecastConfig` accepts. ``native`` is always available;
#: the other two are gated on optional dependencies.
FORECAST_BACKENDS: tuple[str, ...] = ("native", "pymc_forecast", "statespace")

_EPS = 1e-12

#: Smallest Student-t degrees of freedom used when standardising a factor to unit
#: variance. ``Var[t_nu] = nu / (nu - 2)`` diverges at ``nu = 2``; the v2 model floors
#: ``nu`` at 2.5 and the AR simulator it is contrasted against floors at 3.0. Matching
#: the latter keeps the two engines' tails comparable — see
#: ``_price_target_mc.simulate_lagged_risk_adjusted_returns``.
_MIN_NU: float = 3.0


@dataclass(frozen=True)
class ForecastConfig:
    """Knobs for the forward simulation. Frozen; override with :func:`dataclasses.replace`.

    Attributes
    ----------
    horizon_days
        Total forecast horizon in calendar days. Defaults to 365 because the
        response the model fits is a ~12-month analyst price target
        (``PRICE_TARGET_HORIZON_YEARS = 1.0`` in ``dashboards/geib/metrics.py``).
    step_days
        Length of one simulation step in calendar days. The default 91 gives four
        steps over a 365-day horizon, which is deliberately the same step count as
        the AR simulator's ``mc_horizon = 4`` so that
        ``compare_forecast_engines`` contrasts like with like.
    n_scenarios
        Number of joint scenarios drawn. Each scenario fixes ONE posterior sample
        across every name, which is what makes parameter uncertainty a shared
        component rather than noise. When this exceeds the number of available
        posterior samples the samples are drawn with replacement.
    backend
        One of :data:`FORECAST_BACKENDS`.
    factor_levels
        Panel coordinate columns that each get a shared factor. Must be present in
        ``KalmanPanelV2.coord_idx``.
    n_market_factors
        Number of universe-wide factors. Currently only 0 or 1 is meaningful; a
        second market factor is unidentified without a loadings model.
    factor_share
        Fraction of each name's forward shock VARIANCE carried by the shared
        factors, split evenly across the market factor and each entry of
        ``factor_levels``. ``0.0`` reproduces the independent-shock behaviour of
        the AR simulator. The remaining ``1 - factor_share`` is idiosyncratic, so
        total per-name variance is invariant to this knob.

        This is a **prior, not an estimate**. The v2 panel is a cross-section of
        analyst price-target trails and cannot identify a return factor structure;
        0.35 is a plausible equity-market common share, not a measurement. Treat it
        the way the project treats ``forecast_error_multiplier``: grid it and report
        the sensitivity rather than quoting the point value.
    uplift_clip
        ``(lo, hi)`` clip on the cumulative simple return, applied in LOG space so
        it is sign-preserving. The SSOT is ``UPLIFT_CLIP_LO`` / ``UPLIFT_CLIP_HI``
        in ``pymc_kalman_filter_pt_v2.py``; the workflow passes its own values in,
        and these defaults exist so the module is usable standalone.
    random_seed
        Seed for the scenario PRNG. Given the same posterior and config the draws
        are bit-for-bit reproducible.
    """

    horizon_days: int = 365
    step_days: int = 91
    n_scenarios: int = 2000
    backend: str = "native"
    factor_levels: tuple[str, ...] = ("trading_region", "sector")
    n_market_factors: int = 1
    factor_share: float = 0.35
    uplift_clip: tuple[float, float] = (-0.95, 5.0)
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.backend not in FORECAST_BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Valid: {FORECAST_BACKENDS}"
            )
        if self.horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days!r}")
        if self.step_days <= 0:
            raise ValueError(f"step_days must be positive, got {self.step_days!r}")
        if self.step_days > self.horizon_days:
            raise ValueError(
                f"step_days ({self.step_days}) exceeds horizon_days "
                f"({self.horizon_days}); the horizon would be a single partial step."
            )
        if self.n_scenarios < 2:
            raise ValueError(f"n_scenarios must be >= 2, got {self.n_scenarios!r}")
        if self.n_market_factors not in (0, 1):
            raise ValueError(
                "n_market_factors must be 0 or 1; a second universe-wide factor is "
                f"unidentified without a loadings model (got {self.n_market_factors!r})"
            )
        if not (0.0 <= self.factor_share < 1.0):
            raise ValueError(
                f"factor_share must be in [0, 1), got {self.factor_share!r}. "
                "At 1.0 a name has no idiosyncratic risk at all."
            )
        lo, hi = self.uplift_clip
        if not lo > -1.0:
            raise ValueError(f"uplift_clip lower bound must exceed -1, got {lo!r}")
        if not hi > lo:
            raise ValueError(f"uplift_clip must be increasing, got {self.uplift_clip!r}")

    @property
    def n_steps(self) -> int:
        """Number of simulation steps: the horizon divided by the step, to nearest.

        Rounding to *nearest* rather than up is deliberate. ``ceil`` on the default
        365/91 grid yields five steps, the last one a single day — a stub that
        carries almost no variance, breaks the intended like-for-like with the AR
        simulator's four periods, and makes ``time_days`` read as though the horizon
        were sampled more finely than it is. To nearest gives four steps, the last
        stretched to 92 days.
        """
        return max(1, int(round(self.horizon_days / self.step_days)))

    @property
    def time_days(self) -> np.ndarray:
        """Day offsets of each step end. The last entry is exactly ``horizon_days``.

        Intermediate edges sit on the nominal step grid; the final step absorbs
        whatever remains, so the horizon is covered exactly once with no stub.
        """
        edges = np.arange(1, self.n_steps + 1, dtype="float64") * float(self.step_days)
        edges[-1] = float(self.horizon_days)
        return np.maximum.accumulate(np.minimum(edges, float(self.horizon_days)))

    @property
    def step_fractions(self) -> np.ndarray:
        """Fraction of the total horizon carried by each step; sums to exactly 1."""
        ends = self.time_days
        starts = np.concatenate(([0.0], ends[:-1]))
        return (ends - starts) / float(self.horizon_days)

    @property
    def n_factor_blocks(self) -> int:
        """Number of shared factor blocks: the market factor plus one per level."""
        return int(self.n_market_factors) + len(self.factor_levels)

    @property
    def log_clip(self) -> tuple[float, float]:
        """:attr:`uplift_clip` on the log scale, where the clip is applied."""
        lo, hi = self.uplift_clip
        return float(np.log1p(lo)), float(np.log1p(hi))

    @classmethod
    def from_env(cls) -> "ForecastConfig":
        """Build from ``KALMAN_FORECAST_*`` environment variables.

        Reads ``KALMAN_FORECAST_BACKEND``, ``KALMAN_FORECAST_HORIZON_DAYS``,
        ``KALMAN_FORECAST_STEP_DAYS``, ``KALMAN_FORECAST_SCENARIOS``,
        ``KALMAN_FORECAST_FACTOR_SHARE`` and ``RANDOM_SEED``. Everything else keeps
        its dataclass default and is overridden programmatically, which is the
        convention ``KalmanRunConfigV2.from_env`` follows.
        """

        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError:
                logger.warning("%s=%r is not an integer; using %d", name, raw, default)
                return default

        def _float(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                return float(raw)
            except ValueError:
                logger.warning("%s=%r is not a float; using %s", name, raw, default)
                return default

        backend = os.environ.get("KALMAN_FORECAST_BACKEND", "native").strip() or "native"
        if backend not in FORECAST_BACKENDS:
            logger.warning(
                "KALMAN_FORECAST_BACKEND=%r is not one of %s; using 'native'",
                backend,
                FORECAST_BACKENDS,
            )
            backend = "native"
        return cls(
            horizon_days=_int("KALMAN_FORECAST_HORIZON_DAYS", 365),
            step_days=_int("KALMAN_FORECAST_STEP_DAYS", 91),
            n_scenarios=_int("KALMAN_FORECAST_SCENARIOS", 2000),
            backend=backend,
            factor_share=_float("KALMAN_FORECAST_FACTOR_SHARE", 0.35),
            random_seed=_int("RANDOM_SEED", 42),
        )


@dataclass(frozen=True)
class ForecastInputs:
    """Per-name posterior quantities the scenario generator needs, in RETURN space.

    Built by :func:`prepare_forecast_inputs`. Separated from the simulation so that
    the de-standardisation happens exactly once, in one place, and so a caller that
    has already resolved a *shrunk* decision latent can hand it over rather than
    letting this module re-resolve an unshrunk one — the failure
    :class:`ScreenDraws` was introduced to prevent.

    Attributes
    ----------
    isins
        ``(n_isin,)`` identifiers. Carried everywhere; never implied by position.
    mu_log
        ``(n_isin, n_samples)`` posterior draws of the total expected LOG uplift
        over the price-target horizon, already de-standardised.
    sigma_log
        ``(n_isin, n_samples)`` posterior draws of the per-name observation scale
        on the same log scale.
    nu
        ``(n_samples,)`` Student-t degrees of freedom, or a length-1 array.
    group_index
        Level name -> ``(n_isin,)`` integer codes into that level's factor.
    ou_length_scale_days
        Posterior mean of the fitted OU length scale, in days. ``None`` when the
        posterior does not carry it, in which case the forward state does not decay.
    """

    isins: np.ndarray
    mu_log: np.ndarray
    sigma_log: np.ndarray
    nu: np.ndarray
    group_index: dict[str, np.ndarray] = field(default_factory=dict)
    ou_length_scale_days: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mu_log.ndim != 2:
            raise ValueError(f"mu_log must be 2-D (n_isin, n_samples), got {self.mu_log.shape}")
        if self.sigma_log.shape != self.mu_log.shape:
            raise ValueError(
                f"sigma_log {self.sigma_log.shape} must match mu_log {self.mu_log.shape}"
            )
        if len(self.isins) != self.mu_log.shape[0]:
            raise ValueError(
                f"isins has {len(self.isins)} entries but mu_log has "
                f"{self.mu_log.shape[0]} rows"
            )
        n_samples = self.mu_log.shape[1]
        if self.nu.size not in (1, n_samples):
            raise ValueError(
                f"nu must be scalar or length {n_samples}, got size {self.nu.size}"
            )
        for level, codes in self.group_index.items():
            if len(codes) != len(self.isins):
                raise ValueError(
                    f"group_index[{level!r}] has {len(codes)} codes for "
                    f"{len(self.isins)} names"
                )

    @property
    def n_isin(self) -> int:
        return int(self.mu_log.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.mu_log.shape[1])


@dataclass(frozen=True)
class ForecastDraws:
    """Joint forward-return scenarios, the output contract of this module.

    Attributes
    ----------
    isins
        ``(n_isin,)`` identifiers aligned to axis 0 of :attr:`paths` and
        :attr:`terminal`. Pass these alongside the draws to every consumer —
        ``compute_cvar_aware_book`` takes ``return_draws_isins`` for exactly this
        reason, after a positional join silently attributed every risk column to the
        wrong name.
    paths
        ``(n_isin, n_scenarios, n_steps)`` simple returns for each step.
    terminal
        ``(n_isin, n_scenarios)`` cumulative simple return over the whole horizon.
        **This is the decision quantity.** :attr:`pooled_returns` pools the per-step
        marginals, which is what ``er_sd`` has always been, but a portfolio decision
        at a 12-month horizon is about the compounded outcome, not about the average
        of four quarterly moves.
    time_days
        ``(n_steps,)`` day offsets of each step end.
    backend
        Which engine produced the draws. Exported as a column so a stored frame says
        what made it, following the ``compare_arms_fast`` precedent.
    factor_share
        The common-variance share used, recorded because it is a prior.
    factor_draws
        ``(n_blocks, n_scenarios, n_steps)`` standardised shared factors, or ``None``
        when ``factor_share == 0``. Kept for attribution: a book's tail can be
        decomposed into common and idiosyncratic contributions from these.
    factor_labels
        Names of the factor blocks, aligned to axis 0 of :attr:`factor_draws`.
    """

    isins: np.ndarray
    paths: np.ndarray
    terminal: np.ndarray
    time_days: np.ndarray
    backend: str = "native"
    factor_share: float = 0.0
    factor_draws: Optional[np.ndarray] = None
    factor_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.paths.ndim != 3:
            raise ValueError(
                f"paths must be 3-D (n_isin, n_scenarios, n_steps), got {self.paths.shape}"
            )
        n_isin, n_scen, n_steps = self.paths.shape
        if len(self.isins) != n_isin:
            raise ValueError(
                f"isins has {len(self.isins)} entries for {n_isin} rows of paths"
            )
        if self.terminal.shape != (n_isin, n_scen):
            raise ValueError(
                f"terminal {self.terminal.shape} must be ({n_isin}, {n_scen})"
            )
        if len(self.time_days) != n_steps:
            raise ValueError(
                f"time_days has {len(self.time_days)} entries for {n_steps} steps"
            )
        if self.factor_draws is not None:
            if self.factor_draws.shape[1:] != (n_scen, n_steps):
                raise ValueError(
                    f"factor_draws {self.factor_draws.shape} must end in "
                    f"({n_scen}, {n_steps})"
                )
            if len(self.factor_labels) != self.factor_draws.shape[0]:
                raise ValueError(
                    f"{len(self.factor_labels)} factor labels for "
                    f"{self.factor_draws.shape[0]} factor blocks"
                )

    @property
    def n_isin(self) -> int:
        return int(self.paths.shape[0])

    @property
    def n_scenarios(self) -> int:
        return int(self.paths.shape[1])

    @property
    def n_steps(self) -> int:
        return int(self.paths.shape[2])

    @property
    def horizon_days(self) -> float:
        return float(self.time_days[-1]) if len(self.time_days) else 0.0

    @property
    def pooled_returns(self) -> np.ndarray:
        """``paths`` flattened to ``(n_isin, n_scenarios * n_steps)``.

        Deliberately identical in shape and pooling to
        ``ScreenDraws.pooled_returns``, so this object is a drop-in ``return_draws``
        for ``compute_cvar_aware_book``. That routine asserts ``exp_vol == er_sd`` on
        every call; the identity holds because both are the standard deviation of
        *this* array, and it is preserved only as long as the same array also feeds
        ``summarize_mc_returns``.
        """
        return self.paths.reshape(self.n_isin, -1)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def _posterior_group(idata: "InferenceLike") -> Any:
    """Return the ``posterior`` group of a DataTree or InferenceData."""
    if hasattr(idata, "posterior"):
        return idata.posterior
    try:
        return idata["posterior"]
    except (TypeError, KeyError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"cannot find a posterior group on {type(idata).__name__}"
        ) from exc


def _flat_draws(
        idata: "InferenceLike",
        name: str,
        *,
        per_isin: bool = True,
) -> np.ndarray:
    """Flatten a posterior variable to ``(isin, sample)`` or ``(sample,)``.

    Mirrors ``pymc_kalman_filter_pt_v2._posterior_draws`` term for term. It is
    reimplemented rather than imported because the workflow script imports this
    package, so importing the script from here would close the cycle.
    """
    post = _posterior_group(idata)
    if name not in post:
        raise KeyError(
            f"{name!r} not in posterior. Available: {sorted(map(str, post.data_vars))}"
        )
    arr = np.asarray(post[name])
    if per_isin:
        return arr.reshape(-1, arr.shape[-1]).T
    return arr.reshape(-1)


def prepare_forecast_inputs(
        idata: "InferenceLike",
        panel: Any,
        *,
        config: Optional[ForecastConfig] = None,
        latent: Optional[Any] = None,
) -> ForecastInputs:
    """De-standardise the posterior into the log-return quantities the sim needs.

    Parameters
    ----------
    idata
        Fitted v2 inference data.
    panel
        The :class:`~KalmanFilterModel_v2.KalmanPanelV2` the model was fitted on.
        Supplies ``isins``, ``response_mean`` / ``response_std`` and the group
        coordinate codes.
    config
        Forecast configuration; only ``factor_levels`` and ``random_seed`` are read
        here. Defaults to :class:`ForecastConfig`.
    latent
        The decision latent, dims ``(chain, draw, isin)``, on the STANDARDISED
        scale. **Pass the shrunk latent** when the caller has applied
        ``apply_forecast_error_shrinkage`` — resolving it again here would return
        the unshrunk one, so the book would be sized on a different quantity from
        the one the screen reports, and every gate would still pass. When ``None``
        the latent is resolved through
        :func:`~KalmanFilterModel_v2.resolve_screen_latent_v2`, which is the
        project's single name for this quantity.

    Returns
    -------
    ForecastInputs
        Posterior draws in log-return space, plus the group codes and the fitted
        OU length scale.

    Raises
    ------
    KeyError
        If ``sigma_isin`` is absent from the posterior — without a per-name scale
        there is no forward dispersion to simulate.
    """
    cfg = config or ForecastConfig()

    if latent is None:
        from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
            KALMAN_V2_SCREEN_LATENT,
            resolve_screen_latent_v2,
        )

        logger.info(
            "resolving the decision latent from the posterior; pass latent= if the "
            "caller has already applied forecast-error shrinkage"
        )
        latent = resolve_screen_latent_v2(
            idata, latent=KALMAN_V2_SCREEN_LATENT, random_seed=cfg.random_seed
        )

    latent_arr = np.asarray(latent)
    if latent_arr.ndim < 2:
        raise ValueError(
            f"latent must carry at least (sample, isin) axes, got shape {latent_arr.shape}"
        )
    # (sample, isin) -> (isin, sample), matching the posterior helpers' convention.
    mu_std = latent_arr.reshape(-1, latent_arr.shape[-1]).T

    response_std = float(getattr(panel, "response_std", 1.0))
    response_mean = float(getattr(panel, "response_mean", 0.0))

    mu_log = mu_std * response_std + response_mean
    sigma_log = _flat_draws(idata, "sigma_isin") * response_std

    if sigma_log.shape != mu_log.shape:
        raise ValueError(
            f"sigma_isin draws {sigma_log.shape} do not match the latent's "
            f"{mu_log.shape}. The two must come from the same fit."
        )

    try:
        nu = _flat_draws(idata, "nu", per_isin=False)
    except KeyError:
        logger.info("no 'nu' in the posterior; forward shocks will be Gaussian")
        nu = np.array([np.inf])

    group_index: dict[str, np.ndarray] = {}
    coord_idx = getattr(panel, "coord_idx", {}) or {}
    for level in cfg.factor_levels:
        codes = coord_idx.get(level)
        if codes is None:
            logger.warning(
                "factor level %r is not in panel.coord_idx (have: %s); it will carry "
                "no shared factor",
                level,
                sorted(coord_idx),
            )
            continue
        group_index[level] = np.asarray(codes, dtype="int64")

    ou_days: Optional[float] = None
    post = _posterior_group(idata)
    if "ou_length_scale_days" in post:
        ou_days = float(np.asarray(post["ou_length_scale_days"]).mean())

    isins = np.asarray(getattr(panel, "isins"))
    logger.info(
        "forecast inputs: %d names, %d posterior samples, %d factor level(s), "
        "OU length scale %s",
        len(isins),
        mu_log.shape[1],
        len(group_index),
        f"{ou_days:.1f}d" if ou_days is not None else "absent",
    )
    return ForecastInputs(
        isins=isins,
        mu_log=mu_log,
        sigma_log=sigma_log,
        nu=np.atleast_1d(np.asarray(nu, dtype="float64")),
        group_index=group_index,
        ou_length_scale_days=ou_days,
    )


# ---------------------------------------------------------------------------
# The native scenario generator
# ---------------------------------------------------------------------------


def _ou_step_correlation(step_days: float, length_scale_days: Optional[float]) -> float:
    """Lag-one correlation of the forward state over one step.

    The v2 likelihood correlates two observations of the same name at a calendar
    gap ``d`` by ``exp(-d / ell)`` (``KalmanFilterModel_v2._ou_correlation``), with
    ``ell`` the fitted ``ou_length_scale_days``. Carrying the *same* kernel forward
    is the whole point of this module: the AR simulator's ``rho = 0.85`` is a second
    persistence parameter, chosen rather than fitted, and unitless besides.

    Returns ``0.0`` when the posterior carries no length scale, which makes the
    forward path a plain random walk of independent steps.
    """
    if length_scale_days is None or not np.isfinite(length_scale_days):
        return 0.0
    if length_scale_days <= 0:
        return 0.0
    return float(np.exp(-float(step_days) / float(length_scale_days)))


def _cumulative_variance_factor(n_steps: int, phi: float) -> float:
    """``Var`` of the sum of ``n_steps`` unit-variance AR(1) steps with lag-1 ``phi``.

    ``V_n = n + 2 * sum_{k=1}^{n-1} (n - k) * phi**k``.

    Dividing the per-step shocks by ``sqrt(V_n)`` makes the *cumulative* horizon
    dispersion equal ``sigma_isin`` regardless of how persistent the state is, so
    changing the fitted length scale re-shapes the path without silently re-scaling
    the terminal distribution the decision layer reads.
    """
    n = int(n_steps)
    if n <= 1:
        return 1.0
    ks = np.arange(1, n, dtype="float64")
    return float(n + 2.0 * np.sum((n - ks) * np.power(float(phi), ks)))


def _standardised_shocks(
        rng: np.random.Generator,
        shape: tuple[int, ...],
        nu: np.ndarray,
) -> np.ndarray:
    """Unit-variance innovations, Student-t when ``nu`` is finite.

    ``Var[t_nu] = nu / (nu - 2)``, so a raw ``standard_t`` draw would inflate every
    downstream dispersion by that factor. Dividing it out keeps the tails heavy and
    the second moment where the caller asked for it. ``nu`` broadcasts against the
    scenario axis, which is axis ``-2`` of ``shape``.
    """
    nu_arr = np.atleast_1d(np.asarray(nu, dtype="float64"))
    if not np.all(np.isfinite(nu_arr)):
        return rng.standard_normal(size=shape)
    df = np.maximum(nu_arr, _MIN_NU)
    if df.size == 1:
        raw = rng.standard_t(df=float(df[0]), size=shape)
        return raw / np.sqrt(float(df[0]) / (float(df[0]) - 2.0))
    # One df per scenario: broadcast over the scenario axis.
    df_b = df.reshape((1,) * (len(shape) - 2) + (df.size, 1))
    raw = rng.standard_t(df=np.broadcast_to(df_b, shape))
    return raw / np.sqrt(df_b / (df_b - 2.0))


def _ar1_process(
        rng: np.random.Generator,
        shape: tuple[int, ...],
        phi: float,
        nu: np.ndarray,
) -> np.ndarray:
    """A stationary unit-variance AR(1) along the LAST axis.

    ``u_0 = e_0``; ``u_t = phi * u_{t-1} + sqrt(1 - phi**2) * e_t``. The
    ``sqrt(1 - phi**2)`` scaling is what keeps the marginal variance at 1 for every
    ``t`` rather than letting it grow with the path.
    """
    eps = _standardised_shocks(rng, shape, nu)
    if phi <= 0.0:
        return eps
    out = np.empty_like(eps)
    out[..., 0] = eps[..., 0]
    innov_scale = np.sqrt(max(1.0 - phi * phi, 0.0))
    for t in range(1, shape[-1]):
        out[..., t] = phi * out[..., t - 1] + innov_scale * eps[..., t]
    return out


@dataclass(frozen=True)
class _FactorBlocks:
    """Shared factor paths and the map from each name to its row in each block.

    Attributes
    ----------
    paths
        One ``(width, n_scenarios, n_steps)`` array per block, ``width`` being the
        number of distinct values of that level (1 for the market factor).
    labels
        Block names, aligned to :attr:`paths`.
    codes
        ``(n_blocks, n_isin)`` row index of each name within its block.
    weight
        ``sqrt(factor_share / n_blocks)`` — the loading each block carries.
    idio_weight
        ``sqrt(1 - factor_share)`` — the idiosyncratic loading.
    """

    paths: tuple[np.ndarray, ...]
    labels: tuple[str, ...]
    codes: np.ndarray
    weight: float
    idio_weight: float

    @property
    def n_blocks(self) -> int:
        return len(self.paths)

    def summary(self) -> Optional[np.ndarray]:
        """``(n_blocks, n_scenarios, n_steps)`` block means, for attribution.

        Averaged to one row per block because storing every level value of a
        100-level coordinate would exceed the scenario array itself.
        """
        if not self.paths:
            return None
        return np.stack([p.mean(axis=0) for p in self.paths], axis=0)


def _common_factor_shocks(
        rng: np.random.Generator,
        inputs: ForecastInputs,
        config: ForecastConfig,
        *,
        phi: float,
        nu_sel: np.ndarray,
) -> _FactorBlocks:
    """Build the shared factors and each name's loading onto them.

    The decomposition is

    ``u[i, s, t] = sqrt(c) * f_market[s, t]
                 + sum_g sqrt(c) * f_g[level_of(i), s, t]
                 + sqrt(1 - n_blocks * c) * z[i, s, t]``

    with every component a unit-variance AR(1) and ``c = factor_share / n_blocks``.
    Because the components are independent and their variance shares sum to one,
    ``u[i, s, t]`` has unit variance **for any** ``factor_share``. That is the
    invariance the whole design rests on: switching the factor structure on changes
    which names move together and leaves every per-name marginal — hence ``er_sd``,
    hence ``exp_vol`` — untouched.

    Returns
    -------
    _FactorBlocks
        Empty (``n_blocks == 0``, ``idio_weight == 1``) when no factor structure is
        requested or none could be resolved, which reproduces the AR simulator's
        cross-sectionally independent shocks exactly.
    """
    n_scen = config.n_scenarios
    n_steps = config.n_steps

    labels: list[str] = []
    level_codes: list[np.ndarray] = []
    if config.n_market_factors:
        labels.append("market")
        level_codes.append(np.zeros(inputs.n_isin, dtype="int64"))
    for level in config.factor_levels:
        codes = inputs.group_index.get(level)
        if codes is None:
            continue
        labels.append(level)
        level_codes.append(codes)

    n_blocks = len(labels)
    empty = _FactorBlocks(
        paths=(),
        labels=(),
        codes=np.empty((0, inputs.n_isin), dtype="int64"),
        weight=0.0,
        idio_weight=1.0,
    )
    if n_blocks == 0 or config.factor_share <= 0.0:
        if config.factor_share > 0.0:
            logger.warning(
                "factor_share=%.3f requested but no factor block resolved; the "
                "forward shocks will be cross-sectionally independent",
                config.factor_share,
            )
        return empty

    share = config.factor_share / n_blocks
    # ``1 - n_blocks * share`` is exactly ``1 - factor_share``; written out so the
    # variance budget is visible at the point it is spent.
    idio_weight = float(np.sqrt(max(1.0 - n_blocks * share, 0.0)))

    # One factor per level VALUE, so names in the same sector share a path. The
    # per-block arrays are ragged in width, so they are drawn separately and then
    # gathered onto a common (n_blocks, n_scen, n_steps) view at combine time.
    factors: list[np.ndarray] = []
    widths: list[int] = []
    for codes in level_codes:
        width = int(codes.max()) + 1 if codes.size else 1
        widths.append(width)
        factors.append(_ar1_process(rng, (width, n_scen, n_steps), phi, nu_sel))

    logger.info(
        "factor blocks: %s (widths %s), common share %.3f split %.4f each, "
        "idiosyncratic weight %.4f",
        labels,
        widths,
        config.factor_share,
        share,
        idio_weight,
    )
    return _FactorBlocks(
        paths=tuple(factors),
        labels=tuple(labels),
        codes=np.stack(level_codes, axis=0),
        weight=float(np.sqrt(share)),
        idio_weight=idio_weight,
    )


def simulate_forecast(
        inputs: ForecastInputs,
        config: Optional[ForecastConfig] = None,
        *,
        chunk_size: int = 512,
) -> ForecastDraws:
    """Draw joint forward-return scenarios from the posterior.

    The generative form, per name ``i``, scenario ``s`` and step ``t``:

    .. code-block:: text

        log r[i, s, t] = mu_log[i, s] * w[t]  +  sigma_log[i, s] / sqrt(V_n) * u[i, s, t]

        u[i, s, t]     = sqrt(c) * f_market[s, t]
                       + sum_g sqrt(c) * f_g[level(i), s, t]
                       + sqrt(1 - factor_share) * z[i, s, t]

    with ``w[t]`` the step's share of the horizon (summing to 1), every component a
    unit-variance AR(1) whose lag-one correlation is the fitted OU kernel evaluated
    at one step, and ``V_n`` the cumulative-variance factor of that AR(1).

    Three properties follow, and each is worth stating because each is a departure
    from the AR simulator this contrasts against:

    * ``E[sum_t log r] = mu_log`` exactly — the expected total log uplift over the
      horizon is the model's own decision latent, not a quantity that drifts with a
      chosen persistence.
    * ``sd[sum_t log r | s] = sigma_log[i, s]`` exactly, **conditional on a posterior
      draw**, whatever the length scale does to the path's shape. Marginally the
      terminal spread is wider than ``sigma_log``, because it also carries the
      posterior spread of ``mu_log`` across draws. That is intended and is the point
      of scenario-per-draw sampling: parameter uncertainty reaches the decision
      instead of being averaged away first. Do not read the marginal terminal sd as
      an estimate of ``sigma_isin``.
    * every per-name marginal is invariant to ``factor_share``, so the shared
      structure changes only the joint distribution.

    The clip is applied to each step and again to the cumulative total, in LOG space
    so it is sign-preserving. Clipping after converting to simple returns would
    distort ``prob_pos``, which is the reason the workflow clips in log space too.

    Parameters
    ----------
    inputs
        From :func:`prepare_forecast_inputs`.
    config
        Defaults to :class:`ForecastConfig`.
    chunk_size
        Names simulated per block. The result is
        ``n_isin * n_scenarios * n_steps`` float64 — 417 MB at 6,511 names, 2,000
        scenarios and 4 steps — so the intermediates are the thing worth bounding.

    Returns
    -------
    ForecastDraws
        Scenarios labelled by ISIN.
    """
    cfg = config or ForecastConfig()
    if cfg.backend != "native":
        raise ValueError(
            f"simulate_forecast is the native generator; got backend={cfg.backend!r}. "
            "Use forecast_from_posterior, which dispatches on the backend."
        )

    rng = np.random.default_rng(cfg.random_seed)
    n_isin = inputs.n_isin
    n_scen = cfg.n_scenarios
    n_steps = cfg.n_steps

    # One posterior sample per scenario, SHARED across every name. This is what
    # makes parameter uncertainty a common component rather than per-name noise,
    # and it is the only cross-sectional dependence the AR simulator has.
    replace = n_scen > inputs.n_samples
    if replace:
        logger.warning(
            "n_scenarios=%d exceeds the %d posterior samples; sampling with "
            "replacement, so scenarios are not distinct parameter draws",
            n_scen,
            inputs.n_samples,
        )
    sample_idx = rng.choice(inputs.n_samples, size=n_scen, replace=replace)

    mu_sel = np.ascontiguousarray(inputs.mu_log[:, sample_idx])        # (n_isin, n_scen)
    sigma_sel = np.ascontiguousarray(inputs.sigma_log[:, sample_idx])  # (n_isin, n_scen)
    nu_sel = (
        inputs.nu[sample_idx] if inputs.nu.size == inputs.n_samples else inputs.nu
    )

    phi = _ou_step_correlation(cfg.step_days, inputs.ou_length_scale_days)
    var_factor = _cumulative_variance_factor(n_steps, phi)
    step_scale = 1.0 / np.sqrt(var_factor)
    weights = cfg.step_fractions.astype("float64")  # (n_steps,), sums to 1

    logger.info(
        "simulating %d names x %d scenarios x %d steps; OU lag-1 phi=%.4f over %dd, "
        "cumulative variance factor %.4f",
        n_isin,
        n_scen,
        n_steps,
        phi,
        cfg.step_days,
        var_factor,
    )

    blocks = _common_factor_shocks(rng, inputs, cfg, phi=phi, nu_sel=nu_sel)

    log_lo, log_hi = cfg.log_clip
    paths = np.empty((n_isin, n_scen, n_steps), dtype="float64")

    for start in range(0, n_isin, chunk_size):
        stop = min(start + chunk_size, n_isin)
        m = stop - start

        u = blocks.idio_weight * _ar1_process(rng, (m, n_scen, n_steps), phi, nu_sel)
        for b, block_path in enumerate(blocks.paths):
            codes = blocks.codes[b, start:stop]
            u += blocks.weight * block_path[codes]

        chunk = mu_sel[start:stop, :, None] * weights[None, None, :]
        chunk += (sigma_sel[start:stop, :, None] * step_scale) * u
        np.clip(chunk, log_lo, log_hi, out=chunk)
        paths[start:stop] = chunk

    # Terminal FIRST, from the log paths, then convert the steps in place. The other
    # order would need a second full-size array.
    terminal_log = paths.sum(axis=2)
    np.clip(terminal_log, log_lo, log_hi, out=terminal_log)
    terminal = np.expm1(terminal_log)
    np.expm1(paths, out=paths)

    return ForecastDraws(
        isins=np.asarray(inputs.isins),
        paths=paths,
        terminal=terminal,
        time_days=cfg.time_days,
        backend="native",
        factor_share=float(cfg.factor_share) if blocks.n_blocks else 0.0,
        factor_draws=blocks.summary(),
        factor_labels=blocks.labels,
    )


def forecast_from_posterior(
        idata: "InferenceLike",
        panel: Any,
        *,
        config: Optional[ForecastConfig] = None,
        latent: Optional[Any] = None,
) -> ForecastDraws:
    """Prepare inputs and simulate, dispatching on ``config.backend``.

    This is the entry point the workflow calls. See
    :func:`prepare_forecast_inputs` for why ``latent`` should be passed when the
    caller has already applied forecast-error shrinkage.

    Raises
    ------
    NotImplementedError
        For the ``pymc_forecast`` and ``statespace`` backends, which are declared in
        :data:`FORECAST_BACKENDS` and not yet built. The message names the missing
        dependency where that is the reason.
    """
    cfg = config or ForecastConfig()
    inputs = prepare_forecast_inputs(idata, panel, config=cfg, latent=latent)

    if cfg.backend == "native":
        return simulate_forecast(inputs, cfg)

    if cfg.backend == "pymc_forecast":
        if _pmf is None:
            raise NotImplementedError(
                "backend='pymc_forecast' needs the optional pymc-forecast package. "
                "It carries no upper pin on pymc or arviz, so it installs alongside "
                "the current stack."
            )
        raise NotImplementedError(
            "the pymc_forecast backend is not built yet; use backend='native'."
        )

    if cfg.backend == "statespace":
        if _PyMCStateSpace is None:
            raise NotImplementedError(
                "backend='statespace' needs pymc-extras, which pins pymc<6.3 and "
                "pytensor<3.3. Installing it against the current stack would "
                "DOWNGRADE both, so it is deliberately not a dependency. Use "
                "backend='native' until pymc-extras raises its cap."
            )
        raise NotImplementedError(
            "the statespace backend is not built yet; use backend='native'."
        )

    raise ValueError(f"Unknown backend {cfg.backend!r}")  # pragma: no cover


def summarize_forecast(
        draws: ForecastDraws,
        *,
        quantiles: Sequence[float] = (0.05, 0.50, 0.95),
        terminal: bool = False,
) -> "pd.DataFrame":
    """Per-ISIN summary of the forward returns, in the exported ``er_*`` layout.

    Parameters
    ----------
    draws
        From :func:`simulate_forecast`.
    quantiles
        Reported as ``er_p05`` / ``er_p50`` / ``er_p95`` at the defaults, named by
        percentage exactly as ``_price_target_mc.summarize_mc_returns`` does.
    terminal
        ``False`` (default) pools the per-step marginals, reproducing
        ``summarize_mc_returns`` so the numbers are comparable with the shipped
        export and so ``er_sd`` stays the quantity ``exp_vol`` asserts against.
        ``True`` summarises the *cumulative horizon* return instead, which is the
        decision quantity. The two differ and the column names do not say which, so
        a frame built with ``terminal=True`` must record that it was.

    Returns
    -------
    pandas.DataFrame
        ``isin``, ``er_mean``, ``er_sd``, the quantile columns, ``prob_pos``.
    """
    import pandas as pd

    vals = draws.terminal if terminal else draws.pooled_returns
    out = {
        "isin": np.asarray(draws.isins),
        "er_mean": vals.mean(axis=1),
        "er_sd": vals.std(axis=1),
    }
    for q in quantiles:
        out[f"er_p{int(round(q * 100)):02d}"] = np.quantile(vals, q, axis=1)
    out["prob_pos"] = (vals > 0.0).mean(axis=1)
    return pd.DataFrame(out)
