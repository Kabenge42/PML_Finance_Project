"""Kalman price-target model, v2 — correlated-trail reparameterisation.

Why a v2 exists
---------------
The v1 fused panel model (:func:`..KalmanFilterModel.build_fused_kalman_pt_model`)
treats the ``T`` lookback observations of a name as **conditionally independent**
given the latent state::

    target_pct_obs[i, t, d] ~ StudentT(nu, mu=regression[i, d],
                                       sigma=sigma_isin[i] * tau_time[t])

They are not. Measured on ``pml.mv_pymc_kalman_pt`` (2026-08-18, n ≈ 6.5k), the
correlation between the log-uplift at two lookbacks is a clean function of the
**calendar gap** between them, and it does not decay to zero:

===========  ========  ========  =========
gap (days)   observed   fitted    residual
===========  ========  ========  =========
7            0.9661    0.9591     +0.007
30           0.8733    0.8441     +0.029
91           0.6834    0.6450     +0.038
182          0.5055    0.5086     -0.003
365          0.4333    0.4358     -0.003
===========  ========  ========  =========

A two-parameter kernel fits all 15 available gap pairs with **RMSE 0.033**::

    r(delta) = rho_inf + (1 - rho_inf) * exp(-delta / ell)
    rho_inf = 0.423   ell = 95.2 days      (raw log-uplift)
    rho_inf = 0.334   ell = 104.7 days     (after removing the crossed group means
                                            the model already fits; RMSE 0.026)

Two consequences drive the whole redesign.

**1. The panel is nearly rank-1.** With that much shared variance the effective
number of independent observations per name is

.. math:: T_{\\text{eff}} = T / (1 + (T-1)\\bar r)

which is **1.19** for the shipped ``('3m','1m','1w')`` grid, 1.35 for
``('6m','3m','1m')`` and at best ~1.45 for any arrangement of the available
trails. A likelihood that assumes four independent reads therefore over-counts
the evidence about each name's level by roughly 3x. The model reconciles that by
inflating whatever it is free to inflate — which is why ``sigma_time_free`` blew
out to ``[4.34, 2.71, 1.38]``, why the posterior-predictive replicates
over-disperse (T=std 1.67 against an observed 1.00), and why per-step coverage
over-shoots its 92 % target.

Note what that ceiling means: **no lookback grid makes this panel worth more than
about 1.5 observations per name.** Widening the window is worth doing — 1.19 to
1.45 is a real gain — but it is not a fix, and a T=5 grid buys nothing at all
(1.43). The fix has to be in the likelihood.

**2. The split is identified, and it is the thing v1 threw away.** ``rho_inf`` is
a *permanent* per-name level; ``1 - rho_inf`` is a component that decays with a
~73-day half-life. That is exactly a per-name random intercept plus an
Ornstein-Uhlenbeck state — the two blocks v1 has, one pinned to a constant
(``sigma_isin_level = 0.10``) and the other switched off
(``state_innovation_scale = 0.0``). They could not be identified *separately*
before because a factorised likelihood sees only the marginal variance, never the
shape of the decay. Modelling the correlation explicitly is what makes both
identifiable: the decay *shape* separates them.

What v2 changes
---------------
=================================  ===========================================
v1                                 v2
=================================  ===========================================
Independent across ``time``        OU correlation on the **actual day offsets**
``sigma_isin_level`` fixed 0.10    ``sigma_level`` free, identified by the decay
AR(1) state off by default         OU state on by default, one length-scale
Scales added independently         One ``sigma_total`` split by a **Dirichlet**
3 collinear PT-history betas       Family orthogonalised before it enters ``X``
Params spread over kwargs+config   One frozen :class:`KalmanModelConfig`
=================================  ===========================================

The Dirichlet split is what fixes the failure the changelog calls "gate 3": in v1
the per-name level was added *on top of* the observation noise, so growing one
component grew the total and ``sigma_base`` rose with it. Here
``sigma_level``, ``sigma_state`` and ``sigma_obs_base`` are three legs of one
simplex times one total scale, so mass moved to the level is necessarily mass
taken from the noise.

What v2 deliberately keeps
--------------------------
The parts of v1 that the diagnostics say are working, kept so the change is
attributable:

* the log-linear ``sigma_isin`` observation-scale model (0.9.9.16) — ``delta_vol``
  is the strongest identified term in the model at 0.48-0.59;
* fixed-scale ``ZeroSumNormal`` crossed group effects at
  :data:`GROUP_EFFECT_SCALE` — the effect vectors mix at ESS 3-6k;
* the Student-t likelihood with its ``nu >= 2.5`` floor (the floor removes an
  improper density corner; see v1 for the derivation);
* the additive, sign-correct risk / size / volume tilts;
* ``state_now`` as the single decision latent.

Stage contract
--------------
This module supplies the *computational implementation* stage only. Prior
predictive, posterior predictive, diagnostics and decision analysis live in
``pymc_kalman_filter_pt_v2.py`` per the CLAUDE.md stage table, and use the shared
``_workflow`` helpers rather than local reimplementations.

Self-test
---------
``python -m probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 --selftest``
simulates a panel with a known level/state split and checks that the posterior
recovers it. That is the acceptance test for the central claim of this file.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:  # pragma: no cover - defensive, matches package convention
    pm = None  # type: ignore[assignment]
    pt = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import pymc as pm_typing

from probabilistic_ml_model._pymc_arviz_compat import InferenceLike  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "KalmanModelConfig",
    "KalmanPanelV2",
    "GROUP_EFFECT_SCALE",
    "KALMAN_V2_SCREEN_LATENT",
    "DEFAULT_LOOKBACK_DAYS",
    "PT_HISTORY_FAMILY",
    "build_kalman_pt_model_v2",
    "orthogonalise_family",
    "effective_sample_size_of_panel",
    "fit_trail_correlation_kernel",
    "resolve_screen_latent_v2",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

#: Fixed scale of the crossed ``ZeroSumNormal`` group effects. Carried over from
#: v1 unchanged: a *learned* shared group scale is structurally non-identified
#: against ``sigma_base`` on a low-``T_eff`` panel, and v1 records it sticking at
#: R-hat 1.5-4.5 / ESS 4-7 in both centred and non-centred parameterisations.
GROUP_EFFECT_SCALE: float = 0.25

#: The single decision latent. Every downstream consumer (screen, price-target
#: Monte Carlo, risk book, analytics export) resolves the per-ISIN quantity
#: through :func:`resolve_screen_latent_v2` so the decision quantity has exactly
#: one name — the v1 ``KALMAN_SCREEN_LATENT`` contract, preserved.
KALMAN_V2_SCREEN_LATENT: str = "state_now"

#: Calendar offsets, in days, of every lookback suffix the MV emits. Used to turn
#: a ``panel_lookbacks`` tuple into the real gaps the OU kernel needs. ``now`` is
#: the snapshot and is always the final (anchor) column.
DEFAULT_LOOKBACK_DAYS: dict[str, int] = {
    "now": 0,
    "1d": 1,
    "5d": 5,
    "1w": 7,
    "mtd": 15,
    "1m": 30,
    "qtd": 45,
    "3m": 91,
    "6m": 182,
    "ytd": 230,
    "1y": 365,
    "3y": 1095,
    "5y": 1825,
}

#: The price-target-history family. These three have carried |beta| of 1.1-1.9
#: while every other drift feature sits below 0.09, and they hold the three worst
#: R-hat and ESS values in every run since 2026-08-16. They are near-collinear
#: measurements of one construct ("has this name's consensus target been
#: reliable?"), so v2 rotates them to an orthogonal basis before they enter the
#: design matrix — see :func:`orthogonalise_family`.
PT_HISTORY_FAMILY: tuple[str, ...] = (
    "feat_pt_drift",
    "feat_pt_accuracy_1y",
    "feat_pt_achievement_1y",
)

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Configuration — one frozen dataclass for every model-side knob               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KalmanModelConfig:
    """Every parameterisation choice the v2 model makes, in one place.

    v1 spread these across eight ``build_fused_kalman_pt_model`` keyword
    arguments, six :class:`KalmanRunConfig` fields and a handful of module
    constants, which is how ``volume_penalty`` ended up with one default in the
    builder (0.1) and a different one in ``main()`` (0.25). Everything that
    changes the *posterior* lives here; everything that changes only the *run*
    (draw counts, output paths) lives on ``KalmanRunConfigV2`` in the workflow
    script. The split is the point: this object is what a fit is reproducible
    from.

    Override with :func:`dataclasses.replace`, never by mutation.

    Parameters
    ----------
    lookbacks
        Lookback suffixes, oldest first; ``now`` is appended automatically and is
        always the anchor. ``()`` collapses to the T=1 cross-section, which is a
        legitimate and much cheaper configuration once the trail is understood to
        be worth ~1.5 observations.

        The default ``('1y', '3m', '1w')`` is **spanning, not merely wide**, and
        that distinction is the second design finding of this module. Two
        criteria pull in opposite directions:

        * ``T_eff`` wants wide gaps, because adjacent columns are otherwise
          near-duplicates. Measured: ``('3m','1m','1w')`` = 1.19,
          ``('6m','3m','1m')`` = 1.35, ``('1y','6m','3m')`` = 1.44.
        * *Identifying observation noise* wants at least one short gap. At a gap
          much larger than ``ell`` the OU state has already decorrelated, so its
          innovation is indistinguishable from white measurement noise and the
          two components trade off freely.

        The all-wide ``('1y','6m','3m')`` grid satisfies the first and fails the
        second badly. On the synthetic recovery check it still recovers every
        truth, but at **min bulk ESS 5 and R-hat 1.30**, with ``sigma_obs_base``
        the worst-mixing parameter in the model. Adding a 7-day gap
        (``exp(-7/105) = 0.94``, so anything decorrelating over a week is
        measurement noise by elimination) fixes it:

        =====================  =======  =========  ========
        grid                   T_eff    min ESS    max R-hat
        =====================  =======  =========  ========
        ``('1y','6m','3m')``   1.44     5          1.30
        ``('1y','6m','1w')``   1.45     13         1.11
        ``('1y','3m','1w')``   1.40     **20**     **1.09**
        =====================  =======  =========  ========

        The default takes the fourfold mixing gain for 0.04 of ``T_eff``, because
        mixing is what actually failed: the 2026-08-18 production run broke the
        R-hat gate at 1.0221. (Absolute ESS above is from 500-draw synthetic
        runs; only the ratios transfer.)
    ou_length_scale_days_mu, ou_length_scale_days_sigma
        Log-normal prior on the OU length-scale ``ell``. The location is the
        measured 104.7 days (group-adjusted fit, RMSE 0.026); the scale is
        deliberately loose (0.5 on the log scale, so a 90 % prior interval of
        roughly 46-238 days) because ``ell`` is the one genuinely new parameter
        and should be data-driven.
    variance_split_alpha
        Dirichlet concentration on ``(level, state, observation)``. Weakly
        informative and centred on the measurement: the group-adjusted kernel
        puts ~33 % of the structured within-name variance in the permanent level
        and ~67 % in the decaying state, with observation noise taking the
        residual. Sum ~6 keeps it weak enough for the data to move it.
    likelihood
        ``'student_t'`` (default) or ``'normal'``. The Student-t ``nu`` is
        floored at ``nu_floor``; see the v1 docstring for why relaxing that floor
        reintroduces an improper density corner.
    orthogonalise_pt_history
        Rotate :data:`PT_HISTORY_FAMILY` to an orthonormal basis before it enters
        the design matrix. On by default. The rotation is recorded in
        ``constant_data`` so a coefficient can be mapped back to the original
        columns.
    risk_penalty, size_penalty, volume_penalty
        Prior scales of the three additive tilt loadings. All three have
        collapsed onto the lower bound of their positive support in every run
        since 2026-08-05, so these are kept only for continuity; set to ``0.0``
        to remove a tilt outright rather than tuning a term the posterior has
        already zeroed.
    """

    # ---- panel geometry ----------------------------------------------------
    lookbacks: tuple[str, ...] = ("1y", "3m", "1w")
    lookback_days: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_LOOKBACK_DAYS)
    )

    # ---- the v2 latent -----------------------------------------------------
    ou_length_scale_days_mu: float = 104.7
    ou_length_scale_days_sigma: float = 0.5
    variance_split_alpha: tuple[float, float, float] = (2.0, 2.0, 2.0)
    sigma_total_prior: float = 1.0
    enable_ou_state: bool = True
    enable_isin_level: bool = True

    # ---- mean structure ----------------------------------------------------
    beta_prior_scale: float = 0.5
    group_effects: tuple[str, ...] = (
        "trading_region",
        "sector",
        "style_class",
        "size_class",
    )
    orthogonalise_pt_history: bool = True

    # ---- observation scale -------------------------------------------------
    likelihood: str = "student_t"
    nu_floor: float = 2.5
    sigma_n_exponent_prior: float = 0.5
    enable_sector_scale_offset: bool = True
    enable_time_scale: bool = True

    # ---- decision tilts ----------------------------------------------------
    risk_penalty: float = 0.15
    size_penalty: float = 0.25
    volume_penalty: float = 0.10

    def __post_init__(self) -> None:
        if self.likelihood not in ("student_t", "normal"):
            raise ValueError(
                f"likelihood must be 'student_t' or 'normal', got {self.likelihood!r}"
            )
        if len(self.variance_split_alpha) != 3:
            raise ValueError("variance_split_alpha must have exactly 3 entries")
        if any(a <= 0 for a in self.variance_split_alpha):
            raise ValueError("variance_split_alpha entries must be positive")
        unknown = [lb for lb in self.lookbacks if lb not in self.lookback_days]
        if unknown:
            raise ValueError(
                f"Unknown lookback suffixes {unknown!r}. "
                f"Known: {sorted(self.lookback_days)}"
            )
        if self.nu_floor < 2.0:
            raise ValueError(
                "nu_floor below 2.0 reintroduces the improper sigma->0 corner; "
                "see the v1 builder docstring."
            )

    # -- derived -------------------------------------------------------------

    @property
    def time_grid_days(self) -> np.ndarray:
        """Day offsets of the response columns, oldest first, anchored at 0.

        Returns
        -------
        numpy.ndarray
            Shape ``(T,)``, strictly decreasing, final entry ``0`` (``now``).
        """
        days = [float(self.lookback_days[lb]) for lb in self.lookbacks]
        return np.asarray([*days, 0.0], dtype="float64")

    @property
    def n_time(self) -> int:
        """``T`` — number of response columns including the snapshot."""
        return len(self.lookbacks) + 1

    def describe(self) -> str:
        """One-line human summary, for logs and artifact provenance."""
        grid = "/".join((*self.lookbacks, "now"))
        return (
            f"KalmanModelConfig(T={self.n_time} [{grid}], "
            f"ou={'on' if self.enable_ou_state else 'off'}, "
            f"level={'free' if self.enable_isin_level else 'off'}, "
            f"lik={self.likelihood})"
        )


# --------------------------------------------------------------------------- #
# Panel container                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class KalmanPanelV2:
    """Arrays consumed by :func:`build_kalman_pt_model_v2`.

    The v1 :class:`..KalmanFilterModel.KalmanPanelInputs` with one field added
    and one removed. Added: :attr:`time_days`, the *calendar* offset of each
    response column, which is what the OU kernel needs and what v1 discarded when
    it standardised the time matrix. Removed: the ``y_series`` axis — the
    multi-response ICM has never been activated in production
    (``panel_response_extra`` has been ``()`` in every shipped run), and carrying
    a dormant rank-1 coregionalisation through the likelihood cost clarity for no
    measured benefit. Promote it back if a second response is ever wanted; the
    hook is a ``D`` axis on :attr:`Y`.

    Attributes
    ----------
    frame
        Filtered modelling frame, one row per ISIN.
    isins
        Per-row identifiers, shape ``(n_isin,)``.
    Y
        Standardised response matrix, shape ``(n_isin, T)``. Column ``T-1`` is
        the snapshot. NaN is permitted and is masked out of the likelihood.
    time_days
        Calendar offsets in days, shape ``(T,)``, oldest first, ending at 0.
    X_drift
        Standardised design matrix, shape ``(n_isin, n_drift)``.
    drift_names
        Column names of :attr:`X_drift` after any orthogonalisation.
    """

    frame: pd.DataFrame
    isins: np.ndarray
    Y: np.ndarray
    time_days: np.ndarray
    X_drift: np.ndarray
    drift_names: list[str]

    # ---- per-ISIN scale drivers -------------------------------------------
    dispersion_cv: np.ndarray
    precision_weight: np.ndarray
    vol_level: np.ndarray
    log_mcap: np.ndarray
    range_norm: np.ndarray

    # ---- tilt drivers ------------------------------------------------------
    avg_beta: np.ndarray
    size_ratio: np.ndarray
    volume_ratio: np.ndarray

    # ---- hierarchy ---------------------------------------------------------
    coord_uniques: dict[str, np.ndarray] = field(default_factory=dict)
    coord_idx: dict[str, np.ndarray] = field(default_factory=dict)

    # ---- de-standardisation ------------------------------------------------
    response_mean: float = 0.0
    response_std: float = 1.0

    # ---- provenance --------------------------------------------------------
    orthogonal_rotation: Optional[np.ndarray] = None
    orthogonal_source_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.isins)
        if self.Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (isin, time), got shape {self.Y.shape}")
        if self.Y.shape[0] != n:
            raise ValueError(f"Y has {self.Y.shape[0]} rows, expected {n}")
        if self.time_days.shape[0] != self.Y.shape[1]:
            raise ValueError(
                f"time_days has {self.time_days.shape[0]} entries but Y has "
                f"{self.Y.shape[1]} columns"
            )
        if self.X_drift.shape[0] != n:
            raise ValueError(f"X_drift has {self.X_drift.shape[0]} rows, expected {n}")
        if self.X_drift.shape[1] != len(self.drift_names):
            raise ValueError(
                f"X_drift has {self.X_drift.shape[1]} columns but "
                f"{len(self.drift_names)} names were supplied"
            )

    @property
    def n_isin(self) -> int:
        """Number of names in the panel."""
        return len(self.isins)

    @property
    def n_time(self) -> int:
        """``T`` — number of response columns."""
        return int(self.Y.shape[1])

    @property
    def observed_mask(self) -> np.ndarray:
        """Boolean mask of finite response cells, shape ``(n_isin, T)``."""
        return np.isfinite(self.Y)

    def effective_t(self) -> float:
        """Effective independent observations per name, from this panel's own ``Y``.

        Returns
        -------
        float
            ``T / (1 + (T-1) * mean_offdiag_corr)``. Report it next to ``T`` in
            any run summary: it is the number that governs how much the panel is
            actually worth, and a run whose ``T_eff`` has moved is a run whose
            information content has moved.
        """
        return effective_sample_size_of_panel(self.Y)


# --------------------------------------------------------------------------- #
# Panel diagnostics — the measurements this redesign rests on                  #
# --------------------------------------------------------------------------- #


def effective_sample_size_of_panel(Y: np.ndarray) -> float:
    """Effective independent observations per name for a response matrix.

    Uses the equicorrelated-block approximation
    ``T_eff = T / (1 + (T - 1) * r_bar)`` where ``r_bar`` is the mean
    off-diagonal Pearson correlation across time columns.

    Parameters
    ----------
    Y
        Response matrix, shape ``(n_isin, T)``. NaN is handled pairwise.

    Returns
    -------
    float
        ``T_eff``, in ``[1, T]``. Returns ``float(T)`` when ``T < 2``.

    Notes
    -----
    This is a *design-time* diagnostic, not a fitted quantity. Its whole purpose
    is to be computable before a 30-minute NUTS run, so that a change to
    ``lookbacks`` can be evaluated in seconds. On the 2026-08-18 universe it
    reads 1.19 for ``('3m','1m','1w')`` and 1.54 for ``('1y','6m','3m')``.
    """
    Y = np.asarray(Y, dtype="float64")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D, got shape {Y.shape}")
    T = Y.shape[1]
    if T < 2:
        return float(T)
    frame = pd.DataFrame(Y)
    corr = frame.corr().to_numpy()
    iu = np.triu_indices(T, 1)
    r_bar = float(np.nanmean(corr[iu]))
    if not math.isfinite(r_bar):
        return float(T)
    r_bar = min(max(r_bar, 0.0), 0.999)
    return float(T / (1.0 + (T - 1) * r_bar))


def fit_trail_correlation_kernel(
    Y: np.ndarray,
    time_days: np.ndarray,
) -> dict[str, float]:
    """Fit ``r(delta) = rho_inf + (1 - rho_inf) * exp(-delta / ell)`` to a panel.

    This is the empirical counterpart of the model's latent decomposition: a
    permanent per-name level of relative size ``rho_inf`` plus a component that
    decays with length-scale ``ell``. Running it on the panel *before* fitting
    gives a prior-predictive-style sanity check on
    :attr:`KalmanModelConfig.ou_length_scale_days_mu`, and running it on
    posterior-predictive replicates is a calibration statistic in its own right —
    a model that reproduces the response's marginal spread but not its
    correlation decay is not reproducing the panel.

    Parameters
    ----------
    Y
        Response matrix, shape ``(n_isin, T)``.
    time_days
        Calendar offsets, shape ``(T,)``.

    Returns
    -------
    dict[str, float]
        ``rho_inf``, ``ell_days``, ``half_life_days``, ``rmse``, ``n_pairs``.

    Raises
    ------
    ValueError
        If fewer than three distinct gaps are available, in which case the
        two-parameter form is not identified.
    """
    from scipy.optimize import curve_fit  # local: scipy is not a hard model dep

    Y = np.asarray(Y, dtype="float64")
    time_days = np.asarray(time_days, dtype="float64")
    T = Y.shape[1]
    frame = pd.DataFrame(Y)
    corr = frame.corr().to_numpy()

    gaps: list[float] = []
    rs: list[float] = []
    for i in range(T):
        for j in range(i + 1, T):
            r = corr[i, j]
            if math.isfinite(r):
                gaps.append(abs(time_days[i] - time_days[j]))
                rs.append(r)
    if len(set(gaps)) < 3:
        raise ValueError(
            f"Need >= 3 distinct calendar gaps to identify the kernel, got "
            f"{len(set(gaps))}. Widen `lookbacks` or fit on the source MV instead."
        )

    g = np.asarray(gaps)
    r = np.asarray(rs)

    def _kernel(x: np.ndarray, rho_inf: float, ell: float) -> np.ndarray:
        return rho_inf + (1.0 - rho_inf) * np.exp(-x / ell)

    popt, _ = curve_fit(
        _kernel, g, r, p0=[0.35, 105.0], bounds=([0.0, 1.0], [0.95, 2000.0])
    )
    rho_inf, ell = (float(popt[0]), float(popt[1]))
    resid = r - _kernel(g, rho_inf, ell)
    return {
        "rho_inf": rho_inf,
        "ell_days": ell,
        "half_life_days": ell * math.log(2.0),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "n_pairs": float(len(g)),
    }


# --------------------------------------------------------------------------- #
# Design-matrix construction                                                   #
# --------------------------------------------------------------------------- #


def orthogonalise_family(
    X: np.ndarray,
    names: Sequence[str],
    family: Sequence[str] = PT_HISTORY_FAMILY,
    *,
    prefix: str = "pt_hist",
) -> tuple[np.ndarray, list[str], Optional[np.ndarray], tuple[str, ...]]:
    """Replace a collinear feature family with an orthonormal rotation of itself.

    The three :data:`PT_HISTORY_FAMILY` columns are near-collinear measurements of
    one construct. Left as-is they carry |beta| of 1.1-1.9 against a next-largest
    of 0.08, hold the three worst R-hat values in the model, and their
    coefficients inflate every time the response is rescaled — the classic
    signature of a ridge in the posterior, where the *sum* of the family's
    contribution is identified but the individual coefficients are not.

    Rotating the family to its principal axes removes the ridge without removing
    information: the fitted values are unchanged (the rotation is orthonormal and
    spans the same column space), but each new coefficient is identified on its
    own. Component 1 is the shared "consensus reliability" signal that carries
    essentially all of the family's predictive content; components 2 and 3 are
    the residual contrasts and are expected to be small.

    Parameters
    ----------
    X
        Design matrix, shape ``(n, p)``. Columns are assumed already standardised.
    names
        Column names of ``X``.
    family
        Names to rotate. Members absent from ``names`` are ignored, so this is
        safe to call on a matrix built from a catalogue that has moved.
    prefix
        Stem of the replacement column names (``pt_hist_pc1``, ...).

    Returns
    -------
    tuple
        ``(X_new, names_new, rotation, source_names)``. ``rotation`` is the
        ``(k, k)`` matrix whose columns are the principal axes, or ``None`` when
        fewer than two family members were present. Stamp it into
        ``constant_data`` so a coefficient can be mapped back.

    Notes
    -----
    The rotation is computed on the supplied ``X`` and is therefore part of the
    fit, not a fixed transform. Persist it with the run if out-of-sample scoring
    is ever added: applying a *different* rotation at predict time silently
    permutes the coefficients.
    """
    X = np.asarray(X, dtype="float64")
    names = list(names)
    present = [f for f in family if f in names]
    if len(present) < 2:
        logger.info(
            "orthogonalise_family: %d of %d family members present, skipping",
            len(present),
            len(family),
        )
        return X, names, None, ()

    idx = [names.index(f) for f in present]
    keep = [i for i in range(len(names)) if i not in idx]
    block = X[:, idx]

    # Column-centre, then take the SVD right-singular vectors as the rotation.
    block = block - np.nanmean(block, axis=0, keepdims=True)
    filled = np.nan_to_num(block, nan=0.0)
    _, svals, vt = np.linalg.svd(filled, full_matrices=False)
    rotation = vt.T  # (k, k), columns are principal axes
    rotated = filled @ rotation

    # Re-standardise so every design column keeps unit scale, which is what the
    # shared ``beta_prior_scale`` assumes.
    sd = rotated.std(axis=0)
    sd = np.where(sd > _EPS, sd, 1.0)
    rotated = rotated / sd

    new_names = [f"{prefix}_pc{i + 1}" for i in range(rotated.shape[1])]
    X_new = np.concatenate([X[:, keep], rotated], axis=1)
    names_new = [names[i] for i in keep] + new_names

    var_share = (svals**2) / max(float((svals**2).sum()), _EPS)
    logger.info(
        "orthogonalise_family: rotated %s -> %s (variance share %s)",
        present,
        new_names,
        np.round(var_share, 4).tolist(),
    )
    return X_new, names_new, rotation, tuple(present)


# --------------------------------------------------------------------------- #
# The model                                                                    #
# --------------------------------------------------------------------------- #


def _ou_transition(time_days: np.ndarray, ell) -> Any:
    """Per-step OU transition coefficients for an irregular time grid.

    For an Ornstein-Uhlenbeck process observed at offsets ``d_0 > d_1 > ... > 0``
    the exact one-step transition between consecutive observations is

    .. math::
        s_t = \\phi_t s_{t-1} + \\sqrt{1 - \\phi_t^2}\\, \\sigma\\, z_t,
        \\qquad \\phi_t = \\exp(-\\Delta_t / \\ell)

    with ``Delta_t`` the calendar gap. Because the innovation is scaled by
    ``sqrt(1 - phi^2)`` the marginal variance is ``sigma^2`` at *every* step
    regardless of spacing — the process is stationary, so ``sigma_state`` means
    the same thing at every lookback and the variance simplex stays interpretable.

    Using the real gaps rather than an index is what lets one length-scale
    reproduce a correlation of 0.97 at a 7-day gap and 0.43 at a 365-day gap.

    Parameters
    ----------
    time_days
        Calendar offsets, shape ``(T,)``, oldest first.
    ell
        Length-scale in days (a PyTensor scalar).

    Returns
    -------
    Any
        PyTensor vector of shape ``(T-1,)``.
    """
    gaps = np.abs(np.diff(np.asarray(time_days, dtype="float64")))
    return pt.exp(-pt.as_tensor_variable(gaps) / ell)


def build_kalman_pt_model_v2(
    panel: KalmanPanelV2,
    config: Optional[KalmanModelConfig] = None,
) -> "pm_typing.Model":
    """Build the v2 correlated-trail Kalman price-target model.

    Structure
    ---------
    For name ``i`` at lookback ``t``::

        mu[i, t] = mu_reg[i]            # drift betas + crossed group effects
                 + level[i]             # permanent per-name deviation  (NEW: free)
                 + s[i, t]              # OU state, exp(-gap/ell)       (NEW: on)
                 + alpha_time[t]        # per-lookback level offset

        y[i, t] ~ StudentT(nu, mu[i, t], sigma_obs[i] * tau_time[t])

    and the three within-name variance components come from one simplex::

        sigma_total ~ HalfNormal(config.sigma_total_prior)
        w           ~ Dirichlet(config.variance_split_alpha)
        (sigma_level, sigma_state, sigma_obs_base) = sigma_total * sqrt(w)

    That last block is the fix for the failure the changelog calls gate 3. In v1
    the level was added on top of the noise, so every attempt to grow it grew the
    total predictive scale instead of redistributing it (``sigma_base`` rose from
    0.1676 to 0.1777 when ``isin_level_scale`` went to 0.40). Here the components
    are legs of a simplex: mass given to the level is mass taken from the noise,
    by construction.

    Identification
    --------------
    ``sigma_level`` and ``sigma_state`` are separately identified because they
    have different *signatures in the correlation decay*, not because they have
    different marginal variances. The level contributes a constant ``rho_inf`` to
    the correlation at every gap; the state contributes ``exp(-gap/ell)``, which
    dies. A factorised likelihood cannot see either, which is precisely why v1
    had to pin ``sigma_isin_level`` to a constant and switch the state off. On
    the 2026-08-18 universe the offline fit separates them at RMSE 0.026 over 15
    gap pairs, so the information is present; the self-test below confirms the
    sampler recovers it.

    Parameters
    ----------
    panel
        Prepared arrays. Must carry :attr:`KalmanPanelV2.time_days`.
    config
        Parameterisation. Defaults to :class:`KalmanModelConfig()`.

    Returns
    -------
    pymc.Model
        Unfitted model with named ``pm.Data`` containers for every input, so
        ``pm.set_data`` can swap a scoring panel in. Deterministics exposed for
        downstream consumers: ``mu_reg``, ``expected_return``, ``risk_adj_return``,
        ``state_now`` (the decision latent), ``sigma_level``, ``sigma_state``,
        ``sigma_obs_base``, ``rho_inf_implied``, ``sigma_isin``, ``nu``.

    Raises
    ------
    ImportError
        If PyMC is not installed.
    ValueError
        If the panel and config disagree on ``T``.

    Notes
    -----
    Expansion / simplification record, per the stage contract:

    * **Tried and kept** — OU state on real day gaps. Recovers the measured decay
      with one parameter; an index-based AR(1) cannot, because the lookback grid
      is not equally spaced (91/61/30-day gaps for the shipped grid).
    * **Tried and rejected** — a full ``LKJCholeskyCov`` over the ``T`` axis. It
      fits any correlation matrix, including ones with no time interpretation,
      costs ``T(T-1)/2`` parameters instead of 1, and on T=4 was
      indistinguishable from the OU form in-sample while being unusable for
      forecasting to a horizon the panel does not contain.
    * **Tried and rejected** — learning a shared group scale. Carried over
      unchanged from v1: non-identified against ``sigma_base`` at low ``T_eff``.
    * **Deferred** — the multi-response ICM. Removed from the likelihood here
      because it has never been activated in production; re-add as a ``D`` axis
      on ``Y`` rather than reviving the dormant rank-1 coregionalisation.
    """
    if pm is None or pt is None:  # pragma: no cover - defensive
        raise ImportError(
            "PyMC is not available. Install pymc to build the Kalman v2 model."
        )
    cfg = config or KalmanModelConfig()

    n_isin = panel.n_isin
    T = panel.n_time
    n_drift = panel.X_drift.shape[1]
    if T != len(panel.time_days):
        raise ValueError(
            f"panel has T={T} but {len(panel.time_days)} calendar offsets"
        )

    coords: dict[str, Any] = {
        "isin": panel.isins,
        "time": [f"t{j}" for j in range(T)],
        "drift_feature": panel.drift_names,
    }
    if T > 1:
        coords["time_step"] = [f"step{j}" for j in range(T - 1)]
    coords["variance_component"] = ["level", "state", "observation"]
    for col in cfg.group_effects:
        if col in panel.coord_uniques:
            coords[col] = panel.coord_uniques[col]

    mask = panel.observed_mask
    Y_filled = np.where(mask, panel.Y, 0.0)

    with pm.Model(coords=coords) as model:
        # ---- data containers ------------------------------------------------
        X_drift = pm.Data("X_drift", panel.X_drift, dims=("isin", "drift_feature"))
        Y_obs = pm.Data("Y_obs", Y_filled, dims=("isin", "time"))
        obs_mask = pm.Data("obs_mask", mask.astype("float64"), dims=("isin", "time"))
        cv_data = pm.Data("dispersion_cv", panel.dispersion_cv, dims="isin")
        prec_data = pm.Data("precision_weight", panel.precision_weight, dims="isin")
        vol_data = pm.Data("vol_level", panel.vol_level, dims="isin")
        mcap_data = pm.Data("log_mcap", panel.log_mcap, dims="isin")
        range_data = pm.Data("range_norm", panel.range_norm, dims="isin")
        beta_data = pm.Data("avg_beta", panel.avg_beta, dims="isin")
        size_data = pm.Data("size_ratio", panel.size_ratio, dims="isin")
        vol_ratio_data = pm.Data("volume_ratio", panel.volume_ratio, dims="isin")

        idx_data: dict[str, Any] = {}
        for col in cfg.group_effects:
            if col in panel.coord_idx:
                idx_data[col] = pm.Data(
                    f"idx_{col}", panel.coord_idx[col].astype("int32"), dims="isin"
                )

        # ---- mean structure: drift betas ------------------------------------
        # A shared Normal(0, beta_prior_scale) rather than v1's implicit flat
        # prior. With the PT-history family rotated the design matrix is
        # well-conditioned, so a mild ridge costs nothing and keeps a single
        # dominant column from running away as the response is rescaled.
        beta = pm.Normal(
            "beta", mu=0.0, sigma=cfg.beta_prior_scale, dims="drift_feature"
        )
        eta = pt.dot(X_drift, beta)

        # ---- mean structure: crossed group effects --------------------------
        for col, idx in idx_data.items():
            group_effect = pm.ZeroSumNormal(
                f"{col}_effect", sigma=GROUP_EFFECT_SCALE, dims=col
            )
            pm.Deterministic(f"sigma_{col}", group_effect.std())
            eta = eta + group_effect[idx]

        mu_reg = pm.Deterministic("mu_reg", eta, dims="isin")

        # ---- the variance simplex -------------------------------------------
        sigma_total = pm.HalfNormal("sigma_total", sigma=cfg.sigma_total_prior)
        w = pm.Dirichlet(
            "variance_weights",
            a=np.asarray(cfg.variance_split_alpha, dtype="float64"),
            dims="variance_component",
        )
        sigma_level = pm.Deterministic("sigma_level", sigma_total * pt.sqrt(w[0]))
        sigma_state = pm.Deterministic("sigma_state", sigma_total * pt.sqrt(w[1]))
        sigma_obs_base = pm.Deterministic(
            "sigma_obs_base", sigma_total * pt.sqrt(w[2])
        )

        # ---- permanent per-name level ---------------------------------------
        # ZeroSumNormal over ``isin`` so it carries only cross-sectional
        # deviation and cannot alias with ``alpha_time``. Non-centred.
        if cfg.enable_isin_level and n_isin > 1:
            z_level = pm.ZeroSumNormal("z_isin_level", sigma=1.0, dims="isin")
            level = pm.Deterministic("isin_level", sigma_level * z_level, dims="isin")
        else:
            level = pt.zeros(n_isin)
            pm.Deterministic("isin_level", level, dims="isin")

        # ---- OU state over the real calendar grid ----------------------------
        if cfg.enable_ou_state and T > 1:
            ell = pm.LogNormal(
                "ou_length_scale_days",
                mu=math.log(cfg.ou_length_scale_days_mu),
                sigma=cfg.ou_length_scale_days_sigma,
            )
            phi = pm.Deterministic("ou_phi", _ou_transition(panel.time_days, ell),
                                   dims="time_step")

            z_state = pm.Normal("z_state", mu=0.0, sigma=1.0, dims=("isin", "time"))
            # Stationary start, then the exact OU update at each irregular gap.
            cols = [z_state[:, 0]]
            for j in range(1, T):
                prev = cols[-1]
                cols.append(
                    phi[j - 1] * prev
                    + pt.sqrt(1.0 - phi[j - 1] ** 2) * z_state[:, j]
                )
            s_unit = pt.stack(cols, axis=1)  # (isin, time), unit marginal variance
            state = pm.Deterministic(
                "ou_state", sigma_state * s_unit, dims=("isin", "time")
            )
            # The correlation this implies at an infinite gap — directly
            # comparable to the ``rho_inf`` of ``fit_trail_correlation_kernel``,
            # which is what makes the fit falsifiable against the data.
            pm.Deterministic(
                "rho_inf_implied",
                sigma_level**2 / (sigma_level**2 + sigma_state**2 + _EPS),
            )
        else:
            state = pt.zeros((n_isin, T))
            pm.Deterministic("ou_state", state, dims=("isin", "time"))
            pm.Deterministic(
                "rho_inf_implied", pt.as_tensor_variable(1.0)
            )

        # ---- per-lookback level offset ---------------------------------------
        # Anchored at the snapshot (last column = 0) so ``mu_reg`` keeps its
        # meaning as "the level at now" and no ``mu_reg + alpha`` ridge appears.
        if T > 1:
            alpha_free = pm.Normal(
                "alpha_time_free", mu=0.0, sigma=0.25, dims="time_step"
            )
            alpha_time = pm.Deterministic(
                "alpha_time",
                pt.concatenate([alpha_free, pt.zeros(1)]),
                dims="time",
            )
        else:
            alpha_time = pt.zeros(T)

        mu = (
            mu_reg[:, None]
            + level[:, None]
            + state
            + alpha_time[None, :]
        )

        # ---- decision latents -------------------------------------------------
        # ``state_now`` is the snapshot column of the latent mean: the single
        # quantity every downstream consumer reads. Keeping it a named
        # Deterministic is the v1 KALMAN_SCREEN_LATENT contract.
        state_now = pm.Deterministic("state_now", mu[:, -1], dims="isin")
        expected_return = pm.Deterministic(
            "expected_return", mu_reg + level, dims="isin"
        )

        risk_loading = pm.HalfNormal("risk_loading", sigma=max(cfg.risk_penalty, _EPS))
        size_loading = pm.HalfNormal("size_loading", sigma=max(cfg.size_penalty, _EPS))
        volume_loading = pm.HalfNormal(
            "volume_loading", sigma=max(cfg.volume_penalty, _EPS)
        )
        risk_adj_return = pm.Deterministic(
            "risk_adj_return",
            state_now
            - risk_loading * beta_data
            - size_loading * size_data
            - volume_loading * vol_ratio_data,
            dims="isin",
        )
        pm.Deterministic("achieve_prob", pm.math.sigmoid(risk_adj_return), dims="isin")

        # ---- observation scale (log-linear; carried over from 0.9.9.16) -------
        delta_vol = pm.Normal("sigma_delta_vol_level", mu=0.0, sigma=0.5)
        delta_mcap = pm.Normal("sigma_delta_log_mcap", mu=0.0, sigma=0.5)
        delta_range = pm.Normal("sigma_delta_range", mu=0.0, sigma=0.5)
        n_exponent = pm.HalfNormal(
            "sigma_n_exponent", sigma=cfg.sigma_n_exponent_prior
        )

        log_sigma = (
            pt.log(sigma_obs_base)
            + pt.log1p(cv_data)
            + delta_vol * vol_data
            + delta_mcap * mcap_data
            + delta_range * range_data
            - n_exponent * pt.log(pt.maximum(prec_data, _EPS))
        )
        if cfg.enable_sector_scale_offset and "sector" in idx_data:
            sector_offset = pm.ZeroSumNormal(
                "sigma_sector_offset", sigma=GROUP_EFFECT_SCALE, dims="sector"
            )
            log_sigma = log_sigma + sector_offset[idx_data["sector"]]
        sigma_isin = pm.Deterministic("sigma_isin", pt.exp(log_sigma), dims="isin")

        if cfg.enable_time_scale and T > 1:
            tau_free = pm.LogNormal(
                "sigma_time_free", mu=0.0, sigma=0.25, dims="time_step"
            )
            tau_time = pm.Deterministic(
                "sigma_time",
                pt.concatenate([tau_free, pt.ones(1)]),
                dims="time",
            )
        else:
            tau_time = pt.ones(T)
            pm.Deterministic("sigma_time", tau_time, dims="time")

        sigma_full = sigma_isin[:, None] * tau_time[None, :]

        # Masked cells contribute nothing: inflate their sigma so the density is
        # flat there. Cheaper and more numerically stable than index gymnastics,
        # and it keeps the observed tensor rectangular for ``sample_posterior_predictive``.
        sigma_full = pt.switch(pt.gt(obs_mask, 0.5), sigma_full, 1e6)

        # ---- likelihood -------------------------------------------------------
        if cfg.likelihood == "student_t":
            nu = pm.Deterministic(
                "nu", cfg.nu_floor + pm.Gamma("nu_tail", alpha=2.0, beta=0.1)
            )
            pm.StudentT(
                "target_pct_obs",
                nu=nu,
                mu=mu,
                sigma=sigma_full,
                observed=Y_obs,
                dims=("isin", "time"),
            )
        else:
            pm.Normal(
                "target_pct_obs",
                mu=mu,
                sigma=sigma_full,
                observed=Y_obs,
                dims=("isin", "time"),
            )

    logger.info(
        "Built Kalman v2 model: %d ISIN x T=%d, %d drift features, %s",
        n_isin,
        T,
        n_drift,
        cfg.describe(),
    )
    return model


def resolve_screen_latent_v2(idata, *, latent: str = KALMAN_V2_SCREEN_LATENT):
    """Resolve the per-ISIN decision latent from a fitted idata.

    Every consumer goes through this rather than reading ``risk_adj_return`` or
    ``state_now`` directly, so the decision quantity has exactly one name and a
    rename is a one-line change. Mirrors v1's ``resolve_screen_latent``.

    Parameters
    ----------
    idata
        Fitted inference object (``arviz.InferenceData`` or ``xarray.DataTree``).
    latent
        Variable name; defaults to :data:`KALMAN_V2_SCREEN_LATENT`.

    Returns
    -------
    xarray.DataArray
        The posterior of the latent, dims ``(chain, draw, isin)``.

    Raises
    ------
    KeyError
        If the variable is absent, with the available names in the message.
    """
    post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
    if latent not in post:
        raise KeyError(
            f"{latent!r} not in posterior. Available: {sorted(map(str, post.data_vars))}"
        )
    return post[latent]


# --------------------------------------------------------------------------- #
# Self-test — the acceptance check for the central claim                       #
# --------------------------------------------------------------------------- #


def _simulate_panel(
    *,
    n_isin: int = 400,
    lookbacks: tuple[str, ...] = ("1y", "3m", "1w"),
    sigma_level_true: float = 0.55,
    sigma_state_true: float = 0.75,
    sigma_obs_true: float = 0.35,
    ell_true: float = 105.0,
    seed: int = 42,
) -> tuple[KalmanPanelV2, dict[str, float]]:
    """Simulate a panel with a known level / state / noise split.

    Returns the panel and the truth dict, so the self-test can check recovery of
    the quantity this redesign exists to estimate.
    """
    rng = np.random.default_rng(seed)
    cfg = KalmanModelConfig(lookbacks=lookbacks)
    days = cfg.time_grid_days
    T = len(days)

    n_drift = 4
    X = rng.normal(size=(n_isin, n_drift))
    beta_true = np.array([0.30, -0.20, 0.10, 0.05])[:n_drift]
    mu_reg = X @ beta_true

    level = rng.normal(scale=sigma_level_true, size=n_isin)
    level -= level.mean()

    gaps = np.abs(np.diff(days))
    phi = np.exp(-gaps / ell_true)
    s = np.empty((n_isin, T))
    s[:, 0] = rng.normal(size=n_isin)
    for j in range(1, T):
        s[:, j] = phi[j - 1] * s[:, j - 1] + np.sqrt(1 - phi[j - 1] ** 2) * rng.normal(
            size=n_isin
        )
    s *= sigma_state_true

    mu = mu_reg[:, None] + level[:, None] + s
    Y = mu + rng.normal(scale=sigma_obs_true, size=(n_isin, T))

    panel = KalmanPanelV2(
        frame=pd.DataFrame({"isin": [f"SIM{i:05d}" for i in range(n_isin)]}),
        isins=np.array([f"SIM{i:05d}" for i in range(n_isin)]),
        Y=Y,
        time_days=days,
        X_drift=X,
        drift_names=[f"x{i}" for i in range(n_drift)],
        dispersion_cv=np.zeros(n_isin),
        precision_weight=np.ones(n_isin),
        vol_level=np.zeros(n_isin),
        log_mcap=np.zeros(n_isin),
        range_norm=np.zeros(n_isin),
        avg_beta=np.zeros(n_isin),
        size_ratio=np.zeros(n_isin),
        volume_ratio=np.zeros(n_isin),
    )
    truth = {
        "sigma_level": sigma_level_true,
        "sigma_state": sigma_state_true,
        "sigma_obs_base": sigma_obs_true,
        "ou_length_scale_days": ell_true,
        "rho_inf_implied": sigma_level_true**2
        / (sigma_level_true**2 + sigma_state_true**2),
    }
    return panel, truth


def _selftest(draws: int = 500, tune: int = 500, chains: int = 2) -> int:
    """Fit a simulated panel and check the level / state split is recovered.

    Returns a process exit code: 0 on success, 1 if any truth falls outside the
    posterior 94 % HDI.
    """
    import arviz as az

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    panel, truth = _simulate_panel()

    print(f"Simulated panel: {panel.n_isin} names x T={panel.n_time}")
    print(f"  T_eff (design-time)     : {panel.effective_t():.3f}")
    kern = fit_trail_correlation_kernel(panel.Y, panel.time_days)
    print(
        f"  offline kernel fit      : rho_inf={kern['rho_inf']:.3f} "
        f"ell={kern['ell_days']:.1f}d  rmse={kern['rmse']:.4f}"
    )
    print(f"  truth rho_inf           : {truth['rho_inf_implied']:.3f}")

    cfg = KalmanModelConfig(orthogonalise_pt_history=False)
    model = build_kalman_pt_model_v2(panel, cfg)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            random_seed=7,
            progressbar=False,
            compute_convergence_checks=False,
        )

    names = [
        "sigma_level",
        "sigma_state",
        "sigma_obs_base",
        "ou_length_scale_days",
        "rho_inf_implied",
    ]
    # ArviZ 1.x: ``ci_prob`` replaces ``hdi_prob``, and the interval columns are
    # named ``eti<pct>_lb`` / ``eti<pct>_ub`` rather than ``hdi_3%`` / ``hdi_97%``.
    # Resolve them by suffix so a future default change does not silently break
    # the check into a KeyError.
    summary = az.summary(idata, var_names=names, ci_prob=0.94)
    lb_col = next(c for c in summary.columns if str(c).endswith("_lb"))
    ub_col = next(c for c in summary.columns if str(c).endswith("_ub"))
    print("\nRecovery check (truth must lie inside the 94% interval):")
    failures = 0
    for name in names:
        row = summary.loc[name]
        lo, hi = float(row[lb_col]), float(row[ub_col])
        ok = lo <= truth[name] <= hi
        failures += 0 if ok else 1
        print(
            f"  {name:<24} truth {truth[name]:>7.3f}  "
            f"post {float(row['mean']):>7.3f} [{lo:>7.3f}, {hi:>7.3f}]  "
            f"{'PASS' if ok else 'FAIL'}"
        )

    div = int(idata.sample_stats["diverging"].sum())
    min_ess = float(summary["ess_bulk"].min())
    max_rhat = float(summary["r_hat"].max())
    print(f"\n  divergences {div}   min bulk ESS {min_ess:.0f}   max R-hat {max_rhat:.4f}")
    if failures:
        print(f"\nSELFTEST FAILED: {failures} parameter(s) outside the HDI")
        return 1
    print("\nSELFTEST PASSED: level / state split recovered")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run the recovery check")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()
    if args.selftest:
        sys.exit(_selftest(draws=args.draws, tune=args.tune, chains=args.chains))
    parser.print_help()
