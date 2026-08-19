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
``sigma_isin_level`` fixed 0.10    ``w_level`` free, identified by the decay
AR(1) state off by default         OU state on by default, one length-scale
Latents sampled per name/time      Latents **marginalised** — 32,503 free
                                   parameters become ~60
Missing cells given ``sigma=1e6``  Likelihood on the observed sub-vector, names
                                   grouped by missingness pattern
Scales added independently         One total scale split by a **Dirichlet**
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
    "KALMAN_V2_SCREEN_LATENT_SD",
    "DEFAULT_LOOKBACK_DAYS",
    "PT_HISTORY_FAMILY",
    "build_kalman_pt_model_v2",
    "orthogonalise_family",
    "effective_sample_size_of_panel",
    "partition_covariance_groups",
    "bucket_scale_index",
    "covariance_groups_for",
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
#:
#: Since the latents are marginalised (see :func:`build_kalman_pt_model_v2`) the
#: model reports the smoother's *conditional mean* and *sd* rather than a sampled
#: path. ``resolve_screen_latent_v2`` reconstitutes draws from the pair, so
#: consumers see the same thing they always did.
KALMAN_V2_SCREEN_LATENT: str = "state_now_mean"
KALMAN_V2_SCREEN_LATENT_SD: str = "state_now_sd"

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
#:
#: Two of the three left the design matrix on 2026-08-18: ``feat_pt_drift`` and
#: ``feat_pt_accuracy_1y`` are increments of the response trail rather than
#: measurements of its level (see ``DRIFT_EXCLUSIONS`` in
#: ``pymc_kalman_filter_pt_v2.py``), so the family is now a single column and
#: :func:`orthogonalise_family` correctly no-ops on it. The constant and the
#: rotation stay: the collinearity they fix is a property of the family, not of
#: the run that first hit it, and a catalogue change can repopulate it.
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
        Concentration on ``(level, state, observation)``. Read as two Beta
        priors rather than one Dirichlet — see the builder for why the
        coordinates matter — so ``(a_level, a_state)`` is the prior on
        ``rho_inf`` and ``(a_obs, a_level + a_state)`` the prior on the
        observation share. Weakly informative and centred on the measurement:
        the group-adjusted kernel puts ~33 % of the structured within-name
        variance in the permanent level and ~67 % in the decaying state. Small
        concentrations keep it weak enough for the data to move it.
    likelihood
        ``'student_t'`` (default) or ``'normal'``. The Student-t ``nu`` is
        floored at ``nu_floor``; see the v1 docstring for why relaxing that floor
        reintroduces an improper density corner.
    time_scale_applies_to
        What the per-lookback scale ``sigma_time`` multiplies — ``'covariance'``
        (default) or ``'observation'`` (the pre-2026-08-18 behaviour, retained
        only as a comparison arm).

        Under ``'observation'`` the within-name covariance is
        ``w_L*J + w_S*K + w_O*diag(tau^2)``, so inflating a far column's variance
        necessarily divides every one of its correlations by
        ``sqrt(A_ss * A_tt)``. That family **cannot represent this panel**: solved
        against the measured residual structure (variance ratios
        ``[2.13, 1.49, 1.11, 1.0]``; correlations 0.921 / 0.315 / -0.065 at gaps of
        7 / 91 / 365 days) it requires ``w_S > 1``, infeasible by 74 %. The
        2026-08-18 run took the only escape available to it and drove ``tau`` to
        ``[12.3, 9.2, 2.5, 1.0]`` — ten prior sd out — buying the variance ratio by
        destroying the correlation, which is what failed ``ppc_decay`` and
        over-covered steps 0 and 1.

        Under ``'covariance'`` the same scale multiplies the whole matrix,
        ``A = diag(tau) (w_L*J + w_S*K + w_O*I) diag(tau)``, so correlations are
        ``tau``-free and variance ratios are ``tau^2``. The same measurement then
        solves at ``tau ~ [1.46, 1.22, 1.05, 1.0]``, ``ell ~ 82 d``, ``w_L ~ 0.02``,
        ``w_O ~ 0`` — feasible to within 1.4 %, with ``tau`` inside its prior
        instead of far outside it.
    time_scale_prior_sigma
        Log-scale sd of the ``sigma_time`` prior. Widened 0.25 -> 0.40 with the
        restructure: the measured optimum of 1.05-1.46 sat at the edge of the old
        90 % interval of ``[0.66, 1.51]``.
    rho_scale_buckets
        Number of quantile buckets of :attr:`KalmanPanelV2.scale_index` that get
        their own ``rho_inf``. ``1`` (the default) is the single global split.

        **Built, measured, and defaulted OFF on 2026-08-19.** The motivation was
        sound and the mechanism works — on simulated data the tilt is recovered
        (truth 0.800, posterior 0.553 ``[0.237, 0.925]``) — but on the production
        panel ``rho_scale_slope`` came back **0.578 +/- 0.828, an 89 % interval of
        [-0.836, 1.761] that spans zero**, at ESS 1986, so the width is posterior
        uncertainty and not something a longer run will sharpen. The buckets came
        out monotone as predicted, ``[0.0041, 0.0043, 0.0052, 0.0076, 0.0196]``,
        and ``ppc_decay`` did not move: 0.429 observed against [0.334, 0.399]
        replicated, versus [0.320, 0.393] with one global split. It costs +23 %
        per gradient and takes the covariance-group count from 6 to 21. Set it to
        ``5`` to re-enable; the machinery, the self-test and the scale index all
        remain.

        **Why the motivating measurement overstated the effect.** Residual
        correlation decay by fitted-``sigma_isin`` quintile, 2026-08-18 universe,
        measured on the raw residual and again after standardising it by the
        model's own ``sigma_i * tau_t``:

        ==========  ==========  ==============  =====================
        quintile    mean sigma  raw rho_inf     standardised rho_inf
        ==========  ==========  ==============  =====================
        Q1          0.258       0.0000          0.0000
        Q2          0.353       0.0000          0.0000
        Q3          0.440       0.0000          0.0000
        Q4          0.568       0.0000          0.0000
        Q5          0.910       0.1170          **0.0723**
        pooled      0.506       0.0706          **0.0000**
        ==========  ==========  ==============  =====================

        The pooled column is the finding. A per-name scale multiplies a name's
        whole trail together, which in any correlation estimate is a rank-1
        common component — indistinguishable from a permanent level. Standardise
        it away and the pooled permanent correlation goes to **exactly zero**:
        there is no pooled level, and the model fitting ``rho_inf ~ 0.006`` was
        right, not under-powered. Only Q5 keeps a genuine level after
        standardisation, and at 0.072 rather than the 0.117 the raw statistic
        showed — real, but half the apparent size, carried by 20 % of the
        universe, and worth about one gate-width of the decay statistic.

        Bucketing rather than a per-name ``rho_i`` was a performance constraint,
        not a modelling preference: a per-name composition means a per-name
        covariance and a batched Cholesky over 6,497 matrices, measured at
        4,400 ms/gradient against 2.5 ms for the shared form. Names in one bucket
        share one Cholesky, exactly as names in one missingness pattern already
        do.
    rho_scale_slope_prior
        Prior sd of ``rho_scale_slope``, the logit-scale tilt of ``rho_inf`` per
        unit of the standardised scale index. Centred at **zero**, so the null
        hypothesis is the single global split and a tilt has to be earned — which
        on this panel it did not. At ``rho_scale_buckets = 1`` the parameter is
        not created at all.
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
    variance_weight_floor: float = 1e-3
    enable_ou_state: bool = True
    enable_isin_level: bool = True

    # ---- numerical guards --------------------------------------------------
    #: Ridge added to the diagonal of ``A`` before its Cholesky. ``A`` is a
    #: correlation-like matrix built from a simplex, so it is PSD by
    #: construction but can approach singularity when the state share dominates
    #: and ``ell`` is large (every row nearly identical).
    cholesky_jitter: float = 1e-8
    #: Clip range for ``log sigma_isin``. The scale model is a sum of six terms,
    #: several with unbounded coefficients; without a clip ``exp`` of the sum can
    #: overflow to ``inf``. The default admits scales from ~6e-6 to ~55 on the
    #: standardised response, which brackets anything meaningful by orders of
    #: magnitude.
    log_sigma_clip: tuple[float, float] = (-12.0, 4.0)
    #: Prior on ``log sigma_total``. Sampled on the log scale rather than as a
    #: ``HalfNormal``, because the scale model needs ``log(sigma_total)`` and a
    #: half-normal can reach 0 exactly, where that log is ``-inf``.
    log_sigma_total_mu: float = -1.6
    log_sigma_total_sigma: float = 1.0

    #: Number of per-lookback-coverage buckets. Names are partitioned by
    #: ``(missingness pattern, coverage bucket)`` and each group gets its own
    #: covariance sub-block, so a name whose one-year consensus rests on four
    #: analysts is not charged the same measurement precision as one built on
    #: thirty. ``1`` disables bucketing and keeps one shared covariance shape per
    #: missingness pattern. Each extra group costs one 4x4 Cholesky per gradient,
    #: so the knob is cheap; it is defaulted off because its benefit is not yet
    #: measured, and every group is an extra observed variable to reassemble in
    #: the posterior-predictive stage.
    coverage_profile_buckets: int = 1

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
    time_scale_applies_to: str = "covariance"
    time_scale_prior_sigma: float = 0.40
    rho_scale_buckets: int = 1
    rho_scale_slope_prior: float = 1.0

    # ---- decision tilts ----------------------------------------------------
    risk_penalty: float = 0.15
    size_penalty: float = 0.25
    volume_penalty: float = 0.10

    def __post_init__(self) -> None:
        if self.likelihood not in ("student_t", "normal"):
            raise ValueError(
                f"likelihood must be 'student_t' or 'normal', got {self.likelihood!r}"
            )
        if self.time_scale_applies_to not in ("covariance", "observation"):
            raise ValueError(
                "time_scale_applies_to must be 'covariance' or 'observation', got "
                f"{self.time_scale_applies_to!r}"
            )
        if self.time_scale_prior_sigma <= 0:
            raise ValueError("time_scale_prior_sigma must be positive")
        if self.rho_scale_buckets < 1:
            raise ValueError("rho_scale_buckets must be >= 1")
        if self.rho_scale_slope_prior <= 0:
            raise ValueError("rho_scale_slope_prior must be positive")
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
        if self.coverage_profile_buckets < 1:
            raise ValueError("coverage_profile_buckets must be >= 1")
        if not 0.0 <= self.variance_weight_floor < 0.3:
            raise ValueError("variance_weight_floor must be in [0, 0.3)")
        lo, hi = self.log_sigma_clip
        if not lo < hi:
            raise ValueError(f"log_sigma_clip must be increasing, got {self.log_sigma_clip!r}")

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
            f"tau={self.time_scale_applies_to if self.enable_time_scale else 'off'}, "
            f"rho={'global' if self.rho_scale_buckets < 2 else f'{self.rho_scale_buckets}x scale'}, "
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

    # ---- per-lookback analyst coverage -------------------------------------
    #: Relative analyst coverage per cell, shape ``(n_isin, T)``, normalised so
    #: the snapshot column is 1.0. Sourced from the ``n_analysts_{lb}`` columns
    #: that ``mv_pymc_kalman_pt_v2`` emits. Used to bucket names into covariance
    #: groups when :attr:`KalmanModelConfig.coverage_profile_buckets` > 1, and to
    #: form the trail-average :attr:`precision_weight`. Empty means "no profile
    #: available", and every name falls in one bucket.
    coverage_profile: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype="float64")
    )

    # ---- variance-composition driver ---------------------------------------
    #: Per-name standardised scale index, shape ``(n_isin,)``. A data-only
    #: stand-in for ``log sigma_isin``, which cannot be used directly because the
    #: covariance-group partition has to be fixed before any coefficient is
    #: sampled. Built in ``prepare_panel`` from the three strongest terms of the
    #: scale model itself and measured at Pearson **0.966** / Spearman 0.967
    #: against the fitted ``log sigma_isin`` on the 2026-08-18 universe; its
    #: quintiles reproduce the residual-decay pattern that motivates the split
    #: (``[0, 0, 0, 0, 0.109]`` against ``[0, 0, 0, 0, 0.116]`` for the fitted
    #: scale's own quintiles). Empty leaves every name in one bucket.
    scale_index: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="float64")
    )

    # ---- provenance --------------------------------------------------------
    orthogonal_rotation: Optional[np.ndarray] = None
    orthogonal_source_names: tuple[str, ...] = ()
    #: Per-feature level-vs-contrast correlation table, measured on the design
    #: matrix **before** any orthogonalisation — see
    #: ``pymc_kalman_filter_pt_v2.screen_contrast_identities``. Carried on the
    #: panel rather than recomputed downstream so the audit and the fit cannot
    #: disagree about which matrix was screened. Empty when the workflow built
    #: the panel without running the screen.
    contrast_screen: pd.DataFrame = field(default_factory=pd.DataFrame)

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
    decays with length-scale ``ell``.

    **Mind which matrix you hand it.** On a *raw* response panel the constant
    term absorbs everything common to a name across time — including the spread
    of the regression mean, which is by far the largest such component. The
    2026-08-18 universe reads ``rho_inf = 0.429`` raw and **below 0.1** on the
    residual once the model's own mean structure is removed: next to no permanent
    per-name level is left over, because the level *is* ``mu_reg``. Both readings
    are useful and they answer different questions, so §8 of the workflow fits
    the kernel on both and reports them side by side.

    Running it on the panel *before* fitting
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


def _ou_correlation(time_days: np.ndarray, ell) -> Any:
    """Stationary OU correlation matrix on an irregular calendar grid.

    .. math:: K_{st} = \\exp(-|d_s - d_t| / \\ell)

    This is the *marginal* correlation of an Ornstein-Uhlenbeck process observed
    at offsets ``d``. v2 uses the matrix directly rather than the equivalent
    step-by-step recursion, because the recursion requires sampling the state
    path — 25,988 latent parameters on the production panel, multiplied by a
    learned scale, which is a Neal funnel at scale and is what made the first
    v2 build sample at 400 seconds per draw. Written as a covariance the state
    is integrated out in closed form and never sampled at all.

    Using the *real* calendar gaps rather than a time index is what lets one
    length-scale reproduce a correlation of 0.97 at a 7-day gap and 0.43 at a
    365-day gap. An index-based AR(1) cannot: the lookback grid is not equally
    spaced.

    Parameters
    ----------
    time_days
        Calendar offsets, shape ``(T,)``, oldest first, anchored at 0.
    ell
        Length-scale in days (a PyTensor scalar).

    Returns
    -------
    Any
        PyTensor ``(T, T)`` symmetric matrix with unit diagonal.
    """
    days = np.asarray(time_days, dtype="float64")
    gaps = np.abs(days[:, None] - days[None, :])
    return pt.exp(-pt.as_tensor_variable(gaps) / ell)


def bucket_scale_index(
    scale_index: np.ndarray,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretise a per-name scale index into quantile buckets.

    The variance composition varies with a name's scale (see
    :attr:`KalmanModelConfig.rho_scale_buckets`), but a per-name composition
    means a per-name covariance and a batched Cholesky over every name — 4,400
    ms/gradient against 2.5 ms for the shared form. Bucketing keeps one Cholesky
    per (missingness pattern, coverage bucket, scale bucket) group while letting
    the composition move across the universe.

    The buckets discretise where ``rho_inf`` is *evaluated*, not how it is
    parameterised: the builder fits one logit-linear tilt and reads it at each
    bucket's mean index, so the number of free parameters does not grow with
    ``n_buckets`` and the relationship stays monotone.

    Parameters
    ----------
    scale_index
        Per-name standardised index, shape ``(n_isin,)``. NaN is filled with the
        median.
    n_buckets
        Requested bucket count. ``< 2`` returns a single bucket.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(bucket_of_name, bucket_mean_index)``. Empty buckets are dropped and
        the remaining ones renumbered, so ``bucket_of_name.max() + 1 ==
        len(bucket_mean_index)`` always holds — the builder relies on it to size
        the ``rho_bucket`` coordinate.
    """
    x = np.asarray(scale_index, dtype="float64").ravel()
    n = x.shape[0]
    if n_buckets < 2 or n == 0 or not np.isfinite(x).any():
        return np.zeros(max(n, 0), dtype="int64"), np.zeros(1, dtype="float64")
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    if float(x.std()) < _EPS:
        return np.zeros(n, dtype="int64"), np.array([float(x.mean())])

    edges = np.quantile(x, np.linspace(0.0, 1.0, n_buckets + 1)[1:-1])
    raw = np.searchsorted(edges, x, side="right").astype("int64")
    uniq = np.unique(raw)
    remap = np.full(int(uniq.max()) + 1, -1, dtype="int64")
    remap[uniq] = np.arange(uniq.size, dtype="int64")
    bucket = remap[raw]
    means = np.array([float(x[bucket == k].mean()) for k in range(uniq.size)])
    logger.info(
        "Scale buckets: %d (sizes %s, mean index %s)",
        uniq.size,
        [int((bucket == k).sum()) for k in range(uniq.size)],
        np.round(means, 3).tolist(),
    )
    return bucket, means


def covariance_groups_for(
    panel: "KalmanPanelV2",
    config: "KalmanModelConfig",
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], np.ndarray]:
    """Resolve the covariance-group partition and the scale-bucket means.

    The **single** definition of how names are grouped. The builder names one
    observed variable per group and the posterior-predictive stage reassembles
    replicates by that name, so a second implementation that disagreed by one
    group would stitch the wrong rows back onto the panel silently.

    Returns
    -------
    tuple
        ``(groups, bucket_mean_index)`` where ``groups`` is a list of
        ``(row_index, observed_column_index, scale_bucket)``.
    """
    bucket, means = bucket_scale_index(panel.scale_index, config.rho_scale_buckets)
    if bucket.shape[0] != panel.n_isin:
        bucket = np.zeros(panel.n_isin, dtype="int64")
        means = np.zeros(1, dtype="float64")
    groups = partition_covariance_groups(
        panel.observed_mask,
        panel.coverage_profile,
        config.coverage_profile_buckets,
        scale_bucket=bucket,
    )
    return groups, means


def partition_covariance_groups(
    mask: np.ndarray,
    coverage_profile: np.ndarray,
    n_buckets: int,
    scale_bucket: Optional[np.ndarray] = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Partition names into groups that share one covariance sub-block.

    Two names can share a Cholesky factorisation when they observe the same set
    of lookback columns and (optionally) sit in the same analyst-coverage
    bucket. Grouping this way is what lets the model handle missing trail cells
    *exactly* — the likelihood is evaluated on the observed sub-vector — instead
    of v2's first approach, which kept the grid rectangular by setting the
    missing cells' sigma to 1e6. That trick left thousands of near-flat
    likelihood directions for NUTS to traverse at the global step size.

    Parameters
    ----------
    mask
        Boolean ``(n_isin, T)`` observation mask.
    coverage_profile
        ``(n_isin, T)`` relative analyst coverage, or an empty array.
    n_buckets
        Number of coverage buckets; ``1`` groups by missingness pattern only.
    scale_bucket
        Optional per-name scale bucket from :func:`bucket_scale_index`. Names in
        different scale buckets get different variance *compositions*, hence
        different covariance matrices, so they cannot share a Cholesky.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray, int]]
        ``(row_index, observed_column_index, scale_bucket)`` per group. Groups
        are ordered largest first, so the dominant pattern is the first observed
        variable in the model and reads first in any summary.
    """
    n, T = mask.shape
    keys = mask.astype(np.uint8) @ (1 << np.arange(T, dtype=np.uint64))

    if n_buckets > 1 and coverage_profile.size:
        # Bucket on the trail-average coverage, which is the summary the
        # covariance actually cares about: a name whose whole trail is thin
        # deserves a wider observation leg than one thin only at the far end.
        avg = np.nanmean(np.where(mask, coverage_profile, np.nan), axis=1)
        avg = np.nan_to_num(avg, nan=1.0)
        # Rank-based edges so buckets are populated regardless of the shape of
        # the coverage distribution.
        edges = np.nanquantile(avg, np.linspace(0, 1, n_buckets + 1)[1:-1])
        bucket = np.searchsorted(edges, avg, side="right").astype(np.uint64)
        keys = keys * np.uint64(n_buckets) + bucket

    if scale_bucket is not None:
        sb = np.asarray(scale_bucket, dtype="int64")
        n_scale = int(sb.max()) + 1 if sb.size else 1
        if n_scale > 1:
            keys = keys * np.uint64(n_scale) + sb.astype(np.uint64)
    else:
        sb = np.zeros(n, dtype="int64")

    groups: list[tuple[np.ndarray, np.ndarray, int]] = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        cols = np.flatnonzero(mask[rows[0]])
        if cols.size == 0:
            logger.warning("Dropping %d name(s) with no observed trail cell", rows.size)
            continue
        groups.append((rows, cols, int(sb[rows[0]])))
    groups.sort(key=lambda g: -len(g[0]))
    logger.info(
        "Covariance groups: %d (sizes %s, widths %s, scale buckets %s)",
        len(groups),
        [len(r) for r, _, _ in groups][:8],
        [len(c) for _, c, _ in groups][:8],
        [b for _, _, b in groups][:8],
    )
    return groups


def build_kalman_pt_model_v2(
    panel: KalmanPanelV2,
    config: Optional[KalmanModelConfig] = None,
) -> "pm_typing.Model":
    """Build the v2 correlated-trail Kalman price-target model.

    Structure
    ---------
    The generative story for name ``i`` at lookback ``t`` is::

        y[i, t] = mu_reg[i]                     # drift betas + crossed group effects
                + alpha_time[t]                 # per-lookback level offset
                + tau[t] * ( level[i]           # permanent per-name deviation
                           + s[i, t]            # OU state, corr = exp(-gap/ell)
                           + eps[i, t] )        # measurement noise

    but ``level`` and ``s`` are **never sampled**. Both are Gaussian and both
    enter the mean linearly, so they integrate out in closed form, leaving a
    multivariate Student-t over each name's whole trail::

        C       = w_L*J + w_S*K(ell) + w_O*I               # correlation shape
        A       = diag(tau) C diag(tau)                    # T x T, shared
        y[i, :] ~ MvStudentT(nu, mu_reg[i] + alpha_time, sigma_isin[i]^2 * A)

    ``tau`` scales a whole lookback column — every component of it — rather than
    its white-noise leg alone. That is a statement about the trail's data
    quality: the further back a vintage is, the noisier all of it is. It is also
    the only form that can reach this panel; see
    :attr:`KalmanModelConfig.time_scale_applies_to`.

    with ``(w_L, w_S, w_O)`` read as **shares of within-name variance** — and
    sampled as two Betas rather than one Dirichlet, in the coordinates the data
    identifies (see the builder body) — which is exactly what the empirical
    kernel measures, since ``rho_inf`` is a share and not a level.

    Why marginalise, in numbers
    ---------------------------
    The first v2 build sampled the state path. On the production panel that is
    ``6497 x 4 = 25,988`` latent parameters plus a 6,497-element level vector,
    each multiplied by a learned scale — a Neal funnel at scale. Measured:

    ===========================================  ==========  ================
    form                                         free params gradient (n=6497)
    ===========================================  ==========  ================
    sampled ``z_state`` + ``z_isin_level``       32,503      ~1,600 ms
    marginal, per-name covariance (batched chol) ~60         ~4,400 ms
    marginal, shared shape + pre-formed chol     ~60         **2.3 ms**
    ===========================================  ==========  ================

    It also sampled at a step size of 0.001 with 255 gradient evaluations per
    draw — 400 s/draw, i.e. months for a production run. Marginalisation removes
    the funnel, the serial OU graph chain and the near-flat directions that the
    masked cells used to contribute.

    Two consequences of the marginal form, both deliberate:

    * **The Student-t is now per-name, not per-cell.** A multivariate t is the
      marginal of a per-name scale mixture, so an outlier is a whole name with an
      odd trail rather than one odd cell. For a sticky analyst consensus that is
      arguably the better model, but it *is* a change from v1.
    * **The per-name scale multiplies the whole covariance**, so the smoother
      gain is common within a covariance group while the posterior **variance**
      still scales with ``sigma_isin``. Per-name shrinkage *weight* would need
      ``Sigma_i = B + sigma_i^2 D``, which requires a batched Cholesky over 6,497
      matrices — the 4,400 ms row above. Rejected on measurement.

    Identification
    --------------
    ``w_L`` and ``w_S`` are separately identified because they have different
    *signatures in the correlation decay*, not because they have different
    marginal variances. The level contributes a constant ``rho_inf`` at every
    gap; the state contributes ``exp(-gap/ell)``, which dies. A factorised
    likelihood sees neither, which is precisely why v1 had to pin
    ``sigma_isin_level`` to a constant and switch the state off. On the
    2026-08-18 universe the offline fit separates them at RMSE 0.026 over 15 gap
    pairs, so the information is present; ``--selftest`` confirms the sampler
    recovers it.

    The simplex is also the fix for the failure the changelog calls gate 3. In v1
    the level was added on top of the noise, so every attempt to grow it grew the
    total predictive scale instead of redistributing it (``sigma_base`` rose from
    0.1676 to 0.1777 when ``isin_level_scale`` went to 0.40). Here the components
    are legs of one simplex: mass given to the level is mass taken from the
    noise, by construction.

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
    * **Tried and rejected** — ``tau`` on the observation leg alone
      (``time_scale_applies_to='observation'``, the form shipped until
      2026-08-18). Retained as a comparison arm and nothing else: solved against
      the panel's measured residual structure the family needs ``w_S > 1``, so it
      can only trade correlation for variance, and the production run duly drove
      ``tau`` to ``[12.3, 9.2, 2.5, 1.0]`` — ten prior sd out — failing
      ``ppc_decay`` and over-covering the two oldest steps at 0.958 / 0.960.
    * **Tried and rejected** — per-cell data augmentation for the Student-t
      (``n x T`` conjugate scale variables), proposed when the replicate spread
      and decay failures were attributed to the marginal t being a per-NAME
      scale mixture. The attribution was wrong: a multivariate t's correlation
      matrix is its scale matrix's, and reconstructing the 2026-08-18 posterior
      shows the replicate sd ratio (1.76 predicted vs 1.75 measured) and the flat
      replicate decay are both accounted for by ``Var(mu_reg) = 2.50`` against a
      response variance of 0.87. It would have restored ~26k latents for no gate
      movement.
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
        "time_b": [f"t{j}" for j in range(T)],
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

    # The snapshot column anchors the OU grid and is the only cell every
    # downstream decision reads, so a name missing it is not a panel member.
    if not mask[:, T - 1].all():
        n_bad = int((~mask[:, T - 1]).sum())
        raise ValueError(
            f"{n_bad} name(s) have no snapshot observation (column {T - 1}). "
            "Filter them out in prepare_panel; the smoother has no anchor for them."
        )
    if np.any(panel.precision_weight <= 0):
        raise ValueError(
            "precision_weight must be strictly positive — it enters the scale "
            "model through log(). Clip it at panel-prep time."
        )

    groups, bucket_mean = covariance_groups_for(panel, cfg)
    if not groups:
        raise ValueError("No name has an observed trail cell; nothing to fit.")
    n_rho = int(bucket_mean.shape[0])
    coords["rho_bucket"] = [f"q{k + 1}" for k in range(n_rho)]

    with pm.Model(coords=coords) as model:
        # ---- data containers ------------------------------------------------
        X_drift = pm.Data("X_drift", panel.X_drift, dims=("isin", "drift_feature"))
        # Provenance containers. The likelihood reads the observed sub-blocks
        # directly per covariance group (rectangular slices of ``panel.Y``), so
        # these are not on the logp path — they are here so the fitted idata
        # carries the response grid and its missingness pattern, which the §8
        # posterior-predictive reassembly and the T_eff audit both need.
        pm.Data("Y_obs", Y_filled, dims=("isin", "time"))
        pm.Data("obs_mask", mask.astype("float64"), dims=("isin", "time"))
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
        # ``sigma_total`` is sampled on the LOG scale. The observation-scale
        # model below needs ``log(sigma_total)``, and a HalfNormal can reach 0
        # exactly, where that log is -inf — one of the boundary failures the
        # first v2 build carried.
        log_sigma_total = pm.Normal(
            "log_sigma_total",
            mu=cfg.log_sigma_total_mu,
            sigma=cfg.log_sigma_total_sigma,
        )
        sigma_total = pm.Deterministic("sigma_total", pt.exp(log_sigma_total))

        # Two Betas rather than a 3-simplex, and the choice is not cosmetic.
        #
        # A Dirichlet is transformed to unconstrained space by stick-breaking,
        # whose coordinates mix "level vs (state+obs)" — a direction the data
        # does not identify cleanly. What the data DOES identify, and what the
        # offline kernel fit measures directly, is:
        #
        #   rho_inf   = level / (level + state)   from the SHAPE of the decay
        #   obs_share = observation / total       from decorrelation at short gaps
        #
        # Sampling those two directly makes each a primitive with its own ESS and
        # its own prior, instead of a ratio of two simplex legs. On a 6.5k panel
        # the first parameterisation stalled at R-hat 1.21 with ESS 7 on
        # ``rho_inf_implied``; these are the coordinates the likelihood is
        # actually curved in.
        a_lvl, a_st, a_obs = cfg.variance_split_alpha
        structured = a_lvl + a_st
        if cfg.enable_isin_level and cfg.enable_ou_state and T > 1:
            rho_inf = pm.Beta("rho_inf", alpha=a_lvl, beta=a_st)
        elif cfg.enable_isin_level:
            rho_inf = pt.ones(())
        else:
            rho_inf = pt.zeros(())
        obs_share = pm.Beta("obs_share", alpha=a_obs, beta=structured)

        # ---- the split varies with a name's SCALE ----------------------------
        # ``rho_inf`` above is the baseline: the composition of a name sitting at
        # index 0, i.e. of average scale. ``rho_scale_slope`` tilts it on the
        # logit scale across the scale buckets, so a slope of exactly 0 restores
        # the single global split *and its prior* — which is what makes the two
        # arms comparable. The tilt is one parameter regardless of the bucket
        # count: the buckets discretise where it is evaluated, not how it is
        # parameterised, and the relationship stays monotone by construction.
        #
        # Why it is needed at all: the panel's permanent per-name level lives
        # almost entirely in the noisiest fifth of the universe (residual
        # rho_inf 0.116 in Q5 against 0.000 in Q1-Q4), while a single rho_inf
        # fitted in a metric weighting by 1/sigma_i^2 lands at 0.0067 -- right
        # for 80 % of names and wrong for the fifth that dominates every
        # unweighted statistic. See KalmanModelConfig.rho_scale_buckets.
        tilted = (
            n_rho > 1 and cfg.enable_isin_level and cfg.enable_ou_state and T > 1
        )
        if tilted:
            rho_slope = pm.Normal(
                "rho_scale_slope", mu=0.0, sigma=cfg.rho_scale_slope_prior
            )
            rho_c = pt.clip(rho_inf, 1e-6, 1.0 - 1e-6)
            rho_vec = pm.math.sigmoid(
                pt.log(rho_c)
                - pt.log1p(-rho_c)
                + rho_slope * pt.as_tensor_variable(bucket_mean)
            )
        else:
            rho_vec = pt.repeat(pt.as_tensor_variable(rho_inf).reshape((1,)), n_rho)
        pm.Deterministic("rho_inf_by_bucket", rho_vec, dims="rho_bucket")

        f = cfg.variance_weight_floor

        def _split(rho: Any) -> tuple[Any, Any, Any]:
            """Normalised (level, state, observation) weights for a given rho.

            Elementwise, so it serves both the scalar baseline and the per-bucket
            vector. The diagonal of the resulting correlation shape is 1 for any
            rho, which is what keeps this a *composition*: mass given to the
            level is mass taken from the state, never added to the total. That
            property is the one v1 lacked, and it survives the tilt unchanged.
            """
            # Floor the legs off the boundary: sqrt(w) has unbounded derivative
            # at 0 and the level/state legs sit inside a Cholesky, so a leg
            # collapsing to exactly 0 is both a geometry problem and a
            # near-singular matrix.
            wl = (1.0 - obs_share) * rho + f
            ws = (1.0 - obs_share) * (1.0 - rho) + f
            wo = obs_share + f
            tot = wl + ws + wo
            return wl / tot, ws / tot, wo / tot

        # Baseline weights — reported, and used for the summary Deterministics so
        # they stay scalars comparable with every earlier run. They describe the
        # average-scale name, not any particular bucket.
        w_level, w_state, w_obs = _split(rho_inf)
        wl_vec, ws_vec, wo_vec = _split(rho_vec)

        pm.Deterministic(
            "variance_weights",
            pt.stack([w_level, w_state, w_obs]),
            dims="variance_component",
        )
        pm.Deterministic("sigma_level", sigma_total * pt.sqrt(w_level))
        pm.Deterministic("sigma_state", sigma_total * pt.sqrt(w_state))
        pm.Deterministic("sigma_obs_base", sigma_total * pt.sqrt(w_obs))
        # Reported under the same name as before so consumers are unchanged, and
        # directly comparable to the ``rho_inf`` that
        # :func:`fit_trail_correlation_kernel` computes on the raw panel. Having
        # both is what makes the level/state split falsifiable rather than merely
        # fitted — the §4b audit prints the empirical value before the fit runs.
        pm.Deterministic("rho_inf_implied", w_level / (w_level + w_state))

        # ---- the OU correlation, as a matrix rather than a sampled path -------
        if cfg.enable_ou_state and T > 1:
            ell = pm.LogNormal(
                "ou_length_scale_days",
                mu=math.log(cfg.ou_length_scale_days_mu),
                sigma=cfg.ou_length_scale_days_sigma,
            )
            K = _ou_correlation(panel.time_days, ell)
        else:
            ell = None
            K = pt.eye(T)

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

        # ---- observation scale (log-linear; carried over from 0.9.9.16) -------
        delta_vol = pm.Normal("sigma_delta_vol_level", mu=0.0, sigma=0.5)
        delta_mcap = pm.Normal("sigma_delta_log_mcap", mu=0.0, sigma=0.5)
        delta_range = pm.Normal("sigma_delta_range", mu=0.0, sigma=0.5)
        n_exponent = pm.HalfNormal(
            "sigma_n_exponent", sigma=cfg.sigma_n_exponent_prior
        )

        log_sigma = (
            log_sigma_total
            + pt.log1p(cv_data)
            + delta_vol * vol_data
            + delta_mcap * mcap_data
            + delta_range * range_data
            - n_exponent * pt.log(prec_data)
        )
        if cfg.enable_sector_scale_offset and "sector" in idx_data:
            sector_offset = pm.ZeroSumNormal(
                "sigma_sector_offset", sigma=GROUP_EFFECT_SCALE, dims="sector"
            )
            log_sigma = log_sigma + sector_offset[idx_data["sector"]]
        # Six additive terms, several with unbounded coefficients: clip before
        # exponentiating or the product can overflow to inf mid-trajectory.
        log_sigma = pt.clip(log_sigma, *cfg.log_sigma_clip)
        sigma_isin = pm.Deterministic("sigma_isin", pt.exp(log_sigma), dims="isin")

        if cfg.enable_time_scale and T > 1:
            tau_free = pm.LogNormal(
                "sigma_time_free",
                mu=0.0,
                sigma=cfg.time_scale_prior_sigma,
                dims="time_step",
            )
            tau_time = pm.Deterministic(
                "sigma_time",
                pt.concatenate([tau_free, pt.ones(1)]),
                dims="time",
            )
        else:
            tau_time = pt.ones(T)
            pm.Deterministic("sigma_time", tau_time, dims="time")

        # ---- the marginal within-name covariance shape ------------------------
        # C is the correlation SHAPE -- level + OU state + white noise, summing to
        # 1 on the diagonal -- and ``tau`` rescales each lookback column. The
        # per-name scale enters as the multiplier sigma_i^2. Integrating the level
        # and the state out of the mean and into this matrix is the whole v2
        # change: 32,503 sampled parameters become ~60, the Neal funnel between
        # the scales and their latent fields disappears, and the gradient goes
        # from ~1.6 s to ~2.3 ms on the production panel.
        #
        # WHERE tau goes is load-bearing, and the 2026-08-18 run is what proved
        # it -- see ``KalmanModelConfig.time_scale_applies_to``. On the
        # covariance (the default) correlations are tau-free and only the
        # variances move; on the observation leg alone, raising a far column's
        # variance divides all of its correlations by sqrt(A_ss * A_tt), and that
        # family cannot reach this panel's measured structure at any parameter
        # value.
        def _cov(wl: Any, ws: Any, wo: Any) -> Any:
            """Within-name covariance shape for one variance composition."""
            if cfg.time_scale_applies_to == "covariance":
                shape = wl * pt.ones((T, T)) + ws * K + wo * pt.eye(T)
                return shape * tau_time[:, None] * tau_time[None, :]
            return wl * pt.ones((T, T)) + ws * K + wo * pt.diag(tau_time**2)

        # One matrix per scale bucket, each shared by every name in it — so the
        # Cholesky count tracks the number of GROUPS, not the number of names.
        A_by_bucket = [_cov(wl_vec[k], ws_vec[k], wo_vec[k]) for k in range(n_rho)]
        A_full = A_by_bucket[0] if n_rho == 1 else _cov(w_level, w_state, w_obs)
        pm.Deterministic("within_name_cov", A_full, dims=("time", "time_b"))

        if cfg.likelihood == "student_t":
            nu = pm.Deterministic(
                "nu", cfg.nu_floor + pm.Gamma("nu_tail", alpha=2.0, beta=0.1)
            )
        else:
            nu = None

        mean_full = mu_reg[:, None] + alpha_time[None, :]

        # ---- likelihood, one observed variable per covariance group ------------
        # Names sharing a missingness pattern (and optionally a coverage bucket)
        # share one Cholesky. Passing ``chol`` pre-formed as sigma_i * L is what
        # keeps this fast: PyMC then does a triangular solve rather than
        # factorising 6,497 matrices, which measured 1.35 s/gradient at n=2000.
        gain_full = pt.zeros(n_isin)
        var_full = pt.zeros(n_isin)
        for gi, (rows, cols, bkt) in enumerate(groups):
            Tg = len(cols)
            col_idx = pt.as_tensor_variable(cols.astype("int64"))
            row_idx = pt.as_tensor_variable(rows.astype("int64"))

            A_g = A_by_bucket[bkt][col_idx][:, col_idx]
            L_g = pt.linalg.cholesky(A_g + cfg.cholesky_jitter * pt.eye(Tg))

            sigma_g = sigma_isin[row_idx]
            mean_g = mean_full[row_idx][:, col_idx]
            chol_g = sigma_g[:, None, None] * L_g[None, :, :]

            obs_g = panel.Y[np.ix_(rows, cols)]
            name = f"target_pct_obs_g{gi}" if len(groups) > 1 else "target_pct_obs"
            if cfg.likelihood == "student_t":
                pm.MvStudentT(name, nu=nu, mu=mean_g, chol=chol_g, observed=obs_g)
            else:
                pm.MvNormal(name, mu=mean_g, chol=chol_g, observed=obs_g)

            # ---- Kalman smoother: recover the decision latent in closed form ---
            # With Cov(latent, y) = sigma_i^2 * c and Cov(y) = sigma_i^2 * A:
            #
            #   E[latent | y] = c' A^-1 (y - mean)          <- sigma_i^2 cancels
            #   Var[latent | y] = sigma_i^2 (w_L + w_S - c' A^-1 c)
            #
            # This IS the smoother step, and it is exact. It is strictly better
            # than sampling the state path was: Rao-Blackwellised over the latent
            # we removed, so it carries no Monte-Carlo error at all. Note the
            # asymmetry that matters downstream — the GAIN is common to the
            # group, but the posterior VARIANCE still scales with sigma_i, so the
            # per-name uncertainty the coverage-gradient gate checks is preserved.
            now_pos = int(np.flatnonzero(cols == (T - 1))[0])
            # Cov(latent_now, y_t) in units of sigma_i^2. The latent is the state
            # at *now*, on the now scale (tau is anchored at 1 there), so under
            # the covariance scaling each column's covariance with it carries that
            # column's tau -- the same factor its own variance carries. sigma_i^2
            # cancels out of the gain and tau cancels out of ``var_unit``
            # entirely, which is the assertion the self-test checks.
            c_g = (
                wl_vec[bkt] * pt.ones(Tg)
                + ws_vec[bkt] * K[col_idx][:, col_idx][:, now_pos]
            )
            if cfg.time_scale_applies_to == "covariance":
                c_g = c_g * tau_time[col_idx]
            Ainv_c = pt.linalg.solve_triangular(
                L_g.T,
                pt.linalg.solve_triangular(L_g, c_g, lower=True),
                lower=False,
            )
            gain_full = pt.set_subtensor(
                gain_full[row_idx],
                pt.dot(panel.Y[np.ix_(rows, cols)] - mean_g, Ainv_c),
            )
            # Gaussian conditional variance. Under the Student-t the exact
            # conditional carries an extra (nu + q_i) / (nu + Tg - 2) inflation
            # that depends on the name's own Mahalanobis distance; it is omitted
            # deliberately, because at the fitted nu it is a few per cent and
            # including it would make the reported per-name sd a function of how
            # surprising the name is, which reads as confidence where it is not.
            var_unit = pt.maximum(
                wl_vec[bkt] + ws_vec[bkt] - pt.dot(c_g, Ainv_c), cfg.cholesky_jitter
            )
            var_full = pt.set_subtensor(var_full[row_idx], var_unit * sigma_g**2)

        # ---- decision latents -------------------------------------------------
        # ``state_now_mean`` / ``state_now_sd`` describe the posterior of the
        # per-name latent at the snapshot. The workflow draws ``state_now`` from
        # them; keeping mean and sd separate is what makes the draw exact rather
        # than a by-product of having sampled the state.
        state_now = pm.Deterministic("state_now_mean", mu_reg + gain_full, dims="isin")
        pm.Deterministic("state_now_sd", pt.sqrt(var_full), dims="isin")
        pm.Deterministic("isin_level", gain_full, dims="isin")
        pm.Deterministic("expected_return", state_now, dims="isin")

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

    logger.info(
        "Built Kalman v2 model: %d ISIN x T=%d, %d drift features, %s",
        n_isin,
        T,
        n_drift,
        cfg.describe(),
    )
    return model


def resolve_screen_latent_v2(
    idata,
    *,
    latent: str = KALMAN_V2_SCREEN_LATENT,
    latent_sd: str = KALMAN_V2_SCREEN_LATENT_SD,
    include_latent_noise: bool = True,
    random_seed: int = 42,
):
    """Resolve the per-ISIN decision latent from a fitted idata.

    Every consumer goes through this rather than reading ``risk_adj_return`` or a
    raw variable, so the decision quantity has exactly one name and a rename is a
    one-line change. Mirrors v1's ``resolve_screen_latent``.

    Because v2 marginalises the level and state, the posterior stores the
    smoother's conditional **mean** and **sd** rather than a sampled path. With
    ``include_latent_noise=True`` this draws one latent value per posterior
    sample, so the returned array carries *both* parameter uncertainty and the
    residual uncertainty about where the name's latent actually sits. Setting it
    to ``False`` returns the conditional mean alone, which understates per-name
    spread and should only be used for point-estimate diagnostics.

    Parameters
    ----------
    idata
        Fitted inference object (``arviz.InferenceData`` or ``xarray.DataTree``).
    latent, latent_sd
        Variable names; default to the module constants.
    include_latent_noise
        Add the smoother's residual uncertainty. Default ``True``.
    random_seed
        Seed for the latent draw.

    Returns
    -------
    xarray.DataArray
        Dims ``(chain, draw, isin)``.

    Raises
    ------
    KeyError
        If the mean variable is absent, with the available names in the message.
    """
    post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
    if latent not in post:
        raise KeyError(
            f"{latent!r} not in posterior. Available: {sorted(map(str, post.data_vars))}"
        )
    mean = post[latent]
    if not include_latent_noise or latent_sd not in post:
        if include_latent_noise:
            logger.warning(
                "%s absent from the posterior; returning the conditional mean "
                "only, which understates per-name spread.",
                latent_sd,
            )
        return mean
    sd = post[latent_sd]
    rng = np.random.default_rng(random_seed)
    return mean + sd * rng.standard_normal(size=mean.shape)


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
    tau_true: tuple[float, ...] = (1.45, 1.20, 1.05),
    rho_slope_true: float = 0.8,
    seed: int = 42,
) -> tuple[KalmanPanelV2, dict[str, float]]:
    """Simulate a panel with a known level / state / noise split and time scale.

    Returns the panel and the truth dict, so the self-test can check recovery of
    the quantities this redesign exists to estimate.

    ``tau_true`` gives the per-lookback scale of the *whole* residual, oldest
    first and implicitly anchored at 1.0 for the snapshot — the structure
    :attr:`KalmanModelConfig.time_scale_applies_to` = ``'covariance'`` fits. It is
    simulated because the production path fits it: a self-test that generates a
    flat trail certifies a model nobody runs.

    ``rho_slope_true`` tilts each name's level/state *composition* by its scale
    index on the logit scale, which is the structure
    :attr:`KalmanModelConfig.rho_scale_buckets` fits. ``sigma_level_true`` and
    ``sigma_state_true`` then describe the **baseline** name, at index 0; a name
    a standard deviation noisier carries proportionally more permanent level.
    Set it to ``0.0`` to simulate the single global split.
    """
    rng = np.random.default_rng(seed)
    cfg = KalmanModelConfig(lookbacks=lookbacks)
    days = cfg.time_grid_days
    T = len(days)

    n_drift = 4
    X = rng.normal(size=(n_isin, n_drift))
    beta_true = np.array([0.30, -0.20, 0.10, 0.05])[:n_drift]
    mu_reg = X @ beta_true

    # Per-name variance composition, tilted by the scale index. The structured
    # variance is held FIXED across names — only its split moves — so the tilt
    # cannot be recovered from marginal spread and has to come from the shape of
    # the decay, which is the whole claim being tested.
    scale_index = rng.normal(size=n_isin)
    struct_var = sigma_level_true**2 + sigma_state_true**2
    rho_base = sigma_level_true**2 / struct_var
    base_logit = math.log(rho_base / (1.0 - rho_base))
    rho_i = 1.0 / (1.0 + np.exp(-(base_logit + rho_slope_true * scale_index)))

    level = rng.normal(size=n_isin) * np.sqrt(struct_var * rho_i)
    level -= level.mean()

    gaps = np.abs(np.diff(days))
    phi = np.exp(-gaps / ell_true)
    s = np.empty((n_isin, T))
    s[:, 0] = rng.normal(size=n_isin)
    for j in range(1, T):
        s[:, j] = phi[j - 1] * s[:, j - 1] + np.sqrt(1 - phi[j - 1] ** 2) * rng.normal(
            size=n_isin
        )
    s *= np.sqrt(struct_var * (1.0 - rho_i))[:, None]

    tau = np.asarray([*tau_true, 1.0], dtype="float64")
    if tau.shape[0] != T:
        raise ValueError(
            f"tau_true has {len(tau_true)} entries; {T - 1} are needed for "
            f"lookbacks={lookbacks!r} (the snapshot is anchored at 1.0)."
        )
    resid = (
        level[:, None]
        + s
        + rng.normal(scale=sigma_obs_true, size=(n_isin, T))
    )
    Y = mu_reg[:, None] + tau[None, :] * resid

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
        scale_index=scale_index,
    )
    truth = {
        "sigma_level": sigma_level_true,
        "sigma_state": sigma_state_true,
        "sigma_obs_base": sigma_obs_true,
        "ou_length_scale_days": ell_true,
        "rho_inf_implied": sigma_level_true**2
        / (sigma_level_true**2 + sigma_state_true**2),
    }
    truth.update({f"sigma_time[t{j}]": float(tau[j]) for j in range(T)})
    truth["rho_scale_slope"] = float(rho_slope_true)
    return panel, truth


def _selftest(draws: int = 500, tune: int = 500, chains: int = 2) -> int:
    """Fit a simulated panel and check the latent decomposition is recovered.

    Covers the level / state / noise split, the OU length-scale and the
    per-lookback time scale — every quantity the production posterior reports and
    the decision layer reads through.

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

    # ``sigma_time`` is a vector, so it is summarised as one variable and checked
    # per element. The snapshot entry is pinned at 1.0 by construction and is
    # included only so a mis-anchored grid shows up as a failure rather than as a
    # silently shifted scale.
    var_names = [
        "sigma_level",
        "sigma_state",
        "sigma_obs_base",
        "ou_length_scale_days",
        "rho_inf_implied",
        "sigma_time",
        "rho_scale_slope",
    ]
    names = [n for n in var_names if n != "sigma_time"]
    names += [f"sigma_time[t{j}]" for j in range(panel.n_time)]
    # ArviZ 1.x: ``ci_prob`` replaces ``hdi_prob``, and the interval columns are
    # named ``eti<pct>_lb`` / ``eti<pct>_ub`` rather than ``hdi_3%`` / ``hdi_97%``.
    # Resolve them by suffix so a future default change does not silently break
    # the check into a KeyError.
    summary = az.summary(idata, var_names=var_names, ci_prob=0.94)
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
    # ``sigma_time[t{T-1}]`` is a pinned constant: sd 0, ESS == total draws and
    # R-hat NaN by construction. Reporting it as the model's mixing would be
    # reporting an assumption.
    moving = summary.drop(index=f"sigma_time[t{panel.n_time - 1}]", errors="ignore")
    min_ess = float(moving["ess_bulk"].min())
    max_rhat = float(moving["r_hat"].max())
    print(f"\n  divergences {div}   min bulk ESS {min_ess:.0f}   max R-hat {max_rhat:.4f}")
    if failures:
        print(f"\nSELFTEST FAILED: {failures} parameter(s) outside the HDI")
        return 1
    print("\nSELFTEST PASSED: split, time scale and scale-dependent composition recovered")
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
