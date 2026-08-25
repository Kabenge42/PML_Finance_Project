"""Kalman price-target workflow, v2 — gate-driven, correlated-trail panel.

What this script is
-------------------
The v2 counterpart of ``pymc_kalman_filter_pt.py``. It runs the same Bayesian
workflow stages against :func:`KalmanFilterModel_v2.build_kalman_pt_model_v2`,
with three structural differences that between them are the reason a v2 exists.

**1. Every stage is a gate that returns a verdict.** v1 self-reports diagnostics
by logging warnings, which is how the screen drifted from 62 % to 81 % of names
above consensus between two runs without anything failing. Here each stage
returns :class:`GateResult` objects, ``main()`` aggregates them into a
:class:`GateReport`, and a run that fails a blocking gate exits non-zero and
refuses to export. The gate list *is* the acceptance criteria of the model, in
one readable place — see :data:`GATE_CATALOGUE`.

**2. The panel is measured before it is fitted.** A new stage (§4b,
:func:`run_panel_diagnostics`) computes the effective independent observations
per name and fits the empirical correlation kernel — both in seconds, before any
NUTS run. On 2026-08-18 that would have caught the shipped ``('3m','1m','1w')``
grid immediately: it scores ``T_eff = 1.19`` against 1.54 for ``('1y','6m','3m')``,
and the 30-minute fit that followed broke the R-hat gate.

**3. Configuration is split by what it changes.** :class:`KalmanModelConfig`
(in the model module) holds everything that changes the posterior;
:class:`KalmanRunConfigV2` holds everything that changes only the run. v1 mixed
them, which is how ``volume_penalty`` acquired one default in the builder and a
different one in ``main()``, and how the lookback grid — a first-order modelling
decision — came to live next to the figure width.

Stage map
---------
=======  ==================================  ==============================
Section  Function                            Stage (CLAUDE.md contract)
=======  ==================================  ==============================
1        :func:`load_kalman_frame`           Data
3        :func:`select_drift_features_v2`    Conceptual model building
4        :func:`prepare_panel`               Computational implementation
4b       :func:`run_panel_diagnostics`       **NEW** — panel information audit
6        :func:`run_prior_predictive`        Prior predictive
7        :func:`sample_posterior`            Fitting
8        :func:`run_posterior_predictive`    Model evaluation (PPC)
9        :func:`run_diagnostics`             Fitting diagnostics
9b       :func:`run_model_comparison`        Model comparison (opt-in, exact)
9b-fast  :func:`compare_arms_fast`           Model comparison (Max-and-Smooth screen)
10       :func:`run_screen`                  Decision analysis
10c      :func:`export_analytics`            Export
14       :func:`summarise`                   Summary + gate report
=======  ==================================  ==============================

There is no §5: the v1 legacy single-observation model has no v2 equivalent, and
the §10K/§12 pooled GRW sections are deliberately absent — they have been
unidentified in every review (``sigma_obs`` 1.05-1.19 against 0.028 for the same
code path on one name) and are not carried forward.

Usage
-----
.. code-block:: powershell

    . .\\set_env.ps1
    python pymc_kalman_filter_pt_v2.py --dry-run     # stages 1-4b only, seconds
    python pymc_kalman_filter_pt_v2.py               # full run, no export
    python pymc_kalman_filter_pt_v2.py --write       # full run + analytics export

``--dry-run`` is the one to reach for when changing ``lookbacks``: it answers
"is this panel worth fitting?" without fitting it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import uuid
import warnings
from dataclasses import dataclass, field, fields as dc_fields, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

# PyTensor backend guard MUST precede the first pymc/pytensor import.
from probabilistic_ml_model import _pytensor_env  # noqa: F401  (side effect)

from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
    GROUP_EFFECTS_FINE,
    KALMAN_V2_SCREEN_LATENT,
    KalmanModelConfig,
    KalmanPanelV2,
    apply_forecast_error_shrinkage,
    covariance_groups_for,
    build_kalman_pt_model_v2,
    effective_sample_size_of_panel,
    fit_trail_correlation_kernel,
    orthogonalise_family,
    resolve_screen_latent_v2,
)
from probabilistic_ml_model.pymc_models._workflow import (
    MIN_ESS_GATE,
    attach_log_likelihood,
    build_sample_kwargs,
    log_sample_diagnostics,
    posterior_predictive_check,
    prior_predictive_check,
)

logger = logging.getLogger("kalman_v2")

__all__ = [
    "KalmanRunConfigV2",
    "GateResult",
    "GateReport",
    "GATE_CATALOGUE",
    "load_kalman_frame",
    "DRIFT_COVERAGE_MIN",
    "DRIFT_SIGNAL_MIN_FLOOR",
    "drift_signal_min",
    "select_drift_features_v2",
    "screen_contrast_identities",
    "prepare_panel",
    "run_panel_diagnostics",
    "run_prior_predictive",
    "sample_posterior",
    "run_posterior_predictive",
    "run_diagnostics",
    "COMPARISON_ARMS",
    "subsample_panel_v2",
    "run_model_comparison",
    "compare_arms_fast",
    "PROVENANCE_COLUMNS",
    "stamp_export_provenance",
    "resolve_source_revision",
    "run_screen",
    "export_analytics",
    "main",
]

_EPS = 1e-12
_MV = "pml.mv_pymc_kalman_pt_v2"

#: Forward-return clip, carried over from v1 (``pymc_kalman_filter_pt.py:313``).
#: Applied in LOG space so it is sign-preserving and leaves ``prob_pos`` intact.
UPLIFT_CLIP_LO, UPLIFT_CLIP_HI = -0.95, 5.0
LOG_UPLIFT_CLIP_LO = float(np.log1p(UPLIFT_CLIP_LO))  # ~= -3.00
LOG_UPLIFT_CLIP_HI = float(np.log1p(UPLIFT_CLIP_HI))  # ~= +1.79

#: Ranking columns NULLed for an out-of-support row. Identity, price targets and
#: the raw ``er_*`` distribution are deliberately retained — the row is still
#: informative, it just must not be *ranked*.
_RANKING_COLS: tuple[str, ...] = (
    "expected_sharpe_ratio",
    "reward_to_cvar",
    "cvar_book_weight",
    # The probability column that actually orders names, so it is a ranking
    # metric and must be suppressed on a clip-pinned row like the others.
    # ``prob_pos`` is deliberately NOT here: it is pinned at 1.0 for most of the
    # universe, carries no ordering, and is retained as a reported diagnostic.
    "p_upside_pos_cond",
)

#: Columns the ``export_ranking_range`` gate bounds. Both the exported names and
#: the intermediate risk-book names, because v1 guards only the former -- which
#: is how an ``expected_sharpe_ratio`` of -2,142 once reached a SQL consumer.
#:
#: The short ``expected_sharpe`` alias was removed from this list on 2026-08-24
#: along with the column itself: ``compute_cvar_aware_book`` now emits
#: ``expected_sharpe_ratio`` only, so bounding a name nothing produces would
#: quietly bound nothing.
_RANKING_RANGE_COLS: tuple[str, ...] = (
    "expected_sharpe_ratio",
    "reward_to_cvar",
    "p_upside_pos_cond",
)

#: The canonical v2 analytics table. **v2-suffixed on purpose**: v1 and the live
#: GEIB dashboard keep reading ``analytics.kalman_filtered_price_targets``, so
#: the two models coexist and can be compared on one database. Promoting v2 is a
#: deliberate edit here plus a dashboard deploy — never a side effect of a run.
_ANALYTICS_TABLE_V2 = "kalman_filtered_price_targets_v2"
_RISK_BOOK_KEY = "10b_risk_book_v2"

#: Where the run's own gate verdicts land. Written by :func:`export_analytics`
#: after every other frame, so it captures the gates that function adds; and by
#: :func:`summarise` to CSV, so the early-abort paths that never reach an export
#: still leave a record. One row per gate, keyed on the same ``run_id`` as the
#: rest of the run's tables.
_GATE_REPORT_KEY = "09_gate_report_v2"

#: Every column this workflow stamps onto an exported frame so a reader can tell
#: which run — and which *source* — produced it.
#:
#: ``run_id`` and ``exported_at`` were stamped inline in three separate places
#: before 2026-08-22; this is the SSOT they now share, matching v1's
#: ``PROVENANCE_COLUMNS``.
#:
#: **Why the source revision was added.** Neither of the original pair says what
#: CODE produced the row, and writing the fourth edition of the post-run analysis
#: that gap cost a conclusion: run ``78801513e2cf`` was fitted from a working
#: tree about two hours before commit ``c08422d`` landed, and ``0121366fbabf``
#: was the first fit on committed source — so when the book's tail composition
#: swung 13 -> 17 names there was no way to establish from the artefacts whether
#: the two runs shared a specification at all. A three-run series was read as
#: estimator spread on the strength of a git log and a diffstat, which is not
#: provenance.
#:
#: It matters more now that :func:`run_model_comparison` decides questions by
#: contrasting arms across fits: an ELPD contrast between two runs whose source
#: is not pinned is not a contrast.
#:
#: ``source_dirty`` is **not an error flag** — most of these runs had an
#: uncommitted tree. It is the fact a reader needs in order to know which
#: cross-run comparisons are legitimate.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "exported_at",
    "source_sha",
    "source_dirty",
)


@lru_cache(maxsize=1)
def resolve_source_revision() -> tuple[Optional[str], Optional[bool]]:
    """Return ``(head_sha, dirty)`` for the working tree, or ``(None, None)``.

    Cached: it shells out to git, the answer cannot change mid-run, and every
    exported frame asks for it.

    Never raises. A missing git, a non-repository directory or a timeout all
    report ``(None, None)`` — the same "a failed export must not abort the
    workflow" rule the rest of this module follows. ``None`` is honest and
    readable; a fabricated SHA would not be.

    Returns
    -------
    tuple[str or None, bool or None]
        Full 40-character ``HEAD`` SHA, and whether tracked files differ from it.
    """
    import subprocess

    root = Path(__file__).resolve().parent

    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("git %s failed: %s", " ".join(args), exc)
            return None
        if out.returncode != 0:
            logger.debug("git %s exited %d", " ".join(args), out.returncode)
            return None
        return out.stdout

    sha = _git("rev-parse", "HEAD")
    if sha is None:
        logger.info(
            "source revision unavailable (not a git checkout?): exports carry a "
            "NULL source_sha."
        )
        return None, None
    status = _git("status", "--porcelain", "--untracked-files=no")
    dirty = None if status is None else bool(status.strip())
    return sha.strip() or None, dirty


def stamp_export_provenance(
    frame: pd.DataFrame, run_id: str, stamped: Any
) -> pd.DataFrame:
    """Return ``frame`` with :data:`PROVENANCE_COLUMNS` appended.

    Idempotent — re-stamping overwrites in place rather than duplicating.

    Parameters
    ----------
    frame
        Frame to stamp. Not mutated; a shallow copy is returned.
    run_id
        The run's identifier, shared by every table it writes.
    stamped
        Export timestamp (UTC), shared likewise.
    """
    sha, dirty = resolve_source_revision()
    out = frame.copy(deep=False)
    out["run_id"] = run_id
    out["exported_at"] = stamped
    out["source_sha"] = sha
    out["source_dirty"] = dirty
    return out


# =========================================================================== #
# §0  Gates                                                                   #
# =========================================================================== #


@dataclass(frozen=True)
class GateResult:
    """Verdict of one acceptance check.

    Attributes
    ----------
    name
        Stable identifier, matching a key of :data:`GATE_CATALOGUE`.
    passed
        Whether the check met its threshold.
    value
        The measured quantity, for the report table.
    threshold
        Human-readable statement of what would have passed.
    blocking
        A failed blocking gate aborts the run before export. Non-blocking gates
        are recorded and warned about but do not stop a run — used for checks
        that are informative but not yet trustworthy enough to gate on.
    detail
        Optional sentence of context shown under the row.
    """

    name: str
    passed: bool
    value: str
    threshold: str
    blocking: bool = True
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - formatting only
        mark = "PASS" if self.passed else ("FAIL" if self.blocking else "WARN")
        return f"[{mark:4}] {self.name:<28} {self.value:<24} (want {self.threshold})"


#: Every acceptance criterion the v2 workflow applies, with why it exists.
#: This is the model's contract with its consumers; adding a gate here and
#: emitting a :class:`GateResult` for it is how a new failure mode becomes
#: something the pipeline can catch instead of something a review discovers.
GATE_CATALOGUE: dict[str, str] = {
    "panel_t_eff": (
        "Effective independent observations per name. Below ~1.3 the trail is a "
        "near-duplicate of the snapshot and a T>1 likelihood over-counts evidence."
    ),
    "panel_kernel_fit": (
        "The empirical correlation decay must be describable by the level+OU form "
        "the model assumes. A poor fit means the model's latent is the wrong shape."
    ),
    "prior_scale": (
        "Prior-predictive implied upside must cover the empirical range without "
        "putting most of its mass outside it."
    ),
    "divergences": "NUTS divergences. Any non-zero value invalidates the geometry.",
    "r_hat": "Between-chain convergence. The project gate is < 1.01.",
    "ess_bulk": f"Bulk ESS floor, MIN_ESS_GATE = {MIN_ESS_GATE}.",
    "ppc_t_spread": (
        "Replicated INTERQUARTILE RANGE must contain the observed one. Measured "
        "robustly on purpose: comparing standard deviations under a Student-t "
        "mostly reports nu, since the t's variance is dominated by tail mass its "
        "density fit does not chase."
    ),
    "ppc_coverage": (
        "Per-time predictive interval coverage against its nominal target. "
        "Over-shooting means intervals are too wide, not that the model is safe."
    ),
    "ppc_decay": (
        "NEW: replicated data must reproduce the observed correlation decay, not "
        "just the marginal spread. A model can match the variance of a panel while "
        "getting its time structure completely wrong -- v1 did."
    ),
    "drift_contrast_leakage": (
        "NEW: no drift feature may correlate materially more strongly with a "
        "trail CONTRAST than with the response level -- 'materially' being "
        "CONTRAST_DOMINANCE_MARGIN, since the bare comparison fires on sampling "
        "noise. mu_reg is constant in t, so such content cannot enter the mean "
        "-- and under a correlated likelihood the contrast directions carry ~12x "
        "the weight of the level, so it is absorbed by distorting the covariance "
        "instead. That is the whole of the 2026-08-18 ppc_t_spread / "
        "ppc_coverage / ppc_decay failure, and it costs milliseconds to check "
        "before the fit."
    ),
    "drift_selection": (
        "NEW, reported not gated: how much margin the admitted drift columns "
        "have over the coverage and signal floors. A column enters the design "
        "matrix mean-imputed wherever it is NULL, so a thin one is not an "
        "error, it is an attenuated slope -- feat_eps_signal_surprise sat at "
        "79.1% coverage and |r| 0.007 and was fitted at beta +0.002."
    ),
    "model_comparison": (
        "NEW, reported not gated: ELPD contrast between named model arms, each "
        "refit on the same ISIN subsample with a post-hoc log_likelihood. The "
        "only stage that can decide whether a component EARNS its place rather "
        "than merely converging -- the v2 workflow row had it blank."
    ),
    "export_duplicate_content": (
        "WARN: no exported frame should carry the same QUANTITY under two "
        "names. The sibling of export_unique_columns, which checks names and "
        "therefore passes on `weight` beside a byte-identical `book_weight`. "
        "Not blocking: an all-zero column legitimately equals another all-zero "
        "column, and a warning must never cost a run a fit already paid for."
    ),
    "export_unique_columns": (
        "BLOCKING: no exported frame may carry the same column name twice. A "
        "duplicate makes df[col] a DataFrame rather than a Series, so the "
        "ranking-range gate dies with 'arg must be a list, tuple, 1-d array, or "
        "Series' AFTER the fit has been paid for. Any rename() mapping one "
        "existing column onto another existing one creates this silently."
    ),
    "model_comparison_fast": (
        "SCREENING ONLY, reported not gated: a Max-and-Smooth ELPD contrast. It "
        "scores arms on per-ISIN PSEUDO-OBSERVATIONS built from one baseline "
        "fit's covariance, not on the exact likelihood, and it costs seconds "
        "rather than one production fit per arm. Deliberately a different gate "
        "name from `model_comparison` so a screening verdict is never read as "
        "the exact one: use it to pick the candidate, then confirm with "
        "`--compare` before promoting anything."
    ),
    "mean_calibration": (
        "NEW: the slope of the response on the fitted mean must be ~1. An "
        "over-shrunk mean does not lose its missing variance, it parks it in "
        "Cov(mu, resid) -- and because mu is constant in t that is PERMANENT "
        "variance no replicate carries, which is exactly how run fa532b925732 "
        "failed ppc_decay (slope 1.230, 15.1% of Var(y_now)) while mean_spread "
        "read a healthy 0.33. mean_spread is one-sided and structurally cannot "
        "see this; that is why both exist."
    ),
    "mean_spread": (
        "NEW: the fitted mean may not have more variance than the response it "
        "predicts. Var(mu_reg) reached 2.9x Var(y_snapshot) on 2026-08-18 while "
        "every convergence gate passed; one line here would have caught it "
        "before the posterior predictive ran."
    ),
    "ppc_decay_residual": (
        "NEW, reported not gated: ppc_decay with the posterior-mean mean removed "
        "from both sides. A mean that is constant in time contributes the same "
        "constant at every gap, so it reads as a permanent level and hides "
        "whether a decay failure is in the mean or in the covariance."
    ),
    "shrinkage_slope": (
        "NEW: regressing expected upside on analyst-implied upside must give a "
        "slope near 1 and an intercept near 0. An intercept is a uniform offset "
        "applied to the whole universe, which is what took names-above-consensus "
        "from 62% to 81% between two runs with nothing failing."
    ),
    "coverage_gradient": (
        "NEW: posterior uncertainty must decrease monotonically with analyst "
        "coverage. A flat or inverted gradient means the hierarchy is not pricing "
        "information."
    ),
    "runtime_estimate": (
        "NEW: measured gradient cost x the NUTS budget must fit the runtime "
        "budget, on a compiled sampler. Catches in seconds what the 2026-08-18 "
        "run discovered 45 minutes in at 400 s/draw."
    ),
    "prob_pos_degenerate": (
        "NEW: warn when prob_pos is pinned at 1.0 for most of the universe. It "
        "cannot order a ranking in that state and p_upside_pos_cond should be "
        "used instead."
    ),
    "export_finite": (
        "Every exported ranking metric is finite and in range, and clip-pinned "
        "rows are flagged."
    ),
    "export_rowcount": (
        "Every curated frame is non-empty and agrees on row count. Catches the "
        "'table exists with zero rows' failure that passes a naive vintage check."
    ),
}


@dataclass
class GateReport:
    """Collected gate results for one run."""

    results: list[GateResult] = field(default_factory=list)

    def add(self, result: GateResult) -> GateResult:
        """Record a result and log it at the appropriate level."""
        self.results.append(result)
        if result.passed:
            logger.info("%s", result)
        elif result.blocking:
            logger.error("%s -- %s", result, result.detail or GATE_CATALOGUE.get(result.name, ""))
        else:
            logger.warning("%s -- %s", result, result.detail or "")
        return result

    @property
    def blocking_failures(self) -> list[GateResult]:
        """Failed gates that must stop the run."""
        return [r for r in self.results if not r.passed and r.blocking]

    @property
    def ok(self) -> bool:
        """True when no blocking gate failed."""
        return not self.blocking_failures

    def to_frame(self) -> pd.DataFrame:
        """Tabular form, for export alongside the run's artifacts."""
        return pd.DataFrame(
            [
                {
                    "gate": r.name,
                    "status": "PASS" if r.passed else ("FAIL" if r.blocking else "WARN"),
                    "value": r.value,
                    "threshold": r.threshold,
                    "blocking": r.blocking,
                    "detail": r.detail,
                    "rationale": GATE_CATALOGUE.get(r.name, ""),
                }
                for r in self.results
            ]
        )

    def render(self) -> str:
        """Multi-line human summary."""
        lines = ["", "=" * 78, "GATE REPORT", "=" * 78]
        lines += [str(r) for r in self.results]
        lines.append("-" * 78)
        if self.ok:
            lines.append(f"All {len(self.results)} gates passed (or warned non-blocking).")
        else:
            names = ", ".join(r.name for r in self.blocking_failures)
            lines.append(f"BLOCKING FAILURES: {names}")
        lines.append("=" * 78)
        return "\n".join(lines)


# =========================================================================== #
# §0b  Run configuration                                                      #
# =========================================================================== #


@dataclass(frozen=True)
class KalmanRunConfigV2:
    """Everything that changes how a run *executes*, not what it *infers*.

    The split from :class:`KalmanModelConfig` is the point. If two runs share a
    :class:`KalmanModelConfig` they are fitting the same model and their
    posteriors are comparable; if they differ only here, any difference is
    wall-clock or output location. v1 had no such boundary, so a lookback grid
    and a figure width lived in the same dataclass.

    ``from_env`` reads only the five variables the deployment actually sets;
    everything else is overridden programmatically with
    :func:`dataclasses.replace`.
    """

    # ---- NUTS budget -------------------------------------------------------
    draws: int = 2000
    tune: int = 4000
    chains: int = 4
    cores: int = 4
    target_accept: float = 0.9
    random_seed: int = 42
    #: Named explicitly, never left to PyMC's auto-selection. With
    #: ``compile_kwargs={"mode": Mode(linker="py")}`` in play — which
    #: ``build_sample_kwargs`` always sets — auto-selection disqualifies nutpie
    #: and lands on PyMC's own NUTS over the pure-Python VM. That path sampled
    #: the 2026-08-18 v2 run at 400 seconds per draw.
    nuts_sampler: str = "nutpie"
    #: Refuse to start a full-size fit whose projected wall clock exceeds this.
    #: Checked by the ``runtime_estimate`` gate against a measured gradient time,
    #: so a mis-specified model is caught in seconds rather than 45 minutes in.
    max_runtime_minutes: float = 90.0
    #: Gradient evaluations timed by the runtime estimate.
    benchmark_evals: int = 25

    # ---- predictive budgets ------------------------------------------------
    prior_draws: int = 1000
    ppc_draws: int = 1000

    # ---- universe filters --------------------------------------------------
    min_next_earnings: str = "2026-01-01"
    min_report_date: str = "2025-12-01"
    min_trail_obs: int = 2
    excluded_isins: tuple[str, ...] = ()
    #: Winsorise the pooled log-uplift response at these quantiles before
    #: standardising; ``None`` disables it.
    #:
    #: The response has a measured kurtosis of **34.35** against 3 for a Normal,
    #: with 0.69% of cells beyond |z|=4 and 23 beyond |z|=6. That is a
    #: data-quality property, not a signal, and handling it in the likelihood has
    #: failed twice in different directions: a Normal arm stalls a chain outright
    #: (step size 0, 1023 gradient evals), while the marginal Student-t is a
    #: per-NAME scale mixture, so one odd cell inflates a name's whole trail and
    #: pushes both the replicate spread and the measured within-name correlation
    #: too high. Clipping at the boundary lets the likelihood model the bulk,
    #: which is what it is for.
    #:
    #: At 0.5/99.5 this bounds log-uplift to about [-0.33, +1.12], i.e. returns
    #: of [-28%, +208%], clipping 1.00% of cells.
    #:
    #: **Default OFF, and measured.** Tried on the 2026-08-18 panel: it did not
    #: move the two calibration gates it was aimed at (t-spread 1.70-1.80 ->
    #: 1.77-1.87, decay rho_inf 0.66-0.74 -> 0.70-0.75, both marginally worse)
    #: and it *broke* a gate that had been passing — shrinkage slope fell 0.968
    #: -> 0.878 and the intercept rose 0.007 -> 0.027, because compressing the
    #: response compresses the regression onto it. Clipping the tails costs
    #: calibration of the bulk, which is the part the decision layer reads. Left
    #: in as a knob because it is cheap to re-test after a likelihood change.
    response_winsorise: Optional[tuple[float, float]] = None

    # ---- gate thresholds ---------------------------------------------------
    gate_t_eff_min: float = 1.30
    gate_kernel_rmse_max: float = 0.08
    gate_r_hat_max: float = 1.01
    gate_ess_min: int = MIN_ESS_GATE
    #: Percentiles bounding the replicated predictive interval in §8. The
    #: coverage target is **derived** from them (:attr:`gate_coverage_target`)
    #: rather than set independently, because the two were inconsistent: the
    #: interval has always been ``[3, 97]``, i.e. 94 % nominal, while the target
    #: was a hard-coded 0.92 carried over from a v1 statistic that used a
    #: different interval. That asks a *correctly calibrated* model for 92 %
    #: coverage of its own 94 % interval, so the gate fired on calibration rather
    #: than on miscalibration — measured 2026-08-18: 0.936-0.948 against a 0.94
    #: nominal, which is calibrated to within 0.008 and failed against 0.92.
    gate_coverage_percentiles: tuple[float, float] = (3.0, 97.0)
    gate_coverage_tol: float = 0.02
    gate_shrinkage_slope_lo: float = 0.80
    #: Ceiling on the model-on-consensus regression slope. **Lowered 1.20 ->
    #: 0.98 on 2026-08-20**, because the old band contained identity and so
    #: could not distinguish a working shrinkage estimator from a pass-through.
    #: Run ``49e84d7e9d59`` passed this gate at slope 0.979 / intercept +0.0051
    #: while reproducing the analyst consensus at Spearman 0.999995 and a median
    #: revision of 0.03pp — the exact failure the gate was written to catch,
    #: scored as a pass. A slope at or above 1 was never a legitimate outcome
    #: for a shrinkage estimator; only the *lower* bound was ever doing work.
    gate_shrinkage_slope_hi: float = 0.98
    #: Ceiling on ``|mean(expected_upside) - mean(implied_upside)|`` — the
    #: universe-wide shift the screen applies.
    #:
    #: **Replaced the raw y-intercept test on 2026-08-20, because that test was
    #: the slope in disguise.** Shrinking a cloud toward a centre ``c`` gives
    #: ``eu = c + slope*(iu - c)``, hence ``intercept = (1 - slope) * c`` as an
    #: algebraic identity. Measured across a nine-point multiplier sweep on the
    #: fitted run, ``intercept / (1 - slope)`` came out 0.2018, 0.2016, 0.2010,
    #: 0.2006, 0.2006, 0.2007 — i.e. exactly the response centre, every time.
    #: Any genuine shrinkage of a response centred at +20% therefore MUST show a
    #: positive intercept, and pairing ``|intercept| <= 0.02`` with the new
    #: ``rho <= 0.995`` left an EMPTY feasible set: the first needs slope >= 0.90,
    #: the second needs slope <= ~0.88.
    #:
    #: The failure the old threshold was written for — a run where names above
    #: consensus went 62% to 81% — is a shift of the centre, not a non-zero
    #: intercept, and that is what this measures. The gate still reports the
    #: intercept, as a diagnostic rather than as a pass condition.
    gate_shrinkage_center_shift_max: float = 0.02
    #: Ceiling on Spearman rho between model expected upside and analyst implied
    #: upside. Slope and intercept are both blind to a pass-through — an exact
    #: copy scores 1.0 and 0.0, which the band admits — so the gate needs a
    #: statistic that measures *disagreement* rather than calibration. The
    #: shipped model reorders the universe; a ranking identical to its own input
    #: is a fault regardless of how well-calibrated the regression looks.
    gate_shrinkage_rho_max: float = 0.995
    #: Floor on the median absolute revision from consensus, in percentage
    #: points. The scale-free companion to :attr:`gate_shrinkage_rho_max`: rho
    #: catches a rank-preserving copy, this catches a uniform rescale that
    #: reorders nothing of consequence. 0.25pp is two orders of magnitude above
    #: the 0.03pp of run ``49e84d7e9d59`` and an order below v1's 14%.
    gate_shrinkage_revision_min_pp: float = 0.25
    gate_decay_rmse_max: float = 0.10
    #: Ceiling on ``Var(mu_reg) / Var(y_snapshot)``. 1.0 is not a tuning choice:
    #: an additive mean with more variance than its response implies a negative
    #: residual variance, so anything above it is arithmetic, not fit.
    gate_mean_spread_max: float = 1.0
    #: Accepted range for the slope of ``y_now`` on the fitted mean. A
    #: calibrated mean gives exactly 1; +/-0.1 is about twice the spread the
    #: 2026-08-19 profile showed across the plausible ``signal_exponent``
    #: band, so it admits a fitted lambda anywhere in that band and still
    #: rejects the 1.230 that failed ppc_decay.
    gate_mean_calibration: tuple[float, float] = (0.90, 1.10)

    # ---- decision layer ----------------------------------------------------
    #: Shrink the decision latent toward the fitted mean by an explicit
    #: forecast-error term. See
    #: :func:`~probabilistic_ml_model.pymc_models.KalmanFilterModel_v2.apply_forecast_error_shrinkage`
    #: for why it is here and not in the likelihood. ``False`` restores the
    #: pre-2026-08-20 pass-through exactly, and is the comparison arm.
    enable_forecast_error_shrinkage: bool = True
    #: Scalar on the standard error of the analyst consensus mean. **This is a
    #: PRIOR, not an identified parameter** — the panel's own autocorrelation
    #: cannot separate forecast error from reporting noise, which is the whole
    #: reason the term is supplied rather than fitted. Only realised returns can
    #: estimate it, which is what ``scripts/score_panel_vintages.py`` is being
    #: built to do; until then, ``scripts/profile_forecast_error.py`` shows what
    #: each value does to the gates so the choice is auditable.
    #: **1.0, chosen on measurement, not on taste.** 1.0 means the forecast error
    #: is exactly the standard error of the consensus mean, with no inflation --
    #: the only value on the grid that needs no justification of its own.
    #:
    #: The first build defaulted to 2.0, picked to hit a target shrinkage. The
    #: production fit ``760751604647`` rejected it: slope 0.723 against a
    #: ``[0.80, 0.98]`` band, i.e. over-shrunk. Sweeping the multiplier offline
    #: against that run's own output (the shrinkage is post-posterior, so no
    #: refit is needed -- see ``scripts/profile_forecast_error.py``):
    #:
    #: ======  ========  ======  =====  ============
    #: kappa   median g  slope   rho    revision pp
    #: ======  ========  ======  =====  ============
    #: 0.50    0.966     0.944   0.999  0.28
    #: **1.00**  **0.876**  **0.858**  **0.992**  **1.00**
    #: 1.25    0.818     0.819   0.986  1.45
    #: 1.50    0.758     0.783   0.978  1.91
    #: 2.00    0.638     0.723   0.960  2.83
    #: ======  ========  ======  =====  ============
    #:
    #: The feasible band is roughly ``[0.85, 1.35]``: below it ``rho`` stays above
    #: its 0.995 ceiling (the screen is still a consensus sort), above it the
    #: slope falls out of the band. 1.0 sits mid-band and is the interpretable
    #: choice, so the two criteria agree.
    forecast_error_multiplier: float = 1.0
    #: Exponent on the analyst count in that standard error. 0.5 is the textbook
    #: standard error of the mean; raise it to steepen ``coverage_gradient``.
    #: **Do not pre-tune** — measure the gate first, since the shrinkage gain
    #: already inherits a coverage gradient at 0.5.
    forecast_error_n_exponent: float = 0.5
    #: Floor on ``tail_risk`` as a fraction of the name's own Monte-Carlo return
    #: sd. The absolute 1pp floor in :mod:`RiskBookModel` binds for 13.4 % of the
    #: universe and 14 of 25 book names on run ``49e84d7e9d59``, turning STARR
    #: into ``100 x expected_upside`` for exactly the names whose simulated 5 %
    #: quantile happens to be positive. A relative floor charges a name for the
    #: dispersion it has instead of rewarding it for having no simulated loss.
    tail_risk_vol_floor_k: float = 0.25
    mc_horizon: int = 4
    mc_rho: float = 0.85
    cvar_alpha: float = 0.05
    weight_cap: float = 0.10
    k_book: int = 25
    #: Baseline long-probability threshold. Scaled by the universe-average
    #: kalman_gain inside the risk book to give ``p_long_cond``, which is what
    #: ``p_upside_pos_cond`` is actually tested against.
    p_long: float = 0.50
    #: Market-cap pre-selection threshold for long-book eligibility: a name is
    #: eligible when ``mcap_global_r < mcap_global_r_max``.
    #:
    #: **Renamed from ``mcap_country_r_max`` on 2026-08-22.** The old name said
    #: *country* while the column it is compared against — and has always been
    #: compared against, in ``RiskBookModel`` and in v1 before it — is
    #: ``mcap_global_r``. Two different rank bases, one name, and no way to tell
    #: from the call site which was meant. The column is the thing that cannot
    #: move (``feat_mcap_global_r`` is the MV's contract and the size-tilt
    #: driver), so the knob is what changes. :attr:`mcap_country_r_max` remains
    #: as a read-only alias for one release; ``replace(cfg, mcap_country_r_max=…)``
    #: raises rather than silently setting nothing.
    mcap_global_r_max: float = 0.01

    # ---- model comparison (§9b) --------------------------------------------
    #: Run the ELPD comparison stage. **Off by default and deliberately not in
    #: `from_env`**: each arm is a full production fit plus a pointwise
    #: `log_likelihood`, so an N-arm comparison costs roughly N times a normal
    #: run. It is a decision you make once about a component, not something a
    #: production export should ever trip over.
    #: Run the §9b-fast Max-and-Smooth SCREEN over comparison arms.
    #:
    #: Unlike :attr:`enable_model_comparison` this reuses the production fit that
    #: has just completed instead of refitting per arm, so it costs seconds. It
    #: ranks arms; it does not decide them — take the winner to ``--compare``.
    #: Off by default only because a screen nobody reads is noise in the gate
    #: report, not because it is expensive.
    enable_fast_comparison: bool = False
    #: Arms for the fast screen. Empty means every registered arm that the Max
    #: step can see (``drift_strict`` is skipped: it changes the design matrix).
    fast_comparison_arms: tuple[str, ...] = ()
    enable_model_comparison: bool = False
    #: ISIN subsample each arm is fitted on. The `log_likelihood` group is
    #: `chains x draws x n_isin x T` floats — at the full 6.5k panel that is
    #: hundreds of MB per arm, held simultaneously because `az.compare` needs
    #: every arm at once. 800 keeps the contrast well-powered (the arms differ
    #: in structure, not in a per-name effect) at a fraction of the memory.
    comparison_max_isins: int = 800
    #: Arms to run when `enable_model_comparison` is set. Keys of
    #: :data:`COMPARISON_ARMS`. Empty means every registered arm.
    comparison_arms: tuple[str, ...] = ()

    # ---- output ------------------------------------------------------------
    results_dir: Optional[str] = None

    # ---- figures -----------------------------------------------------------
    #: Draw the §4b/§6/§8/§9/§10/§10b panels and write the statistics tables
    #: beside them. v2 shipped with none of this, which is why every chart in the
    #: post-run analysis was hand-drawn and went stale between runs.
    #:
    #: Figures are written AFTER the export, never before, and every panel is
    #: individually wrapped -- a plotting failure must not cost a run the
    #: analytics write it has already paid a fit for. ``--no-figures`` turns the
    #: set off for a run where only the tables are wanted.
    export_figures: bool = True

    #: Target figure width in px, matching v1's ``PML_FIG_WIDTH_PX`` knob so one
    #: environment variable sizes both workflows. Read by the shared figure layer
    #: through the resolver ``kalman_viz_v2.install`` hands it.
    fig_width_px: int = 1150
    write_analytics: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Reject the decision-layer knobs that would fail silently downstream.

        A negative multiplier gives a negative variance and a gain above 1, i.e.
        anti-shrinkage; a slope band that does not contain a shrinkage estimator
        makes the gate unsatisfiable. Both produce plausible-looking numbers
        several stages later rather than an error here.
        """
        if self.forecast_error_multiplier < 0:
            raise ValueError(
                "forecast_error_multiplier must be non-negative, got "
                f"{self.forecast_error_multiplier!r}"
            )
        if self.forecast_error_n_exponent < 0:
            raise ValueError(
                "forecast_error_n_exponent must be non-negative, got "
                f"{self.forecast_error_n_exponent!r}"
            )
        if self.tail_risk_vol_floor_k < 0:
            raise ValueError(
                f"tail_risk_vol_floor_k must be non-negative, got "
                f"{self.tail_risk_vol_floor_k!r}"
            )
        if not self.gate_shrinkage_slope_lo < self.gate_shrinkage_slope_hi:
            raise ValueError(
                "gate_shrinkage_slope band must be increasing, got "
                f"[{self.gate_shrinkage_slope_lo}, {self.gate_shrinkage_slope_hi}]"
            )
        if not 0.0 < self.gate_shrinkage_rho_max <= 1.0:
            raise ValueError(
                "gate_shrinkage_rho_max must be in (0, 1], got "
                f"{self.gate_shrinkage_rho_max!r}"
            )

    @property
    def mcap_country_r_max(self) -> float:
        """Deprecated alias of :attr:`mcap_global_r_max`.

        The old name claimed a country-relative rank; the comparison has always
        been against ``mcap_global_r``. Read-only on purpose — a writable alias
        on a frozen dataclass would let ``replace()`` appear to work while
        setting nothing.
        """
        warnings.warn(
            "KalmanRunConfigV2.mcap_country_r_max is deprecated; it was renamed "
            "to mcap_global_r_max because the threshold is compared against the "
            "mcap_global_r column, not a country rank.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mcap_global_r_max

    @property
    def gate_coverage_target(self) -> float:
        """Nominal coverage of the §8 predictive interval, as a fraction.

        Derived from :attr:`gate_coverage_percentiles` so the target and the
        interval it grades cannot disagree.
        """
        lo, hi = self.gate_coverage_percentiles
        return (hi - lo) / 100.0

    @classmethod
    def from_env(cls) -> "KalmanRunConfigV2":
        """Build from the environment variables the deployment sets.

        ``PML_FIG_WIDTH_PX`` joins the set on 2026-08-24. v1 has always honoured
        it; v2 ignored it because it had no figures to size, and leaving it
        unread once v2 gained them would mean one knob sizing one of two
        workflows.
        """

        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            try:
                return int(raw) if raw else default
            except ValueError:
                logger.warning("%s=%r is not an int; using %d", name, raw, default)
                return default

        return cls(
            random_seed=_int("RANDOM_SEED", 42),
            results_dir=os.environ.get("KALMAN_PT_RESULTS_DIR") or None,
            write_analytics=os.environ.get("KALMAN_PT_SQL_EXPORT", "1") != "0",
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            fig_width_px=_int("PML_FIG_WIDTH_PX", 1150),
        )

    @property
    def results_path(self) -> Path:
        """Root of the artifact tree."""
        return Path(self.results_dir or "pymc_kalman_filter_pt_v2_results")


# =========================================================================== #
# §1  Data                                                                    #
# =========================================================================== #


def kalman_v2_query(run_cfg: KalmanRunConfigV2) -> str:
    """SQL for the v2 modelling frame.

    Notes
    -----
    Exclusions are parameterised rather than inlined. v1 grew two bare
    ``AND isin <> '...'`` literals inside the query string with no comment, which
    is an unexplained data decision in a place nobody reviews. Anything dropped
    here must come from :attr:`KalmanRunConfigV2.excluded_isins` so it appears in
    the run's own configuration record.
    """
    excl = ""
    if run_cfg.excluded_isins:
        joined = ", ".join(f"'{i}'" for i in run_cfg.excluded_isins)
        excl = f"\n             AND isin NOT IN ({joined})"
    return f"""
        SELECT *
        FROM {_MV}
        WHERE feat_log_uplift_now IS NOT NULL
          AND size_class <> 'n/a'
          AND n_trail_obs >= {run_cfg.min_trail_obs}
          AND next_earnings >= '{run_cfg.min_next_earnings}'
          AND income_statement_report_date >= '{run_cfg.min_report_date}'{excl}
          
    """


def load_kalman_frame(run_cfg: KalmanRunConfigV2) -> pd.DataFrame:
    """Load the v2 feature matrix.

    Returns
    -------
    pandas.DataFrame
        One row per ISIN.

    Raises
    ------
    RuntimeError
        If ``DB_URL`` is unset or the query returns nothing — both are hard
        failures, not conditions to warn about and continue past.
    """
    from sqlalchemy import create_engine

    url = os.environ.get("DB_URL")
    if not url:
        raise RuntimeError("DB_URL is not set; run `. .\\set_env.ps1` first.")
    engine = create_engine(url)
    frame = pd.read_sql(kalman_v2_query(run_cfg), engine)
    if frame.empty:
        raise RuntimeError(
            f"{_MV} returned no rows under the configured filters. "
            "Has pml.refresh_kalman_pt_v2() been run?"
        )
    logger.info("Loaded %d names x %d columns from %s", len(frame), frame.shape[1], _MV)
    return frame


# =========================================================================== #
# §3  Feature selection                                                       #
# =========================================================================== #

#: Columns barred from the drift design matrix, with the reason each is out.
#: The single SSOT for v2 exclusions — the v1 partition (leakage, wideners, tilt
#: drivers, support counters, rating counts, collinear legs, Piotroski
#: components, ``days_*``) plus three v2 additions.
DRIFT_EXCLUSIONS: dict[str, str] = {
    # -- leakage: these ARE the response, or trivially determine it -----------
    "feat_implied_upside": "the response in level form",
    "feat_log_uplift_now": "the response",
    "feat_log_uplift_1w": "response trail",
    "feat_log_uplift_1m": "response trail",
    "feat_log_uplift_3m": "response trail",
    "feat_log_uplift_6m": "response trail",
    "feat_log_uplift_1y": "response trail",
    # -- observation-scale drivers: they belong in sigma, not mu --------------
    "feat_pt_noise_sigma": "noise widener",
    "feat_pt_range_norm": "noise widener (sigma_delta_range)",
    "feat_vol_level": "noise widener (sigma_delta_vol_level)",
    "feat_log_mcap": "noise widener (sigma_delta_log_mcap)",
    "feat_vol_drift": "provenance container; corr -0.035 with log|residual|",
    # -- tilt drivers: they enter risk_adj_return additively ------------------
    "feat_avg_beta": "risk tilt driver",
    "feat_rel_volume": "volume tilt driver",
    "feat_mcap_global_r": "size tilt driver",
    # -- support counters, not signals ---------------------------------------
    "feat_pt_drift_n": "support counter",
    "feat_price_drift_n": "support counter",
    "feat_vol_drift_n": "support counter",
    "feat_net_eps_drift_n": "support counter",
    # -- collinear composition legs (one representative kept) ----------------
    "feat_analyst_bullish_pct": "composition leg of feat_analyst_rating",
    "feat_analyst_bearish_pct": "composition leg of feat_analyst_rating",
    "feat_analyst_neutral_pct": "composition leg of feat_analyst_rating",
    "feat_analyst_conviction": "composition leg of feat_analyst_rating",
    "feat_holds": "raw rating count",
    "feat_buys": "raw rating count",
    "feat_sells": "raw rating count",
    "feat_no_opinion": "raw rating count",
    "feat_pt_high_drift": "collinear with feat_pt_drift",
    "feat_pt_low_drift": "collinear with feat_pt_drift",
    "feat_pt_median_drift": "collinear with feat_pt_drift",
    # -- Piotroski components (median kept) ----------------------------------
    "feat_piotroski_f_score_fy": "component of the median composite",
    "feat_piotroski_f_score_neg1fy": "component of the median composite",
    "feat_piotroski_f_score_neg2fy": "component of the median composite",
    "feat_piotroski_f_score_neg3fy": "component of the median composite",
    # -- trail-contrast identities (2026-08-18) -------------------------------
    # feat_log_uplift = log(price_target) - log(price), so the DIFFERENCES of the
    # response trail are log(PT_t / PT_s) - log(P_t / P_s). These six are the two
    # legs of that identity: three price-return windows and three price-target
    # drift terms. Each correlates more strongly with a trail CONTRAST than with
    # the response level, which is the measured rule `screen_contrast_identities`
    # applies and the `drift_contrast_leakage` gate enforces.
    #
    # They were harmless in v1 (|beta| <= 0.09) because a factorised likelihood
    # never weights a contrast. The v2 correlated likelihood weights the
    # (1w, now) contrast -- sd 0.248 against a level sd of 0.934 -- by
    # 1/(1-rho) ~ 12x, and `mu_reg` is constant in t, so that content cannot be
    # expressed in the mean and can only be absorbed by distorting the
    # covariance. On the 2026-08-18 run it was: |beta| reached 1.414 against a
    # next-largest of 0.077 and Var(mu_reg) reached 2.50 against a response
    # variance of 0.87, which is the whole of the ppc_t_spread / ppc_coverage /
    # ppc_decay failure.
    "feat_one_day_return": "trail-contrast identity: corr 0.03 level, -0.46 (now-1w)",
    "feat_price_chg_pct_3m": "trail-contrast identity: corr 0.40 level, -0.67 (now-3m)",
    "feat_price_drift": "trail-contrast identity: corr -0.20 level, -0.33 (now-1y)",
    "feat_pt_drift": "trail-contrast identity: corr -0.00 level, +0.35 (now-3m)",
    "feat_pt_accuracy_1y": "trail-contrast identity: corr 0.02 level, -0.23 (now-1w)",
    "feat_pt_noise_drift": "trail-contrast identity: corr 0.02 level, +0.24 (now-3m)",
    # -- v2 additions ---------------------------------------------------------
    # The four raw EPS legs below are superseded by the consolidated block
    # mv_pymc_kalman_pt_v2 now emits: feat_eps_signal_surprise (the two _pct
    # legs, rescaled /100), feat_eps_signal_beat (the two beat rates) and
    # feat_eps_signal_coverage. They are three columns rather than one because
    # the legs sit on three scales -- percent, share, ratio -- and averaging
    # across them just reproduces whichever leg carries the largest units.
    #
    # feat_net_eps_drift is NOT excluded: it is the trend leg of that same
    # block, already a raw decimal ratio, so it enters the drift matrix on its
    # own rather than being folded into an average. Its support counter
    # feat_net_eps_drift_n stays excluded above.
    "feat_last_q_surprise": "52.1% NULL, |beta| 0.0003; -> feat_eps_signal_surprise",
    "feat_last_y_surprise": "23.2% NULL, |beta| 0.0054; -> feat_eps_signal_surprise",
    "feat_eps_beat_rate": "45.4% NULL, |beta| 0.0115; -> feat_eps_signal_beat",
    "feat_eps_beat_rate_annual": "14.6% NULL, |beta| 0.0013; -> feat_eps_signal_beat",
    "feat_mcap_country_sec_r": "near-duplicate of feat_mcap_country_r",
    "feat_mcap_region_r": "near-duplicate of feat_mcap_country_r",
    "feat_mcap_region_sec_r": "near-duplicate of feat_mcap_country_r",
    "feat_mcap_global_sec_r": "near-duplicate of feat_mcap_country_r",
    # -- the consolidated EPS block, one leg retired (2026-08-21) -------------
    # Run 37e6d8966250 fitted this at beta = +0.0020 with an 89 % ETI of
    # [-0.0092, +0.0132]. Measured against the response on the same universe it
    # is the ONLY drift column failing either admission test, and it fails both
    # by a wide margin -- see DRIFT_COVERAGE_MIN for the full table.
    #
    # The mechanism is not that earnings surprise is uninformative in general;
    # it is that ``_standardise`` z-scores NaN-safely and then fills NaN with
    # 0.0, i.e. imputes the column mean. At 79.1 % coverage a fifth of the
    # universe is pinned at the mean, which attenuates the slope toward zero
    # before the sampler sees it.
    #
    # NOT replaced. ``eps_gaap_est_avg_rev_pct_fy1e_{1w,1m,3m,6m}`` were the
    # proposed substitutes on coverage grounds (84.5-87.0 %) and were measured
    # and rejected: |r(level)| 0.008 / 0.010 / 0.020 / 0.037 against 0.116 for
    # feat_eps_signal_beat and 0.173 for feat_eps_signal_coverage, which are the
    # two legs that would have been retired to make room. ``_1y`` and the
    # non-GAAP ``_6m``/``_1y`` are trail-contrast identities (dominance 7.02 /
    # 4.41 / 14.99) that `drift_contrast_leakage` would reject anyway, and the
    # GAAP ``_1m``/``_3m`` pair correlates 0.969, so at most one could enter.
    # Coverage alone was the wrong criterion; do not reopen this from coverage
    # figures without the correlations beside them.
    "feat_eps_signal_surprise": "79.1% cov, |r(level)| 0.007, contrast dominance 4.53",
}

#: Prefixes barred wholesale. ``feat_total_return_`` and ``feat_tr_cagr_`` are 18
#: windows of one construct that is mechanically anti-correlated with implied
#: upside (a stock that rallied has less distance to its target) — v1 measured
#: Spearman -0.545 and correctly assigned near-null coefficients, which is a lot
#: of design-matrix width to spend on a known accounting identity. See
#: :data:`MOMENTUM_REPRESENTATIVES` for the re-admission hook, which is currently
#: empty.
DRIFT_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "days_",
    "feat_total_return_",
    "feat_tr_cagr_",
    "feat_piotroski_f_score_neg",
)

#: Momentum representatives kept despite the prefix ban. **Empty since
#: 2026-08-18**: both former members (``feat_price_chg_pct_3m``,
#: ``feat_one_day_return``) are trail-contrast identities and are excluded above,
#: which takes precedence anyway since :func:`select_drift_features_v2` tests
#: :data:`DRIFT_EXCLUSIONS` first. Kept as the hook for re-admitting a future
#: representative — the prefix ban is about design-matrix width, and a
#: measurement that survives the contrast screen is welcome back through here.
MOMENTUM_REPRESENTATIVES: tuple[str, ...] = ()


#: Minimum non-NULL share for a column to enter the drift design matrix.
#:
#: **Why a floor is needed at all.** :func:`_standardise` is NaN-safe by
#: z-scoring and then filling NaN with ``0.0`` — that is, imputing the column
#: mean. A thinly-covered column therefore does not fail loudly; it arrives with
#: a large block of rows pinned at its own mean, which attenuates its slope
#: toward zero and spends a design-matrix column on nothing. That is a
#: measurement problem, not a modelling preference, which is why this is a
#: selection rule rather than a judgement call per column.
#:
#: **Calibrated 2026-08-21** against run ``37e6d8966250``'s own universe
#: (n = 6,533, ``n_trail_obs >= 2``). ``dominance`` is the
#: :func:`screen_contrast_identities` statistic, repeated here so the three
#: admission tests can be read together:
#:
#: =================================  =====  ==========  =========
#: feature                            cov %  |r(level)|  dominance
#: =================================  =====  ==========  =========
#: ``feat_pt_achievement_1y``          97.0      0.5146       0.23
#: ``feat_analyst_rating``             99.5      0.4651       0.10
#: ``feat_mcap_country_r``            100.0      0.3230       0.13
#: ``feat_eps_signal_coverage``       100.0      0.1731       0.14
#: ``feat_median_piotroski_f_score``  100.0      0.1358       0.14
#: ``feat_pt_range_hit_rate``         100.0      0.1313       0.47
#: ``feat_eps_signal_beat``            86.3      0.1156       0.93
#: ``feat_coverage_drift``            100.0      0.1053       0.53
#: ``feat_net_eps_drift``             100.0      0.0862       0.37
#: -- both thresholds sit here --      80.0      0.0247       1.50
#: ``feat_eps_signal_surprise``        79.1      0.0071       4.53
#: =================================  =====  ==========  =========
#:
#: One column fails, and it fails all three. The gaps either side are wide —
#: 79.1 to 86.3 on coverage, 0.0071 to 0.0862 on signal — so neither threshold
#: is fitted to its boundary case: both can move a long way without changing a
#: verdict on this universe.
#:
#: The rule is the safety net for *future* candidates. Columns already known to
#: fail are named in :data:`DRIFT_EXCLUSIONS`, which is tested first, so a
#: retired column cannot silently re-enter when its coverage drifts back across
#: 80 % on a later refresh.
DRIFT_COVERAGE_MIN: float = 0.80

#: Floor on ``|corr(column, response)|`` for drift-matrix entry, as a multiple of
#: the sampling null rather than a bare literal.
#:
#: A correlation estimated on ``n`` rows carries se ~ ``1 / sqrt(n)``, so
#: ``2 / sqrt(n)`` is roughly "distinguishable from zero at two standard
#: errors". Writing the threshold as a constant would silently mean something
#: different on a panel of a different size — tighter on a large one, and on a
#: small one it would admit columns that are pure noise.
#:
#: The absolute floor of 0.02 keeps the rule meaningful if the universe ever
#: grows large enough for the sampling null to fall below what is worth a design
#: column at all. At n = 6,533 the binding term is ``2 / sqrt(n)`` = 0.0247.
DRIFT_SIGNAL_MIN_FLOOR: float = 0.02


def drift_signal_min(n: int) -> float:
    """Return the minimum ``|r|`` a drift column must clear on ``n`` rows.

    Parameters
    ----------
    n
        Number of rows the correlation is estimated on.

    Returns
    -------
    float
        ``max(DRIFT_SIGNAL_MIN_FLOOR, 2 / sqrt(n))`` — see
        :data:`DRIFT_SIGNAL_MIN_FLOOR`.
    """
    if n < 2:
        return float("inf")
    return max(DRIFT_SIGNAL_MIN_FLOOR, 2.0 / math.sqrt(float(n)))


def select_drift_features_v2(
    frame: pd.DataFrame,
    *,
    response_col: str = "feat_log_uplift_now",
) -> list[str]:
    """Resolve the drift design matrix columns from the frame.

    Per the stage contract the *catalogue* decides which columns exist; this
    function decides which of them belong in the state-transition mean. The two
    are different questions and v1 conflated them once, by flipping a
    ``pymc_role`` to ``'excluded'`` in SQL — which drops the row from
    ``vw_pymc_feature_catalogue`` while the MV still emits the column, and
    ``assert_pymc_catalogue_coverage()`` then raises ``MISSING_FROM_CATALOGUE``.
    Exclusions live here, in Python, always.

    Four tests, applied in order. The first two are *named*: a column somebody
    decided about, recorded in :data:`DRIFT_EXCLUSIONS` or barred by family in
    :data:`DRIFT_EXCLUDED_PREFIXES`. The last two are *measured*, and exist so a
    column nobody has looked at yet cannot enter unexamined — a coverage floor
    (:data:`DRIFT_COVERAGE_MIN`) and a signal floor (:func:`drift_signal_min`).
    Both orders matter: named first, so a retired column stays retired even if a
    later refresh moves it back across a threshold.

    The measured pair is deliberately *not* the contrast screen, which runs
    separately in :func:`screen_contrast_identities` on the standardised
    response after this returns. That one asks whether a column's content can
    legally enter a mean that is constant in ``t``; these two ask whether the
    column carries enough of anything to be worth a design column at all.

    Parameters
    ----------
    frame
        The loaded modelling frame.
    response_col
        Column the signal floor measures against. When it is absent from
        ``frame`` the signal test is skipped and the omission is logged — never
        silently passed, since skipping it admits every candidate.

    Returns
    -------
    list[str]
        Column names, sorted for stability.
    """
    candidates = [c for c in frame.columns if c.startswith("feat_")]
    kept: list[str] = []
    dropped: dict[str, str] = {}

    response: Optional[pd.Series] = None
    if response_col in frame.columns:
        response = pd.to_numeric(frame[response_col], errors="coerce")
    else:
        logger.warning(
            "%s absent from the frame; the drift SIGNAL floor is skipped this "
            "run and only coverage is enforced",
            response_col,
        )

    n_rows = len(frame)
    signal_min = drift_signal_min(n_rows) if response is not None else 0.0

    for col in candidates:
        if col in DRIFT_EXCLUSIONS:
            dropped[col] = DRIFT_EXCLUSIONS[col]
            continue
        if col.startswith(DRIFT_EXCLUDED_PREFIXES) and col not in MOMENTUM_REPRESENTATIVES:
            dropped[col] = "prefix-excluded family"
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            dropped[col] = "non-numeric"
            continue

        series = pd.to_numeric(frame[col], errors="coerce")
        coverage = float(series.notna().mean()) if n_rows else 0.0
        if coverage < DRIFT_COVERAGE_MIN:
            dropped[col] = (
                f"coverage {coverage:.1%} < {DRIFT_COVERAGE_MIN:.0%} "
                f"(mean-imputed rows attenuate the slope)"
            )
            continue

        if response is not None:
            corr = series.corr(response)
            corr = 0.0 if not np.isfinite(corr) else abs(float(corr))
            if corr < signal_min:
                dropped[col] = (
                    f"|r({response_col})| {corr:.4f} < {signal_min:.4f} "
                    f"(2/sqrt(n), n={n_rows})"
                )
                continue

        kept.append(col)

    kept = sorted(kept)
    logger.info(
        "Drift features: %d kept, %d excluded (of %d feat_ columns)",
        len(kept),
        len(dropped),
        len(candidates),
    )
    for col, reason in sorted(dropped.items()):
        logger.debug("  dropped %-34s %s", col, reason)
    logger.debug("kept: %s", kept)
    return kept


#: Correlation below which the contrast screen declines to judge. Without it a
#: feature uncorrelated with everything (0.002 against 0.003, say) would be
#: flagged on noise, since the rule is a ratio of two small numbers.
CONTRAST_CORR_FLOOR: float = 0.05

#: How far the contrast correlation must *exceed* the level correlation before
#: the screen calls a feature an identity leg.
#:
#: :data:`CONTRAST_CORR_FLOOR` guards the absolute size of the two correlations;
#: it does nothing about their *ratio*, and the rule is a ratio. Without a
#: margin the test is ``|corr_contrast| > |corr_level|``, which any feature
#: whose two correlations are equal up to sampling noise passes half the time.
#: At n ~ 6.5k names a correlation carries se ~ 1/sqrt(n) = 0.013, so a 0.014
#: excess -- one standard error -- is not evidence of anything.
#:
#: Calibrated 2026-08-19 by re-admitting the six known identity legs to the
#: design matrix and measuring both correlations un-rotated against the same
#: universe the gate runs on (6499 names, T = 4). ``dominance`` is
#: ``|corr_contrast| / max(|corr_level|, CONTRAST_CORR_FLOOR)``:
#:
#: ===============================  =========  ==========
#: feature                          dominance  identity?
#: ===============================  =========  ==========
#: ``feat_pt_drift``                     7.17  yes
#: ``feat_pt_noise_drift``               5.09  yes
#: ``feat_pt_accuracy_1y``               3.26  yes
#: ``feat_price_drift``                  1.97  yes
#: ``feat_one_day_return``               1.88  yes
#: ``feat_price_chg_pct_3m``             1.65  yes
#: -- margin sits here --                1.50
#: ``feat_eps_signal_beat``              1.14  no
#: ``feat_median_piotroski_f_score``     0.62  no
#: ``feat_coverage_drift``               0.46  no
#: ===============================  =========  ==========
#:
#: The two populations are separated by a factor of 1.45 with nothing in
#: between, so the margin is not fitted to the boundary case: it can move
#: anywhere in ``(1.15, 1.65)`` without changing a single verdict on this
#: universe. Raise it only with a measurement like the one above, never to
#: clear a failing run.
CONTRAST_DOMINANCE_MARGIN: float = 1.5


def screen_contrast_identities(
    Y: np.ndarray,
    X: np.ndarray,
    names: Sequence[str],
    time_days: np.ndarray,
) -> pd.DataFrame:
    """Measure each drift feature against the response *level* and its *contrasts*.

    The model's mean is ``mu_reg[i] + alpha_time[t]`` — constant in ``t`` up to an
    offset shared by the whole universe. A feature therefore has exactly one
    legal channel: the level. Whatever it knows about a **contrast**
    ``y[i, now] - y[i, lb]`` cannot be expressed in the mean, and under a
    correlated likelihood it is not simply ignored either — the contrast
    directions are the small eigenvalues of the within-name covariance, so they
    carry ``1 / (1 - rho)`` times the weight of the level. Content the mean
    cannot hold and the likelihood weights heavily is content that gets absorbed
    by distorting something else.

    That is not a hypothetical. On the 2026-08-18 run the six features this
    screen flags took ``|beta|`` to 1.414 against a next-largest of 0.077 and
    ``Var(mu_reg)`` to 2.50 against a response variance of 0.87, which accounts
    for the whole of the ``ppc_t_spread`` / ``ppc_coverage`` / ``ppc_decay``
    failure. They are the two legs of an accounting identity —
    ``feat_log_uplift = log PT - log P``, so its increments are
    ``Δ log PT - Δ log P`` — and the design matrix was carrying both.

    This function only *measures*; the exclusions themselves live in
    :data:`DRIFT_EXCLUSIONS` so a run's design matrix is reproducible from the
    source rather than re-derived from whatever universe it happened to load.
    The ``drift_contrast_leakage`` gate is what enforces the rule.

    Run it on the design matrix **before** :func:`orthogonalise_family`. A
    rotation mixes an identity column into its neighbours, so on the rotated
    basis the flag lands on whichever principal axis inherited the contrast
    content rather than on the column responsible for it — measured on the
    2026-08-18 design, the rotated view flags ``pt_hist_pc1`` and clears
    ``feat_pt_drift``, which is the wrong answer to act on.
    :func:`prepare_panel` therefore runs it pre-rotation and stores the result on
    :attr:`KalmanPanelV2.contrast_screen`.

    Parameters
    ----------
    Y
        Standardised response matrix, shape ``(n_isin, T)``.
    X
        Standardised design matrix, shape ``(n_isin, p)``, pre-rotation.
    names
        Column names of ``X``.
    time_days
        Calendar offsets, shape ``(T,)``, oldest first and ending at 0.

    Returns
    -------
    pandas.DataFrame
        One row per drift feature, sorted worst first:
        ``feature``, ``corr_level``, ``corr_contrast``, ``contrast_gap_days``,
        ``dominance``, ``is_identity``.

    Notes
    -----
    A feature is an identity when its **dominance**

    .. code-block:: text

        dominance = abs(corr_contrast) / max(abs(corr_level), CONTRAST_CORR_FLOOR)

    exceeds :data:`CONTRAST_DOMINANCE_MARGIN`. The margin is the whole of the
    test's tolerance for sampling noise: the un-margined form
    ``abs(corr_contrast) > max(abs(corr_level), CONTRAST_CORR_FLOOR)`` fires on a
    one-standard-error excess, which on 2026-08-19 flagged
    ``feat_eps_signal_beat`` at -0.101 level against -0.115 contrast -- a
    difference of 0.014 where a correlation on 6.5k names carries se 0.013. That
    feature is an EPS beat rate; it contains no price and no price-target term,
    so it cannot be a leg of ``Delta log PT - Delta log P`` by construction, and
    the six features that are sit at dominance 1.65 and above. See the constant
    for the calibration table.

    Both correlations are computed pairwise-complete, and the contrast is taken
    against the snapshot column because that is the one every name has and every
    decision reads.
    """
    Y = np.asarray(Y, dtype="float64")
    X = np.asarray(X, dtype="float64")
    names = list(names)
    T = Y.shape[1]
    snapshot = Y[:, T - 1]
    contrasts = {float(time_days[j]): snapshot - Y[:, j] for j in range(T - 1)}

    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return float("nan")
        xs, ys = x[ok], y[ok]
        if xs.std() < _EPS or ys.std() < _EPS:
            return 0.0
        return float(np.corrcoef(xs, ys)[0, 1])

    rows: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        x = X[:, j]
        level = _corr(x, snapshot)
        best_gap, best_corr = float("nan"), 0.0
        for gap, contrast in contrasts.items():
            r = _corr(x, contrast)
            if math.isfinite(r) and abs(r) > abs(best_corr):
                best_gap, best_corr = gap, r
        reference = max(abs(level) if math.isfinite(level) else 0.0, CONTRAST_CORR_FLOOR)
        dominance = abs(best_corr) / reference
        rows.append(
            {
                "feature": name,
                "corr_level": level,
                "corr_contrast": best_corr,
                "contrast_gap_days": best_gap,
                "dominance": dominance,
                "is_identity": dominance > CONTRAST_DOMINANCE_MARGIN,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["is_identity", "dominance"], ascending=False
    ).reset_index(drop=True)


# =========================================================================== #
# §4  Panel preparation                                                       #
# =========================================================================== #


def _standardise(a: np.ndarray) -> np.ndarray:
    """Z-score, NaN-safe, with a zero-variance guard."""
    a = np.asarray(a, dtype="float64")
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    if not math.isfinite(sd) or sd < _EPS:
        return np.zeros_like(a)
    out = (a - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_panel(
    frame: pd.DataFrame,
    model_cfg: KalmanModelConfig,
    run_cfg: Optional[KalmanRunConfigV2] = None,
    *,
    drift_names: Optional[Sequence[str]] = None,
) -> KalmanPanelV2:
    """Assemble a :class:`KalmanPanelV2` from the loaded frame.

    The response trail comes straight from the MV's ``feat_log_uplift_*``
    columns — v2 does not rebuild it in Python, so the definition of the modelled
    quantity has one home. The trail is standardised by a *single* pooled mean
    and sd across all time columns, not per column: per-column standardisation
    would remove exactly the level differences between lookbacks that
    ``alpha_time`` exists to estimate.

    Parameters
    ----------
    frame
        Loaded modelling frame.
    model_cfg
        Supplies ``lookbacks``, hence which trail columns are read and in what
        order, and whether the PT-history family is rotated.
    drift_names
        Optional explicit column list; defaults to :func:`select_drift_features_v2`.

    Returns
    -------
    KalmanPanelV2
    """
    lookbacks = (*model_cfg.lookbacks, "now")
    missing = [lb for lb in lookbacks if f"feat_log_uplift_{lb}" not in frame.columns]
    if missing:
        raise ValueError(
            f"{_MV} does not emit feat_log_uplift_{{{','.join(missing)}}}. "
            "Add the column to the MV rather than reconstructing it in Python."
        )

    Y_raw = np.column_stack(
        [pd.to_numeric(frame[f"feat_log_uplift_{lb}"], errors="coerce") for lb in lookbacks]
    ).astype("float64")

    # The snapshot anchors the OU grid and is the only cell every downstream
    # decision reads, so a name without it is not a panel member. The MV's WHERE
    # clause already enforces this; the filter is here so a hand-built frame
    # cannot reach the model and trip its assertion instead.
    has_snapshot = np.isfinite(Y_raw[:, -1])
    if not has_snapshot.all():
        logger.warning(
            "Dropping %d name(s) with no snapshot observation", int((~has_snapshot).sum())
        )
        frame = frame.loc[has_snapshot].reset_index(drop=True)
        Y_raw = Y_raw[has_snapshot]

    # Winsorise BEFORE standardising, so the clip is defined on the
    # interpretable log-uplift scale and the reported mean/sd describe the data
    # the model actually sees.
    if run_cfg is not None and run_cfg.response_winsorise is not None:
        lo_q, hi_q = run_cfg.response_winsorise
        finite = Y_raw[np.isfinite(Y_raw)]
        lo, hi = np.quantile(finite, [lo_q, hi_q])
        n_clipped = int(((finite < lo) | (finite > hi)).sum())
        Y_raw = np.clip(Y_raw, lo, hi)
        logger.info(
            "Winsorised the response at q%.1f/%.1f -> log-uplift [%.4f, %.4f] "
            "(returns [%.1f%%, %.1f%%]); %d of %d cells clipped (%.2f%%)",
            lo_q * 100, hi_q * 100, lo, hi,
            np.expm1(lo) * 100, np.expm1(hi) * 100,
            n_clipped, finite.size, 100 * n_clipped / max(finite.size, 1),
        )

    # Pooled standardisation — see the docstring.
    mu = float(np.nanmean(Y_raw))
    sd = float(np.nanstd(Y_raw))
    if not math.isfinite(sd) or sd < _EPS:
        raise ValueError("Response trail has zero variance; check the MV.")
    Y = (Y_raw - mu) / sd

    names = list(drift_names) if drift_names is not None else select_drift_features_v2(frame)
    X = np.column_stack([_standardise(frame[c].to_numpy()) for c in names])

    # Screen BEFORE rotating: an orthonormal rotation mixes a trail-contrast
    # identity into its neighbours, so on the rotated basis the flag lands on
    # whichever principal axis inherited the contrast content instead of on the
    # column responsible for it.
    contrast_screen = screen_contrast_identities(Y, X, names, model_cfg.time_grid_days)

    rotation: Optional[np.ndarray] = None
    sources: tuple[str, ...] = ()
    if model_cfg.orthogonalise_pt_history:
        X, names, rotation, sources = orthogonalise_family(X, names)

    def _col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in frame.columns:
            logger.warning("%s absent from the frame; using %.1f", name, default)
            return np.full(len(frame), default)
        return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy()

    # ---- per-lookback analyst coverage -------------------------------------
    # mv_pymc_kalman_pt_v2 emits n_analysts_{lb} for every trail column. v1 used
    # a single snapshot count for all T, which charges a 4-analyst consensus
    # from a year ago the same measurement precision as a 30-analyst one from
    # today. Two uses here: the trail-average feeds `precision_weight` (so the
    # per-name scale reflects the whole trail, not just today), and the profile
    # is handed to the model for covariance bucketing.
    snapshot_n = np.clip(_col("n_analysts", 1.0), 1.0, None)
    cov_cols: list[np.ndarray] = []
    for lb in lookbacks:
        col = "n_analysts" if lb == "now" else f"n_analysts_{lb}"
        cov_cols.append(np.clip(_col(col, np.nan), 1.0, None) if col in frame.columns
                        else np.full(len(frame), np.nan))
    coverage = np.column_stack(cov_cols).astype("float64")
    n_missing_cov = int(np.isnan(coverage).all(axis=1).sum())
    if n_missing_cov:
        logger.warning(
            "%d name(s) have no per-lookback analyst counts; falling back to the "
            "snapshot count for them", n_missing_cov,
        )
    # Fill gaps with the snapshot count, then normalise so the snapshot column
    # is 1.0 — the model wants a *relative* profile, not a level.
    coverage = np.where(np.isnan(coverage), snapshot_n[:, None], coverage)
    coverage_profile = coverage / np.maximum(coverage[:, [-1]], 1.0)

    observed = np.isfinite(Y)
    trail_avg_n = np.nanmean(np.where(observed, coverage, np.nan), axis=1)
    trail_avg_n = np.clip(np.nan_to_num(trail_avg_n, nan=1.0), 1.0, None)

    # The variance-composition driver. A data-only stand-in for
    # ``log sigma_isin``, which the covariance-group partition needs *before* any
    # coefficient is sampled. Built from the three strongest terms of the scale
    # model itself, at unit weight: dispersion, volatility and analyst support.
    # Measured against the fitted log sigma on the 2026-08-18 universe at Pearson
    # 0.966 / Spearman 0.967, and its quintiles reproduce the residual-decay
    # pattern the split exists to capture. Alternatives measured and rejected:
    # feat_vol_level alone (0.928, but its quintiles are non-monotone in residual
    # rho_inf) and log1p(cv) alone (0.572).
    vol_level_z = _standardise(
        np.log1p(np.clip(_col("feat_vol_level", 0.0), 0.0, None))
    )

    pt_sd = _col("feat_pt_noise_sigma", 0.0)
    pt_level = np.abs(_col("observed_pt", 1.0))
    # log1p(cv) is NaN for cv < -1 and the dispersion is a ratio of magnitudes,
    # so clamp it non-negative at source rather than guarding inside the model.
    cv = np.where(pt_level > _EPS, pt_sd / np.maximum(pt_level, _EPS), 0.0)
    cv = np.clip(np.nan_to_num(cv, nan=0.0, posinf=0.0), 0.0, 5.0)

    scale_index = _standardise(
        np.log1p(cv) + vol_level_z - np.log(np.sqrt(trail_avg_n))
    )

    coord_cols = [c for c in model_cfg.group_effects if c in frame.columns]
    coord_uniques: dict[str, np.ndarray] = {}
    coord_idx: dict[str, np.ndarray] = {}
    for col in coord_cols:
        codes, uniques = pd.factorize(frame[col].astype(str), sort=True)
        coord_uniques[col] = np.asarray(uniques)
        coord_idx[col] = codes.astype("int32")

    panel = KalmanPanelV2(
        frame=frame,
        isins=frame["isin"].astype(str).to_numpy(),
        Y=Y,
        time_days=model_cfg.time_grid_days,
        X_drift=X,
        drift_names=names,
        dispersion_cv=cv,
        precision_weight=np.sqrt(trail_avg_n),
        coverage_profile=coverage_profile,
        vol_level=vol_level_z,
        scale_index=scale_index,
        log_mcap=_standardise(_col("feat_log_mcap", 0.0)),
        range_norm=_standardise(_col("feat_pt_range_norm", 0.0)),
        avg_beta=_standardise(_col("feat_avg_beta", 1.0)),
        size_ratio=_standardise(_col("feat_mcap_global_r", 0.5)),
        volume_ratio=_standardise(_col("feat_rel_volume", 1.0)),
        coord_uniques=coord_uniques,
        coord_idx=coord_idx,
        response_mean=mu,
        response_std=sd,
        orthogonal_rotation=rotation,
        orthogonal_source_names=sources,
        contrast_screen=contrast_screen,
    )
    logger.info(
        "Panel: %d names x T=%d, %d drift features (response mean %.4f sd %.4f)",
        panel.n_isin,
        panel.n_time,
        len(names),
        mu,
        sd,
    )
    return panel


# =========================================================================== #
# §4b  Panel information audit  — NEW STAGE                                   #
# =========================================================================== #


def run_panel_diagnostics(
    panel: KalmanPanelV2,
    run_cfg: KalmanRunConfigV2,
    report: GateReport,
) -> dict[str, Any]:
    """Measure how much information the panel actually carries, before fitting.

    Two questions, both answerable in seconds and neither asked by v1:

    1. **How many independent observations per name are there?** If the answer is
       close to 1 the trail is a near-duplicate of the snapshot, and a likelihood
       that treats it as ``T`` independent reads will over-count the evidence.
    2. **Does the correlation decay have the shape the model assumes?** v2's
       latent is a permanent level plus an exponentially-decaying state, which
       implies ``r(gap) = rho_inf + (1 - rho_inf) exp(-gap/ell)``. If that form
       does not fit, the model is the wrong shape and no amount of sampling will
       help.
    3. **Is every drift feature something the mean can actually hold?** A
       feature that measures a trail *contrast* rather than the response level
       has no legal channel in a mean that is constant in time, and a correlated
       likelihood weights it far too heavily to ignore. See
       :func:`screen_contrast_identities`.

    Returns
    -------
    dict[str, Any]
        ``t_eff``, ``rho_inf``, ``ell_days``, ``half_life_days``, ``rmse``,
        ``corr`` (the full ``T x T`` matrix) and ``per_step_coverage``.
    """
    t_eff = panel.effective_t()
    report.add(
        GateResult(
            name="panel_t_eff",
            passed=t_eff >= run_cfg.gate_t_eff_min,
            value=f"T_eff = {t_eff:.2f} of T = {panel.n_time}",
            threshold=f">= {run_cfg.gate_t_eff_min:.2f}",
            detail=(
                "Widen the lookback grid. Measured optima on the 2026-08-18 "
                "universe: ('1y','6m','3m') = 1.54, ('6m','3m','1m') = 1.35, "
                "('3m','1m','1w') = 1.19."
            ),
        )
    )

    out: dict[str, Any] = {"t_eff": t_eff}
    try:
        kern = fit_trail_correlation_kernel(panel.Y, panel.time_days)
        out.update(kern)
        report.add(
            GateResult(
                name="panel_kernel_fit",
                passed=kern["rmse"] <= run_cfg.gate_kernel_rmse_max,
                value=(
                    f"rho_inf {kern['rho_inf']:.3f}, ell {kern['ell_days']:.0f}d, "
                    f"rmse {kern['rmse']:.4f}"
                ),
                threshold=f"rmse <= {run_cfg.gate_kernel_rmse_max}",
                detail=(
                    f"Implied split: {kern['rho_inf'] * 100:.0f}% permanent level, "
                    f"{(1 - kern['rho_inf']) * 100:.0f}% decaying "
                    f"(half-life {kern['half_life_days']:.0f}d). Compare against "
                    "the posterior's rho_inf_implied after fitting."
                ),
            )
        )
    except ValueError as exc:
        report.add(
            GateResult(
                name="panel_kernel_fit",
                passed=False,
                value=str(exc)[:60],
                threshold="3+ distinct calendar gaps",
                blocking=False,
                detail="T is too small or the grid too regular to identify the kernel.",
            )
        )

    screen = panel.contrast_screen
    if screen is not None and not screen.empty:
        out["contrast_screen"] = screen
        offenders = screen.loc[screen["is_identity"]]
        if offenders.empty:
            worst = screen.iloc[0]
            detail = (
                "every feature is measured on the level, as the mean requires. "
                f"Closest to the margin: {worst.feature} at dominance "
                f"{worst.dominance:.2f} ({worst.corr_level:+.2f} level vs "
                f"{worst.corr_contrast:+.2f} at a {worst.contrast_gap_days:.0f}d gap)."
            )
        else:
            detail = (
                "; ".join(
                    f"{r.feature} {r.corr_level:+.2f} level vs {r.corr_contrast:+.2f} "
                    f"at a {r.contrast_gap_days:.0f}d gap (dominance {r.dominance:.2f})"
                    for r in offenders.itertuples()
                )
                + " -- add the offenders to DRIFT_EXCLUSIONS; they cannot be "
                "fitted by a mean that is constant in time."
            )
        report.add(
            GateResult(
                name="drift_contrast_leakage",
                passed=offenders.empty,
                value=(
                    f"{len(offenders)} of {len(screen)} feature(s) measure the "
                    f"trail's increments (max dominance "
                    f"{screen['dominance'].max():.2f})"
                ),
                threshold=(
                    f"|corr(contrast)| <= {CONTRAST_DOMINANCE_MARGIN} x "
                    f"max(|corr(level)|, {CONTRAST_CORR_FLOOR})"
                ),
                detail=detail,
            )
        )

    # ---- what the admission rule let through -------------------------------
    # Reported before the fit, next to the contrast screen, because all three
    # admission tests are cheap and a design matrix is easier to argue about
    # while it is still a list of names than after 40 minutes of NUTS.
    #
    # Non-blocking on purpose: the rule already removed the offenders inside
    # select_drift_features_v2, so by the time this runs there is nothing left
    # to fail on. It exists to make the margin VISIBLE — a surviving column
    # sitting at 80.4 % coverage is a different situation from one at 97 %, and
    # only one of them is a refresh away from silently changing the model.
    # `drift_names` carries POST-rotation names, and orthogonalise_family
    # renames the PT_HISTORY_FAMILY members to composites that have no frame
    # column. Measure the ones that are still real columns and say how many were
    # skipped rather than reporting a coverage figure for a linear combination.
    n_rows = len(panel.frame)
    measurable = [c for c in panel.drift_names if c in panel.frame.columns]
    n_rotated = len(panel.drift_names) - len(measurable)
    coverages = {
        c: float(pd.to_numeric(panel.frame[c], errors="coerce").notna().mean())
        for c in measurable
    }
    if coverages:
        cov_at = min(coverages, key=coverages.__getitem__)
        cov_min = coverages[cov_at]
        out["drift_coverage"] = coverages
        report.add(
            GateResult(
                name="drift_selection",
                passed=True,
                blocking=False,
                value=(
                    f"{len(panel.drift_names)} column(s) admitted; thinnest "
                    f"{cov_at} at {cov_min:.1%}"
                    + (f" ({n_rotated} rotated, not measurable)" if n_rotated else "")
                ),
                threshold=(
                    f"coverage >= {DRIFT_COVERAGE_MIN:.0%} and "
                    f"|r(response)| >= {drift_signal_min(n_rows):.4f}"
                ),
                detail=(
                    "Reported, not gated: select_drift_features_v2 has already "
                    "applied both floors. A column near the coverage threshold "
                    "is one refresh away from leaving the design matrix, which "
                    "changes the model without changing any code."
                ),
            )
        )

    corr = pd.DataFrame(panel.Y).corr().to_numpy()
    out["corr"] = corr
    out["per_step_coverage"] = panel.observed_mask.mean(axis=0)
    logger.info(
        "Panel audit: T_eff %.2f, per-step coverage %s",
        t_eff,
        np.round(out["per_step_coverage"], 3).tolist(),
    )
    return out


# =========================================================================== #
# §6  Prior predictive                                                        #
# =========================================================================== #


def run_prior_predictive(
    model: Any,
    panel: KalmanPanelV2,
    run_cfg: KalmanRunConfigV2,
    report: GateReport,
) -> Any:
    """Draw the prior predictive and check it on the *interpretable* scale.

    De-standardises the implied log-uplift back to a percentage upside so the
    check is against something a person can judge. The v1 prior put visible mass
    at +200 % against an empirical peak near +10 %, which is the kind of thing
    that is obvious in percent and invisible in standardised units.
    """
    # The grouped likelihood names its observed variables target_pct_obs_g{k},
    # so ask the model rather than assuming a single name. ``prior_predictive_check``
    # drops names that are absent, which would silently leave nothing to check.
    obs_names = [
        str(v) for v in model.observed_RVs if str(v.name).startswith("target_pct_obs")
    ]
    idata = prior_predictive_check(
        model,
        var_names=[*obs_names, KALMAN_V2_SCREEN_LATENT, "sigma_level", "sigma_state"],
        draws=run_cfg.prior_draws,
        random_seed=run_cfg.random_seed,
    )
    try:
        key = next(
            (k for k in idata.prior_predictive.data_vars if str(k).startswith("target_pct_obs")),
            None,
        )
        if key is None:
            raise KeyError("no target_pct_obs* variable in the prior predictive")
        rep = np.asarray(idata.prior_predictive[key]).ravel()
        # Clip in LOG space before expm1. A Student-t prior with a free scale
        # produces draws far into the tail, and expm1 of those overflows to inf
        # (the RuntimeWarning the 2026-08-18 run emitted). Clipping at the same
        # bounds the decision layer uses keeps the check on a comparable scale
        # instead of letting a handful of infinities define the percentiles.
        upside = np.expm1(
            np.clip(
                rep * panel.response_std + panel.response_mean,
                LOG_UPLIFT_CLIP_LO,
                LOG_UPLIFT_CLIP_HI,
            )
        )
        obs = np.expm1(
            np.clip(
                panel.Y[np.isfinite(panel.Y)] * panel.response_std + panel.response_mean,
                LOG_UPLIFT_CLIP_LO,
                LOG_UPLIFT_CLIP_HI,
            )
        )
        lo, hi = np.nanpercentile(upside, [5, 95])
        obs_lo, obs_hi = np.nanpercentile(obs, [5, 95])
        # The prior should be wider than the data but not absurdly so: a prior
        # 90% interval more than 10x the empirical one is not weakly informative,
        # it is uninformative in a way that leaks into the posterior tails.
        ratio = (hi - lo) / max(obs_hi - obs_lo, _EPS)
        report.add(
            GateResult(
                name="prior_scale",
                passed=1.0 <= ratio <= 10.0,
                value=f"prior/empirical 90% width = {ratio:.1f}x",
                threshold="1x - 10x",
                blocking=False,
                detail=(
                    f"prior 90% upside [{lo:+.1%}, {hi:+.1%}] vs empirical "
                    f"[{obs_lo:+.1%}, {obs_hi:+.1%}]"
                ),
            )
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("prior-predictive scale check failed: %s", exc)
    return idata


# =========================================================================== #
# §7  Sampling                                                                #
# =========================================================================== #


def _resolve_sampler(run_cfg: KalmanRunConfigV2) -> tuple[str, dict[str, Any]]:
    """Resolve the sampler name and the exact ``pm.sample`` kwargs.

    Single source of truth for the sampling configuration, so the
    ``runtime_estimate`` gate measures the same thing :func:`sample_posterior`
    runs. Keeping these in two places is how a gate ends up certifying a
    configuration that never executes.

    Returns
    -------
    tuple[str, dict[str, Any]]
        ``(sampler_name, sample_kwargs)``.
    """
    env = describe_sampler_environment()
    sampler = (run_cfg.nuts_sampler or "nutpie").lower()
    if sampler == "nutpie" and not env["nutpie_ok"]:
        logger.error(
            "nutpie %s is below PyMC's 0.16.10 minimum, so this run falls back to "
            "the 'pymc' sampler on the pure-Python VM — the configuration that "
            "sampled at 400 s/draw on 2026-08-18. Install nutpie>=0.16.10, or "
            "launch under the interpreter that has it.",
            env["nutpie_version"],
        )
        sampler = "pymc"

    kwargs = build_sample_kwargs(
        samples=run_cfg.draws,
        tune=run_cfg.tune,
        chains=run_cfg.chains,
        cores=run_cfg.cores,
        target_accept=run_cfg.target_accept,
        random_seed=run_cfg.random_seed,
        nuts_sampler=sampler,
        model_name="KalmanPriceTargetV2",
    )
    # ``idata_kwargs`` raises a deprecation FutureWarning and nutpie ignores it;
    # use _workflow.attach_log_likelihood post hoc instead.
    kwargs.pop("idata_kwargs", None)
    # Mode(linker="py") is what disqualifies nutpie from auto-selection in the
    # first place. With the sampler named it is popped by PyMC anyway; removing
    # it here makes the intent visible at the call site.
    if sampler == "nutpie":
        kwargs.pop("compile_kwargs", None)
    return sampler, kwargs


def run_runtime_estimate(
    model: Any,
    run_cfg: KalmanRunConfigV2,
    report: GateReport,
) -> dict[str, Any]:
    """Time the gradient and refuse to start a fit that cannot finish.

    The 2026-08-18 v2 run was aborted after 45 minutes, having produced eight
    draws. Everything needed to predict that was available in under a second:
    the free-parameter count, the resolved sampler, and the cost of one gradient
    evaluation. This gate measures all three *before* sampling.

    The projection assumes a healthy ~2^5 leapfrog steps per draw. A model with
    bad geometry will exceed it — but a model with bad geometry is a failure
    anyway, and the sibling ``divergences`` / ``r_hat`` gates catch that after
    the fact. The purpose here is to catch the case where a *single gradient* is
    so expensive that no amount of good geometry could rescue the run.

    Returns
    -------
    dict[str, Any]
        ``ms_per_grad``, ``n_free_params``, ``projected_minutes``, ``sampler``,
        ``linker``, ``nutpie_version``.
    """
    import time

    import pytensor

    # Resolve the sampler and the compile mode EXACTLY as sample_posterior will,
    # then time the gradient under that mode. Timing under the default mode
    # would be dishonest in precisely the case this gate exists to catch: the
    # PyMC sampler keeps build_sample_kwargs' Mode(linker="py"), so a numba-timed
    # gradient would report a runtime the run will never achieve.
    sampler, sample_kwargs = _resolve_sampler(run_cfg)
    mode = (sample_kwargs.get("compile_kwargs") or {}).get("mode")
    linker = (
        type(mode.linker).__name__
        if mode is not None
        else type(pytensor.compile.mode.get_default_mode().linker).__name__
    )
    degraded = sampler == "pymc" and linker.lower().startswith("py")

    with model:
        point = model.initial_point()
        n_free = int(sum(np.asarray(v).size for v in point.values()))
        dlogp = model.compile_dlogp(**({"mode": mode} if mode is not None else {}))
        dlogp(point)  # warm the compile out of the timing
        t0 = time.perf_counter()
        for _ in range(run_cfg.benchmark_evals):
            dlogp(point)
        ms = (time.perf_counter() - t0) / run_cfg.benchmark_evals * 1000.0

    # nutpie parallelises chains across cores; PyMC's sampler does too, so the
    # wall clock is (iterations x steps x gradient) for the slowest chain.
    steps_per_draw = 32.0
    iterations = run_cfg.draws + run_cfg.tune
    chains_per_core = max(1.0, run_cfg.chains / max(run_cfg.cores, 1))
    projected = ms * steps_per_draw * iterations * chains_per_core / 1000.0 / 60.0

    env = describe_sampler_environment()
    report.add(
        GateResult(
            name="runtime_estimate",
            passed=(projected <= run_cfg.max_runtime_minutes) and not degraded,
            value=(
                f"{ms:.2f} ms/grad, {n_free} free params, ~{projected:.1f} min "
                f"[{sampler}/{linker}]"
            ),
            threshold=f"<= {run_cfg.max_runtime_minutes:.0f} min on a compiled sampler",
            detail=(
                "Resolved to PyMC's NUTS on the pure-Python VM. This is the "
                "configuration that sampled at 400 s/draw on 2026-08-18; install "
                "nutpie>=0.16.10 or pass nuts_sampler explicitly."
                if degraded
                else f"nutpie {env['nutpie_version']}, {run_cfg.chains} chains "
                f"on {run_cfg.cores} cores, assuming ~{steps_per_draw:.0f} "
                "leapfrog steps/draw."
            ),
        )
    )
    return {
        "ms_per_grad": ms,
        "n_free_params": n_free,
        "projected_minutes": projected,
        "sampler": sampler,
        "linker": linker,
        "nutpie_version": env["nutpie_version"],
    }


def describe_sampler_environment() -> dict[str, Any]:
    """Report which sampler and PyTensor linker a fit would actually use.

    This exists because the answer is not obvious and getting it wrong is
    catastrophic rather than merely slow. ``build_sample_kwargs`` sets
    ``compile_kwargs={"mode": Mode(linker="py")}``; PyMC only auto-selects nutpie
    when the linker is Numba or JAX (``pymc/sampling/mcmc.py``), so a bare
    ``nuts_sampler=None`` silently lands on PyMC's own NUTS running the gradient
    on the **pure-Python VM**. That is what produced 400 seconds per draw on the
    2026-08-18 v2 run, and what v1's ``sample_with_fallback`` docstring warns
    about in almost these words.

    Returns
    -------
    dict[str, Any]
        ``sampler``, ``linker``, ``nutpie_version``, ``nutpie_ok`` and
        ``degraded`` (True when the combination is the pure-Python path).
    """
    info: dict[str, Any] = {"nutpie_version": None, "nutpie_ok": False}
    try:
        import nutpie  # noqa: F401

        info["nutpie_version"] = getattr(nutpie, "__version__", "unknown")
        parts = str(info["nutpie_version"]).split(".")[:3]
        info["nutpie_ok"] = tuple(int(p) for p in parts if p.isdigit()) >= (0, 16, 10)
    except Exception as exc:  # pragma: no cover - environment dependent
        info["nutpie_error"] = str(exc)
    return info


def sample_posterior(model: Any, run_cfg: KalmanRunConfigV2) -> Any:
    """Fit with NUTS, naming the sampler explicitly.

    Uses ``build_sample_kwargs`` for the shared policy (chains<2 warning,
    log-likelihood policy, compile kwargs) but **overrides two of its defaults**:

    * ``nuts_sampler`` is passed by name rather than left to PyMC's
      auto-selection. See :func:`describe_sampler_environment` for why the
      default resolves to the pure-Python VM.
    * ``compile_kwargs`` is dropped when nutpie is in use. PyMC pops ``mode``
      itself for a named sampler, but leaving a ``Mode(linker="py")`` in the
      dictionary is exactly the thing that disqualified nutpie in the first
      place, so it is removed here where the intent is visible.
    * ``idata_kwargs`` is dropped outright — it raises a deprecation
      ``FutureWarning`` and is ignored by nutpie anyway. Use
      ``_workflow.attach_log_likelihood`` post hoc if a log-likelihood is wanted.
    """
    import pymc as pm

    sampler, kwargs = _resolve_sampler(run_cfg)
    logger.info(
        "Sampling with nuts_sampler=%r, %d chains x %d draws / %d tune",
        sampler,
        run_cfg.chains,
        run_cfg.draws,
        run_cfg.tune,
    )
    with model:
        idata = pm.sample(**kwargs)
    log_sample_diagnostics(idata, model_name="KalmanPriceTargetV2")
    return idata


# =========================================================================== #
# §8  Posterior predictive                                                    #
# =========================================================================== #


def _thin(idata: Any, max_draws: int) -> Any:
    """Thin the posterior to at most ``max_draws`` total draws.

    ``pm.sample_posterior_predictive`` replicates once per posterior sample and
    takes no draw count, so an un-thinned call replays the whole
    ``chains x draws`` grid to compute statistics that are averages over tens of
    thousands of observations.
    """
    post = idata.posterior
    total = int(post.sizes["chain"] * post.sizes["draw"])
    if total <= max_draws:
        return idata
    step = max(1, total // max_draws)
    return idata.isel(draw=slice(None, None, step))


def _decay_interval(
    obs: np.ndarray,
    rep: np.ndarray,
    mask: np.ndarray,
    time_days: np.ndarray,
) -> Optional[dict[str, float]]:
    """Observed ``rho_inf`` and the replicated 94 % interval around it.

    Shared by both decay statistics so the raw and residual readings cannot drift
    apart in how they are measured — only in what they are measured on.

    Returns ``None`` when the observed kernel or every replicate kernel fails to
    fit, which is a diagnosable state rather than an error.
    """
    try:
        obs_kern = fit_trail_correlation_kernel(obs, time_days)
    except Exception as exc:  # pragma: no cover - degenerate panel
        logger.warning("correlation-decay check unavailable: %s", exc)
        return None

    rho_rep: list[float] = []
    for r in rep[:: max(1, len(rep) // 100)]:
        try:
            rho_rep.append(
                fit_trail_correlation_kernel(np.where(mask, r, np.nan), time_days)["rho_inf"]
            )
        except Exception:  # pragma: no cover - individual replicate failure
            continue
    if not rho_rep:
        return None
    lo, hi = np.percentile(np.asarray(rho_rep), [3, 97])
    return {
        "observed_rho_inf": float(obs_kern["rho_inf"]),
        "observed_ell_days": float(obs_kern["ell_days"]),
        "replicated_lo": float(lo),
        "replicated_hi": float(hi),
        "n_replicates": float(len(rho_rep)),
    }


def run_posterior_predictive(
    model: Any,
    idata: Any,
    panel: KalmanPanelV2,
    run_cfg: KalmanRunConfigV2,
    model_cfg: KalmanModelConfig,
    report: GateReport,
) -> dict[str, Any]:
    """Replicate the data and run the calibration battery.

    Four statistics, of which two are new in v2:

    ``T = std``
        The observed response spread must lie inside the replicated
        distribution. Has failed in every review since 2026-08-05.
    ``per-time coverage``
        Share of observations inside the nominal predictive interval, per
        lookback. Over-shooting is a failure, not a safety margin.
    ``correlation decay`` *(new)*
        The replicates must reproduce the observed ``r(gap)`` curve, not merely
        the marginal spread. A factorised likelihood can match the variance of a
        panel exactly while producing replicates with no time structure at all —
        which is what v1 did, and no v1 statistic could see it.
    ``correlation decay of the residual`` *(new, reported not gated)*
        The same statistic with the posterior-mean mean removed from both sides,
        which is what separates a mean failure from a covariance one.
    ``PIT`` *(carried over)*
        Reported, not gated: it rejects at any ``n`` this large for effects too
        small to matter, so it earns a WARN rather than a block.
    """
    thinned = _thin(idata, run_cfg.ppc_draws)
    # The likelihood is one MvStudentT per covariance group, so the observed
    # variables are ``target_pct_obs_g{k}`` over ragged (rows, cols) blocks.
    # Replicate them all, then stitch back onto the full (isin, time) grid so
    # every statistic below is computed on the same shape as the observed panel.
    obs_names = [
        str(v) for v in model.observed_RVs if str(v.name).startswith("target_pct_obs")
    ]
    if not obs_names:
        raise RuntimeError("no target_pct_obs* observed variable in the model")
    ppc = posterior_predictive_check(
        model, thinned, var_names=obs_names, random_seed=run_cfg.random_seed
    )

    mask = panel.observed_mask
    obs = panel.Y
    # Resolved through the same helper the builder used, so the group order and
    # the ``target_pct_obs_g{k}`` names cannot disagree — a partition that
    # differed by one group would stitch the wrong rows back onto the panel and
    # every statistic below would be computed on scrambled data.
    groups, _ = covariance_groups_for(panel, model_cfg)
    first = np.asarray(ppc.posterior_predictive[obs_names[0]])
    n_draw = int(np.prod(first.shape[:-2]))
    rep = np.full((n_draw, panel.n_isin, panel.n_time), np.nan)
    for gi, (rows, cols, _bucket) in enumerate(groups):
        name = f"target_pct_obs_g{gi}" if len(groups) > 1 else "target_pct_obs"
        if name not in ppc.posterior_predictive:
            logger.warning("replicate %s missing; that group is skipped", name)
            continue
        block = np.asarray(ppc.posterior_predictive[name])
        block = block.reshape(n_draw, len(rows), len(cols))
        rep[np.ix_(np.arange(n_draw), rows, cols)] = block

    out: dict[str, Any] = {}

    # ---- T = spread, measured robustly -------------------------------------
    # Deliberately the INTERQUARTILE RANGE, not the standard deviation.
    #
    # v1 compared replicate sd to observed sd and it failed in every review. On a
    # Student-t likelihood that comparison is close to unfalsifiable: the t's
    # variance is nu/(nu-2) times its scale, so it is dominated by tail mass the
    # density fit does not chase, and a t fitted to data with a few extreme cells
    # has a theoretical variance far above the empirical one *by construction*.
    # Measured here: replicate sd 1.77-1.87 against an observed 1.00, which
    # implies nu ~ 2.9 — the statistic is largely reporting nu, not misfit.
    #
    # The IQR asks the question that was meant: does the model reproduce the
    # spread of the BULK? It is the standard robust answer and it is falsifiable
    # for a heavy-tailed model. The sd ratio is still computed and reported in
    # the detail line, so the tail behaviour stays visible rather than hidden.
    def _iqr(a: np.ndarray) -> float:
        q1, q3 = np.nanpercentile(a, [25, 75])
        return float(q3 - q1)

    obs_iqr = _iqr(obs[mask])
    rep_iqr = np.array([_iqr(r[mask]) for r in rep])
    lo, hi = np.percentile(rep_iqr, [3, 97])
    obs_std = float(np.nanstd(obs[mask]))
    rep_std_med = float(np.median([np.nanstd(r[mask]) for r in rep]))
    out["t_spread"] = {
        "observed_iqr": obs_iqr,
        "lo": float(lo),
        "hi": float(hi),
        "observed_sd": obs_std,
        "replicated_sd_median": rep_std_med,
    }
    report.add(
        GateResult(
            name="ppc_t_spread",
            passed=bool(lo <= obs_iqr <= hi),
            value=f"IQR obs {obs_iqr:.3f} vs rep [{lo:.3f}, {hi:.3f}]",
            threshold="observed IQR inside the replicated 94% interval",
            detail=(
                f"sd for reference: obs {obs_std:.3f}, replicated median "
                f"{rep_std_med:.3f} (ratio {rep_std_med / max(obs_std, _EPS):.2f}x "
                "— expected to exceed 1 under a Student-t and not gated)"
            ),
        )
    )

    # ---- per-time coverage -------------------------------------------------
    # `rep` is NaN wherever a cell has no replicate at all -- an unobserved
    # (isin, time) is never drawn -- and reducing over an all-NaN slice makes
    # numpy warn once per cell rather than telling you how many there are.
    # Exclude those cells from the reduction and report the count instead.
    has_rep = np.isfinite(rep).any(axis=0)  # (isin, time)
    n_no_rep = int((mask & ~has_rep).sum())
    ql, qh = np.nanpercentile(
        np.where(has_rep[None, :, :], rep, 0.0),
        list(run_cfg.gate_coverage_percentiles),
        axis=0,
    )  # (isin, time)
    scored = mask & has_rep
    inside = (obs >= ql) & (obs <= qh) & scored
    cov = inside.sum(axis=0) / np.maximum(scored.sum(axis=0), 1)
    out["coverage"] = cov
    out["n_cells_without_replicates"] = n_no_rep
    worst = float(np.max(np.abs(cov - run_cfg.gate_coverage_target)))
    report.add(
        GateResult(
            name="ppc_coverage",
            passed=worst <= run_cfg.gate_coverage_tol,
            value=f"{cov.min():.3f} - {cov.max():.3f} (target {run_cfg.gate_coverage_target:.2f})",
            threshold=f"within +/-{run_cfg.gate_coverage_tol}",
            detail=(
                f"per-step: {np.round(cov, 4).tolist()}; scored cells per step "
                f"{scored.sum(axis=0).tolist()}"
                + (
                    f"; {n_no_rep} observed cell(s) had no replicate and were excluded"
                    if n_no_rep
                    else ""
                )
            ),
        )
    )

    # ---- correlation decay (new) -------------------------------------------
    # A mean that is constant in t contributes the same constant at every gap, so
    # it reads as a permanent level and the raw statistic mixes the mean's share
    # with the covariance's. Report the split alongside the verdict: f is the
    # mean's variance share, and f alone is the floor the replicated rho_inf
    # cannot go below. On run fa532b925732 that floor WAS the failure -- f was
    # 0.33 against an observed permanent share near 0.48.
    mu_dec = _fitted_mean(idata, panel)
    var_mu_dec = float(np.nanvar(mu_dec))
    var_resid_dec = float(np.nanvar(np.where(mask, obs - mu_dec[:, None], np.nan)))
    f_share = var_mu_dec / max(var_mu_dec + var_resid_dec, _EPS)

    decay = _decay_interval(obs, rep, mask, panel.time_days)
    if decay is not None:
        out["decay"] = decay
        out["decay_mean_share"] = f_share
        report.add(
            GateResult(
                name="ppc_decay",
                passed=bool(
                    decay["replicated_lo"]
                    <= decay["observed_rho_inf"]
                    <= decay["replicated_hi"]
                ),
                value=(
                    f"rho_inf obs {decay['observed_rho_inf']:.3f} vs rep "
                    f"[{decay['replicated_lo']:.3f}, {decay['replicated_hi']:.3f}]"
                ),
                threshold="observed inside the replicated 94% interval",
                detail=(
                    f"Mean carries Var(mu)/(Var(mu)+Var(resid)) = {f_share:.3f} of "
                    "the response; the covariance carries the rest. The mean's "
                    "share is a FLOOR on the replicated rho_inf, so a replicated "
                    "value below the observed one with a healthy residual decay "
                    "means the mean is too small -- check mean_calibration before "
                    "touching rho_inf or the OU length scale. Read alongside "
                    "ppc_decay_residual, which removes the mean from both sides."
                ),
            )
        )

    # ---- correlation decay of the RESIDUAL (reported, not gated) -------------
    # The raw statistic above cannot distinguish "the covariance has the wrong
    # time structure" from "the mean has the wrong spread", because a mean that
    # is constant in t contributes an identical constant at every gap and so
    # reads as a permanent level. On 2026-08-18 the raw reading was 0.429
    # observed against 0.678-0.740 replicated and the entire gap was
    # Var(mu_reg) = 2.50 against a response variance of 0.87 -- a mean failure
    # wearing a covariance failure's clothes, which took hours to unpick by hand.
    # Subtracting the same posterior-mean mean from both sides separates them.
    try:
        # The mean the likelihood actually centres on -- ``mu_scaled`` after the
        # signal-scaling exponent. Centring on the unscaled ``mu_reg`` would
        # leave the scaling in the residual.
        centre = _fitted_mean(idata, panel)[:, None]
        try:
            centre = centre + _posterior_draws(idata, "alpha_time").mean(axis=1)[None, :]
        except KeyError:  # T == 1 registers no alpha_time
            pass
        resid_decay = _decay_interval(obs - centre, rep - centre, mask, panel.time_days)
        if resid_decay is not None:
            out["decay_residual"] = resid_decay
            report.add(
                GateResult(
                    name="ppc_decay_residual",
                    passed=bool(
                        resid_decay["replicated_lo"]
                        <= resid_decay["observed_rho_inf"]
                        <= resid_decay["replicated_hi"]
                    ),
                    value=(
                        f"rho_inf obs {resid_decay['observed_rho_inf']:.3f} vs rep "
                        f"[{resid_decay['replicated_lo']:.3f}, "
                        f"{resid_decay['replicated_hi']:.3f}]"
                    ),
                    threshold="observed inside the replicated 94% interval",
                    blocking=False,
                    detail=(
                        "Same statistic as ppc_decay with the posterior-mean mean "
                        "removed from both sides. Failing here too points at the "
                        "covariance; passing here while ppc_decay fails points at "
                        "the mean, so read mean_spread next."
                    ),
                )
            )
    except Exception as exc:  # pragma: no cover - diagnostic only, never fatal
        logger.warning("residual correlation-decay diagnostic unavailable: %s", exc)

    return out


# =========================================================================== #
# §9  Diagnostics                                                             #
# =========================================================================== #


def free_global_summary(idata: Any, *, ci_prob: float = 0.89) -> pd.DataFrame:
    """Return the convergence summary over the FREE global parameters.

    The single definition of "which parameters the convergence numbers describe",
    shared by :func:`run_diagnostics` and :func:`run_model_comparison`. A second
    implementation would let the production fit and the comparison arms report
    min-ESS on different parameter sets, and two numbers under one name is how a
    reader concludes that an arm mixes worse than the baseline when it was only
    measured differently.

    Two filters, both load-bearing:

    ``globals only``
        Per-ISIN vectors have thousands of entries whose extreme order
        statistics are dominated by the tail of a large sample, so their max
        R-hat is not a convergence signal. The size cap additionally excludes
        wide non-ISIN tensors.
    ``sd > 0``
        ``alpha_time[t3]`` and ``sigma_time[t3]`` are the pinned anchors of
        ``pt.concatenate([free, zeros(1)])``. R-hat and ESS are between-chain
        statistics dividing by a within-chain variance of exactly zero, so they
        come back NaN -- and an unfiltered ``ess_bulk.min()`` would report a
        pinned constant as the thinnest parameter in the model.

    Parameters
    ----------
    idata
        A fitted arm or production fit.
    ci_prob
        Interval width for the summary table.

    Returns
    -------
    pandas.DataFrame
        ``az.summary`` restricted to the free globals. Empty if none qualify.
    """
    import arviz as az

    post = idata.posterior
    globals_ = [
        v
        for v in post.data_vars
        if "isin" not in post[v].dims
        and post[v].size <= post.sizes["chain"] * post.sizes["draw"] * 32
    ]
    if not globals_:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered", category=RuntimeWarning
        )
        summary = az.summary(idata, var_names=globals_, ci_prob=ci_prob)
    return summary.loc[summary["sd"] > _EPS]


def run_diagnostics(
    idata: Any,
    panel: KalmanPanelV2,
    run_cfg: KalmanRunConfigV2,
    report: GateReport,
) -> pd.DataFrame:
    """Convergence gates over the global (non-per-ISIN) parameters, plus mean spread.

    Gating on *global* parameters is deliberate: per-ISIN vectors have thousands
    of entries whose extreme order statistics are dominated by the tail of a
    large sample, so their max R-hat is not a convergence signal. v1's own §9
    table takes the same subset, which is what makes these numbers comparable
    across versions.

    ``mean_spread`` sits here rather than in §8 because it is not a predictive
    check: it is a one-line arithmetic consistency test on the posterior itself,
    and it belongs *before* the expensive replication. The 2026-08-18 run cleared
    divergences, R-hat and ESS while fitting a mean with 2.9x the variance of its
    own response — the three predictive gates that followed were all downstream
    of that one number.
    """
    import arviz as az

    post = idata.posterior
    globals_ = [
        v
        for v in post.data_vars
        if "isin" not in post[v].dims and post[v].size <= post.sizes["chain"] * post.sizes["draw"] * 32
    ]
    # The global / free-parameter selection lives in `free_global_summary`, which
    # `run_model_comparison` also calls -- see there for why the pinned anchors
    # must be excluded before any `.min()`. `summary` is still built here because
    # the exported table keeps the pinned rows (they are real parameters) while
    # the GATES read only the free ones.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered", category=RuntimeWarning
        )
        summary = az.summary(idata, var_names=globals_, ci_prob=0.89)
    free = free_global_summary(idata)
    n_pinned = len(summary) - len(free)
    if n_pinned:
        logger.debug(
            "%d pinned parameter(s) excluded from the convergence gates: %s",
            n_pinned,
            summary.index[summary["sd"] <= _EPS].tolist(),
        )

    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
    report.add(
        GateResult(
            name="divergences",
            passed=div == 0,
            value=str(div),
            threshold="0",
            detail="A non-zero count invalidates the geometry regardless of R-hat.",
        )
    )

    max_rhat = float(free["r_hat"].max())
    worst_rhat = str(free["r_hat"].idxmax())
    report.add(
        GateResult(
            name="r_hat",
            passed=max_rhat < run_cfg.gate_r_hat_max,
            value=f"{max_rhat:.4f} ({worst_rhat})",
            threshold=f"< {run_cfg.gate_r_hat_max}",
        )
    )

    min_ess = float(free["ess_bulk"].min())
    worst_ess = str(free["ess_bulk"].idxmin())
    report.add(
        GateResult(
            name="ess_bulk",
            passed=min_ess >= run_cfg.gate_ess_min,
            value=f"{min_ess:.0f} ({worst_ess})",
            threshold=f">= {run_cfg.gate_ess_min}",
        )
    )

    # ---- mean spread --------------------------------------------------------
    # The regression mean is what every replicate is centred on, so its spread
    # sets a floor under the replicated variance no likelihood can undo.
    mu_mean = _fitted_mean(idata, panel)
    var_mu = float(np.nanvar(mu_mean))
    y_now = panel.Y[:, panel.n_time - 1]
    var_y = float(np.nanvar(y_now))
    ratio = var_mu / max(var_y, _EPS)
    beta_max = float(np.abs(_posterior_draws(idata, "beta").mean(axis=1)).max())
    report.add(
        GateResult(
            name="mean_spread",
            passed=math.isfinite(ratio) and ratio <= run_cfg.gate_mean_spread_max,
            value=f"Var(mu)/Var(y_now) = {ratio:.2f}",
            threshold=f"<= {run_cfg.gate_mean_spread_max:.2f}",
            detail=(
                f"Var(mu) {var_mu:.3f}, Var(y_now) {var_y:.3f}, "
                f"max |beta| {beta_max:.3f}. Above 1 the mean is not explaining "
                "the response, it is adding variance to it -- look for a design "
                "column the mean cannot legally use (drift_contrast_leakage) "
                "before touching the likelihood. This gate is ONE-SIDED and "
                "cannot see an under-dispersed mean; mean_calibration can."
            ),
        )
    )

    # ---- mean calibration ---------------------------------------------------
    # The gate mean_spread could not have caught the 2026-08-19 failure: it
    # reported a healthy-looking 0.33 while the mean was shrunk by 19 %. Regress
    # the response on the fitted mean instead. A calibrated mean gives slope 1;
    # above 1 the mean is too small, and the shortfall does not vanish -- it sits
    # in Cov(mu, resid), which is PERMANENT variance (mu is constant in t) that
    # the generative model sets to exactly zero. That is what depresses the
    # replicated rho_inf and fails ppc_decay, so this gate runs first and names
    # the cause.
    ok = np.isfinite(y_now) & np.isfinite(mu_mean)
    if ok.sum() > 2 and np.nanstd(mu_mean[ok]) > _EPS:
        slope = float(np.polyfit(mu_mean[ok], y_now[ok], 1)[0])
        resid = y_now[ok] - mu_mean[ok]
        cov_share = 2.0 * float(np.cov(mu_mean[ok], resid)[0, 1]) / max(var_y, _EPS)
        lo, hi = run_cfg.gate_mean_calibration
        report.add(
            GateResult(
                name="mean_calibration",
                passed=bool(math.isfinite(slope) and lo <= slope <= hi),
                value=f"slope {slope:.3f}, 2Cov(mu,resid)/Var(y) = {cov_share:+.3f}",
                threshold=f"slope in [{lo:.2f}, {hi:.2f}]",
                # NON-BLOCKING on purpose, and not because the check is weak.
                # Every way this gate fails also fails ppc_decay, which IS
                # blocking -- the mis-calibration and the decay shortfall are the
                # same number seen twice. So the run is still refused; this gate's
                # job is to say WHY in one line instead of requiring a posterior
                # reconstruction. Promote it to blocking once a production run has
                # established what slope is actually achievable (the 2026-08-19
                # smoke fit landed at 1.078 with lambda free).
                blocking=False,
                detail=(
                    "Slope > 1 means the mean is over-shrunk and the missing "
                    "permanent variance is parked in Cov(mu, resid), which no "
                    "replicate reproduces -- read ppc_decay next. Slope < 1 means "
                    "the mean is over-fitted; read mean_spread and ppc_t_spread. "
                    "Measured 1.230 / +0.151 on run fa532b925732, which is the "
                    "whole of that run's ppc_decay failure."
                ),
            )
        )
    return summary


# =========================================================================== #
# §9b  Model comparison                                                       #
# =========================================================================== #

#: Named model arms the ELPD comparison can run, as edits to a base
#: :class:`KalmanModelConfig`.
#:
#: Each entry answers exactly one question, and the *point* of a registry rather
#: than ad-hoc ``replace`` calls at the call site is that the question is named
#: and its arm is reproducible six months later.
#:
#: ``baseline``
#:     The shipped configuration, unchanged. Always include it — an ELPD table
#:     without a reference arm says which alternative is better, not whether
#:     either beats what is running.
#: ``level_off``
#:     Pins the permanent per-ISIN level off. **The rec-01 question.** The level
#:     block is what v2 exists to identify, and run ``37e6d8966250`` gave it
#:     0.59 % of the variance with an 89 % interval of [0.20 %, 1.20 %] — near
#:     zero, and confidently so rather than uncertainly. Two independent
#:     measurements say that is correct rather than under-powered: standardising
#:     the residual by the model's own ``sigma_i * tau_t`` takes the pooled
#:     permanent correlation to exactly zero (see
#:     :attr:`KalmanModelConfig.rho_scale_buckets`), and removing the fitted
#:     regression mean from the raw trail takes the empirical asymptote from
#:     0.420 to 0.071. What neither measurement settles is whether the block
#:     should stay at 0.59 % or be pinned off, because a residual statistic is
#:     not a predictive one. This arm settles it.
#:
#:     Clean at the shipped ``rho_scale_buckets = 1``: ``enable_isin_level =
#:     False`` sets ``rho_inf`` to a constant zero and nothing else, since
#:     ``rho_scale_slope`` is not created at all when buckets is 1. **Raise
#:     ``rho_scale_buckets`` above 1 and this becomes a two-change arm** — the
#:     tilt also requires ``enable_isin_level`` — at which point the contrast is
#:     no longer attributable.
#: ``hierarchy_fine``
#:     Adds ``country`` and ``industry`` to the crossed group effects. **The
#:     rec-02 question.** See :data:`GROUP_EFFECTS_FINE` for the OLS evidence and
#:     the ESS watch.
#: ``drift_strict``
#:     The drift matrix as :func:`select_drift_features_v2` now selects it,
#:     against the pre-2026-08-21 list that also carried
#:     ``feat_eps_signal_surprise``. **The rec-03 question**, and the one arm
#:     that changes the design matrix rather than the model graph — which is why
#:     :func:`run_model_comparison` prepares a panel per arm instead of sharing
#:     one.
COMPARISON_ARMS: dict[str, Callable[[KalmanModelConfig], KalmanModelConfig]] = {
    "baseline": lambda cfg: cfg,
    "level_off": lambda cfg: replace(cfg, enable_isin_level=False),
    "hierarchy_fine": lambda cfg: replace(cfg, group_effects=GROUP_EFFECTS_FINE),
    "drift_strict": lambda cfg: cfg,
}

#: Columns the ``drift_strict`` arm re-admits to form its comparison baseline.
#: The arm itself is the *current* selection; the contrast is against the design
#: matrix as it stood before the admission rule landed.
_DRIFT_LOOSE_READMIT: tuple[str, ...] = ("feat_eps_signal_surprise",)

#: Fields of :class:`KalmanPanelV2` that are NOT indexed by ISIN and must survive
#: subsampling unchanged. Everything else with a leading axis of length
#: ``n_isin`` is sliced.
_PANEL_NON_ISIN_FIELDS: frozenset[str] = frozenset(
    {
        "time_days",
        "drift_names",
        "coord_uniques",
        "response_mean",
        "response_std",
        "orthogonal_rotation",
        "orthogonal_source_names",
        "contrast_screen",
    }
)


def subsample_panel_v2(
    panel: KalmanPanelV2,
    max_isins: int,
    *,
    random_seed: int = 42,
    keep_isins: Optional[Sequence[str]] = None,
) -> KalmanPanelV2:
    """Return ``panel`` restricted to at most ``max_isins`` names.

    Slices every ISIN-indexed field by introspecting the dataclass rather than
    listing them, so a field added to :class:`KalmanPanelV2` later is carried
    automatically instead of silently keeping its full-panel length and failing
    deep inside the likelihood. Fields in :data:`_PANEL_NON_ISIN_FIELDS` pass
    through untouched.

    ``coord_idx`` is sliced but ``coord_uniques`` is **not re-factorised**. A
    group level left with no members is a harmless unused prior draw; re-indexing
    would give two arms different ``coords``, and an ELPD contrast between models
    with different coordinate sets is not a contrast between the two models.

    Parameters
    ----------
    panel
        The full panel.
    max_isins
        Cap. Values at or above ``panel.n_isin`` return the panel unchanged.
    random_seed
        Seed for the subsample, so repeated arms score identical rows.
    keep_isins
        Restrict to these ISINs before sampling. Used to intersect arms whose
        panels dropped different rows.

    Returns
    -------
    KalmanPanelV2
        A new panel; the original is not mutated.
    """
    n = panel.n_isin
    idx = np.arange(n)
    if keep_isins is not None:
        wanted = set(map(str, keep_isins))
        idx = idx[np.array([str(i) in wanted for i in panel.isins], dtype=bool)]
    if 0 < max_isins < idx.size:
        idx = np.sort(
            np.random.default_rng(random_seed).choice(idx, size=max_isins, replace=False)
        )
    if idx.size == n and keep_isins is None:
        return panel

    updates: dict[str, Any] = {}
    for f in dc_fields(panel):
        if f.name in _PANEL_NON_ISIN_FIELDS:
            continue
        value = getattr(panel, f.name)
        if isinstance(value, pd.DataFrame):
            if len(value) == n:
                updates[f.name] = value.iloc[idx].reset_index(drop=True)
        elif isinstance(value, dict):
            updates[f.name] = {
                k: (v[idx] if getattr(v, "shape", (0,))[0] == n else v)
                for k, v in value.items()
            }
        elif isinstance(value, np.ndarray) and value.shape and value.shape[0] == n:
            updates[f.name] = value[idx]

    return replace(panel, **updates)


def collapse_group_loglik(idata: Any, panel: KalmanPanelV2,
                          model_cfg: KalmanModelConfig) -> Any:
    """Stitch the per-covariance-group ``log_likelihood`` into ONE variable.

    Without this the ELPD comparison cannot run on any real panel, and that is
    not a hypothetical: the v2 likelihood is one ``MvStudentT`` per covariance
    group, so ``log_likelihood`` carries ``target_pct_obs_g0..gN`` and
    ``az.compare`` raises ``TypeError: Encountered error trying to compute ELPD
    from model <arm>`` because it cannot choose among them. Measured 2026-08-25:
    a ``baseline`` vs ``level_off`` contrast fitted both arms cleanly at zero
    divergences in 9.7 minutes and then produced nothing, on a panel that
    partitioned into 3 groups of [776, 20, 4]. The single-group case
    (``target_pct_obs``, no suffix) works, which is why the failure survived a
    self-test on a synthetic panel that never partitions.

    **The pointwise unit is the NAME, not the cell**, and that is a modelling
    statement rather than a convenience. Each group's ``MvStudentT`` is a
    multivariate density over one name's T observations, so it already emits one
    log-density per row; a name's cells are correlated by construction -- that
    correlation *is* the model -- so leaving out a single cell would not be a
    leave-one-out at all. LOO here is leave-one-name-out, which is exactly the
    question the level-vs-state contrast asks.

    Parameters
    ----------
    idata
        A fitted arm carrying a ``log_likelihood`` group.
    panel
        The panel the arm was fitted on; supplies ``n_isin`` and the row order.
    model_cfg
        Resolves the partition through :func:`covariance_groups_for` -- the SAME
        helper the builder and §8 use, so the group order and the
        ``target_pct_obs_g{k}`` names cannot disagree.

    Returns
    -------
    Any
        ``idata`` with its ``log_likelihood`` group replaced by a single
        ``target_pct_obs`` variable of dims ``(chain, draw, isin)``.

    Raises
    ------
    KeyError
        If a group's variable is absent -- scoring a subset silently would drop
        names from one arm and not the other, which is not a contrast.

    Notes
    -----
    **Mutates ``idata`` in place** and returns it, matching the convention of
    :func:`attach_log_likelihood`, which it always follows. It is IDEMPOTENT:
    calling it on an already-stitched arm returns that arm untouched rather than
    raising about the group variables it consumed on the first pass. That is not
    defensive decoration -- an in-place rewrite that is not idempotent turns any
    second call into a confusing ``KeyError`` about variables the caller never
    removed.
    """
    import xarray as xr

    ll = idata.log_likelihood
    # Already stitched: one variable, over names. Return unchanged.
    _vars = [str(v) for v in ll.data_vars]
    if _vars == ["target_pct_obs"] and "isin" in ll["target_pct_obs"].dims:
        return idata

    groups, _ = covariance_groups_for(panel, model_cfg)
    names = [
        f"target_pct_obs_g{gi}" if len(groups) > 1 else "target_pct_obs"
        for gi in range(len(groups))
    ]
    missing = [n for n in names if n not in ll.data_vars]
    if missing:
        raise KeyError(
            f"log_likelihood lacks {missing}; cannot assemble a pointwise "
            "log-likelihood over names"
        )
    if len(names) == 1:
        return idata  # single group; the variable is already unsuffixed

    first = ll[names[0]]
    n_chain = int(first.sizes["chain"])
    n_draw = int(first.sizes["draw"])
    out = np.full((n_chain, n_draw, panel.n_isin), np.nan, dtype="float64")
    for (rows, _cols, _bkt), nm in zip(groups, names):
        arr = np.asarray(ll[nm], dtype="float64")
        # (chain, draw, rows_k) -- one log-density per NAME in this group.
        if arr.ndim != 3 or arr.shape[2] != len(rows):
            raise KeyError(
                f"{nm} has shape {arr.shape}, expected (chain, draw, {len(rows)})"
            )
        out[:, :, np.asarray(rows, dtype=int)] = arr

    if np.isnan(out).any():
        n_bad = int(np.isnan(out).any(axis=(0, 1)).sum())
        raise KeyError(
            f"{n_bad} name(s) received no log-likelihood from any group; the "
            "partition does not cover the panel"
        )

    combined = xr.Dataset(
        {"target_pct_obs": (("chain", "draw", "isin"), out)},
        coords={
            "chain": np.asarray(first["chain"]),
            "draw": np.asarray(first["draw"]),
            "isin": np.asarray(panel.isins, dtype=object),
        },
    )
    try:
        idata.log_likelihood = combined
    except Exception:  # pragma: no cover - DataTree vs InferenceData surface
        idata["log_likelihood"] = combined
    return idata


def run_model_comparison(
    frame: pd.DataFrame,
    model_cfg: KalmanModelConfig,
    run_cfg: KalmanRunConfigV2,
    report: GateReport,
    *,
    arms: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """Compare named model arms on ELPD and record the result as a gate.

    Closes the one blank cell in v2's Bayesian-workflow row. Every other gate in
    this module scores the model against the analyst trail it was fitted to,
    which is why a pass-through once cleared 19 of 21 of them; this is the only
    stage that asks whether a component *earns* its place.

    Each arm is prepared, fitted and scored independently:

    1. :func:`prepare_panel` **per arm**, because ``drift_strict`` changes the
       design matrix and a shared panel could not express it.
    2. The **intersection** of ISINs across arms, then one subsample of that,
       so every arm is scored on identical rows. Comparing ELPD across different
       observation sets is meaningless, and prepare_panel's own filters can drop
       different names under different configs.
    3. ``attach_log_likelihood`` **post hoc**. :func:`sample_posterior` hard-codes
       ``log_likelihood: False`` and nutpie — the default sampler — discards
       ``idata_kwargs`` wholesale, so the ``idata_kwargs={'log_likelihood':
       True}`` route silently does nothing here. A missing group is reported as a
       failure, never treated as a tie.

    Parameters
    ----------
    frame
        The loaded modelling frame, before ``prepare_panel``.
    model_cfg
        Base config each arm edits.
    run_cfg
        Supplies the NUTS budget and :attr:`KalmanRunConfigV2.comparison_max_isins`.
    report
        Gate report the ``model_comparison`` result is added to.
    arms
        Names from :data:`COMPARISON_ARMS`. Defaults to
        ``run_cfg.comparison_arms``, or every registered arm.

    Returns
    -------
    pandas.DataFrame or None
        The ``az.compare`` table, or ``None`` when the comparison could not be
        completed. A ``None`` is always accompanied by a failed gate.
    """
    import arviz as az

    names = list(arms or run_cfg.comparison_arms or COMPARISON_ARMS)
    unknown = [a for a in names if a not in COMPARISON_ARMS]
    if unknown:
        raise ValueError(
            f"Unknown comparison arm(s) {unknown!r}. Known: {sorted(COMPARISON_ARMS)}"
        )
    if len(names) < 2:
        raise ValueError(f"Model comparison needs >= 2 arms, got {names!r}")

    def _fail(msg: str) -> None:
        logger.error("Model comparison: %s", msg)
        report.add(
            GateResult(
                name="model_comparison",
                passed=False,
                blocking=False,
                value=msg[:80],
                threshold=">= 2 arms with a log_likelihood group",
                detail="No ELPD contrast was produced; nothing is decided.",
            )
        )

    # ---- resolve each arm's drift matrix, then build its panel --------------
    # `drift_strict` is the only arm that changes the DESIGN rather than the
    # graph, and it is expressed as a contrast: it keeps the current selection
    # while every other arm gets the pre-rule list, which also re-admits
    # _DRIFT_LOOSE_READMIT. With that arm absent, all arms share the default
    # selection and `drift_names=None` lets prepare_panel resolve it.
    drift_for: dict[str, Optional[list[str]]] = {arm: None for arm in names}
    if "drift_strict" in names:
        strict = select_drift_features_v2(frame)
        loose = sorted(
            set(strict) | {c for c in _DRIFT_LOOSE_READMIT if c in frame.columns}
        )
        if set(loose) == set(strict):
            logger.warning(
                "drift_strict arm is identical to its baseline: none of %s is in "
                "the frame, so there is nothing to re-admit",
                _DRIFT_LOOSE_READMIT,
            )
        drift_for = {arm: (strict if arm == "drift_strict" else loose) for arm in names}

    panels: dict[str, KalmanPanelV2] = {
        arm: prepare_panel(
            frame, COMPARISON_ARMS[arm](model_cfg), run_cfg, drift_names=drift_for[arm]
        )
        for arm in names
    }

    first = panels[names[0]]
    if first.n_time < 2:
        _fail(f"T={first.n_time}; the level/state contrast needs T > 1")
        return None

    common = set(map(str, first.isins))
    for arm in names[1:]:
        common &= set(map(str, panels[arm].isins))
    if not common:
        _fail("arms share no ISINs after panel preparation")
        return None

    subs = {
        arm: subsample_panel_v2(
            p,
            run_cfg.comparison_max_isins,
            random_seed=run_cfg.random_seed,
            keep_isins=sorted(common),
        )
        for arm, p in panels.items()
    }
    n_scored = subs[names[0]].n_isin
    # Never let a truncated comparison read as a full one.
    logger.info(
        "Model comparison: %d arms on %d of %d ISINs (%.0f%%; cap=%d), T=%d, "
        "%d chains x %d draws",
        len(names), n_scored, first.n_isin, 100 * n_scored / max(first.n_isin, 1),
        run_cfg.comparison_max_isins, first.n_time, run_cfg.chains, run_cfg.draws,
    )

    # ---- fit each arm --------------------------------------------------------
    fits: dict[str, Any] = {}
    #: Per-arm convergence, carried onto the returned frame so a reader can see
    #: whether a ranking came from an arm that actually mixed.
    convergence: dict[str, dict[str, Any]] = {}
    for arm in names:
        cfg = COMPARISON_ARMS[arm](model_cfg)
        sub = subs[arm]
        logger.info(
            "  [%s] D=%d drift column(s), %d group level(s)",
            arm, sub.X_drift.shape[1], sum(len(v) for v in sub.coord_uniques.values()),
        )
        model = build_kalman_pt_model_v2(sub, config=cfg)
        idata = sample_posterior(model, run_cfg)
        if idata is None:
            _fail(f"sampling failed on the {arm!r} arm")
            return None
        attach_log_likelihood(idata, model)
        if not hasattr(idata, "log_likelihood"):
            _fail(f"could not attach a log_likelihood group to the {arm!r} arm")
            return None
        # The likelihood is one MvStudentT per covariance group, so the group
        # this just attached carries `target_pct_obs_g0..gN` and `az.compare`
        # cannot choose among them. Collapse to one variable over NAMES before
        # scoring -- see `collapse_group_loglik` for why the name is the right
        # pointwise unit and why this was invisible until the harness was
        # actually run.
        try:
            idata = collapse_group_loglik(idata, sub, cfg)
        except Exception as exc:
            _fail(f"could not assemble a pointwise log-likelihood for {arm!r}: {exc}")
            return None
        div = (
            int(idata.sample_stats["diverging"].sum())
            if "diverging" in getattr(idata, "sample_stats", {})
            else 0
        )
        # Per-arm convergence, on the SAME parameter selection the production
        # gates use (`free_global_summary`).
        #
        # Why this is not decoration. An arm that ranks first on ELPD while
        # mixing badly has not won -- its ELPD is computed from draws that do
        # not represent the posterior -- and without this the reader cannot
        # tell the two apart from the comparison's own output. It was a live
        # gap: the `hierarchy_fine` arm adds 147 shrunk group levels, 24 of
        # `country`'s 82 carrying fewer than 5 names and the smallest carrying
        # 1, and the standing instruction is "if the arm mixes badly, drop
        # country before industry". That instruction was unactionable, because
        # the run reported divergences and nothing else -- and zero divergences
        # is exactly what a hard-shrunk, badly-identified level produces.
        #
        # The THINNEST PARAMETER's name is the actionable half. `min ESS 14 on
        # country[XK]` says drop country; `min ESS 1465 on log_sigma_total` says
        # the arm is fine and the binding parameter is the one it always is.
        min_ess = max_rhat = float("nan")
        worst_ess_param = worst_rhat_param = ""
        try:
            free_arm = free_global_summary(idata)
            if len(free_arm):
                min_ess = float(free_arm["ess_bulk"].min())
                worst_ess_param = str(free_arm["ess_bulk"].idxmin())
                max_rhat = float(free_arm["r_hat"].max())
                worst_rhat_param = str(free_arm["r_hat"].idxmax())
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("  [%s] convergence summary unavailable: %s", arm, exc)

        logger.info(
            "  [%s] divergences=%d  min ESS %s (%s)  max R-hat %s (%s)",
            arm, div,
            f"{min_ess:,.0f}" if np.isfinite(min_ess) else "n/a", worst_ess_param or "n/a",
            f"{max_rhat:.4f}" if np.isfinite(max_rhat) else "n/a", worst_rhat_param or "n/a",
        )
        if np.isfinite(min_ess) and min_ess < run_cfg.gate_ess_min:
            logger.warning(
                "  [%s] min bulk ESS %.0f is BELOW the %.0f gate (thinnest: %s). "
                "This arm's ELPD is computed from draws that may not represent "
                "its posterior -- treat its ranking as unmeasured, not as a win.",
                arm, min_ess, run_cfg.gate_ess_min, worst_ess_param,
            )
        convergence[arm] = {
            "divergences": div,
            "min_ess_bulk": min_ess,
            "min_ess_param": worst_ess_param,
            "max_r_hat": max_rhat,
            "max_r_hat_param": worst_rhat_param,
            "n_group_levels": sum(len(v) for v in sub.coord_uniques.values()),
        }
        fits[arm] = idata

    try:
        cmp_df = az.compare(fits)
    except Exception as exc:  # pragma: no cover - arviz surface
        _fail(f"az.compare failed: {exc!r}")
        return None

    # ArviZ 1.x exposes the value as `.elpd`. `.elpd_loo` was REMOVED, and a
    # getattr fallback on the old name yields a silent nan -- even though the
    # ELPDData repr still prints the "elpd_loo" row label.
    elpds: dict[str, float] = {}
    for arm, idata in fits.items():
        try:
            loo = az.loo(idata)
            elpds[arm] = float(loo.elpd)
        except Exception as exc:  # pragma: no cover
            logger.warning("  [%s] az.loo failed: %r", arm, exc)

    winner = str(cmp_df.index[0])
    runner = str(cmp_df.index[1]) if len(cmp_df.index) > 1 else ""
    detail = (
        f"Scored on {n_scored} of {first.n_isin} ISINs. "
        + ", ".join(f"{k} elpd {v:,.1f}" for k, v in elpds.items())
    )
    if "baseline" in cmp_df.index and winner != "baseline":
        detail += (
            f". {winner!r} ranks above the shipped configuration -- promote it by "
            "editing the corresponding default, not by leaving this arm enabled."
        )
    # A thin arm is reported in the VERDICT, not just the frame: a ranking from
    # an arm that did not mix is not a ranking, and the gate line is what a
    # reader sees first.
    _thin = [
        f"{a} min ESS {c['min_ess_bulk']:,.0f} ({c['min_ess_param']})"
        for a, c in convergence.items()
        if np.isfinite(c.get("min_ess_bulk", float("nan")))
        and c["min_ess_bulk"] < run_cfg.gate_ess_min
    ]
    if not detail.endswith("."):
        detail += "."
    if _thin:
        detail += (
            f" CONVERGENCE WARNING -- {'; '.join(_thin)}, below the "
            f"{run_cfg.gate_ess_min:.0f} gate. Treat those arms' rankings as "
            "unmeasured rather than as results."
        )
    else:
        _worst = min(
            (c for c in convergence.values() if np.isfinite(c.get("min_ess_bulk", float("nan")))),
            key=lambda c: c["min_ess_bulk"], default=None,
        )
        if _worst is not None:
            detail += (
                f" Thinnest arm: min ESS {_worst['min_ess_bulk']:,.0f} "
                f"({_worst['min_ess_param']}), above the "
                f"{run_cfg.gate_ess_min:.0f} gate."
            )
    report.add(
        GateResult(
            name="model_comparison",
            passed=True,
            blocking=False,
            value=f"{winner} ranks first of {len(names)}"
            + (f" (next: {runner})" if runner else ""),
            threshold="reported, not gated",
            detail=detail,
        )
    )
    out = cmp_df.reset_index().rename(columns={"index": "arm"})
    # Convergence travels WITH the ELPD table, into `09b_comparison_v2`. A
    # contrast archived without it cannot be re-read later for whether the
    # winning arm mixed -- which is precisely the question the numbers alone
    # invite and cannot answer.
    for col in ("divergences", "min_ess_bulk", "min_ess_param", "max_r_hat",
                "max_r_hat_param", "n_group_levels"):
        out[col] = out["arm"].map(lambda a, _c=col: convergence.get(a, {}).get(_c))
    return out


def compare_arms_fast(
    panel: KalmanPanelV2,
    idata: Any,
    model_cfg: KalmanModelConfig,
    run_cfg: KalmanRunConfigV2,
    report: GateReport,
    *,
    arms: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """§9b-fast — screen comparison arms with Max-and-Smooth.

    Runs against **one** baseline fit instead of one fit per arm. The Max step
    turns each name's trail into a Gaussian pseudo-observation of ``mu_reg``
    (conditioning on the baseline's covariance), and each arm is then a
    linear-Gaussian regression over ~6.5k rows with ~20-175 free parameters.
    Seconds per arm against roughly a production run per arm.

    **This screens; it does not decide.** The contrast is on pseudo-observations,
    so it ranks arms rather than measuring their exact ELPD. Take the winner to
    :func:`run_model_comparison` before editing any default. The gate it writes is
    named ``model_comparison_fast`` for exactly that reason.

    Arms are refused when their config delta touches a field the Max step
    conditioned on — see
    :data:`~probabilistic_ml_model.pymc_models._max_and_smooth.COVARIANCE_FIELDS`.
    ``level_off`` is admissible on purpose: the pseudo model carries the
    permanent level as a free scale, so dropping it is a latent-side change.

    Parameters
    ----------
    panel
        The panel the baseline was fitted to.
    idata
        The baseline posterior.
    model_cfg
        The baseline config each arm edits.
    run_cfg
        Supplies ``random_seed`` and the draw budget.
    report
        Gate report the ``model_comparison_fast`` result is added to.
    arms
        Names from :data:`COMPARISON_ARMS`. Defaults to every registered arm.

    Returns
    -------
    pandas.DataFrame or None
        The ``az.compare`` table with a ``backend`` column, or ``None`` when the
        screen could not be completed (always with a failed gate).
    """
    import arviz as az
    import pymc as pm

    from probabilistic_ml_model.pymc_models._max_and_smooth import (
        assert_arm_is_screenable,
        build_pseudo_model,
        gaussian_likelihood_approximation,
    )

    names = list(arms or COMPARISON_ARMS)
    unknown = [a for a in names if a not in COMPARISON_ARMS]
    if unknown:
        raise ValueError(
            f"Unknown comparison arm(s) {unknown!r}. Known: {sorted(COMPARISON_ARMS)}"
        )
    if len(names) < 2:
        raise ValueError(f"A contrast needs >= 2 arms, got {names!r}")

    def _fail(msg: str) -> None:
        logger.error("Fast model comparison: %s", msg)
        report.add(
            GateResult(
                name="model_comparison_fast",
                passed=False,
                blocking=False,
                value=msg[:80],
                threshold=">= 2 screenable arms",
                detail="No screening contrast was produced; nothing is indicated.",
            )
        )

    # `drift_strict` changes the DESIGN matrix, which the pseudo-observations
    # carry per name -- so it would need its own Max step against its own panel.
    # Out of scope for a screen that exists to reuse one.
    if "drift_strict" in names:
        _fail("the drift_strict arm changes the design matrix; use --compare")
        return None

    # The UNION of every arm's group effects, not the baseline's. `prepare_panel`
    # indexes only `model_cfg.group_effects`, so a panel prepared for the shipped
    # four-level hierarchy carries no `country` / `industry` index -- and
    # `hierarchy_fine` would reduce to the baseline and screen as "no
    # difference". The Max step factorises the extras from `panel.frame`.
    extra_groups = sorted(
        {
            col
            for arm in names
            for col in COMPARISON_ARMS[arm](model_cfg).group_effects
        }
        - set(panel.coord_idx)
    )
    try:
        pseudo = gaussian_likelihood_approximation(
            panel, idata, model_cfg, extra_group_cols=extra_groups
        )
    except Exception as exc:
        _fail(f"Max step failed: {exc}")
        return None

    fits: dict[str, Any] = {}
    for arm in names:
        arm_cfg = COMPARISON_ARMS[arm](model_cfg)
        try:
            assert_arm_is_screenable(model_cfg, arm_cfg, arm)
        except ValueError as exc:
            logger.warning("%s", exc)
            continue
        try:
            with build_pseudo_model(pseudo, arm_cfg) as pseudo_model:
                post = pm.sample(
                    draws=500, tune=500, chains=4, cores=1, target_accept=0.9,
                    random_seed=run_cfg.random_seed, progressbar=False,
                )
                pm.compute_log_likelihood(post, model=pseudo_model, progressbar=False)
            fits[arm] = post
            logger.info("fast arm %r fitted (%d pseudo-observations)", arm, len(pseudo))
        except Exception as exc:
            logger.error("fast arm %r failed: %s", arm, exc, exc_info=True)

    if len(fits) < 2:
        _fail(f"only {len(fits)} arm(s) produced a screenable fit")
        return None

    cmp_df = az.compare(fits)
    winner = str(cmp_df.index[0])
    runner = str(cmp_df.index[1]) if len(cmp_df) > 1 else ""
    report.add(
        GateResult(
            name="model_comparison_fast",
            passed=True,
            blocking=False,
            value=f"{winner} ranks first of {len(fits)}"
            + (f" (next: {runner})" if runner else ""),
            threshold="reported, not gated",
            detail=(
                f"Max-and-Smooth screen over {len(pseudo)} pseudo-observations "
                f"from run baseline; w_level {pseudo.diagnostics['w_level']:.4f}, "
                f"t-inflation {pseudo.diagnostics['t_inflation']:.3f}. SCREEN ONLY "
                f"-- confirm {winner!r} with --compare before editing a default."
            ),
        )
    )
    out = cmp_df.reset_index().rename(columns={"index": "arm"})
    out["backend"] = "max_and_smooth"
    return out


# =========================================================================== #
# §10  Screen + decision gates                                                #
# =========================================================================== #


@dataclass(frozen=True)
class ScreenDraws:
    """The draw arrays §10 builds and §10b needs, carried explicitly.

    ``run_risk_book`` used to re-resolve the decision latent from the idata
    itself. That was harmless while the screen was a thin summary of that same
    latent, but it stopped being true once §10 gained forecast-error shrinkage:
    a re-resolve would have sized the book on the UNSHRUNK latent while the
    screen reported the shrunk one, and every gate would still have passed.

    The Monte-Carlo array is carried for the same reason in reverse — it existed
    only inside ``run_screen``, so ``compute_cvar_aware_book`` had no return
    distribution to compute a CVaR from and fell back to the posterior upside
    draws, which is why the exported ``cvar05`` was positive for 88 % of names.

    Attributes
    ----------
    eu
        Expected-upside draws in RETURN space, dims ``(chain, draw, isin)``.
    mc_returns
        Monte-Carlo forward returns, shape ``(n_isin, n_samples, horizon)``.
    isins
        Identifiers, aligned to the ``isin`` axis of both.
    """

    eu: Any
    mc_returns: np.ndarray
    isins: np.ndarray

    @property
    def pooled_returns(self) -> np.ndarray:
        """``mc_returns`` flattened to ``(n_isin, n_samples * horizon)``.

        The pooling ``summarize_mc_returns`` uses, so a CVaR taken from this and
        the exported ``er_p05`` describe the same distribution.
        """
        return np.asarray(self.mc_returns).reshape(len(self.isins), -1)


def _posterior_draws(idata: Any, name: str, *, per_isin: bool = True) -> np.ndarray:
    """Flatten a posterior variable to ``(isin, sample)`` or ``(sample,)``."""
    post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
    if name not in post:
        raise KeyError(
            f"{name!r} not in posterior. Available: {sorted(map(str, post.data_vars))}"
        )
    arr = np.asarray(post[name])
    if per_isin:
        return arr.reshape(-1, arr.shape[-1]).T  # (isin, sample)
    return arr.reshape(-1)


def _posterior_mean(idata: Any, name: str, panel: KalmanPanelV2) -> np.ndarray:
    """Posterior mean of a per-ISIN variable, aligned to ``panel.isins``."""
    try:
        return _posterior_draws(idata, name).mean(axis=1)
    except KeyError:
        logger.warning("%s absent from the posterior; filling with NaN", name)
        return np.full(panel.n_isin, np.nan)


def _fitted_mean(idata: Any, panel: KalmanPanelV2) -> np.ndarray:
    """Posterior mean of the linear predictor the likelihood actually centres on.

    ``mu_scaled`` since 2026-08-19 — ``signal_scale * mu_reg`` — falling back to
    ``mu_reg`` for an idata fitted before the signal-scaling exponent existed.
    Resolved by membership rather than by catching ``KeyError``: `_posterior_mean`
    swallows that and returns an all-NaN vector, which would have surfaced as a
    spurious ``mean_spread`` failure instead of a fallback.
    """
    post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
    name = "mu_scaled" if "mu_scaled" in post else "mu_reg"
    return _posterior_mean(idata, name, panel)


def _risk_adjusted_prob_positive(
    idata: Any,
    panel: KalmanPanelV2,
    mc_log: np.ndarray,
    *,
    chunk: int = 512,
) -> Optional[np.ndarray]:
    """P(risk-adjusted forward return > 0) per name, from log-space MC draws.

    Replaces ``mc_prob_pos * kalman_gain``, which multiplied a probability by a
    sigmoid of a location parameter: usable as an ordering, meaningless as a
    level. This is one probability of one stated event — that the forward return,
    net of the same risk / size / volume penalties the model applies to
    ``risk_adj_return``, is positive.

    Two implementation notes that are not incidental:

    * The penalty is recovered as ``state_now_mean - risk_adj_return`` rather
      than rebuilt from the three loadings and their data columns. That is the
      penalty the model actually applied, so it cannot drift out of step with the
      builder the way a reimplementation would.
    * The test is on the LOG draws. ``expm1`` is monotone and the clip bounds
      straddle zero, so ``expm1(x) > 0`` exactly when ``x > 0`` — converting
      first would allocate a second multi-gigabyte array to learn nothing.

    Parameters
    ----------
    mc_log
        Simulated log-uplift, shape ``(n_isin, n_samples, horizon)``, already
        clipped. Not modified.
    chunk
        Names per block. The comparison materialises
        ``chunk * n_samples * horizon`` floats, so this bounds peak memory at a
        few hundred MB against a ~1.7 GB source array.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n_isin,)``, or ``None`` when the posterior lacks the variables,
        which signals the caller to fall back to the legacy product.
    """
    try:
        state = _posterior_draws(idata, KALMAN_V2_SCREEN_LATENT)  # (isin, sample)
        rar = _posterior_draws(idata, "risk_adj_return")
    except KeyError as exc:
        logger.warning(
            "risk-adjusted MC probability unavailable (%s); p_upside_pos_cond "
            "falls back to the mc_prob_pos * kalman_gain product, whose LEVEL is "
            "not interpretable.", exc,
        )
        return None

    penalty = (state - rar) * panel.response_std  # (isin, sample), log-uplift
    n_isin = mc_log.shape[0]
    if penalty.shape[0] != n_isin or penalty.shape[1] != mc_log.shape[1]:
        logger.warning(
            "penalty draws %s do not align with the MC array %s; "
            "p_upside_pos_cond falls back to the legacy product.",
            penalty.shape, mc_log.shape,
        )
        return None

    out = np.empty(n_isin, dtype="float64")
    for i0 in range(0, n_isin, chunk):
        sl = slice(i0, min(i0 + chunk, n_isin))
        adj = mc_log[sl] - penalty[sl, :, None]
        out[sl] = (adj > 0.0).mean(axis=(1, 2))
    return out


def run_screen(
    idata: Any,
    panel: KalmanPanelV2,
    run_cfg: KalmanRunConfigV2,
    report: GateReport,
) -> tuple[pd.DataFrame, "ScreenDraws"]:
    """Build the per-ISIN screen and gate the decision layer.

    Returns the screen frame **and** a :class:`ScreenDraws` carrying the draw
    arrays §10b needs — see that class for why they are handed over rather than
    re-derived.

    Two gates here that v1 had nowhere to put, both aimed at failure modes that
    a manual review caught only after they had shipped:

    ``shrinkage_slope``
        Regress model expected upside on analyst-implied upside. A well-behaved
        shrinkage estimator has slope slightly below 1 and intercept near 0. An
        *intercept* is a uniform offset applied to the entire universe — which is
        exactly what drove names-above-consensus from 62 % to 81 % between two
        runs while every other statistic stayed plausible.
    ``coverage_gradient``
        Posterior uncertainty must fall as analyst coverage rises. A flat or
        inverted gradient means the hierarchy is not pricing information, and it
        is the column the risk book divides by.
    """
    import xarray as xr

    from probabilistic_ml_model.pymc_models._price_target_mc import (
        simulate_lagged_risk_adjusted_returns,
        summarize_mc_returns,
    )

    if run_cfg.enable_forecast_error_shrinkage:
        latent, shrink_gain = apply_forecast_error_shrinkage(
            idata,
            panel,
            multiplier=run_cfg.forecast_error_multiplier,
            n_exponent=run_cfg.forecast_error_n_exponent,
            latent=KALMAN_V2_SCREEN_LATENT,
            random_seed=run_cfg.random_seed,
        )
    else:
        latent = resolve_screen_latent_v2(
            idata, latent=KALMAN_V2_SCREEN_LATENT, random_seed=run_cfg.random_seed
        )
        shrink_gain = np.ones(panel.n_isin)
    draws = np.asarray(latent).reshape(-1, latent.shape[-1])  # (sample, isin)

    # De-standardise back to log-uplift, then to a return. The clip is applied in
    # LOG space so it is sign-preserving; converting first and clipping after
    # would distort prob_pos.
    log_uplift = np.clip(
        draws * panel.response_std + panel.response_mean,
        LOG_UPLIFT_CLIP_LO,
        LOG_UPLIFT_CLIP_HI,
    )
    upside = np.expm1(log_uplift)
    # The same quantity with its (chain, draw) axes intact, for the risk book.
    # Built from ``latent`` rather than reshaped back out of ``upside`` so the
    # dims and coords come from the posterior rather than being reconstructed.
    eu_draws = xr.DataArray(
        np.expm1(
            np.clip(
                np.asarray(latent) * panel.response_std + panel.response_mean,
                LOG_UPLIFT_CLIP_LO,
                LOG_UPLIFT_CLIP_HI,
            )
        ),
        dims=latent.dims,
        coords=latent.coords,
    )

    frame = panel.frame
    screen = pd.DataFrame(
        {
            "isin": panel.isins,
            "ticker": frame.get("ticker"),
            "name": frame.get("name"),
            "sector": frame.get("sector"),
            "industry": frame.get("industry"),
            "trading_region": frame.get("trading_region"),
            "country": frame.get("country"),
            "style_class": frame.get("style_class"),
            "size_class": frame.get("size_class"),
            "n_analysts": pd.to_numeric(frame.get("n_analysts"), errors="coerce"),
            "market_cap": pd.to_numeric(frame.get("market_cap"), errors="coerce"),
            "mcap_global_r": pd.to_numeric(frame.get("feat_mcap_global_r"), errors="coerce"),
            "mcap_country_r": pd.to_numeric(frame.get("feat_mcap_country_r"), errors="coerce"),
            "last_price": pd.to_numeric(frame.get("last_price"), errors="coerce"),
            "observed_pt": pd.to_numeric(frame.get("observed_pt"), errors="coerce"),
            "expected_upside": upside.mean(axis=0),
            "expected_upside_sd": upside.std(axis=0),
            # v1's column names, not v2's first draft. compute_cvar_aware_book
            # requires expected_pt_hdi_lo/hi by name; supplying
            # expected_upside_p05/p95 instead silently degrades three risk
            # columns rather than raising.
            "prob_pos": (upside > 0).mean(axis=0),
        }
    )
    screen["implied_upside"] = screen["observed_pt"] / screen["last_price"] - 1.0
    screen["expected_pt"] = screen["last_price"] * (1.0 + screen["expected_upside"])
    screen["expected_pt_hdi_lo"] = screen["last_price"] * (
        1.0 + np.percentile(upside, 3, axis=0)
    )
    screen["expected_pt_hdi_hi"] = screen["last_price"] * (
        1.0 + np.percentile(upside, 97, axis=0)
    )
    screen["shrink_gain"] = shrink_gain
    screen["risk_adj_return"] = _posterior_mean(idata, "risk_adj_return", panel)
    # ``kalman_gain`` was ``sigmoid(risk_adj_return)`` — a sigmoid of a
    # STANDARDISED LOG-UPLIFT, which is not the probability of any defined
    # event. It has no calibration target, it pins at 0.5 wherever the
    # risk-adjusted latent is near zero, and its prior (sigmoid of a wide
    # unconstrained latent) piles mass at both ends. On run 49e84d7e9d59 the
    # exported column correlated -0.004 with analyst count and +0.06 with the
    # shrinkage actually applied. It is now the posterior probability that the
    # risk-adjusted latent is positive: a real tail probability of a stated
    # event, and a reduction over draws rather than a per-draw Deterministic,
    # which is why it lives here and not in the model graph.
    try:
        _rar = _posterior_draws(idata, "risk_adj_return")  # (isin, sample)
        screen["kalman_gain"] = (_rar > 0.0).mean(axis=1)
    except KeyError:  # pragma: no cover - defensive
        logger.warning(
            "risk_adj_return absent from the posterior; kalman_gain falls back "
            "to 1.0 and p_upside_pos_cond degrades to the unconditional MC "
            "probability."
        )
        screen["kalman_gain"] = 1.0

    # ---- Monte-Carlo forward returns (v1 §10 wiring) ------------------------
    # mu and sigma must be de-standardised onto the response scale BEFORE the
    # simulation, and the draws clipped in log space afterwards. Skipping the
    # de-standardisation yields z-scores rather than returns — a real historical
    # bug that reached the exported table.
    # Both arrays must be (n_isin, n_samples) and share the sample ordering.
    # ``_posterior_draws`` already returns that orientation; ``draws`` is
    # (sample, isin) and needs the transpose.
    sigma_draws = _posterior_draws(idata, "sigma_isin") * panel.response_std
    mu_draws = (draws * panel.response_std + panel.response_mean).T  # (isin, sample)
    nu_draws = _posterior_draws(idata, "nu", per_isin=False)
    if sigma_draws.shape != mu_draws.shape:
        raise ValueError(
            f"MC input shape mismatch: sigma {sigma_draws.shape} vs mu "
            f"{mu_draws.shape}. Both must be (n_isin, n_samples)."
        )
    mc = simulate_lagged_risk_adjusted_returns(
        mu_draws,
        sigma_draws,
        nu_draws,
        horizon=run_cfg.mc_horizon,
        rho=run_cfg.mc_rho,
        random_seed=run_cfg.random_seed,
    )
    # In place, and in this order, because the array is large: at the production
    # budget it is (6500, chains*draws, horizon) = ~1.7 GB. Clip -> read the log
    # array -> expm1 over the SAME buffer keeps exactly one copy alive. Binding
    # the clipped log array to its own name and then allocating the return-space
    # array from it holds two, and adding a risk-adjusted copy holds three, which
    # does not fit.
    np.clip(mc, LOG_UPLIFT_CLIP_LO, LOG_UPLIFT_CLIP_HI, out=mc)
    p_cond = _risk_adjusted_prob_positive(idata, panel, mc)
    np.expm1(mc, out=mc)
    mc_summary = summarize_mc_returns(mc, panel.isins)
    screen = screen.merge(
        mc_summary.rename(columns={"prob_pos": "mc_prob_pos"}), on="isin", how="left"
    )

    if p_cond is not None:
        screen["p_upside_pos_cond"] = p_cond
    else:
        # Documented fallback, matching compute_cvar_aware_book's degrade-with-a-
        # warning pattern: the product form still orders names, it just cannot be
        # read as a probability of anything.
        screen["p_upside_pos_cond"] = (
            screen["mc_prob_pos"].fillna(screen["prob_pos"])
            * screen["kalman_gain"].fillna(1.0)
        )

    screen = screen.sort_values("expected_upside", ascending=False).reset_index(drop=True)

    # ---- shrinkage slope gate ---------------------------------------------
    valid = screen[["expected_upside", "implied_upside"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(valid) >= 100:
        slope, intercept = np.polyfit(valid["implied_upside"], valid["expected_upside"], 1)
        above = float((screen["expected_upside"] > screen["implied_upside"]).mean())
        # Slope and intercept grade CALIBRATION; rho and the median revision
        # grade DISAGREEMENT. Both halves are needed: an exact copy of the input
        # scores a perfect 1.0 / 0.0 on the first pair, which is how run
        # 49e84d7e9d59 passed while reproducing consensus at rho 0.999995.
        rho = float(valid["expected_upside"].corr(valid["implied_upside"], method="spearman"))
        revision_pp = float((valid["expected_upside"] - valid["implied_upside"]).abs().median() * 100.0)
        # The universe-wide shift, which is what the old |intercept| threshold
        # was reaching for. The intercept itself is (1 - slope) * centre by
        # construction and so cannot separate an offset from shrinkage; see
        # gate_shrinkage_center_shift_max.
        center_shift = float(valid["expected_upside"].mean() - valid["implied_upside"].mean())
        slope_ok = run_cfg.gate_shrinkage_slope_lo <= slope <= run_cfg.gate_shrinkage_slope_hi
        shift_ok = abs(center_shift) <= run_cfg.gate_shrinkage_center_shift_max
        rho_ok = not (np.isfinite(rho) and rho > run_cfg.gate_shrinkage_rho_max)
        rev_ok = revision_pp >= run_cfg.gate_shrinkage_revision_min_pp
        report.add(
            GateResult(
                name="shrinkage_slope",
                passed=bool(slope_ok and shift_ok and rho_ok and rev_ok),
                value=(
                    f"slope {slope:.3f}, shift {center_shift:+.4f}, above {above:.1%}, "
                    f"rho {rho:.5f}, median revision {revision_pp:.2f}pp "
                    f"(intercept {intercept:+.4f})"
                ),
                threshold=(
                    f"slope in [{run_cfg.gate_shrinkage_slope_lo}, "
                    f"{run_cfg.gate_shrinkage_slope_hi}], |shift| <= "
                    f"{run_cfg.gate_shrinkage_center_shift_max}, rho <= "
                    f"{run_cfg.gate_shrinkage_rho_max}, revision >= "
                    f"{run_cfg.gate_shrinkage_revision_min_pp}pp"
                ),
                detail=(
                    "A shift of the centre is a universe-wide offset, not a "
                    "signal. Expect ~50% of names above consensus, not 80%. A rho "
                    "at the ceiling or a revision at the floor means the opposite "
                    "failure: the screen is a consensus sort, and the drift betas "
                    "and the hierarchy are being estimated but are not reaching "
                    "the exported number. The intercept is reported for continuity "
                    "but is not graded -- it equals (1 - slope) * centre."
                ),
            )
        )

    # ---- coverage gradient gate -------------------------------------------
    cov = screen.dropna(subset=["n_analysts"]).copy()
    if len(cov) >= 500:
        cov["bucket"] = pd.cut(
            cov["n_analysts"], [0, 3, 8, 20, np.inf], labels=["1-3", "4-8", "9-20", "21+"]
        )
        col = "er_sd" if "er_sd" in cov.columns else "expected_upside_sd"
        grad = cov.groupby("bucket", observed=True)[col].mean()
        monotone = bool(grad.is_monotonic_decreasing)
        spread = float(grad.max() / max(grad.min(), _EPS))
        report.add(
            GateResult(
                name="coverage_gradient",
                passed=monotone and spread >= 2.0,
                value=f"{'monotone' if monotone else 'NOT monotone'}, spread {spread:.2f}x",
                threshold="monotone decreasing, spread >= 2x",
                blocking=False,
                detail=f"mean {col} by bucket: {grad.round(4).to_dict()}",
            )
        )

    # ---- prob_pos degeneracy (item 8) --------------------------------------
    pinned = float((screen["prob_pos"] >= 0.99995).mean())
    # The span check is the half that was missing. Grading only prob_pos meant
    # its collapse was reported while the column that INHERITED the ranking went
    # unmeasured; a future collapse of the promoted column would then have been
    # silent. p95 - p05 is the range the ranking actually has to work in.
    _cond = pd.to_numeric(screen.get("p_upside_pos_cond"), errors="coerce")
    span = (
        float(_cond.quantile(0.95) - _cond.quantile(0.05))
        if _cond is not None and _cond.notna().any()
        else float("nan")
    )
    span_ok = bool(np.isfinite(span) and span >= 0.30)
    report.add(
        GateResult(
            name="prob_pos_degenerate",
            passed=(pinned <= 0.60) and span_ok,
            value=(
                f"{pinned:.1%} pinned at 1.0, p_upside_pos_cond span "
                f"{span:.3f}"
            ),
            threshold="pinned <= 60%, p_upside_pos_cond p95-p05 >= 0.30",
            blocking=False,
            detail=(
                "prob_pos is a REPORTED diagnostic and is never ranked on. "
                "p_upside_pos_cond is the primary probability column -- since "
                "2026-08-20 it is P(risk-adjusted forward return > 0) computed "
                "directly from the MC draws, not the old mc_prob_pos * sigmoid "
                "product. The export ranks on it and suppresses it alongside the "
                "other out-of-support metrics."
            ),
        )
    )
    return screen, ScreenDraws(eu=eu_draws, mc_returns=mc, isins=panel.isins)


# =========================================================================== #
# §10b  Risk book                                                             #
# =========================================================================== #


def run_risk_book(
    idata: Any,
    panel: KalmanPanelV2,
    screen: pd.DataFrame,
    run_cfg: KalmanRunConfigV2,
    draws: Optional["ScreenDraws"] = None,
) -> Any:
    """Size a CVaR-aware long book, reusing :mod:`RiskBookModel` unchanged.

    ``compute_cvar_aware_book`` needs the screen frame to already carry
    ``er_mean`` / ``er_sd`` / ``er_p05`` and ``mc_prob_pos``. Without them
    ``expected_sharpe_ratio`` silently becomes NaN, ``tail_risk`` loses its Monte-Carlo
    loss leg, and ``p_upside_pos_cond`` degrades to ``p_upside_pos * kalman_gain``
    — three quiet degradations rather than one loud failure, which is why
    :func:`run_screen` builds those columns first.

    Parameters
    ----------
    draws
        The :class:`ScreenDraws` §10 returned. Supplies the **shrunk** upside
        draws and the Monte-Carlo return distribution. Re-resolving the latent
        here instead — which is what this function used to do — would size the
        book on the unshrunk latent while the screen reported the shrunk one,
        and leave ``cvar05`` computed from estimation uncertainty. Both are
        silent failures, so the fallback path warns loudly.

    Returns
    -------
    RiskBook or None
        ``None`` when the risk book cannot be computed; the caller keeps going
        with the screen alone rather than losing the whole run.
    """
    import xarray as xr

    from probabilistic_ml_model.pymc_models.RiskBookModel import compute_cvar_aware_book

    try:
        if draws is not None:
            eu = draws.eu
            return_draws = draws.pooled_returns
            return_draws_isins = draws.isins
        else:
            logger.warning(
                "run_risk_book called without ScreenDraws: re-resolving the "
                "latent (UNSHRUNK -- it will disagree with the screen) and "
                "computing cvar05 from posterior rather than return draws."
            )
            latent = resolve_screen_latent_v2(
                idata, latent=KALMAN_V2_SCREEN_LATENT, random_seed=run_cfg.random_seed
            )
            eu = xr.DataArray(
                np.expm1(
                    np.clip(
                        np.asarray(latent) * panel.response_std + panel.response_mean,
                        LOG_UPLIFT_CLIP_LO,
                        LOG_UPLIFT_CLIP_HI,
                    )
                ),
                dims=("chain", "draw", "isin"),
                coords={"isin": panel.isins},
            )
            return_draws = None
            return_draws_isins = None
        book = compute_cvar_aware_book(
            idata,
            eu,
            screen,
            alpha=run_cfg.cvar_alpha,
            cap=run_cfg.weight_cap,
            k_book=run_cfg.k_book,
            p_long=run_cfg.p_long,
            mcap_r_max=run_cfg.mcap_global_r_max,
            return_draws=return_draws,
            return_draws_isins=return_draws_isins,
            tail_risk_vol_floor_k=run_cfg.tail_risk_vol_floor_k,
        )
        logger.info(
            "Risk book: %d names, port_up %.3f, STARR %.3f, div %.3f",
            int(book.summary.get("n_book", 0)),
            book.summary.get("port_up", float("nan")),
            book.summary.get("starr_book", float("nan")),
            book.summary.get("div", float("nan")),
        )
        return book
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Risk book failed: %s", exc, exc_info=True)
        return None


def apply_out_of_support(results: pd.DataFrame) -> pd.DataFrame:
    """Flag clip-pinned rows and NULL their ranking metrics.

    Ported from v1 (``pymc_kalman_filter_pt.py:7098-7119``) including the reason
    it works: the test is on the **percentiles**, not the mean. ``er_mean``
    averages the clipped draws, so a handful of draws below the cap drag it a
    fraction under and a mean-based test matched **zero** of the affected names
    on the 2026-08-15 export. A distribution is pinned at the upper bound exactly
    when its 5th percentile has reached it, and at the lower bound exactly when
    its 95th has.

    Identity, price targets and the raw ``er_*`` distribution are deliberately
    retained — the row is still informative, it just must not be *ranked*. An
    unbounded ratio gets noticed; a Sharpe of 717 looks like the best opportunity
    in the book and sorts to the top of every risk-adjusted view, while marking
    the names the model understands least.

    **Both distributions are tested, not just the Monte-Carlo one.** The ranking
    columns this protects are built from two different draws: ``er_*`` comes from
    the forward-return Monte Carlo, while ``cvar05`` / ``exp_vol`` — and so
    ``reward_to_cvar`` and ``expected_sharpe_ratio`` — come from the Kalman
    upside posterior. Testing only ``er_*`` let a row through on 2026-08-18 whose
    Kalman distribution was *entirely* pinned: ``expected_return_kalman`` 5.0,
    ``cvar_5pct_kalman`` 5.0, ``expected_vol_kalman`` 0.0 — a degenerate
    distribution sitting on the +500 % cap — while ``er_p05`` sat at 0.64 and
    cleared the test. Its ``tail_risk`` fell to the floor and its STARR came out
    at exactly 500. ``cvar_5pct_kalman`` is the Kalman distribution's lower-tail
    statistic, so it plays the role ``er_p05`` plays for the Monte Carlo: at the
    upper bound exactly when every draw has reached it.
    """
    out = results.copy()
    hi_key = "er_p05" if "er_p05" in out.columns else "er_mean"
    lo_key = "er_p95" if "er_p95" in out.columns else "er_mean"
    if hi_key not in out.columns:
        out["out_of_support"] = False
        return out

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce")

    pinned_hi = (_num(hi_key) >= UPLIFT_CLIP_HI - 1e-6).fillna(False)
    pinned_lo = (_num(lo_key) <= UPLIFT_CLIP_LO + 1e-6).fillna(False)

    # The Kalman upside posterior, tested the same way on its own tails.
    #
    # This leg USED to be ``cvar_5pct_kalman >= UPLIFT_CLIP_HI``, because that
    # column was the upside posterior's lower-tail statistic and so reached the
    # cap exactly when every posterior draw had. Since 2026-08-20 the column
    # holds the Monte-Carlo return CVaR instead, which is bounded above by
    # ``er_p05`` and therefore only ever re-detects what the ``er_p05`` test
    # above already caught. Testing the upside posterior's MEAN at the cap
    # restores the protection on the correct distribution: the clip is one-sided
    # at each end, so the mean can only sit on a bound when every draw does.
    # Both ends are now tested on the same column, which is what makes it
    # symmetric — the 2026-08-18 row that slipped through (expected upside 5.0,
    # posterior fully degenerate, er_p05 at 0.64) is caught by the upper test.
    kalman_mean = next(
        (c for c in ("expected_return_kalman", "expected_upside") if c in out.columns),
        None,
    )
    if kalman_mean is not None:
        pinned_hi = pinned_hi | (
            _num(kalman_mean) >= UPLIFT_CLIP_HI - 1e-6
        ).fillna(False)
        pinned_lo = pinned_lo | (
            _num(kalman_mean) <= UPLIFT_CLIP_LO + 1e-6
        ).fillna(False)

    oos = (pinned_hi | pinned_lo).to_numpy()
    out["out_of_support"] = oos

    present = [c for c in _RANKING_COLS if c in out.columns]
    for col in present:
        out.loc[oos, col] = np.nan
    # Re-fill the weight so gross exposure stays well defined.
    if "cvar_book_weight" in out.columns:
        out.loc[oos, "cvar_book_weight"] = 0.0
    if oos.any():
        logger.warning(
            "out_of_support: %d name(s) pinned at a clip bound (%d at the +%.0f%% "
            "cap, %d at the %.0f%% floor); %s set to NULL",
            int(oos.sum()),
            int(pinned_hi.sum()),
            UPLIFT_CLIP_HI * 100,
            int(pinned_lo.sum()),
            UPLIFT_CLIP_LO * 100,
            ", ".join(present),
        )
    return out


# =========================================================================== #
# §10c  Export                                                                #
# =========================================================================== #


def export_analytics(
    frames: dict[str, pd.DataFrame],
    run_cfg: KalmanRunConfigV2,
    report: GateReport,
    *,
    run_id: str,
) -> dict[str, int]:
    """Write the curated frames to **v2-suffixed** tables, gated.

    Every table name carries a ``_v2`` suffix, so v1 and the live GEIB dashboard
    are untouched and the two models can be compared on the same database.
    Promotion is a deliberate edit to :data:`_ANALYTICS_TABLE_V2` and the frame
    keys, not a side effect of running this.

    Four blocking gates, each encoding a failure this pipeline has actually
    shipped:

    ``export_rowcount``
        Every frame non-empty **and** the per-ISIN frames agreeing on row count.
        v1's ``10c_kalman_results`` existed with zero rows, so a naive
        one-``run_id``-everywhere vintage check passed over it — a silent failure
        that is worse than a loud one.
    ``export_finite``
        No non-finite value in any numeric column.
    ``export_ranking_range``
        No ranking metric outside +/-100 on a row that was not suppressed. This
        is the ``-4.28e15`` Sharpe that reached the dashboard, and the
        ``-2141.80`` that still sits in v1's intermediate risk table.
    ``export_vintage``
        One ``run_id`` across every table written.
    """
    stamped = datetime.now(timezone.utc)
    counts: dict[str, int] = {}
    frames = {k: v for k, v in frames.items()}

    # ---- gate: non-empty and row-count agreement ---------------------------
    empties = [k for k, v in frames.items() if v is None or v.empty]
    per_isin = {
        k: len(v)
        for k, v in frames.items()
        if v is not None and not v.empty and "isin" in v.columns and k != _RISK_BOOK_KEY
    }
    disagree = len(set(per_isin.values())) > 1
    report.add(
        GateResult(
            name="export_rowcount",
            passed=(not empties) and (not disagree),
            value=(
                f"{len(frames) - len(empties)}/{len(frames)} non-empty; "
                f"per-ISIN counts {sorted(set(per_isin.values()))}"
            ),
            threshold="all non-empty, per-ISIN frames agree",
            detail=(
                f"empty: {empties}" if empties else
                f"row counts differ: {per_isin}" if disagree else ""
            ),
        )
    )

    # ---- gate: finiteness ---------------------------------------------------
    bad: list[str] = []
    for key, df in frames.items():
        if df is None or df.empty:
            continue
        num = df.select_dtypes(include=[np.number])
        if num.size and not np.isfinite(num.to_numpy(dtype="float64", na_value=0.0)).all():
            bad.append(key)
    report.add(
        GateResult(
            name="export_finite",
            passed=not bad,
            value=f"{len(bad)} frame(s) with non-finite values",
            threshold="all numeric cells finite",
            detail=f"offending: {bad}" if bad else "",
        )
    )

    # ---- gate: no duplicate column names -----------------------------------
    # Cheap, and it fires BEFORE the loops that would otherwise die on it. A
    # duplicated name makes `df[col]` a DataFrame, so `pd.to_numeric` below
    # raises "arg must be a list, tuple, 1-d array, or Series" and the export
    # aborts after the fit is already paid for. It is easy to reintroduce: any
    # `rename()` that maps one existing column onto another existing one does it
    # silently, which is how both `expected_sharpe -> expected_sharpe_ratio`
    # sites managed it.
    dupes = {
        key: sorted({c for c in df.columns if list(df.columns).count(c) > 1})
        for key, df in frames.items()
        if df is not None and not df.empty
        and len(df.columns) != len(set(df.columns))
    }
    report.add(
        GateResult(
            name="export_unique_columns",
            passed=not dupes,
            blocking=True,
            value=f"{len(dupes)} frame(s) with duplicate columns",
            threshold="every frame has unique column names",
            detail="; ".join(f"{k}: {v}" for k, v in dupes.items()) if dupes else "",
        )
    )
    if dupes:
        # Returning here is deliberate: every remaining stage indexes by column
        # name, so continuing produces a cascade of confusing errors instead of
        # this one clear verdict.
        logger.error("Duplicate columns block the export: %s", dupes)
        return counts

    # ---- gate: no duplicated column CONTENT --------------------------------
    # The sibling of the gate above, and the reason it needed one. A frame can
    # carry the same numbers under two names and pass `export_unique_columns`
    # perfectly correctly -- that gate checks duplicate NAMES. `10b_risk_book_v2`
    # shipped `weight` beside `book_weight` and `expected_sharpe` beside
    # `expected_sharpe_ratio`, byte-identical, for two releases, because the
    # de-duplicating drop was applied to the analytics frame and not to the book.
    #
    # WARN, never blocking. Two genuinely all-zero or all-constant columns are
    # legitimate (`cvar_book_weight` is 0.0 for every name outside the book), and
    # aborting an export over one would cost a run a fit that is already paid
    # for. The verdict names the PAIRS, so the next reader gets the list rather
    # than a count and can tell a real alias from a coincidence.
    content_dupes: dict[str, list[str]] = {}
    for key, df in frames.items():
        if df is None or df.empty or len(df.columns) < 2:
            continue
        # `duplicated` on the transpose compares whole rows -- i.e. whole columns
        # of the original -- and treats NaN as equal to NaN, which is what is
        # wanted: two all-NaN aliases are still one quantity under two names.
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            continue
        try:
            marks = numeric.T.duplicated(keep=False)
        except TypeError:  # pragma: no cover - unhashable/ragged dtypes
            continue
        if not marks.any():
            continue
        # Group the marked columns into equal-valued sets so the detail reads
        # "book_weight == weight" rather than a flat list of four names.
        groups: dict[tuple, list[str]] = {}
        for col in numeric.columns[marks.to_numpy()]:
            sig = tuple(pd.isna(numeric[col]).tolist()), tuple(
                numeric[col].fillna(0.0).to_numpy().tolist()
            )
            groups.setdefault(sig, []).append(str(col))
        pairs = [" == ".join(sorted(g)) for g in groups.values() if len(g) > 1]
        if pairs:
            content_dupes[key] = sorted(pairs)
    report.add(
        GateResult(
            name="export_duplicate_content",
            passed=not content_dupes,
            blocking=False,
            value=f"{len(content_dupes)} frame(s) with duplicated content",
            threshold="one column per quantity in every frame",
            detail=(
                "; ".join(f"{k}: {', '.join(v)}" for k, v in content_dupes.items())
                if content_dupes
                else ""
            ),
        )
    )
    if content_dupes:
        logger.warning(
            "Frames carry the same quantity under two names: %s. Pick one name "
            "per quantity at the SOURCE (compute_cvar_aware_book), not with a "
            "drop here -- a drop applied to one frame and not another is what "
            "produced this.",
            content_dupes,
        )

    # ---- gate: ranking metrics in range ------------------------------------
    offenders: list[str] = []
    for key, df in frames.items():
        if df is None or df.empty:
            continue
        keep = ~df["out_of_support"].fillna(False) if "out_of_support" in df else slice(None)
        for col in _RANKING_RANGE_COLS:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df.loc[keep, col], errors="coerce").dropna()
            if vals.size and (vals.abs() > 100).any():
                offenders.append(f"{key}.{col} (min {vals.min():.1f}, max {vals.max():.1f})")
    report.add(
        GateResult(
            name="export_ranking_range",
            passed=not offenders,
            value=f"{len(offenders)} column(s) out of range",
            threshold="|metric| <= 100 on non-suppressed rows",
            detail="; ".join(offenders) if offenders else "",
        )
    )

    # ---- write --------------------------------------------------------------
    # CSVs are written unconditionally; only the DATABASE write is gated.
    #
    # The two are different decisions. A failed model gate means the fit is not
    # fit to publish — it does not mean the artifacts are worthless, and they are
    # exactly what someone needs to work out *why* it failed. An earlier build
    # returned before writing anything when any blocking gate failed, and the
    # first thing that happened was a failed run leaving nothing to diagnose it
    # with. Publish nothing, keep everything.
    publish = report.ok
    if not publish:
        logger.error(
            "Blocking gates failed: writing CSV artifacts for diagnosis but "
            "REFUSING the analytics database write. Failed: %s",
            ", ".join(r.name for r in report.blocking_failures),
        )

    out_dir = run_cfg.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = None
    schema = os.environ.get("DB_ANALYTICS_SCHEMA", "analytics")
    if publish and run_cfg.write_analytics and os.environ.get("DB_URL"):
        from sqlalchemy import create_engine

        engine = create_engine(os.environ["DB_URL"])

    # Render the DDL before any table write, and unconditionally: it needs no
    # connection, so an offline run still leaves a reviewable schema.
    if _ANALYTICS_TABLE_V2 in frames and not frames[_ANALYTICS_TABLE_V2].empty:
        stamped_canonical = stamp_export_provenance(
            frames[_ANALYTICS_TABLE_V2], run_id, stamped
        )
        try:
            write_analytics_ddl_v2(stamped_canonical)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("DDL render failed: %s", exc)

    # Every other v2 frame gets a DDL too, since 2026-08-22. Only the canonical
    # table used to, so the six pipeline tables landed via `to_sql` with no
    # readable schema and no record of what their columns mean — the exact gap
    # the v1 convention exists to close, applied to one table out of seven.
    # `_ANALYTICS_COLUMN_COMMENTS_V2` is keyed by column name, so a column
    # documented once is documented everywhere it appears.
    for _key, _frame in frames.items():
        if _key == _ANALYTICS_TABLE_V2 or _frame is None or _frame.empty:
            continue
        try:
            write_analytics_ddl_v2(
                stamp_export_provenance(_frame, run_id, stamped), table=_key
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("DDL render failed for %s: %s", _key, exc)

    sql_ok = engine is not None
    for key, df in frames.items():
        if df is None or df.empty:
            continue
        stamped_df = stamp_export_provenance(df, run_id, stamped)
        counts[key] = len(stamped_df)

        wrote_table = False
        if sql_ok and publish:
            try:
                stamped_df.to_sql(
                    key, engine, schema=schema, if_exists="replace", index=False
                )
                logger.info("wrote %s.%s (%d rows)", schema, key, len(stamped_df))
                wrote_table = True
            except Exception as exc:
                # Memo the failure: a dead connection fails identically for every
                # remaining frame, and retrying it once per frame turns one error
                # into a wall of them.
                logger.error("SQL export failed for %s (%s); CSV from here on", key, exc)
                sql_ok = False
        if not wrote_table:
            stamped_df.to_csv(out_dir / f"{key}.csv", index=False)

    # ---- gate: single vintage ----------------------------------------------
    if engine is not None and counts:
        report.add(_check_export_vintage_v2(engine, schema, list(counts), run_id))

    # ---- the gate report itself, written LAST -------------------------------
    # Deliberately not a member of `frames`: this function ADDS four gates
    # (rowcount, finite, ranking_range, vintage), so a snapshot taken before the
    # loop would omit exactly the gates that describe the export. Writing it here
    # is the only ordering that captures every gate the run produced.
    #
    # Why it exists at all: every gate verdict, the divergence count and all four
    # PPC statistics used to be printed by `summarise()` and dropped. A completed
    # run was unauditable an hour later, and answering any question about its
    # calibration meant refitting it.
    gate_df = report.to_frame()
    if not gate_df.empty:
        gate_df = stamp_export_provenance(gate_df, run_id, stamped)
        wrote = False
        if sql_ok and publish and engine is not None:
            try:
                gate_df.to_sql(
                    _GATE_REPORT_KEY, engine, schema=schema,
                    if_exists="replace", index=False,
                )
                logger.info(
                    "wrote %s.%s (%d gates)", schema, _GATE_REPORT_KEY, len(gate_df)
                )
                wrote = True
            except Exception as exc:
                logger.error("SQL export failed for %s (%s); CSV only", _GATE_REPORT_KEY, exc)
        if not wrote:
            gate_df.to_csv(out_dir / f"{_GATE_REPORT_KEY}.csv", index=False)
        counts[_GATE_REPORT_KEY] = len(gate_df)
    return counts


#: Column documentation for the v2 analytics table. Only the columns whose
#: meaning is *not* obvious from the name, or whose name is actively misleading,
#: are documented — the three at the top are the ones three consecutive reviews
#: had to re-explain because the name says something the column does not do.
_ANALYTICS_COLUMN_COMMENTS_V2: dict[str, str] = {
    "cvar_5pct_kalman": (
        "5% expected shortfall of the Monte-Carlo FORWARD-RETURN distribution: "
        "the mean of the worst 5% of simulated returns. Raw decimal, and "
        "genuinely negative for a name with downside. Changed 2026-08-20 -- it "
        "was previously the tail mean of the posterior EXPECTED-UPSIDE draws, "
        "i.e. estimation uncertainty about a point, which made it positive for "
        "88.4% of names and correlated 0.9998 with expected_return_kalman. For "
        "the estimation-uncertainty view use expected_upside_sd."
    ),
    "expected_sharpe_ratio": (
        "er_mean / er_sd over Monte-Carlo draws of log price-target uplift. A "
        "t-statistic on the distance from price to the smoothed target, NOT an "
        "investment Sharpe ratio -- the numerator is an uplift, not a realised "
        "excess return. Median ~1.05, which is exactly the range a reader "
        "mistakes for a Sharpe. NULL when out_of_support."
    ),
    "expected_vol_kalman": (
        "Standard deviation of the Monte-Carlo forward-return draws -- the same "
        "quantity as er_sd. Raw decimal. Changed 2026-08-20: it was previously "
        "the posterior dispersion of expected upside (median 0.47pp against a "
        "return sd of 19.03pp, a factor of 40), which is why the dashboard "
        "derived its own volatility rather than using this column."
    ),
    "expected_upside_sd": (
        "Posterior sd of the per-name expected upside -- ESTIMATION uncertainty, "
        "not return risk. This is what expected_vol_kalman used to hold. Since "
        "2026-08-20 it also carries the forecast-error term, so it is roughly an "
        "order of magnitude wider than the 0.47pp of run 49e84d7e9d59. Raw decimal."
    ),
    "shrink_gain": (
        "Weight on the name's own smoothed observation in the forecast-error "
        "update; 1 - shrink_gain is the weight on the pooled drift + hierarchy "
        "prediction. Low for thinly covered or widely dispersed consensus. This "
        "is the column kalman_gain was mistakenly believed to be."
    ),
    "expected_return_kalman": "Posterior mean expected upside. Raw decimal (0.25 = +25%).",
    "price_target_kalman": "Smoothed price target, in the security's own currency.",
    "implied_upside": "Analyst consensus upside, original_target/original_price - 1. Raw decimal.",
    "er_mean": "Mean forward return over the Monte-Carlo horizon. Raw decimal.",
    "er_sd": "Pooled sd of the forward-return draws. Denominator of expected_sharpe_ratio.",
    "er_p05": "5th percentile forward return. Drives the upper out_of_support test.",
    "er_p95": "95th percentile forward return. Drives the lower out_of_support test.",
    "mc_prob_pos": "Share of Monte-Carlo forward-return draws above zero.",
    "p_upside_pos_cond": (
        "THE PRIMARY PROBABILITY COLUMN -- rank on this one. P(risk-adjusted "
        "forward return > 0): the share of Monte-Carlo forward-return draws that "
        "are positive after the same risk / size / volume penalties the model "
        "applies to risk_adj_return. Changed 2026-08-20 from mc_prob_pos * "
        "kalman_gain, a probability times a sigmoid, whose ordering was usable "
        "but whose level was the probability of nothing. NULL when "
        "out_of_support."
    ),
    "prob_pos": (
        "NON-RANKING -- reported diagnostic. Share of the per-name posterior "
        "above zero. Do not rank, filter or size on it; rank on "
        "p_upside_pos_cond. It saturates: 59.4% of the universe sat at exactly "
        "1.0 on run 0aa3397b1d01 and 87.4% on the pass-through 49e84d7e9d59, "
        "because the smoother is Rao-Blackwellised over the latent and the "
        "posterior sd collapsed to 0.47pp. The forecast-error term added "
        "2026-08-20 widens it and the prob_pos_degenerate gate warns if it "
        "re-pins -- but that gate passes at 59.4% against a 60% ceiling, so it "
        "is reporting the threshold as much as the model. A column pinned for "
        "three names in five has almost no ordering to offer."
    ),
    "kalman_gain": (
        "NON-RANKING -- reported diagnostic, and a DEPRECATED NAME. Removed from "
        "the GEIB selectable-metric surface on 2026-08-24: 54.0% of the universe "
        "sat at exactly 0 or exactly 1 on run 0aa3397b1d01, up from 50.6%, so it "
        "orders barely half the names and is degenerate for the rest. Rank on "
        "p_upside_pos_cond instead. "
        "Definition, for the rows where it is not pinned: P(risk_adj_return > 0) "
        "over posterior draws -- a real tail probability since 2026-08-20. It "
        "was sigmoid(risk_adj_return), a sigmoid of a standardised log-uplift, "
        "which is not the probability of any event and correlated -0.004 with "
        "analyst count. It is NOT a Kalman gain and never was: for the shrinkage "
        "weight, which is the quantity this name has always suggested, see "
        "shrink_gain."
    ),
    "reward_to_cvar": "expected_return_kalman / tail_risk (STARR). NULL when out_of_support.",
    "cvar_book_weight": "Weight in the CVaR-sized book, 0 outside it and 0 when out_of_support.",
    "out_of_support": (
        "TRUE when the forward-return distribution is pinned at a clip bound "
        "(er_p05 at the +500% cap or er_p95 at the -95% floor). Ranking metrics "
        "are NULL for these rows; identity and the raw er_* distribution are kept."
    ),
    "state_now_sd": "Posterior sd of the per-name latent at the snapshot, standardised scale.",
    "run_id": "Export provenance. Every table written by one run shares it.",
    "exported_at": "Export provenance timestamp (UTC).",
    "source_sha": (
        "Git HEAD SHA of the source that produced this run, or NULL outside a "
        "checkout. Cross-run comparisons -- notably any ELPD contrast between "
        "comparison arms fitted in different runs -- are only legitimate between "
        "rows sharing this value AND having source_dirty = FALSE."
    ),
    "source_dirty": (
        "TRUE when tracked files differed from source_sha at export time. NOT an "
        "error: most runs have an uncommitted tree. It is what tells a reader "
        "that source_sha does not fully determine the code that ran, so two runs "
        "sharing a SHA may still not share a specification."
    ),
}

_ANALYTICS_DDL_HEADER_V2 = """\
-- ===========================================================================
-- analytics.{table}
-- ===========================================================================
-- Generated by pymc_kalman_filter_pt_v2.py -- do not hand-edit; it is rewritten
-- on every export.
--
-- UNITS: all return-like columns are RAW DECIMALS (0.25 = +25%), per the
-- 0.9.9.7 convention. Percent scaling happens only at display boundaries.
--
-- This is the **v2** table. v1 continues to write
-- analytics.kalman_filtered_price_targets, which is what the GEIB dashboard
-- reads; the two coexist so the models can be compared on one database.
-- Promoting v2 means repointing the dashboard, not renaming this table.
-- ===========================================================================
"""


def write_analytics_ddl_v2(
    frame: pd.DataFrame,
    *,
    table: str = _ANALYTICS_TABLE_V2,
    out_path: Optional[Path] = None,
) -> Path:
    """Render a reviewable DDL file for the v2 analytics table.

    ``to_sql(if_exists="replace")`` creates a table but leaves no schema anyone
    can read, and no record of what the columns mean. v1 keeps that
    documentation in ``sql_scripts/analytics/kalman_filtered_price_targets.sql``
    and it is the only place the unit convention and the misleading column names
    are written down. v2 does the same, generated from the frame actually
    exported so schema and table cannot drift apart.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    type_map = {
        "int64": "BIGINT",
        "int32": "INTEGER",
        "float64": "DOUBLE PRECISION",
        "float32": "REAL",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
        "datetime64[ns, UTC]": "TIMESTAMPTZ",
    }
    cols: list[str] = []
    for name, dtype in frame.dtypes.items():
        sql_type = type_map.get(str(dtype), "TEXT")
        cols.append(f'    "{name}" {sql_type}')

    body = [_ANALYTICS_DDL_HEADER_V2.format(table=table)]
    body.append(f'DROP TABLE IF EXISTS analytics."{table}";')
    body.append(f'CREATE TABLE analytics."{table}"\n(\n' + ",\n".join(cols) + "\n);")
    body.append("")
    for name in frame.columns:
        doc = _ANALYTICS_COLUMN_COMMENTS_V2.get(str(name))
        if doc:
            escaped = doc.replace("'", "''")
            body.append(
                f'COMMENT ON COLUMN analytics."{table}"."{name}" IS\n    \'{escaped}\';'
            )
    body.append("")
    # The index is emitted only when the frame actually justifies it. It used to
    # be an unconditional UNIQUE on ``isin`` because this function only ever saw
    # the canonical per-ISIN table; applied to the other six that is either a
    # constraint the data violates (the risk book and the comparison table repeat
    # or omit ``isin``) or a column that is not there at all (the diagnostics and
    # gate frames). A DDL that will not apply is worse than no DDL.
    if "isin" in frame.columns:
        _unique = bool(frame["isin"].notna().all() and not frame["isin"].duplicated().any())
        body.append(
            f'CREATE {"UNIQUE " if _unique else ""}INDEX IF NOT EXISTS idx_{table}_isin'
        )
        body.append(f'    ON analytics."{table}" (isin);')
        body.append("")

    path = out_path or Path("sql_scripts") / "analytics" / f"{table}.sql"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")
    logger.info("wrote %s (%d columns)", path, len(frame.columns))
    return path


def _check_export_vintage_v2(engine: Any, schema: str, tables: list[str], run_id: str):
    """Confirm every written table carries exactly one, expected ``run_id``.

    Resolves which tables actually have a ``run_id`` column from
    ``information_schema`` **first**. A speculative ``SELECT run_id`` against a
    table without the column aborts the PostgreSQL transaction and every later
    query on that connection fails with ``InFailedSqlTransaction`` — the trap
    v1's ``check_export_vintage`` documents.
    """
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            have = {
                r[0]
                for r in conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.columns "
                        "WHERE table_schema = :s AND column_name = 'run_id'"
                    ),
                    {"s": schema},
                )
            }
            found: dict[str, set[str]] = {}
            for tbl in tables:
                if tbl not in have:
                    continue
                rows = conn.execute(
                    text(f'SELECT DISTINCT run_id FROM {schema}."{tbl}"')
                ).fetchall()
                found[tbl] = {r[0] for r in rows}
        stale = {t: v for t, v in found.items() if v != {run_id}}
        return GateResult(
            name="export_vintage",
            passed=not stale and len(found) == len(tables),
            value=f"{len(found)}/{len(tables)} tables stamped {run_id}",
            threshold="every written table carries exactly this run_id",
            detail=f"divergent: {stale}" if stale else "",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return GateResult(
            name="export_vintage",
            passed=False,
            value=f"check failed: {exc}",
            threshold="every written table carries exactly this run_id",
            blocking=False,
        )


# =========================================================================== #
# §14  Orchestration                                                          #
# =========================================================================== #


def summarise(
    report: GateReport,
    extras: dict[str, Any],
    *,
    results_path: Optional[Path] = None,
) -> None:
    """Print the gate report, and persist it when a results directory is given.

    Parameters
    ----------
    report
        The run's gates.
    extras
        Headline numbers printed under the report.
    results_path
        Artifact root. When supplied, ``report.to_frame()`` is also written to
        ``<results_path>/09_gate_report_v2.csv``.

    Notes
    -----
    The CSV is the reason this function takes a path at all. ``summarise`` is
    called on **every** terminating path — dry run, benchmark, panel-audit
    failure, runtime-gate failure and success — whereas
    :func:`export_analytics` runs only on the last of those. Without this, an
    aborted run left its gate verdicts in console scrollback and nowhere else,
    which is precisely the situation where someone needs them.

    On the success path the export writes the same frame again, to the database
    and with the four export gates included. That one is authoritative; this is
    the floor.
    """
    print(report.render())
    if extras:
        print("\nRun summary")
        print("-" * 40)
        for k, v in extras.items():
            print(f"  {k:<28} {v}")
    if results_path is not None:
        try:
            frame = report.to_frame()
            if not frame.empty:
                results_path.mkdir(parents=True, exist_ok=True)
                frame.to_csv(results_path / f"{_GATE_REPORT_KEY}.csv", index=False)
        except Exception as exc:  # pragma: no cover - never lose a run to this
            logger.warning("Could not persist the gate report: %s", exc)


def _render_figures(result: dict[str, Any], panel: KalmanPanelV2,
                    run_cfg: KalmanRunConfigV2) -> None:
    """Draw the run's panels, and never let a figure cost the run anything.

    Called LAST on every terminating path -- after ``export_analytics`` and after
    ``summarise`` -- so the analytics tables and the gate report are already on
    disk before a plotting library gets a chance to fail. The import is deferred
    for the same reason the panels are: ``kalman_viz_v2`` pulls in plotly,
    matplotlib and seaborn, and a workflow that only wants the tables should not
    pay for them, nor fail to run if they are missing.

    On the ``--dry-run`` path this draws the §4b panel audit and the decay
    ladder, which are the two figures that make a dry run worth looking at
    rather than merely reading.
    """
    if not run_cfg.export_figures:
        logger.info("figures disabled (--no-figures)")
        return
    try:
        import kalman_viz_v2 as viz
    except Exception as exc:  # pragma: no cover - optional plotting stack
        logger.warning(
            "figures skipped: kalman_viz_v2 is unavailable (%s). The analytics "
            "export and the gate report are unaffected.", exc,
        )
        return
    try:
        viz.install(run_cfg)
        viz.render_run(result, panel, run_cfg)
    except Exception as exc:  # pragma: no cover - figures are best-effort
        logger.warning("figure rendering failed: %s", exc)


def main(
    *,
    model_config: Optional[KalmanModelConfig] = None,
    run_config: Optional[KalmanRunConfigV2] = None,
    dry_run: bool = False,
    benchmark: bool = False,
) -> dict[str, Any]:
    """Run the v2 workflow end to end.

    Parameters
    ----------
    model_config
        Everything that changes the posterior. Defaults to
        :class:`KalmanModelConfig()`.
    run_config
        Everything that changes only the run. Defaults to
        :meth:`KalmanRunConfigV2.from_env`.
    dry_run
        Stop after the panel audit (§4b). Seconds rather than half an hour, and
        the right way to evaluate a change to ``lookbacks``.

    Returns
    -------
    dict[str, Any]
        ``report``, ``panel``, ``panel_audit`` and — on a full run — ``idata``,
        ``screen``, ``diagnostics``, ``ppc``, ``run_id``.
    """
    model_cfg = model_config or KalmanModelConfig()
    run_cfg = run_config or KalmanRunConfigV2.from_env()
    logging.basicConfig(
        level=getattr(logging, run_cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    run_id = uuid.uuid4().hex[:12]
    logger.info("Kalman v2 run %s -- %s", run_id, model_cfg.describe())

    report = GateReport()
    frame = load_kalman_frame(run_cfg)
    panel = prepare_panel(frame, model_cfg, run_cfg)
    audit = run_panel_diagnostics(panel, run_cfg, report)

    result: dict[str, Any] = {
        "run_id": run_id,
        "panel": panel,
        "panel_audit": audit,
        "report": report,
    }
    if dry_run:
        summarise(
            report,
            {
                "names": panel.n_isin,
                "T": panel.n_time,
                "T_eff": f"{audit['t_eff']:.2f}",
                "drift features": len(panel.drift_names),
                "grid (days)": panel.time_days.tolist(),
            },
            results_path=run_cfg.results_path,
        )
        _render_figures(result, panel, run_cfg)
        return result

    if not report.ok:
        logger.error("Panel audit failed; not fitting. Fix the grid first.")
        summarise(report, {"names": panel.n_isin, "T_eff": f"{audit['t_eff']:.2f}"},
                  results_path=run_cfg.results_path)
        return result

    model = build_kalman_pt_model_v2(panel, model_cfg)

    # Measure before committing. A model that cannot finish is caught here, in
    # seconds, rather than 45 minutes into a run that has produced eight draws.
    result["runtime"] = run_runtime_estimate(model, run_cfg, report)
    if benchmark:
        summarise(report, dict(result["runtime"]), results_path=run_cfg.results_path)
        return result
    if not report.ok:
        logger.error("Runtime gate failed; not sampling.")
        summarise(report, dict(result["runtime"]), results_path=run_cfg.results_path)
        return result

    result["prior_idata"] = run_prior_predictive(model, panel, run_cfg, report)
    idata = sample_posterior(model, run_cfg)
    result["idata"] = idata
    result["diagnostics"] = run_diagnostics(idata, panel, run_cfg, report)
    result["ppc"] = run_posterior_predictive(
        model, idata, panel, run_cfg, model_cfg, report
    )

    # §9b — opt-in, and it refits every arm from scratch, so it runs AFTER the
    # production fit is complete and its gates are recorded. A comparison that
    # crashes must not cost the run its screen and export.
    if run_cfg.enable_model_comparison:
        result["comparison"] = run_model_comparison(frame, model_cfg, run_cfg, report)

    # §9b-fast — the Max-and-Smooth SCREEN. Reuses the production fit that just
    # finished rather than refitting anything, so it is cheap enough to leave on;
    # it ranks arms and never decides one. Same placement rule as §9b: after the
    # production gates are recorded, and a failure here costs nothing else.
    if run_cfg.enable_fast_comparison:
        result["comparison_fast"] = compare_arms_fast(
            panel, idata, model_cfg, run_cfg, report,
            arms=run_cfg.fast_comparison_arms or None,
        )

    screen, screen_draws = run_screen(idata, panel, run_cfg, report)
    result["screen"] = screen
    result["screen_draws"] = screen_draws

    risk_book = run_risk_book(idata, panel, screen, run_cfg, draws=screen_draws)
    result["risk_book"] = risk_book

    # The canonical frame: the risk book's analytics if we have it (it is the
    # screen plus the risk columns), otherwise the screen alone.
    kalman_results = (
        risk_book.analytics.copy() if risk_book is not None else screen.copy()
    )
    # A one-release guard, not a live hazard. `compute_cvar_aware_book` stopped
    # emitting the `expected_sharpe` alias on 2026-08-24, so this is a no-op
    # against the current RiskBookModel. It is retained because a stale or
    # pinned RiskBookModel that still emits both would otherwise rename one onto
    # the other and produce two columns with that name -- not merely untidy: the
    # export_ranking_range gate then hands `pd.to_numeric` a DataFrame instead
    # of a Series and the whole export dies with "arg must be a list, tuple,
    # 1-d array, or Series", after the fit has already been paid for.
    kalman_results = kalman_results.drop(columns=["expected_sharpe"], errors="ignore")
    kalman_results = kalman_results.rename(
        columns={
            "starr": "reward_to_cvar",
            "cvar05": "cvar_5pct_kalman",
            "exp_vol": "expected_vol_kalman",
            "book_weight": "cvar_book_weight",
            "expected_upside": "expected_return_kalman",
            "expected_pt": "price_target_kalman",
            "last_price": "original_price",
            "observed_pt": "original_target",
        }
    )
    # Suppression runs BEFORE anything is written, so every consumer — including
    # the intermediate risk table — sees the same guarded values. In v1 this ran
    # after 10b_risk_analytics was persisted, which is why that table still
    # carries an expected_sharpe_ratio of -2,142.
    kalman_results = apply_out_of_support(kalman_results)
    result["kalman_results"] = kalman_results

    frames: dict[str, pd.DataFrame] = {
        "04_panel_frame_v2": panel.frame,
        "09_diagnostics_v2": result["diagnostics"].reset_index(),
        "10_screen_results_v2": screen,
        _ANALYTICS_TABLE_V2: kalman_results,
    }
    if "er_mean" in screen.columns:
        frames["10_screen_mc_summary_v2"] = screen[
            ["isin", "er_mean", "er_sd", "er_p05", "er_p50", "er_p95", "mc_prob_pos"]
        ]
    # Both contrasts land in one table, distinguished by `backend`, so a reader
    # can never mistake a seconds-long screen for an exact ELPD measurement.
    _cmp_parts = [
        df.assign(backend=df["backend"] if "backend" in df.columns else "nuts")
        for df in (result.get("comparison"), result.get("comparison_fast"))
        if isinstance(df, pd.DataFrame) and not df.empty
    ]
    if _cmp_parts:
        frames["09b_comparison_v2"] = pd.concat(_cmp_parts, ignore_index=True)
    if risk_book is not None:
        # Same one-release guard as above, and the reason this frame is no
        # longer the odd one out: the drop used to be applied HERE and to
        # `kalman_results` but never to `risk_book.book`, so `10b_risk_book_v2`
        # shipped `weight` beside `book_weight` and `expected_sharpe` beside
        # `expected_sharpe_ratio` -- byte-identical pairs that
        # `export_unique_columns` cannot see, because it checks duplicate NAMES.
        # Fixed at the source in `compute_cvar_aware_book` (2026-08-24) so BOTH
        # frames carry one column per quantity, with `export_duplicate_content`
        # below watching for a recurrence.
        frames["10b_risk_analytics_v2"] = apply_out_of_support(
            risk_book.analytics.drop(columns=["expected_sharpe"], errors="ignore")
        )
        frames[_RISK_BOOK_KEY] = risk_book.book
    result["export_counts"] = export_analytics(frames, run_cfg, report, run_id=run_id)

    summarise(
        report,
        {
            "run_id": run_id,
            "names": panel.n_isin,
            "T / T_eff": f"{panel.n_time} / {audit['t_eff']:.2f}",
            "median expected upside": f"{screen['expected_upside'].median():.2%}",
            "median implied upside": f"{screen['implied_upside'].median():.2%}",
            "above consensus": f"{(screen['expected_upside'] > screen['implied_upside']).mean():.1%}",
        },
        results_path=run_cfg.results_path,
    )
    _render_figures(result, panel, run_cfg)
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="stages 1-4b only")
    parser.add_argument("--benchmark", action="store_true",
                        help="build the model, time the gradient, project wall clock, stop")
    parser.add_argument("--write", action="store_true", help="write the analytics tables")
    parser.add_argument(
        "--no-figures", action="store_true",
        help=(
            "skip the §4b/§6/§8/§9/§10/§10b panels and their statistics tables. "
            "The analytics export is unaffected either way -- figures are drawn "
            "after it, never before."
        ),
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help=(
            "run the §9b ELPD comparison over these arms, comma-separated "
            f"(known: {','.join(COMPARISON_ARMS)}). Each arm is a full refit plus "
            "a pointwise log_likelihood, so N arms cost ~N runs."
        ),
    )
    parser.add_argument(
        "--compare-fast", type=str, default=None,
        help=(
            "SCREEN these arms with Max-and-Smooth against the run's own fit, "
            "comma-separated. Seconds per arm instead of a production fit per arm, "
            "because it reuses one posterior's covariance and refits only the "
            "latent structure. It RANKS arms; confirm the winner with --compare "
            "before promoting anything. drift_strict is not screenable (it changes "
            "the design matrix)."
        ),
    )
    parser.add_argument("--lookbacks", type=str, default=None,
                        help="comma-separated, e.g. 1y,6m,3m (omit 'now')")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--cores", type=int, default=None)
    args = parser.parse_args()

    model_cfg = KalmanModelConfig()
    if args.lookbacks is not None:
        model_cfg = replace(
            model_cfg,
            lookbacks=tuple(s.strip() for s in args.lookbacks.split(",") if s.strip()),
        )
    run_cfg = KalmanRunConfigV2.from_env()
    overrides = {
        k: v
        for k, v in {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "cores": args.cores,
        }.items()
        if v is not None
    }
    if args.write:
        overrides["write_analytics"] = True
    if args.no_figures:
        overrides["export_figures"] = False
    if args.compare is not None:
        arms = tuple(s.strip() for s in args.compare.split(",") if s.strip())
        unknown = [a for a in arms if a not in COMPARISON_ARMS]
        if unknown:
            parser.error(
                f"unknown comparison arm(s) {unknown}; known: {sorted(COMPARISON_ARMS)}"
            )
        if len(arms) < 2:
            parser.error("--compare needs at least two arms to contrast")
        overrides["enable_model_comparison"] = True
        overrides["comparison_arms"] = arms
    if args.compare_fast is not None:
        fast_arms = tuple(s.strip() for s in args.compare_fast.split(",") if s.strip())
        unknown = [a for a in fast_arms if a not in COMPARISON_ARMS]
        if unknown:
            parser.error(
                f"unknown comparison arm(s) {unknown}; known: {sorted(COMPARISON_ARMS)}"
            )
        if len(fast_arms) < 2:
            parser.error("--compare-fast needs at least two arms to contrast")
        overrides["enable_fast_comparison"] = True
        overrides["fast_comparison_arms"] = fast_arms
    if overrides:
        run_cfg = replace(run_cfg, **overrides)

    out = main(
        model_config=model_cfg,
        run_config=run_cfg,
        dry_run=args.dry_run,
        benchmark=args.benchmark,
    )
    report: GateReport = out["report"]
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())