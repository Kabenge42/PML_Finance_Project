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
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

# PyTensor backend guard MUST precede the first pymc/pytensor import.
from probabilistic_ml_model import _pytensor_env  # noqa: F401  (side effect)

from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
    KALMAN_V2_SCREEN_LATENT,
    KalmanModelConfig,
    KalmanPanelV2,
    partition_covariance_groups,
    build_kalman_pt_model_v2,
    effective_sample_size_of_panel,
    fit_trail_correlation_kernel,
    orthogonalise_family,
    resolve_screen_latent_v2,
)
from probabilistic_ml_model.pymc_models._workflow import (
    MIN_ESS_GATE,
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
    "select_drift_features_v2",
    "prepare_panel",
    "run_panel_diagnostics",
    "run_prior_predictive",
    "sample_posterior",
    "run_posterior_predictive",
    "run_diagnostics",
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
)

#: Columns the ``export_ranking_range`` gate bounds. Both the exported names and
#: the intermediate risk-book names, because v1 guards only the former and the
#: latter still ships an ``expected_sharpe`` of -2,142 to any SQL consumer.
_RANKING_RANGE_COLS: tuple[str, ...] = (
    "expected_sharpe_ratio",
    "reward_to_cvar",
    "expected_sharpe",
    "ret_vol_ratio",
    "starr",
)

#: The canonical v2 analytics table. **v2-suffixed on purpose**: v1 and the live
#: GEIB dashboard keep reading ``analytics.kalman_filtered_price_targets``, so
#: the two models coexist and can be compared on one database. Promoting v2 is a
#: deliberate edit here plus a dashboard deploy — never a side effect of a run.
_ANALYTICS_TABLE_V2 = "kalman_filtered_price_targets_v2"
_RISK_BOOK_KEY = "10b_risk_book_v2"


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
    "ppc_t_std": (
        "Replicated response spread must contain the observed spread. This is the "
        "check that has failed in every review since 2026-08-05."
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

    # ---- gate thresholds ---------------------------------------------------
    gate_t_eff_min: float = 1.30
    gate_kernel_rmse_max: float = 0.08
    gate_r_hat_max: float = 1.01
    gate_ess_min: int = MIN_ESS_GATE
    gate_coverage_target: float = 0.92
    gate_coverage_tol: float = 0.02
    gate_shrinkage_slope_lo: float = 0.80
    gate_shrinkage_slope_hi: float = 1.20
    gate_shrinkage_intercept_max: float = 0.02
    gate_decay_rmse_max: float = 0.10

    # ---- decision layer ----------------------------------------------------
    mc_horizon: int = 4
    mc_rho: float = 0.85
    cvar_alpha: float = 0.05
    weight_cap: float = 0.10
    k_book: int = 25
    #: Baseline long-probability threshold. Scaled by the universe-average
    #: kalman_gain inside the risk book to give ``p_long_cond``, which is what
    #: ``p_upside_pos_cond`` is actually tested against.
    p_long: float = 0.50
    mcap_country_r_max: float = 0.01

    # ---- output ------------------------------------------------------------
    results_dir: Optional[str] = None
    write_analytics: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "KalmanRunConfigV2":
        """Build from the five environment variables the deployment sets."""

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
    # -- v2 additions ---------------------------------------------------------
    "feat_net_eps_drift": "superseded by feat_eps_signal + feat_eps_coverage",
    "feat_last_q_surprise": "52.1% NULL, |beta| 0.0003; superseded by feat_eps_signal",
    "feat_last_y_surprise": "23.2% NULL, |beta| 0.0054; superseded by feat_eps_signal",
    "feat_eps_beat_rate": "45.4% NULL, |beta| 0.0115; superseded by feat_eps_signal",
    "feat_eps_beat_rate_annual": "14.6% NULL, |beta| 0.0013; superseded by feat_eps_signal",
    "feat_mcap_country_sec_r": "near-duplicate of feat_mcap_country_r",
    "feat_mcap_region_r": "near-duplicate of feat_mcap_country_r",
    "feat_mcap_region_sec_r": "near-duplicate of feat_mcap_country_r",
    "feat_mcap_global_sec_r": "near-duplicate of feat_mcap_country_r",
}

#: Prefixes barred wholesale. ``feat_total_return_`` and ``feat_tr_cagr_`` are 18
#: windows of one construct that is mechanically anti-correlated with implied
#: upside (a stock that rallied has less distance to its target) — v1 measured
#: Spearman -0.545 and correctly assigned near-null coefficients, which is a lot
#: of design-matrix width to spend on a known accounting identity. Two
#: representatives are re-admitted explicitly below.
DRIFT_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "days_",
    "feat_total_return_",
    "feat_tr_cagr_",
    "feat_piotroski_f_score_neg",
)

#: Momentum representatives kept despite the prefix ban.
MOMENTUM_REPRESENTATIVES: tuple[str, ...] = (
    "feat_price_chg_pct_3m",
    "feat_one_day_return",
)


def select_drift_features_v2(frame: pd.DataFrame) -> list[str]:
    """Resolve the drift design matrix columns from the frame.

    Per the stage contract the *catalogue* decides which columns exist; this
    function decides which of them belong in the state-transition mean. The two
    are different questions and v1 conflated them once, by flipping a
    ``pymc_role`` to ``'excluded'`` in SQL — which drops the row from
    ``vw_pymc_feature_catalogue`` while the MV still emits the column, and
    ``assert_pymc_catalogue_coverage()`` then raises ``MISSING_FROM_CATALOGUE``.
    Exclusions live here, in Python, always.

    Parameters
    ----------
    frame
        The loaded modelling frame.

    Returns
    -------
    list[str]
        Column names, sorted for stability.
    """
    candidates = [c for c in frame.columns if c.startswith("feat_")]
    kept: list[str] = []
    dropped: dict[str, str] = {}
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
        kept.append(col)
    kept = sorted(kept)
    logger.info(
        "Drift features: %d kept, %d excluded (of %d feat_ columns)",
        len(kept),
        len(dropped),
        len(candidates),
    )
    logger.debug("kept: %s", kept)
    return kept


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

    # Pooled standardisation — see the docstring.
    mu = float(np.nanmean(Y_raw))
    sd = float(np.nanstd(Y_raw))
    if not math.isfinite(sd) or sd < _EPS:
        raise ValueError("Response trail has zero variance; check the MV.")
    Y = (Y_raw - mu) / sd

    names = list(drift_names) if drift_names is not None else select_drift_features_v2(frame)
    X = np.column_stack([_standardise(frame[c].to_numpy()) for c in names])

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

    pt_sd = _col("feat_pt_noise_sigma", 0.0)
    pt_level = np.abs(_col("observed_pt", 1.0))
    # log1p(cv) is NaN for cv < -1 and the dispersion is a ratio of magnitudes,
    # so clamp it non-negative at source rather than guarding inside the model.
    cv = np.where(pt_level > _EPS, pt_sd / np.maximum(pt_level, _EPS), 0.0)
    cv = np.clip(np.nan_to_num(cv, nan=0.0, posinf=0.0), 0.0, 5.0)

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
        vol_level=_standardise(np.log1p(np.clip(_col("feat_vol_level", 0.0), 0.0, None))),
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
    idata = prior_predictive_check(
        model,
        var_names=["target_pct_obs", "state_now", "sigma_level", "sigma_state"],
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
    groups = partition_covariance_groups(
        mask, panel.coverage_profile, model_cfg.coverage_profile_buckets
    )
    first = np.asarray(ppc.posterior_predictive[obs_names[0]])
    n_draw = int(np.prod(first.shape[:-2]))
    rep = np.full((n_draw, panel.n_isin, panel.n_time), np.nan)
    for gi, (rows, cols) in enumerate(groups):
        name = f"target_pct_obs_g{gi}" if len(groups) > 1 else "target_pct_obs"
        if name not in ppc.posterior_predictive:
            logger.warning("replicate %s missing; that group is skipped", name)
            continue
        block = np.asarray(ppc.posterior_predictive[name])
        block = block.reshape(n_draw, len(rows), len(cols))
        rep[np.ix_(np.arange(n_draw), rows, cols)] = block

    out: dict[str, Any] = {}

    # ---- T = std -----------------------------------------------------------
    obs_std = float(np.nanstd(obs[mask]))
    rep_std = np.array([float(np.nanstd(r[mask])) for r in rep])
    lo, hi = np.percentile(rep_std, [3, 97])
    out["t_std"] = {"observed": obs_std, "lo": float(lo), "hi": float(hi)}
    report.add(
        GateResult(
            name="ppc_t_std",
            passed=bool(lo <= obs_std <= hi),
            value=f"obs {obs_std:.3f} vs rep [{lo:.3f}, {hi:.3f}]",
            threshold="observed inside the replicated 94% interval",
            detail=(
                "Replicates over-dispersing while coverage over-shoots is the "
                "signature of an observation scale absorbing structural misfit."
            ),
        )
    )

    # ---- per-time coverage -------------------------------------------------
    ql, qh = np.nanpercentile(rep, [3, 97], axis=0)  # (isin, time)
    inside = (obs >= ql) & (obs <= qh) & mask
    cov = inside.sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
    out["coverage"] = cov
    worst = float(np.max(np.abs(cov - run_cfg.gate_coverage_target)))
    report.add(
        GateResult(
            name="ppc_coverage",
            passed=worst <= run_cfg.gate_coverage_tol,
            value=f"{cov.min():.3f} - {cov.max():.3f} (target {run_cfg.gate_coverage_target})",
            threshold=f"within +/-{run_cfg.gate_coverage_tol}",
            detail=f"per-step: {np.round(cov, 4).tolist()}",
        )
    )

    # ---- correlation decay (new) -------------------------------------------
    try:
        obs_kern = fit_trail_correlation_kernel(obs, panel.time_days)
        rep_kerns = []
        for r in rep[:: max(1, len(rep) // 100)]:
            masked = np.where(mask, r, np.nan)
            try:
                rep_kerns.append(fit_trail_correlation_kernel(masked, panel.time_days))
            except Exception:  # pragma: no cover - individual replicate failure
                continue
        if rep_kerns:
            rho_rep = np.array([k["rho_inf"] for k in rep_kerns])
            r_lo, r_hi = np.percentile(rho_rep, [3, 97])
            passed = bool(r_lo <= obs_kern["rho_inf"] <= r_hi)
            out["decay"] = {
                "observed_rho_inf": obs_kern["rho_inf"],
                "replicated_lo": float(r_lo),
                "replicated_hi": float(r_hi),
            }
            report.add(
                GateResult(
                    name="ppc_decay",
                    passed=passed,
                    value=(
                        f"rho_inf obs {obs_kern['rho_inf']:.3f} vs rep "
                        f"[{r_lo:.3f}, {r_hi:.3f}]"
                    ),
                    threshold="observed inside the replicated 94% interval",
                    detail=(
                        "The model reproduces the panel's time structure, not only "
                        "its marginal spread."
                    ),
                )
            )
    except Exception as exc:  # pragma: no cover
        logger.warning("correlation-decay check unavailable: %s", exc)

    return out


# =========================================================================== #
# §9  Diagnostics                                                             #
# =========================================================================== #


def run_diagnostics(idata: Any, run_cfg: KalmanRunConfigV2, report: GateReport) -> pd.DataFrame:
    """Convergence gates over the global (non-per-ISIN) parameters.

    Gating on *global* parameters is deliberate: per-ISIN vectors have thousands
    of entries whose extreme order statistics are dominated by the tail of a
    large sample, so their max R-hat is not a convergence signal. v1's own §9
    table takes the same subset, which is what makes these numbers comparable
    across versions.
    """
    import arviz as az

    post = idata.posterior
    globals_ = [
        v
        for v in post.data_vars
        if "isin" not in post[v].dims and post[v].size <= post.sizes["chain"] * post.sizes["draw"] * 32
    ]
    summary = az.summary(idata, var_names=globals_, ci_prob=0.89)

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

    max_rhat = float(summary["r_hat"].max())
    worst_rhat = str(summary["r_hat"].idxmax())
    report.add(
        GateResult(
            name="r_hat",
            passed=max_rhat < run_cfg.gate_r_hat_max,
            value=f"{max_rhat:.4f} ({worst_rhat})",
            threshold=f"< {run_cfg.gate_r_hat_max}",
        )
    )

    min_ess = float(summary["ess_bulk"].min())
    worst_ess = str(summary["ess_bulk"].idxmin())
    report.add(
        GateResult(
            name="ess_bulk",
            passed=min_ess >= run_cfg.gate_ess_min,
            value=f"{min_ess:.0f} ({worst_ess})",
            threshold=f">= {run_cfg.gate_ess_min}",
        )
    )
    return summary


# =========================================================================== #
# §10  Screen + decision gates                                                #
# =========================================================================== #


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


def run_screen(
    idata: Any,
    panel: KalmanPanelV2,
    run_cfg: KalmanRunConfigV2,
    report: GateReport,
) -> pd.DataFrame:
    """Build the per-ISIN screen and gate the decision layer.

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

    latent = resolve_screen_latent_v2(
        idata, latent=KALMAN_V2_SCREEN_LATENT, random_seed=run_cfg.random_seed
    )
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
    screen["risk_adj_return"] = _posterior_mean(idata, "risk_adj_return", panel)
    screen["kalman_gain"] = _posterior_mean(idata, "achieve_prob", panel)

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
    mc = np.expm1(np.clip(mc, LOG_UPLIFT_CLIP_LO, LOG_UPLIFT_CLIP_HI))
    mc_summary = summarize_mc_returns(mc, panel.isins)
    screen = screen.merge(
        mc_summary.rename(columns={"prob_pos": "mc_prob_pos"}), on="isin", how="left"
    )
    screen = screen.sort_values("expected_upside", ascending=False).reset_index(drop=True)

    # ---- shrinkage slope gate ---------------------------------------------
    valid = screen[["expected_upside", "implied_upside"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(valid) >= 100:
        slope, intercept = np.polyfit(valid["implied_upside"], valid["expected_upside"], 1)
        above = float((screen["expected_upside"] > screen["implied_upside"]).mean())
        slope_ok = run_cfg.gate_shrinkage_slope_lo <= slope <= run_cfg.gate_shrinkage_slope_hi
        icpt_ok = abs(intercept) <= run_cfg.gate_shrinkage_intercept_max
        report.add(
            GateResult(
                name="shrinkage_slope",
                passed=bool(slope_ok and icpt_ok),
                value=f"slope {slope:.3f}, intercept {intercept:+.4f}, above {above:.1%}",
                threshold=(
                    f"slope in [{run_cfg.gate_shrinkage_slope_lo}, "
                    f"{run_cfg.gate_shrinkage_slope_hi}], |intercept| <= "
                    f"{run_cfg.gate_shrinkage_intercept_max}"
                ),
                detail=(
                    "A non-zero intercept is a universe-wide offset, not a signal. "
                    "Expect ~50% of names above consensus, not 80%."
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
    report.add(
        GateResult(
            name="prob_pos_degenerate",
            passed=pinned <= 0.60,
            value=f"{pinned:.1%} pinned at 1.0",
            threshold="<= 60%",
            blocking=False,
            detail=(
                "prob_pos cannot order a ranking in this state. Downstream views "
                "should read p_upside_pos_cond, which the risk book computes."
            ),
        )
    )
    return screen


# =========================================================================== #
# §10b  Risk book                                                             #
# =========================================================================== #


def run_risk_book(
    idata: Any,
    panel: KalmanPanelV2,
    screen: pd.DataFrame,
    run_cfg: KalmanRunConfigV2,
) -> Any:
    """Size a CVaR-aware long book, reusing :mod:`RiskBookModel` unchanged.

    ``compute_cvar_aware_book`` needs the screen frame to already carry
    ``er_mean`` / ``er_sd`` / ``er_p05`` and ``mc_prob_pos``. Without them
    ``expected_sharpe`` silently becomes NaN, ``tail_risk`` loses its Monte-Carlo
    loss leg, and ``p_upside_pos_cond`` degrades to ``p_upside_pos * kalman_gain``
    — three quiet degradations rather than one loud failure, which is why
    :func:`run_screen` builds those columns first.

    Returns
    -------
    RiskBook or None
        ``None`` when the risk book cannot be computed; the caller keeps going
        with the screen alone rather than losing the whole run.
    """
    import xarray as xr

    from probabilistic_ml_model.pymc_models.RiskBookModel import compute_cvar_aware_book

    try:
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
        book = compute_cvar_aware_book(
            idata,
            eu,
            screen,
            alpha=run_cfg.cvar_alpha,
            cap=run_cfg.weight_cap,
            k_book=run_cfg.k_book,
            p_long=run_cfg.p_long,
            mcap_r_max=run_cfg.mcap_country_r_max,
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
    """
    out = results.copy()
    hi_key = "er_p05" if "er_p05" in out.columns else "er_mean"
    lo_key = "er_p95" if "er_p95" in out.columns else "er_mean"
    if hi_key not in out.columns:
        out["out_of_support"] = False
        return out

    pinned_hi = (
        pd.to_numeric(out[hi_key], errors="coerce") >= UPLIFT_CLIP_HI - 1e-6
    ).fillna(False)
    pinned_lo = (
        pd.to_numeric(out[lo_key], errors="coerce") <= UPLIFT_CLIP_LO + 1e-6
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

    if not report.ok:
        logger.error("Refusing to write analytics: blocking gates failed.")
        return counts

    # ---- write --------------------------------------------------------------
    out_dir = run_cfg.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = None
    schema = os.environ.get("DB_ANALYTICS_SCHEMA", "analytics")
    if run_cfg.write_analytics and os.environ.get("DB_URL"):
        from sqlalchemy import create_engine

        engine = create_engine(os.environ["DB_URL"])

    # Render the DDL before any table write, and unconditionally: it needs no
    # connection, so an offline run still leaves a reviewable schema.
    if _ANALYTICS_TABLE_V2 in frames and not frames[_ANALYTICS_TABLE_V2].empty:
        stamped_canonical = frames[_ANALYTICS_TABLE_V2].copy()
        stamped_canonical["run_id"] = run_id
        stamped_canonical["exported_at"] = stamped
        try:
            write_analytics_ddl_v2(stamped_canonical)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("DDL render failed: %s", exc)

    sql_ok = engine is not None
    for key, df in frames.items():
        if df is None or df.empty:
            continue
        stamped_df = df.copy()
        stamped_df["run_id"] = run_id
        stamped_df["exported_at"] = stamped
        counts[key] = len(stamped_df)

        wrote_table = False
        if sql_ok:
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
    return counts


#: Column documentation for the v2 analytics table. Only the columns whose
#: meaning is *not* obvious from the name, or whose name is actively misleading,
#: are documented — the three at the top are the ones three consecutive reviews
#: had to re-explain because the name says something the column does not do.
_ANALYTICS_COLUMN_COMMENTS_V2: dict[str, str] = {
    "cvar_5pct_kalman": (
        "Conditional mean of the worst 5% of the expected-upside posterior. "
        "This is a RETURN LEVEL, not a loss: it is positive for most names "
        "because most names have an entirely positive upside posterior. Raw "
        "decimal. For a dispersion measure use reward_to_cvar's denominator."
    ),
    "expected_sharpe_ratio": (
        "er_mean / er_sd over Monte-Carlo draws of log price-target uplift. A "
        "t-statistic on the distance from price to the smoothed target, NOT an "
        "investment Sharpe ratio -- the denominator is parameter uncertainty, "
        "not realised return volatility. NULL when out_of_support."
    ),
    "expected_vol_kalman": (
        "Posterior dispersion of expected upside (~2-3%), NOT equity return "
        "volatility. Raw decimal."
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
        "mc_prob_pos * kalman_gain. The probability column that actually orders "
        "names -- prefer it to prob_pos, which is pinned at 1.0 for most of the "
        "universe and cannot rank."
    ),
    "kalman_gain": "Posterior mean achieve_prob = sigmoid(risk_adj_return).",
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
    body.append(f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_isin')
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


def summarise(report: GateReport, extras: dict[str, Any]) -> None:
    """Print the gate report and the run's headline numbers."""
    print(report.render())
    if extras:
        print("\nRun summary")
        print("-" * 40)
        for k, v in extras.items():
            print(f"  {k:<28} {v}")


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
    panel = prepare_panel(frame, model_cfg)
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
        )
        return result

    if not report.ok:
        logger.error("Panel audit failed; not fitting. Fix the grid first.")
        summarise(report, {"names": panel.n_isin, "T_eff": f"{audit['t_eff']:.2f}"})
        return result

    model = build_kalman_pt_model_v2(panel, model_cfg)

    # Measure before committing. A model that cannot finish is caught here, in
    # seconds, rather than 45 minutes into a run that has produced eight draws.
    result["runtime"] = run_runtime_estimate(model, run_cfg, report)
    if benchmark:
        summarise(report, dict(result["runtime"]))
        return result
    if not report.ok:
        logger.error("Runtime gate failed; not sampling.")
        summarise(report, dict(result["runtime"]))
        return result

    result["prior_idata"] = run_prior_predictive(model, panel, run_cfg, report)
    idata = sample_posterior(model, run_cfg)
    result["idata"] = idata
    result["diagnostics"] = run_diagnostics(idata, run_cfg, report)
    result["ppc"] = run_posterior_predictive(
        model, idata, panel, run_cfg, model_cfg, report
    )
    screen = run_screen(idata, panel, run_cfg, report)
    result["screen"] = screen

    risk_book = run_risk_book(idata, panel, screen, run_cfg)
    result["risk_book"] = risk_book

    # The canonical frame: the risk book's analytics if we have it (it is the
    # screen plus the risk columns), otherwise the screen alone.
    kalman_results = (
        risk_book.analytics.copy() if risk_book is not None else screen.copy()
    )
    kalman_results = kalman_results.rename(
        columns={
            "expected_sharpe": "expected_sharpe_ratio",
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
    # carries an expected_sharpe of -2,142.
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
    if risk_book is not None:
        frames["10b_risk_analytics_v2"] = apply_out_of_support(
            risk_book.analytics.rename(columns={"expected_sharpe": "expected_sharpe_ratio"})
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
    )
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="stages 1-4b only")
    parser.add_argument("--benchmark", action="store_true",
                        help="build the model, time the gradient, project wall clock, stop")
    parser.add_argument("--write", action="store_true", help="write the analytics tables")
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
