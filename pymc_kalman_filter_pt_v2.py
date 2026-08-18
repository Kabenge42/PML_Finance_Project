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
    nuts_sampler: Optional[str] = None

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

    n_analysts = np.clip(_col("n_analysts", 1.0), 1.0, None)
    pt_sd = _col("feat_pt_noise_sigma", 0.0)
    pt_level = np.abs(_col("observed_pt", 1.0))
    cv = np.where(pt_level > _EPS, pt_sd / np.maximum(pt_level, _EPS), 0.0)

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
        dispersion_cv=np.clip(cv, 0.0, 5.0),
        precision_weight=np.sqrt(n_analysts),
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
        rep = np.asarray(idata.prior_predictive["target_pct_obs"]).ravel()
        upside = np.expm1(rep * panel.response_std + panel.response_mean)
        obs = np.expm1(
            panel.Y[np.isfinite(panel.Y)] * panel.response_std + panel.response_mean
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


def sample_posterior(model: Any, run_cfg: KalmanRunConfigV2) -> Any:
    """Fit with NUTS via the shared kwargs builder.

    Never re-copies the sampling boilerplate: ``build_sample_kwargs`` supplies
    the compile kwargs, the ``log_likelihood`` policy, the nutpie
    ``idata_kwargs`` strip and the ``chains < 2`` warning.
    """
    import pymc as pm

    kwargs = build_sample_kwargs(
        samples=run_cfg.draws,
        tune=run_cfg.tune,
        chains=run_cfg.chains,
        cores=run_cfg.cores,
        target_accept=run_cfg.target_accept,
        random_seed=run_cfg.random_seed,
        nuts_sampler=run_cfg.nuts_sampler,
        model_name="KalmanPriceTargetV2",
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
    ppc = posterior_predictive_check(
        model, thinned, var_names=["target_pct_obs"], random_seed=run_cfg.random_seed
    )
    rep = np.asarray(ppc.posterior_predictive["target_pct_obs"])
    rep = rep.reshape(-1, *rep.shape[-2:])  # (draw, isin, time)
    mask = panel.observed_mask
    obs = panel.Y

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
    latent = resolve_screen_latent_v2(idata, latent=KALMAN_V2_SCREEN_LATENT)
    draws = np.asarray(latent).reshape(-1, latent.shape[-1])  # (sample, isin)

    # De-standardise back to log-uplift, then to a return.
    log_uplift = draws * panel.response_std + panel.response_mean
    upside = np.expm1(log_uplift)

    frame = panel.frame
    screen = pd.DataFrame(
        {
            "isin": panel.isins,
            "ticker": frame.get("ticker"),
            "name": frame.get("name"),
            "sector": frame.get("sector"),
            "trading_region": frame.get("trading_region"),
            "n_analysts": pd.to_numeric(frame.get("n_analysts"), errors="coerce"),
            "last_price": pd.to_numeric(frame.get("last_price"), errors="coerce"),
            "observed_pt": pd.to_numeric(frame.get("observed_pt"), errors="coerce"),
            "expected_upside": upside.mean(axis=0),
            "expected_upside_sd": upside.std(axis=0),
            "expected_upside_p05": np.percentile(upside, 5, axis=0),
            "expected_upside_p95": np.percentile(upside, 95, axis=0),
            "prob_pos": (upside > 0).mean(axis=0),
        }
    )
    screen["implied_upside"] = screen["observed_pt"] / screen["last_price"] - 1.0
    screen["expected_pt"] = screen["last_price"] * (1.0 + screen["expected_upside"])

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
        grad = cov.groupby("bucket", observed=True)["expected_upside_sd"].mean()
        monotone = bool(grad.is_monotonic_decreasing)
        spread = float(grad.max() / max(grad.min(), _EPS))
        report.add(
            GateResult(
                name="coverage_gradient",
                passed=monotone and spread >= 2.0,
                value=f"{'monotone' if monotone else 'NOT monotone'}, spread {spread:.2f}x",
                threshold="monotone decreasing, spread >= 2x",
                blocking=False,
                detail=f"means by bucket: {grad.round(4).to_dict()}",
            )
        )
    return screen


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
    """Write the curated frames, gating on finiteness and row counts.

    The two export gates encode failures this pipeline has actually shipped:
    a ranking metric of ``-4.28e15`` reaching the dashboard, and a curated table
    existing with zero rows so a naive one-``run_id``-everywhere check passed
    over it.
    """
    stamped = datetime.now(timezone.utc)
    counts: dict[str, int] = {}

    empties = [k for k, v in frames.items() if v is None or v.empty]
    report.add(
        GateResult(
            name="export_rowcount",
            passed=not empties,
            value=f"{len(frames) - len(empties)}/{len(frames)} frames non-empty",
            threshold="every curated frame non-empty",
            detail=f"empty: {empties}" if empties else "",
        )
    )

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

    if not report.ok:
        logger.error("Refusing to write analytics: blocking gates failed.")
        return counts

    out_dir = run_cfg.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in frames.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df["run_id"] = run_id
        df["exported_at"] = stamped
        path = out_dir / f"{key}.csv"
        df.to_csv(path, index=False)
        counts[key] = len(df)
    if run_cfg.write_analytics and os.environ.get("DB_URL"):
        from sqlalchemy import create_engine

        engine = create_engine(os.environ["DB_URL"])
        schema = os.environ.get("DB_ANALYTICS_SCHEMA", "analytics")
        for key, df in frames.items():
            if df is None or df.empty:
                continue
            df = df.copy()
            df["run_id"] = run_id
            df["exported_at"] = stamped
            df.to_sql(key, engine, schema=schema, if_exists="replace", index=False)
            logger.info("wrote %s.%s (%d rows)", schema, key, len(df))
    return counts


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
    result["prior_idata"] = run_prior_predictive(model, panel, run_cfg, report)
    idata = sample_posterior(model, run_cfg)
    result["idata"] = idata
    result["diagnostics"] = run_diagnostics(idata, run_cfg, report)
    result["ppc"] = run_posterior_predictive(model, idata, panel, run_cfg, report)
    screen = run_screen(idata, panel, run_cfg, report)
    result["screen"] = screen

    export_analytics(
        {"10_screen_results_v2": screen, "09_diagnostics_v2": result["diagnostics"].reset_index()},
        run_cfg,
        report,
        run_id=run_id,
    )

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

    out = main(model_config=model_cfg, run_config=run_cfg, dry_run=args.dry_run)
    report: GateReport = out["report"]
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
