"""PyMC Kalman-filter price-target model — script form of ``pymc_kalman_filter_pt.ipynb``.

The cross-sectional spine is the **fused MvGRW panel model** (Model A + Model B,
:func:`probabilistic_ml_model.pymc_models.KalmanFilterModel.build_fused_kalman_pt_model`):

* **Model B spine** — a rank-1 Intrinsic Coregionalization Model (ICM) over the
  ``(isin, time, y_series)`` response tensor: the response series share the latent
  per-ISIN factor ``mu_isin`` through per-series loadings (primary anchored at 1),
  with direct per-series intercept/slope on the collapsed cross-section (a
  zero-anchored Gaussian-random-walk deviation is re-added only for genuine
  ``T > 1`` panels) and a per-series noise diagonal ``sigma_series``.
* **Model A refinement** — the risk-aware ``expected_return`` →
  ``risk_adj_return`` latent (with ``achieve_prob = sigmoid(risk_adj_return)``)
  *is* the GRW baseline ``mu_isin``, and the heteroscedastic scale
  ``sigma_isin = sigma_base * (1 + cv) / sqrt(n)`` replaces the cv-free form.

The Kalman-specific change vs. the price-target panel: the risk adjustment is keyed on
**systematic risk** (``feat_avg_beta``, the NULL-aware mean of ``beta_{1y,2y,5y}``)
rather than analyst conviction, so the latent target reads ``risk_adj_return =
expected_return`` *given systematic risk*; realized volatility enters only through
the ``feat_vol_drift`` observation-noise widener (drift of the vol term structure,
not absolute levels). The drift design matrix also carries a fundamental-quality
level: ``feat_median_piotroski_f_score``, the median of the four per-fiscal-year
Piotroski F-scores emitted by the MV (the per-year components are barred as
collinear with their median). Per-ISIN screening signals are drawn from the posterior
``risk_adj_return`` / ``sigma_isin`` / ``nu`` via the canonical structural-TS Monte-Carlo
helpers (:func:`simulate_lagged_risk_adjusted_returns` / :func:`summarize_mc_returns`).

The workflow has two halves:

* **Fused cross-sectional panel** (sections 4–10): one row per ISIN from
  ``pml.mv_pymc_kalman_pt``, fit with the fused MvGRW + volatility-conditioned model.
* **Single-security / cohort time-series** (sections 11–14): the literal
  ``KalmanFilterPriceTarget`` GRW filter on the embedded ``*_ago`` price-target and
  spot-price history (the MV now emits the raw ``price_{5d,1w,1m,3m,6m,1y,3y,5y,qtd}_ago``
  trail un-prefixed), plus the mingled earnings-cohort consensus and decision-oriented
  summaries.

Schema-aligned with (single source of truth = the ``pml`` schema):

* MV: ``pml.mv_pymc_kalman_pt``
* Catalogue: ``pml.vw_pymc_feature_catalogue WHERE model_target = 'kalman_pt'``
* Coords: ``pml.vw_pml_df_coords``

Usage::

    python pymc_kalman_filter_pt.py
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util as _ilu
import json
import logging
import os
import re
import shutil
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

# ── PyTensor backend guard — MUST precede the first ``pymc``/``pytensor`` import.
# ``pytensor.config`` freezes ``PYTENSOR_FLAGS`` at import time, so an inherited
# ``cxx=<g++ path>`` (e.g. left in the shell by an earlier PML_ENABLE_PYTENSOR_C=1
# session) re-arms the fragile MinGW C backend unless stripped first. The package
# ``__init__`` runs the same guard, but the third-party ``import pymc`` below would
# otherwise execute before it. Opt back in with ``PML_ENABLE_PYTENSOR_C=1``.
import probabilistic_ml_model._pytensor_env  # noqa: F401  isort: skip

# ArviZ 1.0 split-package imports: arviz-plots owns ``style`` + plotting, arviz-stats
# owns ``summary`` / ``rhat`` / ``ess``. Address each submodule directly.
import arviz_plots as azp
import arviz_stats as azs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors
matplotlib.use("TkAgg")  # use a native window instead of PyCharm's plot tool
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr
from arviz_base import rcParams as _az_rcparams
from arviz_plots import visuals as azv  # low-level primitives for custom composition
from cycler import cycler as _cycler
from sqlalchemy import create_engine, text

# Plotly powers the interactive §2.4 EDA panels. It is an approved visualization
# dependency (see CLAUDE.md); imported defensively so the EDA degrades to its
# matplotlib / arviz_plots panels when the optional dep is unavailable.
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover - optional dependency
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]
    _HAS_PLOTLY = False

from probabilistic_ml_model._pymc_arviz_compat import extend_datatree
from probabilistic_ml_model.pymc_models._feature_alignment import (
    load_feature_metadata_from_db,
    stamp_feature_provenance,
)
from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
    AGO_SUFFIX_PATTERN,
    FISCAL_HORIZONS,
    KALMAN_CONSENSUS_SIGMA_FEATURE,
    KALMAN_DRIFT_EXCLUDED_FEATURES,
    KALMAN_RANGE_WIDENER_FEATURE,
    KALMAN_TILT_FEATURE_ORDER,
    KALMAN_TIME_COVARIATE_PREFIX,
    KALMAN_VOL_DRIFT_FEATURE,
    KalmanFilterPriceTarget,
    KalmanPanelInputs,
    build_fused_kalman_pt_model,
)
from probabilistic_ml_model.pymc_models._price_target_mc import (
    simulate_lagged_risk_adjusted_returns,
    summarize_mc_returns,
)
from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs
from probabilistic_ml_model.pymc_models._workflow import (
    attach_log_likelihood,
    build_sample_kwargs,
    log_sample_diagnostics,
    posterior_dataset as _posterior_dataset,
)
from probabilistic_ml_model.pymc_models.RiskBookModel import (
    RiskBook,
    compute_cvar_aware_book as _compute_cvar_aware_book,
)

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# --- Data-source queries (single source of truth = the pml schema) -----------
# The cross-sectional universe query is built by :func:`kalman_df_query` from
# the :class:`KalmanRunConfig` date/threshold fields (defined below).

# Per-model feature catalogue (pymc_role / feature_role / alias) for kalman_pt.
FEATURE_CATALOGUE_QUERY = """
                          SELECT *
                          FROM pml.vw_pymc_feature_catalogue
                          WHERE model_target = 'kalman_pt'
                          ORDER BY pymc_role, feature_role, feature_alias \
                          """

# Canonical schema of pml.mv_pymc_kalman_pt: per-trail drift feature for every
# price_* / price_target_* family plus the noise wideners. Resilience fallback for
# MV columns absent from the catalogue snapshot. Deliberately a literal (NOT
# derived from the DB at import time): ``pml.vw_pymc_feature_catalogue`` remains
# the SSOT, and this list only keeps the workflow alive offline — keep it in
# sync when the MV DDL changes.
KNOWN_FEATURES = ['feat_pt_drift', 'feat_price_drift',
                  'feat_coverage_drift', 'feat_pt_noise_drift',
                  'feat_pt_noise_sigma', 'feat_pt_range_norm',
                  # Drift of the realized-vol term structure (1m -> 1y) and its
                  # valid-pair counter — replaces the absolute feat_vol_* levels.
                  'feat_vol_drift', 'feat_vol_drift_n',
                  'feat_avg_beta',
                  # Analyst rating mix / conviction and 1y achievement feats
                  # (copied from mv_pymc_price_target).
                  'feat_holds', 'feat_buys', 'feat_sells', 'feat_no_opinion',
                  'feat_analyst_bullish_pct', 'feat_analyst_bearish_pct',
                  'feat_analyst_neutral_pct', 'feat_analyst_conviction',
                  'feat_analyst_rating', 'feat_pt_achievement_1y',
                  'feat_pt_accuracy_1y', 'feat_pt_range_hit_rate',
                  # Short-term momentum: last day's price change (one_day_pct).
                  'feat_one_day_return','feat_price_chg_pct_3m',
                  'feat_total_return_ytd', 'feat_total_return_5y', 'feat_total_return_10y',
                  'feat_tr_cagr_3y', 'feat_tr_cagr_10y', 'feat_tr_cagr_5y', 'feat_tr_cagr_1y',
                  # Short/medium-horizon realised total returns (rolling windows,
                  # period-to-date, and calendar-year buckets) emitted by
                  # mv_pymc_kalman_pt as momentum / drift predictors.
                  'feat_total_return_1d', 'feat_total_return_5d', 'feat_total_return_1w',
                  'feat_total_return_1m', 'feat_total_return_3m', 'feat_total_return_6m',
                  'feat_total_return_1y', 'feat_total_return_3y',
                  'feat_total_return_mtd', 'feat_total_return_qtd',
                  'feat_total_return_2025', 'feat_total_return_2024', 'feat_total_return_2023',
                  'feat_total_return_2022', 'feat_total_return_2021',
                  # Drift of the market_cap / enterprise_value ratio trail (equity share
                  # of EV) across the fiscal-year lags — a state-transition drift predictor.
                  'feat_mv_ev_drift',
                  # Cross-cutting market-cap / EV size & trend feats (added to every
                  # mv_pymc_* view; provenance-only for the fused panel — see §note below).
                  'feat_mcap_trend_1y', 'feat_mcap_vs_3yavg', 'feat_ev_vs_3yavg',
                  # Piotroski F-score fundamental-quality trail: four per-fiscal-year
                  # 9-signal composites plus their median (the drift predictor; the
                  # per-year components are barred as collinear — see
                  # KALMAN_PIOTROSKI_COMPONENT_FEATURES).
                  'feat_piotroski_f_score_fy', 'feat_piotroski_f_score_neg1fy',
                  'feat_piotroski_f_score_neg2fy', 'feat_piotroski_f_score_neg3fy',
                  'feat_median_piotroski_f_score']

# Last-resort drift-feature literal for ``map_state_space_features`` when BOTH
# the passed catalogue frame and the direct ``vw_pymc_feature_catalogue`` query
# are unavailable (e.g. offline unit tests). Mirrors the catalogue-driven
# selection: the ``kalman_pt`` mutable_predictor aliases surviving
# ``KalmanFilterPriceTarget.select_drift_features`` as of 2026-07-30.
_DRIFT_FEATURE_FALLBACK: tuple[str, ...] = (
    # Analyst-target / price / coverage drift trails.
    'feat_pt_drift', 'feat_price_drift','feat_coverage_drift',
    'feat_pt_noise_drift',
    # Short/mid-horizon momentum.
    'feat_one_day_return', 'feat_price_chg_pct_3m',
    # Analyst-sentiment composition (neutral leg dropped as collinear).
    'feat_analyst_bullish_pct', 'feat_analyst_bearish_pct',
    'feat_analyst_conviction', 'feat_analyst_rating',
    # 1y price-target credibility.
    'feat_pt_achievement_1y', 'feat_pt_accuracy_1y', 'feat_pt_range_hit_rate',
    # Market-cap / EV size & trend.
    'feat_mcap_trend_1y', 'feat_mcap_vs_3yavg', 'feat_ev_vs_3yavg',
    'feat_mv_ev_drift',
    # Fundamental quality: median of the four per-fiscal-year Piotroski
    # F-scores (the components are barred as collinear with their median).
    'feat_median_piotroski_f_score',
)

# Hierarchical classification coords (categorical group effects), distinct from the
# fiscal-calendar DATE anchors. Both carry pymc_role='coord', but the date anchors
# define the single-security time axis (sections 11–13) and must NOT be treated as
# categorical effects in the cross-sectional model.
CLASSIFICATION_COORDS_ALL = (
    'region', 'country', 'trading_region', 'trading_country', 'exchange',
    'sector', 'industry', 'style_class', 'size_class', 'unit',
)
# Fiscal-calendar DATE anchors and their day-count covariates, derived from the
# FISCAL_HORIZONS SSOT in KalmanFilterModel.py. ``last_updated`` is the MV
# refresh timestamp — a calendar column on the frame but not a fiscal horizon.
FISCAL_CALENDAR_COLS_ALL = tuple(fh.date_col for fh in FISCAL_HORIZONS) + ('last_updated',)
DAY_COUNT_COLS_ALL = tuple(fh.day_count_col for fh in FISCAL_HORIZONS)

# *_ago price-target history column pattern shared by sections 11–13. The
# suffix alternation comes from the KalmanFilterModel SSOT; the pattern is kept
# free of Python named groups so it stays valid as a PostgreSQL POSIX regex
# (``column_name ~ :pat`` in :func:`fetch_history_columns`).
HIST_COL_PATTERN = (r"^(price_target(_high|_low|_median)?|price)"
                    r"_(" + AGO_SUFFIX_PATTERN + r")_ago$")

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 94% equal-tailed credible-interval bounds shared by every HDI-style band in
# the file (screen tables, forecast tables, sigma_obs paths, group forests).
_HDI_LO, _HDI_HI = 0.04, 0.96

# Approximate calendar day counts for the fixed *_ago lookback suffixes; backs
# the fused panel's t_scaled axis when ``history_lookbacks`` is active. The
# period-to-date suffixes (mtd/qtd/ytd) are date-dependent and deliberately
# absent — they cannot anchor a shared panel time axis.
_AGO_APPROX_DAYS: dict[str, int] = {
    '1d': 1, '5d': 5, '1w': 7, '1m': 30, '3m': 91, '6m': 182,
    '1y': 365, '3y': 1096, '5y': 1826,
}


@dataclass(frozen=True)
class KalmanRunConfig:
    """Workflow-level configuration for the Kalman price-target pipeline.

    Centralises every previously-hardcoded sampling / screening / risk-book /
    universe-query knob (CLAUDE.md "Configuration as Dataclass" pattern).
    Function signatures keep their own literal defaults for direct calls;
    :func:`main` resolves a config once (:meth:`from_env` by default) and
    threads it through the workflow, so environment overrides apply end-to-end.

    Attributes
    ----------
    random_seed
        Workflow RNG seed (env: ``RANDOM_SEED``).
    draws, tune, chains, cores, target_accept
        Fused-panel NUTS budget (:func:`sample_posterior`). ``cores=1`` is the
        kernel-safe default; the CLI path may raise it.
    prior_draws
        Prior-predictive draw count (:func:`run_prior_predictive`).
    mc_horizon, mc_rho
        Structural-TS Monte-Carlo horizon (fiscal quarters) and AR damping
        (:func:`summarize_panel_screen`).
    panel_lookbacks
        ``*_ago`` suffixes building the genuine ``(isin, time)`` fused-panel
        response (``T = len + 1``). Defaults to ``('6m', '3m', '1m')`` — the
        validated **T=4** panel. Pass ``replace(cfg, panel_lookbacks=())`` to
        collapse back to the T=1 cross-section (faster, no time axis). See
        the field comment for the validation record and
        :func:`prepare_kalman_panel_inputs`.
    cvar_alpha, weight_cap, k_book, p_long
        CVaR-aware book parameters (:func:`compute_cvar_aware_book`).
    mcap_country_r_max
        Market-cap pre-selection gate for the long book: candidates must have
        ``mcap_country_r < mcap_country_r_max``, where ``mcap_country_r`` is
        the MV-derived ``feat_mcap_country_r = (100 - market_cap_country_r) /
        100`` ratio (smaller = larger cap). The 0.02 default keeps the top 2%
        of each country by market cap — the ratio-scale mirror of
        ``min_mcap_country_rank`` (98). Names with a missing rank are
        excluded (strict, matching the SQL ``> 98`` candidate filters).
    min_next_earnings, min_report_date
        ISO-date universe filters in :func:`kalman_df_query` (previously
        hardcoded ``'2026-01-01'`` / ``'2025-01-01'`` — roll these forward with
        the modelling horizon).
    min_mcap_country_rank
        ``market_cap_country_r`` floor for the §11–§13 candidate cohorts.
    candidate_limit
        ``LIMIT`` for the §11 single-ISIN candidate pull.
    earnings_window_days
        Half-width (days) of the §12/§13 recent-earnings window around today.
    results_dir, export_draws
        Artifact export root / raw-draw export toggle (env:
        ``KALMAN_PT_RESULTS_DIR`` / ``KALMAN_PT_EXPORT_DRAWS``).
    fig_width_px
        Target figure width override (env: ``PML_FIG_WIDTH_PX``); ``None``
        falls back to :data:`_FIG_WIDTH_DEFAULT_PX`.
    log_level
        Root logging level for :func:`main` (env: ``LOG_LEVEL``).
    """

    # Sampling
    random_seed: int = 42
    # 3000, raised from 1000 on the 2026-08-10 T=4 validation. Every other gate
    # passed there — 0 divergences, per-time PPC coverage within 0.8 pp of
    # nominal, the per-ISIN effect alive — but the drift coefficients ``beta``
    # came in at R-hat 1.026 / bulk-ESS 140 against the 1.01 / 400 gates. With no
    # divergences that is autocorrelation, not bad geometry: the 21 drift features
    # are mutually correlated (the PT / price / momentum trails move together) and
    # now share explanatory space with the per-ISIN intercept, so the chain
    # explores that block slowly. Bulk-ESS scales ~linearly in draws, so ~3x is
    # what clears 140 -> 400. Costs proportionally more wall-clock: budget ~45 min
    # for the T=4 panel rather than the ~16 min the 1000-draw build took.
    draws: int = 3000
    tune: int = 1000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.9
    prior_draws: int = 1000
    # Screen / Monte-Carlo
    mc_horizon: int = 4
    mc_rho: float = 0.85
    # Risk book
    cvar_alpha: float = 0.05
    weight_cap: float = 0.15
    k_book: int = 25
    p_long: float = 0.50
    mcap_country_r_max: float = 0.02
    # Fused-panel time axis: *_ago lookbacks building the genuine (isin, time)
    # log-uplift panel (oldest -> newest is resolved automatically; the current
    # snapshot is always the final step, so T = len(panel_lookbacks) + 1).
    # T=4 is the DEFAULT, validated since the per-time direct-intercept
    # reparameterisation of build_fused_kalman_pt_model: the 2026-08-01
    # T=4 run (tune=1000, target_accept=0.9, nutpie, ~1.5% history cells
    # snapshot-filled) passed with 0 divergences, worst r_hat 1.00 and worst
    # bulk-ESS ~1.6k, in 15.7 min end-to-end. (Earlier failures — 315/190
    # divergences on 2026-07-31/08-01 — were the since-removed aliased
    # level/slope + GRW-innovation block; see build_fused_kalman_pt_model.)
    # Use replace(cfg, panel_lookbacks=()) for the collapsed T=1 cross-section.
    panel_lookbacks: tuple[str, ...] = ('6m', '3m', '1m')
    # AR(1) time-varying state on top of the per-ISIN random intercept.
    # 0.0 = OFF (the default). The per-ISIN intercept restored on T>1 panels
    # carries the per-name signal; the AR deviation on top of it bought +0.013
    # recovery correlation for min ESS 14 vs 69 and drifting sigma_state/rho
    # between runs, so it is disabled pending a longer panel. Set to e.g. 0.1
    # to enable; it is the treatment arm of the §9b model comparison.
    state_innovation_scale: float = 0.0
    # Model comparison (§9b) is OPT-IN: it needs a pointwise log_likelihood group,
    # which costs chains x draws x n_obs floats (~820 MB for the full T=4 panel at
    # 4x1000 draws over 6.4k ISINs) and is computed twice — once per arm.
    # ``comparison_max_isins`` subsamples the ISIN axis to keep that tractable.
    enable_model_comparison: bool = False
    comparison_max_isins: int = 800
    # Second response series (D > 1), OFF by default. Populates the rank-1 ICM
    # (``mu_isin_loading`` / ``sigma_series``), which is inert while D == 1. The
    # D > 1 path is what produced the historic R-hat 4.45 / min-ESS 4.3 freeze, so
    # it must be validated as a change of its own -- see
    # KALMAN_PANEL_RESPONSE_EXTRA for the supported values.
    panel_response_extra: tuple[str, ...] = ()
    # Universe query
    min_next_earnings: str = '2026-01-01'
    min_report_date: str = '2025-01-01'
    min_mcap_country_rank: float = 98.0
    candidate_limit: int = 50
    earnings_window_days: int = 10
    # Plumbing
    results_dir: Optional[str] = None
    export_draws: bool = False
    fig_width_px: Optional[int] = None
    log_level: str = 'INFO'

    def __post_init__(self) -> None:
        for field_name in ('min_next_earnings', 'min_report_date'):
            value = getattr(self, field_name)
            if not _ISO_DATE_RE.match(value):
                raise ValueError(
                    f"{field_name} must be an ISO date (YYYY-MM-DD), got {value!r}."
                )

    @classmethod
    def from_env(cls) -> 'KalmanRunConfig':
        """Build a config from the environment (``environment_variables.txt`` fallback).

        Only the knobs with an established env contract are read
        (``RANDOM_SEED``, ``KALMAN_PT_RESULTS_DIR``, ``KALMAN_PT_EXPORT_DRAWS``,
        ``PML_FIG_WIDTH_PX``, ``LOG_LEVEL``); everything else keeps its
        dataclass default and is overridden programmatically via
        :func:`dataclasses.replace`.
        """
        def _int_or(key: str, default: Optional[int]) -> Optional[int]:
            raw = _resolve_env_setting(key)
            if raw is None or not raw.strip():
                return default
            try:
                return int(float(raw))
            except ValueError:
                logger.warning('%s=%r is not numeric; using %r.', key, raw, default)
                return default

        return cls(
            random_seed=_int_or('RANDOM_SEED', cls.random_seed) or cls.random_seed,
            results_dir=_resolve_env_setting('KALMAN_PT_RESULTS_DIR'),
            export_draws=(_resolve_env_setting('KALMAN_PT_EXPORT_DRAWS', default='0')
                          or '0').strip() == '1',
            fig_width_px=_int_or('PML_FIG_WIDTH_PX', None),
            log_level=_resolve_env_setting('LOG_LEVEL', default='INFO') or 'INFO',
        )


_run_config: Optional[KalmanRunConfig] = None


def get_run_config() -> KalmanRunConfig:
    """Lazy module-level :class:`KalmanRunConfig` singleton (env-resolved once)."""
    global _run_config
    if _run_config is None:
        _run_config = KalmanRunConfig.from_env()
    return _run_config


def set_run_config(config: Optional[KalmanRunConfig]) -> None:
    """Install (or with ``None`` reset) the module-level run config."""
    global _run_config
    _run_config = config


def kalman_df_query(config: Optional[KalmanRunConfig] = None) -> str:
    """Cross-sectional universe query, parameterised on the run config.

    One row per ISIN with a usable analyst target, scoped to names whose next
    earnings land in the modelling horizon. Date literals are validated ISO
    dates (see :meth:`KalmanRunConfig.__post_init__`), so the interpolation is
    injection-safe.
    """
    cfg = config if config is not None else get_run_config()
    return f"""
           SELECT *
           FROM pml.mv_pymc_kalman_pt mpkp
           WHERE observed_pt IS NOT NULL
             AND next_earnings >= '{cfg.min_next_earnings}'
             AND income_statement_report_date >= '{cfg.min_report_date}'
           """


# ``display`` exists only in IPython; fall back to ``print`` for plain-script runs.
# The module-level :func:`display` wrapper below shadows the IPython callable so every
# displayed DataFrame / Series is also captured by the artifact exporter (section 1c)
# when :func:`enable_artifact_export` is active.
try:  # pragma: no cover - depends on runtime
    from IPython.display import display as _display_impl
except ImportError:  # pragma: no cover
    def _display_impl(obj: object) -> None:
        print(obj)


def display(obj: object, *, label: Optional[str] = None) -> None:
    """Show ``obj`` in the front-end, snapshotting tables when artifact export is on.

    DataFrames / Series routed through the notebook ``display`` are transient
    render-time artifacts; when artifact export is enabled they are additionally
    written as CSV snapshots named after the active :func:`export_section`. Figures
    (including the ``_safe_show`` fallback path) pass straight through — only
    pandas objects are captured here, so a figure can never be exported twice.

    Parameters
    ----------
    obj
        Object to display.
    label
        Optional filename slug for the CSV snapshot; auto-numbered when omitted.
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        try:
            if get_export_state().enabled:
                _export_dataframe(obj, _next_stem(label or 'table'))
        except Exception as exc:  # pragma: no cover - export is best-effort
            logger.debug("Table export skipped: %r", exc)
    _display_impl(obj)


# =============================================================================
# 1. Plotting setup + reusable helpers
# =============================================================================
# --- Dark theme (single source of truth) --------------------------------------
# One Plotly template governs every figure in this module. It is applied in
# exactly one place at render time — :func:`_apply_dark_template`, called from
# the :func:`_safe_show` funnel — so a displayed figure and its exported PNG can
# never diverge. arviz-plots 1.x renamed the dark style, hence the candidate
# list; :func:`setup_plotting` warns when neither resolves instead of silently
# leaving PlotCollection figures on the light default.
_PLOTLY_TEMPLATE = 'arviz-tumma'
_ARVIZ_STYLE_CANDIDATES: tuple[str, ...] = ('arviz-tumma', 'arviz-variat')
C_PANEL_BG = '#1e1e1e'     # figure / paper background (matplotlib + marker edges)
C_AXES_BG = '#2a2a2a'      # axes (plot area) background

# Continuous colour ramps. One sequential and one diverging ramp for the whole
# module: panels had drifted across Viridis / Magma / flare (sequential) and vlag
# (diverging), so two views of the same quantity read as different measurements.
# ``*_MPL`` are the matplotlib / seaborn spellings of the same two ramps.
CS_SEQ = 'Viridis'         # magnitude-only quantities (beta, STARR, kalman gain)
CS_DIV = 'RdBu'            # signed quantities centred on zero
CS_SEQ_MPL = 'viridis'
CS_DIV_MPL = 'vlag'


def setup_plotting() -> None:
    """Pin arviz-plots to the **Plotly** backend and install the dark notebook theme.

    All ArviZ figures in this module render through the interactive Plotly backend.
    The authoritative switch is ``arviz_base.rcParams['plot.backend']`` — every
    ``azp.plot_*`` / :class:`arviz_plots.PlotCollection` call that does not pass an
    explicit ``backend`` reads that default. (Assigning ``azp.backend`` has no effect:
    ``arviz_plots.backend`` is a subpackage, so the attribute assignment merely shadows
    the module and never reaches the plotting layer.) The dark ``arviz-tumma`` Plotly
    template is registered as the default so composed collections inherit it.

    Notes
    -----
    The residual hand-built Matplotlib / Seaborn panels (e.g.
    :func:`plot_kalman_forecast`, the §2.4 observation-noise density panels and the
    per-sector error-bar comparisons) are *not* ArviZ figures and keep the Matplotlib
    dark theme installed below. seaborn installs its colour cycle as RGB tuples; those
    are re-expressed as hex strings so any Matplotlib artist that reshapes the active
    cycle behaves under the theme.
    """
    warnings.filterwarnings('ignore', category=FutureWarning)

    # ArviZ -> Plotly. ``rcParams['plot.backend']`` is the default read by every
    # azp.plot_* / PlotCollection call that omits an explicit ``backend`` argument.
    _az_rcparams['plot.backend'] = 'plotly'
    try:
        import plotly.io as _pio
        _pio.templates.default = _PLOTLY_TEMPLATE  # dark ArviZ Plotly template
    except Exception:  # pragma: no cover - plotly optional / renderer-less env
        pass

    # Matplotlib / Seaborn dark theme for the residual hand-built (non-ArviZ) panels.
    plt.style.use('dark_background')
    # A silent failure here used to leave arviz_plots on its light default while
    # everything else went dark -- warn instead, and accept the 1.x rename.
    for _style in _ARVIZ_STYLE_CANDIDATES:
        try:
            azp.style.use(_style)
            break
        except (OSError, ValueError, AttributeError):
            continue
    else:
        logger.warning(
            "No arviz-plots dark style available (tried %s); PlotCollection "
            "figures fall back to the library default and are re-themed at "
            "display time by _apply_dark_template.",
            ', '.join(_ARVIZ_STYLE_CANDIDATES))
    sns.set_theme(style='darkgrid', context='notebook',
                  rc={
                      'figure.facecolor': C_PANEL_BG,
                      'axes.facecolor': C_AXES_BG,
                      'savefig.facecolor': C_PANEL_BG,
                      'axes.edgecolor': '#cccccc',
                      'axes.labelcolor': '#e6e6e6',
                      'xtick.color': '#e6e6e6',
                      'ytick.color': '#e6e6e6',
                      'text.color': '#e6e6e6',
                      'grid.color': '#555555',
                  })

    _cycle_cols = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if _cycle_cols and not all(isinstance(_c, str) for _c in _cycle_cols):
        plt.rcParams['axes.prop_cycle'] = _cycler(
            color=[_mcolors.to_hex(_c) for _c in _cycle_cols]
        )
    plt.rcParams['figure.dpi'] = 110


# --- Figure sizing (single source of truth for every panel) ------------------
# Target display width in pixels. Override with ``PML_FIG_WIDTH_PX`` to match
# the editor / notebook pane. Every ArviZ (arviz_plots), Plotly and matplotlib
# panel derives its size from this one knob; Plotly panels additionally
# autosize to the rendered container width (see :func:`_autosize_plotly`), so
# the pixel width is only the static-export / fallback size.
_FIG_WIDTH_DEFAULT_PX = 1150
_FIG_WIDTH_MIN_PX, _FIG_WIDTH_MAX_PX = 700, 2200

# --- Semantic series palette (single source of truth for every figure) --------
# An Okabe-Ito-ish convention grown organically across the sections; declared
# here so every panel keys colours on ROLE, not on a per-section hex. Reference
# geometry (0-lines, y=x guides, now-boundaries) is always C_REF.
C_POSTERIOR = '#56b4e9'    # posterior / model / expected series
C_OBSERVED = '#ffb000'     # observed / raw consensus series
C_FORECAST = '#cc79a7'     # forward-looking forecast bands / lines
C_REF = '#bbbbbb'          # reference lines (zero, y=x, anchors)
C_HIGHLIGHT = '#e69f00'    # emphasised subset (held book, key feature)
C_VOL = '#ff7f0e'          # volatility paths (sigma_obs, SV panels)
C_DRAWS = '#4daf4a'        # posterior-draw spaghetti / clouds
C_ACCENT = '#2ca02c'       # secondary accent (scale panels, tertiary KDE)
C_MUTED = '#7f7f7f'        # de-emphasised context points

# Standard panel heights (px). Odd one-off heights remain inline where a panel
# is genuinely bespoke; new figures should pick from this ladder.
H_SHORT, H_STD, H_FORECAST, H_TALL, H_PANEL, H_GRID = 380, 440, 540, 560, 760, 900


def _display_width_px() -> int:
    """Resolve the target figure width (px) from the run config (``PML_FIG_WIDTH_PX``).

    Falls back to :data:`_FIG_WIDTH_DEFAULT_PX` and clamps to
    ``[_FIG_WIDTH_MIN_PX, _FIG_WIDTH_MAX_PX]`` so a typo cannot produce an
    unreadable sliver or a multi-screen banner.
    """
    width = get_run_config().fig_width_px
    if width is None:
        width = _FIG_WIDTH_DEFAULT_PX
    return int(np.clip(width, _FIG_WIDTH_MIN_PX, _FIG_WIDTH_MAX_PX))


def _azp_figure_kwargs(height_px: float, *, width_frac: float = 1.0) -> dict[str, Any]:
    """``figure_kwargs`` for ``azp.plot_*`` with sizes declared in **dots**.

    Passing ``figsize_units='dots'`` silences the arviz-plots Plotly backend's
    ``"Assuming dpi=100"`` UserWarning (its default interprets ``figsize`` in
    inches) and pins an explicit pixel geometry. ``width_frac`` shrinks narrow
    single-axis panels (KDEs, PIT ECDFs) below the full display width.
    """
    width = int(round(_display_width_px() * float(np.clip(width_frac, 0.2, 1.0))))
    return {'figsize': (width, int(round(height_px))), 'figsize_units': 'dots'}


def _forest_height_px(n_rows: int, *, per_row: int = 26, base: int = 190,
                      min_px: int = 320, max_px: int = 1650) -> int:
    """Height (px) for row-oriented panels (forest / ridge) from their row count.

    Guarantees a readable per-row band so long forests scroll instead of
    overlapping their labels, while degenerate 1–2 row panels keep a sane
    minimum aspect.
    """
    return int(np.clip(max(int(n_rows), 1) * per_row + base, min_px, max_px))


# --- Diagnostic facet-grid sizing (SSOT for the §9 run_diagnostics panels) ----
# arviz-plots fans a vector variable (e.g. the ~40-element ``beta`` drift-slope
# vector) into one facet per element and wraps the facets into a grid, so panel
# heights must budget for the FACET count, not the variable count. Every §9
# grid (trace / rank-dist / prior-posterior / ESS evolution) derives its
# geometry from these two knobs so the diagnostics share one dynamic rule.
_DIAG_FACET_COLS = 4  # facet-wrap width of the wrapped diagnostic grids
_DIAG_ROW_PX = 260    # vertical budget per grid row (panel + axes + title)


def _n_facets(posterior, var_names: Sequence[str]) -> int:
    """Total non-sample elements (= plot facets) across ``var_names``.

    Scalars count 1; a vector variable contributes one facet per element of
    its non-sample dims (e.g. ``beta`` over ``drift_feature``). Names absent
    from ``posterior`` are skipped.
    """
    total = 0
    for v in var_names:
        if v not in posterior.data_vars:
            continue
        da = posterior[v]
        total += int(np.prod([da.sizes[d] for d in da.dims
                              if d not in ('chain', 'draw')], initial=1))
    return total


def _facet_grid_height_px(n_facets: int, *, cols: int = _DIAG_FACET_COLS,
                          per_row: int = _DIAG_ROW_PX, base: int = 170,
                          min_px: int = 420, max_px: int = 6200) -> int:
    """Height (px) for a wrapped facet grid from its facet count.

    ``cols`` is the facet-wrap width. The raised ``max_px`` (vs
    :func:`_forest_height_px`) lets long grids scroll in the notebook instead
    of compressing dozens of facets into a fixed-height banner.
    """
    n_rows = int(np.ceil(max(int(n_facets), 1) / max(int(cols), 1)))
    return int(np.clip(n_rows * per_row + base, min_px, max_px))


def _diag_figure_kwargs(posterior, var_names: Sequence[str],
                        *, cols: int = _DIAG_FACET_COLS,
                        per_row: int = _DIAG_ROW_PX) -> dict[str, Any]:
    """Standardised ``figure_kwargs`` for the §9 diagnostic facet grids."""
    return _azp_figure_kwargs(
        _facet_grid_height_px(_n_facets(posterior, var_names),
                              cols=cols, per_row=per_row))


_RANK_DIST_ROW_PX = 220  # vertical budget per compact rank-dist row (dist | rank)


def _rank_dist_figure_kwargs(var_names: Sequence[str]) -> dict[str, Any]:
    """``figure_kwargs`` for the §9.3b compact ``plot_rank_dist`` panels.

    ``azp.plot_rank_dist(compact=True)`` pins exactly one grid row per
    *variable* (``rows=['__variable__']``, dist | rank columns) and overlays a
    vector variable's elements within that single row — unlike ``plot_trace``,
    which fans elements into wrapped facets. Height must therefore budget the
    variable count, not the element count :func:`_n_facets` reports (element
    sizing inflated a one-row vector panel to ~2800 px).
    """
    return _azp_figure_kwargs(
        _facet_grid_height_px(len(var_names), cols=1,
                              per_row=_RANK_DIST_ROW_PX, base=150, min_px=360))


def _polish_facet_axes(pc) -> None:
    """Re-enable the tick labels Plotly culls on small facets; pad the margins.

    Best-effort cosmetic pass shared by the §9 grids so every facet keeps
    visible x/y axes regardless of how many panels the grid wraps.
    """
    fig = _plotly_figure_of(pc)
    if fig is None:
        return
    fig.update_xaxes(showticklabels=True, tickfont=dict(size=9))
    fig.update_yaxes(showticklabels=True, tickfont=dict(size=9))
    fig.update_layout(margin=dict(t=70, b=50))


def _mpl_figsize(height_frac: float = 0.45, *, width_frac: float = 1.0) -> tuple[float, float]:
    """Matplotlib ``figsize`` (inches) derived from the shared pixel width.

    ``height_frac`` is the height:width aspect ratio; the pixel width comes from
    :func:`_display_width_px` divided by the active ``figure.dpi`` so mpl panels
    track the same editor-width knob as the Plotly ones.
    """
    dpi = float(plt.rcParams.get('figure.dpi', 100.0)) or 100.0
    w_in = _display_width_px() * float(np.clip(width_frac, 0.2, 1.0)) / dpi
    return w_in, max(2.0, w_in * float(height_frac))


def _plotly_figure_of(obj) -> Optional[object]:
    """Best-effort: the underlying Plotly figure of ``obj``.

    Accepts a raw :class:`plotly.graph_objects.Figure` (returned as-is) or an
    ``arviz_plots.PlotCollection``, whose ``viz`` DataTree stores either a
    root-level ``figure`` node or per-variable ``plot`` targets carrying a
    ``.figure`` attribute (the ``PlotlyPlot`` wrapper). Returns ``None`` when no
    figure can be resolved — callers must treat that as a silent no-op.
    """
    if hasattr(obj, 'update_layout'):
        return obj
    viz = getattr(obj, 'viz', None)
    if viz is None:
        return None
    for key in ('figure', 'chart'):
        try:
            fig = viz[key].item()
            if hasattr(fig, 'update_layout'):
                return fig
        except Exception:
            continue
    try:
        targets = np.asarray(viz['plot'].values).ravel()
        for target in targets:
            fig = getattr(target, 'figure', None)
            if hasattr(fig, 'update_layout'):
                return fig
    except Exception:
        pass
    return None


def _autosize_plotly(obj) -> None:
    """Let a Plotly figure / PlotCollection stretch to the notebook/editor width.

    Plotly treats an explicit ``layout.width`` as fixed, so the width stamped by
    arviz-plots (or a hand-built panel) freezes the figure regardless of the
    rendering pane. Clearing it and setting ``autosize=True`` makes the HTML
    renderer track the container width while the explicit **height** (which
    encodes per-row readability) is preserved. Best-effort: any failure leaves
    the figure exactly as built.
    """
    try:
        fig = _plotly_figure_of(obj)
        if fig is None:
            return
        fig.layout.width = None
        fig.update_layout(autosize=True)
    except Exception:  # pragma: no cover - cosmetic only
        pass


def _apply_dark_template(obj: object) -> None:
    """Stamp the dark :data:`_PLOTLY_TEMPLATE` onto a figure or PlotCollection.

    ``plotly.io.templates.default`` only applies to figures built *after* it is
    pinned, and ``arviz_plots`` composes some collections against its own
    default — so figures reaching the display funnel can still carry the stock
    light template. Applying it here, in the one funnel every figure passes
    through, is what keeps displayed and exported output identical. Best-effort:
    any failure leaves the figure exactly as built.
    """
    try:
        fig = _plotly_figure_of(obj)
        if fig is not None:
            fig.update_layout(template=_PLOTLY_TEMPLATE)
    except Exception:  # pragma: no cover - cosmetic only
        pass


def _render_plotly(fig: object, *, height: Optional[int] = None,
                   label: Optional[str] = None,
                   hovermode: Optional[str] = None) -> None:
    """Render a Plotly figure in the notebook (side-effecting; dark theme).

    Sets the shared margins and forwards to :func:`_safe_show`, which applies the
    dark template (:func:`_apply_dark_template`) for display *and* export alike.
    Mirrors the ``pc.show()`` convention used throughout the module; falls back to
    :func:`display` and is a silent no-op when no renderer is available, so a
    headless / plain-script run never raises. ``label`` is forwarded to the
    artifact exporter via :func:`_safe_show`.

    ``hovermode`` overrides Plotly's default ``'closest'`` — pass ``'x unified'``
    for time-series panels so all series report at the hovered date.
    """
    if fig is None:
        return
    fig.update_layout(margin=dict(l=60, r=30, t=70, b=50))
    if height is not None:
        fig.update_layout(height=height)
    if hovermode is not None:
        fig.update_layout(hovermode=hovermode)
    _autosize_plotly(fig)
    _safe_show(fig, label=label, _fallback=display)


def _safe_show(obj: object, *, label: Optional[str] = None,
               _fallback: Optional[object] = None) -> None:
    """Display any Plotly figure or ``arviz_plots`` PlotCollection, swallowing transport errors.

    IDE-managed display back-ends (e.g. PyCharm's DataLore/Kaleido helper) push the
    rendered image to a local HTTP server; when that socket is aborted the underlying
    ``show()`` raises ``ConnectionAbortedError`` (WinError 10053) mid-run. Rendering is a
    side-effect, never part of the model result, so a failure here must never abort the
    workflow. Handles both :class:`arviz_plots.PlotCollection` (``.show()``) and Plotly
    figures; ``pc.show()`` and raw ``fig.show()`` both route through this one guard.

    This is also the single auto-capture point for artifact export: every displayed
    figure is handed to :func:`_export_figure` *before* ``show()``, so a display
    transport failure still yields the exported file. The dark template is applied
    here too — before both display and export — so the two can never diverge
    (``arviz_plots`` PlotCollections reach this funnel still carrying the light
    default, which is why several exported panels used to render white).

    Parameters
    ----------
    obj
        The figure or PlotCollection to display. ``None`` is a no-op.
    label
        Optional filename slug for the exported artifact; auto-derived from the
        figure title (or just auto-numbered) when omitted.
    _fallback
        Optional callable tried when ``obj.show()`` raises — used by
        :func:`_render_plotly` to fall back to :func:`display` for a returned figure.
    """
    if obj is None:
        return
    _apply_dark_template(obj)  # displayed == exported theme
    _autosize_plotly(obj)  # width tracks the editor pane; heights stay explicit
    _export_figure(obj, label=label)
    try:
        obj.show()
    except Exception as exc:  # pragma: no cover - display transport is environment-dependent
        logger.debug("Figure display skipped (renderer/transport failure): %r", exc)
        if _fallback is not None:
            try:
                _fallback(obj)
            except Exception:
                pass


# --- Hover / axis formatting conventions -------------------------------------
# Every hand-built trace gets a ``name=`` (or an explicit ``hoverinfo='skip'``
# for guides and band edges); every hovertemplate ends in ``<extra></extra>``;
# percent-scaled series format hover values as ``.1f`` + '%', decimal
# probabilities as ``.0%``; band+line pairs share a ``legendgroup`` with
# ``showlegend`` only on the line.
_LEGEND_FONT_SIZE = 9


def _hover_pct(label: str, *, axis: str = 'y', lead_text: bool = False,
               date_x: bool = False) -> str:
    """Hovertemplate for a percent-scaled series (values pre-scaled ×100)."""
    lead = '%{text}<br>' if lead_text else ('%{x|%Y-%m-%d}<br>' if date_x else '')
    return f'{lead}{label} = %{{{axis}:.1f}}%<extra></extra>'


def _hover_price(label: str, *, date_x: bool = True, nd: int = 2) -> str:
    """Hovertemplate for a price-space series along a date axis."""
    lead = '%{x|%Y-%m-%d}<br>' if date_x else ''
    return f'{lead}{label}: %{{y:.{nd}f}}<extra></extra>'


def _hover_prob(label: str, *, axis: str = 'y') -> str:
    """Hovertemplate for a decimal probability in [0, 1]."""
    return f'{label} = %{{{axis}:.0%}}<extra></extra>'


# --- Reference geometry (single source of truth for every guide line) ---------
# Zero lines, break-even markers, y=x guides, now-boundaries and horizon markers
# all key on a ROLE here rather than on per-call-site hexes and widths. Every
# guide draws with ``layer='below'`` so it sits behind the data it annotates.
#
# ``zero``     — the invariant of the panel: 0 return, break-even, y=x, p=0.5.
# ``anchor``   — a datum from the data: last price, now-boundary, horizon edge.
# ``emphasis`` — a highlighted cohort/universe statistic worth reading off.
_REF_LINE_KINDS: dict[str, dict[str, Any]] = {
    'zero': {'color': C_REF, 'dash': 'dash', 'width': 1.0},
    'anchor': {'color': C_REF, 'dash': 'dot', 'width': 1.2},
    'emphasis': {'color': C_HIGHLIGHT, 'dash': 'dash', 'width': 1.4},
}
_REF_LINE_OPACITY = 0.7
_REF_BAND_ALPHA = 0.08


def _add_ref_line(fig, *, x: Optional[float] = None, y: Optional[float] = None,
                  kind: str = 'zero', color: Optional[str] = None,
                  annotation_text: Optional[str] = None,
                  row: Optional[int] = None, col: Optional[int] = None,
                  **kwargs: Any) -> None:
    """Draw a horizontal / vertical reference line under the shared spec.

    Replaces the ad-hoc ``add_hline`` / ``add_vline`` calls that had grown seven
    different widths, three dash treatments and two Plotly APIs for what is
    semantically one piece of geometry.

    Parameters
    ----------
    x, y
        Vertical (``x``) or horizontal (``y``) position. Exactly one is given.
    kind
        Role key into :data:`_REF_LINE_KINDS` — ``'zero'``, ``'anchor'`` or
        ``'emphasis'``.
    color
        Overrides the role colour (e.g. a per-series median marker).
    annotation_text
        Optional label, rendered at :data:`_LEGEND_FONT_SIZE` in the line colour.
    row, col
        Subplot target; both omitted applies to the whole figure.
    **kwargs
        Forwarded to Plotly (``opacity``, ``annotation_position``, …).

    Raises
    ------
    ValueError
        If neither or both of ``x`` / ``y`` are supplied, or ``kind`` is unknown.
    """
    if (x is None) == (y is None):
        raise ValueError("Pass exactly one of x= / y= to _add_ref_line.")
    if kind not in _REF_LINE_KINDS:
        raise ValueError(f"Unknown reference-line kind {kind!r}. "
                         f"Valid: {sorted(_REF_LINE_KINDS)}")

    spec = dict(_REF_LINE_KINDS[kind])
    if color is not None:
        spec['color'] = color
    opts: dict[str, Any] = {
        'line': spec,
        'layer': 'below',
        'opacity': kwargs.pop('opacity', _REF_LINE_OPACITY),
    }
    if annotation_text is not None:
        opts['annotation_text'] = annotation_text
        opts['annotation_font'] = dict(size=_LEGEND_FONT_SIZE, color=spec['color'])
    if row is not None or col is not None:
        opts['row'], opts['col'] = row, col
    opts.update(kwargs)

    if y is not None:
        fig.add_hline(y=y, **opts)
    else:
        fig.add_vline(x=x, **opts)


def _add_ref_band(fig, *, x0: float, x1: float, color: str = C_REF,
                  alpha: float = _REF_BAND_ALPHA, **kwargs: Any) -> None:
    """Shade a vertical reference band (e.g. a tolerance corridor around zero)."""
    fig.add_vrect(x0=x0, x1=x1, fillcolor=_hex_to_rgba(color, alpha),
                  line_width=0, layer='below', **kwargs)


def _fmt_axis(fig, *, x: Optional[str] = None, y: Optional[str] = None,
              x_kind: str = 'linear', y_kind: str = 'linear',
              row: Optional[int] = None, col: Optional[int] = None) -> None:
    """Set axis titles + unit formatting in one call.

    ``*_kind``: ``'linear'`` (no formatting), ``'pct'`` (``ticksuffix='%'`` for
    percent-scaled values), ``'prob'`` (``tickformat='.0%'`` for decimal
    probabilities), ``'date'`` (``tickformat='%b %y'``).
    """
    def _apply(update, title, kind):
        kw: dict[str, Any] = {}
        if title is not None:
            kw['title_text'] = title
        if kind == 'pct':
            kw['ticksuffix'] = '%'
        elif kind == 'prob':
            kw['tickformat'] = '.0%'
        elif kind == 'date':
            kw['tickformat'] = '%b %y'
        if kw:
            if row is not None or col is not None:
                update(row=row, col=col, **kw)
            else:
                update(**kw)
    _apply(fig.update_xaxes, x, x_kind)
    _apply(fig.update_yaxes, y, y_kind)


def _show_fig(fig, *, label: Optional[str] = None) -> None:
    """Surface a Plotly :class:`~plotly.graph_objects.Figure` in any front-end.

    Thin wrapper over :func:`_render_plotly` (dark ``arviz-tumma`` template + a silent
    no-op when no renderer is available) for figures that are *returned* rather than
    self-shown -- e.g. the :func:`plot_kalman_forecast` structural forecasts. Mirrors
    the ``pc.show()`` convention used elsewhere in this module.

    Parameters
    ----------
    fig
        The Plotly figure to surface (typically the first element of a
        :func:`plot_kalman_forecast` return tuple).
    label
        Optional filename slug forwarded to the artifact exporter.
    """
    _render_plotly(fig, label=label)


def _present_vars(idata, candidates: Sequence[str]) -> list[str]:
    """Return the subset of ``candidates`` actually in ``idata.posterior``.

    Which posterior variables exist depends on how ``KalmanFilterPriceTarget.fit`` was
    parameterized (marginalized → ``log_state_init``; non-centred → ``z_innov``;
    stochastic-volatility → ``vol_step_size`` / ``nu_obs`` / ``vol_anchor_offset``), so
    summaries filter to what is present rather than hard-coding names.
    """
    post = getattr(idata, 'posterior', idata)
    have = set(post.data_vars)
    return [v for v in candidates if v in have]


def _kf_fit_kind(idata) -> str:
    """Human label for the parameterization actually fitted (from posterior vars)."""
    post = getattr(idata, 'posterior', idata)
    dv = set(post.data_vars)
    if 'log_vol' in dv:
        return 'Stochastic-volatility GRW (+trend)'
    if 'log_state_init' in dv:
        return 'Marginalized GRW (+trend)'
    return 'Non-centred GRW (+trend)'


@contextlib.contextmanager
def _quiet_degenerate_density():
    """Silence arviz-stats' single-value/non-finite KDE ``UserWarning`` while plotting.

    A stuck chain (poorly-mixed scale) or a structurally-constant slice (e.g. a
    one-level :class:`pm.ZeroSumNormal` effect, which is identically 0) makes
    arviz-stats fall back from a smooth KDE to a delta spike and emit
    ``arviz_stats/base/density.py:672`` *once per affected (chain, slice)* — the
    ``"Your data appears to have a single value or no finite values"`` warnings.
    The plot itself is still valid (the spike correctly depicts a pinned
    marginal), so we suppress only that one message rather than letting it flood
    the diagnostics log. Convergence problems are surfaced numerically by the
    R-hat / ESS report instead, not by raw KDE warnings.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='Your data appears to have a single value or no finite values',
            category=UserWarning,
        )
        yield


def _degenerate_posterior_vars(idata, var_names: Sequence[str],
                               *, atol: float = 1e-8) -> list[str]:
    """Return ``var_names`` whose pooled posterior draws are constant / non-finite.

    A variable is flagged when *every* non-sample slice is either all-non-finite
    or has a (chain+draw) spread ``<= atol``. Such variables produce only the
    delta-spike KDE that triggers ``density.py:672``; reporting them tells the
    reader *which* parameter collapsed (a real modelling signal) rather than
    leaving an anonymous warning in the log.
    """
    post = getattr(idata, 'posterior', idata)
    flagged: list[str] = []
    for v in var_names:
        if v not in post.data_vars:
            continue
        da = post[v]
        arr = np.asarray(da.values, dtype='float64')
        finite = np.isfinite(arr)
        if not finite.any():
            flagged.append(v)
            continue
        sample_axes = tuple(i for i, d in enumerate(da.dims) if d in ('chain', 'draw'))
        masked = np.where(finite, arr, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            spread = (np.nanmax(masked, axis=sample_axes)
                      - np.nanmin(masked, axis=sample_axes))
        if not np.isfinite(spread).any() or float(np.nanmax(spread)) <= atol:
            flagged.append(v)
    return flagged


def plot_price_target_path(
        idata,
        *,
        state_var: str = "state",
        observed: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        last_price: Optional[float] = None,
        ticker: Optional[str] = None,
        hdi_probs: Sequence[float] = (0.94, 0.5),
        figsize: Optional[tuple[float, float]] = None,
        color: str = C_POSTERIOR,
        observed_color: str = C_OBSERVED,
):
    """Compose a Kalman-smoothed price-target trajectory with ``arviz_plots``.

    Builds the plot from the low-level composition API
    (:meth:`arviz_plots.PlotCollection.grid` + :mod:`arviz_plots.visuals`) so the
    posterior latent state, its credible bands, and the raw observations share a
    single time axis.

    Parameters
    ----------
    idata
        Inference object whose ``posterior`` holds ``state_var`` over a ``time`` dim.
    state_var
        Posterior variable carrying the price-space latent state.
    observed
        Observed analyst price targets aligned to the ``time`` axis (plotted as points).
    dates
        Time index aligned to the ``time`` dim. When omitted, the posterior ``time``
        coord is used (treated as datetime if it parses as such).
    last_price
        Reference last price; drawn as a dashed horizontal line when finite.
    ticker
        Optional label for the title.
    hdi_probs
        Credible-interval masses to shade, widest first.
    figsize, color, observed_color
        Cosmetic controls. ``figsize`` is in **dots** (pixels); when omitted it
        derives from :func:`_display_width_px` and the panel autosizes to the
        editor width on display.

    Returns
    -------
    arviz_plots.PlotCollection
        The composed collection (already drawn); call ``.show()`` to display.
    """
    post = idata.posterior[state_var]
    if "time" not in post.dims:
        raise ValueError(f"{state_var!r} has no 'time' dim; dims={post.dims}.")
    n_time = post.sizes["time"]

    # Resolve the x-axis: prefer explicit ``dates``, else the posterior ``time`` coord.
    # Only treat the coord as datetime when its dtype actually is one -- otherwise
    # pd.to_datetime would silently coerce a plain integer time index into 1970-epoch
    # timestamps.
    if dates is None and "time" in post.coords:
        coord_vals = np.asarray(post["time"].values)
        if np.issubdtype(coord_vals.dtype, np.datetime64):
            dates = pd.DatetimeIndex(coord_vals)
        elif coord_vals.dtype == object:
            parsed = pd.to_datetime(coord_vals, errors="coerce")
            if not bool(np.asarray(pd.isna(parsed)).all()):
                dates = pd.DatetimeIndex(parsed)
    use_dates = (
            dates is not None
            and len(dates) == n_time
            and not bool(np.asarray(pd.isna(dates)).all())
    )
    # Plotly consumes datetimes on the x-axis natively (it formats the date ticks),
    # so — unlike the former matplotlib path — no ``mdates.date2num`` projection is
    # needed; pass the datetimes straight through.
    x = xr.DataArray(
        np.asarray(dates) if use_dates else np.arange(n_time),
        dims="time",
    )

    median = post.median(("chain", "draw"))
    ds = post.to_dataset()

    _fk = ({"figsize": figsize, "figsize_units": "dots"} if figsize is not None
           else _azp_figure_kwargs(520))
    pc = azp.PlotCollection.grid(ds, backend="plotly", figure_kwargs=_fk)
    target = pc.get_target(state_var, {})  # arviz_plots PlotlyPlot (figure + row/col)

    # Nested HDI bands: widest first with the lightest alpha so inner masses darken.
    band_alphas = (0.16, 0.28, 0.40, 0.50)
    for prob, alpha in zip(sorted(hdi_probs, reverse=True), band_alphas):
        band = post.azstats.hdi(prob=prob)
        azv.fill_between_y(
            median, target, x=x,
            y_bottom=band.sel(ci_bound="lower"),
            y_top=band.sel(ci_bound="upper"),
            color=color, alpha=alpha,
        )

    azv.line_xy(median, target, x=x, y=median, color=color, width=2.2)

    if observed is not None:
        obs = xr.DataArray(np.asarray(observed, dtype="float64"), dims="time")
        azv.scatter_xy(
            median, target, x=x, y=obs,
            color=observed_color, size=34,
            edgecolor=C_PANEL_BG, width=0.6,
        )

    if last_price is not None and np.isfinite(last_price):
        _add_ref_line(target, y=float(last_price), kind='anchor')

    if use_dates:
        target.update_xaxes(tickformat="%b %y")

    azv.labelled_x(median, target, text="as-of date" if use_dates else "time step")
    azv.labelled_y(median, target, text="price target")
    title = "Kalman-smoothed price-target path"
    if ticker:
        title += f" — {ticker}"
    pc.add_title(title)

    # Hand-built legend: the composition primitives register their traces with
    # ``showlegend=False``, so add invisible proxy traces (x/y = None) carrying only
    # the legend labels — Plotly's analogue of the former Line2D/Patch handle list.
    if go is not None:
        fig = target.figure
        _r, _c = target.row, target.col
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color=color, width=2.2),
            name="posterior median state"), row=_r, col=_c)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, opacity=0.4, size=12, symbol="square"),
            name=f"{int(max(hdi_probs) * 100)}% / {int(min(hdi_probs) * 100)}% HDI"),
            row=_r, col=_c)
        if observed is not None:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=observed_color, size=8,
                            line=dict(color=C_PANEL_BG, width=1)),
                name="observed price target"), row=_r, col=_c)
        if last_price is not None and np.isfinite(last_price):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=C_REF, dash="dash"),
                name="last price"), row=_r, col=_c)
        fig.update_layout(showlegend=True)

    return pc


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` hex string to a Plotly ``rgba(r,g,b,a)`` string.

    Plotly fills/markers take alpha through the colour string (``fillcolor`` /
    ``rgba``) rather than a separate ``alpha`` argument, so the hand-built panels
    that shade credible bands express transparency this way.
    """
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


def _xval(v):
    """Coerce a scalar to a Plotly-friendly x value for shapes/annotations.

    Datetimes become a native :class:`datetime.datetime` (JSON-serialisable, so
    ``write_image`` / ``write_html`` exports succeed — a bare ``pandas.Timestamp`` is
    rejected by Plotly's encoder); everything else becomes a float.
    """
    if isinstance(v, np.datetime64) or np.issubdtype(np.asarray(v).dtype, np.datetime64):
        return pd.Timestamp(v).to_pydatetime()
    return float(v)


def _plotly_band(fig, x, lo, hi, *, color, alpha, name=None, row=None, col=None,
                 showlegend=False):
    """Add a shaded credible band (``fill='tonexty'`` between ``lo`` and ``hi``).

    The upper edge is drawn first as an invisible line, then the lower edge fills up
    to it — the Plotly idiom for the matplotlib ``fill_between`` the panels used.
    """
    fig.add_trace(go.Scatter(x=x, y=hi, mode='lines', line=dict(width=0),
                             hoverinfo='skip', showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=lo, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor=_hex_to_rgba(color, alpha),
                             name=name, hoverinfo='skip',
                             showlegend=bool(showlegend and name)), row=row, col=col)


def _kde_xy(values, *, clip_low: Optional[float] = None, n: int = 200,
            bw=None) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Gaussian-KDE (x, density) for the hand-built Plotly density panels.

    Plotly has no native KDE, so the panels that used ``seaborn.kdeplot`` evaluate a
    :class:`scipy.stats.gaussian_kde` on a robust 0.5–99.5 pct window. Returns
    ``(None, None)`` when the sample is too small or constant (a KDE would be a spike).
    """
    v = np.asarray(values, dtype='float64')
    v = v[np.isfinite(v)]
    if clip_low is not None:
        v = v[v >= clip_low]
    if v.size < 5 or np.allclose(v, v[0]):
        return None, None
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(v, bw_method=bw)
    lo, hi = np.nanpercentile(v, [0.5, 99.5])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    xs = np.linspace(lo, hi, n)
    return xs, kde(xs)


def _annotate_forecast_horizons(fig, fx, pg, *, color, row=1, col=1):
    """Mark each forecast horizon with a dotted vline + its human label.

    ``pg`` is the ``forecast`` ``predictions`` group; when present its ``label``
    coord carries the canonical human-readable horizon names (e.g. ``"Next
    earnings"`` / ``"Next report"``) resolved from
    :data:`KalmanFilterModel.FISCAL_HORIZONS`. Labels are placed at the top of the
    plot (``yref='paper'``) so they read as fiscal-event markers on the time axis
    rather than colliding with the forecast band.
    """
    labels = (np.asarray(pg['label'].values) if 'label' in pg.coords
              else np.asarray([f'+{int(round(v))}d' for v in
                               np.atleast_1d(pg['time_future'].values)
                               if np.isscalar(v) or True]))
    for xv, lb in zip(np.atleast_1d(fx), labels):
        _add_ref_line(fig, x=_xval(xv), kind='anchor', color=color,
                      opacity=0.4, row=row, col=col)
        fig.add_annotation(x=_xval(xv), y=1.0, yref='paper', yanchor='top',
                           text=str(lb), textangle=30, showarrow=False,
                           font=dict(size=8, color=color), xanchor='right',
                           row=row, col=col)


def plot_kalman_forecast(idata_fit, pred, *, observed=None, dates=None,
                         last_price=None, ticker=None, state_var='state',
                         height_px=540, hist_color=C_POSTERIOR,
                         fc_color=C_FORECAST, observed_color=C_OBSERVED,
                         pp_overlay=True, pp_draws=80, pp_color=C_DRAWS,
                         volatility_panel='auto', vol_color=C_VOL,
                         random_seed=RANDOM_SEED):
    """Overlay the fitted smoothed state with the structural forecast bands.

    Mirrors the reference notebook's "Posterior Predictions Plotted": the fitted
    Kalman-smoothed state + HDI up to the last observation, a vertical boundary at
    "now", then ``KalmanFilterPriceTarget.forecast()`` predictive bands extending to
    the future fiscal-calendar events. Each forecast horizon is marked with its
    **human label** (e.g. ``"Next earnings"`` / ``"Next report"``) carried on the
    ``predictions`` group's ``label`` coord — the canonical
    :data:`KalmanFilterModel.FISCAL_HORIZONS` names, so the time axis is annotated
    with the fiscal events the horizon projects to rather than raw column names.

    Following the canonical PyMC stochastic-volatility example (true returns +
    posterior-predictive returns overlaid above, posterior volatility below), the
    forecast band is augmented with a thinned **posterior-predictive spaghetti** of
    ``forecast_pt`` draws, and — when the fit carries a time-varying
    ``sigma_obs(t)`` (``stochastic_volatility=True``) — a companion lower panel
    plots the posterior observation-volatility path over the historical axis.

    Parameters
    ----------
    idata_fit
        InferenceData from :meth:`KalmanFilterPriceTarget.fit` (``state`` over ``time``;
        and, under stochastic volatility, ``sigma_obs`` over ``time``).
    pred
        Output of :meth:`KalmanFilterPriceTarget.forecast` (``predictions`` group or a
        raw ``xarray.Dataset``) with ``forecast_pt`` over ``time_future`` and an
        optional human-label ``label`` coord.
    observed, dates, last_price, ticker
        Observed targets, historical as-of dates, the spot price reference, and a title
        label, all aligned to the fitted ``time`` axis.
    pp_overlay, pp_draws, pp_color
        Whether to overlay ``pp_draws`` thinned posterior-predictive ``forecast_pt``
        draws (green spaghetti, example style), and their colour.
    volatility_panel, vol_color
        ``'auto'`` adds the lower posterior-volatility panel only when the fit exposes
        a time-varying ``sigma_obs``; pass ``True`` / ``False`` to force it. ``vol_color``
        is that panel's colour.
    random_seed
        Seed for thinning the posterior-predictive draws (reproducible spaghetti).

    Returns
    -------
    tuple
        ``(fig, None)`` where ``fig`` is a Plotly :class:`~plotly.graph_objects.Figure`
        (a single panel, or a 2-row state/volatility stack when the posterior-volatility
        panel is drawn). The second element is retained for call-site unpacking parity.
    """
    post = idata_fit.posterior[state_var]
    n_time = post.sizes['time']
    use_dates = dates is not None and len(dates) == n_time
    # Plotly renders datetimes natively, so — unlike the former matplotlib path — the
    # history axis carries the datetimes themselves rather than a ``date2num`` float.
    hx = np.asarray(dates) if use_dates else np.arange(n_time, dtype=float)
    hist_med = post.median(('chain', 'draw')).values
    _hdi = post.azstats.hdi(prob=0.94)
    hlo = _hdi.sel(ci_bound='lower').values
    hhi = _hdi.sel(ci_bound='upper').values

    pg = pred.predictions if hasattr(pred, 'predictions') else pred
    fpt = pg['forecast_pt']
    tf = np.asarray(pg['time_future'].values)
    if np.issubdtype(tf.dtype, np.datetime64):
        # Datetime horizons share the history axis directly when it is on dates; when
        # the history axis fell back to an integer index (``len(dates) != n_time``) the
        # horizons are projected onto it as day-offsets from the last observation.
        if use_dates:
            fx = tf
        else:
            anchor = (np.asarray(dates)[-1] if dates is not None and len(dates) else tf[0])
            fx = hx[-1] + (tf - anchor) / np.timedelta64(1, 'D')
    else:
        fx = hx[-1] + np.asarray(tf, dtype=float)
    f_med = fpt.median(('chain', 'draw')).values
    f_lo = fpt.quantile(_HDI_LO, dim=('chain', 'draw')).values
    f_hi = fpt.quantile(_HDI_HI, dim=('chain', 'draw')).values

    # Resolve the optional stochastic-volatility observation-noise path. The
    # companion volatility panel mirrors the reference example's lower subplot.
    so = idata_fit.posterior['sigma_obs'] if 'sigma_obs' in idata_fit.posterior else None
    has_vol = so is not None and 'time' in so.dims and so.sizes['time'] == n_time
    show_vol = has_vol if volatility_panel == 'auto' else bool(volatility_panel) and has_vol

    if show_vol:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.06,
                            subplot_titles=('', 'Posterior observation volatility'))
    else:
        # A 1x1 subplot grid (not a bare go.Figure) so the shared row/col=1 addressing
        # used throughout — add_trace / add_vline / update_yaxes — resolves a grid ref.
        fig = make_subplots(rows=1, cols=1)

    # 94% HDI band (fit) + Kalman-smoothed state.
    _plotly_band(fig, hx, hlo, hhi, color=hist_color, alpha=0.25,
                 name='94% HDI (fit)', row=1, col=1, showlegend=True)
    fig.add_trace(go.Scatter(x=hx, y=hist_med, mode='lines',
                             line=dict(color=hist_color, width=2.2),
                             name='Kalman state (fit)'), row=1, col=1)
    if observed is not None:
        fig.add_trace(go.Scatter(
            x=hx, y=np.asarray(observed, dtype=float), mode='markers',
            marker=dict(color=observed_color, size=8, line=dict(color=C_PANEL_BG, width=0.6)),
            name='observed target'), row=1, col=1)

    # Posterior-predictive forecast spaghetti (canonical SV-example overlay): a
    # thinned set of forecast_pt draws, each stitched to the now-boundary so the
    # predictive fan reads as draws rather than a single band.
    if pp_overlay and pp_draws > 0:
        draws = fpt.stack(s=('chain', 'draw')).transpose('s', 'time_future').values
        n_draw = draws.shape[0]
        if n_draw:
            rng_local = np.random.default_rng(random_seed)
            idx = (rng_local.choice(n_draw, size=min(pp_draws, n_draw), replace=False)
                   if n_draw > pp_draws else np.arange(n_draw))
            xline = np.r_[hx[-1], fx]
            pp_rgba = _hex_to_rgba(pp_color, 0.18)
            for j in idx:
                fig.add_trace(go.Scatter(
                    x=xline, y=np.r_[hist_med[-1], draws[j]], mode='lines',
                    line=dict(color=pp_rgba, width=0.8), hoverinfo='skip',
                    showlegend=False), row=1, col=1)
            # Single proxy trace for the legend (one label, not pp_draws of them).
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                     line=dict(color=pp_color, width=1.0),
                                     name='posterior-predictive draws'), row=1, col=1)

    # 94% forecast band + forecast median (stitched to the now-boundary).
    _plotly_band(fig, fx, f_lo, f_hi, color=fc_color, alpha=0.22,
                 name='94% PI (forecast)', row=1, col=1, showlegend=True)
    fig.add_trace(go.Scatter(
        x=np.r_[hx[-1], fx], y=np.r_[hist_med[-1], f_med], mode='lines+markers',
        line=dict(color=fc_color, width=2.2, dash='dash'), marker=dict(size=6),
        name='forecast pt'), row=1, col=1)
    _add_ref_line(fig, x=_xval(hx[-1]), kind='anchor', row=1, col=1)
    if last_price is not None and np.isfinite(last_price):
        _add_ref_line(fig, y=float(last_price), kind='anchor',
                      annotation_text='last price', row=1, col=1)

    # Human-labelled fiscal-event horizon markers (Next earnings / Next report / ...).
    _annotate_forecast_horizons(fig, fx, pg, color=fc_color, row=1, col=1)

    fig.update_yaxes(title_text='price target', row=1, col=1)

    if show_vol:
        v_med = so.mean(('chain', 'draw')).values
        v_lo = so.quantile(_HDI_LO, dim=('chain', 'draw')).values
        v_hi = so.quantile(_HDI_HI, dim=('chain', 'draw')).values
        _plotly_band(fig, hx, v_lo, v_hi, color=vol_color, alpha=0.25, row=2, col=1)
        fig.add_trace(go.Scatter(x=hx, y=v_med, mode='lines',
                                 line=dict(color=vol_color, width=2.0),
                                 name='posterior mean σ_obs(t)'), row=2, col=1)
        _add_ref_line(fig, x=_xval(hx[-1]), kind='anchor', row=2, col=1)
        fig.update_yaxes(title_text='σ_obs', row=2, col=1)

    # Numeric (index) axis: pin an explicit finite window; datetime axes autorange.
    _vol_row = 2 if show_vol else 1
    if not use_dates:
        x_all = np.concatenate([np.asarray(hx, dtype=float).ravel(),
                                np.asarray(fx, dtype=float).ravel()])
        x_all = x_all[np.isfinite(x_all)]
        if x_all.size:
            x_lo, x_hi = float(x_all.min()), float(x_all.max())
            pad = (x_hi - x_lo) * 0.02 or 1.0
            fig.update_xaxes(range=[x_lo - pad, x_hi + pad])
    else:
        fig.update_xaxes(tickformat='%b %y')
    fig.update_xaxes(title_text='as-of date' if use_dates else 'time step',
                     row=_vol_row, col=1)

    # Width is left unset: _render_plotly/_autosize_plotly stretch the panel to
    # the editor pane; only the height (readability budget) is pinned, widened
    # when the posterior-volatility companion row is stacked below.
    fig.update_layout(
        title='Kalman state + structural forecast' + (f' — {ticker}' if ticker else ''),
        showlegend=True,
        height=int(height_px + (200 if show_vol else 0)),
    )
    return fig, None


def plot_kalman_forecast_returns(idata_fit, pred, *, observed=None, dates=None,
                                 last_price=None, ticker=None, state_var='state',
                                 height_px=H_FORECAST, hist_color=C_POSTERIOR,
                                 fc_color=C_FORECAST, observed_color=C_OBSERVED,
                                 volatility_panel='auto', vol_color=C_VOL):
    """Return-space structural forecast — observed vs expected returns per fiscal event.

    The return-space twin of :func:`plot_kalman_forecast` (which stays available
    for price-space views but is no longer called by the workflow, 0.9.9.11):
    everything is expressed as an **implied return over the spot price**
    (``x / last_price − 1``, percent-scaled at this display boundary), so the
    chart answers the investment question directly — *what return does the
    smoothed consensus imply, and what return is forecast at each upcoming
    fiscal event?*

    Content:

    - **Observed historical returns** — ``observed / last_price − 1`` as points.
    - **Smoothed implied-upside path** — the latent state's implied return with
      its 94% HDI band.
    - **Per-fiscal-event forecast, nested bands** — the *predictive* band from
      ``forecast_pt / last_price − 1`` (wide: latent + observation noise) and
      the *latent* band from the ``implied_upside_future`` draws computed by
      :meth:`KalmanFilterPriceTarget.forecast` (narrow: state uncertainty
      only). The two are deliberately nested — the gap between them is the
      analyst observation noise. The latent median line is stitched at the
      now-boundary and each horizon carries a ``+X.X%`` median annotation.
    - **0% break-even line** (replaces the price-space last-price line), the
      ``now`` boundary, and the human-labelled fiscal-event markers
      (:func:`_annotate_forecast_horizons`, reused verbatim).
    - The stochastic-volatility companion row (posterior **median**
      ``sigma_obs(t)``) when the fit exposes a time-varying scale.

    Requires a finite positive ``last_price``; without one the return scale is
    undefined and the function falls back to the price-space
    :func:`plot_kalman_forecast` with a logged notice.

    Returns
    -------
    tuple
        ``(fig, None)`` — call-site parity with :func:`plot_kalman_forecast`.
    """
    if last_price is None or not np.isfinite(last_price) or last_price <= 0:
        logger.info('plot_kalman_forecast_returns: no usable last_price — '
                    'falling back to the price-space structural forecast.')
        return plot_kalman_forecast(idata_fit, pred, observed=observed,
                                    dates=dates, last_price=last_price,
                                    ticker=ticker, state_var=state_var)
    lp = float(last_price)

    def _ret_pct(arr):
        return (np.asarray(arr, dtype='float64') / lp - 1.0) * 100.0

    post = idata_fit.posterior[state_var]
    n_time = post.sizes['time']
    use_dates = dates is not None and len(dates) == n_time
    hx = np.asarray(dates) if use_dates else np.arange(n_time, dtype=float)
    hist_med = _ret_pct(post.median(('chain', 'draw')).values)
    _hdi = post.azstats.hdi(prob=0.94)
    hlo = _ret_pct(_hdi.sel(ci_bound='lower').values)
    hhi = _ret_pct(_hdi.sel(ci_bound='upper').values)

    pg = pred.predictions if hasattr(pred, 'predictions') else pred
    fpt = pg['forecast_pt']
    tf = np.asarray(pg['time_future'].values)
    if np.issubdtype(tf.dtype, np.datetime64):
        if use_dates:
            fx = tf
        else:
            anchor = (np.asarray(dates)[-1] if dates is not None and len(dates) else tf[0])
            fx = hx[-1] + (tf - anchor) / np.timedelta64(1, 'D')
    else:
        fx = hx[-1] + np.asarray(tf, dtype=float)

    # Predictive return band (latent + observation noise): forecast_pt / lp − 1.
    p_med = _ret_pct(fpt.median(('chain', 'draw')).values)
    p_lo = _ret_pct(fpt.quantile(_HDI_LO, dim=('chain', 'draw')).values)
    p_hi = _ret_pct(fpt.quantile(_HDI_HI, dim=('chain', 'draw')).values)

    # Latent return band (state uncertainty only): the decimal
    # implied_upside_future draws when the forecast carried a spot anchor,
    # else forecast_state / lp − 1.
    if 'implied_upside_future' in pg:
        _iu = pg['implied_upside_future'] * 100.0
    else:
        _iu = (pg['forecast_state'] / lp - 1.0) * 100.0
    l_med = _iu.median(('chain', 'draw')).values
    l_lo = _iu.quantile(_HDI_LO, dim=('chain', 'draw')).values
    l_hi = _iu.quantile(_HDI_HI, dim=('chain', 'draw')).values

    so = idata_fit.posterior['sigma_obs'] if 'sigma_obs' in idata_fit.posterior else None
    has_vol = so is not None and 'time' in so.dims and so.sizes['time'] == n_time
    show_vol = has_vol if volatility_panel == 'auto' else bool(volatility_panel) and has_vol

    if show_vol:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.06,
                            subplot_titles=('', 'Posterior observation volatility'))
    else:
        fig = make_subplots(rows=1, cols=1)

    # Smoothed implied-return path + band.
    _plotly_band(fig, hx, hlo, hhi, color=hist_color, alpha=0.25,
                 name='94% HDI (smoothed implied return)', row=1, col=1,
                 showlegend=True)
    fig.add_trace(go.Scatter(
        x=hx, y=hist_med, mode='lines',
        line=dict(color=hist_color, width=2.2),
        hovertemplate=_hover_pct('smoothed implied return', date_x=use_dates),
        name='smoothed implied return', legendgroup='state'), row=1, col=1)
    if observed is not None:
        fig.add_trace(go.Scatter(
            x=hx, y=_ret_pct(observed), mode='markers',
            marker=dict(color=observed_color, size=8,
                        line=dict(color=C_PANEL_BG, width=0.6)),
            hovertemplate=_hover_pct('observed implied return', date_x=use_dates),
            name='observed target return', legendgroup='observed'), row=1, col=1)

    # Nested forecast bands: predictive (wide) then latent (narrow) on top.
    _plotly_band(fig, fx, p_lo, p_hi, color=fc_color, alpha=0.18,
                 name='94% PI — predictive return (incl. obs noise)',
                 row=1, col=1, showlegend=True)
    _plotly_band(fig, fx, l_lo, l_hi, color=fc_color, alpha=0.32,
                 name='94% HDI — latent expected return',
                 row=1, col=1, showlegend=True)
    fig.add_trace(go.Scatter(
        x=np.r_[hx[-1], fx], y=np.r_[hist_med[-1], l_med], mode='lines+markers',
        line=dict(color=fc_color, width=2.2, dash='dash'), marker=dict(size=6),
        customdata=np.c_[np.r_[hist_med[-1], p_lo], np.r_[hist_med[-1], p_hi]],
        hovertemplate=('%{x|%Y-%m-%d}<br>expected return = %{y:.1f}%<br>'
                       'predictive 94% = [%{customdata[0]:.1f}%, '
                       '%{customdata[1]:.1f}%]<extra></extra>')
        if use_dates else
        ('expected return = %{y:.1f}%<br>predictive 94% = '
         '[%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>'),
        name='expected return forecast', legendgroup='forecast'), row=1, col=1)

    # Per-horizon +X.X% annotations under each fiscal-event marker.
    for _x, _m in zip(np.atleast_1d(fx), np.atleast_1d(l_med)):
        fig.add_annotation(x=_xval(_x), y=float(_m), yanchor='bottom',
                           text=f'{_m:+.1f}%', showarrow=False,
                           font=dict(size=9, color=fc_color), row=1, col=1)

    _add_ref_line(fig, x=_xval(hx[-1]), kind='anchor', row=1, col=1)
    _add_ref_line(fig, y=0.0, kind='zero', annotation_text='break-even',
                  row=1, col=1)
    _annotate_forecast_horizons(fig, fx, pg, color=fc_color, row=1, col=1)
    fig.update_yaxes(title_text='implied return over spot (%)', ticksuffix='%',
                     row=1, col=1)

    if show_vol:
        v_med = so.median(('chain', 'draw')).values
        v_lo = so.quantile(_HDI_LO, dim=('chain', 'draw')).values
        v_hi = so.quantile(_HDI_HI, dim=('chain', 'draw')).values
        _plotly_band(fig, hx, v_lo, v_hi, color=vol_color, alpha=0.25,
                     name='94% HDI (σ_obs)', row=2, col=1)
        fig.add_trace(go.Scatter(
            x=hx, y=v_med, mode='lines', line=dict(color=vol_color, width=2.0),
            hovertemplate='σ_obs = %{y:.4f}<extra></extra>',
            name='posterior median σ_obs(t)', legendgroup='vol'), row=2, col=1)
        _add_ref_line(fig, x=_xval(hx[-1]), kind='anchor', row=2, col=1)
        fig.update_yaxes(title_text='σ_obs', row=2, col=1)

    _vol_row = 2 if show_vol else 1
    if not use_dates:
        x_all = np.concatenate([np.asarray(hx, dtype=float).ravel(),
                                np.asarray(fx, dtype=float).ravel()])
        x_all = x_all[np.isfinite(x_all)]
        if x_all.size:
            x_lo, x_hi = float(x_all.min()), float(x_all.max())
            pad = (x_hi - x_lo) * 0.02 or 1.0
            fig.update_xaxes(range=[x_lo - pad, x_hi + pad])
    else:
        fig.update_xaxes(tickformat='%b %y')
    fig.update_xaxes(title_text='as-of date' if use_dates else 'time step',
                     row=_vol_row, col=1)

    fig.update_layout(
        title=('Observed vs expected returns — structural forecast to fiscal events'
               + (f' — {ticker}' if ticker else '')),
        showlegend=True,
        legend=dict(font_size=_LEGEND_FONT_SIZE),
        hovermode='x unified',
        height=int(height_px + (200 if show_vol else 0)),
    )
    return fig, None


def build_noise_wideners(df: pd.DataFrame, *, fillna: bool = True) -> dict[str, np.ndarray]:
    """Model-facing observation-noise wideners for one snapshot frame.

    SINGLE SOURCE OF TRUTH for the observation-noise wideners shared by the §2.4c EDA
    panel and the §4.2 model containers, so the picture and the likelihood agree by
    construction. Each quantity is returned in the units the measurement model consumes
    (``sigma_obs = sigma_obs_base * (1 + range + cv + 0.5*max(vol_drift, 0)) /
    sqrt(n_analysts)`` — rising-vol regimes widen the observation noise; falling or
    flat vol contributes 0 via the non-negativity clip).

    Parameters
    ----------
    df
        A ``kalman_df``-shaped frame (universe EDA or the modelling ``model_df``).
    fillna
        ``True`` (the §4.2 model contract): missing inputs -> 0 and the non-negative
        wideners are clipped at 0, so ``sigma_obs`` is always finite. ``False`` (the EDA
        contract): keep NaNs and the raw signed values, so the marginals show the true
        distribution before the model's 0-fill / clip.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``range``, ``cv``, ``vol_drift``, ``sqrt_n`` plus the realised per-row
        ``multiplier`` = ``(1 + range + cv + 0.5*max(vol_drift, 0)) / sqrt_n``.
    """
    range_col = (KALMAN_RANGE_WIDENER_FEATURE
                 if KALMAN_RANGE_WIDENER_FEATURE in df.columns else None)
    sigma_col = (KALMAN_CONSENSUS_SIGMA_FEATURE
                 if KALMAN_CONSENSUS_SIGMA_FEATURE in df.columns else None)
    vol_cols = [c for c in (KALMAN_VOL_DRIFT_FEATURE,) if c in df.columns]

    def _col(name):
        s = df[name].astype('float64') if name and name in df.columns \
            else pd.Series(np.nan, index=df.index)
        return s.fillna(0.0) if fillna else s

    range_s = _col(range_col)
    if sigma_col and sigma_col in df.columns:
        lp = df['last_price'].astype('float64')
        lp = lp.clip(lower=1e-9) if fillna else lp.where(lp > 0)
        cv_s = _col(sigma_col) / lp
    else:
        cv_s = pd.Series(0.0 if fillna else np.nan, index=df.index)
    vol_s = (df[vol_cols].astype('float64').mean(axis=1) if vol_cols
             else pd.Series(0.0 if fillna else np.nan, index=df.index))
    if 'n_analysts' in df.columns:
        n_an = df['n_analysts'].astype('float64').clip(lower=1.0)
    else:
        n_an = pd.Series(1.0, index=df.index)
    sqrt_n = np.sqrt(n_an.to_numpy())

    if fillna:
        # Model contract: non-negative, finite.
        range_s = range_s.clip(lower=0.0)
        cv_s = cv_s.clip(lower=0.0)
        vol_s = vol_s.fillna(0.0).clip(lower=0.0)
        mult = (1.0 + range_s.to_numpy() + cv_s.to_numpy()
                + 0.5 * vol_s.to_numpy()) / sqrt_n
    else:
        # EDA contract: clip only the genuinely-non-negative terms for the multiplier
        # so it stays comparable to the model, but leave per-feature marginals raw.
        mult = (1.0 + range_s.clip(lower=0).fillna(0).to_numpy()
                + cv_s.clip(lower=0).fillna(0).to_numpy()
                + 0.5 * vol_s.clip(lower=0).fillna(0).to_numpy()) / sqrt_n

    return {'range': range_s.to_numpy(), 'cv': cv_s.to_numpy(),
            'vol_drift': vol_s.to_numpy(), 'sqrt_n': sqrt_n, 'multiplier': mult}


def _resolve_env_setting(key: str, env_file: str = 'environment_variables.txt',
                         default: Optional[str] = None) -> Optional[str]:
    """Return ``key`` from ``os.environ``, falling back to environment_variables.txt.

    The process may have been started without sourcing ``set_env.ps1``, in which case
    ``os.environ`` lacks ``key``; we then parse the ``KEY=VALUE`` lines of the project's
    ``environment_variables.txt`` as a fallback before returning ``default``.

    Parameters
    ----------
    key
        Environment variable name to resolve (e.g. ``'DB_URL'``).
    env_file
        Name of the dotenv-style file searched upward from the CWD.
    default
        Value returned when ``key`` is set neither in the environment nor the file.

    Returns
    -------
    Optional[str]
        The resolved value, or ``default`` when not found.
    """
    val = os.environ.get(key)
    if val:
        return val

    here = Path.cwd()
    for base in (here, *here.parents):
        candidate = base / env_file
        if candidate.is_file():
            for raw in candidate.read_text(encoding='utf-8').splitlines():
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, _, value = line.partition('=')
                if k.strip() == key:
                    return value.strip().strip('"').strip("'")
            break
    return default


def resolve_db_url(env_file: str = 'environment_variables.txt') -> str:
    """Return ``DB_URL`` from the environment, falling back to environment_variables.txt.

    The process may have been started without sourcing ``set_env.ps1``, in which case
    ``os.environ`` has no ``DB_URL``; we then parse the ``KEY=VALUE`` lines of the
    project's ``environment_variables.txt`` as a fallback.
    """
    url = _resolve_env_setting('DB_URL', env_file=env_file)
    if url:
        return url
    raise KeyError(
        "DB_URL not set in os.environ and not found in environment_variables.txt. "
        "Run `. .\\set_env.ps1` before launching, or add a DB_URL line."
    )


def export_to_analytics_db(df: pd.DataFrame, table_name: str,
                           if_exists: str = 'replace') -> Optional[int]:
    """Export ``df`` to the PostgreSQL analytics schema.

    Local copy of ``data_utils.export_to_analytics_db`` that resolves the DB connection
    via :func:`resolve_db_url` — so it inherits the ``environment_variables.txt`` fallback
    used everywhere else in this script, instead of the bare ``os.environ['DB_URL']``
    lookup in ``get_analytics_engine`` that raises when ``set_env.ps1`` was not sourced.
    The target schema is read from ``DB_ANALYTICS_SCHEMA`` (default ``analytics``) with
    the same env-file fallback.

    Parameters
    ----------
    df
        DataFrame to export.
    table_name
        Target table name (without schema prefix).
    if_exists
        Behaviour when the table exists: ``'fail'``, ``'replace'``, or ``'append'``.

    Returns
    -------
    Optional[int]
        Number of rows affected, as returned by :meth:`pandas.DataFrame.to_sql`.
    """
    engine = create_engine(resolve_db_url())
    schema = _resolve_env_setting('DB_ANALYTICS_SCHEMA', default='analytics')

    logger.info("Exporting %d rows to %s.%s", len(df), schema, table_name)
    result = df.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
    )
    logger.info("Export complete: %s.%s", schema, table_name)
    return result


def fetch_history_columns(engine, keep: Sequence[str],
                          pattern: str = HIST_COL_PATTERN) -> tuple[list[str], str]:
    """Discover the ``*_ago`` history columns + reference columns present in pml.pml_df.

    Returns the ordered column list and a pre-quoted ``SELECT`` column expression.
    """
    with engine.connect() as conn:
        hist_cols = pd.read_sql(
            text("""
                 SELECT column_name
                 FROM information_schema.columns
                 WHERE table_schema = 'pml'
                   AND table_name = 'pml_df'
                   AND (column_name ~ :pat OR column_name = ANY (:keep))
                 ORDER BY column_name
                 """),
            conn, params={'pat': pattern, 'keep': list(keep)},
        )['column_name'].tolist()
    col_sql = ', '.join(f'"{c}"' for c in hist_cols)
    return hist_cols, col_sql


# =============================================================================
# 1c. Artifact export — figures / tables / DataTrees -> KALMAN_PT_RESULTS_DIR
# =============================================================================
# Every figure shown through the :func:`_safe_show` funnel (which also serves
# :func:`_render_plotly` / :func:`_show_fig`), every DataFrame routed through
# :func:`display`, and the bulk data artifacts returned by :func:`main` are
# persisted under ``KALMAN_PT_RESULTS_DIR`` (env / environment_variables.txt,
# default ``pymc_kalman_filter_pt_results`` next to this script), in a
# **per-section subdirectory** derived from the artifact stem (see
# :data:`_EXPORT_SECTION_DIRS` / :func:`_export_dir_for`):
#
# * figures       -> PNG via plotly/kaleido or matplotlib; self-contained HTML
#                    fallback when kaleido / Chromium is unavailable
# * DataFrames    -> the curated bulk frames in :data:`_SQL_EXPORT_ARTIFACTS`
#                    become ``analytics.<stem>`` tables plus a generated
#                    ``{stem}.sql`` DDL file; every other frame (the
#                    auto-numbered ``display()`` snapshots) stays CSV
# * DataTrees     -> NetCDF (h5netcdf) + compact per-group JSON summary
#
# Exports are side effects in the same sense as rendering: every writer is
# guarded and logs a warning instead of raising, so a failed export can never
# abort the workflow (mirrors the ``_safe_show`` philosophy). The SQL sink
# additionally falls back to CSV when the database is unreachable, so an
# offline run never loses a frame.
_DEFAULT_RESULTS_DIRNAME = 'pymc_kalman_filter_pt_results'
_DEFAULT_EXPORT_PNG_WIDTH = 1400
_DEFAULT_EXPORT_PNG_HEIGHT = 780
_DEFAULT_EXPORT_PNG_SCALE = 2.0
_DEFAULT_EXPORT_DPI = 150
_EXPORT_SLUG_MAXLEN = 48

# Results-tree subdirectories, matched against an artifact stem by longest
# prefix. The ``export_section`` labels in :func:`main` and the hard-coded bulk
# stems in :func:`export_all_artifacts` both key off this one tuple. Two entries
# deliberately differ from their section label so the bulk stems land correctly:
# ``04_panel`` catches ``04_panel_frame`` (exported outside any section) and
# ``10b_risk`` catches ``10b_risk_analytics`` / ``_book`` / ``_summary``
# alongside the ``10b_risk_book_NN_*`` section stems.
_EXPORT_SECTION_DIRS: tuple[str, ...] = (
    '01_data', '02_eda', '03_features', '04_panel', '06_prior', '07_posterior',
    '08_ppc', '09_diagnostics', '09b_comparison', '10_screen', '10b_risk',
    '10c_analytics',
    '10k_universe', '11_single_isin', '11b_single_sv', '12_mingled',
    '12b_mingled_sv', '13_forest', '13b_further_views', '14_summary',
    '14b_recommendations', '00_misc',
)
_EXPORT_MISC_DIR = '00_misc'

# Bulk stems whose prefix does not match their section directory. ``10c`` is the
# only genuine mismatch: the section is ``10c_analytics`` but its bulk artifact
# is ``10c_kalman_results``, so a plain prefix scan would file it under 00_misc.
_EXPORT_DIR_ALIASES: dict[str, str] = {'10c_kalman': '10c_analytics'}

# Artifact stems persisted to the analytics schema instead of CSV. Everything
# else (the auto-numbered ``display()`` snapshots) stays CSV: their stems shift
# between runs as sections gain or lose ``display()`` calls, which would churn
# table names in the schema.
_SQL_EXPORT_ARTIFACTS: frozenset[str] = frozenset({
    '04_panel_frame',
    '09_diagnostics_01_table',
    '10_screen_results',
    '10_screen_mc_summary',
    '10b_risk_analytics',
    '10b_risk_book',
    '10c_kalman_results',
})
# ``10c_kalman_results`` is the same frame :func:`export_analytics` writes to
# ``analytics.kalman_filtered_price_targets``; its SQL write is skipped while
# that canonical write is active (see :func:`note_analytics_written`).
_SQL_REDUNDANT_WHEN_ANALYTICS_WRITTEN = '10c_kalman_results'
_DEFAULT_ANALYTICS_OWNER = 'postgres'
# DataTree groups that get an ``azs.summary`` statistics table in the JSON
# sidecar. ``posterior_predictive`` / ``observed_data`` are inventoried only:
# a second full ESS/R-hat sweep over the (chain, draw, isin, y_series) PPC
# tensor would roughly double the §9 diagnostics runtime for no screening value.
_DATATREE_SUMMARY_GROUPS = ('posterior', 'prior', 'predictions')


@dataclass
class _ExportState:
    """Mutable module-level artifact-export state (lazy singleton).

    Attributes
    ----------
    root
        Resolved output directory (``KALMAN_PT_RESULTS_DIR``).
    enabled
        Master switch; all export helpers no-op while ``False``.
    section
        Active filename prefix, managed by :func:`export_section`.
    counters
        Per-section running counter for auto-numbered artifact stems.
    png_ok
        Memo flag — flips ``False`` after the first kaleido/Chromium failure so
        every subsequent Plotly figure goes straight to the HTML fallback.
    sql_ok
        Memo flag — flips ``False`` after the first analytics-schema write
        failure so every subsequent curated frame goes straight to the CSV
        fallback instead of re-attempting a doomed connection.
    analytics_written
        Set by :func:`note_analytics_written` once
        ``analytics.kalman_filtered_price_targets`` has been written this run;
        suppresses the redundant ``10c_kalman_results`` table.
    cleaned
        Section directories already purged this run (``KALMAN_PT_CLEAN_RESULTS``).
    """

    root: Path
    enabled: bool = False
    section: str = _EXPORT_MISC_DIR
    counters: dict[str, int] = field(default_factory=dict)
    png_ok: bool = True
    sql_ok: bool = True
    analytics_written: bool = False
    cleaned: set[str] = field(default_factory=set)


_export_state_instance: Optional[_ExportState] = None


def get_export_state() -> _ExportState:
    """Return the lazy export-state singleton, resolving the output directory once.

    Resolution order: :attr:`KalmanRunConfig.results_dir` (so ``main(config=…)``
    actually redirects artifacts) → ``KALMAN_PT_RESULTS_DIR`` via
    :func:`_resolve_env_setting` (env → ``environment_variables.txt``) → the
    :data:`_DEFAULT_RESULTS_DIRNAME` default. A relative value is anchored at this
    script's directory, so the default lands at
    ``<project root>/pymc_kalman_filter_pt_results`` regardless of the CWD.
    """
    global _export_state_instance
    if _export_state_instance is None:
        raw: Optional[str] = None
        with contextlib.suppress(Exception):
            raw = get_run_config().results_dir
        if not raw:
            raw = _resolve_env_setting('KALMAN_PT_RESULTS_DIR',
                                       default=_DEFAULT_RESULTS_DIRNAME)
        root = Path(raw or _DEFAULT_RESULTS_DIRNAME)
        if not root.is_absolute():
            root = Path(__file__).resolve().parent / root
        _export_state_instance = _ExportState(root=root)
    return _export_state_instance


def note_analytics_written(written: bool = True) -> None:
    """Record that ``analytics.kalman_filtered_price_targets`` was written this run.

    :func:`export_analytics` calls this after its canonical write so the
    :data:`_SQL_REDUNDANT_WHEN_ANALYTICS_WRITTEN` artifact does not duplicate the
    same frame under a second table name.
    """
    get_export_state().analytics_written = bool(written)


def reset_export_state() -> None:
    """Reset the export-state singleton (re-reads ``KALMAN_PT_RESULTS_DIR``)."""
    global _export_state_instance
    _export_state_instance = None


def enable_artifact_export(enabled: bool = True) -> None:
    """Turn artifact export on (or off) for the current process."""
    get_export_state().enabled = enabled


def set_export_section(label: str) -> str:
    """Set the artifact-export section without a ``with`` block.

    :func:`export_section` is the right tool inside :func:`main`, where sections
    nest and unwind. A notebook has no enclosing block per cell, so this setter
    lets each cell declare which section its figures and tables belong to —
    otherwise every notebook artifact would land in ``00_misc``.

    Parameters
    ----------
    label
        Section prefix, e.g. ``'09_diagnostics'`` (see
        :data:`_EXPORT_SECTION_DIRS`).

    Returns
    -------
    str
        The previous section, so a caller can restore it.
    """
    state = get_export_state()
    prev = state.section
    state.section = label
    _clean_section_dir(label)
    return prev


@contextlib.contextmanager
def export_section(label: str):
    """Scope auto-generated artifact filenames to a workflow section.

    Sets the ``{section}`` prefix used by the auto-capture hooks in
    :func:`_safe_show` and :func:`display` while the block is active; restores
    the previous section on exit (nestable). Entering a section never touches
    the filesystem.

    Parameters
    ----------
    label
        Section prefix, e.g. ``'09_diagnostics'``.
    """
    state = get_export_state()
    prev = state.section
    state.section = label
    _clean_section_dir(label)
    try:
        yield
    finally:
        state.section = prev


def _slugify(raw: str) -> str:
    """Compress ``raw`` to a lowercase ``[a-z0-9_]`` filename slug."""
    parts = re.findall(r'[a-z0-9]+', str(raw).lower())
    return '_'.join(parts)[:_EXPORT_SLUG_MAXLEN].strip('_')


def _export_dir_for(stem: str) -> str:
    """Return the results subdirectory owning ``stem`` (longest-prefix match).

    Matching is done on the **stem** rather than the active
    :func:`export_section`, because :func:`export_all_artifacts` writes its bulk
    stems after every section context has exited.

    Parameters
    ----------
    stem
        Artifact filename stem, e.g. ``'10b_risk_book'`` or ``'02_eda_07'``.

    Returns
    -------
    str
        A member of :data:`_EXPORT_SECTION_DIRS`; :data:`_EXPORT_MISC_DIR` when
        no prefix matches.
    """
    for prefix, directory in _EXPORT_DIR_ALIASES.items():
        if stem == prefix or stem.startswith(f'{prefix}_'):
            return directory
    matches = [d for d in _EXPORT_SECTION_DIRS if stem == d or stem.startswith(f'{d}_')]
    if not matches:
        return _EXPORT_MISC_DIR
    return max(matches, key=len)


def _clean_section_dir(label: str) -> None:
    """Purge a section's subdirectory once per run when ``KALMAN_PT_CLEAN_RESULTS=1``.

    Per-section counters restart on every run while title slugs drift, so stale
    artifacts otherwise interleave with current ones under shifted indices. Now
    that sections are partitioned into their own directories this is a safe,
    surgical reset. Off by default so an interrupted run never destroys the
    previous run's output.
    """
    state = get_export_state()
    if not state.enabled:
        return
    if _resolve_env_setting('KALMAN_PT_CLEAN_RESULTS', default='0') != '1':
        return
    directory = _export_dir_for(label)
    if directory in state.cleaned:
        return
    state.cleaned.add(directory)
    target = state.root / directory
    if not target.is_dir():
        return
    try:
        shutil.rmtree(target)
        logger.info("Cleared stale artifacts in %s", target)
    except Exception as exc:  # pragma: no cover - best-effort hygiene
        logger.warning("Could not clear %s: %r", target, exc)


def _export_path(stem: str, ext: str) -> Path:
    """Return ``root/<section-dir>/stem.ext``, creating the directory on first use.

    The single filesystem touch point for every artifact writer, so the
    per-section tree is applied uniformly to PNG / HTML / CSV / SQL / JSON /
    NetCDF output without touching individual call sites.
    """
    state = get_export_state()
    directory = state.root / _export_dir_for(stem)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}.{ext}"


def _next_stem(label: Optional[str] = None, obj: object = None) -> str:
    """Return the next auto-numbered ``{section}_{NN}[_{slug}]`` filename stem.

    The slug comes from ``label`` when given, else (best-effort) from the Plotly
    figure title of ``obj`` — meaningful names for free on most panels without
    touching call sites.
    """
    state = get_export_state()
    n = state.counters.get(state.section, 0) + 1
    state.counters[state.section] = n
    slug = label
    if slug is None and obj is not None:
        try:
            slug = obj.layout.title.text  # type: ignore[attr-defined]
        except Exception:
            slug = None
    stem = f"{state.section}_{n:02d}"
    if slug:
        slug = _slugify(slug)
        if slug:
            stem = f"{stem}_{slug}"
    return stem


def _write_plotly_figure(fig: object, stem: str) -> None:
    """Write a Plotly figure as PNG, falling back to self-contained HTML.

    PNG needs ``kaleido`` + a Chromium binary (``plotly_get_chrome -y``). The
    first failure flips :attr:`_ExportState.png_ok` so later figures skip the
    doomed ``write_image`` attempt. ``_autosize_plotly`` clears ``layout.width``
    before display, so an explicit pixel width is always passed.
    """
    state = get_export_state()
    if state.png_ok:
        try:
            _apply_dark_template(fig)  # idempotent; guards the direct-call path
            fig.write_image(
                str(_export_path(stem, 'png')),
                width=int(fig.layout.width or _DEFAULT_EXPORT_PNG_WIDTH),
                height=int(fig.layout.height or _DEFAULT_EXPORT_PNG_HEIGHT),
                scale=_DEFAULT_EXPORT_PNG_SCALE,
            )
            return
        except Exception as exc:
            state.png_ok = False
            logger.warning(
                "Plotly PNG export unavailable (install kaleido + run "
                "`plotly_get_chrome -y`): %r -- falling back to HTML", exc)
    fig.write_html(str(_export_path(stem, 'html')),
                   include_plotlyjs=True, full_html=True)


def _export_figure(obj: object, *, label: Optional[str] = None) -> None:
    """Persist any displayed figure object under the active export section.

    Handles a raw Plotly figure or an ``arviz_plots`` PlotCollection (both via
    :func:`_plotly_figure_of`) and matplotlib figures. The matplotlib branch pins
    the figure's own facecolor explicitly rather than relying on the seaborn
    ``savefig.facecolor`` rc surviving — ``bbox_inches='tight'`` otherwise lets a
    light background leak into the exported PNG. No-op while export is disabled;
    never raises.
    """
    state = get_export_state()
    if not state.enabled or obj is None:
        return
    try:
        fig = _plotly_figure_of(obj)
        stem = _next_stem(label, fig if fig is not None else obj)
        if fig is not None:
            _write_plotly_figure(fig, stem)
        elif hasattr(obj, 'viz') and hasattr(obj, 'savefig'):
            # PlotCollection whose figure node could not be resolved directly.
            ext = 'png' if state.png_ok else 'html'
            obj.savefig(str(_export_path(stem, ext)))
        elif hasattr(obj, 'savefig'):  # matplotlib Figure
            obj.savefig(_export_path(stem, 'png'),
                        dpi=_DEFAULT_EXPORT_DPI, bbox_inches='tight',
                        facecolor=obj.get_facecolor(), edgecolor='none')
        else:
            logger.debug("Figure export skipped (unrecognised type %s)", type(obj))
    except Exception as exc:
        logger.warning("Figure export skipped (%s): %r", label or '?', exc)


def _analytics_schema() -> str:
    """Return the analytics schema name (``DB_ANALYTICS_SCHEMA``, default ``analytics``)."""
    return _resolve_env_setting('DB_ANALYTICS_SCHEMA', default='analytics') or 'analytics'


def _sql_column_type(series: pd.Series) -> str:
    """Return the PostgreSQL type ``series`` lands as under ``DataFrame.to_sql``.

    ``pandas.io.sql.get_schema`` would need a live connectable to emit
    PostgreSQL (it falls back to SQLite types otherwise), and DDL generation must
    work offline. Object columns are probed for ``datetime.date`` payloads, which
    is how the fiscal-calendar columns reach the analytics table as ``date``
    rather than ``text``.
    """
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'boolean'
    if pd.api.types.is_integer_dtype(dtype):
        return 'bigint'
    if pd.api.types.is_float_dtype(dtype):
        return 'double precision'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'timestamp'
    if pd.api.types.is_object_dtype(dtype):
        non_null = series.dropna()
        if len(non_null):
            sample = non_null.iloc[0]
            if isinstance(sample, datetime.datetime):
                return 'timestamp'
            if isinstance(sample, datetime.date):
                return 'date'
    return 'text'


def _sql_table_ddl(frame: pd.DataFrame, table: str, *,
                   schema: Optional[str] = None,
                   quote_table: bool = True,
                   comments: Optional[dict[str, str]] = None,
                   header: Optional[str] = None) -> str:
    """Render the ``CREATE TABLE`` DDL for ``frame`` in the analytics schema.

    Reproduces the layout of the hand-written DDL already in
    ``sql_scripts/analytics`` (tab-indented, name-aligned columns, trailing
    ``ALTER TABLE … OWNER TO`` stanza) so generated and checked-in files diff
    cleanly. Needs no database connection.

    Parameters
    ----------
    frame
        Frame whose schema is rendered.
    table
        Bare table name (no schema prefix).
    schema
        Target schema; defaults to :func:`_analytics_schema`.
    quote_table
        Double-quote the table name — required for the artifact stems, which
        start with a digit. ``False`` for plain identifiers such as
        ``kalman_filtered_price_targets``.
    comments
        Optional ``{column: comment}`` mapping rendered as ``COMMENT ON COLUMN``
        statements — used to persist the decimal-unit convention.
    header
        Optional comment block prepended to the file.

    Returns
    -------
    str
        The complete DDL script, newline-terminated.
    """
    schema = schema or _analytics_schema()
    owner = _resolve_env_setting('DB_ANALYTICS_OWNER',
                                 default=_DEFAULT_ANALYTICS_OWNER)
    ident = f'{schema}."{table}"' if quote_table else f'{schema}.{table}'
    names = [str(c) for c in frame.columns]
    pad = max((len(n) for n in names), default=0)

    parts: list[str] = []
    if header:
        parts.append(header.rstrip() + '\n\n')
    parts.append(f'CREATE TABLE {ident}\n(\n')
    parts.append(',\n'.join(
        f'\t{name:<{pad}} {_sql_column_type(frame[col])}'
        for name, col in zip(names, frame.columns)))
    parts.append('\n);\n')
    if owner:
        parts.append(f'\nALTER TABLE {ident}\n\tOWNER TO {owner};\n')
    for column, comment in (comments or {}).items():
        if column not in frame.columns:
            continue
        escaped = str(comment).replace("'", "''")
        parts.append(f'\nCOMMENT ON COLUMN {ident}."{column}"\n'
                     f"\tIS '{escaped}';\n")
    return ''.join(parts)


def _export_table(frame: pd.DataFrame, name: str) -> bool:
    """Persist ``frame`` as ``analytics."{name}"`` plus a generated ``{name}.sql``.

    The DDL file is written first because it needs no database connection: an
    offline run still yields a reviewable schema. The table write reuses
    :func:`export_to_analytics_db` (which resolves ``DB_URL`` /
    ``DB_ANALYTICS_SCHEMA`` through the ``environment_variables.txt`` fallback);
    ``KALMAN_PT_SQL_EXPORT=0`` skips it entirely.

    A non-``RangeIndex`` index is materialised as a real column first — the
    ``to_sql(index=False)`` used by :func:`export_to_analytics_db` would
    otherwise silently drop the ``azs.summary`` variable-name index.

    Parameters
    ----------
    frame
        Frame to persist.
    name
        Artifact stem, used verbatim as the table name.

    Returns
    -------
    bool
        ``True`` when the analytics table was written; ``False`` when the caller
        should fall back to CSV (export disabled, DB unreachable, or a prior
        failure already flipped :attr:`_ExportState.sql_ok`).
    """
    state = get_export_state()
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()

    with contextlib.suppress(Exception):
        _export_path(name, 'sql').write_text(
            _sql_table_ddl(frame, name), encoding='utf-8')

    if _resolve_env_setting('KALMAN_PT_SQL_EXPORT', default='1') != '1':
        logger.info("KALMAN_PT_SQL_EXPORT=0 -> %s written as CSV, DDL only", name)
        return False
    if name == _SQL_REDUNDANT_WHEN_ANALYTICS_WRITTEN and state.analytics_written:
        logger.info(
            "%s duplicates analytics.kalman_filtered_price_targets (already "
            "written this run) -- table write skipped, DDL emitted", name)
        return True
    if not state.sql_ok:
        return False
    try:
        export_to_analytics_db(frame, name, if_exists='replace')
        logger.info("Exported table %s.%s (%d rows)",
                    _analytics_schema(), name, len(frame))
        return True
    except Exception as exc:
        state.sql_ok = False
        logger.warning(
            "Analytics-schema export unavailable (%s): %r -- falling back to CSV "
            "for this and every later curated frame", name, exc)
        return False


def _export_dataframe(df: object, name: str, *,
                      index: Optional[bool] = None,
                      also_json: bool = False) -> None:
    """Persist a DataFrame / Series as an analytics table or as ``{name}.csv``.

    Stems listed in :data:`_SQL_EXPORT_ARTIFACTS` — the curated bulk frames — are
    written to ``analytics."{name}"`` alongside a generated ``{name}.sql`` DDL
    file, mirroring how ``analytics.kalman_filtered_price_targets`` is produced.
    Every other frame (the auto-numbered ``display()`` snapshots, whose stems
    shift between runs) is written as ``{name}.csv``. A curated frame falls back
    to CSV when the database is unreachable, so no frame is ever lost.

    Parameters
    ----------
    df
        Frame or Series to export (Series are exported via ``to_frame()``).
    name
        Full filename stem (caller supplies any section prefix); also the
        analytics table name for curated frames.
    index
        Include the index column in the CSV path; ``None`` auto-includes it
        whenever the index is not a bare RangeIndex (e.g. ``azs.summary``
        variable-name indexes). Ignored on the SQL path, which always
        materialises a non-RangeIndex index as a column.
    also_json
        Additionally write JSON records (``default_handler=str`` guards the
        Timestamp columns).
    """
    state = get_export_state()
    if not state.enabled or df is None:
        return
    try:
        frame = df.to_frame() if isinstance(df, pd.Series) else df
        if index is None:
            index = not isinstance(frame.index, pd.RangeIndex)
        if name in _SQL_EXPORT_ARTIFACTS:
            if not _export_table(frame, name):
                frame.to_csv(_export_path(name, 'csv'), index=index)
        else:
            frame.to_csv(_export_path(name, 'csv'), index=index)
        if also_json:
            frame.to_json(_export_path(name, 'json'), orient='records',
                          date_format='iso', default_handler=str, indent=2)
        logger.info("Exported dataframe %s (%d rows)", name, len(frame))
    except Exception as exc:
        logger.warning("DataFrame export skipped (%s): %r", name, exc)


def _export_datatree(dt: object, name: str) -> None:
    """Write a DataTree as ``{name}.nc`` plus a compact ``{name}_summary.json``.

    The NetCDF file (h5netcdf engine) is the full-fidelity, reload-with-arviz
    artifact. The JSON sidecar carries a per-group ``data_vars``/``sizes``
    inventory and, for the groups in :data:`_DATATREE_SUMMARY_GROUPS`, the
    ``azs.summary`` statistics table (``kind='all'`` when the group has
    >= 2 chains and >= 4 draws, else a stats-only summary — r_hat / ESS are
    between-chain diagnostics and would be NaN for the 1-chain ``prior``
    group; the chosen kind is recorded as ``summary_kind``). Each part is
    individually guarded.
    """
    state = get_export_state()
    if not state.enabled or dt is None:
        return
    try:
        path = _export_path(name, 'nc')
        dt.to_netcdf(path, engine='h5netcdf')
        logger.info("Exported DataTree %s (%.1f MB)", name,
                    path.stat().st_size / 1e6)
    except Exception as exc:
        logger.warning("DataTree NetCDF export skipped (%s): %r", name, exc)
    summary: dict[str, Any] = {'groups': {}}
    try:
        for group, node in (getattr(dt, 'children', {}) or {}).items():
            ds = getattr(node, 'ds', node)
            entry: dict[str, Any] = {
                'data_vars': [str(v) for v in getattr(ds, 'data_vars', [])],
                'sizes': {str(k): int(v)
                          for k, v in dict(getattr(ds, 'sizes', {})).items()},
            }
            if group in _DATATREE_SUMMARY_GROUPS and {'chain', 'draw'} <= set(entry['sizes']):
                # r_hat / between-chain ESS need >= 2 chains and >= 4 draws;
                # prior-predictive groups always carry chain=1, so ask for a
                # stats-only summary there instead of tripping the arviz_stats
                # shape validator (which logs and returns NaN diagnostics).
                kind = ('all' if entry['sizes']['chain'] >= 2 and entry['sizes']['draw'] >= 4
                        else 'stats')
                try:
                    stats = azs.summary(dt, group=str(group), kind=kind)
                    entry['summary_kind'] = kind
                    entry['summary'] = stats.reset_index().to_dict(orient='records')
                except Exception as exc:
                    logger.debug("azs.summary skipped for %s/%s: %r", name, group, exc)
            summary['groups'][str(group)] = entry
        _export_json(summary, f"{name}_summary")
    except Exception as exc:
        logger.warning("DataTree summary export skipped (%s): %r", name, exc)


def _export_json(payload: Optional[dict], name: str) -> None:
    """Write ``payload`` as pretty-printed ``{name}.json`` (``default=str``)."""
    state = get_export_state()
    if not state.enabled or payload is None:
        return
    try:
        with open(_export_path(name, 'json'), 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Exported json %s", name)
    except Exception as exc:
        logger.warning("JSON export skipped (%s): %r", name, exc)


def _export_context(ctx: Optional[dict], name: str) -> None:
    """Export a section context dict by underlying value type.

    DataFrames/Series -> ``{name}_{key}.csv``; DataTrees -> NetCDF + summary;
    Datasets -> NetCDF; small ndarrays / DatetimeIndexes / scalars are collected
    into ``{name}_meta.json`` (large arrays become dtype/shape stubs, model
    objects are recorded by type only). ``None`` and the partial early-exit
    dicts (e.g. §12's) are handled — every key is optional.
    """
    state = get_export_state()
    if not state.enabled or not ctx:
        return
    meta: dict[str, Any] = {}
    for key, val in ctx.items():
        try:
            if isinstance(val, (pd.DataFrame, pd.Series)):
                _export_dataframe(val, f"{name}_{key}")
            elif isinstance(val, xr.DataTree):
                _export_datatree(val, f"{name}_{key}")
            elif isinstance(val, xr.Dataset):
                val.to_netcdf(_export_path(f"{name}_{key}", 'nc'), engine='h5netcdf')
            elif isinstance(val, xr.DataArray):
                logger.debug("Context %s.%s: raw DataArray skipped", name, key)
            elif isinstance(val, np.ndarray):
                meta[key] = (val.tolist() if val.size <= 10_000
                             else {'dtype': str(val.dtype), 'shape': list(val.shape)})
            elif isinstance(val, pd.DatetimeIndex):
                meta[key] = [str(v) for v in val]
            elif isinstance(val, (str, int, float, bool, type(None), list, dict)):
                meta[key] = val
            else:
                meta[key] = f"<{type(val).__module__}.{type(val).__qualname__}>"
        except Exception as exc:
            logger.warning("Context export skipped (%s.%s): %r", name, key, exc)
    _export_json(meta, f"{name}_meta")


def export_all_artifacts(artifacts: dict, *, results_dir: Optional[str] = None) -> None:
    """Export the :func:`main` artifact dict to ``KALMAN_PT_RESULTS_DIR``.

    Persists the durable data artifacts (the render-time figure/table snapshots
    are captured separately by the :func:`_safe_show` / :func:`display` hooks),
    each into its per-section subdirectory (:func:`_export_dir_for`):

    * ``prior_idata`` / ``idata`` DataTrees -> NetCDF + JSON summary
    * ``panel.frame``, ``results``, ``kalman_results``, ``screen.mc_summary``,
      ``risk_book.analytics`` / ``.book`` -> ``analytics."<stem>"`` tables plus a
      generated ``<stem>.sql`` DDL file (:data:`_SQL_EXPORT_ARTIFACTS`);
      ``results`` also JSON. These frames fall back to CSV when the analytics
      schema is unreachable.
    * ``risk_book.summary`` and the ``universe_fit`` scalars -> JSON
    * ``universe_fit.idata`` / ``.pred.predictions`` -> NetCDF

    The raw ``screen.eu`` / ``screen.ept`` posterior draws (~200 MB each) are
    skipped unless ``KALMAN_PT_EXPORT_DRAWS=1``, in which case they are bundled
    into ``10_screen_posterior_draws.nc``; their decision content is already in
    ``analytics."10_screen_results"`` and the posterior NetCDF.

    Callable on ``main(export_results=False)`` output — export is force-enabled
    for the duration of this call only. Missing keys / ``None`` values are
    skipped silently; individual failures log warnings and never raise.

    Parameters
    ----------
    artifacts
        The dict returned by :func:`main`.
    results_dir
        Optional override of the resolved output directory for this call.
    """
    state = get_export_state()
    prev_root, prev_enabled = state.root, state.enabled
    if results_dir is not None:
        state.root = Path(results_dir)
    state.enabled = True
    try:
        _export_datatree(artifacts.get('prior_idata'), '06_prior_idata')
        _export_datatree(artifacts.get('idata'), '07_posterior_idata')

        panel = artifacts.get('panel')
        if getattr(panel, 'frame', None) is not None:
            _export_dataframe(panel.frame, '04_panel_frame')

        _export_dataframe(artifacts.get('results'), '10_screen_results',
                          also_json=True)
        screen = artifacts.get('screen')
        if screen is not None:
            _export_dataframe(getattr(screen, 'mc_summary', None), '10_screen_mc_summary')
            if _resolve_env_setting('KALMAN_PT_EXPORT_DRAWS', default='0') == '1':
                try:
                    xr.Dataset({'expected_upside': screen.eu,
                                'expected_pt': screen.ept}).to_netcdf(
                        _export_path('10_screen_posterior_draws', 'nc'),
                        engine='h5netcdf')
                except Exception as exc:
                    logger.warning("Posterior-draw export skipped: %r", exc)
            else:
                logger.info(
                    "Raw eu/ept posterior draw export is off by default "
                    "(~200 MB per array); set KALMAN_PT_EXPORT_DRAWS=1 to bundle "
                    "them into 10_screen_posterior_draws.nc — screen decision "
                    "content is already in analytics.\"10_screen_results\" and "
                    "the posterior NetCDF.")

        risk_book = artifacts.get('risk_book')
        if risk_book is not None:
            _export_dataframe(getattr(risk_book, 'analytics', None), '10b_risk_analytics')
            _export_dataframe(getattr(risk_book, 'book', None), '10b_risk_book')
            _export_json(dict(getattr(risk_book, 'summary', {}) or {}), '10b_risk_summary')

        _export_dataframe(artifacts.get('kalman_results'), '10c_kalman_results')

        fit = artifacts.get('universe_fit')
        if fit is not None:
            _export_datatree(getattr(fit, 'idata', None), '10k_universe_idata')
            preds = getattr(getattr(fit, 'pred', None), 'predictions', None)
            if preds is not None:
                try:
                    preds.to_netcdf(_export_path('10k_universe_predictions', 'nc'),
                                    engine='h5netcdf')
                except Exception as exc:
                    logger.warning("Universe-prediction export skipped: %r", exc)
            fit_meta: dict[str, Any] = {
                'horizons_days': getattr(fit, 'horizons_days', None),
                'fiscal_dates': [str(d) for d in getattr(fit, 'fiscal_dates', []) or []],
                'labels': getattr(fit, 'labels', None),
                'last_obs': str(getattr(fit, 'last_obs', None)),
                'last_price': getattr(fit, 'last_price', None),
            }
            with contextlib.suppress(Exception):
                fit_meta['fit_kind'] = fit.fit_kind
                fit_meta['n_divergences'] = int(fit.n_divergences)
            _export_json(fit_meta, '10k_universe_fit_meta')

        logger.info("Artifact export complete -> %s", state.root)
    except Exception as exc:  # pragma: no cover - belt and braces
        logger.warning("export_all_artifacts aborted early: %r", exc)
    finally:
        state.root, state.enabled = prev_root, prev_enabled


def migrate_results_layout(root: Optional[str] = None, *,
                           dry_run: bool = True) -> dict[str, str]:
    """Move flat legacy artifacts into the per-section subdirectory tree.

    Historic runs wrote every artifact into the top level of
    ``KALMAN_PT_RESULTS_DIR``. This one-off migration re-files them using the
    same :func:`_export_dir_for` rule the writers now apply, so the tree stays
    consistent with what a fresh run produces. Idempotent: files already inside a
    subdirectory are left alone, and a second invocation reports nothing to do.

    Parameters
    ----------
    root
        Results directory; defaults to the resolved
        :attr:`_ExportState.root`.
    dry_run
        Report the planned moves without touching the filesystem (default).

    Returns
    -------
    dict[str, str]
        ``{filename: destination-subdirectory}`` for every file moved (or that
        would be moved under ``dry_run``).
    """
    base = Path(root) if root is not None else get_export_state().root
    if not base.is_dir():
        logger.warning("Results directory does not exist: %s", base)
        return {}

    planned: dict[str, str] = {}
    for path in sorted(base.iterdir()):
        if not path.is_file():
            continue
        planned[path.name] = _export_dir_for(path.stem)

    for filename, subdir in planned.items():
        source = base / filename
        target_dir = base / subdir
        target = target_dir / filename
        if dry_run:
            logger.info("[dry-run] %s -> %s/", filename, subdir)
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(source), str(target))
        except Exception as exc:
            logger.warning("Could not move %s -> %s/: %r", filename, subdir, exc)

    if not planned:
        logger.info("Results layout already migrated: %s", base)
    else:
        logger.info("%s %d artifact(s) into %d subdirectories under %s",
                    'Would move' if dry_run else 'Moved',
                    len(planned), len(set(planned.values())), base)
    return planned


# =============================================================================
# Data loading + feature-role resolution
# =============================================================================
def load_kalman_df(engine, config: Optional[KalmanRunConfig] = None) -> pd.DataFrame:
    """Load the cross-sectional ``pml.mv_pymc_kalman_pt`` snapshot (one row per ISIN)."""
    with engine.connect() as conn:
        df = pd.read_sql(text(kalman_df_query(config)), conn)
    logger.info("Loaded kalman_df: %s", df.shape)
    return df


def load_feature_catalogue(engine) -> pd.DataFrame:
    """Load the ``kalman_pt`` rows of ``pml.vw_pymc_feature_catalogue``."""
    with engine.connect() as conn:
        cat = pd.read_sql(text(FEATURE_CATALOGUE_QUERY), conn)
    logger.info("Loaded feature_catalogue: %s", cat.shape)
    return cat


@dataclass
class FeatureRoles:
    """Column groups resolved by ``pymc_role`` from the feature catalogue + MV schema."""

    predictor_cols: list[str] = field(default_factory=list)
    coord_cols: list[str] = field(default_factory=list)
    response_cols: list[str] = field(default_factory=list)
    classification_coords: list[str] = field(default_factory=list)
    fiscal_calendar_cols: list[str] = field(default_factory=list)
    day_count_cols: list[str] = field(default_factory=list)


def resolve_feature_roles(kalman_df: pd.DataFrame,
                          feature_catalogue: pd.DataFrame) -> FeatureRoles:
    """Resolve column groups by ``pymc_role`` (catalogue SSOT, MV-schema fallback).

    ``predictor_cols`` mirrors ``KalmanFilterPriceTarget._resolve_kalman_feature_aliases``
    (mutable_predictor aliases for ``model_target='kalman_pt'``); ``KNOWN_FEATURES`` is a
    resilience fallback for MV columns absent from the catalogue snapshot.
    """
    if 'model_target' in feature_catalogue.columns:
        if not (feature_catalogue['model_target'] == 'kalman_pt').all():
            warnings.warn("feature_catalogue is not fully filtered to model_target='kalman_pt'.")

    catalogue = feature_catalogue.copy()
    catalogue['present'] = catalogue['feature_alias'].isin(kalman_df.columns)

    role_summary = (
        catalogue.groupby(['pymc_role', 'feature_role'])['present']
        .agg(n_columns='size', n_present='sum')
        .reset_index()
    )
    display(role_summary)

    present = catalogue.loc[catalogue['present']]
    predictor_cols = present.loc[present['pymc_role'] == 'mutable_predictor', 'feature_alias'].tolist()
    coord_cols = present.loc[present['pymc_role'] == 'coord', 'feature_alias'].tolist()
    response_cols = present.loc[present['pymc_role'].isin(['response', 'observed']), 'feature_alias'].tolist()

    # Fallback must not re-promote columns the catalogue now assigns to the
    # observed/response side (e.g. feat_pt_noise_sigma, feat_total_return_*).
    for col in KNOWN_FEATURES:
        if (col in kalman_df.columns and col not in predictor_cols
                and col not in response_cols):
            predictor_cols.append(col)
    if 'observed_pt' in kalman_df.columns and 'observed_pt' not in response_cols:
        response_cols.append('observed_pt')

    classification_coords = [c for c in CLASSIFICATION_COORDS_ALL if c in kalman_df.columns]
    fiscal_calendar_cols = [c for c in FISCAL_CALENDAR_COLS_ALL if c in kalman_df.columns]
    day_count_cols = [c for c in DAY_COUNT_COLS_ALL if c in kalman_df.columns]

    # Keep only non-date coords as categorical-effect candidates; fall back to the
    # curated classification list when the catalogue exposes no plain coords.
    coord_cols = [c for c in coord_cols if c not in fiscal_calendar_cols] or classification_coords.copy()

    roles = FeatureRoles(
        predictor_cols=predictor_cols, coord_cols=coord_cols, response_cols=response_cols,
        classification_coords=classification_coords, fiscal_calendar_cols=fiscal_calendar_cols,
        day_count_cols=day_count_cols,
    )
    print(f'#predictors     : {len(predictor_cols)} -> {predictor_cols}')
    print(f'#response       : {len(response_cols)} -> {response_cols}')
    print(f'#coords         : {len(coord_cols)} -> {coord_cols}')
    print(f'#classification : {len(classification_coords)} -> {classification_coords}')
    print(f'#fiscal-calendar: {len(fiscal_calendar_cols)} -> {fiscal_calendar_cols}')
    print(f'#day-count      : {len(day_count_cols)} -> {day_count_cols}')
    return roles


# =============================================================================
# 2. Exploratory Data Analysis (EDA)
# =============================================================================
# Interactive (Plotly) EDA helpers. Each is side-effecting (mirrors the
# ``plt.show()`` / ``pc.show()`` convention) and aligned to the fused-panel drivers
# the model actually consumes (build_fused_kalman_pt_model): the systematic-risk
# ``feat_avg_beta`` penalty and the ``feat_mcap_country_r`` size discount on
# ``risk_adj_return``, plus the new short-horizon momentum ``feat_one_day_return``.
# When plotly is unavailable, ``run_eda`` falls back to the matplotlib panels.

# (column, hover/axis label) of the cross-sectional drivers the §2.4e facets scan.
# Only the columns present in the snapshot are plotted.
_EDA_DRIVER_SPECS: tuple[tuple[str, str], ...] = (
    ('feat_avg_beta', 'avg beta — systematic-risk penalty'),
    ('feat_mcap_country_r', 'mcap country rank — size discount'),
    ('feat_rel_volume', 'relative volume — liquidity tilt'),
    ('feat_one_day_return', 'one-day return — short-horizon momentum'),
    ('feat_price_chg_pct_3m', 'three-month return — mid-horizon momentum'),
    ('noise_cv', 'consensus noise CV — sigma_obs widener'),
    ('feat_vol_drift', 'realized-vol drift — sigma_obs widener'),
    ('feat_total_return_ytd', 'YTD total return — momentum'),
    # Catalogue mutable_predictors integrated into the drift design matrix.
    ('feat_analyst_bullish_pct', 'bullish rating share — analyst sentiment'),
    ('feat_analyst_conviction', 'net rating conviction — analyst sentiment'),
    ('feat_pt_achievement_1y', '1y PT achievement — target credibility'),
    ('feat_pt_accuracy_1y', '1y PT abs error — target credibility'),
    ('feat_mcap_trend_1y', 'mcap 1y trend — size re-rating'),
    ('feat_median_piotroski_f_score', 'median Piotroski F — fundamental quality'),
)


def _sector_grouped(df: pd.DataFrame, *, max_sectors: int = 8) -> pd.DataFrame:
    """Return ``df`` with a capped ``sector_grp`` column (top-N sectors + ``Other``)."""
    g = df.copy()
    sec = g.get('sector', pd.Series('Unknown', index=g.index)).fillna('Unknown').astype(str)
    top = sec.value_counts().head(max_sectors).index.tolist()
    g['sector_grp'] = np.where(sec.isin(top), sec, 'Other')
    return g


def _eda_upside_frame(kalman_df: pd.DataFrame) -> pd.DataFrame:
    """Filtered ISIN frame carrying the ``upside_pct`` response and the cv widener.

    Implied upside is ``observed_pt / last_price - 1`` (the cross-sectional response the
    fused panel reconstructs), clipped to [-100 %, +500 %]. ``noise_cv`` is the
    consensus-dispersion observation-noise widener (``feat_pt_noise_sigma / last_price``).
    """
    d = kalman_df[(kalman_df['observed_pt'] > 0) & (kalman_df['last_price'] > 0)].copy()
    if d.empty:
        return d
    d['upside_pct'] = ((d['observed_pt'] / d['last_price'] - 1.0) * 100.0).clip(-100, 500)
    if 'feat_pt_noise_sigma' in d.columns:
        d['noise_cv'] = (d['feat_pt_noise_sigma'].astype('float64')
                         / d['last_price'].clip(lower=1e-9)).clip(0, 1)
    return d


def _plot_upside_vs_drivers_plotly(kalman_df: pd.DataFrame) -> None:
    """Interactive implied-upside vs fused-panel risk/return drivers (faceted scatter).

    Refactors the former static §2.4e scatter into a hoverable Plotly view spanning the
    drivers ``build_fused_kalman_pt_model`` actually consumes — the systematic-risk
    ``feat_avg_beta`` tilt and the ``feat_mcap_country_r`` size discount on
    ``risk_adj_return``, the consensus-noise / volatility ``sigma_obs`` wideners, and the
    new short-horizon momentum ``feat_one_day_return``. Each facet's x-axis is independent
    and winsorised to the 1/99 pct for readability; hover surfaces ticker / name.
    """
    d = _eda_upside_frame(kalman_df)
    if d.empty:
        return
    drivers = [(c, lab) for c, lab in _EDA_DRIVER_SPECS if c in d.columns]
    if not drivers:
        return
    g = _sector_grouped(d)
    id_cols = [c for c in ('ticker', 'name') if c in g.columns]
    frames = []
    for col, lab in drivers:
        sub = g[[col, 'upside_pct', 'sector_grp', *id_cols]].copy()
        lo, hi = sub[col].astype('float64').quantile([0.01, 0.99])
        sub[col] = sub[col].astype('float64').clip(lo, hi)
        sub = sub.rename(columns={col: 'driver_value'})
        sub['driver'] = lab
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    hover = {c: True for c in id_cols}
    hover.update({'driver': False, 'sector_grp': False})
    # Per-driver Spearman ρ (signal strength), annotated on each facet title so
    # the reader sees strength without cross-referencing the §2.4g momentum bar.
    rho_by_label: dict[str, float] = {}
    for col, lab in drivers:
        _pair = d[[col, 'upside_pct']].astype('float64').dropna()
        rho_by_label[lab] = (float(_pair[col].corr(_pair['upside_pct'],
                                                   method='spearman'))
                             if len(_pair) > 10 else float('nan'))
    _trend_kw = ({'trendline': 'ols', 'trendline_color_override': C_REF}
                 if _ilu.find_spec('statsmodels') is not None else {})
    fig = px.scatter(
        long, x='driver_value', y='upside_pct', color='sector_grp',
        facet_col='driver', facet_col_wrap=3, opacity=0.55, hover_data=hover,
        category_orders={'driver': [lab for _, lab in drivers]},
        labels={'sector_grp': 'sector', 'driver_value': 'value',
                'upside_pct': 'implied upside (%)'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        **_trend_kw,
    )
    fig.update_xaxes(matches=None, showticklabels=True)

    def _facet_title(a):
        lab = a.text.split('=', 1)[-1]
        rho = rho_by_label.get(lab)
        suffix = f'  (ρ={rho:.2f})' if rho is not None and np.isfinite(rho) else ''
        a.update(text=f'{lab}{suffix}', font_size=10)

    fig.for_each_annotation(_facet_title)
    _add_ref_line(fig, y=0, kind='zero')
    fig.update_layout(
        title='Implied upside vs fused-panel risk / return drivers (interactive)',
        legend_title_text='sector',
    )
    _render_plotly(fig, height=H_TALL)


def _plot_upside_vs_signals_mpl(kalman_df: pd.DataFrame) -> None:
    """Static seaborn fallback for §2.4e (upside vs consensus dispersion / vol drift)."""
    _g = kalman_df[(kalman_df['observed_pt'] > 0) & (kalman_df['last_price'] > 0)].copy()
    _g['upside_pct'] = ((_g['observed_pt'] / _g['last_price'] - 1.0) * 100.0).clip(-100, 500)
    _g['sector'] = _g.get('sector', pd.Series('Unknown', index=_g.index)).fillna('Unknown')
    if 'feat_pt_noise_sigma' in _g.columns:
        _g['noise_cv'] = (_g['feat_pt_noise_sigma'].astype('float64')
                          / _g['last_price'].clip(lower=1e-9)).clip(0, 1)
    _g['vol_drift'] = _g.get('feat_vol_drift', pd.Series(np.nan, index=_g.index)).astype('float64')

    fig, axes = plt.subplots(1, 2, figsize=_mpl_figsize(0.42), sharey=True,
                             layout='constrained')
    _top_sectors = _g['sector'].value_counts().head(9).index.tolist()
    _gs = _g[_g['sector'].isin(_top_sectors)]
    _pal = dict(zip(_top_sectors, sns.color_palette('Set2', len(_top_sectors))))
    if 'noise_cv' in _gs.columns:
        for sec, sub in _gs.groupby('sector'):
            axes[0].scatter(sub['noise_cv'], sub['upside_pct'], s=8, alpha=0.4,
                            color=_pal[sec], label=sec)
        axes[0].set_xlabel('consensus noise CV  (feat_pt_noise_sigma / last_price)')
        axes[0].set_ylabel('implied upside (%)')
        axes[0].set_title('Upside vs consensus dispersion')
        axes[0].set_xlim(0, 0.5)
    for sec, sub in _gs.groupby('sector'):
        axes[1].scatter(sub['vol_drift'], sub['upside_pct'], s=8, alpha=0.4,
                        color=_pal[sec], label=sec)
    axes[1].set_xlabel('realized-vol drift  (feat_vol_drift, winsorised [-1, 1])')
    axes[1].set_title('Upside vs realized-vol drift')
    axes[1].set_xlim(-1, 1)
    axes[1].legend(fontsize=7, framealpha=0.25, title='sector', title_fontsize=8,
                   loc='upper right')
    _export_figure(fig, label='upside_vs_signals')
    plt.show()


def _plot_feature_corr_heatmap_plotly(kalman_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Interactive Spearman-correlation heatmap of the feat_* drift / noise blocks."""
    cols = [c for c in feature_cols if c in kalman_df.columns and kalman_df[c].notna().sum() > 5]
    if len(cols) < 2:
        return
    corr = kalman_df[cols].astype('float64').corr(method='spearman')
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=cols, y=cols, zmin=-1, zmax=1, zmid=0,
        colorscale='RdBu', reversescale=True, colorbar=dict(title='Spearman rho'),
        text=corr.round(2).to_numpy(), texttemplate='%{text}', textfont=dict(size=8),
        hovertemplate='%{y}<br>vs %{x}<br>rho = %{z:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title='feat_* correlation — drift / momentum vs noise-widener blocks (interactive)',
        yaxis_autorange='reversed',
    )
    _render_plotly(fig, height=max(560, 24 * len(cols) + 180))


def _momentum_signal_table(kalman_df: pd.DataFrame,
                           drift_cols: list[str]) -> Optional[pd.DataFrame]:
    """Spearman rho of each momentum / drift feature with implied upside.

    Scans the drift term structure — the new short-horizon ``feat_one_day_return``, the
    analyst-trail drifts, and the YTD -> multi-year realised-return block — to quantify
    where each horizon's marginal signal sits. Returns a tidy, rho-sorted frame
    (``feature``, ``spearman_rho``, ``n``) or ``None`` when no column qualifies.
    """
    d = _eda_upside_frame(kalman_df)
    if d.empty:
        return None
    rows = []
    for c in drift_cols:
        if c not in d.columns:
            continue
        sub = d[[c, 'upside_pct']].astype('float64').dropna()
        if len(sub) <= 5:
            continue
        rows.append({
            'feature': c,
            'spearman_rho': round(float(sub[c].corr(sub['upside_pct'], method='spearman')), 4),
            'n': int(len(sub)),
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values('spearman_rho').reset_index(drop=True)


def _plot_momentum_signal_plotly(corr_df: pd.DataFrame) -> None:
    """Interactive horizontal rho-bar of the momentum table (new feature highlighted)."""
    colors = np.where(corr_df['feature'] == 'feat_one_day_return', C_HIGHLIGHT, C_POSTERIOR)
    fig = go.Figure(go.Bar(
        x=corr_df['spearman_rho'], y=corr_df['feature'], orientation='h',
        marker_color=colors, customdata=corr_df[['n']].to_numpy(),
        hovertemplate='%{y}<br>rho = %{x:.3f}<br>n = %{customdata[0]}<extra></extra>',
    ))
    _add_ref_line(fig, x=0, kind='zero')
    fig.update_layout(
        title='Momentum / drift signal vs implied upside '
              '(Spearman rho; feat_one_day_return highlighted)',
        xaxis_title='Spearman rho with implied upside (%)', yaxis_title='',
        showlegend=False,
    )
    _render_plotly(fig, height=max(320, 26 * len(corr_df) + 140))


def run_eda(kalman_df: pd.DataFrame, roles: FeatureRoles) -> None:
    """Exploratory views over ``kalman_df`` (source: ``pml.mv_pymc_kalman_pt``).

    Treats the ``feat_*`` columns through the lens of their state-space role: drift
    features -> the state-transition mean (``beta`` slopes), noise wideners -> the
    measurement-noise scale ``sigma_obs``. All plots are side-effecting only.
    """
    # 2.1 Shape, dtype, missingness overview.
    eda_overview = pd.DataFrame({
        'dtype': kalman_df.dtypes.astype(str),
        'n_missing': kalman_df.isna().sum(),
        'pct_missing': (kalman_df.isna().mean() * 100).round(1),
        'n_unique': kalman_df.nunique(),
    })
    print(f'kalman_df shape: {kalman_df.shape}')
    display(eda_overview.sort_values('pct_missing', ascending=False).head(30))

    # 2.2 Expected upside by industry — arviz_plots ridge.
    _d = kalman_df[(kalman_df['observed_pt'] > 0) & (kalman_df['last_price'] > 0)].copy()
    _d['upside_pct'] = (_d['observed_pt'] / _d['last_price'] - 1.0) * 100.0
    _d['upside_pct'] = _d['upside_pct'].clip(-100, 500)
    _d['industry'] = (_d['industry'] if 'industry' in _d.columns
                      else pd.Series('Unknown', index=_d.index))
    _d['industry'] = _d['industry'].fillna('Unknown').astype(str)
    _counts = _d['industry'].value_counts()
    _keep = _counts[_counts >= 5].index.tolist()
    _d = _d[_d['industry'].isin(_keep)]
    # Sort industries by median upside so the ridge reads as a ranking.
    _industries = (_d.groupby('industry')['upside_pct'].median()
                   .sort_values().index.tolist())
    if _industries:
        _max_n = int(_d['industry'].value_counts().max())
        _arr = np.full((len(_industries), _max_n), np.nan)
        for i, ind in enumerate(_industries):
            vals = _d.loc[_d['industry'] == ind, 'upside_pct'].to_numpy()
            _arr[i, :len(vals)] = vals
        _ds_ridge = xr.Dataset(
            {'implied_upside_pct': (('industry', 'sample'), _arr)},
            coords={'industry': _industries},
        )
        pc = azp.plot_ridge(_ds_ridge, var_names=['implied_upside_pct'],
                            sample_dims=['sample'], combined=True,
                            figure_kwargs=_azp_figure_kwargs(
                                _forest_height_px(len(_industries), per_row=30)))
        pc.add_title('Implied upside (%) by industry — consensus observed_pt vs '
                     'last_price (sorted by median)')
        with contextlib.suppress(Exception):
            _fx = _plotly_figure_of(pc)
            if _fx is not None:
                _add_ref_line(_fx, x=0, kind='zero')
                _fx.update_xaxes(title_text='implied upside (%)', ticksuffix='%')
        _safe_show(pc)
        display(_d['upside_pct'].describe())

    # 2.3 Classification-coord cardinality.
    card = {c: kalman_df[c].nunique() for c in roles.classification_coords}
    display(pd.Series(card).sort_values(ascending=False))

    # 2.4a Distributional summary of the feat_* columns via arviz-stats.
    eda_drift = [c for c in ('feat_pt_drift', 'feat_price_drift','feat_coverage_drift', 'feat_pt_noise_drift',
                             'feat_one_day_return','feat_price_chg_pct_3m',
                             'feat_total_return_ytd', 'feat_total_return_5y',
                             'feat_total_return_10y', 'feat_tr_cagr_3y',
                             # Curated momentum ladder (mirrors the drift-predictor set).
                             'feat_total_return_1m', 'feat_total_return_3m',
                             'feat_total_return_6m', 'feat_total_return_1y',
                             'feat_tr_cagr_5y'
                             )
                 if c in kalman_df.columns]
    eda_noise = [c for c in ('feat_pt_range_norm', 'feat_pt_noise_sigma',
                             'feat_vol_drift')
                 if c in kalman_df.columns]
    # Size / valuation context (market-cap trend, size-vs-3y-avg, EV-vs-3y-avg).
    eda_size = [c for c in ('feat_mcap_trend_1y', 'feat_mcap_vs_3yavg', 'feat_ev_vs_3yavg')
                if c in kalman_df.columns]
    eda_features = eda_drift + eda_noise + eda_size

    _eda = kalman_df[eda_features].astype('float64')
    _lo = _eda.quantile(0.01)
    _hi = _eda.quantile(0.99)
    _eda_w = _eda.clip(lower=_lo, upper=_hi, axis=1)
    _eda_ds = xr.Dataset({col: ('sample', _eda_w[col].to_numpy()) for col in eda_features})
    feat_summary = azs.summary(
        _eda_ds, var_names=eda_features, kind='stats', round_to=4,
        sample_dims='sample', skipna=True,
    )
    print(f'Distributional summary (winsorised 1/99 pct) for {len(eda_features)} feat_* columns:')
    display(feat_summary)

    # 2.4b Drift-feature marginals as an arviz_plots ridge (z-scored).
    if eda_drift:
        _drift_std = ((_eda_w[eda_drift] - _eda_w[eda_drift].mean())
                      / _eda_w[eda_drift].std(ddof=0).replace(0, 1.0)).dropna()
        _n = len(_drift_std)
        _drift_arr = np.full((len(eda_drift), _n), np.nan)
        for i, col in enumerate(eda_drift):
            _drift_arr[i] = _drift_std[col].to_numpy()
        _ds_drift_ridge = xr.Dataset(
            {'drift_feature_z': (('feature', 'sample'), _drift_arr)},
            coords={'feature': eda_drift},
        )
        pc = azp.plot_ridge(_ds_drift_ridge, var_names=['drift_feature_z'],
                            sample_dims=['sample'], combined=True,
                            figure_kwargs=_azp_figure_kwargs(
                                _forest_height_px(len(eda_drift), per_row=30)))
        pc.add_title('Standardised drift-feature marginals (state-transition mean inputs)')
        _safe_show(pc)

    # 2.4c Observation-noise wideners — RAW (un-winsorised), on the model-facing scale.
    _w = build_noise_wideners(kalman_df, fillna=False)
    _widener_specs = [
        ('range  (feat_pt_range_norm)', _w['range'], False),
        ('cv  (feat_pt_noise_sigma / last_price)', _w['cv'], False),
        ('vol drift  (feat_vol_drift, signed)', _w['vol_drift'], True),
    ]
    if 'feat_pt_noise_drift' in kalman_df.columns:
        _widener_specs.insert(
            2, ('noise drift  (feat_pt_noise_drift, signed)',
                kalman_df['feat_pt_noise_drift'].astype('float64').to_numpy(), True))

    # Two-panel Plotly density view. Plotly has no symlog scale, so each KDE is drawn
    # on a robust 0.5–99.5 pct linear window (the tails are summarised numerically
    # below) rather than the former matplotlib symlog axis.
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.09,
        subplot_titles=('Observation-noise wideners (raw, model-scaled → σ_obs)',
                        'Realised σ_obs multiplier  (1 + range + cv + ½·max(vol_drift, 0)) / √n'))
    _pal = [_mcolors.to_hex(c)
            for c in sns.color_palette(CS_SEQ_MPL, len(_widener_specs))]
    _rows = []
    for (label, arr, is_signed), c in zip(_widener_specs, _pal):
        v = arr[np.isfinite(arr)]
        if v.size <= 5:
            continue
        med, p99 = float(np.nanmedian(v)), float(np.nanpercentile(v, 99))
        xs, ys = _kde_xy(v, clip_low=None if is_signed else 0.0)
        if xs is not None:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', fill='tozeroy',
                line=dict(color=c, width=1.7), fillcolor=_hex_to_rgba(c, 0.12),
                name=f'{label}  (med={med:.2g}, p99={p99:.2g}, min={float(v.min()):.2g})'),
                row=1, col=1)
            _add_ref_line(fig, x=med, kind='anchor', color=c,
                          opacity=0.8, row=1, col=1)
        _rows.append((label, c, med, p99, float(v.min())))
    fig.update_xaxes(title_text='model-facing value (signed where applicable)', row=1, col=1)
    fig.update_yaxes(title_text='density', row=1, col=1)

    mult = _w['multiplier']
    mult = mult[np.isfinite(mult)]
    fig.add_trace(go.Histogram(
        x=mult, histnorm='probability density', nbinsx=80,
        marker=dict(color=_hex_to_rgba(C_POSTERIOR, 0.35)),
        name='multiplier', showlegend=False), row=1, col=2)
    _xs_m, _ys_m = _kde_xy(mult, clip_low=0.0)
    if _xs_m is not None:
        fig.add_trace(go.Scatter(x=_xs_m, y=_ys_m, mode='lines',
                                 line=dict(color=C_POSTERIOR, width=2.0),
                                 name='multiplier KDE', showlegend=False), row=1, col=2)
    _mult_med, _mult_p99 = float(np.nanmedian(mult)), float(np.nanpercentile(mult, 99))
    _add_ref_line(fig, x=1.0, kind='zero', annotation_text='×1 (=base)',
                  row=1, col=2)
    _add_ref_line(fig, x=_mult_med, kind='emphasis', color=C_OBSERVED,
                  annotation_text=f'median={_mult_med:.2f}', row=1, col=2)
    _add_ref_line(fig, x=_mult_p99, kind='emphasis', color=C_FORECAST,
                  annotation_text=f'p99={_mult_p99:.2f}', row=1, col=2)
    fig.update_xaxes(title_text='σ_obs / σ_obs_base', row=1, col=2)
    fig.update_yaxes(title_text='density', row=1, col=2)
    fig.update_layout(barmode='overlay', legend=dict(font_size=9))
    _render_plotly(fig, height=H_STD)

    print('Observation-noise wideners (raw, un-winsorised, model-facing):')
    for lab, _c, m, p, mn in _rows:
        print(f'  - {lab:<42s} median={m:>9.3g}  p99={p:>10.3g}  min={mn:>8.3g}  '
              f'(p99/median tail ratio={p / m if m else float("nan"):.1f})')
    print(f'  sigma_obs multiplier: median={_mult_med:.2f}, p99={_mult_p99:.2f}; '
          f'{(mult > 1.0).mean() * 100:.0f}% of names widen sigma_obs above base, '
          f'{(mult < 1.0).mean() * 100:.0f}% (high-coverage) tighten it below base.')

    # 2.4d Feature collinearity heatmap (Spearman, robust to heavy feat_* tails).
    #      Interactive Plotly heatmap (hover + cell labels) when available; static
    #      seaborn heatmap fallback otherwise.
    _corr_cols = [c for c in eda_features if kalman_df[c].notna().sum() > 5]
    if _HAS_PLOTLY:
        _plot_feature_corr_heatmap_plotly(kalman_df, _corr_cols)
    elif len(_corr_cols) >= 2:
        _corr = kalman_df[_corr_cols].astype('float64').corr(method='spearman')
        fig, ax = plt.subplots(figsize=_mpl_figsize(0.85, width_frac=0.75),
                               layout='constrained')
        sns.heatmap(_corr, ax=ax, cmap=CS_DIV_MPL, center=0.0, vmin=-1, vmax=1,
                    square=True, linewidths=0.4, linecolor=C_PANEL_BG,
                    cbar_kws={'shrink': 0.7, 'label': 'Spearman ρ'},
                    annot=True, fmt='.2f', annot_kws={'size': 6})
        ax.set_title('feat_* correlation — drift vs noise-widener blocks', pad=10)
        _export_figure(fig, label='feature_corr_heatmap')
        plt.show()

    # 2.4e Implied upside vs the fused-panel risk/return drivers. Interactive Plotly
    #      facets — systematic-risk feat_avg_beta, feat_mcap_country_r size discount,
    #      short-horizon feat_one_day_return, plus the sigma_obs wideners — with
    #      ticker/name hover. Static seaborn scatter (noise-CV / 6m-vol) fallback.
    if _HAS_PLOTLY:
        _plot_upside_vs_drivers_plotly(kalman_df)
    else:
        _plot_upside_vs_signals_mpl(kalman_df)

    # 2.4f Empirical per-group implied-upside forest (EDA preview of §5 group
    # effects). Consolidated (0.9.9.11): the former up-to-7 near-identical
    # per-coord figures collapse into ONE faceted panel gated to the coords the
    # fused model actually uses as group effects, each facet sorted by median
    # with universe-median and 0% reference lines.
    _fe = kalman_df[(kalman_df['observed_pt'] > 0) & (kalman_df['last_price'] > 0)].copy()
    _fe['upside_pct'] = ((_fe['observed_pt'] / _fe['last_price'] - 1.0) * 100.0).clip(-100, 200)
    # Mirrors _FUSED_KALMAN_GROUP_EFFECTS minus the high-cardinality
    # exchange/unit coords (which drowned the preview and are model plumbing).
    _group_preview = [c for c in ('exchange_name', 'unit_name', 'country_name', 'industry',)
                      if c in _fe.columns]
    _min_per_level = 15
    _boot_draws = 4000
    _univ_median = float(_fe['upside_pct'].median())
    _facets: list[tuple[str, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for _coord in _group_preview:
        lab = _fe[_coord].fillna('Unknown').astype(str)
        _counts = lab.value_counts()
        levels = [lv for lv in _counts.index if _counts[lv] >= _min_per_level]
        if len(levels) < 2:
            continue
        # Bootstrap each level's per-name upside: point = bootstrap-mean median,
        # CI = the 94% ETI of the bootstrap means.
        _med = np.empty(len(levels))
        _lo = np.empty(len(levels))
        _hi = np.empty(len(levels))
        for i, lv in enumerate(levels):
            vals = _fe.loc[lab == lv, 'upside_pct'].to_numpy()
            _bm = vals[rng.integers(0, vals.size, (_boot_draws, min(vals.size, 512)))
                       ].mean(axis=1)
            _med[i] = float(np.median(_bm))
            _lo[i] = float(np.quantile(_bm, _HDI_LO))
            _hi[i] = float(np.quantile(_bm, _HDI_HI))
        _sorted = np.argsort(_med)
        _n = np.array([_counts[lv] for lv in levels])
        _facets.append((_coord,
                        [f'{levels[j]}  (n={_counts[levels[j]]})' for j in _sorted],
                        _med[_sorted], _lo[_sorted], _hi[_sorted], _n[_sorted]))
    if _facets and _HAS_PLOTLY:
        _tot = sum(len(f[1]) for f in _facets)
        figf = make_subplots(
            rows=len(_facets), cols=1, shared_xaxes=True,
            vertical_spacing=min(0.06, 20.0 / max(_tot * 24, 1)),
            row_heights=[len(f[1]) / _tot for f in _facets],
            subplot_titles=[f[0] for f in _facets])
        for i, (_coord, _lbls, _med, _lo, _hi, _n) in enumerate(_facets, start=1):
            figf.add_trace(go.Scatter(
                x=_med, y=_lbls, mode='markers',
                marker=dict(color=C_POSTERIOR, size=8),
                error_x=dict(type='data', symmetric=False,
                             array=np.clip(_hi - _med, 0, None),
                             arrayminus=np.clip(_med - _lo, 0, None),
                             color=_hex_to_rgba(C_POSTERIOR, 0.5), thickness=1.4),
                customdata=np.c_[_lo, _hi, _n],
                hovertemplate=('%{y}<br>median upside = %{x:.1f}%<br>94% CI = '
                               '[%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]'
                               '<br>n = %{customdata[2]:.0f}<extra></extra>'),
                name=_coord, showlegend=False), row=i, col=1)
            _add_ref_line(figf, x=0, kind='zero', row=i, col=1)
            _add_ref_line(figf, x=_univ_median, kind='emphasis', row=i, col=1)
        figf.update_xaxes(
            title_text=('implied upside vs last_price  '
                        '(observed_pt / last_price − 1, %); dashed = universe median'),
            ticksuffix='%', row=len(_facets), col=1)
        figf.update_yaxes(automargin=True)
        figf.update_layout(
            title='Empirical implied upside by group — EDA preview of the §5 '
                  'group effects (levels sorted by median)',
            showlegend=False)
        _render_plotly(figf, height=int(np.clip(24 * _tot + 80 * len(_facets) + 120,
                                                480, 1800)))

    # 2.4g Momentum / drift term-structure signal. Spearman ρ of each drift feature —
    #      including the new short-horizon feat_one_day_return — against implied upside,
    #      surfacing where the highest-frequency momentum signal sits across the
    #      1-day → multi-year horizons. Statistical table always; interactive ρ-bar
    #      (new feature highlighted) when plotly is available.
    _mom_corr = _momentum_signal_table(kalman_df, eda_drift)
    if _mom_corr is not None:
        print('Momentum / drift signal vs implied upside (Spearman ρ; 1-day → multi-year):')
        display(_mom_corr)
        if _HAS_PLOTLY:
            _plot_momentum_signal_plotly(_mom_corr)


# =============================================================================
# 3. State-space feature mapping (Kalman semantics)
# =============================================================================
def map_state_space_features(
        kalman_df: pd.DataFrame,
        feature_catalogue: Optional[pd.DataFrame] = None,
) -> tuple[list[str], pd.DataFrame]:
    """Map the catalogue's ``kalman_pt`` mutable_predictors onto state-space roles.

    The drift-feature list (state-transition mean / ``beta`` slopes) is derived
    from ``pml.vw_pymc_feature_catalogue`` (``pymc_role = 'mutable_predictor'``)
    — the SQL registry is the SSOT — filtered through
    :meth:`KalmanFilterPriceTarget.select_drift_features`, which drops the
    aliases consumed elsewhere in the fused model (noise wideners, the
    ``feat_avg_beta`` / ``feat_mcap_country_r`` tilts, ``days_*`` time
    covariates, drift-support counters, raw rating counts, the per-fiscal-year
    Piotroski component scores — their median enters instead) and the
    response-leakage alias. Returns the drift-feature list and a tidy
    role-mapping frame covering every mutable_predictor disposition.

    Parameters
    ----------
    kalman_df
        Snapshot frame backed by ``pml.mv_pymc_kalman_pt``; aliases absent from
        its columns are dropped.
    feature_catalogue
        The ``kalman_pt`` rows of ``pml.vw_pymc_feature_catalogue`` (from
        :func:`load_feature_catalogue`). When ``None`` the aliases are fetched
        via ``KalmanFilterPriceTarget._resolve_kalman_feature_aliases`` (a
        direct catalogue query); if that also fails, the curated
        :data:`_DRIFT_FEATURE_FALLBACK` literal keeps the workflow alive.

    Notes
    -----
    LEAKAGE GUARDRAIL: ``feat_implied_upside = (observed_pt - last_price)/last_price`` is
    a deterministic function of the RESPONSE; ``log1p(feat_implied_upside)`` IS the
    log-uplift the model targets, so it must never enter the drift-PREDICTOR matrix.
    """
    # --- Candidate mutable_predictor aliases (catalogue SSOT, layered fallbacks)
    candidates: list[str] = []
    source = 'feature_catalogue frame'
    if feature_catalogue is not None and {'pymc_role', 'feature_alias'} <= set(feature_catalogue.columns):
        candidates = (
            feature_catalogue.loc[
                feature_catalogue['pymc_role'] == 'mutable_predictor', 'feature_alias']
            .dropna().astype(str).tolist()
        )
    if not candidates:
        candidates = list(KalmanFilterPriceTarget._resolve_kalman_feature_aliases())
        source = 'vw_pymc_feature_catalogue (direct query)'
    if not candidates:
        candidates = list(_DRIFT_FEATURE_FALLBACK)
        source = '_DRIFT_FEATURE_FALLBACK literal'

    drift_features = KalmanFilterPriceTarget.select_drift_features(
        candidates, available_columns=kalman_df.columns,
    )
    assert 'feat_implied_upside' not in drift_features, (
        'feat_implied_upside must not be a drift predictor (target leakage).'
    )

    range_col = (KALMAN_RANGE_WIDENER_FEATURE
                 if KALMAN_RANGE_WIDENER_FEATURE in kalman_df.columns else None)
    sigma_col = (KALMAN_CONSENSUS_SIGMA_FEATURE
                 if KALMAN_CONSENSUS_SIGMA_FEATURE in kalman_df.columns else None)
    vol_cols = [c for c in (KALMAN_VOL_DRIFT_FEATURE,) if c in kalman_df.columns]

    mapping_rows: list[tuple[str, str]] = [
        (c, 'drift / state-transition mean (beta)') for c in drift_features
    ]
    if range_col:
        mapping_rows.append((range_col, 'observation-noise widener (range)'))
    if sigma_col:
        mapping_rows.append((sigma_col, 'observation-noise widener (consensus sigma)'))
    mapping_rows += [(c, 'observation-noise widener (realized-vol drift)') for c in vol_cols]
    # Remaining mutable_predictor dispositions, so the mapping frame accounts
    # for every catalogue row (dropped ≠ forgotten).
    mapping_rows += [
        (c, 'risk/size tilt (named pm.Data container)')
        for c in KALMAN_TILT_FEATURE_ORDER if c in kalman_df.columns
    ]
    mapping_rows += [
        (c, 'time covariate (t_scaled axis)')
        for c in candidates
        if c.startswith(KALMAN_TIME_COVARIATE_PREFIX) and c in kalman_df.columns
    ]
    if 'feat_implied_upside' in kalman_df.columns:
        mapping_rows.append(
            ('feat_implied_upside', 'RESPONSE (log1p -> feat_log_uplift; leakage-barred)')
        )
    mapping_rows += [
        (c, 'excluded (drift-support counter / raw rating count / collinear leg / Piotroski component)')
        for c in candidates
        if (c in KALMAN_DRIFT_EXCLUDED_FEATURES and c in kalman_df.columns
            and c not in drift_features
            and all(c != row[0] for row in mapping_rows))
    ]
    mapping = pd.DataFrame(mapping_rows, columns=['mv_column', 'state_space_role'])

    print(f'Drift-feature source: {source} ({len(candidates)} mutable_predictor candidates)')
    print(f'Drift features : {drift_features}')
    print(f'Noise drivers  : range={range_col}, sigma={sigma_col}, vol={vol_cols}')
    display(mapping)
    return drift_features, mapping


# =============================================================================
# 5b. Fused MvGRW panel model (Model A + Model B)
# =============================================================================
# The legacy §4/§5 single-observation cross-sectional model (``ModelData`` /
# ``build_model_data`` / ``build_kalman_pt_model``) was superseded by the fused
# panel path below (``prepare_kalman_panel_inputs`` + ``build_panel_model``)
# and removed in 0.9.9.10.
# D-dimensional joint response series broadcast across the T fiscal anchors.
# Mirrors PANEL_RESPONSE_COLS in _price_target_mc.py, re-homed onto the kalman MV.
KALMAN_PANEL_RESPONSE_COLS: tuple[str, ...] = (
    # Distinct, low-collinearity signals only. The dropped price-target levels
    # (median/high/low) are near-collinear with observed_pt and are already
    # captured via the dispersion path (feat_pt_noise_sigma -> sigma_isin); as
    # response series they only inflated D and added posterior correlation.
    #
    # PRIMARY series = feat_log_uplift == log1p(feat_implied_upside), NOT the raw
    # decimal upside. It MUST stay first — it is the de-standardisation key in
    # panel_posterior_upside, and the per-series loadings anchor index 0 at 1.0.
    # Modelling the LOG uplift (and reconstructing expected_pt = last_price *
    # exp(log_uplift)) guarantees a strictly-positive price target: the raw
    # decimal feat_implied_upside is unbounded below, so de-standardising the
    # Gaussian baseline linearly (last_price * (1 + eu)) produced *negative*
    # expected_pt for names trading far above their analyst targets (extreme YTD
    # momentum run-ups, e.g. JUSUNG +667 %, Sandisk +817 %, IQE +1018 %), where
    # eu < -1 implies an impossible sub-zero price.
    #
    # observed_pt is dropped as a response: it is observed_pt == last_price *
    # (1 + feat_implied_upside), i.e. a deterministic, near-collinear price-level
    # restatement of the primary series whose standardised scale is dominated by
    # a handful of high-price names and which dragged the shared mu_isin factor.
    #
    # feat_pt_drift is dropped as a response (convergence fix). It is ALSO a drift
    # PREDICTOR (see map_state_space_features → X_drift), so using it as a response
    # made the model partly predict it from itself; and on the single-snapshot MV it
    # is a sparse, winsorised([-1,1]) trail that standardises to a near-constant
    # column, leaving its rank-1 ICM loading (mu_isin_loading) + per-series noise
    # (sigma_series) UNIDENTIFIED — a weak ridge that (together with the former dead
    # achieve_prob block) produced the ~600–1650 divergences / R-hat 3–4.5 freeze.
    # With a single response the coregion collapses to a clean hierarchical
    # cross-sectional regression (empirically: 0 divergences, R-hat → ~1.0). The
    # rank-1 ICM remains available for a genuine multi-signal panel (a DISTINCT 2nd
    # series not already in X_drift); add it here then. The TIME dimension is
    # populated from the *_ago trails via ``history_lookbacks`` (T > 1) in
    # ``prepare_kalman_panel_inputs``.
    'feat_log_uplift',
)

# Minimum finite-coverage fraction a NON-PRIMARY response series must clear to
# enter the fused ICM. A sparsely-populated trail (e.g. ``feat_pt_drift``, NULL
# whenever ``price_target_*_ago`` is unpopulated) standardises to a near-constant
# (mostly-zero) column whose rank-1 coregion loading ``mu_isin_loading`` is then
# UNIDENTIFIED — a flat ``loading × mu_isin`` ridge that collapses the NUTS step
# size and freezes the whole posterior (observed: max R-hat 4.45, min ESS 4.3, 0
# divergences). The primary series is always kept; non-primary series below this
# coverage (or with ~0 standardised variance) are dropped with a logged message.
KALMAN_RESPONSE_COVERAGE_MIN: float = 0.60

# --- Optional SECOND response series (D > 1), opt-in --------------------------
# Populating a second series is what makes the rank-1 ICM live: with D == 1 the
# coregion loading ``mu_isin_loading`` and the per-output noise diagonal
# ``sigma_series`` are not created at all, so the whole "MOGP-Coregion-Hadamard"
# apparatus in build_fused_kalman_pt_model is dormant.
#
# The requirement a candidate must meet is NOT merely "another column". Per the
# KALMAN_PANEL_RESPONSE_COLS notes it must be (a) a DISTINCT signal, not a
# near-collinear restatement of the primary log-uplift, (b) NOT already an
# X_drift predictor — otherwise the model partly predicts a response from itself
# — and (c) populated across a genuine ``*_ago`` trail, so it fills the time
# panel rather than sitting NaN in every history cell (a snapshot-only series
# standardises to a near-constant column and leaves its loading unidentified,
# the documented R-hat 4.45 / min-ESS 4.3 freeze).
#
# ``pt_dispersion`` is the one candidate that satisfies all three: the analyst
# dispersion LEVEL, log1p(price_target_stddev_{lb} / price_{lb}), built from the
# stddev trail that mv_pymc_kalman_pt emits for every lookback. It measures
# disagreement rather than direction, so it is genuinely orthogonal to the uplift
# level, and it shares the latent conviction/quality factor the ICM is meant to
# pool. Its DRIFT (``feat_pt_noise_drift``) is an X_drift predictor, so enabling
# this series drops that predictor -- see _resolve_response_extra.
KALMAN_PANEL_RESPONSE_EXTRA: dict[str, dict[str, str]] = {
    'pt_dispersion': {
        'numerator': 'price_target_stddev_{lb}_ago',
        'denominator': 'price_{lb}_ago',
        'snapshot_numerator': 'feat_pt_noise_sigma',
        'snapshot_denominator': 'last_price',
        # Predictor that must leave X_drift when this series is promoted: it is
        # the first difference of exactly this level.
        'conflicting_drift': 'feat_pt_noise_drift',
    },
}


def _dispersion_log_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Return ``log1p(numerator / denominator)`` with non-positive denominators voided.

    The scale map applied to every optional dispersion response series, at the
    snapshot and at each ``*_ago`` lookback alike, so all ``T`` slices of the
    series live on one scale (exactly as the primary log-uplift does). The ratio
    is winsorised at 200 % of price before the log: analyst-dispersion outliers
    are real but a handful of them would otherwise dominate the series std and
    drag the shared coregion factor.

    Parameters
    ----------
    numerator, denominator
        Aligned dispersion / price columns.

    Returns
    -------
    numpy.ndarray
        ``float64`` array; non-finite where the denominator is not > 0.
    """
    num = pd.to_numeric(numerator, errors='coerce')
    den = pd.to_numeric(denominator, errors='coerce')
    ratio = (num / den.where(den > 0)).clip(lower=0.0, upper=2.0)
    # ``copy=True``: callers fill missing history cells in place, and pandas can
    # hand back a read-only view here.
    return np.log1p(ratio).to_numpy(dtype='float64', copy=True)


def prepare_kalman_panel_inputs(
        kalman_df: pd.DataFrame,
        roles: Optional[FeatureRoles] = None,
        drift_features: Optional[list[str]] = None,
        *,
        classification_coords: Optional[Sequence[str]] = None,
        response_cols: Sequence[str] = KALMAN_PANEL_RESPONSE_COLS,
        history_lookbacks: Optional[Sequence[str]] = None,
        time_covariate_cols: Sequence[str] = DAY_COUNT_COLS_ALL,
        response_extra: Sequence[str] = (),
) -> KalmanPanelInputs:
    """Assemble the 3-D MvGRW panel tensors for :func:`build_fused_kalman_pt_model`.

    The Kalman analogue of
    ``probabilistic_ml_model.pymc_models._price_target_mc.prepare_price_target_panel_inputs``.
    Pure-NumPy/pandas (no PyMC import) so it is unit-testable in isolation. Keeps
    rows with strictly-positive ``observed_pt`` / ``last_price`` and reuses
    :func:`build_noise_wideners` so the per-ISIN expected volatility / dispersion
    CV / precision drivers match the §4.2 cross-sectional model by construction.

    Parameters
    ----------
    kalman_df : pandas.DataFrame
        Frame backed by ``pml.mv_pymc_kalman_pt``.
    roles : FeatureRoles, optional
        Resolved column groups. When supplied, ``roles.classification_coords``
        seeds the categorical group-effect coords. May be ``None`` so the helper
        is usable directly from the inline notebook (which never builds a
        ``FeatureRoles``); the coords then fall back to ``classification_coords``
        or :data:`CLASSIFICATION_COORDS_ALL` intersected with the frame columns.
    drift_features : list[str], optional
        Standardised state-transition-mean inputs (from
        :func:`map_state_space_features`). Required — derived by the caller.
    classification_coords : Sequence[str], optional
        Explicit categorical group-effect coords. Overrides ``roles`` when given.
    response_cols : Sequence[str]
        ``D`` response series columns.
    history_lookbacks : Sequence[str], optional
        ``*_ago`` suffixes (e.g. ``('6m', '3m', '1m')``) building a **genuine**
        ``(isin, time)`` log-uplift panel from the MV's
        ``price_target_{lb}_ago`` / ``price_{lb}_ago`` trails, ordered oldest →
        newest with the current snapshot as the final time step (``T =
        len(lookbacks) + 1``). Empty / ``None`` (the default) keeps the
        collapsed single-slice cross-section (``T = 1``). Lookbacks whose
        column pair is absent from the frame are dropped with a logged message.
        This replaces the earlier ``collapse_time=False`` branch, which merely
        **tiled** the snapshot across the fiscal anchors — time-invariant data
        that double-counted every observation ``T``-fold and ill-conditioned
        the posterior.
    response_extra : Sequence[str]
        Keys of :data:`KALMAN_PANEL_RESPONSE_EXTRA` promoting an optional
        **second** response series (``D > 1``), which is what activates the rank-1
        ICM (``mu_isin_loading`` / ``sigma_series``) — dormant while ``D == 1``.
        Each series is derived on the same ``log1p`` scale as the primary and
        carries its own ``*_ago`` trail so it populates the time panel. Promoting
        a series DROPS any drift predictor declared as its ``conflicting_drift``
        (response ↔ predictor disjointness), with a printed note. Empty by
        default: the ``D > 1`` path is the one that produced the historic R-hat
        4.45 / min-ESS 4.3 freeze and must be validated on its own.

    Returns
    -------
    KalmanPanelInputs

    Raises
    ------
    ValueError
        If ``drift_features`` is not supplied or no rows survive filtering.
    """
    if drift_features is None:
        raise ValueError(
            "prepare_kalman_panel_inputs requires drift_features "
            "(call map_state_space_features(kalman_df) first)."
        )
    model_df = kalman_df.loc[
        (kalman_df['observed_pt'] > 0)
        & (kalman_df['last_price'] > 0)
        & kalman_df['observed_pt'].notna()
        & kalman_df['last_price'].notna()
        ].copy().reset_index(drop=True)

    if 'n_analysts' in model_df.columns:
        model_df['n_analysts'] = model_df['n_analysts'].fillna(1).clip(lower=1)
    else:
        model_df['n_analysts'] = 1.0

    # --- Primary response on the LOG-uplift scale (positivity guarantee) -------
    # feat_log_uplift == log1p(feat_implied_upside) == log(observed_pt/last_price).
    # Modelling the log uplift keeps the reconstructed price target strictly
    # positive (expected_pt = last_price * exp(log_uplift)); the raw decimal
    # feat_implied_upside is unbounded below and, de-standardised linearly,
    # produced negative expected_pt for names trading far above their targets.
    # The cross-section is winsorised to a sane band BEFORE the log so a handful
    # of extreme YTD-momentum names (price >> stale analyst target, or vice
    # versa) cannot dominate the response std / drag the shared mu_isin factor.
    _last = model_df['last_price'].astype('float64')
    _obs = model_df['observed_pt'].astype('float64')
    _raw_uplift = (_obs / _last - 1.0)
    if 'feat_implied_upside' in model_df.columns:
        _iu = model_df['feat_implied_upside'].astype('float64')
        _raw_uplift = _iu.where(np.isfinite(_iu), _raw_uplift)
    # Cap to [-95 %, +500 %] implied upside before logging: bounds the heavy
    # momentum tails without discarding the names.
    _uplift_w = _raw_uplift.clip(lower=-0.95, upper=5.0)
    model_df['feat_log_uplift'] = np.log1p(_uplift_w).to_numpy()

    isin_labels = model_df['isin'].astype(str).to_numpy()
    n_isin = len(model_df)
    if n_isin == 0:
        raise ValueError('No modelling rows after filtering observed_pt / last_price.')

    # --- Optional derived SECOND response series (D > 1), opt-in --------------
    # Materialises the snapshot column for each requested extra series and
    # enforces response <-> predictor disjointness before anything is fitted.
    drift_features = list(drift_features)
    extra_specs: dict[str, dict[str, str]] = {}
    for key in response_extra:
        if key not in KALMAN_PANEL_RESPONSE_EXTRA:
            raise ValueError(
                f'Unknown response_extra {key!r}. Supported: '
                f'{sorted(KALMAN_PANEL_RESPONSE_EXTRA)}.')
        spec = KALMAN_PANEL_RESPONSE_EXTRA[key]
        num, den = spec['snapshot_numerator'], spec['snapshot_denominator']
        if num not in model_df.columns or den not in model_df.columns:
            logger.warning(
                'response_extra %r dropped (missing %s / %s).', key, num, den)
            continue
        model_df[key] = _dispersion_log_ratio(model_df[num], model_df[den])
        extra_specs[key] = spec
        # A response must never also be a predictor: the model would partly
        # predict the series from a deterministic function of itself. This is the
        # contract `assert_disjoint_features` exists to enforce; here the collision
        # is known in advance, so the conflicting predictor is dropped loudly.
        clash = spec.get('conflicting_drift')
        if clash and clash in drift_features:
            drift_features.remove(clash)
            print(f'  [response_extra] {key!r} promoted to a response series; '
                  f'dropped the collinear drift predictor {clash!r} '
                  f'(it is the first difference of this level).')
    if extra_specs:
        response_cols = (*response_cols, *extra_specs)

    # --- Time axis + response tensor (n_isin, T, D) standardised --------------
    # ``history_lookbacks`` builds a GENUINE (isin, time) log-uplift panel from
    # the MV's ``price_target_{lb}_ago`` / ``price_{lb}_ago`` trails; without it
    # we model a single collapsed cross-sectional slice (T=1). The old
    # ``collapse_time=False`` branch that np.tile-d the snapshot across the
    # fiscal anchors (time-invariant data, T-fold observation double-counting,
    # the ill-conditioned max-tree-depth posterior) was removed with it.
    resp_all = [c for c in response_cols if c in model_df.columns]
    if not resp_all:
        raise KeyError(f'None of the response columns {list(response_cols)} present.')

    # --- Degeneracy guard: drop near-constant / sparsely-populated responses ---
    # A non-primary series that is mostly NULL (e.g. ``feat_pt_drift`` when the
    # ``price_target_*_ago`` trail is unpopulated) standardises to ~0 and leaves its
    # rank-1 ICM loading unidentified — the multiplicative ridge that froze the
    # sampler. The PRIMARY series (index 0, the de-standardisation key) is always
    # retained. Coverage is measured on the RAW column (before the model's nan→0
    # fill) so the post-fill zero spike cannot masquerade as signal.
    primary_resp = resp_all[0]
    resp = [primary_resp]
    for c in resp_all[1:]:
        raw = pd.to_numeric(model_df[c], errors='coerce').to_numpy(dtype='float64')
        finite = np.isfinite(raw)
        frac = float(finite.mean()) if raw.size else 0.0
        sd = float(np.nanstd(raw)) if finite.any() else 0.0
        if frac < KALMAN_RESPONSE_COVERAGE_MIN or not np.isfinite(sd) or sd < 1e-6:
            print(f'  [guard] dropping degenerate response series {c!r} '
                  f'(finite_frac={frac:.2f}, raw_sd={sd:.3g}; '
                  f'gate finite_frac>={KALMAN_RESPONSE_COVERAGE_MIN:.2f}, sd>=1e-6)')
            continue
        resp.append(c)
    D = len(resp)

    # Resolve the usable history lookbacks: both trail columns must exist and
    # the suffix must have a calendar day-count (period-to-date suffixes are
    # data-dependent and excluded). Ordered oldest -> newest so the model's
    # zero-anchored GRW deviation anchors at the OLDEST step; the snapshot is
    # the final time step.
    lookbacks: list[str] = []
    if history_lookbacks:
        for lb in history_lookbacks:
            pt_col, px_col = f'price_target_{lb}_ago', f'price_{lb}_ago'
            if lb not in _AGO_APPROX_DAYS:
                logger.warning('history lookback %r has no calendar day count; dropped.', lb)
            elif pt_col not in model_df.columns or px_col not in model_df.columns:
                logger.warning('history lookback %r dropped (missing %s / %s).',
                               lb, pt_col, px_col)
            else:
                lookbacks.append(lb)
        lookbacks.sort(key=lambda s: -_AGO_APPROX_DAYS[s])

    if lookbacks:
        # Genuine (n_isin, T, D) history panel. Each lookback's implied uplift
        # (price_target_lb_ago / price_lb_ago - 1) gets the same [-95%, +500%]
        # winsorisation-then-log1p treatment as the snapshot response, so all T
        # slices live on one scale. Missing history cells are filled with the
        # name's OWN snapshot log-uplift — a repeated observation for that
        # minority of names (never a fake cross-sectional-mean observation,
        # which the post-standardisation nan->0 fill would create).
        T = len(lookbacks) + 1
        offsets = np.asarray(
            [-float(_AGO_APPROX_DAYS[lb]) for lb in lookbacks] + [0.0], dtype='float64')
        t_std = float(np.std(offsets)) or 1.0
        t_scaled = np.tile((offsets - float(np.mean(offsets))) / t_std, (n_isin, 1))

        snap_uplift = model_df['feat_log_uplift'].to_numpy(dtype='float64')
        hist_cols: list[np.ndarray] = []
        n_filled = 0
        for lb in lookbacks:
            pt_l = pd.to_numeric(model_df[f'price_target_{lb}_ago'], errors='coerce')
            px_l = pd.to_numeric(model_df[f'price_{lb}_ago'], errors='coerce')
            uplift = (pt_l / px_l.where(px_l > 0) - 1.0).clip(lower=-0.95, upper=5.0)
            col = np.log1p(uplift).to_numpy(dtype='float64', copy=True)
            missing = ~np.isfinite(col)
            n_filled += int(missing.sum())
            col[missing] = snap_uplift[missing]
            hist_cols.append(col)
        primary_panel = np.stack([*hist_cols, snap_uplift], axis=1)  # (n_isin, T)

        # Non-primary series: build a genuine trail when the series declares one
        # (KALMAN_PANEL_RESPONSE_EXTRA), else observe it at the snapshot step only
        # and leave the history cells NaN (-> series mean 0 after the nan-aware
        # standardisation below). A snapshot-only series in a T > 1 panel is a
        # near-constant column whose ICM loading is unidentified, so this branch
        # is where an opt-in extra series earns its place.
        per_series = [primary_panel]
        for c in resp[1:]:
            snap_vals = model_df[c].astype('float64').to_numpy()
            spec = extra_specs.get(c)
            if spec is None:
                filled = np.full((n_isin, T), np.nan, dtype='float64')
                filled[:, -1] = snap_vals
                per_series.append(filled)
                continue
            cols: list[np.ndarray] = []
            for lb in lookbacks:
                num_c = spec['numerator'].format(lb=lb)
                den_c = spec['denominator'].format(lb=lb)
                if num_c in model_df.columns and den_c in model_df.columns:
                    col = _dispersion_log_ratio(model_df[num_c], model_df[den_c])
                else:
                    logger.warning(
                        'response_extra %r: lookback %r missing %s / %s; that '
                        'history cell falls back to the snapshot value.',
                        c, lb, num_c, den_c)
                    col = np.full(n_isin, np.nan, dtype='float64')
                missing = ~np.isfinite(col)
                col[missing] = snap_vals[missing]
                cols.append(col)
            per_series.append(np.stack([*cols, snap_vals], axis=1))
        Y = np.stack(per_series, axis=-1)
        print(f'  history panel: lookbacks={lookbacks} -> T={T}; '
              f'{n_filled} missing history cells filled with the snapshot uplift '
              f'({n_filled / max(n_isin * (T - 1), 1):.1%} of history cells).')
    else:
        T = 1
        # A single standardised time-to-event covariate keeps the beta_t slope a
        # meaningful cross-sectional tilt instead of multiplying by an all-zero
        # anchor difference; falls back to zeros when no day-count column exists.
        tcov_col = next((c for c in time_covariate_cols if c in model_df.columns), None)
        if tcov_col is not None:
            tcov = pd.to_numeric(model_df[tcov_col], errors='coerce').to_numpy(dtype='float64')
            tc_mean = np.nanmean(tcov) if np.isfinite(tcov).any() else 0.0
            tc_std = np.nanstd(tcov)
            tc_std = tc_std if tc_std and np.isfinite(tc_std) else 1.0
            t_scaled = np.nan_to_num((tcov - tc_mean) / tc_std, nan=0.0)[:, None]
        else:
            t_scaled = np.zeros((n_isin, 1), dtype='float64')
        # Response tensor (n_isin, 1, D) — per-ISIN scalar values, no tiling.
        Y = np.stack(
            [model_df[c].astype('float64').to_numpy()[:, None] for c in resp],
            axis=-1,
        )

    # Standardise the response tensor across (isin, time) per series. Use NaN-AWARE
    # reductions: a plain ``.mean()`` / ``.std()`` propagates a single NaN to the
    # whole series statistic, so ``(Y - NaN) / 1`` then ``nan_to_num``→0 silently
    # ZEROED any series carrying even one NaN — turning a partially-observed trail
    # into a fully-degenerate constant. ``nanmean`` / ``nanstd`` standardise on the
    # observed entries; surviving NaNs (coverage already ≥ the guard threshold) map
    # to the series mean (0 after centering).
    flat = Y.reshape(-1, D)
    y_mean = np.nanmean(flat, axis=0)
    y_std = np.nanstd(flat, axis=0)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0)
    y_std = np.where(np.isfinite(y_std) & (y_std > 1e-6), y_std, 1.0)
    Y_std = np.nan_to_num((Y - y_mean) / y_std, nan=0.0)

    # --- Standardised drift design matrix (state-transition mean inputs) ------
    x_raw = model_df[drift_features].astype(float)
    x_std = ((x_raw - x_raw.mean()) / x_raw.std(ddof=0).replace(0, 1.0)).fillna(0.0).to_numpy()

    # --- Model-A / σ drivers (shared SSOT helper) -----------------------------
    # vol_drift == feat_vol_drift (drift of the realized-vol term structure; the
    # σ_obs widener carrier); cv == consensus noise CV widening σ.
    _nw = build_noise_wideners(model_df, fillna=True)
    vol_drift = _nw['vol_drift']
    dispersion_cv = _nw['cv']
    sqrt_n = _nw['sqrt_n']
    n_analysts = model_df['n_analysts'].astype('float64').clip(lower=1).to_numpy()

    # --- Systematic-risk driver (feat_avg_beta) -------------------------------
    # feat_avg_beta == NULL-aware mean of beta_{1y,2y,5y} (mv_pymc_kalman_pt). It
    # is the risk_adj_return discount key (realized vol enters only via the
    # vol_drift widener); the raw beta is carried here and z-scored inside
    # build_fused_kalman_pt_model. Falls back to NaN (→ 0 tilt after the model's
    # nan_to_num) when the column is absent.
    if 'feat_avg_beta' in model_df.columns:
        avg_beta = pd.to_numeric(model_df['feat_avg_beta'], errors='coerce').to_numpy('float64')
    else:
        avg_beta = np.full(n_isin, np.nan, dtype='float64')

    # --- Size driver (feat_mcap_country_r) -------------------------------------
    # feat_mcap_country_r == market_cap / market_cap_3yavg (mv_pymc_kalman_pt): the
    # firm's current size relative to its own 3y-average size. Carried raw here and
    # z-scored inside build_fused_kalman_pt_model, where it discounts risk_adj_return
    # additively via - size_loading * z(size). Falls back to NaN
    # (-> neutral 0.0 tilt after the model's nan_to_num) when the column is absent.
    if 'feat_mcap_country_r' in model_df.columns:
        size_ratio = pd.to_numeric(model_df['feat_mcap_country_r'], errors='coerce').to_numpy('float64')
    else:
        size_ratio = np.full(n_isin, np.nan, dtype='float64')

    # --- Volume driver (feat_rel_volume) ---------------------------------------
    # feat_rel_volume == relative trading volume (mv_pymc_kalman_pt). Carried raw
    # here and z-scored inside build_fused_kalman_pt_model, where it discounts
    # risk_adj_return additively via - volume_loading * z(volume). Falls back to
    # NaN (-> neutral 0.0 tilt after the model's nan_to_num) when the column is
    # absent.
    if 'feat_rel_volume' in model_df.columns:
        volume_ratio = pd.to_numeric(model_df['feat_rel_volume'], errors='coerce').to_numpy('float64')
    else:
        volume_ratio = np.full(n_isin, np.nan, dtype='float64')

    # --- Categorical group-effect coords (classification coords only) ---------
    # Resolve the coord source: explicit arg > roles object > module default.
    if classification_coords is not None:
        _coord_src: Sequence[str] = classification_coords
    elif roles is not None:
        _coord_src = roles.classification_coords
    else:
        _coord_src = CLASSIFICATION_COORDS_ALL
    categorical_coords = [c for c in _coord_src
                          if c in model_df.columns and c not in ('isin', 'ticker')]
    coord_uniques: dict[str, np.ndarray] = {}
    coord_idx: dict[str, np.ndarray] = {}
    for col in categorical_coords:
        labels = model_df[col].fillna('Unknown').astype(str).to_numpy()
        uniques, idx = np.unique(labels, return_inverse=True)
        coord_uniques[col] = uniques
        coord_idx[col] = idx.astype('int64')

    # avg_beta / size_ratio / volume_ratio are carried NaN-bearing (see above) when
    # their source column is absent or fails numeric coercion for every row; guard
    # each summary mean the same way as the tcov/gain reductions elsewhere in this
    # module (np.isfinite(...).any()) so an all-missing driver logs 0.0 instead of
    # tripping numpy's "Mean of empty slice" RuntimeWarning.
    _safe_mean = lambda arr: float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
    print(f'Fused panel — isins:{n_isin}  T:{T}  D:{D} ({resp})')
    print(f'  drift_features:{len(drift_features)}  avg_beta(mean):{_safe_mean(avg_beta):.3f}'
          f'  size_ratio(mean):{_safe_mean(size_ratio):.3f}'
          f'  volume_ratio(mean):{_safe_mean(volume_ratio):.3f}'
          f'  vol_drift(mean):{_safe_mean(vol_drift):.3f}'
          f'  cv(mean):{_safe_mean(dispersion_cv):.3f}')

    return KalmanPanelInputs(
        frame=model_df, isins=isin_labels, Y=Y_std, t_scaled=t_scaled,
        X_drift=x_std, n_analysts=n_analysts, sqrt_n_analysts=sqrt_n,
        vol_drift=vol_drift, dispersion_cv=dispersion_cv, avg_beta=avg_beta,
        size_ratio=size_ratio, volume_ratio=volume_ratio,
        drift_names=list(drift_features), response_names=list(resp),
        coord_uniques=coord_uniques, coord_idx=coord_idx,
        # The EXACT moments applied above — carried so the posterior can be
        # de-standardised without recomputation. See _panel_response_stats.
        response_mean=np.asarray(y_mean, dtype='float64'),
        response_std=np.asarray(y_std, dtype='float64'),
    )


# --- Fused-model posterior variable groupings (single source of truth) --------
# Scalars (global hyper-parameters); group-effect scales ``sigma_<coord>`` are
# appended at runtime from the panel coords actually present.
FUSED_SCALAR_VARS: tuple[str, ...] = (
    # ``mu_logit`` / ``sigma_logit`` were removed with the dead achievement block
    # in build_fused_kalman_pt_model (they sampled the prior only and dominated the
    # divergences); ``achieve_prob`` is now a Deterministic of ``state_now``.
    #
    # ``sigma_state`` is the local-level state's per-step innovation sd. It exists
    # only on a genuine T > 1 panel with the state enabled; run_diagnostics skips
    # absent vars. Watch it: a posterior pressed against 0 means the panel carries
    # no per-name time dynamics and the state layer is dead weight.
    'sigma_base', 'nu', 'sigma_state',
    # Learned, sign-fixed risk / size loadings (additive tilts on the per-ISIN
    # baseline keyed on feat_avg_beta / feat_mcap_country_r). Present only when the
    # corresponding penalty prior scale is > 0; run_diagnostics skips absent vars.
    'risk_loading', 'size_loading', 'volume_loading',
)
# Vector hyper-parameters (have a non-sample dim): drift slopes and the
# per-series coregion level/slope/loading/noise terms. run_diagnostics skips any
# that are absent for the fitted shape (``beta_slope`` exists only when t_scaled
# varies across ISINs; ``sigma_series`` / ``mu_isin_loading`` only when D > 1).
FUSED_VECTOR_VARS: tuple[str, ...] = (
    'beta', 'alpha_level', 'beta_slope', 'mu_isin_loading', 'sigma_series',
)


def _max_posterior_rhat(posterior: "xr.Dataset",
                        var_names: Optional[Sequence[str]] = None) -> float:
    """Return the worst-case split R-hat across ``var_names``.

    Uses nan-aware reductions: deterministically-anchored entries (constant
    across draws, e.g. the pinned primary-series ICM loading
    ``mu_isin_loading`` / noise ``sigma_series``) yield a NaN R-hat by
    construction (0/0 within/between variance) and must not mask the genuine
    worst case.

    Parameters
    ----------
    posterior
        Posterior variables as a plain Dataset (see :func:`_posterior_dataset`).
    var_names
        Variables to reduce over; defaults to every posterior variable. Names
        absent from ``posterior`` are skipped.

    Returns
    -------
    float
        Maximum finite R-hat, or ``nan`` when nothing can be computed.
    """
    keys = [v for v in (var_names if var_names is not None else posterior.data_vars)
            if v in posterior.data_vars]
    if not keys:
        return float('nan')
    try:
        rhat_ds = azs.rhat(posterior[keys])
        return float(np.nanmax([float(rhat_ds[v].max()) for v in rhat_ds.data_vars]))
    except Exception as exc:
        logger.warning('Max R-hat computation failed for %r: %s', keys, exc)
        return float('nan')


def sample_with_fallback(model: "pm.Model", config: Optional[KalmanRunConfig] = None,
                         *, model_name: str = 'kalman_pt',
                         progressbar: bool = True) -> Optional[Any]:
    """Sample ``model``, trying nutpie → numpyro → pymc in priority order.

    The same sampler priority :func:`sample_posterior` uses, factored out so
    secondary fits (§9b model comparison, the validation script) cannot silently
    diverge from the production path.

    **This is not a micro-optimisation.** The project forces the PyTensor
    pure-Python VM (``PML_ENABLE_PYTENSOR_C=0``), under which PyMC's own NUTS is
    orders of magnitude slower than the nutpie numba backend. On the local-level
    panel — which adds ``n_isin × (T-1)`` state innovations, ~16.8k parameters on
    a 5.6k-ISIN T=4 panel — the pure-Python sampler produced **zero draws in 42
    minutes of CPU**. Anything that samples this model must go through here (or
    pass ``nuts_sampler`` explicitly), never bare ``build_sample_kwargs``, whose
    ``nuts_sampler=None`` default lands on the pure-Python path.

    Parameters
    ----------
    model
        The built model to sample.
    config
        Supplies the NUTS budget. Defaults to :func:`get_run_config`.
    model_name
        Label used in the diagnostics log lines.
    progressbar
        ``False`` for back-to-back fits — the bars are noise, and nutpie's
        thin-space glyphs crash a cp1252 Windows console.

    Returns
    -------
    Any or None
        The inference object, or ``None`` when every candidate sampler failed.
    """
    cfg = config if config is not None else get_run_config()
    candidates = [s for s in ('nutpie', 'numpyro')
                  if _ilu.find_spec(s) is not None] + ['pymc']
    logger.info('%s: sampler priority %s', model_name, candidates)
    for sampler in candidates:
        try:
            with model:
                idata = pm.sample(**build_sample_kwargs(
                    samples=cfg.draws, tune=cfg.tune, chains=cfg.chains,
                    target_accept=cfg.target_accept, random_seed=cfg.random_seed,
                    cores=cfg.cores, nuts_sampler=sampler,
                    model_name=model_name, progressbar=progressbar))
            log_sample_diagnostics(idata, model_name=model_name)
            return idata
        except Exception as exc:
            logger.warning('%s: sampler %r failed (%s); falling back.',
                           model_name, sampler, exc)
    return None


def build_panel_model(
        panel: KalmanPanelInputs, *, robust: bool = True, volume_penalty: float = 0.25,
        config: Optional[KalmanRunConfig] = None,
        state_innovation_scale: Optional[float] = None,
) -> "pm.Model":
    """Build the fused local-level-state model and render its graph.

    Parameters
    ----------
    panel
        Output of :func:`prepare_kalman_panel_inputs`.
    robust
        Student-t (``True``) vs Normal panel likelihood.
    volume_penalty
        Prior scale of the learned relative-volume tilt.
    config
        Supplies ``state_innovation_scale`` when it is not passed explicitly.
    state_innovation_scale
        Explicit override of the local-level innovation prior scale. ``0.0``
        pins the state at its t=0 anchor (the static comparison baseline).
    """
    if state_innovation_scale is None:
        cfg = config if config is not None else get_run_config()
        state_innovation_scale = cfg.state_innovation_scale
    model = build_fused_kalman_pt_model(
        panel, robust=robust, volume_penalty=volume_penalty,
        state_innovation_scale=float(state_innovation_scale))
    try:
        pm.model_to_graphviz(model)
    except Exception:  # pragma: no cover - graphviz optional
        pass
    return model


def present_group_effects(idata) -> list[str]:
    """Return the coord names that received a hierarchical ``<coord>_effect`` in the fit.

    Derived from the posterior rather than a hard-coded list so it tracks whichever
    of :data:`_FUSED_KALMAN_GROUP_EFFECTS` survived the panel-coord intersection.
    """
    post = getattr(idata, 'posterior', idata)
    return [v[:-len('_effect')] for v in post.data_vars if str(v).endswith('_effect')]


# --- Decision latent (single source of truth) ---------------------------------
# The per-ISIN quantity every decision consumer de-standardises: the screen, the
# price-target Monte-Carlo, the risk book, the analytics export and the §13b
# plots. Since the local-level state landed this is the FILTERED level at the
# final (snapshot) time step, ``state_now`` — not ``risk_adj_return``, which is
# now only the t=0 structural anchor of the walk. They coincide exactly when
# T == 1 (or the state is pinned off), so the fallback is not a degraded path.
#
# The fallback exists for idata produced BEFORE the state layer (archived
# NetCDF artifacts, the notebook twin mid-migration), which carry
# ``risk_adj_return`` and no ``state_now``.
KALMAN_SCREEN_LATENT: str = 'state_now'
KALMAN_SCREEN_LATENT_FALLBACK: str = 'risk_adj_return'


def resolve_screen_latent(group: "xr.Dataset") -> "xr.DataArray":
    """Return the decision latent from a posterior/prior group.

    Resolves :data:`KALMAN_SCREEN_LATENT` (``state_now``, the filtered level at
    the snapshot), falling back to :data:`KALMAN_SCREEN_LATENT_FALLBACK`
    (``risk_adj_return``, the t=0 anchor) for pre-state-layer inference objects.

    Parameters
    ----------
    group
        A posterior or prior group (``idata.posterior`` / ``idata.prior``).

    Returns
    -------
    xarray.DataArray
        Per-``isin`` latent draws over ``(chain, draw, isin)``.

    Raises
    ------
    KeyError
        If neither name is present.
    """
    if KALMAN_SCREEN_LATENT in group:
        return group[KALMAN_SCREEN_LATENT]
    if KALMAN_SCREEN_LATENT_FALLBACK in group:
        logger.debug(
            '%r absent; falling back to %r (pre-state-layer inference object).',
            KALMAN_SCREEN_LATENT, KALMAN_SCREEN_LATENT_FALLBACK)
        return group[KALMAN_SCREEN_LATENT_FALLBACK]
    raise KeyError(
        f'Neither {KALMAN_SCREEN_LATENT!r} nor {KALMAN_SCREEN_LATENT_FALLBACK!r} '
        f'is present in the group (have: {sorted(group.data_vars)[:12]}...).')


def _panel_response_stats(panel: KalmanPanelInputs) -> dict[str, tuple[float, float]]:
    """Per-response-series ``(mean, std)`` reproducing ``prepare_kalman_panel_inputs``.

    The fused model standardises each ``y_series`` column before fitting, so the
    per-ISIN baseline ``mu_isin`` / ``state_now`` lives on the dimensionless
    standardised scale. These stats invert that standardisation back onto a chosen
    response series (the primary ``feat_log_uplift`` target) for human-facing
    screening readouts.

    The moments are read from :attr:`KalmanPanelInputs.response_mean` /
    ``response_std`` — the values the panel builder ACTUALLY applied — so the
    inverse is exact by construction.

    .. note::

       This previously recomputed the moments by tiling the **snapshot** column
       across ``T``, on the (then-correct, now-false) reasoning that "tiling
       across ``T`` leaves the moments unchanged". That holds only for the
       removed tile-based panel. With ``history_lookbacks`` the primary series is
       built from genuine ``price_target_{lb}_ago`` / ``price_{lb}_ago`` trails
       and standardised on the **pooled (isin × time)** moments, which differ
       from the snapshot's — measured on the 2026-08 6 401-name T=4 run, pooled
       ``(0.20754, 0.24939)`` vs snapshot ``(0.22475, 0.24163)``. Inverting with
       the snapshot moments inflated every exported ``expected_upside`` /
       ``expected_pt`` by **+1.5 to +2.3 percentage points**, biasing the screen,
       the CVaR risk book and ``analytics.kalman_filtered_price_targets`` alike.

    Falls back to the legacy snapshot-tiling computation only for panels built
    before those fields existed (e.g. an unpickled pre-0.9.9.14 object), logging
    a warning so a silently-biased readout can never pass unnoticed.
    """
    stats: dict[str, tuple[float, float]] = {}
    recorded_mean = np.asarray(getattr(panel, 'response_mean', ()), dtype='float64')
    recorded_std = np.asarray(getattr(panel, 'response_std', ()), dtype='float64')
    have_recorded = (recorded_mean.size == len(panel.response_names)
                     and recorded_std.size == len(panel.response_names))
    if not have_recorded:
        logger.warning(
            'KalmanPanelInputs carries no fit-time response moments; falling back '
            'to snapshot-tiling, which is BIASED on a T>1 history panel. Rebuild '
            'the panel with prepare_kalman_panel_inputs to get the exact inverse.')

    T = panel.Y.shape[1]
    for d, col in enumerate(panel.response_names):
        if have_recorded:
            mean, std = float(recorded_mean[d]), float(recorded_std[d])
        else:
            v = panel.frame[col].astype('float64').to_numpy()
            tiled = np.tile(v[:, None], (1, T)).reshape(-1)
            mean, std = float(np.mean(tiled)), float(np.std(tiled))
            if not np.isfinite(mean):
                mean = (float(np.nanmean(tiled))
                        if np.isfinite(np.nanmean(tiled)) else 0.0)
            if not np.isfinite(std) or std <= 1e-6:
                std = float(np.nanstd(tiled))
        # Same guards the builder applies, so a degenerate series maps through
        # the identity rather than dividing by ~0.
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std <= 1e-6:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def panel_posterior_upside(
        idata, panel: KalmanPanelInputs,
        *, source: str = 'posterior',
) -> tuple[xr.DataArray, xr.DataArray]:
    """De-standardise the decision latent into ``(expected_upside, expected_pt)`` draws.

    The latent is resolved through :func:`resolve_screen_latent` — ``state_now``,
    the local-level state's **filtered** level at the snapshot time step, falling
    back to ``risk_adj_return`` (the t=0 anchor) for pre-state-layer inference
    objects. It is a per-ISIN shift on the standardised response scale shared
    across the ``D`` series. Mapping it through the primary ``feat_log_uplift``
    series' standardisation recovers an interpretable implied-upside (decimal)
    per posterior draw, and ``expected_pt = last_price * exp(log_uplift)`` lifts
    it to price units.

    Parameters
    ----------
    idata
        Fitted inference object (or one carrying a ``prior`` group when
        ``source='prior'``).
    panel
        The :class:`KalmanPanelInputs` the model was fit on.
    source
        ``'posterior'`` (default) or ``'prior'`` — which group to read
        ``risk_adj_return`` from.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        ``(expected_upside, expected_pt)`` over ``(chain, draw, isin)``.
    """
    group = getattr(idata, source)
    rar = resolve_screen_latent(group)
    stats = _panel_response_stats(panel)
    if 'feat_log_uplift' in stats:
        key = 'feat_log_uplift'
    elif 'feat_implied_upside' in stats:
        key = 'feat_implied_upside'
    else:
        key = panel.response_names[0]
    mean, std = stats[key]
    latent = mean + rar * std  # de-standardised baseline on the primary scale
    last = xr.DataArray(
        panel.frame['last_price'].astype('float64').to_numpy(),
        dims='isin', coords={'isin': rar.coords['isin']},
    )
    if key == 'feat_log_uplift':
        # ``latent`` is a log-uplift: exp() makes the price target positive by
        # construction and ``expm1`` recovers the decimal expected upside.
        eu = np.expm1(latent).rename('expected_upside')
        ept = (last * np.exp(latent)).rename('expected_pt')
    else:
        # Legacy decimal-upside scale: floor the gross multiple at a small
        # positive value so a deeply-negative baseline cannot drive the
        # reconstructed price target non-positive.
        eu = latent.rename('expected_upside')
        ept = (last * np.maximum(1.0 + eu, 1e-3)).rename('expected_pt')
    return eu, ept


@dataclass(frozen=True, eq=False)
class ScreenContext:
    """Screening artifacts derived from the fused-panel posterior.

    Attributes
    ----------
    eu, ept : xarray.DataArray
        Posterior ``expected_upside`` (decimal) / ``expected_pt`` (price) draws over
        ``(chain, draw, isin)``, de-standardised via :func:`panel_posterior_upside`.
    results : pandas.DataFrame
        Per-ISIN screening table (sorted by expected upside), consumed by §13/§14.
    mc_summary : pandas.DataFrame
        Structural-TS Monte-Carlo per-ISIN risk-adjusted-return summary
        (:func:`summarize_mc_returns`).
    """

    eu: xr.DataArray
    ept: xr.DataArray
    results: pd.DataFrame
    mc_summary: pd.DataFrame


# =============================================================================
# 6. Prior predictive checks
# =============================================================================
def run_prior_predictive(model: "pm.Model", panel: KalmanPanelInputs,
                         config: Optional[KalmanRunConfig] = None):
    """Sample the fused-model prior and sanity-check the implied-upside / risk scale.

    The fused MvGRW baseline ``risk_adj_return`` is de-standardised onto the primary
    ``feat_implied_upside`` series (:func:`panel_posterior_upside`) so the prior over
    implied upside can be eyeballed against the empirical consensus upside. The
    ``achieve_prob`` (= ``sigmoid(risk_adj_return)``) and heteroscedastic
    ``sigma_isin`` priors are shown alongside so the §5b refinements are visible
    before any data is seen.
    """
    # Also capture the global hyper-parameters so §9 can overlay prior vs
    # posterior marginals (azp.plot_prior_posterior) — only vars the fitted
    # parameterization actually materialises (e.g. risk_loading exists only
    # when its penalty prior scale is > 0).
    cfg = config if config is not None else get_run_config()
    _hyper = [v for v in (*FUSED_SCALAR_VARS, 'beta') if v in model.named_vars]
    # ``state_now`` (the filtered decision latent) is what panel_posterior_upside
    # de-standardises, so it must be drawn for the prior implied-upside panel;
    # ``state_path`` is skipped — an (isin, time) tensor over prior draws is large
    # and adds nothing the terminal state does not already show.
    var_names = [v for v in ('expected_return', 'risk_adj_return', 'state_now',
                             'achieve_prob', 'sigma_isin')
                 if v in model.named_vars]
    var_names.extend(_hyper)
    with model:
        prior_idata = pm.sample_prior_predictive(
            draws=cfg.prior_draws, var_names=var_names,
            random_seed=cfg.random_seed, return_inferencedata=True,
            compile_kwargs=get_pytensor_compile_kwargs(),
        )

    eu_prior, _ = panel_posterior_upside(prior_idata, panel, source='prior')
    prior_up = eu_prior.values.reshape(-1)
    emp_up = (panel.frame['observed_pt'] / panel.frame['last_price'] - 1.0).to_numpy()

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.07,
        subplot_titles=('Prior implied upside vs empirical',
                        'Prior achieve_prob (sigmoid risk-adj return)',
                        'Prior sigma_isin  (heteroscedastic scale)'))
    # Percent boundary: prior/empirical upside are decimals — scale ×100 here so
    # this panel matches every other upside axis in the module (0.9.9.11 fix of
    # the sole decimal-scale figure).
    fig.add_trace(go.Histogram(
        x=np.clip(prior_up, -1, 2) * 100.0, histnorm='probability density', nbinsx=80,
        marker=dict(color=_hex_to_rgba(C_POSTERIOR, 0.6)),
        hovertemplate='prior upside = %{x:.0f}%<extra></extra>',
        name='prior expected_upside'), row=1, col=1)
    # Empirical distribution as a stepped outline (matplotlib ``histtype='step'`` analogue).
    _emp = np.clip(emp_up[np.isfinite(emp_up)], -1, 2) * 100.0
    _eh, _ee = np.histogram(_emp, bins=80, density=True)
    _ec = 0.5 * (_ee[:-1] + _ee[1:])
    fig.add_trace(go.Scatter(
        x=_ec, y=_eh, mode='lines', line=dict(color=C_OBSERVED, width=1.5, shape='hvh'),
        hovertemplate='empirical upside = %{x:.0f}%<extra></extra>',
        name='empirical observed_pt/last_price − 1'), row=1, col=1)
    _add_ref_line(fig, x=0, kind='zero', row=1, col=1)
    fig.update_xaxes(title_text='implied upside (%)', ticksuffix='%', row=1, col=1)
    fig.update_yaxes(title_text='density', row=1, col=1)

    # Model-A achievement probability prior (= sigmoid(risk_adj_return)).
    ap = prior_idata.prior['achieve_prob'].values.reshape(-1)
    fig.add_trace(go.Histogram(
        x=ap, histnorm='probability density', nbinsx=60,
        marker=dict(color=_hex_to_rgba(C_FORECAST, 0.7)),
        hovertemplate='P(achieve) = %{x:.0%}<extra></extra>',
        name='achieve_prob', showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text='P(achieve)', range=[0, 1], tickformat='.0%',
                     row=1, col=2)

    # Heteroscedastic measurement scale sigma_isin = sigma_base * (1 + cv) / sqrt(n).
    si = prior_idata.prior['sigma_isin'].values.reshape(-1)
    si = si[np.isfinite(si)]
    fig.add_trace(go.Histogram(
        x=np.clip(si, 0, np.nanpercentile(si, 99)), histnorm='probability density',
        nbinsx=60, marker=dict(color=_hex_to_rgba(C_ACCENT, 0.7)),
        hovertemplate='σ_isin = %{x:.3f}<extra></extra>',
        name='sigma_isin', showlegend=False), row=1, col=3)
    fig.update_xaxes(title_text='sigma_isin (standardised scale)', row=1, col=3)
    fig.update_layout(showlegend=True, legend=dict(font_size=_LEGEND_FONT_SIZE))
    _render_plotly(fig, height=H_SHORT)

    print(f'Prior expected_upside: median={np.nanmedian(prior_up):.3f}, '
          f'p01/p99=({np.nanpercentile(prior_up, 1):.2f}, {np.nanpercentile(prior_up, 99):.2f}); '
          f'empirical median={np.nanmedian(emp_up):.3f}.')
    return prior_idata


# =============================================================================
# 7. Posterior inference (NUTS)
# =============================================================================
def sample_posterior(model: "pm.Model", prior_idata, *, cores: Optional[int] = None,
                     panel: Optional[KalmanPanelInputs] = None,
                     config: Optional[KalmanRunConfig] = None):
    """Sample the posterior, trying nutpie -> numpyro -> pymc in priority order.

    Merges the prior groups into the posterior idata for one-object downstream access.

    Parameters
    ----------
    model
        The built PyMC model to sample.
    prior_idata
        Prior-predictive groups to merge into the returned posterior ``DataTree``.
    panel
        When given, the drift-feature aliases (``panel.drift_names``) plus their
        catalogue metadata are stamped onto ``constant_data['drift_features']``
        via ``stamp_feature_provenance`` (best-effort; the ``pm.Data`` container
        of the same name lands in ``constant_data`` automatically).
    config
        Optional :class:`KalmanRunConfig` supplying the NUTS budget
        (``draws`` / ``tune`` / ``chains`` / ``cores`` / ``target_accept`` /
        ``random_seed``); defaults to :func:`get_run_config`.
    cores
        Number of chains to run in parallel; overrides ``config.cores`` when
        given. The config default is ``1`` (kernel-safe,
        chains run sequentially); pass ``cores=4`` on the standalone script /
        CLI path, where the native (nutpie numba/rust) sampler runs
        happily in parallel. **Keep ``cores=1`` in an IDE-managed Jupyter kernel
        (e.g. PyCharm / DataSpell) on Windows:** launching nutpie's parallel native
        worker threads inside the embedded kernel can crash the kernel process
        outright — a native crash that no Python ``try``/``except`` can catch, so
        the IDE only reports *"Connection to IDE-Managed Server is lost"* with no
        traceback. Running chains sequentially (``cores=1``) removes that parallel
        native-thread launch and keeps the kernel alive; per-chain nutpie speed is
        unaffected, only wall-clock (chains no longer overlap).
    """
    # After the §4 structural fix (inert tiled time axis collapsed to a single
    # cross-sectional slice) AND the identifiability fixes in
    # ``build_fused_kalman_pt_model`` the cross-sectional variance structure is now
    # genuinely identified. The decisive fix is structural, not budgetary: the
    # per-ISIN signal latent (``sigma_expected_return`` + ``z_expected_return``)
    # was dropped (``expected_return`` is now the deterministic structural mean
    # ``mu_reg``), and — the last unidentified variance component — the learned
    # group scale ``sigma_group`` was removed: the crossed effects are now
    # fixed-scale ``ZeroSumNormal`` regularized effects (``GROUP_EFFECT_SCALE``),
    # with ``industry`` dropped. Previously those variance components formed an
    # unidentified partition over the same per-ISIN dispersion (every learned scale
    # stuck at R-hat ≈ 1.5–4.5, ESS ≈ 4–7) — a sampling budget could never have
    # fixed that, since a single collapsed slice (T=1) cannot identify a
    # between-group variance. With the model well-conditioned the cross-section
    # adapts fast: draws=1000, tune=1000, target_accept=0.90, chains=4, cores=4
    # (all chains in parallel) clears ESS > 400 comfortably. Since the per-time
    # direct-intercept reparameterisation a genuine (isin, time) panel
    # (history_lookbacks, T > 1) shares this geometry and needs no extra tune.
    # ``draws=1000`` clears the ESS > 400 gate with margin; ``target_accept`` is
    # relaxed 0.97 → 0.9 now that the degenerate-series ICM ridge is removed (the
    # coverage guard in ``prepare_kalman_panel_inputs`` + the sign-fixed
    # ``mu_isin_loading`` prior). The 0.97 setting was a band-aid that, against that
    # ridge, drove the step size toward 0 and froze every chain; with a
    # well-conditioned posterior the default-ish 0.9 mixes fast.
    cfg = config if config is not None else get_run_config()
    # No extra tune budget for T > 1: since the per-time direct-intercept
    # reparameterisation (2026-08-01) the genuine (isin, time) panel shares the
    # T=1 geometry — the former tune >= 2000 bump targeted the removed
    # GRW-deviation ridge and only added wall-clock.
    sample_kwargs = dict(
        draws=cfg.draws, tune=cfg.tune, chains=cfg.chains,
        cores=cores if cores is not None else cfg.cores,
        target_accept=cfg.target_accept, random_seed=cfg.random_seed,
        progressbar=True, return_inferencedata=True,
        idata_kwargs={"log_likelihood": False},
    )

    candidate_samplers = []
    if _ilu.find_spec("nutpie") is not None:
        candidate_samplers.append("nutpie")
    if _ilu.find_spec("numpyro") is not None:
        candidate_samplers.append("numpyro")
    candidate_samplers.append("pymc")  # always available — pure-Python NUTS
    print(f"Available NUTS samplers (in priority order): {candidate_samplers}")

    sampling_errors = []
    idata = None
    for sampler in candidate_samplers:
        try:
            with model:
                idata = pm.sample(
                    nuts_sampler=sampler,
                    compile_kwargs=get_pytensor_compile_kwargs(),
                    **sample_kwargs,
                )
            print(f"Sampled successfully with nuts_sampler={sampler!r}.")
            break
        except Exception as e:  # pragma: no cover - environment-dependent fallback
            sampling_errors.append((sampler, repr(e)))
            print(f"nuts_sampler={sampler!r} failed: {e!r}")
            # Every candidate sampler ultimately relies on PyTensor's C backend.
            # A CompileError is therefore a toolchain/interpreter problem, not a
            # sampler problem — retrying the other samplers only reproduces the
            # identical failure. Stop early and raise an actionable diagnostic.
            if type(e).__name__ == "CompileError":
                raise RuntimeError(
                    "PyTensor failed to compile its C backend, so no NUTS "
                    "sampler can run. This is an environment problem, not a "
                    "model problem.\n"
                    f"  Python: {sys.version.split()[0]} "
                    f"(interpreter: {sys.executable})\n"
                    "  Likely causes:\n"
                    "    * Python 3.14 is not yet supported by the compiled "
                    "PyMC/PyTensor stack — recreate the venv on Python 3.11/3.12.\n"
                    "    * MSYS2 UCRT64 g++ linking against an MSVC-built "
                    "python*.dll (ABI mismatch). Prefer a conda-forge env with a "
                    "matched toolchain.\n"
                    "  Verify with: import pytensor; print(pytensor.config.cxx)\n"
                    f"  Original CompileError: {e!r}"
                ) from e

    if idata is None:
        raise RuntimeError(
            "All NUTS samplers failed:\n"
            + "\n".join(f"  - {s}: {err}" for s, err in sampling_errors)
        )

    # Feature provenance: record which catalogue aliases back the
    # ``drift_feature`` dim (plus their data_type / category metadata) on the
    # constant_data group, so exported idata is self-describing. Best-effort —
    # a missing DB / constant_data group must never fail the sampling path.
    if panel is not None:
        try:
            stamp_feature_provenance(
                idata, "drift_features", panel.drift_names,
                load_feature_metadata_from_db(),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not stamp drift-feature provenance: %s", exc)

    return extend_datatree(idata, prior_idata)


# =============================================================================
# 8. Posterior predictive checks
# =============================================================================
def run_posterior_predictive(model: "pm.Model", idata, panel: KalmanPanelInputs) -> None:
    """Sample the fused-panel posterior predictive and draw calibration diagnostics.

    The fused likelihood ``target_pct_obs`` is the standardised ``(isin, time,
    y_series)`` response tensor, so the checks pool the replicated draws against the
    observed standardised responses: an ECDF overlay, a t-stat calibration check,
    a per-``y_series`` 94% coverage table, and (best-effort) a PIT calibration
    ECDF via ``azp.plot_ecdf_pit`` on manually computed PIT values.
    """
    with model:
        pm.sample_posterior_predictive(
            idata, extend_inferencedata=True,
            random_seed=RANDOM_SEED, progressbar=True,
            compile_kwargs=get_pytensor_compile_kwargs(),
        )

    # (The former arviz plot_ppc_dist ECDF overlay was removed as a duplicate of
    # the pooled hand-built ECDF overlay below — 0.9.9.11 consolidation.)

    # (a2) t-stat calibration (gallery: plot_ppc_tstat) — where the observed
    # summary statistic sits inside the replicated T(y_rep) distribution. The
    # mean checks the location calibration; sd checks the dispersion the
    # heteroscedastic sigma_isin / sigma_series structure is meant to absorb.
    for _stat in ('mean', 'std'):
        try:
            pc_t = azp.plot_ppc_tstat(
                idata, var_names=["target_pct_obs"], t_stat=_stat,
                backend="plotly",
                figure_kwargs=_azp_figure_kwargs(380, width_frac=0.6),
            )
            pc_t.add_title(f'PPC t-stat calibration — T = {_stat}')
            _safe_show(pc_t)
        except Exception as e:  # pragma: no cover - best-effort diagnostic
            print(f"arviz plot_ppc_tstat({_stat!r}) skipped: {e!r}")

    pp = idata.posterior_predictive['target_pct_obs']
    obs = idata.observed_data['target_pct_obs']

    # (b) Robust pooled ECDF overlay (observed standardised response vs predictive draws).
    pp_stack = pp.stack(sample=('chain', 'draw'))
    n_samp = pp_stack.sizes['sample']
    pick = np.linspace(0, n_samp - 1, min(60, n_samp)).astype(int)
    obs_flat = np.asarray(obs.values).reshape(-1)
    obs_flat = obs_flat[np.isfinite(obs_flat)]
    obs_sorted = np.sort(obs_flat)
    ecdf_y = np.linspace(0, 1, len(obs_sorted))

    fig = go.Figure()
    _pp_rgba = _hex_to_rgba(C_POSTERIOR, 0.12)
    for s in pick:
        rep = np.asarray(pp_stack.isel(sample=s).values).reshape(-1)
        rep = np.sort(rep[np.isfinite(rep)])
        fig.add_trace(go.Scatter(x=rep, y=np.linspace(0, 1, len(rep)), mode='lines',
                                 line=dict(color=_pp_rgba, width=0.8),
                                 hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=obs_sorted, y=ecdf_y, mode='lines',
                             line=dict(color=C_OBSERVED, width=2.2), name='observed',
                             hovertemplate=('standardised response = %{x:.2f}<br>'
                                            'ECDF = %{y:.0%}<extra></extra>')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color=C_POSTERIOR, width=1.2),
                             name='posterior-predictive draws'))
    fig.update_xaxes(range=[float(np.nanpercentile(obs_flat, 0.5)),
                            float(np.nanpercentile(obs_flat, 99.5))],
                     title_text='standardised response  (target_pct_obs)')
    fig.update_yaxes(title_text='ECDF')
    fig.update_layout(title='Posterior-predictive ECDF overlay — fused MvGRW panel')
    _render_plotly(fig, height=H_STD)

    # (c) Per-y_series 94% predictive-interval coverage — printed AND charted
    # against the 0.94 target so miscalibration is visible at a glance.
    lo = pp.quantile(_HDI_LO, dim=('chain', 'draw'))
    hi = pp.quantile(_HDI_HI, dim=('chain', 'draw'))
    inside = ((obs >= lo) & (obs <= hi))
    cover = inside.mean(('isin', 'time'))
    _tgt = _HDI_HI - _HDI_LO
    print(f'Per-y_series {_tgt:.0%} posterior-predictive coverage '
          f'(target ≈ {_tgt:.2f}):')
    _cov_names, _cov_vals = [], []
    for name in panel.response_names:
        try:
            c = float(cover.sel(y_series=name).values)
            print(f'  - {name:<24s}: {c:.2%}')
            _cov_names.append(str(name))
            _cov_vals.append(c)
        except Exception:
            continue
    if _cov_names:
        _target = _HDI_HI - _HDI_LO
        figc = go.Figure(go.Bar(
            x=_cov_vals, y=_cov_names, orientation='h',
            marker=dict(color=[C_POSTERIOR if abs(v - _target) <= 0.03
                               else C_HIGHLIGHT for v in _cov_vals]),
            hovertemplate='%{y}<br>coverage = %{x:.1%}<extra></extra>',
            name=f'{_target:.0%} PI coverage'))
        _add_ref_line(figc, x=_target, kind='zero',
                      annotation_text=f'target {_target:.0%}')
        figc.update_xaxes(title_text='share of observations inside the 94% PI',
                          range=[0, 1], tickformat='.0%')
        figc.update_layout(title='Posterior-predictive coverage by response series',
                           showlegend=False)
        _render_plotly(figc, height=max(240, 60 * len(_cov_names) + 160))

    # (c2) Per-TIME predictive-interval coverage — the calibration statistic that
    # actually tests
    # the local-level state. Coverage pooled over time can look fine while the
    # model is systematically over-confident at the oldest lookbacks and
    # over-dispersed at the snapshot (or vice versa); only a per-time breakdown
    # exposes that. A monotone drift across t is the signature of a state whose
    # innovation scale is mis-set — too small and the early slices fall outside
    # the interval, too large and the late ones are over-covered.
    if 'time' in inside.dims and inside.sizes['time'] > 1:
        cover_t = inside.mean(('isin', 'y_series')) if 'y_series' in inside.dims \
            else inside.mean('isin')
        _t_idx = [int(v) for v in np.asarray(cover_t['time'].values)]
        _t_vals = [float(v) for v in np.asarray(cover_t.values)]
        _target = _HDI_HI - _HDI_LO
        print(f'Per-time {_target:.0%} posterior-predictive coverage '
              f'(target ≈ {_target:.2f}; oldest → snapshot):')
        for _ti, _tv in zip(_t_idx, _t_vals):
            _flag = '' if abs(_tv - _target) <= 0.03 else '   <-- off target'
            print(f'  - t={_ti} : {_tv:.2%}{_flag}')
        figt = go.Figure(go.Scatter(
            x=_t_idx, y=_t_vals, mode='lines+markers',
            marker=dict(size=8, color=C_POSTERIOR), line=dict(width=1.8),
            hovertemplate='t = %{x}<br>coverage = %{y:.1%}<extra></extra>',
            name=f'{_target:.0%} PI coverage'))
        _add_ref_line(figt, y=_target, kind='zero',
                      annotation_text=f'target {_target:.0%}')
        figt.update_xaxes(title_text='time index (lookback anchor, oldest → now)',
                          tickmode='array', tickvals=_t_idx)
        figt.update_yaxes(title_text='share inside the 94% PI', tickformat='.0%')
        figt.update_layout(
            title='Posterior-predictive coverage by time step — local-level state',
            showlegend=False)
        _render_plotly(figt, height=H_SHORT)

    # (d) PIT calibration ECDF (best-effort; band-hugging = calibrated).
    # ``azp.plot_ppc_pit`` cannot digest the multi-dim (isin, time, y_series)
    # response tensor — it coerces multi-element sub-arrays to scalars and dies
    # with ``TypeError: only 0-dimensional arrays can be converted to Python
    # scalars``. Instead the PIT values are computed directly — u_i =
    # P(y_rep <= y_obs_i) over the pooled (chain, draw) predictive draws, per
    # ``y_series`` — and handed to ``azp.plot_ecdf_pit``, the gallery visual
    # dedicated to pre-computed PIT samples (uniform when calibrated, drawn as a
    # Delta-ECDF with simultaneous confidence bands).
    try:
        series_dim = 'y_series' if 'y_series' in obs.dims else None
        series_vals = list(obs[series_dim].values) if series_dim else [None]

        any_pit = False
        for sv in series_vals:
            sel = {series_dim: sv} if sv is not None else {}
            pit = (pp.sel(sel) <= obs.sel(sel)).mean(('chain', 'draw'))
            pit_flat = np.asarray(pit.values, dtype='float64').ravel()
            pit_flat = pit_flat[np.isfinite(pit_flat)]
            if pit_flat.size < 10:
                continue
            _pit_ds = xr.Dataset({'pit': (('chain', 'draw'), pit_flat[None, :])})
            pc_pit = azp.plot_ecdf_pit(
                _pit_ds, var_names=['pit'], backend='plotly',
                figure_kwargs=_azp_figure_kwargs(400, width_frac=0.7),
            )
            title = (f'PPC PIT ECDF — {sv}' if sv is not None else 'PPC PIT ECDF')
            with contextlib.suppress(Exception):
                pc_pit.add_title(f'{title}  (n={pit_flat.size}; in-band = calibrated)')
            _safe_show(pc_pit)
            any_pit = True
        if not any_pit:
            print("PPC PIT calibration plot skipped: no response series resolved.")
    except Exception as e:  # pragma: no cover - diagnostic is best-effort
        print(f"PPC PIT calibration plot skipped: {e!r}")


# =============================================================================
# 9. MCMC diagnostics
# =============================================================================
def run_diagnostics(idata, panel: KalmanPanelInputs) -> None:
    """R-hat / ESS summary, divergences, trace / rank-dist / forest plots, plus
    the NUTS energy (E-BFMI), prior-vs-posterior contraction and ESS-evolution
    diagnostics.

    Targets the fused-model hyper-parameters: the global scalars
    (:data:`FUSED_SCALAR_VARS`), the per-coord hierarchical scales ``sigma_<coord>``,
    the drift slopes ``beta`` and the per-series coregion terms
    (:data:`FUSED_VECTOR_VARS`).
    """
    group_effects = present_group_effects(idata)
    posterior = idata.posterior
    requested = [*FUSED_SCALAR_VARS, *FUSED_VECTOR_VARS]
    for grp in group_effects:
        requested.extend([f'sigma_{grp}', f'{grp}_effect'])

    available, skipped = [], []
    for v in requested:
        if v not in posterior.data_vars:
            skipped.append((v, 'not in posterior'))
            continue
        da = posterior[v]
        non_sample_sizes = [da.sizes[d] for d in da.dims if d not in ('chain', 'draw')]
        if any(s == 0 for s in non_sample_sizes):
            skipped.append((v, f'empty dim(s): {dict(da.sizes)}'))
            continue
        available.append(v)

    if skipped:
        print('Skipping variables:')
        for name, reason in skipped:
            print(f'  - {name}: {reason}')
    if not available:
        raise RuntimeError('No non-empty variables to summarise.')

    summary = azs.summary(idata, var_names=available, round_to=4)
    # Explicit label: this frame is a curated SQL artifact
    # (``_SQL_EXPORT_ARTIFACTS``), so its stem must not drift if the section
    # gains or loses an earlier ``display()`` call.
    display(summary.sort_values('r_hat', ascending=False).head(100), label='table')

    # 9.2 Divergences and aggregated R-hat / ESS.
    n_div = int(idata.sample_stats['diverging'].sum())

    def _non_empty_vars(ds):
        keep = []
        for name, da in ds.data_vars.items():
            sizes = [da.sizes[d] for d in da.dims if d not in ('chain', 'draw')]
            if all(s > 0 for s in sizes):
                keep.append(name)
        return keep

    posterior = _posterior_dataset(idata)
    keep_vars = _non_empty_vars(posterior)
    # Drop variables that are CONSTANT across draws before the sweep. A
    # deterministically-anchored entry (the D>1 primary ICM loading / noise pinned
    # at 1.0; historically the structurally-zero ``beta_t``) has zero within- AND
    # between-chain variance, so arviz-stats evaluates 0/0 and emits
    # "RuntimeWarning: invalid value encountered in scalar divide" before returning
    # NaN. The nan-aware reductions below already ignored the NaN, but the warning
    # still leaked into every run log. Excluding the constants removes it at source
    # and keeps the sweep over quantities where R-hat is actually defined.
    constant_vars = set(_degenerate_posterior_vars(idata, keep_vars))
    swept_vars = [v for v in keep_vars if v not in constant_vars]
    if not swept_vars:  # pragma: no cover - a wholly-degenerate posterior
        raise RuntimeError('Every posterior variable is constant across draws.')
    rhat_ds = azs.rhat(posterior[swept_vars])
    ess_ds = azs.ess(posterior[swept_vars], method='bulk')

    ess_tail_ds = azs.ess(posterior[swept_vars], method='tail')

    # Use nan-aware reductions: deterministically-anchored entries (e.g. the
    # primary-series ICM loading ``mu_isin_loading``/noise ``sigma_series`` pinned
    # at 1.0) are constant across draws, so arviz returns a NaN R-hat for them
    # (0/0 within/between variance). Those NaNs are expected and must not mask the
    # genuine worst-case R-hat / ESS — ``np.nanmax`` / ``np.nanmin`` skip them.
    max_rhat = float(np.nanmax([float(rhat_ds[v].max()) for v in rhat_ds.data_vars]))
    min_ess = float(np.nanmin([float(ess_ds[v].min()) for v in ess_ds.data_vars]))

    # Convergence gates (Vehtari et al. 2021): R-hat < 1.01 and ESS > 400.
    _RHAT_GATE, _ESS_GATE = 1.01, 400.0
    grp_keys = [f'sigma_{g}' for g in group_effects if f'sigma_{g}' in rhat_ds.data_vars]
    # Report the scale together with its (non-centred) deviation vector so the
    # funnel partners are visible side by side.
    grp_report: dict[str, tuple[float, float, float]] = {}
    for v in grp_keys:
        grp_report[v] = (float(rhat_ds[v].max()),
                         float(ess_ds[v].min()),
                         float(ess_tail_ds[v].min()))

    print(f'Divergences: {n_div}'
          + ('  <-- non-zero: inspect the funnel partners below' if n_div else ''))
    print(f'Max R-hat:   {max_rhat:.4f}  (gate < {_RHAT_GATE})')
    print(f'Min ESS:     {min_ess:.1f}  (gate > {_ESS_GATE:.0f})')

    if grp_report:
        print('Group-effect scale diagnostics  (R-hat, ESS-bulk, ESS-tail, status):')
        for v, (r, e_bulk, e_tail) in grp_report.items():
            ok = (r < _RHAT_GATE) and (e_bulk > _ESS_GATE)
            status = 'PASS' if ok else 'WARN'
            print(f'  - {v:>20s}: r_hat={r:6.3f}  ess_bulk={e_bulk:7.1f}  '
                  f'ess_tail={e_tail:7.1f}  [{status}]')

        # Variance-partition readout: each genuine per-coord between-group sd
        # ``sigma_<col>`` (the empirical sd of that coord's fixed-scale
        # ``ZeroSumNormal`` effect — well-identified, no longer a stuck shared
        # ``sigma_group`` scalar) vs the residual/measurement base scale
        # ``sigma_base``. A small ``sigma_<col>`` signals little between-group
        # signal for that coord; a larger one signals real between-group structure.
        partition = {
            name: float(posterior[name].mean())
            for name in (*grp_keys, 'sigma_base')
            if name in posterior.data_vars
        }
        if partition:
            denom = sum(partition.values())
            print('Variance partition (posterior-mean scales, share of total):')
            for name, m in partition.items():
                share = f'  ({m / denom * 100:4.0f}%)' if denom > 0 else ''
                print(f'  - {name:>34s}: {m:7.4f}{share}')
            # Same partition as a single stacked bar — the share of the total
            # posterior-mean scale attributable to each between-group coord vs
            # the residual/measurement base.
            if denom > 0 and _HAS_PLOTLY:
                figv = go.Figure()
                for _i, (name, m) in enumerate(
                        sorted(partition.items(), key=lambda kv: -kv[1])):
                    figv.add_trace(go.Bar(
                        x=[m / denom * 100.0], y=['scale share'], orientation='h',
                        name=name,
                        hovertemplate=(f'{name}<br>share = %{{x:.0f}}%'
                                       f'<br>scale = {m:.4f}<extra></extra>')))
                figv.update_layout(
                    barmode='stack',
                    title='Variance partition — between-group scales vs sigma_base',
                    legend=dict(font_size=_LEGEND_FONT_SIZE, title_text='scale'),
                    height=220)
                figv.update_xaxes(title_text='share of total posterior-mean scale (%)',
                                  ticksuffix='%', range=[0, 100])
                figv.update_yaxes(showticklabels=False)
                _render_plotly(figv)

    # Flag parameters whose draws collapsed to a constant / non-finite spike: these
    # are exactly the slices that would otherwise emit arviz-stats' single-value KDE
    # warning, so naming them here turns an anonymous warning into a model signal.
    _degenerate = _degenerate_posterior_vars(
        idata, [*FUSED_SCALAR_VARS, *FUSED_VECTOR_VARS, *grp_keys,
                *(f'{g}_effect' for g in group_effects)],
    )
    if _degenerate:
        print('Degenerate (constant / non-finite) posterior vars '
              '[densities shown as delta spikes]:')
        for v in _degenerate:
            print(f'  - {v}')

    # 9.3 Trace + marginal densities. plot_trace can crash when a single call
    # mixes variables whose non-sample dims differ, so we split scalar vs
    # vector vars and keep a per-variable fallback. Sizing budgets the FACET
    # count (one facet per vector element) via the shared _diag_figure_kwargs
    # helper — sizing by len(_vars) squeezed beta's ~40 panels into ~300 px.
    post_trace = idata.posterior
    requested = [*FUSED_SCALAR_VARS, *FUSED_VECTOR_VARS,
                 *(f'sigma_{g}' for g in group_effects)]
    trace_vars = [v for v in requested if v in post_trace.data_vars]

    def _extra_dims(_v):
        return [d for d in post_trace[_v].dims if d not in ('chain', 'draw')]

    scalar_vars = [v for v in trace_vars if not _extra_dims(v)]
    vector_vars = [v for v in trace_vars if _extra_dims(v)]

    # The ``_quiet_degenerate_density`` guard suppresses arviz-stats' single-value
    # KDE UserWarning (density.py:672) that a collapsed/stuck slice would emit; the
    # numeric R-hat/ESS report and the degenerate-var list above already carry that
    # signal, so the raw per-slice warnings are pure noise here.
    def _show_trace(_vars):
        with _quiet_degenerate_density():
            pc = azp.plot_trace(
                idata, var_names=_vars, backend='plotly',
                figure_kwargs=_diag_figure_kwargs(post_trace, _vars),
            )
            with contextlib.suppress(Exception):
                pc.add_title('Trace + marginal density — '
                             + ', '.join(str(v) for v in _vars[:4])
                             + (' …' if len(_vars) > 4 else ''))
            _polish_facet_axes(pc)
            _safe_show(pc)

    if scalar_vars:
        try:
            _show_trace(scalar_vars)
        except ValueError as exc:
            print(f'Combined scalar trace failed ({exc}); plotting per variable.')
            for sv in scalar_vars:
                _show_trace([sv])
    for vv in vector_vars:
        _show_trace([vv])
    if not scalar_vars and not vector_vars:
        print('No trace-eligible variables available.')

    # 9.3b Fractional-rank Delta-ECDF plots (same scalar/vector split, but
    # sized per VARIABLE row via _rank_dist_figure_kwargs: compact
    # plot_rank_dist overlays vector elements in one row instead of fanning
    # them into facets, so the trace grid's facet-count budget over-sizes it).
    def _show_rank_dist(_vars):
        with _quiet_degenerate_density():
            pc = azp.plot_rank_dist(
                idata, var_names=_vars, backend='plotly',
                figure_kwargs=_rank_dist_figure_kwargs(_vars),
            )
            pc.add_title('Fractional-rank Δ-ECDF — ' + ', '.join(_vars))
            _polish_facet_axes(pc)
            _safe_show(pc)

    if scalar_vars:
        try:
            _show_rank_dist(scalar_vars)
        except ValueError as exc:
            print(f'Combined scalar rank-dist failed ({exc}); plotting per variable.')
            for sv in scalar_vars:
                _show_rank_dist([sv])
    for vv in vector_vars:
        _show_rank_dist([vv])

    # 9.4 Forest of hierarchical group-effect scales, drift slopes and GRW innovations.
    forest_vars = [f'sigma_{g}' for g in group_effects
                   if f'sigma_{g}' in idata.posterior.data_vars]
    if forest_vars:
        # One forest row per non-sample element (vector vars span multiple rows).
        _n_rows = _n_facets(post_trace, forest_vars)
        with _quiet_degenerate_density():
            pc = azp.plot_forest(
                idata, var_names=forest_vars, combined=True,
                figure_kwargs=_azp_figure_kwargs(_forest_height_px(_n_rows)),
            )
            pc.add_title('Group-effect scales (sigma_<coord>), drift slopes (beta) '
                         'and GRW innovation scales')
            _safe_show(pc)
    else:
        print('No group-effect / beta variables in posterior - skipped.')

    # 9.5 NUTS energy diagnostic (E-BFMI): the marginal vs transition energy
    # overlap. Poor overlap (low BFMI) flags a sampler that cannot explore the
    # heavy Student-t tails — complementary to R-hat / ESS above.
    try:
        pc_e = azp.plot_energy(idata, backend='plotly',
                               figure_kwargs=_azp_figure_kwargs(400, width_frac=0.7))
        pc_e.add_title('NUTS energy — marginal vs transition (E-BFMI check)')
        _safe_show(pc_e)
    except Exception as exc:  # pragma: no cover - sampler-dependent
        print(f'plot_energy skipped: {exc!r}')

    # 9.6 Prior -> posterior contraction of the global hyper-parameters. The
    # prior group carries these vars since run_prior_predictive samples them
    # (FUSED_SCALAR_VARS + beta), so the overlay shows how much the fused
    # likelihood actually updates sigma_base / nu / risk & size loadings and
    # the drift slopes — a direct "did the data speak?" readout per parameter.
    try:
        _prior_ds = getattr(idata, 'prior', None)
        _prior_vars = set(_prior_ds.data_vars) if _prior_ds is not None else set()
        _pp_vars = [v for v in (*FUSED_SCALAR_VARS, 'beta')
                    if v in posterior.data_vars and v in _prior_vars]
        if _pp_vars:
            with _quiet_degenerate_density():
                pc_pp = azp.plot_prior_posterior(
                    idata, var_names=_pp_vars, backend='plotly',
                    figure_kwargs=_diag_figure_kwargs(posterior, _pp_vars),
                )
                pc_pp.add_title('Prior vs posterior — global hyper-parameters '
                                '(contraction = information gained from data)')
                _polish_facet_axes(pc_pp)
                _safe_show(pc_pp)
        else:
            print('plot_prior_posterior skipped: no hyper-parameter overlaps the '
                  'prior group (re-run run_prior_predictive to capture them).')
    except Exception as exc:  # pragma: no cover - best-effort diagnostic
        print(f'plot_prior_posterior skipped: {exc!r}')

    # 9.7 ESS evolution: does the effective sample size keep growing linearly in
    # draws (healthy) or plateau (sticky chains)? Scalars only — the vector vars
    # would swamp the grid.
    if scalar_vars:
        try:
            pc_ess = azp.plot_ess_evolution(
                idata, var_names=scalar_vars, backend='plotly',
                figure_kwargs=_diag_figure_kwargs(post_trace, scalar_vars),
            )
            pc_ess.add_title('ESS evolution (bulk / tail) — should grow ~linearly '
                             'with draws')
            _polish_facet_axes(pc_ess)
            _safe_show(pc_ess)
        except Exception as exc:  # pragma: no cover - best-effort diagnostic
            print(f'plot_ess_evolution skipped: {exc!r}')


# =============================================================================
# 9b. Model comparison (ELPD / LOO)
# =============================================================================
def _subsample_panel(panel: KalmanPanelInputs, max_isins: int,
                     *, random_seed: int = RANDOM_SEED) -> KalmanPanelInputs:
    """Return ``panel`` restricted to at most ``max_isins`` randomly-chosen names.

    Model comparison needs a pointwise ``log_likelihood`` group, whose size is
    ``chains × draws × n_isin × T × D`` floats — ~820 MB for the full T=4 panel
    at 4×1000 draws over 6.4k ISINs, and it is paid once per arm. Comparing on a
    random ISIN subsample keeps the ELPD contrast meaningful while making the
    memory tractable.

    Parameters
    ----------
    panel
        The full panel.
    max_isins
        Upper bound on the retained ISIN count. Values ``<= 0`` or ``>= n_isin``
        return ``panel`` unchanged.
    random_seed
        Seed for the subsample draw (reproducible).

    Returns
    -------
    KalmanPanelInputs
        A panel with every per-ISIN array, the frame and the coord index arrays
        sliced consistently. Coord uniques are RE-derived from the retained rows
        so a level that lost all its members does not linger as an empty group.
    """
    n = len(panel.isins)
    if max_isins <= 0 or n <= max_isins:
        return panel
    rng = np.random.default_rng(random_seed)
    keep = np.sort(rng.choice(n, size=max_isins, replace=False))
    frame = panel.frame.iloc[keep].reset_index(drop=True)
    coord_uniques: dict[str, np.ndarray] = {}
    coord_idx: dict[str, np.ndarray] = {}
    for col, idx in panel.coord_idx.items():
        labels = np.asarray(panel.coord_uniques[col])[np.asarray(idx)[keep]]
        uniques, reindexed = np.unique(labels, return_inverse=True)
        coord_uniques[col] = uniques
        coord_idx[col] = reindexed.astype('int64')
    return replace(
        panel, frame=frame, isins=panel.isins[keep], Y=panel.Y[keep],
        t_scaled=panel.t_scaled[keep], X_drift=panel.X_drift[keep],
        n_analysts=panel.n_analysts[keep],
        sqrt_n_analysts=panel.sqrt_n_analysts[keep],
        vol_drift=panel.vol_drift[keep], dispersion_cv=panel.dispersion_cv[keep],
        avg_beta=panel.avg_beta[keep], size_ratio=panel.size_ratio[keep],
        volume_ratio=panel.volume_ratio[keep],
        coord_uniques=coord_uniques, coord_idx=coord_idx,
    )


def run_model_comparison(panel: KalmanPanelInputs, *,
                         config: Optional[KalmanRunConfig] = None,
                         robust: bool = True,
                         volume_penalty: float = 0.25) -> Optional[pd.DataFrame]:
    """Compare the local-level state model against its static twin on ELPD.

    Closes the one ``❌`` in this module's Bayesian-workflow coverage row. The two
    arms differ in exactly one thing — whether the per-ISIN latent may evolve
    over the panel:

    * ``local_level`` — ``state_innovation_scale`` from ``config`` (default 0.1);
    * ``static`` — ``state_innovation_scale=0.0``, pinning the state at its t=0
      anchor, i.e. the pre-0.9.9.14 time-constant build.

    Both are refit on the same (subsampled) panel so the ELPD contrast is
    like-for-like, then compared with :func:`arviz.compare`.

    .. note::

       ``log_likelihood`` is off project-wide and ``sample_posterior`` hard-codes
       it to ``False``, so each arm's group is attached post-hoc with
       :func:`~probabilistic_ml_model.pymc_models._workflow.attach_log_likelihood`
       (``pm.compute_log_likelihood``). The ``idata_kwargs={'log_likelihood':
       True}`` route does NOT work here: nutpie — the default sampler — ignores
       ``idata_kwargs`` and ``build_sample_kwargs`` strips it.

    Parameters
    ----------
    panel
        The fused panel. Subsampled to ``config.comparison_max_isins`` names.
    config
        Run config supplying the NUTS budget, the innovation scale and the
        subsample size. Defaults to :func:`get_run_config`.
    robust, volume_penalty
        Passed through to the builder so both arms match the production model.

    Returns
    -------
    pandas.DataFrame or None
        The ``az.compare`` table (also displayed / exported), or ``None`` when
        the comparison could not be completed — a missing ``log_likelihood``
        group is reported, never silently treated as a tie.
    """
    cfg = config if config is not None else get_run_config()
    if panel.Y.shape[1] < 2:
        print('Model comparison skipped: the state layer needs a T > 1 panel '
              '(got T=1). Use panel_lookbacks=("6m","3m","1m").')
        return None

    sub = _subsample_panel(panel, cfg.comparison_max_isins,
                           random_seed=cfg.random_seed)
    n_full, n_sub = len(panel.isins), len(sub.isins)
    # Never let a truncated comparison read as a full one.
    print(f'Model comparison on {n_sub} of {n_full} ISINs '
          f'({n_sub / max(n_full, 1):.0%}; cap={cfg.comparison_max_isins}), '
          f'T={sub.Y.shape[1]}, D={sub.Y.shape[2]}, '
          f'{cfg.chains} chains x {cfg.draws} draws.')

    arms = {
        'local_level': float(cfg.state_innovation_scale),
        'static': 0.0,
    }
    fits: dict[str, Any] = {}
    for name, scale in arms.items():
        model = build_fused_kalman_pt_model(
            sub, robust=robust, volume_penalty=volume_penalty,
            state_innovation_scale=scale)
        # Two full fits back to back: the bars are noise, and nutpie's
        # thin-space glyphs crash a cp1252 console.
        idata = sample_with_fallback(model, cfg, model_name=f'kalman_pt[{name}]',
                                     progressbar=False)
        if idata is None:
            print(f'Model comparison aborted: every sampler failed on the '
                  f'{name!r} arm.')
            return None
        attach_log_likelihood(idata, model)
        if not hasattr(idata, 'log_likelihood'):
            print(f'Model comparison aborted: could not attach a log_likelihood '
                  f'group to the {name!r} arm (az.compare needs one).')
            return None
        fits[name] = idata
        print(f'  [{name}] divergences='
              f'{int(idata.sample_stats["diverging"].sum())}')

    try:
        cmp_df = azs.compare(fits)
    except Exception as exc:
        print(f'compare failed: {exc!r}')
        return None

    display(cmp_df, label='table')
    # ArviZ 1.x exposes the value as ``.elpd``; ``.elpd_loo`` was REMOVED and a
    # getattr fallback on the old name silently yields nan, even though the
    # ELPDData repr still prints the "elpd_loo" row label.
    for name, idata in fits.items():
        with contextlib.suppress(Exception):
            loo = azs.loo(idata)
            print(f'  {name:12s} elpd={float(loo.elpd):10.2f}  '
                  f'se={float(loo.se):7.2f}')
    best = str(cmp_df.index[0])
    print(f'Best by ELPD: {best!r}'
          + ('  <-- the local-level state earns its parameters'
             if best == 'local_level' else
             '  <-- the static twin wins; the state layer is not paying for '
             'itself on this panel'))
    # Read the k-hat column, don't ignore it. The local-level arm carries a
    # per-ISIN latent path, so a slice of the pointwise contributions is
    # influential by construction and PSIS-LOO flags them (k-hat > 0.7); this is
    # the documented weakness of LOO for models with per-observation latents, not
    # evidence of a bad fit. Trust the verdict when |elpd_diff| is several times
    # ``dse`` (empirically ~80 vs dse ~10 on the smoke panel); treat a margin
    # inside ~2 dse as inconclusive rather than a win.
    print('Note: high Pareto k-hat on the state arm is expected (per-ISIN latent '
          'path). Judge the result on elpd_diff vs dse, not on k-hat alone.')
    return cmp_df


# =============================================================================
# 10. Expected price targets — posterior summary
# =============================================================================
def summarize_panel_screen(idata, panel: KalmanPanelInputs,
                           *, horizon: int = 4, rho: float = 0.85) -> ScreenContext:
    """Build the per-ISIN screening table from the fused-panel posterior.

    The screen has two complementary readouts:

    * **De-standardised upside** — the decision latent (``state_now``, the
      local-level state's filtered level at the snapshot; see
      :func:`resolve_screen_latent`) mapped back onto the primary
      ``feat_log_uplift`` series (:func:`panel_posterior_upside`) gives
      ``expected_upside`` / ``expected_pt`` and their 94% HDI bands.
    * **Structural-TS Monte-Carlo** — the same per-ISIN latent plus ``sigma_isin``
      / ``nu`` draws feed :func:`simulate_lagged_risk_adjusted_returns` /
      :func:`summarize_mc_returns` for the canonical risk-adjusted forward-return
      distribution (``er_mean``, percentiles, ``prob_pos``).

    Returns a :class:`ScreenContext` carrying the upside draws and the sorted
    ``results`` frame consumed by §13/§14.
    """
    post = idata.posterior
    frame = panel.frame

    # De-standardised expected upside / price-target draws (chain, draw, isin).
    eu, ept = panel_posterior_upside(idata, panel)
    exp_up = eu.mean(('chain', 'draw')).values
    exp_pt = ept.mean(('chain', 'draw')).values
    _ept_s = ept.stack(s=('chain', 'draw'))
    pt_lo = _ept_s.quantile(_HDI_LO, dim='s').values
    pt_hi = _ept_s.quantile(_HDI_HI, dim='s').values
    prob_pos = (eu > 0).mean(('chain', 'draw')).values
    # The ``risk_adj_return`` column keeps its name/units for the analytics export
    # and the dashboard, but now reports the FILTERED level (``state_now``) rather
    # than the t=0 anchor — the two coincide when T == 1.
    latent_da = resolve_screen_latent(post)
    risk_adj = latent_da.mean(('chain', 'draw')).values

    # Structural-TS Monte-Carlo over the per-ISIN risk-adjusted-return latent.
    #
    # The posterior ``risk_adj_return`` / ``sigma_isin`` live on the *standardised*
    # response scale (mean 0, sd ~1), so simulating them directly yielded ``er_*``
    # columns that were standardised z-scores — not returns — and ``prob_pos`` was
    # P(standardised baseline > 0), i.e. P(name beats the cross-sectional average
    # upside), which read as a near-zero "probability of a positive return" for
    # below-average names. De-standardise the latent onto the primary scale first
    # (the same (mean, std) used by :func:`panel_posterior_upside`) so the MC runs
    # on genuine units; on the log-uplift scale ``expm1`` then maps the simulated
    # log-returns to decimal returns and ``prob_pos`` becomes P(return > 0).
    _stats = _panel_response_stats(panel)
    _key = ('feat_log_uplift' if 'feat_log_uplift' in _stats
            else 'feat_implied_upside' if 'feat_implied_upside' in _stats
    else panel.response_names[0])
    _mean, _std = _stats[_key]
    mu_draws = (latent_da.stack(sample=('chain', 'draw'))
                .transpose('isin', 'sample').values) * _std + _mean
    sigma_draws = (post['sigma_isin'].stack(sample=('chain', 'draw'))
                   .transpose('isin', 'sample').values) * _std
    nu_draws = post['nu'].stack(sample=('chain', 'draw')).values
    mc = simulate_lagged_risk_adjusted_returns(
        mu_draws, sigma_draws, nu_draws, horizon=horizon, rho=rho,
        random_seed=RANDOM_SEED,
    )
    if _key == 'feat_log_uplift':
        # Simulated quantities are log-uplifts -> decimal returns (sign-preserving,
        # so ``prob_pos`` is unchanged and reads as P(return > 0)).
        mc = np.expm1(mc)
    mc_summary = summarize_mc_returns(mc, np.asarray(panel.isins))

    results = pd.DataFrame({
        'isin': np.asarray(panel.isins),
        'ticker': frame.get('ticker'),
        'name': frame.get('name'),
        'trading_region': frame.get('trading_region'),
        'region': frame.get('region'),
        'country': frame.get('country'),
        'unit': frame.get('unit'),
        'exchange': frame.get('exchange'),
        'sector': frame.get('sector'),
        'industry': frame.get('industry'),
        'size_class': frame.get('size_class'),
        'style_class': frame.get('style_class'),
        'market_cap': frame.get('market_cap'),
        # (100 - market_cap_country_r) / 100 — smaller = larger cap in country;
        # feeds the compute_cvar_aware_book mcap pre-selection gate.
        'mcap_country_r': frame.get('feat_mcap_country_r'),
        'enterprise_value': frame.get('enterprise_value'),
        'last_price': frame['last_price'].to_numpy(),
        'observed_pt': frame['observed_pt'].to_numpy(),
        'expected_pt': exp_pt,
        'expected_pt_hdi_lo': pt_lo,
        'expected_pt_hdi_hi': pt_hi,
        # All return/upside columns are stored as raw decimals (0.25 = +25%);
        # percent scaling happens only at display boundaries.
        'expected_upside': exp_up,
        'risk_adj_return': risk_adj,
        'prob_pos': prob_pos,
        'implied_upside': (
            frame['feat_implied_upside'].to_numpy()
            if 'feat_implied_upside' in frame.columns
            else (frame['observed_pt'] / frame['last_price'] - 1.0).to_numpy()
        ),
        'total_return_ytd': (
            frame['feat_total_return_ytd'].to_numpy()
            if 'feat_total_return_ytd' in frame.columns else np.nan
        ),
        'total_return_5y': (
            frame['feat_total_return_5y'].to_numpy()
            if 'feat_total_return_5y' in frame.columns else np.nan
        ),
        'total_return_10y': (
            frame['feat_total_return_10y'].to_numpy()
            if 'feat_total_return_10y' in frame.columns else np.nan
        ),
        'tr_cagr_3y': (
            frame['feat_tr_cagr_3y'].to_numpy()
            if 'feat_tr_cagr_3y' in frame.columns else np.nan
        ),
        'n_analysts': frame['n_analysts'].to_numpy(),
    })
    # Merge the MC risk-adjusted-return summary (er_mean / percentiles / prob_pos_mc).
    results = results.merge(
        mc_summary.rename(columns={'prob_pos': 'mc_prob_pos'}), on='isin', how='left',
    )
    results = results.sort_values('expected_upside', ascending=False).reset_index(drop=True)
    print(f'Fused-panel screen for {len(results)} ISINs '
          f'(MC horizon={horizon}, rho={rho}).')
    display(results.head(50))

    model_df = frame

    # (The former price-space shrinkage scatter was removed as redundant with
    # the percent-space shrinkage view in ``_plot_comparative_returns`` and the
    # signed-log ``create_kalman_vs_raw_scatter`` reused in §10c — 0.9.9.11
    # consolidation.)

    # Per-industry expected_upside posterior — arviz_plots forest with HDIs.
    eu_pct = eu * 100.0
    _industry_per_isin = model_df['industry'].fillna('Unknown').astype(str).to_numpy()
    _industry_da = xr.DataArray(
        _industry_per_isin, dims='isin', coords={'isin': eu.coords['isin']},
    )
    expected_upside_by_industry = eu_pct.groupby(_industry_da.rename('industry')).mean('isin')
    _ds_forest = xr.Dataset({'expected_upside_pct': expected_upside_by_industry})
    pc = azp.plot_forest(
        _ds_forest, var_names=['expected_upside_pct'], combined=True,
        figure_kwargs=_azp_figure_kwargs(
            _forest_height_px(expected_upside_by_industry.sizes.get('industry', 1))),
    )
    pc.add_title('Per-industry expected upside (%) — posterior mean and 94% HDI')
    with contextlib.suppress(Exception):
        _fx = _plotly_figure_of(pc)
        if _fx is not None:
            _add_ref_line(_fx, x=0, kind='zero')
            _fx.update_xaxes(title_text='expected upside (%)', ticksuffix='%')
    _safe_show(pc)

    # §5b model internals (Model A risk discount + Model B GRW components).
    plot_fused_model_effects(idata, panel)

    _plot_comparative_returns(eu, results, model_df)
    return ScreenContext(eu=eu, ept=ept, results=results, mc_summary=mc_summary)


def plot_fused_model_effects(idata, panel: KalmanPanelInputs) -> None:
    """Visualise the fused MvGRW internals — Model-A risk discount + Model-B GRW.

    Four panels make the §5b structure legible:

    (a) ``expected_return`` → ``risk_adj_return`` coloured by the per-ISIN average
        market beta — the ``- risk_loading * z(avg_beta)`` systematic-risk
        discount that is the Kalman-specific refinement.
    (b) ``achieve_prob`` (= ``sigmoid(state_now)``) against the risk-adjusted return.
    (c) the heteroscedastic ``sigma_isin`` against analyst count, coloured by the
        consensus dispersion CV (``sigma_isin = sigma_base * (1 + cv) / sqrt(n)``).
    (d) the local-level ``state_path`` (median + 10–90 % cross-sectional band) over
        time — falling back to the ``beta_t`` slope on an isin-varying time axis,
        where the slope is identified and the state is not materialised.
    """
    post = idata.posterior
    er = post['expected_return'].mean(('chain', 'draw')).values
    # Panel (a) genuinely wants the t=0 ANCHOR — it visualises the
    # expected_return -> risk_adj_return tilt. Panel (b) wants the decision latent,
    # since achieve_prob is now sigmoid(state_now).
    rar = post['risk_adj_return'].mean(('chain', 'draw')).values
    latent = resolve_screen_latent(post).mean(('chain', 'draw')).values
    ap = post['achieve_prob'].mean(('chain', 'draw')).values
    si = post['sigma_isin'].mean(('chain', 'draw')).values
    avg_beta = np.asarray(panel.avg_beta, dtype='float64')
    n_an = np.asarray(panel.n_analysts, dtype='float64')
    cv = np.asarray(panel.dispersion_cv, dtype='float64')

    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.12, vertical_spacing=0.12,
        subplot_titles=('Model A — systematic-risk (beta) discount',
                        'Model A — achievement probability',
                        'Model A — heteroscedastic scale  σ·(1+cv)/√n',
                        'Model B — MvGRW slope per y_series'))

    # (a) Systematic-risk (beta) discount. Per-trace marker colourbar (Viridis),
    # positioned over its own subplot (Plotly has no shared per-axes colourbar).
    fig.add_trace(go.Scatter(
        x=er, y=rar, mode='markers',
        marker=dict(color=avg_beta, colorscale=CS_SEQ, size=6, opacity=0.75,
                    colorbar=dict(title='avg β', len=0.42, y=0.79, x=0.455, thickness=12)),
        customdata=np.c_[avg_beta],
        hovertemplate=('ER = %{x:.2f}  RAR = %{y:.2f}'
                       '<br>avg β = %{customdata[0]:.2f}<extra></extra>'),
        name='beta discount', showlegend=False), row=1, col=1)
    _lim = [float(np.nanmin([er.min(), rar.min()])), float(np.nanmax([er.max(), rar.max()]))]
    fig.add_trace(go.Scatter(x=_lim, y=_lim, mode='lines',
                             line=dict(color=C_REF, dash='dash', width=1.1),
                             showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.update_xaxes(title_text='expected_return (latent)', row=1, col=1)
    fig.update_yaxes(title_text='risk_adj_return = ER − λ·z(β) − γ·z(size)', row=1, col=1)

    # (b) Achievement probability vs risk-adjusted return.
    fig.add_trace(go.Scatter(x=latent, y=ap, mode='markers',
                             marker=dict(color=C_FORECAST, size=6, opacity=0.6),
                             name='achieve_prob',
                             hovertemplate=('latent = %{x:.2f}<br>'
                                            'P(achieve) = %{y:.0%}<extra></extra>'),
                             showlegend=False), row=1, col=2)
    _add_ref_line(fig, y=0.5, kind='zero', row=1, col=2)
    fig.update_xaxes(title_text=f'{KALMAN_SCREEN_LATENT} (standardised latent)',
                     row=1, col=2)
    fig.update_yaxes(title_text='achieve_prob (sigmoid filtered level)', range=[0, 1],
                     tickformat='.0%', row=1, col=2)

    # (c) Heteroscedastic measurement scale (log analyst axis).
    fig.add_trace(go.Scatter(
        x=n_an, y=si, mode='markers',
        marker=dict(color=np.clip(cv, 0, np.nanpercentile(cv, 99)), colorscale=CS_SEQ,
                    size=6, opacity=0.75,
                    colorbar=dict(title='dispersion CV', len=0.42, y=0.21, x=0.455,
                                  thickness=12)),
        customdata=np.c_[cv],
        hovertemplate=('n = %{x:.0f}<br>σ_isin = %{y:.3f}'
                       '<br>cv = %{customdata[0]:.3f}<extra></extra>'),
        name='sigma_isin', showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text='n_analysts (log)', type='log', row=2, col=1)
    fig.update_yaxes(title_text='sigma_isin', row=2, col=1)

    # (d) The local-level state path — the cross-sectional spread of the per-ISIN
    # latent at each time step. This replaces the former ``beta_t`` slope panel,
    # which on the default (isin-constant) time axis drew a flat line at zero: the
    # slope is not identified there and is no longer materialised at all. The
    # state path is the quantity that genuinely carries per-name time structure,
    # so a visibly widening band from t=0 is the local-level layer doing its job;
    # a flat band means ``sigma_state`` collapsed and the panel has no dynamics.
    if 'state_path' in post.data_vars:
        sp = post['state_path'].mean(('chain', 'draw'))  # (isin, time)
        times = np.asarray(sp['time'].values)
        vals = np.asarray(sp.transpose('time', 'isin').values, dtype='float64')
        med = np.nanmedian(vals, axis=1)
        lo = np.nanpercentile(vals, 10, axis=1)
        hi = np.nanpercentile(vals, 90, axis=1)
        fig.add_trace(go.Scatter(
            x=np.r_[times, times[::-1]], y=np.r_[hi, lo[::-1]], fill='toself',
            fillcolor=_hex_to_rgba(C_POSTERIOR, 0.20), line=dict(width=0),
            hoverinfo='skip', showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=times, y=med, mode='lines+markers', marker=dict(size=5),
            line=dict(color=C_POSTERIOR, width=1.6), name='state_path (median)',
            hovertemplate=('t = %{x}<br>median state = %{y:.3f}<extra></extra>'),
            legendgroup='state'), row=2, col=2)
        _add_ref_line(fig, y=0, kind='zero', row=2, col=2)
        fig.update_xaxes(title_text='time index (lookback anchor, oldest → now)',
                         row=2, col=2)
        fig.update_yaxes(title_text='state_path (median, 10–90% band)', row=2, col=2)
    elif 'beta_t' in post.data_vars:
        # Isin-VARYING time axis (the T=1 days-to-event covariate): the per-series
        # slope is identified and materialised, so show it.
        beta_t = post['beta_t'].mean(('chain', 'draw'))  # (time, y_series)
        times = np.asarray(beta_t['time'].values)
        for name in panel.response_names:
            try:
                fig.add_trace(go.Scatter(
                    x=times, y=beta_t.sel(y_series=name).values, mode='lines+markers',
                    marker=dict(size=4), line=dict(width=1.4), name=str(name),
                    legendgroup='beta_t'), row=2, col=2)
            except Exception:
                continue
        _add_ref_line(fig, y=0, kind='zero', row=2, col=2)
        fig.update_xaxes(title_text='time index (fiscal anchor)', row=2, col=2)
        fig.update_yaxes(title_text='beta_t (GRW slope)', row=2, col=2)

    fig.update_layout(showlegend=True,
                      legend=dict(font_size=_LEGEND_FONT_SIZE, title_text='y_series'))
    _render_plotly(fig, height=H_PANEL)


def _plot_comparative_returns(eu, results: pd.DataFrame, model_df: pd.DataFrame) -> None:
    """Section 10b: feat_implied_upside vs expected_upside vs total_return_ytd views.

    ``eu`` is the de-standardised ``expected_upside`` posterior (chain, draw, isin).
    """
    _comp = results.dropna(subset=['expected_upside']).copy()
    # Display boundary: the screen frame stores decimals — build local percent
    # columns for the plots only.
    for _c in ('expected_upside', 'implied_upside', 'total_return_ytd'):
        _comp[f'{_c}_pct'] = pd.to_numeric(_comp.get(_c), errors='coerce') * 100.0

    # (1) Shrinkage scatter: raw analyst-implied upside vs Kalman-smoothed expected upside.
    _both = np.r_[_comp['implied_upside_pct'].to_numpy(), _comp['expected_upside_pct'].to_numpy()]
    _both = _both[np.isfinite(_both)]
    _lo, _hi = float(np.nanpercentile(_both, 1)), float(np.nanpercentile(_both, 99))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_comp['implied_upside_pct'], y=_comp['expected_upside_pct'], mode='markers',
        marker=dict(size=6, opacity=0.45, color=C_POSTERIOR),
        text=_comp.get('ticker'), name='ISIN',
        hovertemplate=('%{text}<br>implied = %{x:.1f}%<br>'
                       'expected = %{y:.1f}%<extra></extra>')))
    fig.add_trace(go.Scatter(x=[_lo, _hi], y=[_lo, _hi], mode='lines',
                             line=dict(color=C_REF, dash='dash', width=1.1),
                             name='y = x (no shrinkage)', hoverinfo='skip'))
    _add_ref_line(fig, y=0, kind='zero')
    _add_ref_line(fig, x=0, kind='zero')
    fig.update_xaxes(range=[_lo, _hi], ticksuffix='%',
                     title_text='raw implied upside  feat_implied_upside (%)')
    fig.update_yaxes(range=[_lo, _hi], ticksuffix='%',
                     title_text='Kalman-smoothed expected upside (%)',
                     scaleanchor='x', scaleratio=1)
    fig.update_layout(title='Posterior shrinkage of analyst-implied upside',
                      height=H_TALL)
    _render_plotly(fig)

    # (2) Overlaid KDEs. (The former cross-sectional-average posterior KDE was
    # removed as redundant: §14's cohort-vs-universe KDE is the canonical view
    # of that quantity — see 0.9.9.11 consolidation.)
    eu_pct = eu * 100.0
    fig = go.Figure()
    for _col, _lab, _c in [
        ('implied_upside_pct', 'raw implied upside (consensus)', C_OBSERVED),
        ('expected_upside_pct', 'Kalman-smoothed expected upside', C_POSTERIOR),
        ('total_return_ytd_pct', 'realised total return YTD', C_FORECAST),
    ]:
        _v = pd.to_numeric(_comp.get(_col), errors='coerce').to_numpy()
        _v = _v[np.isfinite(_v)]
        if _v.size > 5:
            _v = _v[(_v >= np.nanpercentile(_v, 1)) & (_v <= np.nanpercentile(_v, 99))]
        _xs, _ys = _kde_xy(_v)
        if _xs is not None:
            fig.add_trace(go.Scatter(x=_xs, y=_ys, mode='lines', fill='tozeroy',
                                     line=dict(color=_c, width=1.8),
                                     fillcolor=_hex_to_rgba(_c, 0.18), name=_lab,
                                     hovertemplate=(f'{_lab}<br>'
                                                    'return = %{x:.1f}%<extra></extra>')))
    _add_ref_line(fig, x=0, kind='zero')
    fig.update_xaxes(title_text='return / upside (%)', ticksuffix='%')
    fig.update_yaxes(title_text='density')
    fig.update_layout(title='Expected vs implied vs realised returns - distributional comparison',
                      legend=dict(font_size=_LEGEND_FONT_SIZE))
    _render_plotly(fig, height=H_SHORT)

    # (3) Per-sector forest-style comparison.
    _sector_da = xr.DataArray(
        model_df['sector'].fillna('Unknown').astype(str).to_numpy(),
        dims='isin', coords={'isin': eu_pct.coords['isin']},
    )
    eu_by_sector = eu_pct.groupby(_sector_da.rename('sector')).mean('isin')
    _stack = eu_by_sector.stack(s=('chain', 'draw'))
    _sec = [str(s) for s in eu_by_sector['sector'].values]
    _mean = _stack.mean('s').values
    _q_lo = _stack.quantile(_HDI_LO, 's').values
    _q_hi = _stack.quantile(_HDI_HI, 's').values
    _ref = (_comp.assign(sector=_comp['sector'].fillna('Unknown').astype(str))
            .groupby('sector')[['implied_upside_pct', 'total_return_ytd_pct']].mean()
            .reindex(_sec))
    _order = np.argsort(_mean)
    _y = np.arange(len(_sec))
    _labels = np.array(_sec)[_order]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_mean[_order], y=_labels, mode='markers',
        marker=dict(color=C_POSTERIOR, size=9),
        error_x=dict(type='data', symmetric=False,
                     array=(_q_hi - _mean)[_order], arrayminus=(_mean - _q_lo)[_order],
                     color=C_POSTERIOR, thickness=1.4, width=4),
        customdata=np.c_[_q_lo[_order], _q_hi[_order]],
        hovertemplate=('%{y}<br>expected = %{x:.1f}%<br>94% HDI = '
                       '[%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>'),
        name='expected upside (posterior mean, 94% HDI)'))
    # Reference markers: mask sectors with no coverage so hovers never show null.
    for _col, _lab2, _c2, _sym, _sz in [
        ('implied_upside_pct', 'raw implied upside (mean)', C_OBSERVED, 'square', 9),
        ('total_return_ytd_pct', 'realised total return YTD (mean)', C_FORECAST, 'x', 11),
    ]:
        _vals = _ref[_col].to_numpy()[_order]
        _ok = np.isfinite(_vals)
        fig.add_trace(go.Scatter(
            x=_vals[_ok], y=_labels[_ok], mode='markers',
            marker=dict(color=_c2, size=_sz, symbol=_sym),
            hovertemplate=(f'%{{y}}<br>{_lab2} = %{{x:.1f}}%<extra></extra>'),
            name=_lab2))
    _add_ref_line(fig, x=0, kind='zero')
    fig.update_xaxes(title_text='return / upside (%)', ticksuffix='%')
    fig.update_layout(title='Per-sector: expected vs implied vs realised returns',
                      height=int(max(320, 34 * len(_sec) + 140)),
                      legend=dict(font_size=_LEGEND_FONT_SIZE,
                                  title_text='series'))
    _render_plotly(fig)


# =============================================================================
# 10b. CVaR-aware risk analytics + sizing (SSOT for §10c export and §14b screen)
# =============================================================================
def compute_cvar_aware_book(
        idata, panel: KalmanPanelInputs, screen: ScreenContext, results: pd.DataFrame,
        *, alpha: Optional[float] = None, cap: Optional[float] = None,
        k_book: Optional[int] = None, p_long: Optional[float] = None,
        mcap_r_max: Optional[float] = None,
        config: Optional[KalmanRunConfig] = None,
) -> RiskBook:
    """Resolve the sizing knobs from :class:`KalmanRunConfig` and build the book.

    Thin workflow-level wrapper around
    :func:`probabilistic_ml_model.pymc_models.RiskBookModel.compute_cvar_aware_book`,
    which holds the model logic. This layer exists so the workflow keeps its
    config-driven contract (every ``None`` falls back to the corresponding
    ``KalmanRunConfig`` field) while the package function stays free of any
    dependency on the script.

    Parameters
    ----------
    idata
        Fused-panel inference data.
    panel
        Unused; retained for call-site compatibility with the rest of §10b.
    screen
        The :class:`ScreenContext` carrying the ``expected_upside`` draws ``eu``.
    results
        Per-ISIN screen table (see the package function for required columns).
    alpha, cap, k_book, p_long, mcap_r_max
        Sizing overrides; ``None`` resolves to ``config.cvar_alpha`` /
        ``weight_cap`` / ``k_book`` / ``p_long`` / ``mcap_country_r_max``.
    config
        Optional :class:`KalmanRunConfig`; defaults to :func:`get_run_config`.

    Returns
    -------
    RiskBook
        Per-name ``analytics``, the sized ``book`` and the portfolio ``summary``.
    """
    cfg = config if config is not None else get_run_config()
    return _compute_cvar_aware_book(
        idata,
        screen.eu,
        results,
        alpha=cfg.cvar_alpha if alpha is None else alpha,
        cap=cfg.weight_cap if cap is None else cap,
        k_book=cfg.k_book if k_book is None else k_book,
        p_long=cfg.p_long if p_long is None else p_long,
        mcap_r_max=cfg.mcap_country_r_max if mcap_r_max is None else mcap_r_max,
    )


def plot_kalman_results_overview(kalman_results: pd.DataFrame,
                                 book_summary: Optional[dict] = None) -> None:
    """Decision dashboard over the exported ``kalman_results`` row-set (§10c).

    ``book_summary`` (optional :attr:`RiskBook.summary`) supplies the aggregate
    portfolio ``port_up`` / ``port_vol`` / ``port_cvar`` / ``starr_book``
    (decimal returns, scaled here at the display boundary); when given, panels
    (a)/(d) carry the portfolio point and (a) adds the held-name upper-hull
    efficient line.

    Four linked views of the ``analytics.kalman_filtered_price_targets`` columns
    (DDL: ``sql_scripts/analytics/kalman_filtered_price_targets.sql``) that read
    the expected-return / volatility trade-off directly off the export:

    (a) **Risk-return map** — Monte-Carlo expected return (``er_mean``) vs
        expected volatility (``expected_vol_kalman``), coloured by the STARR
        ratio (``reward_to_cvar``) and sized by the CVaR-aware book weight, so
        the held names visibly cluster in the high-reward / low-tail corner.
    (b) **MC return fan (sized book)** — per-name ``er_p05``–``er_p95`` interval
        with the ``er_p50`` median and the posterior ``expected_return_kalman``
        overlaid, for the largest book weights: the simulated return
        distribution behind each sized position.
    (c) **Kalman-gain conditioning** — ``mc_prob_pos`` vs ``p_upside_pos_cond``
        coloured by ``kalman_gain``: how much the smoother's state confidence
        discounts each name's raw positive-return probability (distance below
        the y = x line).
    (d) **Tail asymmetry** — expected return vs the 5% expected shortfall
        (``cvar_5pct_kalman``); held names should sit in the favourable
        upper-right (positive reward, shallow tail) region.
    """
    if not _HAS_PLOTLY or kalman_results is None or len(kalman_results) == 0:
        return

    df = kalman_results.copy()

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(df.get(col), errors='coerce')

    er_mean_pct = _num('er_mean') * 100.0
    er_p05_pct = _num('er_p05') * 100.0
    er_p50_pct = _num('er_p50') * 100.0
    er_p95_pct = _num('er_p95') * 100.0
    exp_ret_pct = _num('expected_return_kalman') * 100.0
    exp_vol_pct = _num('expected_vol_kalman') * 100.0
    cvar_pct = _num('cvar_5pct_kalman') * 100.0
    starr = _num('reward_to_cvar')  # dimensionless STARR ratio — never scaled
    weight = _num('cvar_book_weight').fillna(0.0)
    mc_pos = _num('mc_prob_pos')
    p_cond = _num('p_upside_pos_cond')
    gain = _num('kalman_gain')
    _tk = df.get('ticker', pd.Series(index=df.index, dtype=object))
    _nm = df.get('name', pd.Series(index=df.index, dtype=object))
    label = _tk.astype('string').fillna(_nm.astype('string')).fillna(
        df['isin'].astype('string'))
    held = weight > 0

    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.12, vertical_spacing=0.14,
        subplot_titles=(
            'Risk-return map  (colour = STARR, size = book weight)',
            'MC return fan — CVaR-sized book  (p05–p95, | = p50, ◆ = posterior)',
            'Kalman-gain conditioning of P(return > 0)',
            'Tail asymmetry — expected return vs 5% expected shortfall'))

    # (a) Risk-return map.
    _m = exp_vol_pct.notna() & er_mean_pct.notna()
    if _m.any():
        _starr_clip = starr.clip(lower=float(np.nanquantile(starr, 0.02)),
                                 upper=float(np.nanquantile(starr, 0.98)))
        fig.add_trace(go.Scatter(
            x=exp_vol_pct[_m], y=er_mean_pct[_m], mode='markers',
            marker=dict(
                color=_starr_clip[_m], colorscale=CS_SEQ, opacity=0.8,
                size=(6.0 + 60.0 * weight[_m]).clip(upper=22.0),
                colorbar=dict(title='STARR', len=0.42, y=0.79, x=0.455,
                              thickness=12)),
            text=label[_m],
            customdata=np.c_[weight[_m] * 100.0, cvar_pct[_m]],
            hovertemplate=('%{text}<br>E[r]=%{y:.1f}%  vol=%{x:.1f}%'
                           '<br>weight=%{customdata[0]:.1f}%  '
                           'CVaR5=%{customdata[1]:.1f}%<extra></extra>'),
            name='universe', showlegend=False), row=1, col=1)
        # Upper-hull efficient line over the HELD names: sort by vol and keep
        # the running-max return staircase (no held name above-left of it).
        _hm = _m & held
        if _hm.sum() >= 2:
            _hv = exp_vol_pct[_hm].to_numpy()
            _hr = er_mean_pct[_hm].to_numpy()
            _hs = np.argsort(_hv)
            _hull_v, _hull_r, _best = [], [], -np.inf
            for _v, _r in zip(_hv[_hs], _hr[_hs]):
                if _r > _best:
                    _hull_v.append(_v)
                    _hull_r.append(_r)
                    _best = _r
            fig.add_trace(go.Scatter(
                x=_hull_v, y=_hull_r, mode='lines',
                line=dict(color=C_HIGHLIGHT, width=1.6, dash='dot', shape='hv'),
                name='held-name efficient hull',
                hovertemplate='hull: E[r]=%{y:.1f}% @ vol=%{x:.1f}%<extra></extra>',
            ), row=1, col=1)
        if book_summary:
            _pv = float(book_summary.get('port_vol', float('nan'))) * 100.0
            _pu = float(book_summary.get('port_up', float('nan'))) * 100.0
            _st = float(book_summary.get('starr_book', float('nan')))
            if np.isfinite(_pv) and np.isfinite(_pu):
                fig.add_trace(go.Scatter(
                    x=[_pv], y=[_pu], mode='markers',
                    marker=dict(color=C_HIGHLIGHT, size=16, symbol='star',
                                line=dict(color=C_PANEL_BG, width=1)),
                    name='CVaR book (portfolio)',
                    hovertemplate=(f'portfolio<br>E[r]=%{{y:.1f}}%  '
                                   f'vol=%{{x:.1f}}%<br>STARR={_st:.2f}'
                                   '<extra></extra>')), row=1, col=1)
        _add_ref_line(fig, y=0, kind='zero', row=1, col=1)
        fig.update_xaxes(title_text='expected volatility  expected_vol_kalman (%)',
                         ticksuffix='%', row=1, col=1)
        fig.update_yaxes(title_text='MC expected return  er_mean (%)',
                         ticksuffix='%', row=1, col=1)

    # (b) MC return fan for the largest sized positions. Mask names with a
    # missing MC summary so the fan never renders gapped bars with null hovers.
    _fan_ok = held & er_p05_pct.notna() & er_p50_pct.notna() & er_p95_pct.notna()
    _book = df.index[_fan_ok]
    if len(_book):
        _ord = weight.loc[_book].sort_values(ascending=True).tail(20).index
        _y = label.loc[_ord].to_numpy(dtype=object)
        fig.add_trace(go.Scatter(
            x=er_p50_pct.loc[_ord], y=_y, mode='markers',
            marker=dict(color=C_POSTERIOR, size=9, symbol='line-ns-open',
                        line=dict(width=2)),
            error_x=dict(type='data', symmetric=False,
                         array=(er_p95_pct - er_p50_pct).loc[_ord],
                         arrayminus=(er_p50_pct - er_p05_pct).loc[_ord],
                         color=_hex_to_rgba(C_POSTERIOR, 0.55), thickness=2.4),
            customdata=np.c_[er_p05_pct.loc[_ord], er_p95_pct.loc[_ord]],
            hovertemplate=('%{y}<br>p50 = %{x:.1f}%<br>p05/p95 = '
                           '[%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]'
                           '<extra></extra>'),
            name='MC p05–p95 (| = p50)', legendgroup='fan'), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=exp_ret_pct.loc[_ord], y=_y, mode='markers',
            marker=dict(color=C_OBSERVED, size=8, symbol='diamond'),
            hovertemplate='%{y}<br>posterior E[r] = %{x:.1f}%<extra></extra>',
            name='posterior expected return', legendgroup='fan'), row=1, col=2)
        _add_ref_line(fig, x=0, kind='zero', row=1, col=2)
        fig.update_xaxes(title_text='simulated return distribution (%)',
                         ticksuffix='%', row=1, col=2)

    # (c) Conditioning: raw MC P(>0) vs the gain-conditioned probability.
    _c = mc_pos.notna() & p_cond.notna()
    if _c.any():
        fig.add_trace(go.Scatter(
            x=mc_pos[_c], y=p_cond[_c], mode='markers',
            marker=dict(color=gain[_c], colorscale=CS_SEQ, size=6, opacity=0.75,
                        colorbar=dict(title='kalman gain', len=0.42, y=0.21,
                                      x=0.455, thickness=12)),
            text=label[_c],
            hovertemplate=('%{text}<br>MC P(>0)=%{x:.0%}  '
                           'conditional=%{y:.0%}<extra></extra>'),
            name='conditioning', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 line=dict(color=C_REF, dash='dash', width=1.1),
                                 name='y = x', showlegend=False,
                                 hoverinfo='skip'), row=2, col=1)
        fig.update_xaxes(title_text='mc_prob_pos (unconditional)', range=[0, 1],
                         tickformat='.0%', row=2, col=1)
        fig.update_yaxes(title_text='p_upside_pos_cond (× kalman_gain)',
                         range=[0, 1], tickformat='.0%', row=2, col=1)

    # (d) Tail asymmetry: reward vs expected shortfall, held names highlighted.
    _t = cvar_pct.notna() & exp_ret_pct.notna()
    if _t.any():
        for _mask, _name, _color, _size in (
                (_t & ~held, 'not held', C_MUTED, 5),
                (_t & held, 'CVaR book', C_HIGHLIGHT, 8)):
            if not _mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=cvar_pct[_mask], y=exp_ret_pct[_mask], mode='markers',
                marker=dict(color=_color, size=_size, opacity=0.7),
                text=label[_mask],
                hovertemplate=('%{text}<br>CVaR5=%{x:.1f}%  '
                               'E[r]=%{y:.1f}%<extra></extra>'),
                name=_name, legendgroup='tail'), row=2, col=2)
        if book_summary:
            _pc = float(book_summary.get('port_cvar', float('nan'))) * 100.0
            _pu = float(book_summary.get('port_up', float('nan'))) * 100.0
            if np.isfinite(_pc) and np.isfinite(_pu):
                fig.add_trace(go.Scatter(
                    x=[_pc], y=[_pu], mode='markers',
                    marker=dict(color=C_HIGHLIGHT, size=16, symbol='star',
                                line=dict(color=C_PANEL_BG, width=1)),
                    name='CVaR book (portfolio)', showlegend=False,
                    hovertemplate=('portfolio<br>CVaR5=%{x:.1f}%  '
                                   'E[r]=%{y:.1f}%<extra></extra>')), row=2, col=2)
        _add_ref_line(fig, y=0, kind='zero', row=2, col=2)
        _add_ref_line(fig, x=0, kind='zero', row=2, col=2)
        fig.update_xaxes(title_text='cvar_5pct_kalman — 5% expected shortfall (%)',
                         ticksuffix='%', row=2, col=2)
        fig.update_yaxes(title_text='expected_return_kalman (%)',
                         ticksuffix='%', row=2, col=2)

    fig.update_layout(
        title='kalman_filtered_price_targets — expected return / volatility / tail overview',
        legend=dict(font_size=_LEGEND_FONT_SIZE))
    _render_plotly(fig, height=H_GRID)


# =============================================================================
# 10c. Export — analytics.kalman_filtered_price_targets
# =============================================================================
_ANALYTICS_TABLE = 'kalman_filtered_price_targets'
_ANALYTICS_DDL_PATH = (Path(__file__).resolve().parent / 'sql_scripts' /
                       'analytics' / f'{_ANALYTICS_TABLE}.sql')

# Unit-convention header + per-column comments persisted with the regenerated
# DDL. ``if_exists='replace'`` drops and recreates the table on every export, so
# the comments only survive if the checked-in DDL carries them (CHANGELOG 0.9.9.7
# / CLAUDE.md document this contract; the file had drifted without them).
_ANALYTICS_DDL_HEADER = """\
-- analytics.kalman_filtered_price_targets
--
-- Generated by ``export_analytics`` in pymc_kalman_filter_pt.py -- do not edit by
-- hand; ``export_to_analytics_db(..., if_exists='replace')`` drops and recreates
-- the table on every run and this file is rewritten alongside it.
--
-- UNIT CONVENTION (since 0.9.9.7): every return / risk column stores a **raw
-- decimal** return (0.25 = +25%), including ``cvar_5pct_kalman`` and
-- ``expected_vol_kalman``. Percent scaling happens only at visualization and
-- print boundaries. Probabilities are decimals in [0, 1]."""

_ANALYTICS_COLUMN_COMMENTS: dict[str, str] = {
    'price_target_kalman': 'Posterior-mean Kalman-smoothed price target (price units).',
    'implied_upside': 'Raw analyst-consensus implied upside vs last price (decimal return).',
    'expected_return_kalman': 'Posterior-mean expected return (decimal return, 0.25 = +25%).',
    'kalman_variance': 'Posterior variance of the smoothed price target (price units squared).',
    'kalman_gain': 'achieve_prob = sigmoid(risk_adj_return): probability the target is '
                   'achieved (decimal in [0, 1]).',
    'signal_strength': '|E[risk_adj_return]| / sd(risk_adj_return) (dimensionless).',
    'expected_pt_hdi_lo': 'Lower bound of the 94% posterior price-target HDI (price units).',
    'expected_pt_hdi_hi': 'Upper bound of the 94% posterior price-target HDI (price units).',
    'risk_adj_return': 'Posterior-mean risk-adjusted-return latent (decimal return).',
    'er_mean': 'Structural-TS Monte-Carlo mean forward return (decimal return).',
    'er_sd': 'Pooled std of the structural-TS Monte-Carlo forward-return draws '
             '(decimal return); denominator of expected_sharpe_ratio.',
    'er_p05': '5th percentile of the Monte-Carlo forward-return draws (decimal return).',
    'er_p50': 'Median of the Monte-Carlo forward-return draws (decimal return).',
    'er_p95': '95th percentile of the Monte-Carlo forward-return draws (decimal return).',
    'mc_prob_pos': 'Monte-Carlo probability of a positive forward return (decimal in [0, 1]).',
    'cvar_book_weight': 'Normalised CVaR-aware long-book weight; held names sum to 1 '
                        '(0 for names outside the sized book).',
    'cvar_5pct_kalman': '5% expected shortfall (CVaR) of the posterior upside draws '
                        '(decimal return, negative = loss).',
    'reward_to_cvar': 'STARR ratio: expected upside / binding tail risk (dimensionless).',
    'expected_vol_kalman': 'Std of the posterior upside draws (decimal return).',
    'expected_sharpe_ratio': 'er_mean / er_sd of the structural-TS Monte-Carlo forward-return '
                             'distribution (dimensionless).',
    'p_upside_pos_cond': 'mc_prob_pos x kalman_gain: probability of a positive upside given '
                         'state confidence (decimal in [0, 1]); the p_long gate applies here.',
    'pt_achievement_1y': 'Historic 1y price-target achievement rate (decimal in [0, 1]).',
    'pt_range_hit_rate': 'Share of history inside the analyst target range (decimal in [0, 1]).',
    'analyst_bullish_pct': 'Share of analysts rating the name a buy (decimal in [0, 1]).',
    'analyst_bearish_pct': 'Share of analysts rating the name a sell (decimal in [0, 1]).',
    'analyst_neutral_pct': 'Share of analysts rating the name a hold (decimal in [0, 1]).',
    'analyst_conviction': 'Net buy-minus-sell analyst conviction (decimal in [-1, 1]).',
}


def write_analytics_ddl(frame: pd.DataFrame,
                        path: Optional[Path] = None) -> Optional[Path]:
    """Rewrite the checked-in ``kalman_filtered_price_targets`` DDL from ``frame``.

    Keeps ``sql_scripts/analytics/kalman_filtered_price_targets.sql`` in step with
    the frame the export actually writes, and re-attaches the unit-convention
    header plus the ``COMMENT ON COLUMN`` statements that the drop-and-recreate
    write destroys each run.

    Parameters
    ----------
    frame
        The exported analytics frame.
    path
        Destination; defaults to :data:`_ANALYTICS_DDL_PATH`.

    Returns
    -------
    Optional[Path]
        The written path, or ``None`` when the write failed (never raises).
    """
    target = Path(path) if path is not None else _ANALYTICS_DDL_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _sql_table_ddl(frame, _ANALYTICS_TABLE, quote_table=False,
                           comments=_ANALYTICS_COLUMN_COMMENTS,
                           header=_ANALYTICS_DDL_HEADER),
            encoding='utf-8')
        logger.info("Regenerated analytics DDL -> %s", target)
        return target
    except Exception as exc:
        logger.warning("Analytics DDL regeneration skipped: %r", exc)
        return None


def export_analytics(idata, panel: KalmanPanelInputs, screen: ScreenContext,
                     *, risk_book: Optional[RiskBook] = None,
                     write: bool = True) -> pd.DataFrame:
    """Build the ``analytics.kalman_filtered_price_targets`` row-set from the fused posterior.

    Maps the fused MvGRW + volatility-conditioned posterior onto the analytics table's
    Kalman columns:

    * ``price_target_kalman`` / ``kalman_estimate`` — de-standardised ``expected_pt``.
    * ``implied_return_kalman`` / ``expected_upside_kalman`` — de-standardised
      ``expected_upside`` (decimal).
    * ``kalman_gain`` — ``achieve_prob`` = ``sigmoid(risk_adj_return)`` (the smoother's
      confidence analogue: probability the implied target is achieved).
    * ``signal_strength`` — ``|E[risk_adj_return]| / sd(risk_adj_return)``.

    It enriches each row with the valuation, posterior-band and structural-TS Monte-Carlo
    columns pulled straight from the §10 screen table (``screen.results``), so the export
    reuses — and never drifts from — those figures rather than recomputing them:

    * ``market_cap`` / ``enterprise_value`` — size context from the panel frame.
    * ``expected_pt_hdi_lo`` / ``expected_pt_hdi_hi`` — 94% posterior price-target band.
    * ``risk_adj_return`` — posterior-mean risk-adjusted-return latent.
    * ``er_mean`` / ``er_sd`` / ``er_p05`` / ``er_p50`` / ``er_p95`` — MC return
      distribution summary (decimal returns).
    * ``mc_prob_pos`` — MC probability of a positive return.

    It also wires the §10b CVaR-aware sizing onto each row:

    * ``cvar_book_weight`` — the normalised long-book weight (0 for names outside the
      sized book; the held names sum to 1 / 100% gross).
    * ``cvar_5pct_kalman`` — per-name 5% expected shortfall (CVaR) of the posterior
      upside draws (decimal return units).
    * ``expected_vol_kalman`` — per-name std of the posterior upside draws (decimal).
    * ``reward_to_cvar`` — the STARR ratio (expected upside / binding tail risk) the
      weights are ranked on (dimensionless).
    * ``expected_sharpe_ratio`` — ``er_mean / er_sd`` of the structural-TS MC
      forward-return distribution (dimensionless).
    * ``p_upside_pos_cond`` — conditional probability of a positive upside given the
      smoother's state confidence (``mc_prob_pos`` × ``kalman_gain``), the quantity
      the book's ``p_long`` eligibility gate is applied to.

    The :class:`RiskBook` is reused when passed via ``risk_book`` (so the export and the
    §14b screen share one computation); otherwise it is recomputed from ``screen``.

    Set ``write=True`` to persist the rows via ``export_to_analytics_db``
    (``DB_ANALYTICS_SCHEMA``, default ``analytics``) with ``if_exists='replace'`` — the
    table is **dropped and recreated** on every run, not appended to, so the
    hand-maintained types and column comments do not survive. The regenerated DDL
    (including the unit-convention header and ``COMMENT ON COLUMN`` statements) is
    therefore rewritten to ``sql_scripts/analytics/kalman_filtered_price_targets.sql``
    on each write, keeping the checked-in schema honest.
    """
    model_df = panel.frame
    _post = idata.posterior
    _est = screen.ept  # (chain, draw, isin) — de-standardised price target
    _eu = screen.eu  # (chain, draw, isin) — de-standardised implied upside
    # (chain, draw, isin) — the decision latent: the local-level state's FILTERED
    # level at the snapshot (``state_now``), falling back to the t=0 anchor
    # ``risk_adj_return`` for pre-state-layer inference objects. ``signal_strength``
    # below is |mean| / sd of this latent, so it must be the same quantity the
    # screen and the price-target Monte-Carlo used.
    _rar = resolve_screen_latent(_post)

    kalman_estimate = _est.mean(('chain', 'draw')).values
    kalman_variance = _est.var(('chain', 'draw')).values
    expected_upside_kalman = _eu.mean(('chain', 'draw')).values
    implied_return_kalman = expected_upside_kalman
    kalman_gain = _post['achieve_prob'].mean(('chain', 'draw')).values
    _rar_mean = _rar.mean(('chain', 'draw')).values
    _rar_sd = _rar.std(('chain', 'draw')).values
    signal_strength = np.where(_rar_sd > 0, np.abs(_rar_mean) / _rar_sd, np.nan)

    def _idcol(name):
        if name in model_df.columns:
            return model_df[name].to_numpy()
        return np.full(len(model_df), None, dtype=object)

    def _numcol(name):
        """Numeric (double precision) column with a NaN fallback when absent."""
        if name in model_df.columns:
            return pd.to_numeric(model_df[name], errors='coerce').to_numpy(dtype='float64')
        return np.full(len(model_df), np.nan, dtype='float64')

    kalman_results = pd.DataFrame({
        'isin': np.asarray(panel.isins),
        'ticker': _idcol('ticker'),
        'name': _idcol('name'),
        'trading_region': _idcol('trading_region'),
        'trading_country': _idcol('trading_country'),
        'trading_country_name': _idcol('trading_country_name'),
        'region': _idcol('region'),
        'country': _idcol('country'),
        'country_name': _idcol('country_name'),
        'unit': _idcol('unit'),
        'unit_name': _idcol('unit_name'),
        'exchange': _idcol('exchange'),
        'exchange_name': _idcol('exchange_name'),
        'sector': _idcol('sector'),
        'industry': _idcol('industry'),
        'style_class': _idcol('style_class'),
        'size_class': _idcol('size_class'),
        'last_updated': _idcol('last_updated'),
        'next_earnings': _idcol('next_earnings'),
        'next_earnings_when': _idcol('next_earnings_when'),
        'next_earnings_status': _idcol('next_earnings_status'),
        'fy_end_date': _idcol('fy_end_date'),
        'next_fiscal_quarter': _idcol('next_fiscal_quarter'),
        'income_statement_report_date': _idcol('income_statement_report_date'),
        'next_income_statement_report_date': _idcol('next_income_statement_report_date'),
        'next_fy_end_date': _idcol('next_fy_end_date'),
        'expected_report_date': _idcol('expected_report_date'),
        'days_to_next_earnings': _idcol('days_to_next_earnings'),
        'days_since_last_report': _idcol('days_since_last_report'),
        'days_to_next_fy_end': _idcol('days_to_next_fy_end'),
        'days_to_next_fiscal_quarter': _idcol('days_to_next_fiscal_quarter'),
        'days_to_next_report': _idcol('days_to_next_report'),
        'days_to_expected_report': _idcol('days_to_expected_report'),
        'days_since_fy_end': _idcol('days_since_fy_end'),
        'market_cap': model_df['market_cap'].to_numpy(),
        'enterprise_value': model_df['enterprise_value'].to_numpy(),
        'mcap_country_r': _numcol('feat_mcap_country_r'),
        'beta': model_df['feat_avg_beta'].to_numpy(),
        'original_price': model_df['last_price'].to_numpy(),
        'original_target': model_df['observed_pt'].to_numpy(),
        'price_target_kalman': kalman_estimate,
        'implied_upside': model_df['feat_implied_upside'].to_numpy(),
        'expected_return_kalman': expected_upside_kalman,
        'kalman_variance': kalman_variance,
        'kalman_gain': kalman_gain,
        'signal_strength': signal_strength,
        # Analyst Rating columns.
        'n_holds': _numcol('feat_holds'),
        'n_buys': _numcol('feat_buys'),
        'n_sells': _numcol('feat_sells'),
        'n_analysts': _numcol('n_analysts'),
        'feat_no_opinion': _numcol('feat_no_opinion'),
        'analyst_bullish_pct': _numcol('feat_analyst_bullish_pct'),
        'analyst_bearish_pct': _numcol('feat_analyst_bearish_pct'),
        'analyst_neutral_pct': _numcol('feat_analyst_neutral_pct'),
        'analyst_conviction': _numcol('feat_analyst_conviction'),
        'analyst_rating': _numcol('feat_analyst_rating'),
        'piotroski_f_score_median': _numcol('feat_median_piotroski_f_score'),
        'piotroski_f_score_fy': _numcol('feat_piotroski_f_score_fy'),
        'piotroski_f_score_neg1fy': _numcol('feat_piotroski_f_score_neg1fy'),
        'piotroski_f_score_neg2fy': _numcol('feat_piotroski_f_score_neg2fy'),
        'piotroski_f_score_neg3fy': _numcol('feat_piotroski_f_score_neg3fy'),
        'pt_achievement_1y': _numcol('feat_pt_achievement_1y'),
        'pt_range_hit_rate': _numcol('feat_pt_range_hit_rate'),
        # Consensus price-target levels + the *_ago price-target trail.
        'price_target_median': _numcol('price_target_median'),
        'price_target_high': _numcol('price_target_high'),
        'price_target_low': _numcol('price_target_low'),
        'price_target_1w_ago': _numcol('price_target_1w_ago'),
        'price_target_mtd_ago': _numcol('price_target_mtd_ago'),
        'price_target_1m_ago': _numcol('price_target_1m_ago'),
        'price_target_qtd_ago': _numcol('price_target_qtd_ago'),
        'price_target_3m_ago': _numcol('price_target_3m_ago'),
        'price_target_6m_ago': _numcol('price_target_6m_ago'),
        'price_target_ytd_ago': _numcol('price_target_ytd_ago'),
        'price_target_1y_ago': _numcol('price_target_1y_ago'),
        # Realised spot-price *_ago trail.
        'price_5d_ago': _numcol('price_5d_ago'),
        'price_3m_ago': _numcol('price_3m_ago'),
        'price_5y_ago': _numcol('price_5y_ago'),
        'price_1y_ago': _numcol('price_1y_ago'),
        'price_6m_ago': _numcol('price_6m_ago'),
        'price_qtd_ago': _numcol('price_qtd_ago'),
        'price_1m_ago': _numcol('price_1m_ago'),
        'price_1w_ago': _numcol('price_1w_ago'),
        'price_3y_ago': _numcol('price_3y_ago'),
    })

    # Pull valuation, posterior-band and MC summary columns straight from the §10 screen
    # table (the SSOT), keyed on isin — so the export never recomputes, nor drifts from,
    # those figures. Columns absent from an older screen are simply skipped.
    _screen_cols = ['expected_pt_hdi_lo', 'expected_pt_hdi_hi', 'risk_adj_return',
                    'er_mean', 'er_sd', 'er_p05', 'er_p50', 'er_p95', 'mc_prob_pos'
                    ]
    _present = [c for c in _screen_cols if c in screen.results.columns]
    if _present:
        kalman_results = kalman_results.merge(
            screen.results[['isin', *_present]], on='isin', how='left')

    # Resolve the CVaR-aware sizing book: reuse the one passed in (so the export and the
    # §14b screen share a single computation), otherwise recompute it from the screen —
    # exactly as the docstring promises. ``RiskBook.analytics`` carries the per-ISIN
    # book_weight / cvar05 / starr keyed on isin.
    rb = _resolve_risk_book(risk_book, idata, panel, screen, screen.results)
    _sized = (rb.analytics[['isin', 'book_weight', 'cvar05', 'starr', 'exp_vol',
                            'expected_sharpe', 'p_upside_pos_cond']]
              .rename(columns={'book_weight': 'cvar_book_weight',
                               'cvar05': 'cvar_5pct_kalman',
                               'starr': 'reward_to_cvar',
                               'exp_vol': 'expected_vol_kalman',
                               'expected_sharpe': 'expected_sharpe_ratio'}))
    kalman_results = kalman_results.merge(_sized, on='isin', how='left')

    # Guarantee the column exists regardless of whether a risk book was passed.
    if 'cvar_book_weight' not in kalman_results.columns:
        kalman_results['cvar_book_weight'] = 0.0
    kalman_results['cvar_book_weight'] = kalman_results['cvar_book_weight'].fillna(0.0)
    _held = int((kalman_results['cvar_book_weight'] > 0).sum())
    print(f'Built kalman_filtered_price_targets row-set: {kalman_results.shape}  '
          f'(CVaR-aware book: {_held} sized names, gross='
          f'{kalman_results["cvar_book_weight"].sum() * 100:.0f}%).')

    # Decision dashboard over the export columns (risk-return map, MC return fan,
    # gain conditioning, tail asymmetry). Display-only and best-effort — a plot
    # failure must never block the DB export below.
    try:
        plot_kalman_results_overview(kalman_results, book_summary=rb.summary)
    except Exception as exc:  # pragma: no cover - display-only
        print(f'kalman_results overview dashboard skipped: {exc!r}')

    # Shrinkage view via the shared package builder (signed-log axes handle the
    # analyst fat tails; replaces the former in-script price-space scatter —
    # 0.9.9.11 consolidation). Guarded import: the visualizations package uses
    # Python-3.14-only multi-exception syntax and may be absent/broken on older
    # interpreters; the figure is display-only either way.
    try:
        from probabilistic_ml_model.visualizations.expected_returns_viz import (
            create_kalman_vs_raw_scatter,
        )
        # The shared builder keys on ``implied_return_kalman``; the export
        # names the same decimal quantity ``expected_return_kalman``.
        _render_plotly(create_kalman_vs_raw_scatter(kalman_results.rename(
            columns={'expected_return_kalman': 'implied_return_kalman'})))
    except Exception as exc:  # pragma: no cover - optional reuse
        print(f'kalman-vs-raw shrinkage scatter skipped: {exc!r}')

    # Curate the top-25 preview so the fiscal-calendar date columns surface up front
    # (after the identifiers) rather than being buried at the right edge. Round only
    # the numeric columns -- the DATE / day-count columns are date/object dtypes that
    # ``DataFrame.round`` would either ignore or choke on.
    _date_cols = [c for c in (*FISCAL_CALENDAR_COLS_ALL, 'next_earnings_when',
                              'next_earnings_status', *DAY_COUNT_COLS_ALL)
                  if c in kalman_results.columns]
    _id_cols = [c for c in
                ('isin', 'ticker', 'name', 'trading_region', 'trading_country',
                 'trading_country_name', 'region', 'country', 'unit', 'exchange',
                 'exchange_name', 'unit_name', 'country_name',
                 'sector', 'industry', 'size_class', 'style_class')
                if c in kalman_results.columns]
    _other_cols = [c for c in kalman_results.columns
                   if c not in (*_id_cols, *_date_cols)]
    _preview_cols = [*_id_cols, *_date_cols, *_other_cols]
    _num_cols = kalman_results[_other_cols].select_dtypes(include='number').columns
    display(kalman_results[_preview_cols]
            .sort_values('cvar_book_weight', ascending=False)
            .head(25).round({c: 4 for c in _num_cols}))

    write_analytics_ddl(kalman_results)
    if write:
        _n = export_to_analytics_db(kalman_results, _ANALYTICS_TABLE,
                                    if_exists='replace')
        note_analytics_written()
        print(f'Replaced analytics.{_ANALYTICS_TABLE} with {_n} rows.')
    else:
        print('write=False -> not persisted. Pass write=True to replace the DB sink.')
    return kalman_results


# =============================================================================
# 10K. Shared single-series fit + forecast driver (+ universe application)
# =============================================================================
# Canonical wrapper around ``KalmanFilterModel.KalmanFilterPriceTarget.fit`` /
# ``.forecast`` so every time-series section (§10K universe consensus, §11
# single-ISIN, §12 mingled cohort, and the §11b/§12b stochastic-volatility twins)
# shares one instantiate -> fit -> build_forecast_specs -> forecast chain rather
# than re-inlining it per call site.

# Posterior variables summarised after every single-series fit (§10K/§11/§12);
# SV twins report / trace the volatility-walk subset.
_KF_SUMMARY_VARS: tuple[str, ...] = (
    'sigma_state', 'sigma_obs', 'beta_trend', 'log_state_init',
    'vol_step_size', 'nu_obs', 'vol_anchor_offset',
)
_SV_SUMMARY_VARS: tuple[str, ...] = (
    'sigma_state', 'vol_step_size', 'nu_obs', 'vol_anchor_offset', 'beta_trend',
)
_SV_TRACE_VARS: tuple[str, ...] = ('vol_step_size', 'nu_obs', 'vol_anchor_offset')


@dataclass
class KalmanFitResult:
    """Bundle returned by :func:`fit_kalman_model`.

    Carries the fitted :class:`KalmanFilterPriceTarget`, its inference object and
    PyMC model, and — when a fiscal-calendar ``forecast_df`` was supplied and at
    least one future event qualified — the forward structural forecast plus the
    resolved ``(horizons_days, fiscal_dates, labels)`` triple.

    Attributes
    ----------
    kf
        The fitted model instance (retains fit context for further forecasts).
    idata
        Inference data / DataTree from :meth:`KalmanFilterPriceTarget.fit`.
    model
        The PyMC model object.
    pred
        The ``forecast`` ``predictions`` group (``None`` when no forecast ran).
    horizons_days, fiscal_dates, labels
        The resolved forecast-horizon triple (empty when no forecast ran).
    last_obs
        As-of anchor the forecast horizons were measured beyond.
    last_price
        Spot price forwarded to both ``fit`` and ``forecast`` (spot anchoring).
    """

    kf: KalmanFilterPriceTarget
    idata: Any
    model: Any
    pred: Optional[Any] = None
    horizons_days: list[int] = field(default_factory=list)
    fiscal_dates: list = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    last_obs: Optional[pd.Timestamp] = None
    last_price: Optional[float] = None

    @property
    def n_divergences(self) -> int:
        """Post-tuning divergence count (0 when ``sample_stats`` is unavailable)."""
        ss = getattr(self.idata, 'sample_stats', None)
        if ss is None or 'diverging' not in getattr(ss, 'data_vars', {}):
            return 0
        try:
            return int(ss['diverging'].sum())
        except Exception:  # pragma: no cover - defensive
            return 0

    @property
    def fit_kind(self) -> str:
        """Human label for the parameterization actually fitted."""
        return _kf_fit_kind(self.idata)

    @property
    def has_forecast(self) -> bool:
        """Whether a structural forecast was produced."""
        return self.pred is not None


def _forecast_table(res: "KalmanFitResult",
                    last_price: Optional[float] = None) -> pd.DataFrame:
    """Tidy per-fiscal-event forecast table from a fitted :class:`KalmanFitResult`.

    Shared by §10K/§11/§12 so the three sections render byte-identical forecast
    tables. ``implied_upside_pct`` is appended only when a spot price is given
    (display-boundary percent scaling).
    """
    _pg = res.pred.predictions
    tbl = pd.DataFrame({
        'fiscal_event': res.labels,
        'date': [d.strftime('%Y-%m-%d') for d in res.fiscal_dates],
        'horizon_days': res.horizons_days,
        'forecast_pt': _pg['forecast_pt'].mean(('chain', 'draw')).values,
        'forecast_pt_lo': _pg['forecast_pt'].quantile(_HDI_LO, dim=('chain', 'draw')).values,
        'forecast_pt_hi': _pg['forecast_pt'].quantile(_HDI_HI, dim=('chain', 'draw')).values,
    })
    if last_price:
        tbl['implied_upside_pct'] = (tbl['forecast_pt'] / last_price - 1.0) * 100
    return tbl


def _plot_sigma_obs_path(dates, so, *, color: str, title: str) -> None:
    """Render the posterior mean + 94% band of a time-varying ``sigma_obs`` path.

    Shared by the §11b/§12b stochastic-volatility twins (identical apart from
    accent color and title).
    """
    _mean = so.mean(('chain', 'draw')).values
    _lo = so.quantile(_HDI_LO, dim=('chain', 'draw')).values
    _hi = so.quantile(_HDI_HI, dim=('chain', 'draw')).values
    fig = go.Figure()
    _plotly_band(fig, dates, _lo, _hi, color=color, alpha=0.25,
                 name='94% HDI', showlegend=True)
    fig.add_trace(go.Scatter(x=dates, y=_mean, mode='lines',
                             line=dict(color=color, width=2),
                             hovertemplate=('%{x|%Y-%m-%d}<br>'
                                            'σ_obs = %{y:.4f}<extra></extra>'),
                             name='posterior mean σ_obs(t)', legendgroup='sigma'))
    fig.update_yaxes(title_text='σ_obs (log-price scale)')
    fig.update_xaxes(title_text='asof_date', tickformat='%b %y')
    fig.update_layout(title=title, legend=dict(font_size=_LEGEND_FONT_SIZE))
    _render_plotly(fig, height=360, hovermode='x unified')


def _resolve_forecast_anchor(
        forecast_df: Optional[pd.DataFrame],
        dates: Optional[pd.DatetimeIndex],
        anchor_col: Optional[str],
        aggregate: str,
) -> Optional[pd.Timestamp]:
    """Resolve the as-of anchor the forecast horizons are measured beyond.

    Prefers the ``anchor_col`` snapshot timestamp (``last_updated`` on
    ``pml.mv_pymc_kalman_pt`` by default) so the projection starts from the
    freshest as-of date the MV was refreshed at rather than from the last
    ``*_ago`` observation. Falls back to ``dates.max()`` when the column is
    absent / unparseable, and never returns an anchor *earlier* than the last
    observation (a horizon must land in the future).

    Parameters
    ----------
    forecast_df
        Frame carrying ``anchor_col`` (and the fiscal-calendar DATE coords).
    dates
        Observation timestamps of the fitted series.
    anchor_col
        Snapshot timestamp column (``"last_updated"`` by default). ``None``
        disables the snapshot anchor.
    aggregate
        ``"first"`` (single ISIN) takes the first non-null anchor; anything else
        (``"median"`` cohort) takes the freshest (``max``) anchor.
    """
    last_obs = (pd.Timestamp(pd.DatetimeIndex(dates).max())
                if dates is not None and len(dates) else None)
    if forecast_df is not None and anchor_col and anchor_col in forecast_df.columns:
        parsed = pd.to_datetime(forecast_df[anchor_col], errors='coerce').dropna()
        if not parsed.empty:
            anchor = pd.Timestamp(parsed.iloc[0] if aggregate == 'first' else parsed.max())
            last_obs = anchor if last_obs is None else max(last_obs, anchor)
    return last_obs


def fit_kalman_model(
        price_targets: np.ndarray,
        isin: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        last_price: Optional[float] = None,
        sectors: Optional[np.ndarray] = None,
        categories_df: Optional[pd.DataFrame] = None,
        hierarchy_levels: Optional[list[str]] = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        random_seed: int = RANDOM_SEED,
        trend: bool = True,
        trend_sigma: float = 0.5,
        stochastic_volatility: bool = False,
        realized_vol: Optional[np.ndarray] = None,
        parameterization: str = 'marginalized',
        nuts_sampler: Optional[str] = None,
        *,
        forecast_df: Optional[pd.DataFrame] = None,
        forecast_aggregate: Literal['first', 'median'] = 'median',
        forecast_anchor_col: Optional[str] = 'last_updated',
        forecast_horizons: Optional[Sequence] = None,
        **sample_kwargs: Any,
) -> KalmanFitResult:
    """Fit the single-series GRW filter and project it to future fiscal events.

    Thin orchestration wrapper around :meth:`KalmanFilterPriceTarget.fit` and
    :meth:`KalmanFilterPriceTarget.forecast` (both from
    ``probabilistic_ml_model.pymc_models.KalmanFilterModel``) so every time-series
    section of this workflow shares one canonical fit + forecast driver instead of
    re-inlining the instantiate -> fit -> :meth:`build_forecast_specs` -> forecast
    chain.

    Spot anchoring is **adopted**: ``last_price`` is forwarded to ``fit`` (anchoring
    the latent log-level prior at ``log(last_price)`` and exposing
    ``implied_upside``) *and* reused by ``forecast``. The forecast as-of anchor is
    resolved from ``forecast_anchor_col`` (``last_updated`` on
    ``pml.mv_pymc_kalman_pt`` by default) together with the fiscal-calendar DATE
    coords walked by :meth:`build_forecast_specs`.

    Parameters
    ----------
    price_targets, isin, dates, last_price, sectors, categories_df, hierarchy_levels, samples, tune, chains, target_accept, random_seed, trend, trend_sigma, stochastic_volatility, realized_vol, parameterization, nuts_sampler, **sample_kwargs
        Forwarded verbatim to :meth:`KalmanFilterPriceTarget.fit` (see that method
        for the full contract).
    forecast_df : pandas.DataFrame, optional
        Frame carrying the fiscal-calendar DATE columns (and ``forecast_anchor_col``)
        consumed by :meth:`build_forecast_specs`. When ``None`` no forecast is run
        and the returned :class:`KalmanFitResult` has ``pred=None``.
    forecast_aggregate : {"first", "median"}
        ``"first"`` for a single-ISIN row; ``"median"`` for a cohort frame.
    forecast_anchor_col : str, optional
        Snapshot timestamp column used as the as-of anchor (default
        ``"last_updated"``). Falls back to ``dates.max()`` when absent / unparseable.
    forecast_horizons : Sequence, optional
        Subset / ordering of :data:`FISCAL_HORIZONS` to consider. ``None`` uses the
        full canonical set.

    Returns
    -------
    KalmanFitResult
        The fitted model, inference object, PyMC model, and (optional) forecast.
    """
    kf = KalmanFilterPriceTarget()
    idata, model = kf.fit(
        price_targets=price_targets,
        isin=isin,
        dates=dates,
        last_price=last_price,
        sectors=sectors,
        categories_df=categories_df,
        hierarchy_levels=hierarchy_levels,
        samples=samples,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        trend=trend,
        trend_sigma=trend_sigma,
        stochastic_volatility=stochastic_volatility,
        realized_vol=realized_vol,
        parameterization=parameterization,
        nuts_sampler=nuts_sampler,
        **sample_kwargs,
    )

    result = KalmanFitResult(kf=kf, idata=idata, model=model, last_price=last_price)
    if forecast_df is None:
        return result

    last_obs = _resolve_forecast_anchor(
        forecast_df, dates, forecast_anchor_col, forecast_aggregate)
    if last_obs is None:
        return result
    result.last_obs = last_obs

    spec_kwargs = {} if forecast_horizons is None else {'horizons': forecast_horizons}
    horizons_days, fiscal_dates, labels = KalmanFilterPriceTarget.build_forecast_specs(
        forecast_df, last_obs, aggregate=forecast_aggregate, **spec_kwargs)
    if not horizons_days:
        return result

    result.horizons_days = horizons_days
    result.fiscal_dates = fiscal_dates
    result.labels = labels
    result.pred = kf.forecast(
        horizons_days, fiscal_dates=fiscal_dates, labels=labels,
        last_price=last_price, random_seed=random_seed)
    return result


@dataclass(frozen=True)
class UniverseConsensus:
    """Universe-wide consensus price-target trail pooled from ``kalman_df``.

    Attributes
    ----------
    consensus
        One row per ``asof_date`` bucket: median ``price_target`` and ``n_isin``
        (distinct contributing names).
    dates
        ``consensus['asof_date']`` as a :class:`pandas.DatetimeIndex`.
    observed
        ``consensus['price_target']`` as a NumPy array (the fit's response).
    last_price
        Universe-median spot price (``None`` when unavailable) — the spot anchor.
    label
        Human label used as the fit's ``isin`` metadata.
    n_rows, n_isin
        Size of the contributing ``kalman_df`` snapshot (rows / distinct ISINs).
    """

    consensus: pd.DataFrame
    dates: pd.DatetimeIndex
    observed: np.ndarray
    last_price: Optional[float]
    label: str
    n_rows: int
    n_isin: int


def build_universe_consensus(
        kalman_df: pd.DataFrame,
        *,
        freq: str = 'W',
        min_observations: int = 2,
) -> Optional[UniverseConsensus]:
    """Collapse the full ``kalman_df`` snapshot into one consensus PT series.

    Every ``kalman_df`` row contributes: each name's embedded ``price_target*_ago``
    and raw spot-price ``price_*_ago`` cohort is unpivoted (anchored per ISIN on
    ``income_statement_report_date``), pooled across the whole universe, and the
    cross-sectional **median** taken per ``freq`` bucket — the universe-wide
    analogue of the §12 earnings-window mingle. The spot-price trail entered the
    MV alongside the target trail, so the pooled fair-value consensus blends the
    realised price path with the analyst-target path (mirroring the §12/§13
    ``pml_df``-sourced cohorts).

    Parameters
    ----------
    kalman_df
        Full ``pml.mv_pymc_kalman_pt`` snapshot (one row per ISIN); the MV's
        ``observed_pt`` seeds the point-in-time observation.
    freq
        Pandas period alias the pooled ``asof_date`` axis is bucketed to. Weekly by
        default so the marginalized ``MvNormal`` covariance stays small even though
        the per-ISIN fiscal anchors scatter the raw as-of dates almost daily.
    min_observations
        Minimum number of consensus buckets required for a Kalman fit.

    Returns
    -------
    Optional[UniverseConsensus]
        The pooled consensus trail, or ``None`` when no ``*_ago`` history is
        present / fewer than ``min_observations`` buckets survive.
    """
    long_df, _eligible, date_col = KalmanFilterPriceTarget.build_price_target_history(
        kalman_df, now_cols=('observed_pt',),
        timestamp_col='last_updated',
        fiscal_anchor_col='fy_end_date',
    )
    if date_col is None or long_df.empty:
        print('No price_target*_ago / price_*_ago history in kalman_df; '
              'universe consensus unavailable.')
        return None

    bucket = long_df['asof_date'].dt.to_period(freq).dt.start_time
    consensus = (
        long_df.assign(asof_date=bucket)
        .groupby('asof_date', as_index=False)
        .agg(price_target=('price_target', 'median'), n_isin=('isin', 'nunique'))
        .sort_values('asof_date').reset_index(drop=True)
    )
    if len(consensus) < min_observations:
        print(f'Universe consensus has only {len(consensus)} bucket(s); '
              f'need >= {min_observations} for a Kalman fit. Section skipped.')
        return None

    last_price = None
    if 'last_price' in kalman_df.columns:
        _med = float(np.nanmedian(kalman_df['last_price']))
        if np.isfinite(_med) and _med > 0:
            last_price = _med
    n_isin = int(kalman_df['isin'].nunique()) if 'isin' in kalman_df.columns else len(kalman_df)
    print(f'Universe consensus: {len(kalman_df)} kalman_df rows / {n_isin} ISINs -> '
          f'{len(long_df)} pooled (isin, asof_date) observations -> {len(consensus)} '
          f'{freq!r}-bucket median consensus points spanning '
          f'{consensus["asof_date"].min():%Y-%m-%d} ... '
          f'{consensus["asof_date"].max():%Y-%m-%d}.')
    return UniverseConsensus(
        consensus=consensus,
        dates=pd.DatetimeIndex(consensus['asof_date']),
        observed=consensus['price_target'].to_numpy(),
        last_price=last_price,
        label=f'UNIVERSE (n={n_isin})',
        n_rows=int(len(kalman_df)),
        n_isin=n_isin,
    )


def report_universe_kalman_fit(res: KalmanFitResult,
                               universe: UniverseConsensus) -> pd.DataFrame:
    """Render the §10K universe fit: sampler summary, smoothed path, forecast, table.

    Mirrors the §12 mingled-cohort presentation so the universe consensus reads on
    the same axes: posterior summary of the state-space parameters, the smoothed
    ``expected_pt`` path vs the observed consensus, the structural forecast to the
    universe-median fiscal events (when one ran), and a tidy per-bucket comparison.

    Parameters
    ----------
    res
        Bundle returned by :func:`fit_kalman_model`.
    universe
        The pooled consensus trail the fit was run on.

    Returns
    -------
    pandas.DataFrame
        Per-bucket comparison of observed vs smoothed consensus (with HDI bounds).
    """
    dates, observed = universe.dates, universe.observed
    label, last_price = universe.label, universe.last_price

    print(f'{res.fit_kind} fit: {len(observed)} obs, '
          f'{(dates.max() - dates.min()).days}d span, divergences={res.n_divergences}.')
    display(azs.summary(
        res.idata,
        var_names=_present_vars(res.idata, _KF_SUMMARY_VARS),
        round_to=4))

    _safe_show(plot_price_target_path(
        res.idata, observed=observed, dates=dates,
        last_price=last_price, ticker=label,
    ))

    if res.has_forecast:
        fig_fc, _ = plot_kalman_forecast_returns(
            res.idata, res.pred, observed=observed, dates=dates,
            last_price=last_price, ticker=label)
        _show_fig(fig_fc)
        fc_tbl = _forecast_table(res, last_price=last_price)
        print(f'Universe structural forecast to {len(res.horizons_days)} fiscal '
              f'event(s) beyond {res.last_obs:%Y-%m-%d}:')
        display(fc_tbl.round(3))
    else:
        print('No qualifying future fiscal events -> structural forecast skipped.')

    _post = res.idata.posterior['state']
    _stk = _post.stack(s=('chain', 'draw'))
    comparison = pd.DataFrame({
        'asof_date': dates.strftime('%Y-%m-%d'),
        'n_isin': universe.consensus['n_isin'].to_numpy(),
        'observed_pt': observed,
        'expected_pt': _post.mean(('chain', 'draw')).values,
        'expected_pt_hdi_lo': _stk.quantile(_HDI_LO, dim='s').values,
        'expected_pt_hdi_hi': _stk.quantile(_HDI_HI, dim='s').values,
        'last_price': last_price,
    })
    comparison['expected_vs_observed_pct'] = (
            (comparison['expected_pt'] / comparison['observed_pt'] - 1.0) * 100
    )
    print('Universe consensus comparison (observed_pt vs smoothed expected_pt):')
    display(comparison.round(3))
    return comparison


def run_universe_kalman_fit(kalman_df: pd.DataFrame,
                            **fit_kwargs: Any) -> Optional[KalmanFitResult]:
    """Section 10K: fit the shared driver on the full ``kalman_df`` universe.

    Pools **every** ``kalman_df`` row into the median consensus trail
    (:func:`build_universe_consensus`) and routes it through the canonical
    :func:`fit_kalman_model` driver — spot-anchored at the universe-median
    ``last_price`` and structurally forecast to the universe-median fiscal events
    (``forecast_df=kalman_df``, ``forecast_aggregate='median'``). ``fit_kwargs``
    override the marginalized-GRW (+trend, nutpie) defaults.

    Parameters
    ----------
    kalman_df
        Full ``pml.mv_pymc_kalman_pt`` snapshot (one row per ISIN).
    **fit_kwargs
        Overrides forwarded to :func:`fit_kalman_model`.

    Returns
    -------
    Optional[KalmanFitResult]
        The fitted bundle, or ``None`` when the section is skipped (no history /
        DB-independent guard tripped).
    """
    try:
        universe = build_universe_consensus(kalman_df)
        if universe is None:
            return None

        kwargs: dict[str, Any] = dict(
            samples=2500, tune=2500, chains=4, target_accept=0.90,
            random_seed=RANDOM_SEED, parameterization='marginalized', trend=True,
            nuts_sampler='nutpie',
            forecast_df=kalman_df, forecast_aggregate='median',
            forecast_anchor_col='fy_end_date',
        )
        kwargs.update(fit_kwargs)

        print(f'Fitting universe-consensus Kalman filter on {len(universe.observed)} '
              f'consensus observations ({universe.label}).')
        res = fit_kalman_model(price_targets=universe.observed, isin=universe.label, dates=universe.dates,
                               last_price=universe.last_price, **kwargs)
        report_universe_kalman_fit(res, universe)
        return res
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 10K (universe shared-driver fit) skipped: {e!r}')
        return None


# =============================================================================
# 11. Single-ISIN time-series Kalman filter (+ 11b stochastic volatility)
# =============================================================================
def run_single_isin_filter(frame: pd.DataFrame, engine,
                           config: Optional[KalmanRunConfig] = None) -> Optional[dict]:
    """Fit the literal single-security GRW filter on the richest ``*_ago`` history.

    The time axis is reconstructed from the embedded ``*_ago`` price-target cohort,
    anchored on ``income_statement_report_date`` and projected forward to the next
    fiscal events via ``KalmanFilterPriceTarget.forecast()``. Uses the funnel-free
    *marginalized* GRW with a structural trend. ``frame`` is the fused panel's
    modelling frame (``panel.frame``). Returns context for the §11b SV variant
    (or ``None`` when no ISIN has >=2 ``*_ago`` observations / DB is unavailable).
    """
    model_df = frame
    cfg = config if config is not None else get_run_config()
    try:
        keep = ('isin', 'ticker', 'name', 'last_price', 'price_target', 'market_cap','market_cap_country_r', 'enterprise_value',
                'income_statement_report_date', 'next_earnings', 'fy_end_date', 'next_fiscal_quarter', 'last_updated',
                'next_income_statement_report_date', 'next_fy_end_date', 'expected_report_date')
        hist_cols, col_sql = fetch_history_columns(engine, keep)
        cohort = model_df['isin'].astype(str).tolist()
        with engine.connect() as conn:
            # Rank the cohort by the most recent earnings (next_earnings closest to
            # today), breaking ties by the largest market cap, and pull the top 50
            # candidates.  cf. SELECT GREATEST(market_cap) WHERE next_earnings = current_date.
            # The first of these that has >=2 *_ago observations is fitted below.
            snap = pd.read_sql(
                text(
                    f'SELECT {col_sql} FROM pml.pml_df '
                    'WHERE isin = ANY(:isins) '
                    '  AND next_earnings IS NOT NULL '
                    '  AND market_cap_country_r >= :min_rank '
                    'ORDER BY ABS(next_earnings - CURRENT_DATE) , '
                    '         market_cap DESC '
                    'LIMIT :lim'
                ),
                conn, params={'isins': cohort,
                              'min_rank': cfg.min_mcap_country_rank,
                              'lim': cfg.candidate_limit},
            )
        n_ago = sum(c.endswith('_ago') for c in hist_cols)
        print(f'Pulled pml.pml_df history frame: {snap.shape}  ({n_ago} *_ago columns).')

        long_df, eligible, date_col = KalmanFilterPriceTarget.build_price_target_history(
            snap, now_cols=('price_target',),
            fiscal_anchor_col='fy_end_date',
        )
        # Honour the SQL ranking: take the first candidate (most recent earnings /
        # largest market cap) that is eligible, falling back to select_target_isin.
        eligible_isins = set(eligible.index.astype(str)) if eligible is not None else set()
        ranked = snap['isin'].astype(str).tolist()
        chosen = next((i for i in ranked if i in eligible_isins), None)
        if chosen is None:
            chosen = KalmanFilterPriceTarget.select_target_isin(eligible, cohort=cohort)

        if chosen is None or date_col is None:
            print('No ISIN has >= 2 *_ago price-target observations; section skipped.')
            return None

        ts = (long_df.loc[long_df['isin'] == chosen, ['asof_date', 'price_target']]
              .dropna().sort_values('asof_date').reset_index(drop=True))
        dates = pd.DatetimeIndex(ts['asof_date'])
        observed = ts['price_target'].to_numpy()

        _row = model_df.loc[model_df['isin'] == chosen]
        ticker = str(_row['ticker'].iloc[0]) if len(_row) else str(chosen)
        last_price = float(_row['last_price'].iloc[0]) if len(_row) else None

        print(f'Fitting single-ISIN Kalman filter for {chosen} ({ticker}) '
              f'on {len(observed)} observations spanning '
              f'{dates.min():%Y-%m-%d} – {dates.max():%Y-%m-%d}.')

        # Fit + structural forecast via the shared driver. Spot anchoring is adopted
        # (last_price flows into fit and forecast); the forecast as-of anchor is the
        # snapshot ``last_updated`` (fused-panel row) plus the fiscal-calendar DATE
        # coords resolved by build_forecast_specs.
        res = fit_kalman_model(price_targets=observed, isin=str(chosen), dates=dates, last_price=last_price,
                               samples=2500, tune=2500, chains=4, target_accept=0.90, random_seed=RANDOM_SEED,
                               trend=True, parameterization='auto', nuts_sampler='nutpie', forecast_df=_row,
                               forecast_aggregate='first')
        kf, kf_idata = res.kf, res.idata

        print(f'{res.fit_kind} fit: {len(observed)} obs, '
              f'{dates.max().year - dates.min().year}y span, divergences={res.n_divergences}.')
        display(azs.summary(
            kf_idata,
            var_names=_present_vars(kf_idata, _KF_SUMMARY_VARS),
            round_to=4))

        _safe_show(plot_price_target_path(
            kf_idata, observed=observed, dates=dates,
            last_price=last_price, ticker=ticker,
        ))

        # Structural forecast to the next fiscal events. Horizons, future dates and
        # human labels are resolved from the canonical FISCAL_HORIZONS map (SSOT =
        # the days_* aliases on pml.mv_pymc_kalman_pt), so every label matches its
        # own date column instead of the previously mislabelled hand-built tuples.
        if res.has_forecast:
            fig_fc, _ = plot_kalman_forecast_returns(
                kf_idata, res.pred, observed=observed, dates=dates,
                last_price=last_price, ticker=ticker)
            _show_fig(fig_fc)
            fc_tbl = _forecast_table(res, last_price=last_price)
            print(f'Structural forecast to {len(res.horizons_days)} fiscal event(s) '
                  f'for {ticker}:')
            display(fc_tbl.round(3))
        else:
            print(f'No future fiscal event beyond the last observation for {chosen}; '
                  f'forecast skipped.')

        return {'kf': kf, 'chosen': chosen, 'dates': dates, 'observed': observed,
                'ticker': ticker, 'last_price': last_price, 'row': _row}
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 11 (single-ISIN time-series Kalman) skipped: {e!r}')
        return None


def run_single_isin_stochastic_vol(ctx: Optional[dict]) -> None:
    """Section 11b: re-fit the single-ISIN series with stochastic volatility.

    The scalar ``sigma_obs`` is replaced by a latent log-volatility random walk
    (``sigma_obs_t = exp(log_vol_t)``) under a robust Student-t likelihood. The
    log-vol walk is anchored at the constant ``log(scale)`` of the observed scatter
    (the absolute ``feat_vol_*`` term-structure anchor was retired with the
    ``feat_vol_drift`` MV refactor; ``fit(realized_vol=...)`` still accepts an
    external volatility series when one is available).
    """
    try:
        if not ctx:
            print('Section 11 produced no fitted ISIN; stochastic-volatility variant skipped.')
            return
        chosen, dates, observed = ctx['chosen'], ctx['dates'], ctx['observed']
        ticker, _row = ctx['ticker'], ctx['row']
        last_price = ctx.get('last_price')

        # Fit + forecast via the shared driver (SV path; spot anchoring adopted).
        res_sv = fit_kalman_model(price_targets=observed, isin=str(chosen), dates=dates, last_price=last_price,
                                  samples=2500, tune=2500, chains=4, random_seed=RANDOM_SEED, trend=True,
                                  stochastic_volatility=False, parameterization='auto',
                                  nuts_sampler='nutpie', forecast_df=_row, forecast_aggregate='first')
        kf_sv, kf_sv_idata = res_sv.kf, res_sv.idata

        print(f'Stochastic-volatility fit: {len(observed)} obs, divergences={res_sv.n_divergences}.')
        _sv_vars = [v for v in _SV_SUMMARY_VARS if v in kf_sv_idata.posterior]
        display(azs.summary(kf_sv_idata, var_names=_sv_vars, round_to=4))

        # Structural forecast to the next fiscal events (human-labelled horizons from
        # the canonical FISCAL_HORIZONS map). Rendered through the unified
        # plot_kalman_forecast: its SV-aware lower panel draws the posterior
        # sigma_obs(t) path beneath the state + posterior-predictive forecast,
        # mirroring the reference SV example (returns + posterior predictive above,
        # posterior volatility below).
        if res_sv.has_forecast:
            pred_sv = res_sv.pred
            fig_sv, _ = plot_kalman_forecast_returns(
                kf_sv_idata, pred_sv, observed=observed, dates=dates,
                last_price=last_price, ticker=f'{ticker} (SV)')
            _show_fig(fig_sv)
        else:
            # No future fiscal event beyond the last observation: still surface the
            # standalone posterior observation-volatility path.
            _plot_sigma_obs_path(
                dates, kf_sv_idata.posterior['sigma_obs'], color=C_VOL,
                title=f'Stochastic volatility - time-varying observation noise ({ticker})')

        _trace_vars = [v for v in _sv_vars if v in _SV_TRACE_VARS]
        if _trace_vars:
            _pc_sv = azp.plot_trace(
                kf_sv_idata, var_names=_trace_vars,
                figure_kwargs=_azp_figure_kwargs(
                    _forest_height_px(len(_trace_vars), per_row=210, base=110)),
            )
            with contextlib.suppress(Exception):
                _pc_sv.add_title('SV fit — volatility-walk trace ('
                                 + ', '.join(_trace_vars) + ')')
            _polish_facet_axes(_pc_sv)
            _safe_show(_pc_sv)
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 11b (stochastic volatility) skipped: {e!r}')


# =============================================================================
# 12. Mingle-ISIN earnings-window Kalman filter (+ 12b stochastic volatility)
# =============================================================================
def run_mingled_cohort_filter(frame: pd.DataFrame, engine,
                              config: Optional[KalmanRunConfig] = None) -> Optional[dict]:
    """Mingle the recent-earnings cohort into one consensus series and refit the filter.

    For every ISIN whose ``next_earnings`` lands in the ±``earnings_window_days``
    window (:class:`KalmanRunConfig`, default ±5 days) the embedded
    ``*_ago`` cohort is unpivoted and the cross-sectional **median** price target taken at
    each shared ``asof_date`` — a single earnings-cohort consensus over time. Fit with the
    marginalized GRW (+trend). ``frame`` is the fused panel's modelling frame
    (``panel.frame``). Returns context (``comparison``, ``snap``, ``mingled`` ...)
    for §12b and §14, or ``None`` when the window yields fewer than 2 observations.
    """
    model_df = frame
    cfg = config if config is not None else get_run_config()
    try:
        keep = ('isin', 'name', 'ticker', 'last_price', 'price_target', 'market_cap', 'market_cap_country_r', 'enterprise_value',
                'income_statement_report_date', 'next_earnings', 'fy_end_date', 'next_fiscal_quarter', 'last_updated',
                'next_income_statement_report_date', 'next_fy_end_date', 'expected_report_date',)
        hist_cols, col_sql = fetch_history_columns(engine, keep)
        with engine.connect() as conn:
            snap = pd.read_sql(
                text(f"""
                    SELECT {col_sql}
                    FROM pml.pml_df
                    WHERE next_earnings >= '{cfg.min_next_earnings}'
                      AND income_statement_report_date >= '{cfg.min_report_date}'
                      AND next_earnings >= current_date - INTERVAL ':w days'
                      AND next_earnings <= current_date + INTERVAL ':w days'
                      AND market_cap_country_r > :min_rank
                    ORDER BY market_cap DESC
                """.replace(':w', str(int(cfg.earnings_window_days)))),
                conn, params={'min_rank': cfg.min_mcap_country_rank},
            )
        n_ago = sum(c.endswith('_ago') for c in hist_cols)
        n_cohort = snap['isin'].nunique() if 'isin' in snap.columns else 0
        print(f'Recent-earnings window cohort: {snap.shape[0]} rows / {n_cohort} ISINs '
              f'({n_ago} *_ago columns).')

        long_df, _eligible, date_col = KalmanFilterPriceTarget.build_price_target_history(
            snap, now_cols=('price_target',),
            fiscal_anchor_col='fy_end_date',
        )
        if date_col is None or long_df.empty:
            print('No *_ago price-target history in the earnings window; section skipped.')
            return {'snap': snap, 'col_sql': col_sql, 'comparison': None, 'mingled': None}

        mingled = (
            long_df.groupby('asof_date', as_index=False)
            .agg(price_target=('price_target', 'median'), n_isin=('isin', 'nunique'))
            .sort_values('asof_date').reset_index(drop=True)
        )
        if len(mingled) < 2:
            print(f'Mingled cohort has only {len(mingled)} distinct as-of date(s); '
                  'need >= 2 for a Kalman fit. Section skipped.')
            return {'snap': snap, 'col_sql': col_sql, 'comparison': None, 'mingled': mingled}

        dates = pd.DatetimeIndex(mingled['asof_date'])
        observed = mingled['price_target'].to_numpy()
        last_price = (float(np.nanmedian(snap['last_price']))
                      if 'last_price' in snap.columns
                         and np.isfinite(np.nanmedian(snap['last_price'])) else None)
        label = f'EARNINGS-COHORT (n={n_cohort})'

        print(f'Fitting mingled-cohort Kalman filter on {len(observed)} consensus '
              f'observations spanning {dates.min():%Y-%m-%d} ... {dates.max():%Y-%m-%d}.')

        # Fit + cohort structural forecast via the shared driver (spot anchoring
        # adopted; cohort-median fiscal dates + last_updated anchor).
        res = fit_kalman_model(price_targets=observed, isin=label, dates=dates, last_price=last_price, samples=2500,
                               tune=2500, chains=4, target_accept=0.97, random_seed=RANDOM_SEED, trend=True,
                               parameterization='auto', nuts_sampler='nutpie', forecast_df=snap,
                               forecast_aggregate='first')
        kf, kf_idata = res.kf, res.idata

        print(f'{res.fit_kind} fit: {len(observed)} obs, '
              f'{(dates.max() - dates.min()).days}d span, divergences={res.n_divergences}.')
        display(azs.summary(
            kf_idata,
            var_names=_present_vars(kf_idata, _KF_SUMMARY_VARS),
            round_to=4))

        # (a) Headline composition.
        _safe_show(plot_price_target_path(
            kf_idata, observed=observed, dates=dates,
            last_price=last_price, ticker=label,
        ))

        # (a2) Structural forecast to the cohort's next fiscal events. Cohort-median
        # dates, day-offsets and human labels come from the canonical FISCAL_HORIZONS
        # map (SSOT = the days_* aliases on pml.mv_pymc_kalman_pt), so each label
        # tracks its own date column rather than the old mislabelled tuples.
        if res.has_forecast:
            fig_fc, _ = plot_kalman_forecast_returns(
                kf_idata, res.pred, observed=observed, dates=dates,
                last_price=last_price, ticker=label)
            _show_fig(fig_fc)
            fc_tbl = _forecast_table(res, last_price=last_price)
            print(f'Cohort structural forecast to {len(res.horizons_days)} fiscal event(s):')
            display(fc_tbl.round(3))
        else:
            print('No future fiscal event beyond the cohort\'s last observation; '
                  'structural forecast skipped.')

        # (b) ArviZ forest of the per-as-of-date expected_pt posterior HDIs.
        _state = kf_idata.posterior['state']
        _state = _state.assign_coords(time=[d.strftime('%Y-%m-%d') for d in dates])
        pc_state = azp.plot_forest(_state.to_dataset(), var_names=['state'],
                                   combined=True, backend='plotly',
                                   figure_kwargs=_azp_figure_kwargs(
                                       _forest_height_px(len(dates), per_row=30)))
        _ax_state = pc_state.viz['plot'].sel(column='forest').item()  # PlotlyPlot
        if last_price is not None:
            _add_ref_line(_ax_state, x=last_price, kind='anchor',
                          annotation_text='cohort last_price')
        _ax_state.update_xaxes(title_text='expected_pt (price)')
        pc_state.add_title(f'Expected price target (Kalman state) per as-of date - {label}')
        _safe_show(pc_state)

        # (c) Tidy comparison table.
        _post = kf_idata.posterior['state']
        _mean = _post.mean(('chain', 'draw')).values
        _stk = _post.stack(s=('chain', 'draw'))
        _lo = _stk.quantile(_HDI_LO, dim='s').values
        _hi = _stk.quantile(_HDI_HI, dim='s').values
        comparison = pd.DataFrame({
            'asof_date': dates.strftime('%Y-%m-%d'),
            'n_isin': mingled['n_isin'].to_numpy(),
            'observed_pt': observed,
            'expected_pt': _mean,
            'expected_pt_hdi_lo': _lo,
            'expected_pt_hdi_hi': _hi,
            'last_price': last_price,
        })
        comparison['expected_vs_observed_pct'] = (
                (comparison['expected_pt'] / comparison['observed_pt'] - 1.0) * 100
        )
        print('Mingled-cohort comparison (last_price vs observed_pt vs expected_pt):')
        display(comparison.round(3))

        return {'snap': snap, 'col_sql': col_sql, 'comparison': comparison,
                'mingled': mingled, 'label': label, 'last_price': last_price}
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 12 (mingle-ISIN earnings-window Kalman) skipped: {e!r}')
        return None


def run_mingled_cohort_stochastic_vol(frame: pd.DataFrame, ctx: Optional[dict]) -> None:
    """Section 12b: stochastic-volatility variant of the mingled earnings cohort.

    The log-vol walk is anchored at the constant ``log(scale)`` of the cohort's
    observed scatter (the cohort-median ``feat_vol_*`` term-structure anchor was
    retired with the ``feat_vol_drift`` MV refactor; ``fit(realized_vol=...)``
    still accepts an external volatility series when one is available).
    """
    try:
        mingled = ctx.get('mingled') if ctx else None
        if mingled is None or len(mingled) < 2:
            print('Section 12 produced no mingled cohort; stochastic-volatility variant skipped.')
            return
        label = ctx.get('label', 'EARNINGS-COHORT')
        last_price = ctx.get('last_price')

        _dates_sv = pd.DatetimeIndex(mingled['asof_date'])
        _observed_sv = mingled['price_target'].to_numpy()

        # Fit via the shared driver (SV path; no forward forecast in this section,
        # so ``forecast_df`` is omitted). Spot anchoring adopted.
        res_sv = fit_kalman_model(price_targets=_observed_sv, isin=label, dates=_dates_sv, last_price=last_price,
                                  samples=2500, tune=2500, chains=4, random_seed=RANDOM_SEED, trend=True,
                                  stochastic_volatility=True, parameterization='non_centered',
                                  nuts_sampler='nutpie')
        kf_sv, kf_sv_idata = res_sv.kf, res_sv.idata

        print(f'Mingled-cohort stochastic-volatility fit: {len(_observed_sv)} obs, '
              f'divergences={res_sv.n_divergences}.')
        _sv_vars = [v for v in _SV_SUMMARY_VARS if v in kf_sv_idata.posterior]
        display(azs.summary(kf_sv_idata, var_names=_sv_vars, round_to=4))

        _plot_sigma_obs_path(
            _dates_sv, kf_sv_idata.posterior['sigma_obs'], color=C_VOL,
            title=f'Stochastic volatility - mingled cohort observation noise ({label})')

        _trace_vars = [v for v in _sv_vars if v in _SV_TRACE_VARS]
        if _trace_vars:
            _pc_sv = azp.plot_trace(
                kf_sv_idata, var_names=_trace_vars,
                figure_kwargs=_azp_figure_kwargs(
                    _forest_height_px(len(_trace_vars), per_row=210, base=110)),
            )
            with contextlib.suppress(Exception):
                _pc_sv.add_title('SV fit — volatility-walk trace ('
                                 + ', '.join(_trace_vars) + ')')
            _polish_facet_axes(_pc_sv)
            _safe_show(_pc_sv)
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 12b (stochastic volatility) skipped: {e!r}')


# =============================================================================
# 13. Granular earnings-cohort posterior-predictive forest (+ 13.1 further views)
# =============================================================================
def run_granular_forest(idata, results: pd.DataFrame, panel: KalmanPanelInputs,
                        screen: ScreenContext, engine,
                        config: Optional[KalmanRunConfig] = None) -> Optional[dict]:
    """Per-ISIN forest of the fused-panel expected-price posterior for the cohort.

    Keeps the same earnings-cohort definition as §12 but stays per-ISIN granular and
    reuses the fitted fused posterior: the de-standardised ``expected_pt`` draws
    (:attr:`ScreenContext.ept`) per cohort name are plotted as a forest, with the raw
    analyst ``observed_pt`` overlaid as points. Returns context (``ppc_tree``, ``keep``,
    ``cohort_meta`` ...) for §13.1 / §14.
    """
    try:
        keep_cols_tuple = ('isin', 'ticker', 'name', 'description', 'trading_region', 'region', 'country',
                           'trading_country', 'trading_country_name', 'exchange',
                           'unit', 'exchange_name', 'unit_name', 'country_name', 'sector',
                           'industry', 'market_cap','market_cap_country_r', 'enterprise_value', 'last_price', 'price_target', 'next_earnings',
                           'days_to_earnings','fy_end_date', 'last_updated', 'next_fiscal_quarter',
                           'income_statement_report_date', 'next_income_statement_report_date', 'next_fy_end_date')
        _hist_cols, col_sql = fetch_history_columns(engine, keep_cols_tuple)
        # Earnings-window bounds (single source of truth for both the SQL filter and
        # the print label): names whose next_earnings lands within the
        # ±earnings_window_days window around today (KalmanRunConfig).
        cfg = config if config is not None else get_run_config()
        earnings_past_days = int(cfg.earnings_window_days)
        earnings_future_days = int(cfg.earnings_window_days)
        with engine.connect() as conn:
            cohort_meta = pd.read_sql(
                text(f"""
                        SELECT {col_sql}
                        FROM pml.pml_df
                        WHERE income_statement_report_date >= '{cfg.min_report_date}'
                          AND next_earnings >= current_date - INTERVAL ':past_days days'
                          AND next_earnings <= current_date + INTERVAL ':future_days days'
                          AND market_cap_country_r > :min_rank
                    """.replace(':past_days', str(earnings_past_days))
                     .replace(':future_days', str(earnings_future_days))),
                conn, params={'min_rank': cfg.min_mcap_country_rank},
            )
        cohort_isins_all = cohort_meta['isin'].astype(str).unique().tolist()
        print(f'Recent-earnings cohort (next_earnings +{earnings_future_days}/-{earnings_past_days}d): '
              f'{len(cohort_isins_all)} ISINs.')

        ept = screen.ept
        modelled = set(np.asarray(ept.coords['isin'].values).astype(str).tolist())
        cohort_isins = [i for i in cohort_isins_all if i in modelled]
        if not cohort_isins:
            raise RuntimeError(
                'No earnings-window ISIN overlaps the fitted fused panel posterior '
                f'({len(cohort_isins_all)} cohort ISINs, {len(modelled)} modelled).'
            )
        print(f'Cohort ISINs overlapping the fitted posterior: {len(cohort_isins)}.')

        MAX_FOREST = 50
        cohort_results = (results[results['isin'].isin(cohort_isins)]
                          .sort_values('expected_upside', ascending=False))
        if len(cohort_results) > MAX_FOREST:
            half = MAX_FOREST // 2
            keep = pd.concat([cohort_results.head(half), cohort_results.tail(half)])
            print(f'Cohort has {len(cohort_results)} ISINs; showing the {MAX_FOREST} most '
                  f'extreme by expected upside (top/bottom {half}).')
        else:
            keep = cohort_results
        forest_isins = keep['isin'].astype(str).tolist()

        # Fused-panel expected-price posterior draws + observed analyst target per name.
        pp_price = ept.sel(isin=forest_isins).rename('expected_price')
        _obs = (panel.frame.assign(isin=panel.frame['isin'].astype(str))
                .drop_duplicates('isin').set_index('isin')['observed_pt']
                .reindex(forest_isins).astype('float64'))
        obs_price = xr.DataArray(
            _obs.to_numpy(), dims='isin', coords={'isin': forest_isins},
        ).rename('expected_price')
        ppc_tree = xr.DataTree.from_dict({
            'posterior': pp_price.to_dataset(),
            'observed_data': obs_price.to_dataset(),
        })

        # Reference bands from the POSTERIOR HDIs of pooled expected_pt.
        exp_pt_pool = ept.sel(isin=forest_isins).stack(s=('chain', 'draw', 'isin'))
        _q = lambda p: float(exp_pt_pool.quantile(p).values)
        band94 = (_q(_HDI_LO), _q(_HDI_HI))
        band50 = (_q(0.25), _q(0.75))
        cohort_last_price = float(np.nanmedian(
            cohort_meta.loc[cohort_meta['isin'].isin(forest_isins), 'last_price']
        ))

        pc = azp.plot_forest(
            ppc_tree, group='posterior', combined=True,
            labels=['isin'], backend='plotly',
            figure_kwargs=_azp_figure_kwargs(_forest_height_px(len(forest_isins))),
        )
        pc.map(azv.scatter_x, 'observations', data=ppc_tree.observed_data.ds,
               coords={'column': 'forest'}, color=C_OBSERVED)
        pc.map(azv.labelled_x, 'xlabel', coords={'column': 'forest'},
               text='expected price (simulated)  -  points = observed analyst target',
               ignore_aes='y')
        pc.coords = {'column': 'forest'}
        pc = azp.add_bands(pc, values=[band94],
                           visuals={'ref_band': {'color': C_POSTERIOR, 'alpha': 0.12}})
        pc = azp.add_bands(pc, values=[band50],
                           visuals={'ref_band': {'color': C_POSTERIOR, 'alpha': 0.24}})
        pc = azp.add_lines(pc, values=cohort_last_price,
                           visuals={'ref_line': {'color': C_REF,
                                                 'linestyle': 'dash', 'width': 1.3}})
        _safe_show(pc)
        print(f'Cohort expected_pt 94% HDI band: ({band94[0]:.2f}, {band94[1]:.2f});  '
              f'50% HDI band: ({band50[0]:.2f}, {band50[1]:.2f});  '
              f'cohort last_price ref = {cohort_last_price:.2f}.')

        _cols = ['isin', 'ticker', 'name', 'description', 'trading_region', 'region', 'country_name', 'exchange_name',
                 'unit_name',
                 'sector', 'industry',
                 'market_cap', 'enterprise_value', 'last_price', 'observed_pt',
                 'expected_pt', 'expected_pt_hdi_lo', 'expected_pt_hdi_hi',
                 'expected_upside']
        display(keep[[c for c in _cols if c in keep.columns]]
                .round(3).reset_index(drop=True))

        return {'ppc_tree': ppc_tree, 'keep': keep, 'forest_isins': forest_isins,
                'cohort_last_price': cohort_last_price, 'cohort_meta': cohort_meta}
    except Exception as e:  # pragma: no cover - optional / environment-dependent
        print(f'Section 13 (granular earnings-cohort posterior-predictive forest) skipped: {e!r}')
        return None


def run_granular_further_views(prior_idata, panel: KalmanPanelInputs,
                               screen: ScreenContext,
                               forest_ctx: Optional[dict]) -> None:
    """Section 13.1: results-keyed reference-band forest + cohort distribution KDE."""
    if not forest_ctx:
        print('Run the Section 13 forest first (ppc_tree / keep / cohort_last_price '
              'not in scope).')
        return
    ppc_tree = forest_ctx['ppc_tree']
    keep = forest_ctx['keep']
    forest_isins = forest_ctx['forest_isins']
    cohort_last_price = forest_ctx['cohort_last_price']

    # 13a. Forest with reference bands keyed off the stored results dataframe.
    band_lo = float(np.nanmedian(keep['expected_pt_hdi_lo']))
    band_hi = float(np.nanmedian(keep['expected_pt_hdi_hi']))
    band_med = float(np.nanmedian(keep['expected_pt']))

    pc2 = azp.plot_forest(ppc_tree, group='posterior', combined=True,
                          labels=['isin'], backend='plotly',
                          figure_kwargs=_azp_figure_kwargs(
                              _forest_height_px(len(forest_isins))))
    pc2.map(azv.scatter_x, 'observations', data=ppc_tree.observed_data.ds,
            coords={'column': 'forest'}, color=C_OBSERVED)
    pc2.map(azv.labelled_x, 'xlabel', coords={'column': 'forest'},
            text='expected price (simulated)  -  band = cohort-median results HDI '
                 '[expected_pt_hdi_lo, expected_pt_hdi_hi]',
            ignore_aes='y')
    pc2.coords = {'column': 'forest'}
    pc2 = azp.add_bands(pc2, values=[(band_lo, band_hi)],
                        visuals={'ref_band': {'color': '#9b59b6', 'alpha': 0.15}})
    pc2 = azp.add_lines(pc2, values=band_med,
                        visuals={'ref_line': {'color': '#9b59b6', 'width': 1.4}})
    pc2 = azp.add_lines(pc2, values=cohort_last_price,
                        visuals={'ref_line': {'color': C_REF,
                                              'linestyle': 'dash', 'width': 1.3}})
    _safe_show(pc2)
    print(f'results-df cohort-median 94% HDI band: ({band_lo:.2f}, {band_hi:.2f});  '
          f'median expected_pt = {band_med:.2f};  cohort last_price = {cohort_last_price:.2f}.')

    # 13b. Cohort distribution KDE: implied upside vs expected return.
    model_df = panel.frame
    VAR = 'cohort_expected_upside_pct'
    _fi = [str(s) for s in forest_isins]

    def _cohort_mean_pct(da):
        return (da.sel(isin=forest_isins).mean('isin') * 100.0).rename(VAR)

    def _consensus_upside_pct():
        m = model_df.set_index(model_df['isin'].astype(str))
        if 'feat_implied_upside' in m.columns:
            s, src = m['feat_implied_upside'].astype('float64'), 'feat_implied_upside (SSOT)'
        else:
            s = m['observed_pt'].astype('float64') / m['last_price'].astype('float64') - 1.0
            src = 'observed_pt/last_price - 1 (fallback)'
        return (s.reindex(_fi).dropna() * 100.0), src

    eu_prior, _ = panel_posterior_upside(prior_idata, panel, source='prior')
    cohort_upside = _cohort_mean_pct(screen.eu)
    prior_cohort = _cohort_mean_pct(eu_prior)
    _cons, _cons_src = _consensus_upside_pct()
    cons_da = xr.DataArray(_cons.to_numpy(), dims='isin',
                           coords={'isin': _cons.index.to_numpy()}).rename(VAR)

    # Styles carry Plotly line vocabulary (``width`` / ``linestyle`` dash names) so the
    # same dict feeds both the ``plot_dist`` ``dist`` visual and the proxy legend traces.
    # Colours come straight from the module palette — the local ``_C_*`` aliases this
    # block used to declare shadowed the SSOT names and hid which role each series played.
    series = [
        (cohort_upside, ['chain', 'draw'], dict(color=C_POSTERIOR, width=2.2),
         'posterior E[upside] (cohort mean)'),
        (prior_cohort, ['chain', 'draw'], dict(color=C_VOL, width=2.0, linestyle='dash'),
         'prior E[upside] (cohort mean)'),
    ]
    if len(_cons) >= 2:
        series.append((cons_da, ['isin'], dict(color=C_ACCENT, width=2.2),
                       'consensus implied upside (across names)'))

    pc3 = None
    for da, sample_dims, style, _ in series:
        pc3 = azp.plot_dist(
            da.to_dataset(), kind='kde', var_names=[VAR], sample_dims=sample_dims,
            backend='plotly', plot_collection=pc3, visuals={'dist': style},
            **({'figure_kwargs': _azp_figure_kwargs(480)} if pc3 is None else {}),
        )
    if pc3 is None:
        raise RuntimeError('No KDE series to plot (expected posterior + prior).')

    ax = pc3.get_target(VAR, {})  # PlotlyPlot (figure + row/col)
    fig = ax.figure
    cons_mean = float(_cons.mean()) if len(_cons) else float('nan')
    _add_ref_line(ax, x=0.0, kind='zero')
    if np.isfinite(cons_mean):
        _add_ref_line(ax, x=cons_mean, kind='emphasis', color=C_ACCENT)

    _all = np.concatenate([cohort_upside.values.ravel(), prior_cohort.values.ravel(),
                           _cons.to_numpy()])
    _lo, _hi = np.nanpercentile(_all, [1, 99])
    _pad = 0.05 * (_hi - _lo)
    ax.update_xaxes(range=[_lo - _pad, _hi + _pad], title_text='upside vs last_price (%)')
    ax.update_yaxes(title_text='density')
    pc3.add_title('Cohort upside (%): consensus implied vs expected prior/posterior '
                  '(earnings window +/-10d)')

    # Proxy legend traces (Plotly analogue of the former Line2D handle list): the KDE
    # ``dist`` traces are registered with ``showlegend=False``, so re-express each
    # series style + the two reference lines as invisible named traces.
    if go is not None:
        _r, _c = ax.row, ax.col
        for _, _, style, label in series:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color=style['color'], width=style.get('width', 2.0),
                          dash=style.get('linestyle', 'solid')),
                name=label), row=_r, col=_c)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=_REF_LINE_KINDS['emphasis'] | {'color': C_ACCENT},
            name=f'consensus cohort mean ({cons_mean:.1f}%)'), row=_r, col=_c)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines', line=dict(**_REF_LINE_KINDS['zero']),
            name='0% break-even'), row=_r, col=_c)
        fig.update_layout(showlegend=True)
    _safe_show(pc3)

    _p_pos = float((cohort_upside > 0).mean().values) * 100.0
    _p_vs_cons = (float((cohort_upside > cons_mean).mean().values) * 100.0
                  if np.isfinite(cons_mean) else float('nan'))
    print(f'Consensus implied upside [{_cons_src}]: cohort mean = {cons_mean:.2f}% '
          f'across {len(_cons)} names.')
    print(f'Expected upside (cohort mean): prior = {float(prior_cohort.mean()):.2f}%, '
          f'posterior = {float(cohort_upside.mean()):.2f}%;  '
          f'P(posterior cohort upside > 0) = {_p_pos:.1f}%;  '
          f'P(posterior cohort upside > consensus mean) = {_p_vs_cons:.1f}%.')


# =============================================================================
# 14. Comprehensive summary + actionable recommendations
# =============================================================================
def _fmt_or_na(x, nd: int = 1, suf: str = '') -> str:
    """Nan-safe fixed-point formatter shared by §14/§14b (``'n/a'`` fallback)."""
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return 'n/a'
        return f'{x:.{nd}f}{suf}'
    except Exception:
        return 'n/a'


def _display_label(row) -> str:
    """Company display name with ISIN fallback, shared by §14/§14b."""
    t = row.get('name')
    return t if isinstance(t, str) and t.strip() else str(row['isin'])


def _resolve_risk_book(risk_book, idata, panel, screen,
                       results: pd.DataFrame) -> RiskBook:
    """Reuse the shared §10b :class:`RiskBook` or recompute it from the screen.

    Guard shared by :func:`export_analytics` and :func:`run_recommendations` so
    the exported and displayed sizing always come from a single computation.
    """
    if risk_book is not None:
        return risk_book
    return compute_cvar_aware_book(idata, panel, screen, results)


_VERDICT_COLORS = {'OVERWEIGHT': C_ACCENT, 'UNDERWEIGHT': C_HIGHLIGHT,
                   'NEUTRAL': C_MUTED}


def plot_group_signal_forest(payload: dict[str, dict[str, Any]]) -> None:
    """Stacked shrunk-excess forest over the model's group-effect coords (§14b.4).

    Visual twin of the ``run_recommendations`` block-4 prints: one subplot per
    coord, x = the hierarchically **shrunk excess return vs the universe
    posture** (pp), point + 94% CI (the raw group-mean CI mapped into excess
    space via the group's shrinkage factor λ), the per-coord ±1-sd OW/UW band
    shaded, marker colour by verdict. Hover carries the raw upside, CI,
    conditional P(>0|K) and group size.
    """
    coords = [c for c in payload if payload[c].get('rows')]
    if not coords or not _HAS_PLOTLY:
        return
    row_counts = [len(payload[c]['rows']) for c in coords]
    total = sum(row_counts)
    fig = make_subplots(
        rows=len(coords), cols=1, shared_xaxes=True,
        vertical_spacing=min(0.08, 24.0 / max(total * 24, 1)),
        row_heights=[n / total for n in row_counts],
        subplot_titles=[f'{c}  (±{payload[c]["band"]:.2f}pp OW/UW band)'
                        for c in coords])
    for i, col in enumerate(coords, start=1):
        rows = payload[col]['rows']
        verdicts = payload[col].get('verdicts') or ['NEUTRAL'] * len(rows)
        band = float(payload[col]['band'])
        # rows are sorted by raw upside desc; plot bottom-up so best sits on top.
        names = [r[0] for r in rows][::-1]
        ex = np.array([r[4] for r in rows])[::-1]
        m = np.array([r[1] for r in rows])[::-1]
        lo = np.array([r[2] for r in rows])[::-1]
        hi = np.array([r[3] for r in rows])[::-1]
        pc = np.array([r[5] for r in rows])[::-1]
        n = np.array([r[6] for r in rows])[::-1]
        lam = np.array([r[7] for r in rows])[::-1]
        vcol = [_VERDICT_COLORS.get(v, C_MUTED) for v in verdicts[::-1]]
        if np.isfinite(band) and band > 0:
            _add_ref_band(fig, x0=-band, x1=band, row=i, col=1)
        _add_ref_line(fig, x=0, kind='zero', row=i, col=1)
        fig.add_trace(go.Scatter(
            x=ex, y=names, mode='markers',
            marker=dict(color=vcol, size=9),
            error_x=dict(type='data', symmetric=False,
                         array=np.clip(lam * (hi - m), 0, None),
                         arrayminus=np.clip(lam * (m - lo), 0, None),
                         color=_hex_to_rgba(C_POSTERIOR, 0.5), thickness=1.4),
            customdata=np.c_[m, lo, hi, pc, n],
            hovertemplate=('%{y}<br>shrunk excess = %{x:.2f}pp<br>'
                           'upside = %{customdata[0]:.1f}% '
                           '[%{customdata[1]:.1f}%, %{customdata[2]:.1f}%]<br>'
                           'P(>0|K) = %{customdata[3]:.0%}  '
                           'n = %{customdata[4]:.0f}<extra></extra>'),
            name=col, showlegend=False), row=i, col=1)
    fig.update_xaxes(title_text='shrunk excess return vs universe (pp)',
                     row=len(coords), col=1)
    # Long group labels: automargin overrides _render_plotly's fixed l=60 margin.
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        title=('Group allocation signals — shrunk excess return forest '
               '(green = OVERWEIGHT, orange = UNDERWEIGHT)'),
        showlegend=False)
    _render_plotly(fig, height=int(np.clip(26 * total + 90 * len(coords) + 120,
                                           420, 1800)))


def plot_book_composition(rb: RiskBook) -> None:
    """CVaR-book composition — weights vs per-name tail risk (§14b.10 visual).

    Left: the STARR-ranked book weights (with the per-name cap line). Right:
    each held name's 5% expected shortfall (CVaR) — the risk budget the weight
    is spending. The title annotation carries the portfolio-level aggregates
    (``port_up`` / ``port_cvar`` / ``starr_book`` / ``div`` from
    :attr:`RiskBook.summary`) — numbers that previously appeared only in the
    §14b.10 prints.
    """
    if not _HAS_PLOTLY or rb is None or rb.book is None or len(rb.book) == 0:
        return
    book = rb.book.sort_values('weight', ascending=True)
    _tk = book.get('ticker', pd.Series(index=book.index, dtype=object))
    labels = [t if isinstance(t, str) and t.strip() else str(i)[:8]
              for t, i in zip(_tk, book['isin'])]
    # Disambiguate collisions (e.g. tickers truncated/missing to the same
    # 8-char isin prefix) so distinct names never share one bar category --
    # Plotly silently sums same-category bar values into a single bar
    # otherwise, hiding the rest of the book behind one inflated total.
    _dupe_n: dict[str, int] = {}
    for _i, _lab in enumerate(labels):
        _dupe_n[_lab] = _dupe_n.get(_lab, 0) + 1
        if _dupe_n[_lab] > 1:
            labels[_i] = f'{_lab} ({_dupe_n[_lab]})'
    w_pct = (pd.to_numeric(book['weight'], errors='coerce')
             .replace([np.inf, -np.inf], np.nan) * 100.0)
    cvar_pct = (pd.to_numeric(book.get('cvar05'), errors='coerce')
                .replace([np.inf, -np.inf], np.nan) * 100.0)
    eu_pct = (pd.to_numeric(book.get('expected_upside'), errors='coerce')
              .replace([np.inf, -np.inf], np.nan) * 100.0)
    s = rb.summary or {}
    cap = float(s.get('cap', float('nan')))

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=('book weight', '5% expected shortfall'))
    fig.add_trace(go.Bar(
        x=w_pct, y=labels, orientation='h',
        marker=dict(color=_hex_to_rgba(C_POSTERIOR, 0.85)),
        customdata=np.c_[eu_pct],
        hovertemplate=('%{y}<br>weight = %{x:.1f}%<br>'
                       'E[upside] = %{customdata[0]:.1f}%<extra></extra>'),
        name='weight'), row=1, col=1)
    if np.isfinite(cap):
        _add_ref_line(fig, x=cap * 100.0, kind='anchor',
                      annotation_text=f'cap {cap:.0%}', row=1, col=1)
    fig.add_trace(go.Bar(
        x=cvar_pct, y=labels, orientation='h',
        marker=dict(color=_hex_to_rgba(C_HIGHLIGHT, 0.85)),
        hovertemplate='%{y}<br>CVaR5 = %{x:.1f}%<extra></extra>',
        name='CVaR5'), row=1, col=2)
    fig.update_xaxes(title_text='weight (%)', ticksuffix='%', row=1, col=1)
    fig.update_xaxes(title_text='CVaR5 (%)', ticksuffix='%', row=1, col=2)
    # Force an explicit categorical y-axis (with the STARR-sorted label order
    # pinned via categoryarray) rather than relying on Plotly's implicit
    # type inference -- that inference can flip the shared axis to a
    # numeric/linear type for certain label sets, which collapses every bar
    # into an invisible sliver against an autoranged 0..N numeric axis
    # instead of the intended per-name category rows.
    fig.update_yaxes(type='category', categoryorder='array', categoryarray=labels,
                     automargin=True)

    _fmt = _fmt_or_na
    fig.update_layout(
        title=('CVaR-aware book composition — portfolio: '
               f'E[upside]={_fmt(s.get("port_up", float("nan")) * 100.0, 1, "%")}  '
               f'CVaR5={_fmt(s.get("port_cvar", float("nan")) * 100.0, 1, "%")}  '
               f'reward/CVaR={_fmt(s.get("starr_book", float("nan")), 2)}  '
               f'diversification={_fmt(s.get("div", float("nan")), 2)}'),
        showlegend=False)
    _render_plotly(fig, height=int(np.clip(22 * len(book) + 180, 360, 1400)))


def plot_screen_overview(results_df: pd.DataFrame, *, top_n: int = 15) -> None:
    """Screen distribution + top-N ranked recommendations (promoted from the
    former notebook §14.1 inline cell — SSOT now lives here).

    Left: the cross-sectional expected-upside distribution with median and
    break-even reference lines. Right: the top-``top_n`` names ranked by
    expected upside with their 94% HDI (derived from the ``expected_pt`` HDI
    over ``last_price``).
    """
    if not _HAS_PLOTLY or results_df is None or len(results_df) == 0:
        return
    eu_pct = pd.to_numeric(results_df['expected_upside'], errors='coerce') * 100.0
    eu_pct = eu_pct[np.isfinite(eu_pct)]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=('Cross-sectional expected-upside '
                                        'distribution',
                                        f'Top {top_n} by expected upside '
                                        '(94% HDI)'))
    fig.add_trace(go.Histogram(
        x=eu_pct.clip(*np.nanpercentile(eu_pct, [0.5, 99.5])), nbinsx=60,
        marker=dict(color=_hex_to_rgba(C_POSTERIOR, 0.8)),
        hovertemplate='upside = %{x:.0f}%<extra></extra>',
        name='expected upside', showlegend=False), row=1, col=1)
    _add_ref_line(fig, x=float(eu_pct.median()), kind='emphasis',
                  annotation_text=f'median {eu_pct.median():.1f}%', row=1, col=1)
    _add_ref_line(fig, x=0, kind='zero', row=1, col=1)
    fig.update_xaxes(title_text='expected upside (%)', ticksuffix='%', row=1, col=1)
    fig.update_yaxes(title_text='ISIN count', row=1, col=1)

    top = (results_df.dropna(subset=['expected_upside'])
           .sort_values('expected_upside', ascending=False).head(top_n)
           .iloc[::-1])
    _tk = top.get('ticker', pd.Series(index=top.index, dtype=object))
    lbl = [t if isinstance(t, str) and t.strip() else str(i)[:6]
           for t, i in zip(_tk, top['isin'])]
    mid = pd.to_numeric(top['expected_upside'], errors='coerce') * 100.0
    _lp = pd.to_numeric(top['last_price'], errors='coerce')
    lo = (pd.to_numeric(top['expected_pt_hdi_lo'], errors='coerce') / _lp - 1.0) * 100.0
    hi = (pd.to_numeric(top['expected_pt_hdi_hi'], errors='coerce') / _lp - 1.0) * 100.0
    fig.add_trace(go.Scatter(
        x=mid, y=lbl, mode='markers',
        marker=dict(color=C_ACCENT, size=8),
        error_x=dict(type='data', symmetric=False,
                     array=np.clip(hi - mid, 0, None),
                     arrayminus=np.clip(mid - lo, 0, None),
                     color=_hex_to_rgba(C_ACCENT, 0.55), thickness=1.6),
        customdata=np.c_[lo, hi],
        hovertemplate=('%{y}<br>upside = %{x:.1f}%<br>94% HDI = '
                       '[%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]'
                       '<extra></extra>'),
        name='top names', showlegend=False), row=1, col=2)
    _add_ref_line(fig, x=0, kind='zero', row=1, col=2)
    fig.update_xaxes(title_text='expected upside (%)', ticksuffix='%', row=1, col=2)
    fig.update_layout(title='Screen overview — distribution and ranked '
                            'recommendations', showlegend=False)
    _render_plotly(fig, height=H_FORECAST)


def plot_risk_return_scatter(results_df: pd.DataFrame, *,
                             max_points: int = 7000) -> None:
    """Interactive expected-upside vs posterior-uncertainty screen (promoted
    from the former notebook §14.2 inline cell — SSOT now lives here).

    Uncertainty = the 94% HDI band width of ``expected_pt`` as a fraction of
    ``last_price``; colour = sector, size = market cap.
    """
    if not _HAS_PLOTLY or results_df is None or len(results_df) == 0:
        return
    need = ['expected_upside', 'expected_pt_hdi_lo', 'expected_pt_hdi_hi',
            'last_price', 'market_cap']
    df = results_df.dropna(subset=[c for c in need if c in results_df.columns]).copy()
    if df.empty:
        return
    df['uncertainty_pct'] = ((df['expected_pt_hdi_hi'] - df['expected_pt_hdi_lo'])
                             / df['last_price']) * 100.0
    df['expected_upside_pct'] = df['expected_upside'] * 100.0
    _tk = df.get('ticker', pd.Series(index=df.index, dtype=object))
    df['label'] = [t if isinstance(t, str) and t.strip() else str(i)[:6]
                   for t, i in zip(_tk, df['isin'])]
    if len(df) > max_points:
        df = df.nlargest(max_points, 'expected_upside')
    fig = px.scatter(
        df, x='uncertainty_pct', y='expected_upside_pct',
        color=df['sector'].fillna('Unknown').astype(str), size='market_cap',
        size_max=22, hover_name='label',
        hover_data={'isin': True, 'name': True, 'industry': True,
                    'expected_upside_pct': ':.1f', 'uncertainty_pct': ':.1f',
                    'market_cap': ':.0f', 'label': False},
        labels={'uncertainty_pct': 'posterior uncertainty — 94% HDI width / '
                                   'last price (%)',
                'expected_upside_pct': 'expected upside (%)',
                'color': 'sector'},
        title='Expected upside vs posterior uncertainty (94% HDI width)')
    _add_ref_line(fig, y=0, kind='zero')
    fig.update_xaxes(ticksuffix='%')
    fig.update_yaxes(ticksuffix='%')
    fig.update_layout(legend_title_text='sector',
                      legend=dict(font_size=_LEGEND_FONT_SIZE))
    _render_plotly(fig, height=650)


def plot_top_candidate_forest(screen_ctx: ScreenContext,
                              results_df: pd.DataFrame, *,
                              top_n: int = 20) -> None:
    """Posterior expected-upside forest of the top screen candidates (promoted
    from the former notebook §14.3 inline cell — SSOT now lives here)."""
    top = (results_df.dropna(subset=['expected_upside'])
           .sort_values('expected_upside', ascending=False).head(top_n))
    eu_isins = set(np.asarray(screen_ctx.eu['isin'].values).astype(str))
    top = top[top['isin'].astype(str).isin(eu_isins)]
    top_isins = top['isin'].astype(str).tolist()
    if not top_isins:
        print('No top candidates overlap the posterior draws — nothing to plot.')
        return
    eu = screen_ctx.eu.sel(isin=top_isins) * 100.0
    _tk = top.get('ticker', pd.Series(index=top.index, dtype=object))
    labels = [t if isinstance(t, str) and t.strip() else str(i)[:6]
              for t, i in zip(_tk, top['isin'])]
    eu = eu.assign_coords(isin=labels)
    _ds = xr.Dataset({'expected_upside_pct': eu})
    pc = azp.plot_forest(_ds, var_names=['expected_upside_pct'], combined=True,
                         backend='plotly',
                         figure_kwargs=_azp_figure_kwargs(
                             _forest_height_px(len(labels), per_row=30)))
    pc.add_title(f'Top {len(labels)} candidates — posterior expected upside '
                 '(%) with 94% HDI')
    with contextlib.suppress(Exception):
        _fx = _plotly_figure_of(pc)
        if _fx is not None:
            _add_ref_line(_fx, x=0, kind='zero')
            _fx.update_xaxes(title_text='expected upside (%)', ticksuffix='%')
    _safe_show(pc)


def run_summary(results: pd.DataFrame, screen: ScreenContext,
                cohort_ctx: Optional[dict], mingled_ctx: Optional[dict]) -> None:
    """Consolidate the screen into a decision-oriented earnings-cohort vs baseline read."""
    cohort_meta = cohort_ctx.get('cohort_meta') if cohort_ctx else None
    comparison = mingled_ctx.get('comparison') if mingled_ctx else None

    _fmt = _fmt_or_na
    _label = _display_label

    def _band_width_pct(df):
        denom = df['expected_pt'].replace(0, np.nan)
        return (df['expected_pt_hdi_hi'] - df['expected_pt_hdi_lo']) / denom * 100.0

    def _shrink_pct(df):
        denom = df['observed_pt'].replace(0, np.nan)
        return (df['expected_pt'] / denom - 1.0) * 100.0

    universe = results.copy()
    # Display boundary: the screen frame stores decimal upside — build a local
    # percent column for the summary tables/prints (cohort/rest slice universe,
    # so they inherit it).
    universe['expected_upside_pct'] = (
        pd.to_numeric(universe['expected_upside'], errors='coerce') * 100.0)

    # ---- A. Cross-sectional: earnings cohort vs the rest of the universe ----
    _cohort_ids = set(cohort_meta['isin'].astype(str)) if cohort_meta is not None else None
    if _cohort_ids:
        _in = universe['isin'].astype(str).isin(_cohort_ids)
        cohort, rest = universe[_in].copy(), universe[~_in].copy()
        groups = [('Earnings cohort (+/-10d)', cohort),
                  ('Historical baseline (not reporting)', rest),
                  ('Full universe', universe)]
    else:
        cohort = rest = None
        groups = [('Full universe', universe)]
        print('Section 13 cohort_meta not available - showing the universe only.')

    _rows = []
    for label, df in groups:
        if df is None or len(df) == 0:
            continue
        _rows.append({
            'group': label,
            'n_names': int(len(df)),
            'median_upside_%': df['expected_upside_pct'].median(),
            'mean_upside_%': df['expected_upside_pct'].mean(),
            'positive_upside_%': (df['expected_upside_pct'] > 0).mean() * 100.0,
            'median_band_width_%': _band_width_pct(df).median(),
            'median_shrink_vs_consensus_%': _shrink_pct(df).median(),
            'median_n_analysts': (df['n_analysts'].median() if 'n_analysts' in df else np.nan),
        })
    summary_tbl = pd.DataFrame(_rows).set_index('group').round(2)
    print('Cross-sectional summary - expected price targets by group:')
    display(summary_tbl)

    # A-viz. Cohort vs baseline vs universe on the three decision metrics —
    # grouped bars (visual twin of the table above; best-effort, display only).
    try:
        if _HAS_PLOTLY and len(summary_tbl):
            _metrics = [('median_upside_%', 'median expected upside'),
                        ('positive_upside_%', 'share with positive upside'),
                        ('median_band_width_%', 'median 94% band width')]
            _metrics = [(c, lab) for c, lab in _metrics if c in summary_tbl.columns]
            figs = make_subplots(rows=1, cols=len(_metrics), shared_yaxes=True,
                                 horizontal_spacing=0.05,
                                 subplot_titles=[lab for _, lab in _metrics])
            _grp_colors = [C_HIGHLIGHT, C_MUTED, C_POSTERIOR]
            for j, (c, lab) in enumerate(_metrics, start=1):
                figs.add_trace(go.Bar(
                    x=pd.to_numeric(summary_tbl[c], errors='coerce'),
                    y=summary_tbl.index.astype(str), orientation='h',
                    marker=dict(color=_grp_colors[:len(summary_tbl)]),
                    hovertemplate=f'%{{y}}<br>{lab} = %{{x:.1f}}%<extra></extra>',
                    name=lab, showlegend=False), row=1, col=j)
                _add_ref_line(figs, x=0, kind='zero', row=1, col=j)
                figs.update_xaxes(ticksuffix='%', row=1, col=j)
            figs.update_layout(title='Earnings cohort vs baseline vs universe — '
                                     'decision metrics', showlegend=False)
            _render_plotly(figs, height=320)
    except Exception as exc:  # pragma: no cover - display-only
        print(f'summary decision panel skipped: {exc!r}')

    if cohort is not None and len(cohort) and 'sector' in cohort.columns:
        sector_mix = (cohort.assign(sector=cohort['sector'].fillna('Unknown'))
                      .groupby('sector')
                      .agg(n=('isin', 'size'),
                           median_upside_pct=('expected_upside_pct', 'median'))
                      .sort_values('n', ascending=False).round(2))
        print('\nEarnings-cohort sector tilt:')
        display(sector_mix.head(10))

        # A2-viz. Cohort sector tilt vs the universe sector mix — diverging bar
        # of the share-of-names difference (positive = cohort over-represents).
        try:
            if _HAS_PLOTLY and universe is not None and 'sector' in universe.columns:
                _u_mix = (universe.assign(sector=universe['sector'].fillna('Unknown'))
                          .groupby('sector')['isin'].size())
                _c_share = sector_mix['n'] / max(float(sector_mix['n'].sum()), 1.0)
                _u_share = (_u_mix / max(float(_u_mix.sum()), 1.0)).reindex(
                    _c_share.index).fillna(0.0)
                _tilt = ((_c_share - _u_share) * 100.0).sort_values()
                figt = go.Figure(go.Bar(
                    x=_tilt.to_numpy(), y=_tilt.index.astype(str), orientation='h',
                    marker=dict(color=[C_ACCENT if v >= 0 else C_HIGHLIGHT
                                       for v in _tilt.to_numpy()]),
                    hovertemplate=('%{y}<br>tilt = %{x:+.1f}pp of names'
                                   '<extra></extra>'),
                    name='sector tilt'))
                _add_ref_line(figt, x=0, kind='zero')
                figt.update_xaxes(title_text='cohort share − universe share '
                                             '(pp of names)')
                figt.update_layout(title='Earnings-cohort sector tilt vs universe',
                                   showlegend=False)
                _render_plotly(figt, height=int(np.clip(24 * len(_tilt) + 160,
                                                        300, 900)))
        except Exception as exc:  # pragma: no cover - display-only
            print(f'sector-tilt chart skipped: {exc!r}')

    # ---- B. Time-series: recent vs historical mingled cohort trail ----
    hist_drift = implied_now = first = last = None
    if comparison is not None and len(comparison) >= 2:
        first, last = comparison.iloc[0], comparison.iloc[-1]
        hist_drift = ((last['observed_pt'] / first['observed_pt'] - 1.0) * 100.0
                      if first['observed_pt'] else np.nan)
        implied_now = ((last['expected_pt'] / last['last_price'] - 1.0) * 100.0
                       if last['last_price'] else np.nan)

    # ---- C. Headline narrative ----
    print('\n' + '=' * 74)
    print('KEY INSIGHTS - recent earnings period vs historical data')
    print('=' * 74)
    if cohort is not None and len(cohort):
        cu = cohort['expected_upside_pct'].median()
        ru = rest['expected_upside_pct'].median() if rest is not None and len(rest) else np.nan
        print(f'- {len(cohort)} names report within +/-10d. Median expected upside '
              f'{_fmt(cu, 1, "%")} vs {_fmt(ru, 1, "%")} for non-reporting names '
              f'(delta {_fmt(cu - ru, 1, " pp")}).')
        print(f'- {_fmt((cohort["expected_upside_pct"] > 0).mean() * 100, 0, "%")} of the cohort '
              f'has positive expected upside; median credible band width '
              f'{_fmt(_band_width_pct(cohort).median(), 1, "%")} '
              f'(universe {_fmt(_band_width_pct(universe).median(), 1, "%")}).')
        sh = _shrink_pct(cohort).median()
        print(f'- Kalman-smoothed targets sit {_fmt(abs(sh), 1, "%")} '
              f'{"above" if sh >= 0 else "below"} raw consensus (median) - shrinkage toward '
              f'the hierarchical group mean.')
        _top = cohort.sort_values('expected_upside_pct', ascending=False).head(3)
        _bot = cohort.sort_values('expected_upside_pct').head(3)
        _names = lambda d: ', '.join(f'{_label(r)} ({_fmt(r["expected_upside_pct"], 0, "%")})'
                                     for _, r in d.iterrows())
        print(f'- Highest expected upside: {_names(_top)}.')
        print(f'- Lowest / most downside : {_names(_bot)}.')
    else:
        print('- Earnings cohort not available (Section 13 was skipped).')
    if first is not None and last is not None:
        print(f'- Historical target trail ({first["asof_date"]} -> {last["asof_date"]}): mingled '
              f'cohort consensus {"rose" if (hist_drift or 0) >= 0 else "fell"} '
              f'{_fmt(abs(hist_drift), 1, "%")}; latest Kalman-smoothed target implies '
              f'{_fmt(implied_now, 1, "%")} upside vs cohort last price.')
    else:
        print('- Historical mingled trail not available (Section 12 was skipped).')

    # ---- D. Distributional view: cohort vs universe expected upside ----
    try:
        if _cohort_ids:
            _eu = screen.eu * 100.0
            _modelled = set(_eu.coords['isin'].values.astype(str).tolist())
            _cohort_post = [i for i in _cohort_ids if i in _modelled]
            if _cohort_post:
                _cohort_avg = _eu.sel(isin=_cohort_post).mean('isin')
                _univ_avg = _eu.mean('isin')
                _stacked = xr.concat(
                    [_cohort_avg, _univ_avg],
                    dim=pd.Index(['earnings_cohort', 'universe'], name='group'),
                ).rename('avg_expected_upside_pct')
                pc_sum = azp.plot_dist(
                    _stacked.to_dataset(), kind='kde',
                    var_names=['avg_expected_upside_pct'],
                    sample_dims=['chain', 'draw'], backend='plotly',
                    figure_kwargs=_azp_figure_kwargs(420, width_frac=0.75),
                )
                pc_sum.add_title('Expected upside (%): earnings cohort vs universe '
                                 '(posterior cross-sectional average)')
                pc_sum = azp.add_lines(
                    pc_sum, values=0.0,
                    visuals={'ref_line': {'color': C_REF,
                                          'linestyle': 'dash', 'width': 1.3}},
                )
                _safe_show(pc_sum)
    except Exception as _e:  # pragma: no cover - plot is best-effort
        print(f'Summary KDE overlay skipped: {_e!r}')


def run_recommendations(idata, panel: KalmanPanelInputs, results: pd.DataFrame,
                        screen: ScreenContext, cohort_ctx: Optional[dict],
                        risk_book: Optional[RiskBook] = None) -> None:
    """Section 14b: distil the fused-panel screen into actionable, risk-aware signals.

    Read-only over ``idata``, the panel frame, the de-standardised ``screen.eu``
    expected-upside draws and the §10 ``results`` table. The §8–§10 risk analytics and
    sized book come from the shared :func:`compute_cvar_aware_book` (reused via
    ``risk_book`` when supplied, else recomputed) so they match the §10c export exactly.

    The screen layers three views on top of the raw OVERWEIGHT/NEUTRAL/UNDERWEIGHT
    posture:

    * **§1–§7 — directional signals.** Group- and name-level upside verdicts plus a
      band-width / coverage caution list. All positive-upside probabilities are
      *conditional* on the smoother's state confidence — ``mc_prob_pos`` (name level)
      or the posterior group ``P(upside>0)`` multiplied by the ``kalman_gain``
      (posterior-mean ``achieve_prob``, inverse to ``kalman_variance``) — and the
      ``P_HI`` / ``P_LO`` gates are rescaled by the universe-mean gain to match.
      Group verdicts rank on the hierarchically *shrunk excess return* vs the
      universe posture: ``excess_g = E[group upside] - E[universe upside]`` shrunk by
      ``lambda_g = tau^2 / (tau^2 + s_g^2)`` (between-group signal variance vs the
      group's posterior noise), with a per-coord OW/UW band of ±1 cross-group sd of
      the shrunk excess replacing the former static ±2pp gates. Groups backed by
      fewer than ``MIN_GROUP_N`` (15) names are excluded before the shrinkage
      stats, so thin coord groups neither receive verdicts nor distort
      ``tau^2`` / the OW/UW band.
    * **§8 — risk-adjusted return.** Reward per unit of *expected volatility* (the
      per-name dispersion of the posterior upside draws — a genuinely forward-looking
      vol proxy), i.e. a Sharpe-like ranking that demotes high-upside / high-vol names.
    * **§9 — CVaR tail analytics.** Per-name expected shortfall (CVaR, mean of the worst
      ``ALPHA``-tail) of the posterior upside draws, cross-checked against the
      Student-t Monte-Carlo 5% return. With a low estimated ``nu`` the analyst-target
      outliers are material, so these tails drive sizing rather than the means.
    * **§10 — CVaR-aware sizing.** A long book sized on a reward-to-CVaR (STARR) ratio
      with a per-name cap and a joint-draw portfolio expected shortfall, so no single
      fat-tailed name dominates the book's tail risk.
    """
    model_df = panel.frame
    cohort_meta = cohort_ctx.get('cohort_meta') if cohort_ctx else None

    post = idata.posterior
    eu = screen.eu * 100.0
    isin_dim = eu.coords['isin']
    univ_mean = float(eu.mean(('chain', 'draw', 'isin')))

    # Kalman-gain conditioning: ``achieve_prob`` (the exported ``kalman_gain``) is
    # the smoother's state-confidence analogue, inverse to ``kalman_variance``.
    # Group P(upside>0) is conditioned on the group-mean gain, and the P_HI / P_LO
    # gates are rescaled by the universe-mean gain so verdicts compare
    # like-for-like on the conditional scale.
    if 'achieve_prob' in post:
        gain_da = post['achieve_prob'].mean(('chain', 'draw'))
    else:  # pragma: no cover - defensive
        logger.warning('achieve_prob (kalman_gain) missing from posterior — group '
                       'verdicts fall back to unconditional P(upside>0).')
        gain_da = xr.DataArray(np.ones(eu.sizes['isin']), dims='isin',
                               coords={'isin': isin_dim})
    univ_gain = float(gain_da.mean())
    if not np.isfinite(univ_gain) or univ_gain <= 0:
        univ_gain = 1.0
    P_HI_BASE, P_LO_BASE = 0.67, 0.33
    P_HI, P_LO = P_HI_BASE * univ_gain, P_LO_BASE * univ_gain

    # Minimum-coverage gate for the §4 group allocation signals: coord groups
    # backed by fewer than MIN_GROUP_N names carry too little cross-sectional
    # evidence for a posture verdict (a 1-3 name "exchange" is a stock pick,
    # not an allocation signal). Such groups are dropped BEFORE the shrinkage
    # stats so they neither receive verdicts nor distort tau^2 / the OW/UW band.
    MIN_GROUP_N = 15

    # Shared CVaR-aware analytics + sized book (SSOT with the §10c export).
    rb = _resolve_risk_book(risk_book, idata, panel, screen, results)

    def _verdict(excess_shrunk, p_cond, ow_thr, uw_thr):
        """Posture verdict from the shrunk excess return and conditional P(>0)."""
        if excess_shrunk >= ow_thr and p_cond >= P_HI:
            return 'OVERWEIGHT'
        if excess_shrunk <= uw_thr or p_cond <= P_LO:
            return 'UNDERWEIGHT'
        return 'NEUTRAL'

    def _na(x, nd=0, suf=''):
        return _fmt_or_na(x, nd, suf)

    print('=' * 88)
    print('KALMAN PRICE-TARGET SCREEN - ACTIONABLE INVESTMENT RECOMMENDATIONS')
    print('=' * 88)

    # 1. Posterior reliability — same DataTree unwrap + nan-aware reduction as
    # run_diagnostics, so both report the identical worst-case R-hat.
    max_rhat = _max_posterior_rhat(_posterior_dataset(idata),
                                   [*FUSED_SCALAR_VARS, 'beta'])
    try:
        n_div = int(idata.sample_stats['diverging'].sum())
    except Exception:
        n_div = -1
    print(f'\n1. POSTERIOR RELIABILITY    max R-hat={max_rhat:.4f}    '
          f'divergences={n_div if n_div >= 0 else "n/a"}')
    if np.isfinite(max_rhat) and max_rhat <= 1.01 and n_div == 0:
        print('   [OK]  High-quality posterior - signals are safe to act on.')
    elif np.isfinite(max_rhat) and max_rhat <= 1.05:
        print('   [~]   Acceptable convergence - size positions conservatively.')
    else:
        print('   [!!]  Convergence concerns - treat signals as indicative only.')

    # 2. Tail risk (Student-t df).
    if 'nu' in post:
        nu_mean = float(post['nu'].mean())
        tail = ('heavy - analyst-target outliers are material; prefer CVaR-aware sizing'
                if nu_mean <= 7 else
                'moderate - Gaussian risk approximation broadly acceptable')
        print(f'\n2. TAIL RISK    Student-t df (nu)={nu_mean:.1f}  ->  {tail}.')

    # 3. Universe posture.
    print(f'\n3. UNIVERSE POSTURE    posterior-mean upside={univ_mean:.2f}%    '
          f'P(upside>0)={float((eu > 0).mean()):.0%}    '
          f'mean kalman gain={univ_gain:.2f}    names={eu.sizes["isin"]}')
    print(f'   Rule: excess = group upside - universe posture, shrunk by '
          f'lambda = tau^2/(tau^2+s_g^2); OVERWEIGHT if shrunk excess >= +1 '
          f'cross-group sd and P(>0|gain)>={P_HI:.0%}; UNDERWEIGHT if <= -1 sd '
          f'or P(>0|gain)<={P_LO:.0%}  (gates = {P_HI_BASE:.0%}/{P_LO_BASE:.0%} '
          f'x mean gain). Groups with n<{MIN_GROUP_N} names are excluded from '
          f'the group signals below.')

    # 4. Group allocation signals (hierarchical coords). The prints remain the
    # full record; the model's actual group-effect coords additionally feed the
    # stacked shrunk-excess forest rendered after the loop.
    _coords = [c for c in
               ('region', 'trading_region', 'exchange_name', 'unit_name', 'country_name','trading_country_name', 'sector', 'industry',
                'size_class',
                'style_class')
               if c in model_df.columns]
    _forest_coords = ('sector', 'industry','size_class','style_class')
    _forest_payload: dict[str, dict[str, Any]] = {}
    for col in _coords:
        lab = model_df[col].fillna('Unknown').astype(str).to_numpy()
        da = xr.DataArray(lab, dims='isin', coords={'isin': isin_dim})
        counts = pd.Series(lab).value_counts()

        # Minimum-coverage gate: keep only groups with n >= MIN_GROUP_N names.
        # Applied before the shrinkage stats so thin groups do not inflate
        # tau^2 (between-group variance) or the ±1 sd OW/UW band.
        grp_all = eu.groupby(da.rename(col)).mean('isin')
        eligible = [g for g in grp_all[col].values
                    if int(counts.get(str(g), 0)) >= MIN_GROUP_N]
        n_dropped = grp_all.sizes[col] - len(eligible)
        if not eligible:
            print(f'\n4.{col.upper()} SIGNALS  (0 of {grp_all.sizes[col]} groups '
                  f'with n>={MIN_GROUP_N} - section skipped)')
            continue

        grp = grp_all.sel({col: eligible})
        stk = grp.stack(s=('chain', 'draw'))
        gmean = stk.mean('s')
        glo = stk.quantile(_HDI_LO, 's')
        ghi = stk.quantile(_HDI_HI, 's')
        gpos = (grp > 5).mean(('chain', 'draw'))
        ggain = gain_da.groupby(da.rename(col)).mean('isin').sel({col: eligible})

        # Hierarchical shrinkage of the per-group excess return vs the universe
        # posture: lambda_g = tau^2 / (tau^2 + s_g^2), where tau^2 is the
        # between-group variance of the posterior group means (the signal) and
        # s_g the posterior sd of group g's mean upside (its noise). Thin or
        # noisy groups shrink toward 0pp excess; the OW/UW band is ±1
        # cross-group sd of the *shrunk* excess — a per-coord, posture-relative
        # replacement for the former static ±2pp gates.
        gsd = stk.std('s')
        excess = gmean - univ_mean
        tau2 = float(excess.var())
        lam = (tau2 / (tau2 + gsd ** 2)) if tau2 > 0 else xr.zeros_like(gsd)
        shrunk = lam * excess
        _band = float(shrunk.std())
        ow_thr = _band if _band > 0 else float('inf')
        uw_thr = -ow_thr

        rows = [(str(g), float(gmean.sel({col: g})), float(glo.sel({col: g})),
                 float(ghi.sel({col: g})),
                 float(shrunk.sel({col: g})),
                 float(gpos.sel({col: g})) * float(ggain.sel({col: g})),
                 int(counts.get(str(g), 0)),
                 float(lam.sel({col: g})) if hasattr(lam, 'sel') else 0.0)
                for g in grp[col].values]
        rows.sort(key=lambda r: r[1], reverse=True)
        if col in _forest_coords:
            _forest_payload[col] = {
                'rows': rows, 'band': _band,
                'verdicts': [_verdict(r[4], r[5], ow_thr, uw_thr) for r in rows],
            }
        print(f'\n4.{col.upper()} SIGNALS  ({len(rows)} groups, '
              f'{n_dropped} dropped n<{MIN_GROUP_N})  '
              f'OW/UW band=±{_band:.2f}pp shrunk excess')

        def _emit(r):
            gs, m, lo, hi, ex, pc, n, _lam = r
            print(f'   {_verdict(ex, pc, ow_thr, uw_thr):>11s}  {gs:<26.26s}  '
                  f'upside={m:6.2f}%  CI=[{lo:6.2f},{hi:6.2f}]  xs={ex:5.2f}pp  '
                  f'P(>0|K)={pc:4.0%}  n={n}')

        if len(rows) > 14:
            for r in rows[:7]:
                _emit(r)
            print(f'   ... {len(rows) - 14} mid-ranked {col} groups omitted ...')
            for r in rows[-7:]:
                _emit(r)
        else:
            for r in rows:
                _emit(r)

    # 4-viz. Stacked shrunk-excess forest over the model's group-effect coords
    # (visual twin of the prints above; best-effort, display only).
    try:
        plot_group_signal_forest(_forest_payload)
    except Exception as exc:  # pragma: no cover - display-only
        print(f'group-signal forest skipped: {exc!r}')

    # 5. Name-level action list. ``nm`` carries the §8-§10 risk analytics + book
    # weights from the shared RiskBook (p_upside_pos, band_width, cvar05, ...).
    # The RiskBook frame stores raw decimals — build local percent columns here
    # so this display-only section prints on the % scale.
    nm = rb.analytics.copy()
    for _c in ('expected_upside', 'band_width', 'exp_vol', 'cvar05', 'tail_risk'):
        nm[f'{_c}_pct'] = pd.to_numeric(nm.get(_c), errors='coerce') * 100.0
    _wide = float(np.nanpercentile(nm['band_width_pct'], 95)) if len(nm) else float('inf')

    _nm_label = _display_label

    # High-conviction gates on the conditional scale: 95% / 5% nominal thresholds
    # rescaled by the universe-mean kalman gain, compared against
    # p_upside_pos_cond = mc_prob_pos * kalman_gain from the shared RiskBook.
    _univ_gain_rb = float(rb.summary.get('univ_gain', 1.0))
    _hi_conv, _lo_conv = 0.75 * _univ_gain_rb, 0.25 * _univ_gain_rb
    longs = nm[(nm['expected_upside_pct'] > 0) & (nm['p_upside_pos_cond'] >= _hi_conv)] \
        .sort_values(['p_upside_pos_cond', 'expected_upside_pct'], ascending=False)
    shorts = nm[(nm['expected_upside_pct'] < 0) & (nm['p_upside_pos_cond'] <= _lo_conv)] \
        .sort_values(['p_upside_pos_cond', 'expected_upside_pct'])
    print('\n5. NAME-LEVEL ACTIONS')
    print(f'   --- High-conviction LONGS (upside>0, P(>0|K)>={_hi_conv:.0%}): '
          f'{len(longs)} names ---')
    for _, r in longs.head(10).iterrows():
        print(f'   BUY    {_nm_label(r):<14.14s}  upside={r["expected_upside_pct"]:6.2f}%  '
              f'P(>0|K)={r["p_upside_pos_cond"]:4.0%}  band={_na(r["band_width_pct"], 1, "%")}  '
              f'n_analysts={_na(r.get("n_analysts"))}')
    print(f'   --- AVOID / SHORT candidates (upside<0, P(>0|K)<={_lo_conv:.0%}): '
          f'{len(shorts)} names ---')
    for _, r in shorts.head(10).iterrows():
        print(f'   AVOID  {_nm_label(r):<14.14s}  upside={r["expected_upside_pct"]:6.2f}%  '
              f'P(>0|K)={r["p_upside_pos_cond"]:4.0%}  band={_na(r["band_width_pct"], 1, "%")}  '
              f'n_analysts={_na(r.get("n_analysts"))}')

    # 6. Position-sizing caution.
    risky = nm[(nm['band_width_pct'] >= _wide) | (nm['n_analysts'].fillna(0) <= 2)]
    print(f'\n6. SIZE-DOWN WATCH (band in top quintile >= {_wide:.1f}% or n_analysts<=2): '
          f'{len(risky)} names')
    for _, r in risky.sort_values('band_width_pct', ascending=False).head(8).iterrows():
        print(f'   CAUTION {_nm_label(r):<14.14s}  upside={r["expected_upside_pct"]:6.2f}%  '
              f'band={_na(r["band_width_pct"], 1, "%")}  n_analysts={_na(r.get("n_analysts"))}'
              f'  -> wide posterior: size to the downside of the credible band.')

    # 7. Earnings-cohort tilt.
    _cids = set(cohort_meta['isin'].astype(str)) if cohort_meta is not None else None
    if _cids:
        _coh = nm[nm['isin'].astype(str).isin(_cids)]
        if len(_coh):
            stance = ('lean INTO' if _coh['expected_upside_pct'].median() >= univ_mean
                      else 'lean AWAY from')
            print(f'\n7. EARNINGS-WEEK COHORT ({len(_coh)} names)  median upside='
                  f'{_coh["expected_upside_pct"].median():.2f}%  '
                  f'high-conviction longs={int((_coh["p_upside_pos_cond"] >= _hi_conv).sum())}')
            print(f'   -> Pre-earnings stance: {stance} the reporting cohort vs the '
                  f'broader universe; gate entries on the SIZE-DOWN WATCH list above.')

    # ------------------------------------------------------------------
    # Risk-adjusted analytics + CVaR-aware sizing (computed once in the shared
    # ``rb`` = compute_cvar_aware_book; the blocks below are display-only).
    #
    # The §5b panel uses a Student-t likelihood whose estimated df (``nu``) is low
    # (heavy tails), so analyst-target outliers dominate naive mean-upside rankings.
    # §8 re-ranks on reward per unit of *expected volatility* (the per-name
    # dispersion of the posterior upside draws); §9 measures each name's expected
    # shortfall (CVaR) tail; §10 reports the long book sized on a reward-to-CVaR
    # budget so a single fat-tailed name cannot dominate the portfolio's downside.
    # ------------------------------------------------------------------
    ALPHA = rb.summary['alpha']
    # Long-eligibility gate on the conditional scale (p_long x universe-mean gain).
    _p_long = float(rb.summary.get('p_long_cond', rb.summary['p_long']))

    # 8. Risk-adjusted return (reward per unit expected volatility).
    _vmed = (float(np.nanmedian(nm['exp_vol_pct']))
             if nm['exp_vol_pct'].notna().any() else float('nan'))
    print(f'\n8. RISK-ADJUSTED RETURN  median expected vol={_na(_vmed, 1, "%")}  '
          f'(reward per unit expected volatility; demotes high-upside/high-vol names)')
    _ra_book = nm[(nm['expected_upside_pct'] > 0) & (nm['p_upside_pos_cond'] >= _p_long)
                  & nm['ret_vol_ratio'].notna()]
    if len(_ra_book):
        _ra_top = _ra_book.sort_values('ret_vol_ratio', ascending=False).head(10)
        print(f'   --- Top risk-adjusted LONGS ({len(_ra_book)} eligible, by upside/vol) ---')
        for _, r in _ra_top.iterrows():
            print(f'   BUY-RA {_nm_label(r):<14.14s}  upside={r["expected_upside_pct"]:6.2f}%  '
                  f'vol={_na(r["exp_vol_pct"], 1, "%")}  ret/vol={_na(r["ret_vol_ratio"], 2)}  '
                  f'P(>0|K)={r["p_upside_pos_cond"]:4.0%}')
        # Names the vol screen demotes: top raw upside but bottom-half risk-adjusted.
        _hi_up = _ra_book.sort_values('expected_upside_pct', ascending=False).head(
            max(10, len(_ra_book) // 5))
        _med_rv = float(_ra_book['ret_vol_ratio'].median())
        _demoted = _hi_up[_hi_up['ret_vol_ratio'] < _med_rv].sort_values(
            'expected_upside_pct', ascending=False).head(6)
        if len(_demoted):
            print(f'   --- DEMOTED by volatility (high upside, ret/vol < universe median '
                  f'{_med_rv:.2f}) ---')
            for _, r in _demoted.iterrows():
                print(f'   TRIM   {_nm_label(r):<14.14s}  upside={r["expected_upside_pct"]:6.2f}%  '
                      f'vol={_na(r["exp_vol_pct"], 1, "%")}  ret/vol={_na(r["ret_vol_ratio"], 2)}'
                      f'  -> upside is volatility-inflated; prefer the risk-adjusted leaders.')
    else:
        print('   [~]  No expected-volatility coverage on the long book - vol screen skipped.')

    # 9. Tail risk (CVaR / expected shortfall, 5%).
    _univ_cvar = (float(np.nanmean(nm['cvar05_pct']))
                  if nm['cvar05_pct'].notna().any() else float('nan'))
    _share_mc_loss = (float((nm['er_p05'] < 0).mean()) if 'er_p05' in nm.columns
                                                          and nm['er_p05'].notna().any() else float('nan'))
    print(f'\n9. TAIL RISK (CVaR, {ALPHA:.0%} expected shortfall)  '
          f'universe mean 5% upside-ES={_na(_univ_cvar, 2, "%")}  '
          f'names with simulated loss tail (MC p05<0)={_na(_share_mc_loss * 100, 0, "%")}')
    _tail_book = nm[(nm['expected_upside_pct'] > 0) & (nm['p_upside_pos_cond'] >= _p_long)]
    if len(_tail_book):
        _worst = _tail_book.sort_values('tail_risk_pct', ascending=False).head(8)
        print('   --- Fattest-tail LONGS (largest mean->ES gap / MC loss; sized down) ---')
        for _, r in _worst.iterrows():
            _mc05 = ((r['er_p05'] * 100.0)
                     if 'er_p05' in r and np.isfinite(r.get('er_p05', np.nan)) else np.nan)
            print(f'   TAIL   {_nm_label(r):<14.14s}  upside={r["expected_upside_pct"]:6.2f}%  '
                  f'CVaR5={_na(r["cvar05_pct"], 1, "%")}  MC_p05={_na(_mc05, 1, "%")}  '
                  f'tail_risk={_na(r["tail_risk_pct"], 1, "%")}  STARR={_na(r["starr"], 2)}')
        print('   -> CVaR is the mean of the worst 5% of outcomes; with heavy tails it, not '
              'the mean upside, is the right sizing denominator.')

    # 10. CVaR-aware position sizing (long book, expected-shortfall budget).
    # ``rb.book`` and the ``rb.summary`` return metrics are decimal — scale at print.
    _s = rb.summary
    _book = rb.book.copy()
    for _c in ('expected_upside', 'exp_vol', 'cvar05'):
        _book[f'{_c}_pct'] = pd.to_numeric(_book.get(_c), errors='coerce') * 100.0
    print(f'\n10. CVaR-AWARE SIZING  (top {int(_s["k_book"])} reward-to-CVaR longs, '
          f'{_s["cap"]:.0%} name cap, '
          f'mcap rank < {_s.get("mcap_r_max", float("nan")):.0%} of country, '
          f'100% gross)')
    if len(_book):
        print(f'   {"NAME":<14}  {"wt":>6}  {"upside":>8}  {"vol":>7}  {"CVaR5":>8}  {"STARR":>6}')
        for _, r in _book.iterrows():
            print(f'   {_nm_label(r):<14.14s}  {r["weight"] * 100:5.1f}%  '
                  f'{r["expected_upside_pct"]:7.2f}%  {_na(r["exp_vol_pct"], 1, "%"):>7}  '
                  f'{_na(r["cvar05_pct"], 1, "%"):>8}  {_na(r["starr"], 2):>6}')
        print(f'   PORTFOLIO  expected upside={_na(_s["port_up"] * 100.0, 2, "%")}  '
              f'CVaR5={_na(_s["port_cvar"] * 100.0, 2, "%")}  reward/CVaR={_na(_s["starr_book"], 2)}  '
              f'approx vol(upper bnd)={_na(_s["port_vol"] * 100.0, 1, "%")}')
        print(f'   Diversification: weighted-avg name CVaR5={_na(_s["wavg_cvar"] * 100.0, 2, "%")} vs '
              f'portfolio CVaR5={_na(_s["port_cvar"] * 100.0, 2, "%")} '
              f'(tail lift x{_na(_s["div"], 2)}); weights inverse to expected shortfall, so '
              f'the analyst-target outliers are capped, not chased. Persisted to '
              f'analytics.kalman_filtered_price_targets as cvar_book_weight.')
        # 10-viz. Book composition + portfolio aggregates (best-effort visual
        # twin of the sizing table above).
        try:
            plot_book_composition(rb)
        except Exception as exc:  # pragma: no cover - display-only
            print(f'book-composition chart skipped: {exc!r}')
    else:
        print(f'   [~]  Insufficient long book or posterior draws for CVaR-aware sizing '
              f'({_na(_s.get("n_mcap_eligible"), 0)} names passed the mcap gate).')

    print('\n' + '=' * 88)
    print('Signals are model-implied screens from analyst-target dynamics, NOT investment '
          'advice; size on the §10 CVaR budget and combine with fundamentals, liquidity '
          'and risk limits.')
    print('=' * 88)


# =============================================================================
# Entry point
# =============================================================================
def main(*, run_eda_section: bool = True, write_analytics: bool = True,
         robust: bool = False, volume_penalty: float = 0.25, export_results: bool = True,
         config: Optional[KalmanRunConfig] = None) -> dict[str, Any]:
    """Run the full Kalman price-target workflow end-to-end on the fused panel model.

    The cross-sectional spine is the §5b fused MvGRW + volatility-conditioned model
    (:func:`build_fused_kalman_pt_model`). Per-ISIN screening signals are de-standardised
    from the posterior ``risk_adj_return`` baseline and cross-checked with the canonical
    structural-TS Monte-Carlo of risk-adjusted forward returns.

    Parameters
    ----------
    run_eda_section
        When ``True`` (default), render the section-2 EDA panels.
    write_analytics
        When ``True``, replace ``analytics.kalman_filtered_price_targets`` with the
        §10c screen (drop-and-recreate) and regenerate its checked-in DDL.
    robust
        When ``True``, use the Student-t panel likelihood (absorbs analyst
        outliers); ``False`` (default) selects the Normal-likelihood twin.
    volume_penalty
        Prior scale (``HalfNormal`` sigma) of the learned ``volume_loading``
        tilt on ``risk_adj_return`` via the per-ISIN relative trading volume
        (``feat_rel_volume``, z-scored). Defaults to ``0.25`` (enabled), which
        overrides the ``0.2`` builder default in
        :func:`~probabilistic_ml_model.pymc_models.KalmanFilterModel.build_fused_kalman_pt_model`;
        ``0.0`` disables the factor.
    export_results
        When ``True`` (default), persist every rendered figure / displayed table
        (via the :func:`_safe_show` / :func:`display` hooks) and the bulk data
        artifacts (:func:`export_all_artifacts`) under ``KALMAN_PT_RESULTS_DIR``.
    config
        Optional :class:`KalmanRunConfig` overriding the env-resolved defaults
        (sampling budget, screen / risk-book knobs, universe-query dates).
        Installed as the module run config for the duration of the call.

    Returns
    -------
    dict
        Key artifacts (``idata``, ``results``, ``kalman_results``, ``panel``,
        ``screen``, ``universe_fit``) for programmatic reuse.
    """
    if config is not None:
        set_run_config(config)
    cfg = get_run_config()
    logging.basicConfig(level=cfg.log_level)
    # Headless / redirected-stdout runs on Windows default to cp1252, which
    # cannot encode the Unicode used by the samplers' console output (nutpie
    # emits U+2009; several prints use '≈'/'σ'). A cp1252 UnicodeEncodeError
    # destroyed an otherwise-complete 36-minute nutpie run (2026-07-31), so
    # make stdout/stderr lossy-safe instead of fatal.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, 'reconfigure'):
            with contextlib.suppress(Exception):
                _stream.reconfigure(errors='replace')
    setup_plotting()
    if export_results:
        enable_artifact_export()
        logger.info("Artifact export enabled -> %s", get_export_state().root)

    engine = create_engine(resolve_db_url())

    # Data load + role resolution.
    with export_section('01_data'):
        kalman_df = load_kalman_df(engine, cfg)
        feature_catalogue = load_feature_catalogue(engine)
        roles = resolve_feature_roles(kalman_df, feature_catalogue)

    # §2 EDA (optional).
    if run_eda_section:
        with export_section('02_eda'):
            run_eda(kalman_df, roles)

    # §3 state-space feature mapping + §4 fused-panel data containers.
    with export_section('03_features'):
        drift_features, _mapping = map_state_space_features(kalman_df, feature_catalogue)
        panel = prepare_kalman_panel_inputs(
            kalman_df, roles, drift_features,
            history_lookbacks=cfg.panel_lookbacks,
            response_extra=cfg.panel_response_extra)

    # §5b fused model -> §6 prior -> §7 posterior -> §8 PPC.
    model = build_panel_model(panel, robust=robust, volume_penalty=volume_penalty,
                              config=cfg)
    with export_section('06_prior'):
        prior_idata = run_prior_predictive(model, panel, cfg)
    with export_section('07_posterior'):
        idata = sample_posterior(model, prior_idata, panel=panel, config=cfg)
    with export_section('08_ppc'):
        run_posterior_predictive(model, idata, panel)

    # §9 diagnostics -> §9b comparison -> §10 screening table -> §10b risk book.
    with export_section('09_diagnostics'):
        run_diagnostics(idata, panel)
    # §9b model comparison is OPT-IN: it refits BOTH arms on a subsampled panel
    # and computes a pointwise log_likelihood for each, so it roughly triples the
    # run's sampling cost. Enable with replace(cfg, enable_model_comparison=True).
    if cfg.enable_model_comparison:
        with export_section('09b_comparison'):
            run_model_comparison(panel, config=cfg, robust=robust,
                                 volume_penalty=volume_penalty)
    with export_section('10_screen'):
        screen = summarize_panel_screen(idata, panel,
                                        horizon=cfg.mc_horizon, rho=cfg.mc_rho)
        results = screen.results
    with export_section('10b_risk'):
        risk_book = compute_cvar_aware_book(idata, panel, screen, results, config=cfg)
    with export_section('10c_analytics'):
        kalman_results = export_analytics(idata, panel, screen, risk_book=risk_book,
                                          write=write_analytics)

    # §10K universe-consensus fit via the shared fit_kalman_model driver
    # (all kalman_df rows pooled into one median *_ago consensus trail).
    with export_section('10k_universe'):
        universe_fit = run_universe_kalman_fit(kalman_df)

    # §11 single-ISIN filter (+11b SV).
    with export_section('11_single_isin'):
        single_ctx = run_single_isin_filter(panel.frame, engine, cfg)
        _export_context(single_ctx, '11_single_isin_ctx')
    with export_section('11b_single_sv'):
        run_single_isin_stochastic_vol(single_ctx)

    # §12 mingled cohort (+12b SV).
    with export_section('12_mingled'):
        mingled_ctx = run_mingled_cohort_filter(panel.frame, engine, cfg)
        _export_context(mingled_ctx, '12_mingled_ctx')
    with export_section('12b_mingled_sv'):
        run_mingled_cohort_stochastic_vol(panel.frame, mingled_ctx)

    # §13 granular forest (+13.1 further views).
    with export_section('13_forest'):
        forest_ctx = run_granular_forest(idata, results, panel, screen, engine, cfg)
        _export_context(forest_ctx, '13_forest_ctx')
    with export_section('13b_further_views'):
        run_granular_further_views(prior_idata, panel, screen, forest_ctx)

    # §14 summary + 14b recommendations.
    with export_section('14_summary'):
        run_summary(results, screen, forest_ctx, mingled_ctx)
    with export_section('14b_recommendations'):
        run_recommendations(idata, panel, results, screen, forest_ctx, risk_book=risk_book)

    artifacts = {'idata': idata, 'prior_idata': prior_idata, 'results': results,
                 'kalman_results': kalman_results, 'panel': panel, 'screen': screen,
                 'risk_book': risk_book, 'universe_fit': universe_fit}
    if export_results:
        export_all_artifacts(artifacts)
    return artifacts


def _parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the script's command-line arguments.

    The default invocation (no flags) runs the full workflow, as documented in
    the module docstring; ``--migrate-layout`` runs only the one-off results-tree
    migration and exits.
    """
    parser = argparse.ArgumentParser(
        prog='pymc_kalman_filter_pt',
        description='Bayesian Kalman price-target workflow '
                    '(fused panel model -> screen -> risk book -> analytics export).')
    parser.add_argument(
        '--migrate-layout', action='store_true',
        help='Re-file flat legacy artifacts in KALMAN_PT_RESULTS_DIR into the '
             'per-section subdirectory tree, then exit. Dry-run unless --apply.')
    parser.add_argument(
        '--apply', action='store_true',
        help='Perform the --migrate-layout moves instead of only reporting them.')
    parser.add_argument(
        '--results-dir', default=None,
        help='Override the results directory for --migrate-layout.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    _args = _parse_cli_args()
    if _args.migrate_layout:
        logging.basicConfig(level=logging.INFO)
        migrate_results_layout(_args.results_dir, dry_run=not _args.apply)
    else:
        main()