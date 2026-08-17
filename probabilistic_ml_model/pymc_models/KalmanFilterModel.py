"""
Kalman Filter Price Target Model — GaussianRandomWalk state-space.

Uses PyMC GaussianRandomWalk for latent price state with an observation model
for noisy price targets.

Reference: run_kalman_filter() in expected_returns_v3.py (line 1519);
           probability_models.py Kalman references.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Hashable, Iterable, Literal, Optional, Sequence, TYPE_CHECKING, Union

from pandas import Timestamp
from pandas._libs import NaTType
from pandas.tseries.offsets import DateOffset, MonthBegin, QuarterBegin, YearBegin

# PyMC 6.0 + ArviZ 1.0: top-level ``arviz`` re-exports the modular API,
# so the legacy ``arviz_base`` fallback (PyMC 5.x transition artefact) is
# no longer needed. The ``ImportError`` guard is kept defensively for
# environments where arviz is absent.
try:
    import arviz as az
except ImportError:
    az = None  # type: ignore[assignment]
import numpy as np
import pandas as pd

try:
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:
    pm = None  # type: ignore[assignment]
    pt = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import arviz as az_typing  # noqa: F401
    import pymc as pm_typing  # noqa: F401
    import xarray as xr_typing  # noqa: F401
    from pymc.backends.base import MultiTrace  # noqa: F401

from probabilistic_ml_model._pymc_arviz_compat import InferenceLike
from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs
from probabilistic_ml_model.pymc_models._workflow import (
    MIN_ESS_GATE,
    build_sample_kwargs,
    log_sample_diagnostics,
)
from probabilistic_ml_model.pymc_models._hierarchy import (
    build_hierarchy_indices,
    coerce_categories,
)

logger = logging.getLogger(__name__)

Parameterization = Literal["centered", "non_centered", "marginalized", "auto"]

# Below this many observations the latent random-walk funnel becomes hard to
# sample; ``parameterization="auto"`` collapses to the funnel-free
# ``marginalized`` form for such short ``*_ago`` cohorts.
_SHORT_SERIES_THRESHOLD = 25

# Default MCMC budget shared by ``fit`` / ``fit_from_snapshot`` / ``forecast``.
_DEFAULT_SAMPLES = 2000
_DEFAULT_TUNE = 1000
_DEFAULT_CHAINS = 4
_DEFAULT_TARGET_ACCEPT = 0.90
_DEFAULT_RANDOM_SEED = 42
# Minimum bulk ESS before ``log_sample_diagnostics`` warns. Re-exported from
# the shared workflow module so the gate has exactly one home project-wide.
_MIN_ESS_GATE = MIN_ESS_GATE

# ---------------------------------------------------------------------------
# Historical "*_ago" cohort -> per-ISIN time-series helpers.
#
# Background. ``mv_all_stock_features`` is a point-in-time snapshot, so any
# event-style date column (e.g. ``next_earnings``) only ever holds one row
# per ISIN -> ``groupby(isin).size() >= 2`` is unsatisfiable. The snapshot,
# however, embeds a historical axis through the ``*_ago`` family
# (``price_target_1w_ago``, ``price_target_high_3m_ago``, ``price_1y_ago``,
# ``price_target_median_ytd_ago`` …). Unpivoting these columns yields up to
# ~16 ordered observations per ISIN with a real ``pd.DatetimeIndex`` derived
# from the suffix — exactly what the GaussianRandomWalk state-space model
# in :class:`KalmanFilterPriceTarget` needs.
#
# ``pml.mv_pymc_kalman_pt`` emits the raw spot-price trail
# (``price_{5d,1w,1m,3m,6m,1y,3y,5y,qtd}_ago``) un-prefixed alongside the
# analyst-target trail, so snapshots loaded from the MV (not just ``pml_df``)
# carry the full price + target ``*_ago`` cohort that
# :data:`AGO_HISTORY_RE` unpivots.
#
# This logic lives on the model class so the notebook (Section 7.1 of
# ``pymc_expected_returns_model.ipynb``) and any future batch scorer can
# call a single canonical helper instead of inlining the unpivot.
# ---------------------------------------------------------------------------

# Map textual *_ago suffix -> pandas DateOffset. ``AGO_HISTORY_RE`` is the
# SSOT ``*_ago`` history-column pattern shared with ``pymc_kalman_filter_pt.py``
# (sections 11-13 column discovery against ``pml.pml_df``).
AGO_SUFFIX_PATTERN = r"5d|1w|1m|3m|6m|1y|3y|5y|mtd|qtd|ytd"
AGO_HISTORY_RE = (
    r"^(price_target(?:_high|_low|_median)?|price)"
    r"_(?P<suf>" + AGO_SUFFIX_PATTERN + r")_ago$"
)
_AGO_HISTORY_PATTERN = re.compile(AGO_HISTORY_RE)


@lru_cache(maxsize=1)
def _build_ago_offset_map() -> dict[str, Any]:
    return {
        "1w": DateOffset(weeks=1),
        "1m": DateOffset(months=1),
        "3m": DateOffset(months=3),
        "6m": DateOffset(months=6),
        "1y": DateOffset(years=1),
        "3y": DateOffset(years=3),
        "5y": DateOffset(years=5),
        "5d": DateOffset(days=5),
        # Period-to-date — anchor to the start of the current period.
        "mtd": MonthBegin(0),
        "qtd": QuarterBegin(0, startingMonth=1),
        "ytd": YearBegin(0),
    }


# ---------------------------------------------------------------------------
# Fiscal-calendar forecast horizons (single source of truth = the ``days_*``
# aliases minted by ``pml.mv_pymc_kalman_pt``).
#
# ``mv_pymc_kalman_pt`` derives an integer day-count horizon from each
# fiscal-calendar DATE column, e.g.::
#
#     next_earnings                     - CURRENT_DATE  AS days_to_next_earnings
#     CURRENT_DATE - income_statement_report_date       AS days_since_last_report
#     next_income_statement_report_date - CURRENT_DATE  AS days_to_next_report
#     expected_report_date              - CURRENT_DATE  AS days_to_expected_report
#     next_fy_end_date                  - CURRENT_DATE  AS days_to_next_fy_end
#     fy_end_date                       - CURRENT_DATE  AS days_since_fy_end
#     next_fiscal_quarter               - CURRENT_DATE  AS days_to_next_fiscal_quarter
#
# :meth:`KalmanFilterPriceTarget.forecast` projects to exactly these horizons,
# so the triple (DATE column, day-count column, human label) is captured here
# once and reused by every caller — rather than re-inlined, and re-mislabelled,
# per call site.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiscalHorizon:
    """A single fiscal-calendar forecast horizon on ``pml.mv_pymc_kalman_pt``.

    Attributes
    ----------
    date_col
        Fiscal-calendar DATE column the horizon projects to
        (e.g. ``"next_earnings"``).
    day_count_col
        Integer day-count column the MV derives from ``date_col``
        (the SQL ``AS`` alias, e.g. ``"days_to_next_earnings"``).
    label
        Human-readable label for forecast tables / plot annotations
        (e.g. ``"Next earnings"``).
    """

    date_col: str
    day_count_col: str
    label: str


# Ordered fiscal-calendar horizons, mirroring the ``days_*`` aliases on
# ``pml.mv_pymc_kalman_pt`` (and ``DAY_COUNT_COLS_ALL`` in
# ``pymc_kalman_filter_pt.py``).
FISCAL_HORIZONS: tuple[FiscalHorizon, ...] = (
    FiscalHorizon("next_earnings", "days_to_next_earnings", "Next earnings"),
    FiscalHorizon("income_statement_report_date", "days_since_last_report", "Last report"),
    FiscalHorizon("next_income_statement_report_date", "days_to_next_report", "Next report"),
    FiscalHorizon("expected_report_date", "days_to_expected_report", "Expected report"),
    FiscalHorizon("next_fy_end_date", "days_to_next_fy_end", "Next FY end"),
    FiscalHorizon("fy_end_date", "days_since_fy_end", "FY end"),
    FiscalHorizon("next_fiscal_quarter", "days_to_next_fiscal_quarter", "Next fiscal quarter"),
)


# ---------------------------------------------------------------------------
# Catalogue-driven drift-feature selection (SSOT).
#
# The candidate pool is ``pml.vw_pymc_feature_catalogue`` rows with
# ``model_target = 'kalman_pt' AND pymc_role = 'mutable_predictor'``
# (see :meth:`KalmanFilterPriceTarget._resolve_kalman_feature_aliases`).
# Not every mutable_predictor may enter the drift design matrix: some aliases
# are consumed elsewhere in the fused model (noise wideners, named tilts, the
# ``t_scaled`` time covariate) and one is the response itself. These frozensets
# partition the pool so drift selection stays deterministic and auditable.
# ---------------------------------------------------------------------------

# ``feat_implied_upside`` is a deterministic function of the RESPONSE
# (``feat_log_uplift = log1p(feat_implied_upside)``) — never a drift predictor.
KALMAN_LEAKAGE_FEATURES: frozenset[str] = frozenset({"feat_implied_upside"})

# sigma_obs wideners (``build_noise_wideners`` in ``pymc_kalman_filter_pt.py``):
# they scale the observation noise, so they must not double as mean predictors.
# Each widener has a distinct role in the measurement model, so the individual
# column names are exported alongside the aggregate frozenset — downstream
# consumers must reference these constants rather than re-spelling the strings.
KALMAN_RANGE_WIDENER_FEATURE: str = "feat_pt_range_norm"
KALMAN_CONSENSUS_SIGMA_FEATURE: str = "feat_pt_noise_sigma"
KALMAN_VOL_DRIFT_FEATURE: str = "feat_vol_drift"
# Realized-vol LEVEL and log size, added 2026-08-16. Both are observation-scale
# drivers, so they belong here (barred from the drift design) and enter
# ``sigma_isin`` in ``build_fused_kalman_pt_model``.
#
# Why they exist: measured on the live 6,533-row MV, correlation with
# log|residual| after an OLS fit of the 17 drift features --
#
#     cv = pt_stddev/price   +0.2245   (already in sigma_isin)
#     feat_log_mcap          -0.2100   <- new
#     volatility_1m          +0.1924   <- feat_vol_level
#     volatility_1y          +0.1897   <- feat_vol_level
#     log n_analysts         -0.1696   (already in sigma_isin)
#     feat_pt_range_norm     +0.1150   (documented as a widener; was NEVER wired)
#     feat_vol_drift         -0.0349   <- the drift is nearly useless
#
# 0.9.9.6 replaced the absolute vol LEVELS with feat_vol_drift on the reasoning
# that the drift is the sigma_obs analogue of feat_pt_noise_drift. That traded a
# +0.19 driver for a -0.03 one: how fast volatility is CHANGING says almost
# nothing about how dispersed a name's implied upside is; how volatile it IS says
# a great deal. ``feat_vol_drift`` is kept as a container for provenance and for
# the EDA panel, but it is no longer the volatility term the scale model uses.
KALMAN_VOL_LEVEL_FEATURE: str = "feat_vol_level"
KALMAN_LOG_MCAP_FEATURE: str = "feat_log_mcap"
KALMAN_NOISE_WIDENER_FEATURES: frozenset[str] = frozenset({
    KALMAN_RANGE_WIDENER_FEATURE,
    KALMAN_CONSENSUS_SIGMA_FEATURE,
    KALMAN_VOL_DRIFT_FEATURE,
    KALMAN_VOL_LEVEL_FEATURE,
    KALMAN_LOG_MCAP_FEATURE,
})

# Named tilt drivers with dedicated ``pm.Data`` containers + sign-fixed loadings
# in ``build_fused_kalman_pt_model`` (risk / size / volume adjustments on
# ``risk_adj_return``); folding them into the generic ``beta`` block as well
# would double-count them. The ordered tuple backs display frames; the
# frozenset backs the drift-exclusion union.
KALMAN_TILT_FEATURE_ORDER: tuple[str, ...] = (
    "feat_avg_beta", "feat_mcap_global_r", "feat_rel_volume",
)
KALMAN_TILT_FEATURES: frozenset[str] = frozenset(KALMAN_TILT_FEATURE_ORDER)

# Market-cap percentile-rank siblings of the ``feat_mcap_global_r`` size tilt.
#
# ``mv_pymc_kalman_pt`` emits six nested rank-ratio scopes — global / region /
# country, each also sector-relative — all built as ``(100 - rank) / 100``.
# ``feat_mcap_global_r`` is the ONE the model consumes, as a tilt driver with its
# own ``pm.Data`` container and sign-fixed loading; the other five restate the same
# size contrast against a different peer set. Measured on the four regional CSV
# snapshots (6 274 complete rows), the pairwise correlations run:
#
#     r(global_r, global_sec_r)   = +0.978
#     r(region_r, country_r)      = +0.945
#     r(global_r, country_sec_r)  = +0.783   <- the WEAKEST pair in the family
#
# Admitting them to the drift matrix would therefore add six near-duplicate
# columns and wreck the conditioning the pruned matrix was tuned for (cond 19.7,
# max VIF 4.25). Promoting them to additional tilt drivers is no better: ~0.9
# correlated loadings are weakly identified and fight for the same signal.
#
# They stay in the MV and the catalogue because the screen, the §11-§13 candidate
# cohorts, EDA and the analytics export all use them. As the header below spells
# out, the drift matrix is a Python-side concern — this frozenset is where the
# exclusion belongs, NOT a ``pymc_role='excluded'`` flip in SQL.
KALMAN_SIZE_RANK_SIBLING_FEATURES: frozenset[str] = frozenset({
    "feat_mcap_global_r",
    "feat_mcap_global_sec_r",
    "feat_mcap_region_r",
    "feat_mcap_region_sec_r",
    "feat_mcap_country_sec_r",
})

# Valid-pair counters behind the ``target_drift_n`` SQL helper
# (feature_role='metadata'): drift-support diagnostics, not signals.
KALMAN_DRIFT_SUPPORT_COUNTERS: frozenset[str] = frozenset({
    "feat_pt_drift_n",
    "feat_price_drift_n",
    "feat_vol_drift_n",
    "feat_net_eps_drift_n",
})

# Raw analyst-rating counts. ``feat_analyst_rating`` enters the drift matrix as
# the family's single representative; the raw counts are collinear with it and
# with ``n_analysts``.
KALMAN_RATING_COUNT_FEATURES: frozenset[str] = frozenset({
    "feat_holds",
    "feat_buys",
    "feat_sells",
    "feat_no_opinion",
})

# The analyst-sentiment family collapses to ONE column in the drift matrix.
#
# ``feat_analyst_{bullish_pct,bearish_pct,neutral_pct,conviction,rating}`` are all
# functions of the same six ``num_*_ratings`` buckets, and ``conviction`` is
# literally ``|bullish - bearish|``. The three percentage legs sum to ~1, so the
# triplet is exactly collinear after z-scoring; ``conviction`` and ``rating`` then
# restate the same contrast again. Measured on the live ``mv_pymc_kalman_pt``
# (6 549 rows):
#
#     r(bullish, conviction) = +0.926   VIF(bullish)    = 14.4
#     r(bullish, rating)     = +0.909   VIF(rating)     = 11.8
#     r(conviction, rating)  = +0.837   VIF(conviction) =  7.2
#     r(bearish, rating)     = -0.691   VIF(bearish)    =  2.6
#
# ``feat_analyst_rating`` is kept because it measurably dominates the
# alternatives: retaining it alone scores R^2 0.6499 against the log-uplift
# response versus 0.6463 for the ``bullish_pct`` + ``bearish_pct`` pair, with one
# FEWER column, and it is 99.76 % populated (16 missing of 6 549). It is the
# vendor's weighted composite of all six buckets, where ``bullish_pct`` is a
# crude share — which is why it carries more of the signal.
KALMAN_COLLINEAR_COMPOSITION_FEATURES: frozenset[str] = frozenset({
    "feat_analyst_neutral_pct",
    "feat_analyst_bullish_pct",
    "feat_analyst_bearish_pct",
    "feat_analyst_conviction",
})

# Siblings of ``feat_pt_drift``: the SAME ``pml.target_drift()`` computation run
# over the median / high / low analyst-target trails instead of the mean. The
# four move together almost rigidly, because a consensus revision shifts the whole
# target band at once:
#
#     r(pt_drift, pt_low_drift)      = +0.889   VIF(pt_drift)        = 162.5
#     r(pt_low_drift, pt_median)     = +0.875   VIF(pt_median_drift) =  77.0
#     r(pt_high_drift, pt_low_drift) = +0.805   VIF(pt_high_drift)   =  24.7
#                                               VIF(pt_low_drift)    =   6.6
#
# The consensus MEAN drift is retained as the family's representative. Dropping
# the siblings costs R^2 0.6538 -> 0.6518 (0.3 % relative).
#
# An orthogonal REPLACEMENT was tried instead of a plain drop and rejected:
# ``feat_pt_high_drift - feat_pt_low_drift`` (how the analyst band is widening,
# as distinct from where its centre is moving) correlates -0.003 with the
# response and moves R^2 by 0.0001. The high/low drifts genuinely are duplication
# of the level drift, not a separate dispersion signal — ``feat_pt_noise_drift``
# (the stddev trail's drift) already carries dispersion dynamics.
KALMAN_PT_DRIFT_SIBLING_FEATURES: frozenset[str] = frozenset({
    "feat_pt_median_drift",
    "feat_pt_high_drift",
    "feat_pt_low_drift",
})

# Per-fiscal-year Piotroski F-score components (``pml.piotroski_f_score`` in
# ``mv_pymc_kalman_pt``): their median ``feat_median_piotroski_f_score`` is the
# fundamental-quality drift predictor; the four components are collinear with
# it and stay out of the drift matrix (EDA / analytics export only).
KALMAN_PIOTROSKI_COMPONENT_FEATURES: frozenset[str] = frozenset({
    "feat_piotroski_f_score_fy",
    "feat_piotroski_f_score_neg1fy",
    "feat_piotroski_f_score_neg2fy",
    "feat_piotroski_f_score_neg3fy",
})

# ``days_*`` mutable_predictors are fiscal-calendar time covariates: they feed
# the standardised ``t_scaled`` axis (``DAY_COUNT_COLS_ALL``), not the drift
# design matrix.
KALMAN_TIME_COVARIATE_PREFIX: str = "days_"

# Union of every alias barred from the drift design matrix (``days_*`` columns
# are excluded by prefix, see KALMAN_TIME_COVARIATE_PREFIX).
#
# 2026-08-13 — MARKET-CAP / EV FAMILY SWAPPED FOR AN EPS FAMILY. Four columns
# left ``mv_pymc_kalman_pt`` entirely (so they need no entry here):
# ``feat_mv_ev_drift``, ``feat_mcap_trend_1y``, ``feat_mcap_vs_3yavg`` and
# ``feat_ev_vs_3yavg``. All four were PRICE-derived — ``market_cap`` is
# ``last_price * shrs_out`` — so they restated price history the design already
# carried through ``feat_price_drift``, ``feat_price_chg_pct_3m``,
# ``feat_one_day_return`` and the ``feat_total_return_*`` family, without saying
# anything about WHY analysts revise a target. Their replacements are
# earnings-derived and orthogonal to the price trails: ``feat_net_eps_drift``
# (sign-preserving drift of the net-basic-EPS fiscal-year trail — see
# ``pml.signed_drift``: EPS crosses zero, so the usual ``pml.target_drift``
# denominator would invert the signal for loss-makers), ``feat_last_q_surprise``
# / ``feat_last_y_surprise`` and ``feat_eps_beat_rate`` /
# ``feat_eps_beat_rate_annual``. Only the support counter
# ``feat_net_eps_drift_n`` is excluded, via KALMAN_DRIFT_SUPPORT_COUNTERS. The
# trio survives unchanged in the other six ``mv_pymc_*`` views; only the
# ``kalman_pt`` model_target was dropped.
#
# The surviving drift set is 16 columns. The 2026-07-31 note here
# flagged ``feat_analyst_conviction`` as a null beta to "revisit if it stays null
# on the genuine (isin, time) panel" — it did, and the T=4 panel supplied the
# reason: conviction is ``|bullish - bearish|``, so it never carried information
# the rest of the analyst family did not already have. It is now excluded, along
# with the rest of that family bar ``feat_analyst_rating`` and the three
# ``feat_pt_*_drift`` siblings. (``feat_one_day_return``, the other null beta in
# that note, is KEPT: its VIF is ~1, so it is merely uninformative rather than
# collinear, and dropping it would not improve conditioning.)
#
# Why this mattered: the 21-column design had condition number 1 580 and a
# smallest eigenvalue of 0.004 — near-singular. That left ``beta`` on long, thin
# ridges which NUTS crawls along, and it was the LAST failing gate on the
# 2026-08-10 full-scale validation (R-hat 1.026 / bulk-ESS 140 against 1.01 / 400,
# at ZERO divergences — the signature of poor conditioning rather than bad
# geometry). Pruning to 15:
#
#     set                                  k    cond   maxVIF   R^2
#     current                             21   1 580    162.5   0.6538
#     - pt_drift siblings                 18     106     14.1   0.6518
#     - + analyst family                  15      23      3.8   0.6499
#
# i.e. 69x better conditioning for a 0.6 % relative loss in explanatory power.
#
# The 2026-08-13 mcap/EV -> EPS swap, re-measured on the live 6 538-name MV with
# ONE consistent response construction (log1p of the support-clipped implied
# upside) so the two arms are directly comparable — these numbers are NOT on the
# same scale as the 21/18/15 rows above, which used the earlier construction:
#
#     set                                  k    cond   maxVIF   R^2
#     mcap/EV family (previous)           15    24.7     4.36   0.6695
#     EPS family (this build)             16    19.7     4.25   0.6672
#
# Read that honestly: explanatory power is FLAT (-0.3 % relative), not improved.
# The swap is not an R^2 play. The mcap/EV columns scored as well as they did
# precisely BECAUSE they restate price — and the response is analyst-implied
# upside off that same price, so their contribution was closer to duplication
# than to independent signal. What the swap buys is (a) modestly better
# conditioning, and (b) five predictors that are near-orthogonal to everything
# else in the design: every new column lands at VIF <= 1.19, and the strongest
# correlation among them is r(eps_beat_rate, eps_beat_rate_annual) = +0.369 —
# well inside the range where both legs are worth carrying.
#
# Coverage is the live caveat, measured on the same 6 540-row MV:
# feat_net_eps_drift 100 %, feat_eps_beat_rate_annual 85.2 %,
# feat_last_y_surprise 76.4 %, feat_eps_beat_rate 54.3 %,
# feat_last_q_surprise 46.6 %. The two quarterly columns are thin. That is
# tolerable HERE because a sparse PREDICTOR zero-fills to the column mean after
# standardisation (``prepare_kalman_panel_inputs``), which shrinks its beta
# toward 0 without dropping a single name — unlike a sparse RESPONSE series,
# which is what produced the rank-1 ICM identification failure documented in
# ``pml_feature_catalogue.sql`` (max R-hat 4.45, min ESS 4.3). If the quarterly
# betas come back dead on the next full-scale fit, drop them here rather than in
# SQL, for the reason given below.
#
# These columns are NOT removed from ``mv_pymc_kalman_pt`` or the catalogue. They
# remain valid features for EDA and the analytics export, and the analyst family
# is shared with the ``price_target`` model. Flipping ``pymc_role`` to 'excluded'
# in ``pml_df_feature_alias`` would drop them from ``vw_pymc_feature_catalogue``
# while the MV still emits them, which makes
# ``pml.assert_pymc_catalogue_coverage()`` raise MISSING_FROM_CATALOGUE. The
# drift matrix is a Python-side concern and this frozenset is its SSOT.
KALMAN_DRIFT_EXCLUDED_FEATURES: frozenset[str] = (
    KALMAN_LEAKAGE_FEATURES
    | KALMAN_NOISE_WIDENER_FEATURES
    | KALMAN_TILT_FEATURES
    | KALMAN_SIZE_RANK_SIBLING_FEATURES
    | KALMAN_DRIFT_SUPPORT_COUNTERS
    | KALMAN_RATING_COUNT_FEATURES
    | KALMAN_COLLINEAR_COMPOSITION_FEATURES
    | KALMAN_PIOTROSKI_COMPONENT_FEATURES
    | KALMAN_PT_DRIFT_SIBLING_FEATURES
)


class KalmanFilterPriceTarget:
    """Bayesian state-space model for price target filtering.

    Uses GaussianRandomWalk as latent state with HalfNormal priors on
    state and observation noise (scaled to data std).
    """

    def __init__(self) -> None:
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[InferenceLike] = None
        # Fit context retained so :meth:`forecast` can project the fitted
        # local-level process forward to future fiscal events without refitting.
        self._fit_idata_: Optional[InferenceLike] = None
        self._fit_last_price_: Optional[float] = None
        self._fit_has_trend_: bool = False
        self._fit_stochastic_volatility_: bool = False

    @staticmethod
    @lru_cache(maxsize=4)
    def _resolve_kalman_feature_aliases(
            connection_string: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Resolve the Kalman mutable_predictor aliases from the PyMC catalogue.

        Source of truth (pml schema)::

            SELECT feature_alias
            FROM pml.vw_pymc_feature_catalogue
            WHERE model_target = 'kalman_pt' AND pymc_role = 'mutable_predictor'

        These label the ``kalman_feature`` dim of the auxiliary ``pm.Data``
        container of observed Kalman-feature values (price-target dynamics +
        technical-analysis signals).

        Returns a tuple so the result is hashable / cache-friendly; callers
        should convert to ``list`` if mutation is needed. Returns an empty
        tuple on any failure so callers can fall back gracefully.
        """
        try:
            import os

            import pandas as _pd

            from probabilistic_ml_model.data_utils.data_utils import (
                get_analytics_engine,
            )

            try:
                from sqlalchemy import create_engine
            except ImportError:  # pragma: no cover - defensive
                create_engine = None  # type: ignore[assignment]

            sql = (
                "SELECT DISTINCT feature_alias "
                "FROM pml.vw_pymc_feature_catalogue "
                "WHERE model_target = 'kalman_pt' "
                "AND pymc_role = 'mutable_predictor' "
                "ORDER BY feature_alias"
            )
            url = connection_string or os.environ.get("DB_URL")
            if create_engine is not None and url:
                engine = create_engine(url)
            else:
                engine = get_analytics_engine()
            df = _pd.read_sql(sql, engine)
            return tuple(df["feature_alias"].dropna().astype(str).tolist())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load kalman_pt feature catalogue: %s", exc)
            return tuple()

    @classmethod
    def select_drift_features(
            cls,
            feature_aliases: Iterable[str],
            available_columns: Optional[Iterable[str]] = None,
    ) -> list[str]:
        """Filter mutable_predictor aliases down to drift design-matrix inputs.

        Applies the module-level SSOT partition: drops the response-leakage
        alias, the sigma_obs noise wideners, the named risk/size tilt drivers,
        the drift-support counters, the raw analyst-rating counts, the
        collinear composition leg, the per-fiscal-year Piotroski component
        scores (their median ``feat_median_piotroski_f_score`` enters instead),
        and the ``days_*`` time covariates
        (see :data:`KALMAN_DRIFT_EXCLUDED_FEATURES` /
        :data:`KALMAN_TIME_COVARIATE_PREFIX`).

        Parameters
        ----------
        feature_aliases
            Candidate ``mutable_predictor`` aliases (catalogue order is
            preserved; duplicates are dropped).
        available_columns
            When given, only aliases present in this collection survive
            (typically ``kalman_df.columns``).

        Returns
        -------
        list[str]
            Ordered drift-feature aliases for the ``drift_feature`` dim.
        """
        available = set(available_columns) if available_columns is not None else None
        selected: list[str] = []
        seen: set[str] = set()
        for alias in feature_aliases:
            if alias in seen:
                continue
            seen.add(alias)
            if alias in KALMAN_DRIFT_EXCLUDED_FEATURES:
                continue
            if alias.startswith(KALMAN_TIME_COVARIATE_PREFIX):
                continue
            if available is not None and alias not in available:
                continue
            selected.append(alias)
        return selected

    # ------------------------------------------------------------------
    # Snapshot -> per-ISIN price-target time-series helpers.
    #
    # These mirror Section 7.1 of ``pymc_expected_returns_model.ipynb``
    # so the unpivot of the embedded ``*_ago`` cohort is owned by the
    # model class rather than inlined per-caller. They turn a
    # point-in-time snapshot (``mv_all_stock_features``) into a real
    # ``(isin, asof_date, price_target)`` long panel suitable for the
    # GaussianRandomWalk likelihood.
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_reference_now(
            df: pd.DataFrame,
            timestamp_col: str = "feature_calculated_at",
    ) -> Union[Timestamp, NaTType]:
        """Return a deterministic "now" anchor for ``*_ago`` offsetting.

        Uses the snapshot's ``feature_calculated_at`` column when present
        (so reruns of the same snapshot land on identical timestamps);
        falls back to ``pd.Timestamp.utcnow()`` otherwise. The result is
        always tz-naive.
        """
        if timestamp_col in df.columns:
            ref = pd.to_datetime(df[timestamp_col], errors="coerce").max()
        else:
            ref = pd.Timestamp.utcnow()
        if pd.isna(ref):
            ref = pd.Timestamp.utcnow()
        return (
            pd.Timestamp(ref).tz_localize(None) if pd.Timestamp(ref).tzinfo else pd.Timestamp(ref)
        )

    @staticmethod
    def _resolve_fiscal_anchor(
            df: pd.DataFrame,
            id_col: str,
            anchor_col: Optional[str],
    ) -> Optional[pd.Series]:
        """Return a per-ISIN fiscal-calendar anchor date for ``*_ago`` offsetting.

        When ``anchor_col`` (e.g. ``income_statement_report_date``) is present,
        each ISIN's ``*_ago`` observations are measured back from that ISIN's own
        fiscal anchor instead of a single global "now". The most recent
        (``max``) date per ISIN is used so the latest snapshot lands on the last
        actual reporting date.

        Parameters
        ----------
        df
            Wide snapshot frame.
        id_col
            ISIN column name.
        anchor_col
            Fiscal-calendar date column to anchor on. ``None`` disables fiscal
            anchoring (callers fall back to :meth:`_resolve_reference_now`).

        Returns
        -------
        pandas.Series or None
            ISIN-indexed tz-naive anchor timestamps, or ``None`` when the column
            is absent or holds no parseable dates.
        """
        if not anchor_col or anchor_col not in df.columns or id_col not in df.columns:
            return None
        anchors = pd.to_datetime(df[anchor_col], errors="coerce")
        tmp = pd.DataFrame({id_col: df[id_col].to_numpy(), "_anchor": anchors})
        tmp = tmp.dropna(subset=["_anchor"])
        if tmp.empty:
            return None
        s = tmp.groupby(id_col)["_anchor"].max()
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
        return s

    @classmethod
    def build_price_target_history(
            cls,
            df: pd.DataFrame,
            *,
            id_col: str = "isin",
            now_cols: tuple[str, ...] = ("price_target", "last_price"),
            min_observations: int = 2,
            timestamp_col: str = "feature_calculated_at",
            fiscal_anchor_col: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.Series, Optional[str]]:
        """Build a long ``(isin, asof_date, price_target)`` panel from a snapshot.

        Unpivots the ``*_ago`` cohort
        (``price_target``, ``price_target_high|low|median``, ``price``) using
        :data:`AGO_HISTORY_RE` and translates each suffix to a real
        ``pd.Timestamp`` via :func:`_build_ago_offset_map`. The
        point-in-time ``price_target`` / ``last_price`` values are seeded
        at the resolved "now" anchor.

        Parameters
        ----------
        df : pandas.DataFrame
            Wide snapshot frame (e.g. ``mv_all_stock_features``).
        id_col : str
            Name of the ISIN column. Defaults to ``"isin"``.
        now_cols : tuple of str
            Columns whose values represent the current price target /
            last price; seeded at the snapshot anchor.
        min_observations : int
            Minimum number of distinct ``(isin, asof_date)`` rows required
            to consider an ISIN eligible for Kalman fitting.
        timestamp_col : str
            Column name carrying the snapshot's calculation timestamp.
        fiscal_anchor_col : str, optional
            Fiscal-calendar date column (e.g. ``income_statement_report_date``)
            to anchor the ``*_ago`` axis on. When supplied, each ISIN's
            observations are offset from its own fiscal anchor (resolved via
            :meth:`_resolve_fiscal_anchor`) so the time axis tracks the real
            reporting cadence rather than a single global "now". ISINs without a
            parseable anchor fall back to the global ``ref_now``. ``None`` (the
            default) preserves the prior single-anchor behaviour.

        Returns
        -------
        long_df : pandas.DataFrame
            Long-format panel with columns ``[id_col, "asof_date",
            "price_target"]``, sorted by ISIN then ``asof_date``.
        eligible : pandas.Series
            Per-ISIN observation counts for ISINs with
            ``count >= min_observations``.
        date_col : str, optional
            ``"asof_date"`` when the unpivot succeeded; otherwise ``None``.
        """
        offset_map = _build_ago_offset_map()
        ref_now = cls._resolve_reference_now(df, timestamp_col=timestamp_col)
        anchor_by_isin = cls._resolve_fiscal_anchor(df, id_col, fiscal_anchor_col)

        def _anchor_series(ids: pd.Series) -> pd.Series:
            """Per-row anchor timestamps: the ISIN's fiscal anchor, else ``ref_now``."""
            if anchor_by_isin is None:
                return pd.Series(ref_now, index=ids.index)
            return ids.map(anchor_by_isin).fillna(ref_now)

        history_cols: list[tuple[str, str]] = []
        for col in df.columns:
            m = _AGO_HISTORY_PATTERN.match(col)
            if m:
                history_cols.append((col, m.group("suf")))

        frames: list[pd.DataFrame] = []
        # Historical ``*_ago`` snapshots: offset each row from its own anchor so the
        # axis is fiscal-calendar aligned when ``fiscal_anchor_col`` is supplied.
        for col, suf in history_cols:
            off = offset_map.get(suf.lower())
            if off is None:
                continue
            piece = (
                df[[id_col, col]]
                .rename(columns={col: "price_target"})
                .dropna(subset=["price_target"])
            )
            if piece.empty:
                continue
            piece = piece.assign(
                asof_date=_anchor_series(piece[id_col]) - off, source_col=col
            )
            frames.append(piece)
        # "Now" columns seeded at each ISIN's anchor (the latest reporting date when
        # fiscal anchoring is active, else the global ``ref_now``).
        for col in (c for c in now_cols if c in df.columns):
            piece = (
                df[[id_col, col]]
                .rename(columns={col: "price_target"})
                .dropna(subset=["price_target"])
            )
            if piece.empty:
                continue
            piece = piece.assign(asof_date=_anchor_series(piece[id_col]), source_col=col)
            frames.append(piece)

        if not frames:
            empty = pd.DataFrame(columns=[id_col, "asof_date", "price_target"])
            return empty, pd.Series(dtype="int64"), None

        long_df = pd.concat(frames, ignore_index=True)
        long_df = long_df[(long_df["price_target"] > 0) & np.isfinite(long_df["price_target"])]
        # Average duplicate (isin, asof_date) cells across high/low/median/now
        # cohorts so the per-ISIN time axis is strictly monotonic.
        long_df = (
            long_df.groupby([id_col, "asof_date"], as_index=False)["price_target"]
            .mean()
            .sort_values([id_col, "asof_date"])
        )
        counts = long_df.groupby(id_col).size()
        eligible = counts[counts >= min_observations]
        return long_df, eligible, "asof_date"

    @staticmethod
    def select_target_isin(
            eligible: pd.Series,
            cohort: Optional[Any] = None,
    ) -> Optional[Hashable]:
        """Pick the ISIN with the most history, optionally filtered by ``cohort``.

        ``cohort`` is any iterable of ISIN strings (e.g. the index of the
        PriceTarget model's ``pt_df``). When the intersection is empty,
        falls back to the global argmax over ``eligible``.
        """
        if eligible is None or eligible.empty:
            return None
        if cohort is not None:
            cohort_set = {str(x) for x in cohort}
            candidates = eligible[eligible.index.astype(str).isin(cohort_set)]
            if not candidates.empty:
                return candidates.idxmax()
        return eligible.idxmax()

    @staticmethod
    def _prepare_log_targets(
            price_targets: np.ndarray,
            dates: Optional[pd.DatetimeIndex],
            realized_vol: Optional[np.ndarray] = None,
    ) -> tuple[
        np.ndarray, np.ndarray, float, Optional[pd.DatetimeIndex], Optional[np.ndarray]
    ]:
        """Validate, clean, and log-transform the observed price targets.

        Drops non-finite observations (and the aligned ``dates`` entries),
        verifies strict positivity, and derives a clamped log-space scale
        for the noise priors.

        Parameters
        ----------
        price_targets : numpy.ndarray
            Raw observed price target series (length >= 2).
        dates : pandas.DatetimeIndex, optional
            Time index aligned to ``price_targets``; filtered alongside it.
        realized_vol : numpy.ndarray, optional
            Realized-volatility anchor aligned 1:1 with ``price_targets`` (used
            only by the stochastic-volatility path). Co-filtered with the same
            finite mask so it stays aligned to the cleaned series by
            construction. Must be strictly positive and finite after filtering.

        Returns
        -------
        tuple
            ``(pt_arr, log_pt, scale, dates, rv)`` — the cleaned positive
            series, its log transform, the clamped prior scale, the (possibly
            filtered) ``dates``, and the co-filtered ``realized_vol`` (or
            ``None`` when not supplied).

        Raises
        ------
        ValueError
            If fewer than 2 (finite) observations remain, any value is
            non-positive, or ``realized_vol`` is misaligned / non-positive.
        """
        pt_arr = np.asarray(price_targets, dtype="float64")
        rv: Optional[np.ndarray] = None
        if realized_vol is not None:
            rv = np.asarray(realized_vol, dtype="float64")
            if rv.size != pt_arr.size:
                raise ValueError(
                    "realized_vol must align 1:1 with price_targets "
                    f"(got {rv.size} vs {pt_arr.size})."
                )
        if pt_arr.size < 2:
            raise ValueError("price_target must have length ≥ 2.")

        # Drop non-finite observations to avoid propagating NaN/Inf into the
        # GaussianRandomWalk likelihood (a common overflow trigger).
        finite_mask = np.isfinite(pt_arr)
        if finite_mask.sum() < 2:
            raise ValueError(
                "price_target must contain at least 2 finite observations."
            )
        if not finite_mask.all():
            logger.warning(
                "KalmanFilterPriceTarget: dropping %d non-finite price target obs",
                int((~finite_mask).sum()),
            )
            pt_arr = pt_arr[finite_mask]
            if dates is not None:
                dates = pd.DatetimeIndex(np.asarray(dates)[finite_mask])
            if rv is not None:
                rv = rv[finite_mask]

        # Price targets are strictly positive ⇒ model on log-scale.  This keeps
        # the latent state and noise priors dimensionless and bounded,
        # preventing the float64 overflow observed when raw price levels (often
        # 10²–10³) are combined with a HalfNormal(σ=std(price)) prior on a
        # GaussianRandomWalk: cumulative variance ≈ T·σ² overflows the
        # likelihood for moderate T.  Working in log-space, σ represents
        # log-returns and is naturally O(0.1).
        if np.any(pt_arr <= 0):
            raise ValueError(
                "price_target must be strictly positive for log-space Kalman filter."
            )
        log_pt = np.log(pt_arr)

        # Clamp scale to a sane band so degenerate inputs (constant series, or
        # a single outlier driving std to ∞) cannot blow up the priors.
        raw_scale = float(np.nanstd(log_pt))
        scale = float(np.clip(raw_scale if np.isfinite(raw_scale) else 0.0, 1e-3, 1.0))

        # The stochastic-volatility anchor enters the log-volatility prior mean
        # via ``log(realized_vol)``, so it must be strictly positive and finite
        # after co-filtering.
        if rv is not None and (not np.isfinite(rv).all() or np.any(rv <= 0)):
            raise ValueError(
                "realized_vol must be strictly positive and finite after "
                "aligning with the cleaned price target series."
            )
        return pt_arr, log_pt, scale, dates, rv

    @staticmethod
    def implied_upside_from_state(
            state: np.ndarray,
            last_price: float,
    ) -> np.ndarray:
        """Return the implied upside of a (smoothed) price state vs spot.

        Mirrors the SQL ``pml.calc_change_ratio(price_target, last_price)``
        feature ``feat_implied_upside`` and the cross-sectional notebook's
        ``expected_upside`` so the single-ISIN time-series model reports the
        same, directly-comparable quantity:

        .. math:: \\text{implied\\_upside} = \\frac{\\text{state}}{\\text{last\\_price}} - 1.

        Parameters
        ----------
        state : numpy.ndarray
            Price-space latent state (e.g. posterior-mean ``state``).
        last_price : float
            Reference spot price; must be strictly positive.

        Returns
        -------
        numpy.ndarray
            Implied-upside ratio aligned to ``state``. Returns an all-NaN array
            when ``last_price`` is non-finite or non-positive.
        """
        arr = np.asarray(state, dtype="float64")
        if not np.isfinite(last_price) or last_price <= 0:
            return np.full_like(arr, np.nan)
        return arr / float(last_price) - 1.0

    @staticmethod
    def _resolve_coords(
            pt_arr: np.ndarray,
            dates: Optional[pd.DatetimeIndex],
            isin: Optional[str],
            sectors: Optional[np.ndarray],
            categories_df: Optional[pd.DataFrame],
            hierarchy_levels: Optional[list[str]],
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Assemble PyMC model coords and the singleton ``isin`` index.

        Registers a ``time`` coord (from ``dates`` or a positional range), an
        ``isin`` coord (mirroring ``pml.vw_pml_df_coords`` where
        ``column_name = 'isin'``), and any optional category-hierarchy coords
        via :func:`coerce_categories`.

        Returns
        -------
        tuple
            ``(coords, isins_arr)``.
        """
        T = len(pt_arr)
        time_coords = dates if dates is not None else np.arange(T, dtype=np.int64)
        coords: dict[str, Any] = {"time": time_coords}

        # `isin` mirrors pml.vw_pml_df_coords (column_name='isin',
        # pymc_role='coord'). When ``isin`` is not supplied we fall back to a
        # singleton positional index so the optional category-hierarchy coords
        # can still be wired in.
        if isin is not None:
            isins_arr = np.asarray([isin])
        else:
            isins_arr = np.arange(1, dtype="int64")
        coords["isin"] = isins_arr

        # Optional category hierarchy registers coords for downstream pivots.
        cats_df, levels = coerce_categories(
            isins_arr,
            sectors=sectors,
            categories_df=categories_df,
            hierarchy_levels=hierarchy_levels
                             or (["sector", "industry"] if categories_df is not None else None),
        )
        if cats_df is not None and levels:
            hierarchy_meta = build_hierarchy_indices(cats_df, isins_arr, levels=levels)
            for lv, meta in hierarchy_meta.items():
                coords[lv] = meta["labels"]
        return coords, isins_arr

    @staticmethod
    def _resolve_time_deltas(
            dates: Optional[pd.DatetimeIndex],
            n_obs: int,
    ) -> np.ndarray:
        """Return a non-decreasing elapsed-time vector (years) anchored at zero.

        The marginalized formulation scales the random-walk innovation variance
        by *real* spacing between observations (the ``*_ago`` cohort is highly
        irregular — 1w, 1m, 3m, 6m, 1y). This converts ``dates`` into cumulative
        elapsed years :math:`\\tau` with :math:`\\tau_0 = 0`, used to build the
        Wiener-process covariance kernel :math:`\\min(\\tau_s, \\tau_t)`.

        Falls back to unit steps (``0, 1, …, n-1``) when ``dates`` are absent,
        contain NaT, or are degenerate (all identical) so the kernel is still
        well defined.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, optional
            Observation timestamps aligned to the cleaned price series.
        n_obs : int
            Number of (finite, positive) observations.

        Returns
        -------
        numpy.ndarray
            Float64 elapsed-time vector of length ``n_obs``.
        """
        if dates is not None and len(dates) == n_obs and n_obs > 1:
            idx = pd.DatetimeIndex(dates)
            if not idx.hasnans:
                # Subtract in datetime space (exact) then convert the small
                # deltas to float days — converting the raw nanosecond epoch to
                # float64 first would lose day-level precision (~1e18 ns).
                days = ((idx - idx[0]) / pd.Timedelta(days=1)).to_numpy(dtype="float64")
                tau = np.maximum.accumulate(days) / 365.25
                if np.isfinite(tau).all() and tau[-1] > 0:
                    return tau
        return np.arange(n_obs, dtype="float64")

    @staticmethod
    def _build_log_state(
            parameterization: Parameterization,
            log_pt: np.ndarray,
            sigma_state: Any,
            scale: float,
            tau: np.ndarray,
            init_mu: Optional[float] = None,
            beta_trend: Any = None,
    ) -> Any:
        """Build the explicit latent log-price state (non-marginalized forms).

        Must be called inside an active ``pm.Model`` context. The
        ``marginalized`` parameterization does *not* go through this helper —
        it integrates the latent path out analytically in
        :meth:`_build_marginalized_likelihood`.

        - ``centered`` — an explicit :class:`pymc.GaussianRandomWalk` (registered
          as ``log_level``).
        - ``non_centered`` (default) — ``init + cumsum(sigma_state * z)``.

        When ``beta_trend`` is supplied, a deterministic linear trend
        ``beta_trend * tau`` is added to the latent mean (the reference notebook's
        structural-trend fix that stops the projection decaying to a flat
        baseline). The combined latent path is always exposed as the ``log_state``
        Deterministic so downstream consumers resolve uniformly.

        Parameters
        ----------
        parameterization : Parameterization
            ``"centered"`` or ``"non_centered"``.
        log_pt : numpy.ndarray
            Cleaned log-price-target series (its first element seeds the
            initial-level prior when ``init_mu`` is not supplied).
        sigma_state : pytensor tensor
            Random-walk (process) noise scale.
        scale : float
            Clamped log-scale for the diffuse initial-level prior.
        tau : numpy.ndarray
            Elapsed-time vector (years) from :meth:`_resolve_time_deltas`; scales
            the optional ``beta_trend`` term.
        init_mu : float, optional
            Initial latent log-level anchor. Defaults to ``log_pt[0]``; pass
            ``log(last_price)`` to anchor the smoother to the current spot
            price instead of the first observed target.
        beta_trend : pytensor tensor, optional
            Per-year log-price trend slope. ``None`` (default) omits the trend.
        """
        anchor = float(log_pt[0]) if init_mu is None else float(init_mu)
        if parameterization == "centered":
            state_mean = pm.GaussianRandomWalk(
                "log_level",
                sigma=sigma_state,
                init_dist=pm.Normal.dist(mu=anchor, sigma=scale),
                dims="time",
            )
        else:
            # Non-centred GRW: state = init + cumsum(sigma_state * z).
            z_innov = pm.Normal("z_innov", 0.0, 1.0, dims="time")
            state_mean = anchor + pt.cumsum(sigma_state * z_innov)
        if beta_trend is not None:
            tau_t = pt.as_tensor_variable(np.asarray(tau, dtype="float64"))
            state_mean = state_mean + beta_trend * tau_t
        return pm.Deterministic("log_state", state_mean, dims="time")

    @staticmethod
    def _build_stochastic_volatility(
            log_state: Any,
            log_obs_data: Any,
            scale: float,
            realized_vol_anchor: Optional[np.ndarray] = None,
    ) -> Any:
        """Build a time-varying (stochastic-volatility) observation likelihood.

        Replaces the scalar ``sigma_obs`` with a latent log-volatility random
        walk over ``time`` and a robust Student-t observation model, following
        the canonical PyMC stochastic-volatility example (a
        :class:`pymc.GaussianRandomWalk` in log-volatility, ``Exponential``
        step-size and degrees-of-freedom priors, ``StudentT`` observations with
        ``sigma = exp(log_vol)``). Must be called inside an active ``pm.Model``
        context, *after* ``log_state`` has been built by
        :meth:`_build_log_state`.

        This is fundamentally **incompatible** with the ``marginalized``
        closed-form likelihood (:meth:`_build_marginalized_likelihood`), which
        integrates the latent path out into an :class:`pymc.MvNormal` whose
        covariance diagonal needs a *scalar* observation variance. The caller
        (:meth:`fit`) therefore forces an explicit-state parameterization
        whenever stochastic volatility is requested.

        The latent log-volatility walk is **non-centred**
        (``cumsum(step_size * z_vol)`` with ``z_vol ~ Normal(0, 1)``) to avoid
        the funnel, mirroring the non-centred level walk in
        :meth:`_build_log_state`.

        Parameters
        ----------
        log_state : pytensor tensor
            Latent log-price level (the observation mean), dims ``time``.
        log_obs_data : pytensor tensor
            The :class:`pymc.Data` container holding the observed log-targets,
            dims ``time``.
        scale : float
            Clamped log-scale (from :meth:`_prepare_log_targets`). Sets the
            constant fallback log-volatility anchor ``log(scale)`` when no
            realized-volatility anchor is supplied.
        realized_vol_anchor : numpy.ndarray, optional
            Strictly-positive per-time realized-volatility anchor, already
            aligned to ``time`` (co-filtered with the price series in
            :meth:`_prepare_log_targets`). When supplied, only its **shape**
            (the deviations of ``log(realized_vol)`` from its own mean) informs
            the per-time prior *mean* of the log-volatility walk; the absolute
            *level* is a learned ``vol_anchor_offset ~ Normal(log(scale), 1)``.
            This deliberately makes the anchor **scale-invariant** — realized
            volatility may arrive in percent units (e.g. ≈ 40) while
            the observation noise lives on the log-price-target scatter scale
            (≈ 0.05–0.2), so equating absolute levels would be nonsensical; the
            term-structure still carries the relative time variation. When
            ``None``, the walk is anchored at the constant ``log(scale)``.

        Returns
        -------
        pytensor tensor
            The ``StudentT`` observed variable ``obs``. Side effects register
            the ``vol_step_size``, ``z_vol``, ``nu_obs`` (and, when anchored,
            ``vol_anchor_offset`` — the learned absolute log-volatility level)
            priors and the ``log_vol`` / ``sigma_obs`` Deterministics (both
            over ``time``).

        Notes
        -----
        ``sigma_obs`` is deliberately reused as the Deterministic name (now
        carrying a ``time`` dim rather than being scalar) so existing
        diagnostics and :meth:`forecast` resolve it uniformly; ``forecast``
        sniffs the ``time`` dim and holds the terminal volatility flat.
        """
        # Non-centred latent log-volatility random walk (funnel-free; mirrors
        # the non-centred level walk in _build_log_state).
        step_size = pm.Exponential("vol_step_size", 10.0)
        z_vol = pm.Normal("z_vol", 0.0, 1.0, dims="time")
        log_vol_innov = pt.cumsum(step_size * z_vol)

        # Per-time prior-mean anchor for log-volatility.
        if realized_vol_anchor is not None:
            log_rv = np.log(np.asarray(realized_vol_anchor, dtype="float64"))
            # Use only the *shape* (deviations from the mean) so the anchor is
            # scale-invariant — realized vol may be in percent while the noise
            # lives on the log-price-target scatter scale. The absolute level is
            # learned and anchored at log(scale) (the non-SV scalar level).
            log_rv_shape = pt.as_tensor_variable(log_rv - float(np.mean(log_rv)))
            vol_anchor_offset = pm.Normal("vol_anchor_offset", float(np.log(scale)), 1.0)
            log_vol_mean: Any = vol_anchor_offset + log_rv_shape
        else:
            log_vol_mean = float(np.log(scale))

        log_vol = pm.Deterministic("log_vol", log_vol_mean + log_vol_innov, dims="time")
        sigma_obs_t = pm.Deterministic("sigma_obs", pm.math.exp(log_vol), dims="time")

        # Robust heavy-tailed observation with time-varying scale.
        nu = pm.Exponential("nu_obs", 0.1)
        return pm.StudentT(
            "obs",
            nu=nu,
            mu=log_state,
            sigma=sigma_obs_t,
            observed=log_obs_data,
            dims="time",
        )

    @staticmethod
    def _build_marginalized_likelihood(
            log_obs_data: Any,
            log_pt: np.ndarray,
            sigma_state: Any,
            sigma_obs: Any,
            scale: float,
            tau: np.ndarray,
            init_mu: Optional[float] = None,
            beta_trend: Any = None,
    ) -> Any:
        """Integrate the latent random walk out into the likelihood covariance.

        Must be called inside an active ``pm.Model`` context. Implements a true
        *marginalized* local-level (integrated Wiener) state-space filter: the
        latent log-price path :math:`x_{1:T}` is collapsed analytically, leaving
        a single multivariate-normal likelihood for the observed log-targets
        whose covariance carries **both** the random-walk (process) variance and
        the observation (measurement) variance.

        For a local-level model with continuous-time process noise,

        .. math::

            x_t = x_1 + W(\\tau_t), \\qquad y_t = x_t + \\varepsilon_t,

        marginalizing :math:`x` yields :math:`y \\sim \\mathcal{N}(\\mu_0\\mathbf{1},\\;\\Sigma)`
        with

        .. math::

            \\Sigma_{st} = P_0 + \\sigma_{\\text{state}}^2 \\min(\\tau_s, \\tau_t)
                          + \\sigma_{\\text{obs}}^2 \\, \\delta_{st}.

        Unlike the previous implementation, ``log_pt`` is used **only** as the
        observed series (never as the mean), so the random-walk and observation
        variances are genuinely identified by the data.

        Parameters
        ----------
        log_obs_data : pytensor tensor
            The :class:`pymc.Data` container holding the observed log-targets.
        log_pt : numpy.ndarray
            Cleaned log-price-target series (used only for static shape / seed).
        sigma_state, sigma_obs : pytensor tensor
            Random-walk (process) and observation (measurement) noise scales.
        scale : float
            Clamped log-scale used for the diffuse initial-level variance/prior.
        tau : numpy.ndarray
            Elapsed-time vector (years) from :meth:`_resolve_time_deltas`.
        init_mu : float, optional
            Initial latent log-level prior mean. Defaults to ``log_pt[0]``; pass
            ``log(last_price)`` to anchor the smoother to the current spot price.

        Returns
        -------
        pytensor tensor
            The analytic smoother mean :math:`\\mathbb{E}[x \\mid y]`, registered
            as the ``log_state`` Deterministic so downstream consumers (the
            ``state`` Deterministic and path plots) continue to resolve.
        """
        from pytensor.tensor import linalg as pt_linalg

        n = int(len(log_pt))
        tau_t = pt.as_tensor_variable(np.asarray(tau, dtype="float64"))
        # Wiener-process kernel: Cov(x_s, x_t) = P0 + sigma_state^2 * min(tau_s, tau_t).
        min_tau = pt.minimum(tau_t[:, None], tau_t[None, :])
        eye = pt.eye(n)
        p0 = float(scale) ** 2  # diffuse initial-level variance
        anchor = float(log_pt[0]) if init_mu is None else float(init_mu)
        mu0 = pm.Normal("log_state_init", mu=anchor, sigma=float(scale))
        state_cov = p0 + pt.sqr(sigma_state) * min_tau
        # Marginal observation covariance = signal covariance + measurement noise
        # on the diagonal (+ tiny jitter for numerical positive-definiteness).
        obs_cov = state_cov + (pt.sqr(sigma_obs) + 1e-6) * eye
        # Local-level mean is the diffuse initial level, plus an optional linear
        # trend in elapsed time (the structural-trend term from the reference).
        mu_vec = mu0 * pt.ones(n)
        if beta_trend is not None:
            mu_vec = mu_vec + beta_trend * tau_t
        pm.MvNormal("obs", mu=mu_vec, cov=obs_cov, observed=log_obs_data, dims="time")

        # Analytic Kalman/RTS smoother mean E[x | y] = mu + K Sigma^{-1} (y - mu),
        # where K (signal covariance) = Cov(x, y) since the measurement noise is
        # independent of the state. Exposed as `log_state` for downstream plots.
        resid = log_obs_data - mu_vec
        smoothed = mu_vec + pt.dot(
            state_cov, pt_linalg.solve(obs_cov, resid, assume_a="pos", b_ndim=1)
        )
        return pm.Deterministic("log_state", smoothed, dims="time")

    def fit(
            self,
            price_targets: np.ndarray,
            isin: Optional[str] = None,
            dates: Optional[pd.DatetimeIndex] = None,
            last_price: Optional[float] = None,
            sectors: Optional[np.ndarray] = None,
            categories_df: Optional[pd.DataFrame] = None,
            hierarchy_levels: Optional[list[str]] = None,
            samples: int = _DEFAULT_SAMPLES,
            tune: int = _DEFAULT_TUNE,
            chains: int = _DEFAULT_CHAINS,
            target_accept: float = _DEFAULT_TARGET_ACCEPT,
            random_seed: int = _DEFAULT_RANDOM_SEED,
            trend: bool = False,
            trend_sigma: float = 0.5,
            stochastic_volatility: bool = False,
            realized_vol: Optional[np.ndarray] = None,
            parameterization: Parameterization = "auto",
            nuts_sampler: Optional[str] = None,
            **sample_kwargs: Any,
    ) -> tuple[Union[InferenceLike, "MultiTrace"], "pm_typing.Model"]:
        """
        Fit a state-space model to the provided price targets using a Kalman filter
        approach in log-space. The function handles inference via PyMC and returns
        both the inference data object and model instance.

        Parameters
        ----------
        price_targets : numpy.ndarray
            An array of observed price target values, must have at least 2 elements.
            Values must be finite and strictly positive, as the Kalman filter operates
            in log-space.
        isin : str, optional
            International Securities Identification Number (ISIN) associated with the
            price targets. If provided, it is included in model metadata.
        dates : pandas.DatetimeIndex, optional
            Time index corresponding to `price_targets`. If provided, it will be used
            for naming and plotting axes. Non-finite dates will be dropped alongside
            corresponding price targets.
        last_price : float, optional
            Current spot (last) price. When supplied and strictly positive it
            (a) anchors the latent log-level prior at ``log(last_price)`` instead
            of the first observed target — so the smoother is filtered relative to
            spot — and (b) exposes an ``implied_upside`` Deterministic
            (``state / last_price - 1``) over the ``time`` dim, mirroring the SQL
            ``feat_implied_upside`` feature and the cross-sectional model's
            ``expected_upside``. When ``None`` the model is unchanged.
        sectors : numpy.ndarray, optional
            Optional 1-D array of sector labels aligned to the (singleton) ``isin``
            coord. Forwarded to :func:`coerce_categories` so the model registers a
            ``sector`` coord even when ``categories_df`` is not supplied.
        categories_df : pandas.DataFrame, optional
            Optional ISIN-indexed (or ISIN-columned) frame carrying category columns
            (``sector``, ``industry``, …). When provided, its columns become
            additional model coords keyed by hierarchy level so downstream
            consumers can pivot the latent state by sector / industry.
        hierarchy_levels : list of str, optional
            Subset of column names from ``categories_df`` to register as coords.
            Defaults to ``["sector", "industry"]`` when ``categories_df`` is
            supplied and this argument is omitted.
        samples : int
            Number of posterior samples to draw during the MCMC process. Default is
            2000.
        tune : int
            Number of tuning steps before sampling during MCMC inference. The tuned
            values are not included in the returned samples. Default is 1000.
        chains : int
            Number of independent MCMC chains to run. Default is 4.
        target_accept : float
            Target acceptance rate for the NUTS sampler, influencing the step-size
            adaptation during sampling. Default is 0.9.
        random_seed : int
            Seed for random number generation to ensure reproducibility. Default is 42.
        trend : bool
            When ``True``, add a deterministic linear trend ``beta_trend * tau``
            (per-year log-price slope) to the latent state mean. This is the
            structural-trend component from the *Forecasting with Structural AR
            Timeseries* reference that prevents :meth:`forecast` projections from
            decaying to a flat baseline. Default ``False`` (driftless local level).
        trend_sigma : float
            Prior scale for the ``beta_trend ~ Normal(0, trend_sigma)`` slope (in
            log-price units per year). Only used when ``trend=True``. Default 0.5.
        stochastic_volatility : bool
            When ``True``, replace the single scalar observation-noise prior with
            a latent log-volatility random walk over ``time`` and a robust
            Student-t observation model (see
            :meth:`_build_stochastic_volatility`), following the canonical PyMC
            stochastic-volatility example. Opt-in and default ``False`` — the
            scalar-noise behaviour is otherwise byte-for-byte unchanged. SV is
            incompatible with the ``marginalized`` closed-form, so a resolved
            ``marginalized`` parameterization (including via ``"auto"`` on short
            series) is overridden to ``"non_centered"`` with a warning, and
            ``target_accept`` is bumped to ``0.95`` when left at its ``0.9``
            default (the augmented latent geometry samples harder). ``"auto"``
            never selects SV on its own.
        realized_vol : numpy.ndarray, optional
            Strictly-positive realized-volatility anchor aligned 1:1 with
            ``price_targets``. Only consulted when ``stochastic_volatility=True``.
            It is co-filtered with the same finite mask applied to
            ``price_targets`` and its log sets the per-time prior *mean* of the
            log-volatility walk (with a single learned ``vol_anchor_offset`` to
            reconcile scales). When ``None``, the walk is anchored at the
            constant log-scale, matching the previous scalar prior's level.
        parameterization : Parameterization
            Model parameterization approach used for the latent state. Options are:
            "auto", "non_centered", "centered", or "marginalized". Default is "auto",
            which selects the funnel-free "marginalized" form for short series
            (fewer than ``_SHORT_SERIES_THRESHOLD`` observations) and
            "non_centered" otherwise.

            - "centered" / "non_centered" sample an explicit latent
              :class:`pymc.GaussianRandomWalk` path with a diagonal-Normal
              observation likelihood.
            - "marginalized" integrates the latent path out analytically into an
              :class:`pymc.MvNormal` likelihood whose covariance carries both the
              random-walk (process) and observation (measurement) variances,
              scaled by the real elapsed time between observations. The smoothed
              latent path is recovered as the analytic Kalman-smoother mean. This
              is funnel-free and exact for short, irregular ``*_ago`` cohorts.
        nuts_sampler : str, optional
            Specific NUTS sampler to use. If None, the default sampler provided by PyMC
            is used.
        **sample_kwargs : Any
            Additional keyword arguments passed to the `pm.sample` method. Overrides
            defaults set internally.

        Returns
        -------
        tuple[InferenceData | MultiTrace, Any]
            A tuple containing:
            - The resulting inference data object (ArviZ InferenceData or PyMC
              MultiTrace depending on the sampling method).
            - The PyMC model object representing the state-space Kalman filter.

        Raises
        ------
        ImportError
            If the PyMC library is not available.
        ValueError
            If the `price_targets` array contains fewer than 2 values, fewer than 2
            finite values, or any non-positive values.

        Notes
        -----
        - The Kalman filter operates in log-space to ensure numerical stability and
          avoid overflow errors when working with strictly positive observed price
          targets.
        - Input price targets are filtered to remove non-finite values (NaN or Inf)
          before building the model. A warning is logged indicating how many
          observations were dropped.
        - Observations with constant or near-constant series will have their scale
          constrained to prevent degenerate prior configurations.
        """
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install pymc + arviz to use KalmanFilterPriceTarget."
            )

        pt_arr, log_pt, scale, dates, rv_anchor = self._prepare_log_targets(
            price_targets, dates, realized_vol if stochastic_volatility else None
        )

        # Optional spot-price anchor: when a strictly-positive ``last_price`` is
        # supplied the latent log-level prior is centred at ``log(last_price)``
        # (so the smoother is filtered relative to spot rather than the first
        # observed target), and an ``implied_upside`` Deterministic is exposed.
        has_last_price = last_price is not None and np.isfinite(last_price) and last_price > 0
        init_mu = float(np.log(last_price)) if has_last_price else None

        # Resolve ``"auto"`` to the funnel-free ``marginalized`` parameterization
        # for short ``*_ago`` cohorts (where the explicit latent random walk is
        # poorly identified and funnels) and to ``non_centered`` otherwise. The
        # ``marginalized`` form is now a genuine integrated-out GRW (see
        # :meth:`_build_marginalized_likelihood`) — it analytically collapses the
        # latent path into an MvNormal likelihood, so selecting it for short
        # series is statistically correct rather than the prior no-op.
        if parameterization == "auto":
            parameterization = (
                "marginalized" if len(pt_arr) < _SHORT_SERIES_THRESHOLD else "non_centered"
            )
            logger.info(
                "KalmanFilterPriceTarget: auto-selected %r parameterization for %d obs.",
                parameterization,
                len(pt_arr),
            )

        # Stochastic volatility needs an explicit latent state (its time-varying
        # observation noise cannot be folded into the marginalized closed-form's
        # scalar covariance diagonal). Override marginalized — including when
        # "auto" resolved to it for a short series — to the funnel-free
        # non-centred path.
        if stochastic_volatility and parameterization == "marginalized":
            logger.warning(
                "KalmanFilterPriceTarget: stochastic_volatility is incompatible "
                "with the marginalized closed-form likelihood (needs a scalar "
                "sigma_obs); overriding parameterization to 'non_centered'."
            )
            parameterization = "non_centered"

        # The augmented (level + log-volatility) latent geometry samples harder;
        # nudge the target acceptance up when the caller left it at the default.
        if stochastic_volatility and target_accept == _DEFAULT_TARGET_ACCEPT:
            target_accept = 0.95

        # Elapsed-time vector for the (optionally marginalized) GRW covariance —
        # scales process variance by real spacing across the irregular cohort.
        tau = self._resolve_time_deltas(dates, len(pt_arr))

        coords, _ = self._resolve_coords(
            pt_arr, dates, isin, sectors, categories_df, hierarchy_levels
        )

        with pm.Model(coords=coords) as model:
            # Store the raw price target series for downstream consumers /
            # `pm.set_data` swaps; the model itself operates on log-prices.
            pm.Data("price_target", pt_arr, dims="time")
            log_obs_data = pm.Data("log_price_target", log_pt, dims="time")

            # Mode-anchored, weakly-informative priors keep both variances away
            # from 0 and break the sigma_state/sigma_obs ridge that produces the
            # funnel (divergences, inflated r-hat, tiny ESS) under sparse data.
            # log-returns are naturally O(0.05-0.2), so anchor sigma_state there;
            # sigma_obs is anchored to the observed log-scale scatter.
            sigma_state = pm.Gamma("sigma_state", mu=0.10, sigma=0.05)
            # In the stochastic-volatility path the observation noise is a latent
            # per-time process (built below), so the scalar prior is omitted.
            if not stochastic_volatility:
                sigma_obs = pm.Gamma("sigma_obs", mu=scale, sigma=scale)

            # Optional structural trend: a per-year log-price slope shared by the
            # likelihood mean and recovered by :meth:`forecast` so projections
            # carry directional structure (reference: "Specifying a Trend Model").
            beta_trend = pm.Normal("beta_trend", 0.0, float(trend_sigma)) if trend else None

            if stochastic_volatility:
                # Explicit latent level (marginalized already overridden above) +
                # a latent log-volatility random walk feeding a robust Student-t
                # observation model with time-varying scale.
                log_state = self._build_log_state(
                    parameterization, log_pt, sigma_state, scale, tau,
                    init_mu=init_mu, beta_trend=beta_trend,
                )
                self._build_stochastic_volatility(
                    log_state, log_obs_data, scale, realized_vol_anchor=rv_anchor,
                )
            elif parameterization == "marginalized":
                # Integrate the latent path out: the GRW (process) and
                # observation (measurement) variances both enter the MvNormal
                # likelihood covariance. ``log_state`` is the analytic smoother
                # mean, so the per-time latent series is recovered without an
                # explicit (funnel-prone) random walk.
                log_state = self._build_marginalized_likelihood(
                    log_obs_data, log_pt, sigma_state, sigma_obs, scale, tau,
                    init_mu=init_mu, beta_trend=beta_trend,
                )
            else:
                log_state = self._build_log_state(
                    parameterization, log_pt, sigma_state, scale, tau,
                    init_mu=init_mu, beta_trend=beta_trend,
                )
                pm.Normal(
                    "obs",
                    mu=log_state,
                    sigma=sigma_obs,
                    observed=log_obs_data,
                    dims="time",
                )

            # Expose the latent state in the original price space as a
            # Deterministic so downstream code that referenced ``state``
            # continues to work without overflow risk.
            state = pm.Deterministic("state", pm.math.exp(log_state), dims="time")

            # Implied upside vs spot: state / last_price - 1. Mirrors the SQL
            # ``feat_implied_upside`` feature and the cross-sectional model's
            # ``expected_upside`` so both Kalman variants report the same metric.
            if has_last_price:
                pm.Data("last_price", float(last_price))
                pm.Deterministic(
                    "implied_upside", state / float(last_price) - 1.0, dims="time"
                )

            idata = pm.sample(
                **build_sample_kwargs(
                    samples=samples,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=random_seed,
                    nuts_sampler=nuts_sampler,
                    sample_kwargs=sample_kwargs,
                    model_name="KalmanFilterPriceTarget",
                )
            )

        # Store model metadata for downstream consumers
        if isin is not None:
            try:
                model.name = f"KalmanFilter[{isin}]"
            except Exception as exc:
                logger.debug("Could not set model name for %r: %s", isin, exc)

        # Surface sampler-quality diagnostics so the model self-reports funnel
        # problems (divergences / low ESS) instead of relying on console scraping.
        log_sample_diagnostics(
            idata, model_name="KalmanFilterPriceTarget", tag=isin
        )

        self.model_ = model
        # Detect ArviZ InferenceData (or arviz-base xarray.DataTree per migration
        # guide) at runtime via the shim's typing alias. Falls back to MultiTrace
        # detection so external NUTS samplers that bypass ArviZ still work.
        _az_inference = getattr(az, "InferenceData", None) if az is not None else None
        self.idata_ = (
            idata
            if (_az_inference is not None and isinstance(idata, _az_inference))
            else None
        )

        # Retain fit context so :meth:`forecast` can project forward to future
        # fiscal events. ``_fit_idata_`` keeps whatever has a ``posterior`` group
        # (InferenceData or xarray.DataTree) regardless of the MultiTrace check above.
        self._fit_idata_ = idata if hasattr(idata, "posterior") else None
        self._fit_last_price_ = float(last_price) if has_last_price else None
        self._fit_has_trend_ = bool(trend)
        self._fit_stochastic_volatility_ = bool(stochastic_volatility)
        return idata, model

    @staticmethod
    def build_forecast_specs(
            df: pd.DataFrame,
            last_obs: Any,
            *,
            horizons: Sequence[FiscalHorizon] = FISCAL_HORIZONS,
            aggregate: Literal["first", "median"] = "first",
    ) -> tuple[list[int], list[Timestamp], list[str]]:
        """Resolve future fiscal events into :meth:`forecast` inputs + human labels.

        Walks the canonical :data:`FISCAL_HORIZONS` mapping (the single source of
        truth = the ``days_*`` aliases on ``pml.mv_pymc_kalman_pt``) and, for each
        fiscal-calendar DATE column present in ``df`` whose (aggregated) value
        falls strictly after ``last_obs``, emits the day-offset beyond
        ``last_obs``, the absolute future date, and the **human label** — exactly
        the ``horizons_days`` / ``fiscal_dates`` / ``labels`` triple
        :meth:`forecast` consumes. This replaces the per-call-site, hand-built
        ``(date_col, label)`` tuples (which historically drifted out of sync, e.g.
        labelling ``next_income_statement_report_date`` as ``next_fy_end_date``).

        Parameters
        ----------
        df : pandas.DataFrame
            Snapshot frame (or single-row slice) carrying the fiscal-calendar
            DATE columns. Use ``aggregate="first"`` for a single-ISIN row and
            ``aggregate="median"`` for a cohort.
        last_obs : pandas.Timestamp
            Last observed as-of date; horizons are measured beyond it and any
            event on/before it is dropped.
        horizons : Sequence[FiscalHorizon]
            Subset / ordering of :data:`FISCAL_HORIZONS` to consider. Defaults to
            the full ordered set.
        aggregate : {"first", "median"}
            ``"first"`` takes the first non-null date per column (single ISIN);
            ``"median"`` takes the cohort-median date.

        Returns
        -------
        tuple[list[int], list[pandas.Timestamp], list[str]]
            ``(horizons_days, fiscal_dates, labels)``, aligned 1:1 and ordered as
            ``horizons``. All three are empty when no future event qualifies.
        """
        anchor = pd.Timestamp(last_obs)
        horizons_days: list[int] = []
        fiscal_dates: list[Timestamp] = []
        labels: list[str] = []
        for fh in horizons:
            if fh.date_col not in df.columns:
                continue
            parsed = pd.to_datetime(df[fh.date_col], errors="coerce").dropna()
            if parsed.empty:
                continue
            event = parsed.iloc[0] if aggregate == "first" else parsed.median()
            if pd.isna(event) or event <= anchor:
                continue
            horizons_days.append(int((event - anchor).days))
            fiscal_dates.append(event)
            labels.append(fh.label)
        return horizons_days, fiscal_dates, labels

    def forecast(
            self,
            horizons_days: Any,
            *,
            fiscal_dates: Optional[Any] = None,
            labels: Optional[list[str]] = None,
            last_price: Optional[float] = None,
            random_seed: int = _DEFAULT_RANDOM_SEED,
    ) -> Union[InferenceLike, "xr_typing.Dataset"]:
        """Project the fitted local-level process forward to future fiscal events.

        This is the structural *prediction step* of the *Forecasting with
        Structural AR Timeseries* reference, specialised to the continuous-time
        local-level (Wiener) process fit by :meth:`fit`. It conditions on the
        learned posterior — the terminal smoothed latent log-state
        :math:`x_T = \\text{log\\_state}[-1]`, the process / observation scales
        ``sigma_state`` / ``sigma_obs`` and the optional ``beta_trend`` — and
        propagates it forward to a set of future horizons (typically the
        ``days_to_*`` fiscal-calendar horizons emitted by ``mv_pymc_kalman_pt``).

        For a horizon :math:`\\Delta\\tau` years beyond the last observation the
        latent log-state is

        .. math::

            x_f \\mid x_T \\sim \\mathcal{N}\\bigl(x_T + \\beta_{\\text{trend}}\\,\\Delta\\tau,\\;
                                                  \\sigma_{\\text{state}}^2\\,\\Delta\\tau\\bigr),

        and the predictive (observed) target adds the measurement variance
        :math:`\\sigma_{\\text{obs}}^2` on top of that state variance.
        This is the GRW analogue of the reference's ``ar1_fut`` initialised at
        ``DiracDelta(ar[..., -1])`` and conditioned on the learned coefficients.
        Each horizon is independent of :math:`x_T`, so the predictive variance is
        non-decreasing in the horizon.

        When :meth:`fit` was run with ``stochastic_volatility=True`` the
        posterior ``sigma_obs`` is a per-time latent path rather than a scalar;
        the forward projection holds the **terminal** posterior volatility flat
        for :math:`\\sigma_{\\text{obs}}` (the log-volatility random walk has no
        mean reversion, so its last value is the optimal flat forecast of future
        observation noise).

        Parameters
        ----------
        horizons_days : array-like
            Strictly-positive, finite day-offsets **beyond the last observation**
            (e.g. the ``days_to_next_earnings`` horizon re-based to the last
            observed date). Together with ``fiscal_dates`` / ``labels`` these are
            most conveniently produced by :meth:`build_forecast_specs`, which
            derives them from the canonical :data:`FISCAL_HORIZONS` map.
        fiscal_dates : array-like, optional
            Absolute future dates aligned 1:1 with ``horizons_days``; used as the
            ``time_future`` coordinate when supplied (else the day-offsets are).
        labels : list of str, optional
            Optional human-readable horizon labels aligned 1:1 with
            ``horizons_days`` (e.g. ``["Next earnings", "Next report"]`` from
            the :data:`FISCAL_HORIZONS` labels), attached as a ``label`` coord along
            ``time_future`` for forecast tables / plot annotations.
        last_price : float, optional
            Spot price for ``implied_upside_future``. Defaults to the
            ``last_price`` supplied at :meth:`fit` time.
        random_seed : int
            Seed for the predictive draws. Default 42.

        Returns
        -------
        arviz.InferenceData or xarray.Dataset
            A ``predictions`` group (when ArviZ is available, else a raw
            ``xarray.Dataset``) with dims ``(chain, draw, time_future)`` and
            variables ``forecast_state`` (latent, price space), ``forecast_pt``
            (predictive observed target, price space), ``predictive_sd_log``
            (closed-form log-space predictive sd) and — when a positive
            ``last_price`` is known — ``implied_upside_future``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not produced a sampled posterior.
        ValueError
            If ``horizons_days`` is empty / non-positive / non-finite, or
            ``fiscal_dates`` does not align with ``horizons_days``.
        """
        if self._fit_idata_ is None or not hasattr(self._fit_idata_, "posterior"):
            raise RuntimeError(
                "forecast() requires a completed fit() with a sampled posterior; "
                "call fit(...) first (with an ArviZ-returning sampler)."
            )
        import xarray as xr

        h = np.asarray(horizons_days, dtype="float64").ravel()
        if h.size == 0:
            raise ValueError("horizons_days must contain at least one horizon.")
        if not np.isfinite(h).all() or np.any(h <= 0):
            raise ValueError(
                "horizons_days must be strictly positive, finite day-offsets "
                "measured beyond the last observation."
            )
        delta_tau = h / 365.25  # elapsed years from the last observation

        post = self._fit_idata_.posterior
        sigma_state = np.asarray(post["sigma_state"].values, dtype="float64")
        # Under stochastic volatility ``sigma_obs`` is per-time (dims
        # chain × draw × time); the log-volatility random walk has no mean
        # reversion, so its terminal value is the optimal flat forecast of
        # future observation noise. Continuing the walk would inflate the band
        # with unanchored vol-of-vol. Otherwise it is the usual scalar.
        so_var = post["sigma_obs"]
        sigma_obs = np.asarray(
            (so_var.isel(time=-1) if "time" in so_var.dims else so_var).values,
            dtype="float64",
        )
        # Terminal smoothed latent log-state x_T (per posterior draw).
        x_term = np.asarray(post["log_state"].isel(time=-1).values, dtype="float64")
        beta_trend = (
            np.asarray(post["beta_trend"].values, dtype="float64")
            if "beta_trend" in post
            else np.zeros_like(x_term)
        )

        # Broadcast (chain, draw) x (time_future).
        x_term_ = x_term[..., None]
        ss_ = sigma_state[..., None]
        so_ = sigma_obs[..., None]
        bt_ = beta_trend[..., None]
        dt_ = delta_tau[None, None, :]

        mean_log = x_term_ + bt_ * dt_
        proc_sd = ss_ * np.sqrt(dt_)
        # Closed-form predictive (observation) sd in log-space: process + measure.
        pred_sd_log = np.sqrt(ss_ ** 2 * dt_ + so_ ** 2) * np.ones_like(mean_log)

        rng = np.random.default_rng(random_seed)
        state_log = rng.normal(mean_log, proc_sd)  # latent future log-state
        obs_log = rng.normal(state_log, so_ * np.ones_like(dt_))  # predictive log-target

        lp = last_price if last_price is not None else self._fit_last_price_

        if fiscal_dates is not None:
            tf = pd.DatetimeIndex(pd.to_datetime(np.asarray(fiscal_dates)))
            if len(tf) != h.size:
                raise ValueError(
                    f"fiscal_dates must align 1:1 with horizons_days "
                    f"(got {len(tf)} vs {h.size})."
                )
            time_future: np.ndarray = tf.values
        else:
            time_future = h

        dims = ("chain", "draw", "time_future")
        data_vars: dict[str, Any] = {
            "forecast_state": (dims, np.exp(state_log)),
            "forecast_pt": (dims, np.exp(obs_log)),
            "predictive_sd_log": (dims, pred_sd_log),
        }
        if lp is not None and np.isfinite(lp) and lp > 0:
            data_vars["implied_upside_future"] = (dims, np.exp(state_log) / float(lp) - 1.0)

        ds = xr.Dataset(
            data_vars,
            coords={
                "chain": post.coords["chain"].values,
                "draw": post.coords["draw"].values,
                "time_future": time_future,
            },
        )
        ds = ds.assign_attrs(
            horizons_days=h,
            trend=int(self._fit_has_trend_),
            last_price=float(lp) if (lp is not None and np.isfinite(lp)) else float("nan"),
        )
        if labels is not None:
            if len(labels) != h.size:
                raise ValueError(
                    f"labels must align 1:1 with horizons_days "
                    f"(got {len(labels)} vs {h.size})."
                )
            ds = ds.assign_coords(label=("time_future", list(labels)))

        # Expose as a ``predictions`` group, mirroring
        # pm.sample_posterior_predictive(predictions=True). ArviZ 1.x aliases
        # InferenceData to xarray.DataTree, so build the group as a DataTree node
        # (accessible as ``result.predictions`` / ``result["predictions"]``).
        # Fall back to the bare Dataset if DataTree is unavailable.
        try:
            return xr.DataTree.from_dict({"predictions": ds})
        except Exception:  # pragma: no cover - older xarray without DataTree
            return ds

    def fit_from_snapshot(
            self,
            df: pd.DataFrame,
            *,
            id_col: str = "isin",
            cohort: Optional[Any] = None,
            target_isin: Optional[str] = None,
            min_observations: int = 2,
            timestamp_col: str = "feature_calculated_at",
            now_cols: tuple[str, ...] = ("price_target", "last_price"),
            fiscal_anchor_col: Optional[str] = None,
            **fit_kwargs: Any,
    ) -> tuple[Optional[Union[InferenceLike, "MultiTrace"]], Optional["pm_typing.Model"]]:
        """Build a per-ISIN history from a snapshot frame and fit the model.

        Convenience wrapper around :meth:`build_price_target_history`,
        :meth:`select_target_isin`, and :meth:`fit` that reproduces the
        notebook Section 7.1 / 7.3 flow in a single call.

        Parameters
        ----------
        df : pandas.DataFrame
            Wide snapshot frame containing the ``*_ago`` cohort.
        id_col : str
            ISIN column name.
        cohort : iterable, optional
            Restrict ISIN selection to this cohort (e.g. the
            ``PriceTargetAchievement`` model's ``pt_df`` index).
        target_isin : str, optional
            Skip the auto-selection step and fit on this ISIN directly.
        min_observations : int
            Minimum non-null observations per ISIN required for fitting.
        timestamp_col : str
            Column carrying the snapshot anchor.
        now_cols : tuple of str
            Columns whose values seed the "now" anchor in the long panel.
        fiscal_anchor_col : str, optional
            Forwarded to :meth:`build_price_target_history` to anchor the
            ``*_ago`` axis on a per-ISIN fiscal-calendar date (e.g.
            ``income_statement_report_date``). ``None`` keeps the global anchor.
        **fit_kwargs : Any
            Forwarded verbatim to :meth:`fit` (``samples``, ``tune``,
            ``chains``, ``parameterization``, ``trend``, ``nuts_sampler``, …).

        Returns
        -------
        tuple
            ``(idata, model)`` from :meth:`fit`. When no eligible ISIN
            exists, returns ``(None, None)`` and emits a warning.
        """
        long_df, eligible, date_col = self.build_price_target_history(
            df,
            id_col=id_col,
            now_cols=now_cols,
            min_observations=min_observations,
            timestamp_col=timestamp_col,
            fiscal_anchor_col=fiscal_anchor_col,
        )
        if date_col is None or eligible.empty:
            warnings.warn(
                f"No ISIN has >= {min_observations} non-null price_target "
                "observations across the *_ago cohort. Skipping Kalman fit "
                "(point-in-time snapshot).",
                stacklevel=2,
            )
            return None, None

        chosen = target_isin or self.select_target_isin(eligible, cohort=cohort)
        if chosen is None:
            return None, None

        ts = (
            long_df.loc[long_df[id_col] == chosen, ["asof_date", "price_target"]]
            .dropna()
            .sort_values("asof_date")
            .reset_index(drop=True)
        )
        dates = pd.DatetimeIndex(ts["asof_date"])

        # Pull the chosen ISIN's spot price from the snapshot so the smoother can
        # anchor to it and emit ``implied_upside`` (unless the caller already
        # supplied ``last_price`` explicitly).
        if "last_price" not in fit_kwargs and "last_price" in df.columns:
            lp = pd.to_numeric(
                df.loc[df[id_col] == chosen, "last_price"], errors="coerce"
            ).dropna()
            if not lp.empty:
                fit_kwargs["last_price"] = float(lp.iloc[0])

        return self.fit(
            price_targets=ts["price_target"].to_numpy(),
            isin=chosen,
            dates=dates,
            **fit_kwargs,
        )


# ===========================================================================
# Fused cross-sectional state-space model (Model A + Model B).
#
# Mirrors :func:`probabilistic_ml_model.pymc_models.PriceTargetModel.
# build_fused_price_target_model`, re-homed onto the Kalman price-target panel
# (``pml.mv_pymc_kalman_pt``). The single notebook-Kalman refinement: the
# Model-A risk adjustment is re-keyed from *analyst conviction* to *systematic
# risk*, so the latent target is ``risk_adj_return = expected_return`` **given
# systematic risk** (``feat_avg_beta``); realized vol enters only via the
# ``feat_vol_drift`` observation-noise widener.
# ===========================================================================

# Crossed group intercepts for the cross-sectional drift mean: one fixed-scale
# ``ZeroSumNormal`` per coord, all sharing :data:`GROUP_EFFECT_SCALE` (the group
# SD is fixed, not learned — see ``build_fused_kalman_pt_model``'s docstring).
#
# ONE control per distinct dimension. The previous set — exchange, unit, country,
# industry — was three near-duplicate geography controls plus a high-cardinality
# industry, and it mixed badly: on the 2026-08-10 T=4 validation ``beta`` came in
# at R-hat 1.028 / ESS 134 with ``unit_effect`` 1.024 / 206 and
# ``exchange_effect`` 1.021 / 199, while the per-ISIN latents mixed fine
# (R-hat 1.014, ESS 308). Level counts on the production panel say why:
#
#     country 82 | exchange 81 | unit 47 | industry 59 | sector 9
#
# ``exchange`` is very nearly 1:1 with ``country`` (81 vs 82 levels — a venue sits
# in a country), and ``unit`` (currency) is largely determined by it too, so the
# three effects were mutually aliased and traded off draw to draw. ``industry``
# is near-nested under ``sector``; the comment that used to sit here already said
# industry was excluded for exactly that reason and for being the worst-mixing
# effect (R-hat ≈ 3.4) — the code had silently drifted from its own comment and
# was still including it.
#
# So: keep ``country`` as the geography control (it is also the dimension
# ``feat_mcap_global_r`` is defined against) and collapse ``industry`` to its
# parent ``sector``, the standard de-collinearisation move for a near-nested
# pair. Two orthogonal dimensions, both low-to-moderate cardinality.
# ``region`` is deliberately absent: it and ``trading_region`` are the SAME
# variable for practical purposes — they agree for 96.12 % of the 6 549-name
# universe (Cramer's V = 0.938; only cross-listings differ), against V <= 0.24
# for every other pair here. Carrying both left the two effects trading off along
# a ridge, and after the per-ISIN intercept landed they became the two
# worst-mixing globals in the model: ``region_effect`` R-hat 1.0130 / ESS 243 and
# ``trading_region_effect`` 1.0121 / 236, blocking the convergence gate on the
# 2026-08-10 validation. (They mixed acceptably BEFORE the per-ISIN latent existed
# — R-hat 1.002, ESS ~1.4-1.6k — so the redundancy was always there but only bit
# once another per-name absorber competed for the same variance.)
#
# ``trading_region`` is kept over ``region`` because this model is driven by
# analyst consensus: where a name LISTS determines which analysts cover it and in
# what currency, which is the mechanism the drift features actually measure.
_FUSED_KALMAN_GROUP_EFFECTS: tuple[str, ...] = (
    "trading_region",
    "sector",
    "style_class",
    "size_class",
)

# Fixed (non-learned) scale for the crossed group-intercept ``ZeroSumNormal``
# effects in ``build_fused_kalman_pt_model``. A learned hierarchical scale is
# structurally non-identified on the collapsed T=1 cross-section (it stuck at
# R-hat 1.5–4.5 / ESS 4–7), so the group SD is fixed here instead. 0.25 covers
# the observed effect magnitudes (~0.01–0.17 on the standardised log-uplift
# scale) at ≤ 2 prior sd.
GROUP_EFFECT_SCALE: float = 0.25

# Upper bound on the local-level state's AR(1) persistence in
# ``build_fused_kalman_pt_model``. As ``rho -> 1`` the AR field degenerates into a
# per-name constant, which is precisely what ``mu_isin`` already is, so the two
# trade off along a ridge; the cap keeps the state a genuine *process*. See the
# state block's comment for the measured failure this prevents.
_STATE_RHO_MAX: float = 0.95


def _nan_zscore(raw: np.ndarray, *, eps: float = 1e-9) -> tuple[np.ndarray, float]:
    """Nan-safe cross-sectional z-score for the fused-model tilt containers.

    Missing entries map to a 0 (neutral) tilt; a degenerate spread falls back
    to sd=1 so the z-scores stay finite. Returns ``(z, mean)`` — the mean is
    reused by callers that also need a mean-filled raw container (PyMC 6
    rejects NaN in ``pm.Data``).
    """
    arr = np.asarray(raw, dtype="float64")
    mu = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
    sd = float(np.nanstd(arr))
    sd = sd if (np.isfinite(sd) and sd > eps) else 1.0
    return np.nan_to_num((arr - mu) / sd, nan=0.0), mu


@dataclass(frozen=True, eq=False)
class KalmanPanelInputs:
    """Container of arrays consumed by the fused (3-D MvGRW) Kalman model.

    The Kalman analogue of
    :class:`probabilistic_ml_model.pymc_models._price_target_mc.PriceTargetPanelInputs`.
    Carries the standardised ``(isin, time, y_series)`` response tensor, the
    ``(isin, time)`` standardised time matrix (``*_ago`` lookback offsets when a
    genuine history panel is built via ``history_lookbacks``; a single
    days-to-event covariate on the collapsed T=1 cross-section), the
    standardised drift-feature design matrix, and the per-ISIN Model-A / σ
    inputs. The conviction latent of
    the price-target panel is replaced by ``avg_beta`` (the NULL-aware mean of the
    ``beta_{1y,2y,5y}`` windows), the systematic-risk (CAPM beta) driver of the
    ``risk_adj_return`` adjustment. ``vol_drift`` (``feat_vol_drift``, the drift
    across the realized-vol term structure) is retained for provenance /
    σ-widening but does not drive the risk adjustment.

    Attributes
    ----------
    frame : pandas.DataFrame
        Filtered modelling frame (``observed_pt > 0`` and ``last_price > 0``).
    isins : numpy.ndarray
        Per-row ISIN identifiers (``len == n_isin``).
    Y : numpy.ndarray
        Standardised response tensor, shape ``(n_isin, T, D)``.
    t_scaled : numpy.ndarray
        Standardised time matrix, shape ``(n_isin, T)``.
    X_drift : numpy.ndarray
        Standardised drift-feature design matrix, shape ``(n_isin, n_drift)``.
        Backs the state-transition mean (``beta`` slopes). Columns are the
        catalogue-driven ``kalman_pt`` mutable_predictor aliases surviving
        :meth:`KalmanFilterPriceTarget.select_drift_features`: the
        analyst-target / price drift trails, short/mid-horizon momentum, the
        analyst-sentiment representative ``feat_analyst_rating`` (the
        composition legs are barred via
        :data:`KALMAN_COLLINEAR_COMPOSITION_FEATURES`), the 1-year
        price-target credibility trio (``feat_pt_achievement_1y`` /
        ``feat_pt_accuracy_1y`` / ``feat_pt_range_hit_rate``), the
        consensus-dispersion drift (``feat_pt_noise_drift``), the EPS trend /
        surprise / beat-frequency family (``feat_net_eps_drift`` /
        ``feat_last_q_surprise`` / ``feat_last_y_surprise`` /
        ``feat_eps_beat_rate`` / ``feat_eps_beat_rate_annual``, which replaced
        the price-derived market-cap & EV signals), and the fundamental-quality
        level ``feat_median_piotroski_f_score`` (median of the four
        per-fiscal-year Piotroski F-scores; the per-year components are barred
        via :data:`KALMAN_PIOTROSKI_COMPONENT_FEATURES`).
    n_analysts, sqrt_n_analysts : numpy.ndarray
        Floored analyst count and its NumPy-precomputed square root (so the PyMC
        graph never applies an inplace ``Sqrt`` that NUTS rejects).
    vol_drift : numpy.ndarray
        Per-ISIN realized-vol drift (``feat_vol_drift``, winsorised [-1, 1] drift
        across the volatility term structure).
        Retained for provenance / σ-widening; not the risk-adjustment driver.
    avg_beta : numpy.ndarray
        Per-ISIN average market beta (``feat_avg_beta`` = NULL-aware mean of
        ``beta_{1y,2y,5y}``). The systematic-risk (CAPM) driver of the
        ``exp(-risk_penalty * z(avg_beta))`` risk adjustment.
    size_ratio : numpy.ndarray
        Per-ISIN country market-cap rank ratio (``feat_mcap_global_r`` =
        ``(100 - market_cap_country_r) / 100``): ~0 for the largest name in
        its country, larger for smaller caps. Carried raw and z-scored inside
        :func:`build_fused_kalman_pt_model`, where it discounts ``risk_adj_return``
        additively via ``- size_loading * z(size_ratio)``.
    volume_ratio : numpy.ndarray
        Per-ISIN relative trading volume (``feat_rel_volume``). Carried raw and
        z-scored inside :func:`build_fused_kalman_pt_model`, where it discounts
        ``risk_adj_return`` additively via ``- volume_loading * z(volume_ratio)``.
    dispersion_cv : numpy.ndarray
        Per-ISIN consensus noise coefficient of variation
        (``feat_pt_noise_sigma / last_price``); widens ``sigma_isin``.
    drift_names, response_names : list[str]
        Column labels backing the ``drift_feature`` / ``y_series`` dims.
    coord_uniques : dict[str, numpy.ndarray]
        Per-coord unique label vectors.
    coord_idx : dict[str, numpy.ndarray]
        Per-coord ``int64`` index arrays (``len == n_isin``).
    response_mean, response_std : numpy.ndarray
        The per-series ``(mean, std)`` ACTUALLY used to standardise :attr:`Y`,
        length ``D``. Recorded at construction so the posterior can be mapped
        back onto the raw response scale *exactly*, rather than by recomputing
        the moments downstream.

        This is load-bearing on the ``T > 1`` history panel. ``Y``'s primary
        column is standardised on the **pooled (isin × time)** moments of the
        genuine ``*_ago`` trails, but the downstream de-standardisation used to
        recompute them by tiling the **snapshot** column across ``T`` — correct
        only for the removed tile-based panel, where every time slice was a copy
        of the snapshot. On the real trail panel the two differ (measured on the
        2026-08 6 401-name run: pooled ``(0.20754, 0.24939)`` vs snapshot
        ``(0.22475, 0.24163)``), which biased every exported ``expected_upside``
        / ``expected_pt`` **+1.5 to +2.3 percentage points high**. Carrying the
        fit-time moments makes the inverse exact by construction; see
        ``pymc_kalman_filter_pt._panel_response_stats``.

        Defaults to empty arrays so panels pickled before these fields existed
        still load (the consumer then falls back to recomputation).
    """

    frame: pd.DataFrame
    isins: np.ndarray
    Y: np.ndarray
    t_scaled: np.ndarray
    X_drift: np.ndarray
    n_analysts: np.ndarray
    sqrt_n_analysts: np.ndarray
    vol_drift: np.ndarray
    dispersion_cv: np.ndarray
    avg_beta: np.ndarray
    size_ratio: np.ndarray
    volume_ratio: np.ndarray
    drift_names: list[str]
    response_names: list[str]
    coord_uniques: dict[str, np.ndarray]
    coord_idx: dict[str, np.ndarray]
    response_mean: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="float64")
    )
    response_std: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="float64")
    )
    # ---- Observation-scale drivers (2026-08-16) ----------------------------
    # Inputs to the log-linear ``sigma_isin`` model. Defaulted to empty so a
    # panel built by older code still constructs; the builder falls back to a
    # zero vector (i.e. that term drops out) when a driver is absent, so an old
    # panel reproduces the previous two-term scale exactly.
    #
    # ``range_norm`` is not new data — ``feat_pt_range_norm`` has been emitted by
    # the MV and documented as a widener in ``build_noise_wideners`` all along,
    # but no ``pm.Data`` container for it ever existed and the likelihood never
    # applied it. It correlates +0.115 with log|residual|.
    vol_level: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="float64")
    )
    log_mcap: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="float64")
    )
    range_norm: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="float64")
    )


def build_fused_kalman_pt_model(
        panel: "KalmanPanelInputs",
        *,
        group_effects: Optional[tuple[str, ...]] = None,
        risk_penalty: float = 0.15,
        size_penalty: float = 0.25,
        volume_penalty: float = 0.1,
        robust: bool = True,
        likelihood: Optional[str] = None,
        isin_level_scale: float = 0.10,
        state_innovation_scale: float = 0.0,
) -> "pm_typing.Model":
    """Build the fused coregionalised cross-sectional Kalman price-target model.

    Fuses two notebook models into a single reusable ``pm.Model`` builder:

    * **Model B spine** — a rank-1 **Intrinsic Coregionalization Model** (ICM; the
      PyMC *MOGP-Coregion-Hadamard* pattern) over the ``(isin, time, y_series)``
      response tensor: the ``D`` response series share the single latent per-ISIN
      factor ``mu_isin`` through per-series loadings ``W`` (``mu_isin_loading``,
      primary anchored at 1), with per-series intercept ``alpha`` / time-slope
      ``beta_t`` and a per-series noise diagonal ``sigma_series`` (the ICM
      ``kappa``). On the default collapsed cross-section (``T == 1``) ``alpha`` /
      ``beta_t`` are **direct Normals**; a genuine time panel (``T > 1``) re-adds a
      zero-anchored diagonal Gaussian-random-walk *deviation* (identified by the
      multiple time steps).
    * **Model A refinement** — the risk-aware ``expected_return`` →
      ``risk_adj_return`` latent (with ``achieve_prob = sigmoid(risk_adj_return)``,
      a posterior-informed confidence) becomes the shared factor ``mu_isin``, and
      the richer heteroscedastic scale ``sigma_isin = sigma_base * (1 + cv) / sqrt(n)``
      replaces Model B's cv-free form.

    The single Kalman-specific change versus
    :func:`probabilistic_ml_model.pymc_models.PriceTargetModel.build_fused_price_target_model`:
    the risk adjustment is keyed on *systematic risk* (average market beta) rather
    than analyst conviction, so the latent target reads
    ``risk_adj_return = expected_return - risk_loading * z(avg_beta)`` — i.e.
    ``expected_return`` *given systematic risk*. Beta (``feat_avg_beta``, the
    NULL-aware mean of ``beta_{1y,2y,5y}``) enters as a **standardised** (z-scored)
    cross-sectional tilt, never as a raw level (see the convergence notes below).
    Realized-vol drift (``feat_vol_drift``) is retained as a provenance container
    but does not drive the penalty.

    A second **size tilt** sits alongside the beta penalty on ``risk_adj_return``:
    ``risk_adj_return = expected_return - risk_loading * z(avg_beta)
    - size_loading * z(feat_mcap_global_r)`` where ``feat_mcap_global_r =
    market_cap / market_cap_3yavg`` is the firm's current size relative to its own
    3-year average. With a positive ``size_loading`` names trading **above** their
    3-year average size (re-rated up) are discounted while names **below** it earn a
    premium — a mean-reversion / "don't chase the re-rating" tilt — exactly
    parallel to the systematic-risk (beta) discount.

    A third **volume tilt** discounts ``risk_adj_return`` by the per-ISIN average
    relative trading volume (``feat_rel_volume``): ``- volume_loading *
    z(feat_rel_volume)``. Elevated relative volume (e.g. distribution, event-driven
    churn) discounts the risk-adjusted return the same way an elevated beta or an
    above-average size does; ``volume_penalty`` defaults to ``0.2`` — enabled by
    default (twice the ``risk_penalty`` / ``size_penalty`` prior scale) — so the
    factor is identified from the cross-section on every run unless explicitly
    disabled.

    All three tilts are **additive** in the standardised log-uplift return space, not
    the earlier multiplicative ``exp(-penalty · z)`` factors. ``expected_return`` is a
    zero-mean cross-sectional *deviation*, so a multiplicative factor only rescaled
    its magnitude and inverted the discount direction for below-average names (a
    high-beta, below-average name was lifted toward zero, not penalised); the
    additive form ``- loading · z`` is monotone in the feature for either sign of
    ``expected_return``. ``risk_loading`` / ``size_loading`` / ``volume_loading`` are
    **learned** sign-fixed (``HalfNormal``) loadings whose prior scale is
    ``risk_penalty`` / ``size_penalty`` / ``volume_penalty`` (``0.0`` disables the
    factor), so the penalty magnitude is identified from the cross-section while its
    direction stays a discount.

    NUTS-inplace-safety constraints (preserved from the notebook):

    * ``sqrt(n_analysts)`` is precomputed in NumPy and passed as ``pm.Data``
      (never ``pt.sqrt`` on an integer container).
    * ``sigma_base`` uses :class:`pm.Exponential` (not ``HalfNormal``) so no
      ``Abs→Sqrt`` rewrite fuses into an inplace ``Composite`` op.
    * The GRW uses an explicit diagonal innovation parameterisation instead of
      :class:`pm.LKJCholeskyCov` (whose onion-method Beta→Normal chain triggers
      an inplace rewrite NUTS rejects).

    Sampler-geometry parameterisation (added to escape the ~0.001 step-size /
    max-tree-depth regime observed when sampling the full panel):

    * ``mu_global`` is **dropped**: the per-ISIN baseline ``mu_isin`` is the
      model's sole (identified) level, so the flat ``mu_global`` ↔ walk-level
      ridge — a tiny-step / max-tree-depth driver — is removed at the root.
    * Remaining priors are scaled to the standardised response/design (``beta``
      σ≈1.0) rather than the previous diffuse ``10``/``5``, so the posterior is
      well-conditioned.
    * Crossed group intercepts (see :data:`_FUSED_KALMAN_GROUP_EFFECTS`) use a
      :class:`pm.ZeroSumNormal` at a **FIXED scale** :data:`GROUP_EFFECT_SCALE`
      (0.25) — the group SD is **not learned**. The sum-to-zero constraint removes
      the additive level ridge (level carried by ``alpha``); the fixed scale removes
      the variance-partition ridge. A learned ``sigma_group`` is *structurally
      non-identified* on the collapsed T=1 cross-section (one slice cannot separate
      between-group dispersion from the residual base scale): the non-centred build
      stuck at ``sigma_group`` R-hat ≈ 1.52 / ESS ≈ 7 collapsed to ≈ 0.004 (1 %),
      the centred build at R-hat ≈ 4.5 / ESS ≈ 4 wandering near 0.22 (28 %) — the
      two disagreeing is the non-identifiability signature, not a funnel. The group
      EFFECT vectors themselves already mixed fine (ESS ≈ 4k–9k, R-hat ≈ 1.0); only
      the scale was stuck, dragging the dependent scalars down with it. Fixing the
      scale (Gelman: fix the group SD when one slice can't identify it) deletes
      ``sigma_group`` from the sampler so the group effects are plain regularized
      fixed effects. ``_FUSED_KALMAN_GROUP_EFFECTS`` keeps ONE control
      per distinct dimension (``country``, ``sector``); the near-duplicate
      geography coords and the near-nested ``industry`` were dropped — see that
      constant for the mixing evidence. Per-coord ``sigma_<col>`` are re-exposed as a
      Deterministic of each effect's empirical between-group sd (a genuine,
      well-identified per-coord number — no longer four aliases of one stuck scalar)
      so downstream diagnostics resolve unchanged.
    * ``expected_return`` is the **deterministic** structural mean ``mu_reg``. The
      previous per-ISIN signal latent (``mu_reg + sigma_er * z_expected_return``,
      scale ``sigma_expected_return``) was the largest member of that unidentified
      partition (itself stuck at R-hat ≈ 4.4) and is removed; per-ISIN idiosyncratic
      dispersion is carried by the Student-t measurement noise ``sigma_isin``.
      ``expected_return`` remains exposed as a Deterministic for downstream consumers.
    * Per-series level ``alpha`` / time-slope ``beta_t`` are **direct Normals** on
      the collapsed cross-section (``T == 1``, ``prepare_kalman_panel_inputs``
      without ``history_lookbacks``). The previous build parameterised them as a
      single-step Gaussian random walk ``alpha[0,d] = sigma_alpha_innov[d] *
      z_alpha[0,d]`` — a product of **two** scalars representing **one** intercept.
      The likelihood sees only the product, so the (scale, innovation) pair is
      non-identified: a ridge (not a funnel — it persists with **0 divergences**)
      that trapped the sampler. The ``sigma_*_innov`` within-chain variance drove to
      0 (the arviz R-hat divide-by-zero NaN), ESS collapsed to ≈ 7, R-hat ≈ 1.5 and
      ``sigma_group`` was squeezed to ≈ 0 — even after the funnel above was removed.
      Direct intercepts delete the redundant scale parameters and the model mixes
      (R-hat ≈ 1.0, ESS in the hundreds–thousands). A genuine time panel
      (``T > 1``, built from the ``*_ago`` trails via
      ``prepare_kalman_panel_inputs(history_lookbacks=...)``) uses **T direct
      per-time intercepts** for the same reason: the earlier zero-anchored GRW
      deviations (plus a global slope) were mutually aliased per time slice on
      an isin-constant time axis and reproduced the ridge (2026-08-01 T=4 run:
      190 divergences, ``alpha_level``/``beta_slope`` R-hat 1.06). The
      per-series time slope ``beta_t`` is materialised only when ``t_scaled``
      genuinely varies across ISINs; on an isin-constant axis it is **not created
      at all** (it was previously published as a structurally-zero Deterministic,
      which read as a fitted-and-dead parameter, gave arviz a constant to divide
      0/0 on for R-hat, and drew a flat zero line in the §13b slope panel).
    * Each response series carries its **own** noise scale ``sigma_series`` (the ICM
      ``kappa`` diagonal; primary anchored at 1, others ``HalfNormal(0.5)``) so a
      noisier output (``feat_pt_drift``) no longer inflates the primary series' σ —
      ``sigma_obs[i,d] = sigma_isin[i] * sigma_series[d]``.

    Per-ISIN latent on a genuine panel (``T > 1``) — what the time axis buys:

    * A per-ISIN random intercept ``sigma_isin_level · z_isin_level``
      (``ZeroSumNormal`` over ``isin``, non-centred) is added to
      ``expected_return``. It was dropped in an earlier release as non-identified
      — correctly, **for ``T == 1``**. T repeated observations per name identify
      it in the ordinary way, and restoring it is what fixes the
      pseudo-replication described below.
    * ``state_path`` (``isin``, ``time``) and ``state_now = state_path[:, -1]``
      are always emitted so downstream consumers have one stable name for the
      decision latent; with the AR layer off they equal ``mu_isin``.

    Time-varying state (``state_innovation_scale > 0``) — **off by default**:

    * Adds a stationary AR(1) deviation ``sigma_state · z[i, t]`` with persistence
      ``state_rho`` on top of the intercept, so a name's level may move across the
      lookback window.
    * **It did not pay for itself at T=4 and is disabled.** On the synthetic
      recovery check (N=300, T=4; truth: intercept sd 0.8, AR deviation sd 0.35,
      rho 0.75) it bought +0.013 correlation with the true terminal state (0.964
      vs 0.951) while collapsing mixing — min ESS 14 vs 69, max R-hat 1.13 vs
      1.03 — and ``sigma_state`` / ``rho`` drifted between runs at different draw
      budgets (0.265→0.216, 0.63→0.49), the same "two fits disagree" signature
      this model's docstring already flags for non-identified variance
      components. With T=4 the data cannot cleanly split a per-name constant from
      a persistent per-name deviation. Revisit on a longer panel; the machinery
      is retained and tested.
    * **Why it was added.** Broadcasting a time-constant ``mu_isin`` across the T
      lookback slices treated each name's T serially-correlated log-uplift
      observations as T iid draws around one latent: pseudo-replication that
      shrank the per-name posterior sd by ~√T and produced over-confident
      ``expected_upside`` HDIs in the screen, the CVaR risk book and
      ``analytics.kalman_filtered_price_targets``. With ``beta_t`` also zeroed on
      the tiled time axis, the whole time dimension contributed nothing but T
      cross-sectionally-shared intercepts — no state evolution whatsoever.
    * **Why it is identified here** when the per-ISIN random effect above is not:
      that one was dropped for want of repeated observations per name on the T=1
      cross-section. T repeated observations is precisely what separates a
      persistent per-name level from transient measurement noise; the T=1 argument
      does not carry over to a genuine panel. On ``T == 1`` the state collapses to
      ``mu_isin`` and no innovation parameters are created.
    * **Alternatives tried and rejected.** (a) Re-adding a free ``beta_slope`` on
      the tiled axis — exactly aliased with the T free ``alpha_level`` intercepts;
      this is the 2026-08-01 T=4 failure (190 divergences, ``alpha_level`` /
      ``beta_slope`` R-hat 1.06, bulk-ESS ≈ 75). (b) A **centred** field —
      reintroduces the scale ↔ deviation funnel that gave ``sigma_*_innov``
      ESS ≈ 7 / R-hat ≈ 1.5; the build is non-centred instead. (c) An **uncentred**
      field — its cross-sectional mean at each t is aliased with
      ``alpha_level[t, ·]``. (d) A literal **cumulative random walk** anchored at
      ``state[i,0] = mu_isin[i]`` — its marginal variance grows as ``√t``, which
      the 2026-08-10 full-scale run falsified directly: per-time predictive
      coverage ramped 89.9 % → 95.3 % → 97.2 % → 98.2 % against a 94 % target
      (too narrow at t=0, too wide at the snapshot), with max R-hat 1.054 and min
      ESS 74. Uplift is marginally stationary across this window, so the
      doubly-centred (constant-variance) field replaced it. (e) The
      raw-``Normal``-then-subtract-means form of the same field — over-parameterised
      by ``n_isin + T - 1`` likelihood-invariant directions that merely resample
      their prior while dominating mass-matrix adaptation.
    * **Expected signature vs the static build:** ``sigma_base`` falls (persistent
      per-name signal leaves the residual for the state) and exported
      ``expected_upside`` HDIs widen — the pseudo-replication correction landing,
      not a regression. A ``sigma_state`` posterior collapsing to ~0 is the signal
      that the panel carries no per-name dynamics and the layer should be dropped.

    Parameters
    ----------
    panel : KalmanPanelInputs
        Output of ``prepare_kalman_panel_inputs`` (in ``pymc_kalman_filter_pt``).
    group_effects : tuple[str, ...], optional
        Coord names receiving a hierarchical intercept. Defaults to the
        intersection of :data:`_FUSED_KALMAN_GROUP_EFFECTS` with the panel coords.
    risk_penalty : float
        Prior scale (``HalfNormal`` sigma) of the learned, sign-fixed
        ``risk_loading`` applied to ``expected_return`` via the per-ISIN average
        market beta (z-scored systematic risk) as an additive tilt
        ``- risk_loading * z(feat_avg_beta)``. ``0.0`` disables the factor.
    size_penalty : float
        Prior scale (``HalfNormal`` sigma) of the learned, sign-fixed
        ``size_loading`` applied to ``risk_adj_return`` via the per-ISIN size ratio
        ``feat_mcap_global_r`` (z-scored) as an additive tilt
        ``- size_loading * z(feat_mcap_global_r)`` — discounting names trading above
        their 3-year average size. ``0.0`` disables the size factor entirely.
    volume_penalty : float
        Prior scale (``HalfNormal`` sigma) of the learned, sign-fixed
        ``volume_loading`` applied to ``expected_return`` via the per-ISIN average
        rel_volume (z-scored relative volume) as an additive tilt
        ``- volume_loading * z(feat_rel_volume)``. ``0.0`` disables the factor.
    robust : bool
        Legacy two-way switch, kept so existing callers are unchanged. ``True``
        (default) → Student-t panel likelihood; ``False`` → Normal twin. Ignored
        when ``likelihood`` is given explicitly.
    likelihood : Optional[str]
        Observation model: ``'student_t'``, ``'mixture'`` or ``'normal'``.
        ``None`` (default) resolves from ``robust`` so nothing changes for
        existing callers.

        ``'mixture'`` is a two-component scale mixture — a tight Normal core plus
        a wider Normal outlier component sharing the same mean. It exists because
        the Student-t's single shape parameter cannot be tight in the bulk and
        heavy in the tail simultaneously: on the 2026-08-16 panel it pinned
        ``nu`` at its 2.5 floor and over-dispersed the predictive by ~40 %,
        failing the T=std, T=mean and PIT checks together. Fitted directly to the
        standardised response the mixture beats the t by dBIC 877 and matches the
        observed sd exactly. Under this branch ``nu`` is emitted as a *derived*
        kurtosis-matched summary for downstream consumers, not as a fitted dof.
    isin_level_scale : float
        **Fixed** (not learned) sd of the per-ISIN ``ZeroSumNormal`` intercept
        restored on ``T > 1`` panels — the per-name analogue of
        :data:`GROUP_EFFECT_SCALE`. A learned scale was tried and was the
        worst-mixing parameter in the model (R-hat 1.023 / ESS 110, sliding
        toward 0 and shifting with the group-effect coord set); the T=4 slices
        are too serially correlated to identify a per-name variance component.
        ``0.0`` pins the intercept off, recovering the pre-0.9.9.14 purely
        deterministic ``expected_return`` — the baseline arm for before/after
        comparison. Ignored when ``T == 1``.
    state_innovation_scale : float
        Prior scale (``HalfNormal`` sigma) of ``sigma_state``, the marginal sd of
        the AR(1) state deviation, on the standardised response scale.
        **Defaults to 0.0 — the time-varying state layer is OFF.** The per-ISIN
        time variation the T>1 panel reliably supports is carried by the per-ISIN
        random intercept (``sigma_isin_level``) restored above; the AR deviation
        on top of it did not pay for itself at T=4 (see *Time-varying state* in
        the notes below). Set to e.g. ``0.1`` to enable it — worth revisiting on
        a longer panel. Ignored when ``T == 1``.

    Returns
    -------
    pm.Model
        The unfitted fused model. ``state_now``, ``risk_adj_return``,
        ``sigma_isin`` and ``nu`` keep ``dims='isin'`` so a downstream
        Monte-Carlo helper consumes per-isin mu/sigma draws + scalar nu
        unchanged.
    """
    if pm is None or pt is None:
        raise ImportError(
            "PyMC is not available. Install pymc + pytensor to build the "
            "fused Kalman price-target model."
        )

    # Guard: response-leakage aliases must never reach the drift design matrix
    # (``feat_implied_upside`` IS the log-uplift response after ``log1p``). The
    # catalogue-driven selector already drops them; this backstop protects
    # callers that hand-assemble ``KalmanPanelInputs``.
    leaked = sorted(KALMAN_LEAKAGE_FEATURES.intersection(panel.drift_names))
    if leaked:
        raise ValueError(
            f"build_fused_kalman_pt_model: drift_names contains response-leakage "
            f"features {leaked!r}. Remove them upstream ("
            "KalmanFilterPriceTarget.select_drift_features enforces this)."
        )

    n_isin, T, D = panel.Y.shape

    # Guard: a (near-)zero-variance NON-PRIMARY response series leaves its rank-1
    # coregion loading ``mu_isin_loading`` unidentified — a flat ``loading × mu_isin``
    # ridge that collapses the NUTS step size and freezes the whole posterior. The
    # data-side coverage guard in ``prepare_kalman_panel_inputs`` should already have
    # dropped such series; this is a defensive backstop so a degenerate series can
    # never silently enter the ICM (the primary series, index 0, is exempt — it is
    # the de-standardisation anchor and may legitimately be rescaled).
    if D > 1:
        series_var = np.nanvar(
            np.asarray(panel.Y, dtype="float64").reshape(-1, D), axis=0
        )
        degenerate = [
            panel.response_names[d]
            for d in range(1, D)
            if not np.isfinite(series_var[d]) or series_var[d] < 1e-8
        ]
        if degenerate:
            raise ValueError(
                "build_fused_kalman_pt_model: non-primary response series "
                f"{degenerate!r} have ~zero variance and would leave their ICM "
                "loading unidentified. Drop them upstream (the "
                "KALMAN_RESPONSE_COVERAGE_MIN guard in prepare_kalman_panel_inputs)."
            )

    coords: dict[str, Any] = {
        "isin": np.asarray(panel.isins),
        "time": np.arange(T),
        "y_series": np.asarray(panel.response_names),
        "drift_feature": list(panel.drift_names),
    }
    # Free per-time noise multipliers: every slice except the SNAPSHOT, which is
    # anchored at 1.0 so ``sigma_base`` stays identified.
    if T > 1:
        coords["time_free"] = np.arange(T - 1)
    # Non-primary response series carry a free factor loading on ``mu_isin``; the
    # primary series (index 0, ``feat_implied_upside``) is anchored at loading ≡ 1.
    if D > 1:
        coords["y_series_free"] = np.asarray(panel.response_names[1:])
    for col, uniques in panel.coord_uniques.items():
        coords[col] = uniques

    avail_groups = (
        tuple(group_effects)
        if group_effects is not None
        else tuple(c for c in _FUSED_KALMAN_GROUP_EFFECTS if c in panel.coord_idx)
    )
    avail_groups = tuple(c for c in avail_groups if c in panel.coord_idx)

    # Standardised average market beta for the risk penalty. ``panel.avg_beta`` is
    # ``feat_avg_beta`` (the NULL-aware mean of ``beta_{1y,2y,5y}``), the
    # systematic-risk (CAPM) driver that replaces the prior expected-volatility
    # penalty. Raw beta is O(1) (≈ 1 for the market), but z-scoring keeps the
    # penalty a *relative* cross-sectional tilt — high-beta names discounted,
    # low-beta names lifted, mean discount ≈ 1 (``exp(±risk_penalty)`` over ±1 sd of
    # beta) — which is robust to cross-sectional level shifts and leaves the
    # structural level identified by ``mu_isin``/``alpha``.
    avg_beta_z, _beta_mu = _nan_zscore(panel.avg_beta)
    # ``feat_avg_beta`` is intentionally left NaN-bearing by
    # ``prepare_kalman_panel_inputs`` (missing beta → 0 tilt via ``avg_beta_z``
    # above). The raw container is provenance-only — the math consumes
    # ``beta_z`` — but PyMC 6.0 rejects NaN in ``pm.Data``, so fill missing
    # entries with the cross-sectional mean to keep the stored record honest on
    # the raw beta scale (and unbiased) rather than collapsing them to 0.
    avg_beta_filled = np.nan_to_num(
        np.asarray(panel.avg_beta, dtype="float64"), nan=_beta_mu
    )

    # Standardised size ratio for the learned size tilt on ``expected_return``.
    # ``panel.size_ratio`` is ``feat_mcap_global_r`` (= market_cap / market_cap_3yavg):
    # the firm's current market cap relative to its own 3-year average. Raw values
    # are O(1) around 1.0, so z-scoring keeps the tilt a *relative* cross-sectional
    # factor (currently-shrunk vs currently-extended names) robust to level shifts,
    # exactly as for ``avg_beta``. Missing entries (NaN) map to a 0 tilt via the
    # z-scored container; the raw container is filled with the cross-sectional mean
    # for an honest provenance record (PyMC 6.0 rejects NaN in ``pm.Data``).
    size_ratio_z, _size_mu = _nan_zscore(panel.size_ratio)
    size_ratio_filled = np.nan_to_num(
        np.asarray(panel.size_ratio, dtype="float64"), nan=_size_mu
    )

    # Standardised relative trading volume for the learned volume tilt on
    # ``risk_adj_return``. ``panel.volume_ratio`` is ``feat_rel_volume``; z-scoring
    # keeps the tilt a *relative* cross-sectional factor, exactly as for
    # ``avg_beta`` / ``size_ratio``. Missing entries (NaN) map to a 0 tilt via the
    # z-scored container; the raw container is filled with the cross-sectional mean
    # for an honest provenance record (PyMC 6.0 rejects NaN in ``pm.Data``).
    volume_ratio_z, _volume_mu = _nan_zscore(panel.volume_ratio)
    volume_ratio_filled = np.nan_to_num(
        np.asarray(panel.volume_ratio, dtype="float64"), nan=_volume_mu
    )

    # Standardised realized-vol drift, retained for provenance only.
    # ``feat_vol_drift`` is a winsorised [-1, 1] drift; it is z-scored here so the
    # stored ``feat_vol_drift_z`` container carries the cross-sectional relative
    # position alongside the raw signed drift.
    vol_drift_z, _ = _nan_zscore(panel.vol_drift)

    # ---- Observation-scale drivers (2026-08-16) ------------------------------
    # z-scored so each ``delta_*`` below reads as "log-sigma change per 1 sd of
    # the driver", and so a driver that is absent (empty array on a pre-2026-08-16
    # panel) contributes an all-zero vector, i.e. its term vanishes rather than
    # shifting the intercept. ``feat_vol_level`` is logged first: realized vol is
    # right-skewed and strictly positive, so its z-score would otherwise be
    # dominated by a handful of 300-vol names sitting on the winsorisation cap.
    def _driver_z(raw: np.ndarray, n: int, *, log1p: bool = False) -> np.ndarray:
        arr = np.asarray(raw, dtype="float64")
        if arr.size != n:
            return np.zeros(n, dtype="float64")
        if log1p:
            arr = np.log1p(np.clip(arr, 0.0, None))
        return _nan_zscore(arr)[0]

    vol_level_z = _driver_z(panel.vol_level, n_isin, log1p=True)
    log_mcap_z = _driver_z(panel.log_mcap, n_isin)
    range_norm_z = _driver_z(panel.range_norm, n_isin)

    # Median-normalised analyst-count precision weight for the measurement scale.
    # The response tensor ``Y`` is z-scored per series (std ≈ 1), so dividing the
    # observation noise by the *raw* ``√n`` (n up to ~50) drove ``sigma_isin`` to
    # ≈ 0.04 — ~10× too small for the data scale. Combined with the heavy-tailed
    # (ν≈2.5) likelihood, every per-ISIN residual then sat deep in the Student-t's
    # redescending tail where the gradient w.r.t. the mean ≈ 0, so the likelihood
    # could not distinguish ISINs and the hierarchical scale ``sigma_expected_return``
    # collapsed to 0 → full pooling → the static ≈30 % screen. Normalising by the
    # median analyst count puts the median name at weight 1 (so ``sigma_isin`` ≈
    # ``sigma_base·(1+cv)`` ≈ O(1) on the z-scored scale) while preserving the
    # *relative* precision tilt: high-coverage names tighten (less shrinkage),
    # few-analyst names widen (more shrinkage toward the group mean).
    _n_raw = np.asarray(panel.n_analysts, dtype="float64")
    _n_raw = np.where(np.isfinite(_n_raw) & (_n_raw > 0), _n_raw, 1.0)
    _n_ref = float(np.median(_n_raw))
    _n_ref = _n_ref if (np.isfinite(_n_ref) and _n_ref > 0) else 1.0
    precision_weight = np.sqrt(_n_raw / _n_ref)

    with pm.Model(coords=coords) as model:
        # ---- pm.Data containers (names mirror MV aliases) ----
        Y_obs = pm.Data("Y_obs", panel.Y, dims=("isin", "time", "y_series"))
        t_obs = pm.Data("t_scaled", panel.t_scaled, dims=("isin", "time"))
        X_data = pm.Data("drift_features", panel.X_drift, dims=("isin", "drift_feature"))
        # Average market beta (feat_avg_beta = mean of beta_{1y,2y,5y}) is the
        # systematic-risk driver of the risk adjustment below. Raw beta retained for
        # provenance; the math consumes the standardised ``beta_z``.
        pm.Data("feat_avg_beta", avg_beta_filled, dims="isin")
        beta_z = pm.Data("feat_avg_beta_z", avg_beta_z, dims="isin")
        # Size driver (feat_mcap_global_r = (100 - market_cap_country_r) / 100,
        # ~0 = largest in country): the learned additive size tilt on
        # ``expected_return`` below. Raw container retained for provenance; the
        # math consumes the standardised ``size_z``.
        pm.Data("feat_mcap_global_r", size_ratio_filled, dims="isin")
        size_z = pm.Data("feat_mcap_global_r_z", size_ratio_z, dims="isin")
        # Volume driver (feat_rel_volume, relative trading volume): the learned
        # additive volume tilt on ``risk_adj_return`` below. Raw container retained
        # for provenance; the math consumes the standardised ``volume_z``.
        pm.Data("feat_rel_volume", volume_ratio_filled, dims="isin")
        volume_z = pm.Data("feat_rel_volume_z", volume_ratio_z, dims="isin")
        # Realized-vol drift (feat_vol_drift) retained as a provenance container
        # only — it does not drive the risk adjustment.
        # ``build_noise_wideners(..., fillna=True)`` already 0-fills these for the
        # model contract; ``nan_to_num`` is a defensive guard so any stray NaN
        # (PyMC 6.0 rejects NaN in ``pm.Data``) can never break model build.
        pm.Data("feat_vol_drift", np.nan_to_num(panel.vol_drift), dims="isin")
        pm.Data("feat_vol_drift_z", vol_drift_z, dims="isin")
        cv_data = pm.Data(
            "feat_pt_noise_cv", np.nan_to_num(panel.dispersion_cv), dims="isin"
        )
        # Observation-scale drivers, standardised. Unlike feat_vol_drift above
        # these are NOT provenance-only: they enter sigma_isin directly.
        vol_level_data = pm.Data("feat_vol_level_z", vol_level_z, dims="isin")
        log_mcap_data = pm.Data("feat_log_mcap_z", log_mcap_z, dims="isin")
        range_norm_data = pm.Data("feat_pt_range_norm_z", range_norm_z, dims="isin")
        pm.Data("n_analysts", np.nan_to_num(panel.n_analysts, nan=1.0), dims="isin")
        # Raw √n retained for provenance; the measurement scale uses the
        # median-normalised ``precision_weight`` below.
        pm.Data(
            "sqrt_n_analysts", np.nan_to_num(panel.sqrt_n_analysts, nan=1.0),
            dims="isin",
        )
        precision_weight_data = pm.Data(
            "precision_weight", precision_weight, dims="isin"
        )
        idx_data_vars = {
            col: pm.Data(f"{col}_idx", panel.coord_idx[col], dims="isin")
            for col in panel.coord_idx
        }

        # ---- Model A: achievement probability ----
        # NOTE (convergence fix): the earlier "non-centred logit-normal" block
        # (``mu_logit`` / ``sigma_logit`` / ``z_isin`` → ``achieve_prob``) was
        # DEAD with respect to the likelihood — ``achieve_prob`` was never read by
        # ``mu_isin`` / the regression, so its ``n_isin`` ``z_isin`` latents and the
        # ``sigma_logit`` scale sampled their *prior* only, yet dominated the
        # mass-matrix adaptation (1 000 nuisance dims) and were empirically the
        # WORST-mixing variables in the panel fit (R-hat ≈ 3.1, ESS ≈ 2 — i.e. the
        # ``achieve_prob`` / ``z_isin`` pair sat at the very top of the R-hat
        # ranking, alongside ~600 divergences). Removing the block takes the panel
        # to **0 divergences** and R-hat → ~1.0. ``achieve_prob`` is now a genuine,
        # posterior-informed Deterministic — the logistic map of the risk-adjusted
        # return — defined just after ``risk_adj_return`` below, so downstream
        # consumers (prior/posterior plots, the ``kalman_gain`` readout) resolve
        # unchanged but on a quantity the data actually moves.

        # ---- Cross-sectional drift-regression baseline ----
        # Priors are scaled to the *standardised* response/design — Y and X_drift
        # are z-scored in ``prepare_kalman_panel_inputs`` — so O(1) effects are
        # expected. The previous sigma=10/5 priors left the posterior badly
        # conditioned: the near-flat high-variance directions forced NUTS into a
        # ~0.001 step size with max-tree-depth trajectories.
        #
        # No global intercept: the per-ISIN baseline ``mu_isin`` is the model's
        # sole (identified) level and the GRW ``alpha`` (anchored at t=0 below)
        # carries pure time deviations. A scalar ``mu_global`` traded off against
        # the walk level along a flat additive ridge — the dominant tiny-step /
        # max-tree-depth driver — so it is removed at the root. ``mu_reg`` is now
        # a pure deviation; the response is z-scored per series (mean 0), so the
        # level ``mu_isin`` carries is ≈ 0 and these priors sit on the O(1)
        # standardised scale.
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="drift_feature")

        eta = pt.dot(X_data, beta)
        # Sum-to-zero partial pooling (``ZeroSumNormal``) on the crossed group
        # intercepts, at a FIXED (non-learned) scale. The zero-sum constraint
        # removes the additive *level* ridge (the level is carried uniquely by
        # ``alpha`` per series); the fixed scale removes the variance-partition
        # ridge that trapped the sampler.
        #
        # Why the group scale is NOT learned. Earlier builds sampled a learned
        # shared scale ``sigma_group`` (``effect = sigma_group * z_g`` non-centred,
        # or ``effect ~ ZSN(sigma=sigma_group)`` centred). On the collapsed
        # cross-section (T=1, ~1 informative slice per
        # ISIN) a between-group variance component is *structurally
        # non-identified* — a single slice cannot tell between-group dispersion
        # apart from the residual base scale ``sigma_base``. Neither
        # parameterization fixes that: the non-centred build stuck at
        # ``sigma_group`` R-hat ≈ 1.52 / ESS ≈ 7 collapsed to ≈ 0.004 (1 % of the
        # cross-sectional variance), while the centred build stuck at R-hat ≈ 4.5 /
        # ESS ≈ 4 wandering near 0.22 (28 %) — the two disagreeing wildly is the
        # signature of non-identifiability, not a funnel. Crucially the group
        # EFFECT vectors themselves already mixed fine (ESS ≈ 4k–9k, R-hat ≈ 1.0):
        # only the scale was stuck, and it dragged the dependent scalars
        # (``sigma_base`` / ``risk_loading`` …) down with it.
        #
        # Fixing the scale to a weakly-informative constant (Gelman: fix the group
        # SD when one slice can't identify it) deletes ``sigma_group`` from the
        # sampler entirely, so the ``sigma_group`` ↔ ``sigma_base`` ridge is gone
        # and the crossed effects become plain regularized fixed effects.
        # ``GROUP_EFFECT_SCALE`` = 0.25 comfortably covers the observed effect
        # magnitudes (~0.01–0.17 on the standardised log-uplift scale) at ≤ 2 prior
        # sd. Which coords participate is :data:`_FUSED_KALMAN_GROUP_EFFECTS` —
        # one control per distinct dimension, the near-duplicate geography coords
        # and the near-nested ``industry`` having been dropped for collinearity.
        for col in avail_groups:
            group_effect = pm.ZeroSumNormal(
                f"{col}_effect", sigma=GROUP_EFFECT_SCALE, dims=col
            )
            # Genuine per-coord between-group sd (a Deterministic of the effect
            # draws, so well-identified — inherits the effect vectors' ESS). This
            # replaces the former four-way alias of the single stuck ``sigma_group``
            # scalar, which made the four ``sigma_<col>`` diagnostics identical and
            # overstated one problem as four.
            pm.Deterministic(f"sigma_{col}", group_effect.std())
            eta = eta + group_effect[idx_data_vars[col]]
        mu_reg = pm.Deterministic("mu_reg", eta, dims="isin")

        # ---- Model A: volatility-aware expected / risk-adjusted return ----
        # ``expected_return`` is the DETERMINISTIC structural mean ``mu_reg``
        # (drift β + shared-scale group effects). The previous per-ISIN latent
        # ``mu_reg + sigma_er * z_expected_return`` added an explicit per-ISIN
        # signal random-effect with its own learned scale ``sigma_expected_return``
        # — but on a single cross-sectional slice (collapsed T=1,
        # ~1 obs/ISIN) that scale could not be told apart from the crossed group
        # scales or the measurement noise, so it joined the unidentified variance
        # partition (it was itself stuck at R-hat ≈ 4.4, ESS ≈ 4) and drove most
        # of the divergences. Dropping it (and its ``n_isin`` ``z_expected_return``
        # parameters) leaves a clean, fully-identified hierarchical regression:
        # the per-ISIN mean is the structural mean, and idiosyncratic per-ISIN
        # dispersion is carried by the Student-t measurement noise ``sigma_isin``
        # (which already widens per-ISIN via ``cv`` / ``precision_weight``).
        # ``expected_return`` / ``risk_adj_return`` / ``mu_isin`` Deterministics are
        # retained so downstream Monte-Carlo consumers are unchanged.
        #
        # ...that reasoning is specific to ``T == 1`` and EXPIRES on a genuine
        # panel. With T repeated observations per name a per-ISIN intercept is
        # identified against the measurement noise in the ordinary way, so on
        # ``T > 1`` the random effect is restored.
        #
        # This is not optional once the state layer exists — it is what keeps the
        # two blocks from fighting. Without a free per-name intercept the ONLY
        # place a per-name constant can live is the AR(1) state at ``rho -> 1``,
        # and the synthetic recovery check caught it doing exactly that: against a
        # truth of (per-name intercept sd 0.8, AR deviation sd 0.35, rho 0.75) the
        # sampler drove ``rho`` to its 0.95 cap and ``sigma_state`` to 0.81,
        # over-covering every predictive interval (per-time coverage 99-100 % vs
        # the 94 % target) at min ESS 78. Giving the constant its proper home
        # leaves the AR carrying genuine time variation.
        #
        # ``ZeroSumNormal`` over ``isin`` so the effect carries only cross-sectional
        # deviation and cannot alias with the ``alpha_level`` level; non-centred
        # (scale x unit-ZSN) to avoid the funnel.
        if T > 1 and isin_level_scale > 0:
            # FIXED scale, not learned — the same medicine, and for the same
            # reason, as the crossed group effects above.
            #
            # A learned ``sigma_isin_level`` was tried first and behaved exactly
            # like the ``sigma_group`` scalar this model already abandoned: on the
            # 2026-08-10 full-scale run (6 540 ISINs, T=4) it was the WORST-mixing
            # parameter in the entire model at R-hat 1.023 / ESS 110, and its
            # posterior slid toward the boundary (mean 0.032, 5.5 % quantile
            # 0.0073). It also moved with the group-effect coord set — 0.089 under
            # one set, 0.032 under another — which is the "two fits disagree"
            # signature of a variance component the data cannot pin down, not a
            # funnel.
            #
            # Why the data cannot pin it down: separating per-name SIGNAL
            # dispersion from per-name MEASUREMENT noise needs independent
            # replication within a name, and the T=4 lookback slices are strongly
            # serially correlated (analyst targets are sticky), so they supply far
            # less than 4 independent observations each. The level itself is
            # identified; its scale is not.
            #
            # Fixing the scale deletes the stuck scalar from the sampler and
            # leaves a plain regularized per-name effect, exactly as
            # ``GROUP_EFFECT_SCALE`` does for the crossed intercepts.
            # ``isin_level_scale`` is that constant. The default 0.10 brackets
            # both learned estimates (0.032-0.089) with room above, so it
            # regularizes without pinning the effect near zero.
            #
            # ---- 2026-08-16/17: raising it to 0.40 was TRIED and REVERTED -----
            # The marginal case for a larger value is real and is recorded here so
            # it is not re-derived from scratch. A pooled OLS with shared slopes
            # and free per-time intercepts -- the exact structure this model
            # imposes -- on the live 6,487-name T=4 panel gives:
            #
            #     total residual sd                    0.6910
            #     BETWEEN-name level sd                0.4718
            #     WITHIN-name (across-time) sd         0.5049
            #     per-name level share of residual var  46.6 %
            #
            # Deflated for the estimation noise in each name's own level
            # (Var_true = 0.4718^2 - 0.5049^2/T) that is ~0.40, and re-measuring
            # with the crossed group effects in the design barely moves it
            # (0.3854), so the figure is not double-counting sector/region/style/
            # size variance. On that basis the constant was set to 0.40.
            #
            # It did not behave as the arithmetic predicted. At 0.40 the per-name
            # level does real work -- realised effect sd 0.2383, 60 % of the
            # scale, against 0.0468 at 0.10 -- and ``nu`` rises further (4.77 ->
            # 8.58). But ``sigma_base`` rises WITH it (0.1653 -> 0.2205) instead
            # of falling, so the level is added ON TOP of the observation noise
            # rather than displacing it. Total predictive scale
            # sigma*sqrt(nu/(nu-2)) goes 0.2168 -> 0.2517, per-time PPC coverage
            # over-shoots to 98.4 % against a 92 % target at the decision slice,
            # and the validation gate's "sigma_base falls versus the no-latent
            # baseline" signature inverts.
            #
            # Reverted to 0.10. The log-linear sigma_isin model (realized-vol
            # level, size, analyst range, learned coverage exponent, sector
            # offset) is retained -- it is what freed ``nu`` from its 2.5 floor
            # and improved PIT, and it does so without inflating total dispersion.
            #
            # Why the arithmetic misleads: it measures how much per-name level
            # EXISTS in the residual, not how much this model can absorb without
            # the rest of the scale expanding to match. Re-deriving 0.4718 and
            # concluding "therefore 0.40" is the trap.
            pm.Deterministic("sigma_isin_level", pt.constant(isin_level_scale))
            z_isin_level = pm.ZeroSumNormal(
                "z_isin_level", sigma=float(isin_level_scale), dims="isin"
            )
            expected_return = pm.Deterministic(
                "expected_return", mu_reg + z_isin_level, dims="isin",
            )
        else:
            # T == 1, or the latent deliberately pinned off — the pre-0.9.9.14
            # build, and the baseline arm of the validation comparison.
            expected_return = pm.Deterministic("expected_return", mu_reg, dims="isin")
        # ---- ADDITIVE, sign-correct risk + size tilts (data-learned loadings) ----
        # ``expected_return`` (= ``mu_reg``) is a ZERO-MEAN cross-sectional deviation
        # on the standardised log-uplift scale (the level lives in the GRW ``alpha``
        # intercept, ``mu_global`` having been dropped). The previous build applied a
        # MULTIPLICATIVE ``expected_return * exp(-risk_penalty·beta_z) *
        # exp(-size_penalty·size_z)`` factor, which is only a "discount" on a strictly
        # POSITIVE return: on a sign-ambiguous deviation it merely rescales the
        # magnitude and FLIPS direction for below-average names — a high-beta,
        # below-average name (``expected_return < 0``, ``beta_z > 0``) was pulled
        # *up* toward 0 instead of discounted. That sign inversion scrambled the
        # cross-section, so the mean structure could not order names monotonically;
        # combined with the group-effect funnel above the posterior collapsed toward
        # full pooling and every ISIN landed on ≈ the universe-mean upside (the
        # "stale / uniformly non-negative" ``expected_upside`` / ``log_uplift``).
        #
        # The fix is an ADDITIVE tilt ``-loading · z`` which is monotone in the
        # feature regardless of the sign of ``expected_return``: higher systematic
        # risk (``feat_avg_beta``) and a larger size-vs-own-history (``feat_mcap_global_r``,
        # = market_cap / market_cap_3yavg) ALWAYS reduce the risk-adjusted return.
        # The loadings are LEARNED (sign-fixed ``HalfNormal`` so the direction stays
        # a discount while the magnitude is identified from the cross-section), which
        # also gives the mean structure two genuinely-predictive factors and moves
        # variance off the residual ``sigma_base`` — the converse of the collapse.
        # ``risk_penalty`` / ``size_penalty`` set the loading prior scale; ``0.0``
        # disables the corresponding factor (loading pinned at 0).
        if risk_penalty > 0:
            risk_loading = pm.HalfNormal("risk_loading", sigma=float(risk_penalty))
        else:
            risk_loading = pt.constant(0.0)
        if size_penalty > 0:
            size_loading = pm.HalfNormal("size_loading", sigma=float(size_penalty))
        else:
            size_loading = pt.constant(0.0)
        if volume_penalty > 0:
            volume_loading = pm.HalfNormal("volume_loading", sigma=float(volume_penalty))
        else:
            volume_loading = pt.constant(0.0)
        risk_tilt = pm.Deterministic("risk_tilt", -risk_loading * beta_z, dims="isin")
        size_tilt = pm.Deterministic("size_tilt", -size_loading * size_z, dims="isin")
        volume_tilt = pm.Deterministic("volume_tilt", -volume_loading * volume_z, dims="isin")
        risk_adj_return = pm.Deterministic(
            "risk_adj_return",
            expected_return + risk_tilt + size_tilt + volume_tilt,
            dims="isin",
        )
        # The risk- and size-adjusted return is the GRW baseline.
        # The risk-adjusted return is the t=0 ANCHOR of the local-level state built
        # below; ``achieve_prob`` is defined there, off the filtered level.
        mu_isin = pm.Deterministic("mu_isin", risk_adj_return, dims="isin")

        # ---- Model B: rank-1 coregionalised per-series structure ----
        # The D response series share the single latent per-ISIN factor ``mu_isin``
        # through per-series loadings — a rank-1 Intrinsic Coregionalization Model
        # (ICM; PyMC "MOGP-Coregion-Hadamard"): ``reg[i,d] = alpha[d] +
        # beta_t[d]·t[i] + W[d]·mu_isin[i]`` with the coregion loading ``W``
        # (``mu_isin_loading``) anchored at ``W[0] ≡ 1`` for the primary series so
        # the factor scale is identified, and per-output noise handled by the
        # diagonal ``tau_series`` (the ICM ``kappa`` term) on the σ side below.
        #
        # Per-series level / slope are DIRECT Normals — never a cumulative
        # ``sigma_innov · z`` random walk. On the collapsed cross-section
        # (``T == 1``) the walk had a single step, so scale × innovation was a
        # non-identified product (the historic ridge: ``sigma_*_innov``
        # within-chain variance → 0, ESS ≈ 7, R-hat ≈ 1.5). On a genuine
        # lookback panel (``T > 1``) the SAME ridge reappeared along time: the
        # zero-anchored GRW deviations on ``alpha`` AND ``beta_t`` PLUS a
        # global slope were mutually aliased per time slice — with an
        # isin-constant time axis a per-time mean shift is attributable to the
        # alpha deviation, the beta deviation × t, or the slope, so the T
        # identifiable time-means were parameterised by ``2 + 2·(T-1) + 2``
        # scales/levels. Observed on the 2026-08-01 T=4 validation
        # (tune=2000, target_accept=0.95, nutpie): 190 divergences with
        # ``alpha_level`` / ``beta_slope`` at R-hat 1.06 (bulk-ESS ≈ 75) and
        # ``sigma_alpha_innov`` at 1.04 (ESS ≈ 120) while every other block
        # mixed cleanly. Fix = the T=1 direct-intercept medicine applied per
        # time step: T direct per-time intercepts (exactly identified), and a
        # per-series scalar slope ONLY when ``t_scaled`` genuinely varies
        # across ISINs (the T=1 days-to-event covariate). On an isin-constant
        # axis — including the degenerate all-zero T=1 fallback, where the
        # slope was previously sampled against zero data — the trend is
        # absorbed by the intercepts and ``beta_t`` is NOT MATERIALISED at all
        # (it was formerly published as a Deterministic pinned at 0; see the
        # note on the ``t_isin_varying`` branch below for why that was worse
        # than omitting it). Per-ISIN time structure is carried by the
        # local-level state instead.
        t_isin_varying = bool(np.any(np.ptp(panel.t_scaled, axis=0) > 1e-12))
        if T > 1:
            alpha_expr = pm.Normal("alpha_level", 0.0, 1.0, dims=("time", "y_series"))
        else:
            alpha_level = pm.Normal("alpha_level", 0.0, 1.0, dims="y_series")
            alpha_expr = alpha_level[None, :]
        alpha = pm.Deterministic("alpha", alpha_expr, dims=("time", "y_series"))

        # ``beta_t`` is materialised ONLY on an isin-varying time axis. On the
        # ``history_lookbacks`` panel ``t_scaled`` is ``np.tile``-d from the shared
        # calendar offsets, so it is identical for every ISIN and a slope
        # ``beta_t[d]·t[t]`` is exactly spanned by the T free ``alpha_level[t, d]``
        # intercepts. It used to be published as a structurally-zero Deterministic,
        # which (a) read as a fitted-and-dead parameter in the §9 summary, (b) gave
        # arviz a constant variable to divide 0/0 on when computing R-hat, and (c)
        # drew a flat zero line in the §13b "MvGRW slope per y_series" panel.
        # Not creating it is the honest form: on this axis the linear trend is
        # carried by ``alpha``, and the per-ISIN time variation is now carried by
        # the local-level state below. Downstream consumers already guard on
        # presence (``if 'beta_t' in post.data_vars``).
        if t_isin_varying:
            beta_slope = pm.Normal("beta_slope", 0.0, 0.5, dims="y_series")
            beta_expr = pt.tile(beta_slope[None, :], (T, 1))
            beta_t = pm.Deterministic("beta_t", beta_expr, dims=("time", "y_series"))
            trend_term = beta_t[None, :, :] * t_obs[:, :, None]
        else:
            trend_term = pt.zeros((n_isin, T, D))

        # ---- Local-level state: the per-ISIN latent that makes this a filter ----
        # ``mu_isin`` is the structural, TIME-CONSTANT baseline (drift regression +
        # group effects + risk/size/volume tilts). Broadcasting it unchanged across
        # the T lookback slices — the previous behaviour — meant each name's T
        # log-uplift observations were treated as T iid draws around ONE constant
        # latent. Those slices are strongly serially correlated within a name, so
        # the iid Student-t likelihood shrank the effective posterior sd of the
        # per-name level by ~√T: PSEUDO-REPLICATION, producing over-confident
        # ``expected_upside`` HDIs downstream in the screen, the CVaR risk book and
        # ``analytics.kalman_filtered_price_targets``. Combined with the zeroed
        # ``beta_t`` above, the entire time dimension contributed nothing but T
        # cross-sectionally-shared intercepts — the model had no state evolution at
        # all despite being a "Kalman" filter.
        #
        # The fix is a per-ISIN latent level that evolves as a STATIONARY AR(1)
        # across the lookback steps: ``state[i, t] = mu_isin[i] + sigma_state ·
        # z[i, t]`` with ``z[:, t] = rho·z[:, t-1] + sqrt(1-rho²)·eps[:, t]``.
        # Each name gets its own trajectory, so the panel can separate persistent
        # per-name signal from transient measurement noise instead of
        # double-counting it.
        #
        # The parameterisation is pinned by TWO measured failures, both of which
        # a plausible-looking alternative walked straight into:
        #
        # (1) A cumulative RANDOM WALK anchored at ``state[i,0] = mu_isin[i]``
        #     mis-calibrates the panel. A walk's marginal variance grows as
        #     ``sigma_state·√t``, so t=0 carries no state uncertainty and the
        #     snapshot carries the most. The 2026-08-10 full-scale run (5 605
        #     ISINs, T=4) showed exactly that: per-time posterior-predictive
        #     coverage ramped 89.9 % (t=0, intervals too NARROW) → 95.3 % → 97.2 %
        #     → 98.2 % (snapshot, too WIDE) against a 94 % target, with max R-hat
        #     1.054 and min ESS 74 at 0 divergences. Analyst uplift is marginally
        #     STATIONARY across a 1-6 month window, not diffusing.
        #
        # (2) An UNSTRUCTURED doubly-centred field fixes the calibration and
        #     destroys the identification. With one observation per ``(isin,
        #     time)`` cell, an unstructured state field has the same covariance
        #     structure as the measurement noise — both diagonal — so the two are
        #     not separately identified. Verified on synthetic data with a known
        #     state (N=300, T=4, true sigma_state 0.35): coverage went flat (2.7 pp
        #     spread) and mixing improved (min ESS 736, max R-hat 1.02), but
        #     ``sigma_state`` collapsed to 0.027 and ``corr(state_now, truth)``
        #     fell to 0.04 — the state layer had become decorative.
        #
        # Temporal CORRELATION is the only thing that distinguishes a persistent
        # state from iid noise: within-name off-diagonal covariance is
        # ``sigma_state²·rho^|Δt|``, identically zero when rho = 0. A stationary
        # AR(1) supplies that correlation (identification) AND constant marginal
        # variance (calibration), which is why it is the structure that survives.
        # ``rho`` is estimated — with thousands of names the pooled within-name
        # covariance carries ample information about it.
        #
        # Why this is identified now when the per-ISIN random effect at :2368 was
        # not: that one was dropped because "on a single cross-sectional slice
        # (collapsed T=1, ~1 obs/ISIN) that scale could not be told apart from the
        # crossed group scales or the measurement noise". T repeated observations
        # per name is exactly what breaks that tie — the reasoning does not carry
        # over to a genuine panel. On T == 1 the state correctly collapses to
        # ``mu_isin`` and no innovation parameters are created.
        #
        # Two identification guards, both load-bearing:
        #   * NON-CENTRED (``sigma_state · z``) — a centred field reproduces the
        #     classic funnel between the scale and its deviations, the same ridge
        #     that gave ``sigma_*_innov`` ESS ≈ 7 / R-hat ≈ 1.5.
        #   * ZERO-SUM OVER ``isin`` at each t — otherwise the field shares a
        #     common level shift with ``alpha_level[t, ·]`` (a market-wide move at
        #     time t is attributable to either), precisely the aliasing that
        #     produced the 2026-08-01 T=4 failure (190 divergences,
        #     ``alpha_level`` R-hat 1.06, bulk-ESS ≈ 75). Zero-sum leaves the field
        #     carrying ONLY idiosyncratic deviation, so ``alpha`` keeps the
        #     market-wide time effect and the two blocks are orthogonal.
        #
        # ``mu_isin`` needs no explicit time-centring: a stationary AR(1) has
        # prior mean 0 at every t, so the per-name level is identified against it
        # the same way any random effect is identified against its intercept.
        # (Forcing exact zero-sum over time was tried and is what produced the
        # non-identified unstructured field in (2) above.)
        #
        # Expected diagnostic signature versus the static build: ``sigma_base``
        # FALLS (persistent per-name signal moves out of the residual and into the
        # state) and the exported ``expected_upside`` HDIs WIDEN. The widening is
        # the pseudo-replication fix landing, not a regression.
        if T > 1 and state_innovation_scale > 0:
            sigma_state = pm.HalfNormal(
                "sigma_state", sigma=float(state_innovation_scale)
            )
            # Persistence of the state across lookback steps, bounded to
            # (0, RHO_MAX). BOTH ends are non-identified corners and the bound
            # matters as much as the prior:
            #   * rho -> 0 makes the state indistinguishable from iid
            #     measurement noise (see (2) above);
            #   * rho -> 1 makes the AR field a per-name CONSTANT, which is
            #     exactly what ``mu_isin`` already is — the two then trade off
            #     along a ridge. Left unbounded on the synthetic recovery check
            #     the sampler ran to rho = 0.975 and inflated ``sigma_state``
            #     to 0.83 against a true 0.35, widening every predictive
            #     interval (per-time coverage 98.7-100 % vs the 94 % target) and
            #     collapsing mixing to min ESS 46 / max R-hat 1.06.
            # Capping at 0.95 keeps the AR genuinely a *process* rather than a
            # second intercept; Beta(2, 2) is symmetric inside that range.
            state_rho = pm.Deterministic(
                "state_rho", _STATE_RHO_MAX * pm.Beta("state_rho_raw", 2.0, 2.0)
            )
            # Innovations are zero-sum ACROSS ISINS at each time step (PyMC
            # constrains the TRAILING axis, hence the (time, isin) layout). The
            # AR recursion below is a linear combination of zero-sum vectors, so
            # every state slice inherits the constraint exactly — no aliasing
            # with ``alpha_level[t, ·]``, and no redundant directions for the
            # mass matrix to chase.
            z_state = pm.ZeroSumNormal(
                "z_state", sigma=1.0, dims=("time", "isin"), n_zerosum_axes=1
            )
            # Stationary AR(1) with UNIT marginal variance at every t: scaling
            # the innovations by sqrt(1 - rho^2) is what keeps var(z[:, t])
            # constant instead of growing like a random walk. T is a small
            # Python int (4 on the production panel), so the recursion is
            # unrolled rather than run through pytensor.scan.
            innov_scale = pt.sqrt(1.0 - state_rho ** 2)
            steps = [z_state[0]]
            for _t in range(1, T):
                steps.append(state_rho * steps[-1] + innov_scale * z_state[_t])
            state_dev = sigma_state * pt.stack(steps, axis=0).T  # (isin, time)
        else:
            # T == 1, or the state deliberately pinned off (the static twin used
            # as the §9b model-comparison baseline).
            state_dev = pt.zeros((n_isin, T))
        state_path = pm.Deterministic(
            "state_path", mu_isin[:, None] + state_dev, dims=("isin", "time")
        )
        # The DECISION-relevant latent: the filtered level at the final (snapshot)
        # time step, not the t=0 anchor. This is what the screen, the price-target
        # Monte-Carlo and the analytics export consume — resolved through
        # ``pymc_kalman_filter_pt.KALMAN_SCREEN_LATENT`` so there is one name for it.
        state_now = pm.Deterministic("state_now", state_path[:, -1], dims="isin")
        # Achievement probability keyed on the FILTERED level, so the downstream
        # ``kalman_gain`` / confidence readouts describe "now" rather than the
        # t=0 structural anchor. Collapses to sigmoid(risk_adj_return) when T == 1.
        pm.Deterministic("achieve_prob", pm.math.sigmoid(state_now), dims="isin")

        # ---- Coregion loadings on the shared per-ISIN factor ----
        # ``mu_isin`` is the single latent factor; loading it per series (primary
        # anchored at 1) makes it the common factor *on the feat_log_uplift scale*
        # (so the ``panel_posterior_upside`` de-standardisation stays exact) while
        # weakly-correlated series (feat_pt_drift) load freely instead of dragging
        # the latent. This is the rank-1 ICM weight ``W`` (n_series × 1).
        if D > 1:
            # Sign-FIXED, unit-centred loading: a free ``Normal`` loading let
            # ``loading × mu_isin`` flip sign (the latent factor and its loading are
            # only jointly identified up to a shared sign), giving a bimodal,
            # non-mixing ridge. A ``LogNormal`` (median 1, strictly positive) anchors
            # the secondary series to load in the SAME direction as the primary
            # (anchored at 1), removing the sign-flip multimodality; ``sigma=0.3``
            # keeps it weakly-informative around parity.
            loading_free = pm.LogNormal(
                "mu_isin_loading", mu=0.0, sigma=0.3, dims="y_series_free"
            )
            loading = pm.Deterministic(
                "mu_isin_loading_full",
                pt.concatenate([pt.ones(1), loading_free]),
                dims="y_series",
            )
        else:
            loading = pt.ones(1)

        # ``state_path`` (isin, time) replaces the time-constant ``mu_isin``
        # broadcast: the shared latent factor now MOVES per name across the panel,
        # and the per-series ICM loadings apply to that trajectory.
        regression = (
                alpha[None, :, :]
                + trend_term
                + loading[None, None, :] * state_path[:, :, None]
        )

        # ---- Model A: richer heteroscedastic σ (cv term, Exponential base) ----
        # ``/ precision_weight`` (median-normalised √n) rather than raw √n keeps the
        # median name's scale O(1) on the z-scored response, so the per-ISIN signal
        # lands in the informative region of the Student-t and ``sigma_expected_return``
        # stays > 0 (no full-pooling collapse). The relative precision still widens
        # few-analyst names and tightens high-coverage names.
        #
        # The EXPONENT on that weight is LEARNED, not pinned at 1. Writing
        # ``/ precision_weight`` asserts sd ∝ n^-0.5 — the textbook standard-error
        # rate, which assumes analyst targets are independent draws about a common
        # truth. They are not: analysts anchor on each other and on the same
        # published guidance, so dispersion falls far more slowly than √n.
        # Measured on the 2026-08-16 MV (24 coverage levels, 6,270 names), the
        # observed sd of ``feat_log_uplift`` scales as **n^-0.306**, not n^-0.5:
        #
        #     n_analysts   observed sd   1/√n predicts   ratio
        #        1-2          0.3489        0.3489       1.00
        #        5-8          0.2056        0.1962       1.05
        #        9-14         0.1957        0.1468       1.33
        #       15-24         0.1580        0.1153       1.37
        #       25-60         0.1404        0.0878       1.60
        #
        # Fixing the rate at 0.5 therefore hands well-covered names an observation
        # scale up to 60 % too small, their standardised residuals run ~1.6× too
        # large, and because ``nu`` is a single global parameter it collapses onto
        # its 2.5 floor to accommodate them. At ν = 2.53 the Student-t variance is
        # ν/(ν-2) = 4.77·σ², which over-disperses the whole predictive — the
        # 2026-08-16 PPC measured a replicated std of 1.40-1.60 against an observed
        # 1.03, and both the PIT ECDF and the mean t-statistic rejected.
        #
        # ``precision_weight`` is √n, so the exponent here is 2× the rate: the
        # observed n^-0.306 corresponds to ≈ 0.61, and 1.0 reproduces the previous
        # model exactly. The prior is deliberately centred on the OLD value rather
        # than on the empirical estimate — a posterior that still lands near 0.6 is
        # then evidence from the likelihood, not an echo of a prior fitted to the
        # same data.
        #
        # Identification is clean: ``precision_weight`` is median-normalised, so the
        # median name has weight 1 and ``sigma_isin`` = ``sigma_base·(1+cv)`` there
        # for ANY exponent. ``sigma_base`` is fixed by the median, the exponent by
        # the spread across coverage levels; they do not trade off.
        sigma_base = pm.Exponential("sigma_base", 1.0)
        sigma_n_exponent = pm.TruncatedNormal(
            "sigma_n_exponent", mu=1.0, sigma=0.5, lower=0.0, upper=2.0,
        )
        # ---- Log-linear observation-scale model (2026-08-16) -----------------
        # Previously this was ``sigma_base * (1 + cv) / precision_weight**tau`` --
        # two drivers, of which one (coverage) was mis-specified. Measured on the
        # live 6,533-row panel, correlation of each candidate with log|residual|
        # after an OLS fit of the 17 drift features:
        #
        #     cv = pt_stddev/price   +0.2245   was in
        #     log market_cap         -0.2100   NOT in, not even catalogued
        #     volatility_1m          +0.1924   NOT in
        #     volatility_1y          +0.1897   NOT in
        #     log n_analysts         -0.1696   was in
        #     feat_pt_range_norm     +0.1150   DOCUMENTED as in, never wired
        #     feat_vol_drift         -0.0349   documented as in, never wired,
        #                                      and useless regardless
        #
        # The scale is now log-linear in the z-scored drivers. Log-linear keeps
        # positivity automatic (no clipping, no HalfNormal on a product) and makes
        # each ``delta_*`` read directly as the log-sigma change per 1 sd of its
        # driver. ``log1p(cv)`` preserves the previous multiplicative ``(1 + cv)``
        # exactly, so the old model is the special case delta_* = 0, tau = 1.
        #
        # Identification: ``precision_weight`` is median-normalised and every
        # driver is mean-zero after z-scoring, so ``sigma_base`` remains pinned by
        # the median name and the deltas are identified by cross-sectional spread.
        # They do not trade off with the intercept.
        delta_vol = pm.Normal("sigma_delta_vol_level", 0.0, 0.5)
        delta_mcap = pm.Normal("sigma_delta_log_mcap", 0.0, 0.5)
        delta_range = pm.Normal("sigma_delta_range", 0.0, 0.5)

        # ---- Group structure on the VARIANCE ---------------------------------
        # The model carries crossed group effects on the MEAN but had none on the
        # scale, while residual sd moves 2.14x across sectors (Health Care 0.184
        # vs the tightest 0.086), 1.82x across size_class and 1.68x across
        # trading_region. No per-name driver can absorb a group-level shift of
        # that size, so a single global tail parameter was taking the strain --
        # which is why ``nu`` pinned at its 2.5 floor.
        #
        # Fixed scale (GROUP_EFFECT_SCALE), ZeroSumNormal, exactly as the mean-side
        # crossed effects: a learned hierarchical scale over a single cross-section
        # is the non-identified construction this model already abandoned there.
        # Sum-to-zero keeps it from aliasing with ``sigma_base``.
        log_sigma = (
            pt.log(sigma_base)
            + pt.log1p(cv_data)
            + delta_vol * vol_level_data
            + delta_mcap * log_mcap_data
            + delta_range * range_norm_data
            - sigma_n_exponent * pt.log(precision_weight_data)
        )
        if "sector" in idx_data_vars:
            sigma_sector = pm.ZeroSumNormal(
                "sigma_sector_offset", sigma=GROUP_EFFECT_SCALE, dims="sector"
            )
            log_sigma = log_sigma + sigma_sector[idx_data_vars["sector"]]
        sigma_isin = pm.Deterministic("sigma_isin", pt.exp(log_sigma), dims="isin")
        # Per-output noise diagonal (the ICM ``kappa``): each response series gets
        # its own multiplicative noise scale so a noisier output (feat_pt_drift)
        # does not inflate the primary series' σ (and vice-versa). The primary
        # series is anchored at 1 so the absolute ``sigma_base`` scale stays
        # identified (a free primary scale would form a ``sigma_base · tau[0]``
        # product); the others are HalfNormal(0.5) deviations from it. Collapses to
        # a no-op (all ones) when D == 1.
        if D > 1:
            tau_free = pm.HalfNormal("sigma_series_free", sigma=0.25, dims="y_series_free")
            tau_series = pm.Deterministic(
                "sigma_series",
                pt.concatenate([pt.ones(1), tau_free]),
                dims="y_series",
            )
            sigma_obs = sigma_isin[:, None] * tau_series[None, :]  # (isin, y_series)
        else:
            sigma_obs = sigma_isin[:, None]  # (isin, 1)

        # ---- Per-TIME noise scale (the lookback slices are not equally clean) ----
        # ``sigma_isin`` is a per-name scale applied identically to every time
        # slice. That is wrong for a history panel and the data says so: with the
        # state layer entirely OFF, the 2026-08-10 full-scale run still produced a
        # monotone per-time posterior-predictive coverage ramp — 90.7 % at the
        # oldest lookback (intervals too NARROW), 94.3 %, 95.7 %, 96.7 % at the
        # snapshot (too WIDE) against a 94 % target. Because the ramp survives with
        # no time-varying latent at all, it is not a state-dynamics artefact: the
        # OBSERVATION noise genuinely differs by lookback. That is what one would
        # expect — ``price_target_{6m,3m,1m}_ago`` / ``price_*_ago`` are staler and
        # thinner-covered than today's consensus, and a small share of history
        # cells are snapshot-filled outright.
        #
        # One multiplicative scale per time step fixes it, anchored at the
        # SNAPSHOT (the final index, and the slice every downstream decision reads)
        # so ``sigma_base`` keeps its meaning and no ``sigma_base · tau`` product
        # appears. ``LogNormal(0, 0.25)`` is median-1 and strictly positive — the
        # same median-anchored form used for the ICM loading — so the prior is
        # neutral about which lookbacks are noisier and the data decides. Costs
        # ``T-1`` parameters (3 on the production panel), amply identified by
        # thousands of names.
        if T > 1:
            tau_time_free = pm.LogNormal(
                "sigma_time_free", mu=0.0, sigma=0.25, dims="time_free"
            )
            tau_time = pm.Deterministic(
                "sigma_time",
                pt.concatenate([tau_time_free, pt.ones(1)]),
                dims="time",
            )
            sigma_full = sigma_obs[:, None, :] * tau_time[None, :, None]
        else:
            sigma_full = sigma_obs[:, None, :]
        # Robust-likelihood degrees of freedom, lower-bounded at ν₀ = 2.5.
        # A plain ``Gamma(2, 0.1)`` places non-trivial mass below ν = 2, where the
        # Student-t has infinite variance and — critically — the joint density is
        # *improper* along the ``sigma_base → 0, nu → 0`` ray: the ``nu·log σ``
        # scale penalty vanishes as ν → 0, so the density can be driven arbitrarily
        # high in that corner. With a single shared ``mu_isin`` forced to explain
        # three heterogeneous response series the residuals are large everywhere,
        # and the sampler slid straight into that corner — ν → 0.06,
        # ``sigma_base`` / ``sigma_expected_return`` / ``beta`` / group effects all
        # → 0, the mean structure switched off, and every ISIN collapsed onto the
        # cross-sectional mean upside (the static ≈ 30 % screen / flat shrinkage
        # line). Shifting by a hard floor ν ≥ 2.5 guarantees finite variance and
        # removes the improper corner, so ``σ → 0`` is again penalised by
        # ``2.5·log σ → −∞`` and the per-ISIN mean stays identified; the ``Gamma``
        # tail still lets the data choose genuinely heavy tails above the floor.
        # Exposed as the ``nu`` Deterministic so downstream consumers are unchanged.
        _lik = (likelihood or ("student_t" if robust else "normal")).lower()
        if _lik not in ("student_t", "mixture", "normal"):
            raise ValueError(
                f"likelihood must be 'student_t', 'mixture' or 'normal', got {_lik!r}"
            )

        if _lik == "student_t":
            nu = pm.Deterministic("nu", 2.5 + pm.Gamma("nu_tail", alpha=2.0, beta=0.1))
            pm.StudentT(
                "target_pct_obs",
                nu=nu,
                mu=regression,
                sigma=sigma_full,
                observed=Y_obs,
                dims=("isin", "time", "y_series"),
            )
        elif _lik == "mixture":
            # ---- Two-component scale mixture: tight core + wide outlier ----
            # WHY, over the Student-t above. A t has ONE shape parameter, so it
            # cannot be tight in the bulk and heavy in the tail at the same time:
            # to cover the outliers it makes the whole distribution heavy, and the
            # predictive over-disperses. Measured on the 2026-08-16 panel, the
            # shipped t sat at nu ~ 2.6-2.7 (its 2.5 floor) and replicated a std of
            # 1.31-1.39 against an observed 0.97, which is what failed the T=std,
            # T=mean and PIT checks together.
            #
            # Fitting both families directly to the standardised response
            # (n = 25,962) is unambiguous:
            #
            #     single Student-t (nu=3.76)   loglik -34872.4  k=3  BIC 69775.2
            #     2-component normal mixture   loglik -34423.8  k=5  BIC 68898.3
            #
            # The mixture wins by dBIC 877, and its implied sd is 1.0000 against
            # the data's 1.0000 (the single t manages 1.024). The fitted split is a
            # 76 % core at sd 0.634 and a 24 % outlier component at sd 1.524, i.e.
            # an inflation of ~2.4x -- which is the priors' rough location below.
            #
            # ``outlier_inflation >= 1`` by construction, so the second component
            # is always the wider one and the label-switching that plagues free
            # scale mixtures cannot occur.
            outlier_weight = pm.Beta("outlier_weight", alpha=2.0, beta=8.0)
            outlier_inflation = pm.Deterministic(
                "outlier_inflation", 1.0 + pm.HalfNormal("outlier_inflation_excess", 2.0)
            )
            # ``nu`` is retained as a DERIVED summary, not a fitted parameter: the
            # Student-t degrees of freedom whose excess kurtosis matches this
            # mixture's. Downstream consumers read it (the forward-return Monte
            # Carlo in _price_target_mc draws StudentT(df=nu), run_diagnostics
            # summarises it), and dropping it would break them silently. It is a
            # tail-weight summary of the fitted mixture -- do NOT read it as an
            # estimated dof the way the student_t branch's ``nu`` is.
            _w, _k2 = outlier_weight, outlier_inflation ** 2
            _m2 = (1.0 - _w) + _w * _k2
            _m4 = 3.0 * ((1.0 - _w) + _w * _k2 ** 2)
            _exkurt = pt.maximum(_m4 / _m2 ** 2 - 3.0, 1e-6)
            pm.Deterministic("nu", pt.clip(6.0 / _exkurt + 4.0, 2.5, 500.0))
            pm.Mixture(
                "target_pct_obs",
                w=pt.stack([1.0 - outlier_weight, outlier_weight]),
                comp_dists=[
                    pm.Normal.dist(mu=regression, sigma=sigma_full),
                    pm.Normal.dist(mu=regression, sigma=sigma_full * outlier_inflation),
                ],
                observed=Y_obs,
                dims=("isin", "time", "y_series"),
            )
        else:
            pm.Normal(
                "target_pct_obs",
                mu=regression,
                sigma=sigma_full,
                observed=Y_obs,
                dims=("isin", "time", "y_series"),
            )

    return model