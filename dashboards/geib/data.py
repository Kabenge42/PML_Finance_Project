"""Data access layer for the GEIB dashboard.

Reads the **v2** Kalman export: ``analytics.kalman_filtered_price_targets_v2``
(DDL in ``sql_scripts/analytics/kalman_filtered_price_targets_v2.sql``), joined
to ``analytics."04_panel_frame_v2"`` on ``isin``, using the project's engine
factory :func:`probabilistic_ml_model.data_utils.data_utils.get_analytics_engine`.

Why two tables when this module used to read one
------------------------------------------------
The board was pointed at ``analytics.kalman_filtered_price_targets`` — the v1
table — which **no longer exists in the database**. The v2 pipeline writes its
own ``_v2``-suffixed tables and nothing recreated v1, so every card was
rendering against the empty-frame fallback: not a crash, just a board that
looked like "no data matched the filters".

v2's canonical table is deliberately narrower than v1's (77 columns against
102): it dropped the duplicated risk columns and never carried the descriptive
block. Of the 96 columns this module declares, 56 are in the canonical table and
**38 more live in the panel frame** under ``feat_`` names — beta, the analyst
rating mix, the Piotroski set, the price/price-target ladders. Both frames are
written by the same run, carry the same ``run_id``, and hold one unique row per
ISIN, so the join is 1:1 and costs one extra read.

:data:`VINTAGE_COLUMN` is checked rather than trusted. A mixed vintage is the
failure this join makes possible and it would be invisible in the rendered
board — the columns would simply describe a different fit from the one whose
returns are plotted beside them.

The frame is cached in-process; call :func:`refresh` to invalidate. When
``DB_URL`` is unset or the load fails the loader returns an empty frame carrying
the full column set so the app still renders.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

try:  # reuse the project's analytics engine factory
    from probabilistic_ml_model.data_utils.data_utils import get_analytics_engine
except Exception:  # pragma: no cover - import guard for standalone runs
    get_analytics_engine = None  # type: ignore[assignment]

logger = logging.getLogger("geib.data")

TABLE_NAME = "kalman_filtered_price_targets_v2"
PANEL_TABLE = "04_panel_frame_v2"

#: Provenance column both frames carry. The join is only valid while they agree.
VINTAGE_COLUMN = "run_id"

#: ``dashboard name -> panel column``. v2 kept the catalogue's ``feat_`` prefix on
#: the descriptive block; the board's charts, labels and filter values were all
#: written against the v1 spellings, so the rename happens once here rather than
#: in fourteen chart modules.
#:
#: ``analyst_rating`` is the exception that proves the rule: v2 carries it on the
#: CANONICAL table as ``feat_analyst_rating``, so it is resolved from whichever
#: frame supplies it rather than assumed to be panel-only.
PANEL_RENAME: dict[str, str] = {
    "beta": "feat_avg_beta",
    "analyst_rating": "feat_analyst_rating",
    "analyst_conviction": "feat_analyst_conviction",
    "analyst_bullish_pct": "feat_analyst_bullish_pct",
    "analyst_bearish_pct": "feat_analyst_bearish_pct",
    "analyst_neutral_pct": "feat_analyst_neutral_pct",
    "n_buys": "feat_buys",
    "n_holds": "feat_holds",
    "n_sells": "feat_sells",
    "piotroski_f_score_median": "feat_median_piotroski_f_score",
    "piotroski_f_score_fy": "feat_piotroski_f_score_fy",
    "piotroski_f_score_neg1fy": "feat_piotroski_f_score_neg1fy",
    "piotroski_f_score_neg2fy": "feat_piotroski_f_score_neg2fy",
    "piotroski_f_score_neg3fy": "feat_piotroski_f_score_neg3fy",
    "pt_achievement_1y": "feat_pt_achievement_1y",
    "pt_range_hit_rate": "feat_pt_range_hit_rate",
}

#: Panel columns the board uses under their own names -- the consensus target
#: levels and the two lookback ladders the PT-convergence and structural-forecast
#: cards plot.
PANEL_VERBATIM: list[str] = [
    "price_target_median", "price_target_high", "price_target_low",
    "price_target_1w_ago", "price_target_mtd_ago", "price_target_1m_ago",
    "price_target_qtd_ago", "price_target_3m_ago", "price_target_6m_ago",
    "price_target_ytd_ago", "price_target_1y_ago",
    "price_5d_ago", "price_1w_ago", "price_1m_ago", "price_qtd_ago",
    "price_3m_ago", "price_6m_ago", "price_1y_ago", "price_3y_ago",
    "price_5y_ago",
    "feat_no_opinion",
]

# Column inventory mirrors the analytics DDL exactly (text identifiers first,
# then the double-precision numeric columns).
IDENTIFIER_COLUMNS: list[str] = [
    "isin",
    "ticker",
    "name",
    "region",
    "trading_region",
    "country_name",
    "trading_country_name",
    "unit_name",
    "exchange_name",
    "sector",
    "industry",
    "style_class",
    "size_class",
    # Categorical earnings-timing coords the global filter panel exposes. The
    # live table also carries the raw date coords (income_statement_report_date,
    # next_earnings, fy_end_date, …) via ``SELECT *``; they are not declared
    # here because the board does not surface dates as filters.
    "next_earnings_when",
    "next_earnings_status",
]

NUMERIC_COLUMNS: list[str] = [
    "expected_return_kalman",
    "price_target_kalman",
    "kalman_gain",
    "original_price",
    "original_target",
    "market_cap",
    "enterprise_value",
    "expected_pt_hdi_lo",
    "expected_pt_hdi_hi",
    "risk_adj_return",
    "er_mean",
    "er_sd",
    "er_p05",
    "er_p50",
    "er_p95",
    "mc_prob_pos",
    "p_upside_pos_cond",
    "cvar_book_weight",
    "cvar_5pct_kalman",
    "reward_to_cvar",
    "expected_sharpe_ratio",
    # v2 risk block. ``tail_risk`` is STARR's denominator,
    # max(-cvar05, k*er_sd, MIN_TAIL_RISK) -- see charts/kelly.py, which mirrors
    # the same two constants because the dashboard must not import the PyMC
    # stack. ``out_of_support`` flags a name the screen suppressed.
    "tail_risk",
    "ret_vol_ratio",
    "band_width",
    "prob_pos",
    "out_of_support",
    # ``shrink_gain`` is the weight the forecast-error update puts on the name's
    # own smoothed observation (1 - it is the weight on the pooled drift +
    # hierarchy prediction); ``expected_upside_sd`` is the posterior sd of
    # expected upside.
    #
    # THREE v1 COLUMNS ARE GONE AND ARE NOT COMING BACK:
    #   ``expected_vol_kalman`` -- retired 2026-08-27 as a duplicate of ``er_sd``,
    #     which it had equalled by construction since the ISIN-merge fix. That
    #     equality is ``compute_cvar_aware_book``'s own alignment self-check.
    #   ``kalman_variance``     -- the price-target LEVEL variance in currency^2,
    #     which ``metrics.return_volatility`` divided by spot to get a return sd.
    #     ``er_sd`` is that return sd directly, in raw decimal, with no unit
    #     conversion to get wrong.
    #   ``signal_strength``     -- no v2 successor. The board now sizes and ranks
    #     on ``p_upside_pos_cond``, P(risk-adjusted forward return > 0), which is
    #     the screen's documented primary ranking column and is bounded [0, 1].
    "shrink_gain",
    "expected_upside_sd",
    # CAPM market sensitivity / analyst-consensus target levels.
    "beta",
    "implied_upside",
    "mcap_country_r",
    "price_target_median",
    "price_target_high",
    "price_target_low",
    # Analyst rating-mix block (0.9.9.9 export additions). ``analyst_rating``
    # is the raw 1-5 vendor consensus, higher = more bullish.
    "analyst_rating",
    "analyst_conviction",
    "analyst_bullish_pct",
    "analyst_bearish_pct",
    "analyst_neutral_pct",
    "n_analysts",
    "n_buys",
    "n_holds",
    "n_sells",
    "feat_no_opinion",
    # Piotroski F-score composites (0-9 per fiscal year; ``_median`` is the
    # fused-model drift feature, the per-year components are analytics-only).
    "piotroski_f_score_median",
    "piotroski_f_score_fy",
    "piotroski_f_score_neg1fy",
    "piotroski_f_score_neg2fy",
    "piotroski_f_score_neg3fy",
    # Consensus price-target achievement diagnostics.
    "pt_achievement_1y",
    "pt_range_hit_rate",
    # Price-target history ladder (consensus target as of the suffix lookback).
    "price_target_1w_ago",
    "price_target_mtd_ago",
    "price_target_1m_ago",
    "price_target_qtd_ago",
    "price_target_3m_ago",
    "price_target_6m_ago",
    "price_target_ytd_ago",
    "price_target_1y_ago",
    # Price history ladder (spot as of the suffix lookback).
    "price_5d_ago",
    "price_1w_ago",
    "price_1m_ago",
    "price_qtd_ago",
    "price_3m_ago",
    "price_6m_ago",
    "price_1y_ago",
    "price_3y_ago",
    "price_5y_ago",
    # Earnings-calendar distances (bigint / double in the DDL; float64 here so
    # the empty-frame fallback carries NaN support).
    "days_to_next_earnings",
    "days_since_last_report",
    "days_to_next_fy_end",
    "days_to_next_fiscal_quarter",
    "days_to_next_report",
    "days_to_expected_report",
    "days_since_fy_end",
]

# Raw date coords carried through ``SELECT *`` from the analytics DDL. Declared
# here so they survive the empty-frame fallback with a datetime dtype and are
# coerced from whatever pandas / the driver infers. Consumed by the Kalman
# structural-forecast chart (event lines), the KPI data-as-of stamp
# (``last_updated``) and available to any future date-aware panel.
DATE_COLUMNS: list[str] = [
    "last_updated",
    "next_earnings",
    "fy_end_date",
    "next_fiscal_quarter",
    "income_statement_report_date",
    "next_income_statement_report_date",
    "next_fy_end_date",
    "expected_report_date",
]

ALL_COLUMNS: list[str] = IDENTIFIER_COLUMNS + NUMERIC_COLUMNS + DATE_COLUMNS

_CACHE: Optional[pd.DataFrame] = None


def _empty_dtype(col: str) -> str:
    """Return the empty-frame dtype for *col* (numeric / datetime / object)."""
    if col in NUMERIC_COLUMNS:
        return "float64"
    if col in DATE_COLUMNS:
        return "datetime64[ns]"
    return "object"


def _empty_frame() -> pd.DataFrame:
    """Return an empty frame carrying every expected column with a sane dtype."""
    data = {col: pd.Series(dtype=_empty_dtype(col)) for col in ALL_COLUMNS}
    return pd.DataFrame(data)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to float, date coords to datetime, identifiers to str."""
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in IDENTIFIER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("object")
    return df


def _join_panel(df: pd.DataFrame, engine: object, schema: str) -> pd.DataFrame:
    """Left-join the panel frame's descriptive block onto the canonical frame.

    The canonical v2 table carries the decision columns; the panel frame carries
    the descriptive ones the board plots beside them. Pulls only the columns
    actually wanted rather than ``SELECT *`` -- the panel is 204 columns wide and
    the board uses 37 of them.

    **Vintage is asserted, not assumed.** If the two frames come from different
    runs the join still succeeds and the board still renders; it would just be
    describing one fit with another fit's features, which no chart could reveal.
    On a mismatch the panel block is dropped and the canonical frame returned
    alone, so the cards that depend on it degrade to empty rather than to wrong.

    A missing panel table is a WARNING, not an error: the decision cards work
    without it.
    """
    wanted = {src: dash for dash, src in PANEL_RENAME.items()} | {c: c for c in PANEL_VERBATIM}
    # Anything the canonical frame already supplies is left alone -- it is the
    # authority for its own columns, and re-joining it would collide.
    need = [src for src in wanted if src not in df.columns]
    if not need:
        return df.rename(columns={s: d for s, d in wanted.items() if s in df.columns})

    cols = ", ".join(f'"{c}"' for c in ["isin", VINTAGE_COLUMN, *need])
    try:
        panel = pd.read_sql(f'SELECT {cols} FROM {schema}."{PANEL_TABLE}"', engine)
    except Exception as exc:  # pragma: no cover - network/db dependent
        logger.warning(
            "Panel frame %s.%s is unavailable (%s). The decision cards render "
            "normally; beta / Piotroski / rating-mix / price-ladder cards will "
            "be empty.", schema, PANEL_TABLE, exc,
        )
        return df.rename(columns={s: d for s, d in wanted.items() if s in df.columns})

    left = set(df.get(VINTAGE_COLUMN, pd.Series(dtype=object)).dropna().unique())
    right = set(panel.get(VINTAGE_COLUMN, pd.Series(dtype=object)).dropna().unique())
    if left != right:
        logger.error(
            "VINTAGE MISMATCH: %s is at run_id %s but %s is at %s. Dropping the "
            "panel block rather than describing one run with another's features. "
            "Re-run the v2 export so both tables carry one run_id.",
            TABLE_NAME, sorted(left) or "<none>", PANEL_TABLE, sorted(right) or "<none>",
        )
        return df.rename(columns={s: d for s, d in wanted.items() if s in df.columns})

    panel = panel.drop(columns=[VINTAGE_COLUMN])
    merged = df.merge(panel, on="isin", how="left", validate="one_to_one")
    logger.info(
        "Joined %d panel column(s) from %s.%s at run_id %s",
        len(need), schema, PANEL_TABLE, next(iter(left), "<unknown>"),
    )
    return merged.rename(columns={s: d for s, d in wanted.items() if s in merged.columns})


def _load_from_db() -> pd.DataFrame:
    """Load the analytics table, failing soft to an empty frame."""
    if get_analytics_engine is None:
        # An IMPORT failure, not a connection failure — probabilistic_ml_model is
        # not on sys.path. Every card then renders against an empty frame, which
        # looks like "no data matched the filters" rather than a broken install,
        # so say exactly what is wrong and how to fix it.
        logger.error(
            "probabilistic_ml_model.data_utils is not importable, so "
            "get_analytics_engine is None and the board will render EMPTY. "
            "The repo root is missing from sys.path — launch via "
            "`python dashboards/global_equity_investment_dashboard.py` from the "
            "repo root (its launcher inserts the root), or set PYTHONPATH to it."
        )
        return _empty_frame()

    if not os.environ.get("DB_URL"):
        logger.error(
            "DB_URL is not set, so the board will render EMPTY. "
            "Run `. .\\set_env.ps1` before launching."
        )
        return _empty_frame()

    schema = os.environ.get("DB_ANALYTICS_SCHEMA", "analytics")
    try:
        engine = get_analytics_engine()
        df = pd.read_sql(f'SELECT * FROM {schema}."{TABLE_NAME}"', engine)
    except Exception as exc:  # pragma: no cover - network/db dependent
        logger.error("Failed to load %s.%s: %s", schema, TABLE_NAME, exc)
        return _empty_frame()

    df = _join_panel(df, engine, schema)
    logger.info("Loaded %d rows from %s.%s", len(df), schema, TABLE_NAME)
    return _coerce_dtypes(df)


def get_data(force_reload: bool = False) -> pd.DataFrame:
    """Return a copy of the cached analytics frame.

    Parameters
    ----------
    force_reload
        Reload from the database, bypassing the cache.

    Returns
    -------
    pandas.DataFrame
        A defensive copy so downstream callbacks may mutate freely.
    """
    global _CACHE
    if _CACHE is None or force_reload:
        _CACHE = _load_from_db()
    return _CACHE.copy()


def refresh() -> None:
    """Invalidate the in-process cache so the next ``get_data`` reloads."""
    global _CACHE
    _CACHE = None