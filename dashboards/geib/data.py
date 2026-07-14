"""Data access layer for the GEIB dashboard.

Loads the single source table ``analytics.kalman_filtered_price_targets`` (DDL in
``sql_scripts/analytics/kalman_filtered_price_targets.sql``) using the project's
existing engine factory
:func:`probabilistic_ml_model.data_utils.data_utils.get_analytics_engine`.

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

TABLE_NAME = "kalman_filtered_price_targets"

# Column inventory mirrors the analytics DDL exactly (text identifiers first,
# then the double-precision numeric columns).
IDENTIFIER_COLUMNS: list[str] = [
    "isin",
    "ticker",
    "name",
    "region",
    "trading_region",
    "country",
    "trading_country",
    "unit",
    "exchange",
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
    "kalman_variance",
    "kalman_gain",
    "signal_strength",
    "original_price",
    "original_target",
    "market_cap",
    "enterprise_value",
    "expected_pt_hdi_lo",
    "expected_pt_hdi_hi",
    "risk_adj_return",
    "er_mean",
    "er_p05",
    "er_p50",
    "er_p95",
    "mc_prob_pos",
    "cvar_book_weight",
    "cvar_5pct_kalman",
    "reward_to_cvar",
    # CAPM market sensitivity / analyst-consensus target levels.
    "beta",
    "price_target_median",
    "price_target_high",
    "price_target_low",
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


def _load_from_db() -> pd.DataFrame:
    """Load the analytics table, failing soft to an empty frame."""
    if get_analytics_engine is None:
        logger.warning("get_analytics_engine unavailable; returning empty frame")
        return _empty_frame()

    if not os.environ.get("DB_URL"):
        logger.warning("DB_URL not set; returning empty frame")
        return _empty_frame()

    schema = os.environ.get("DB_ANALYTICS_SCHEMA", "analytics")
    try:
        engine = get_analytics_engine()
        df = pd.read_sql(f'SELECT * FROM {schema}."{TABLE_NAME}"', engine)
    except Exception as exc:  # pragma: no cover - network/db dependent
        logger.error("Failed to load %s.%s: %s", schema, TABLE_NAME, exc)
        return _empty_frame()

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