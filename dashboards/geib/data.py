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
    "country",
    "unit",
    "exchange",
    "sector",
    "industry",
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
]

ALL_COLUMNS: list[str] = IDENTIFIER_COLUMNS + NUMERIC_COLUMNS

_CACHE: Optional[pd.DataFrame] = None


def _empty_frame() -> pd.DataFrame:
    """Return an empty frame carrying every expected column with a sane dtype."""
    data = {
        col: pd.Series(dtype="float64" if col in NUMERIC_COLUMNS else "object")
        for col in ALL_COLUMNS
    }
    return pd.DataFrame(data)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to float; leave identifiers as object strings."""
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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
