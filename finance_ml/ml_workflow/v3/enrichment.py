from __future__ import annotations

import pandas as pd


# Thin adapters to legacy enrichment helpers — allow progressive migration.

def safe_pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    from expected_returns_v3 import _safe_pct_change as _legacy

    return _legacy(current, previous)


def compute_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    from expected_returns_v3 import _compute_spread as _legacy

    return _legacy(high, low)


def enrich_with_historical_target_drift(df: pd.DataFrame, hist_available: dict[str, list[str]]) -> pd.DataFrame:
    from expected_returns_v3 import _enrich_with_historical_target_drift as _legacy

    return _legacy(df, hist_available)


def add_drift_columns(
    df: pd.DataFrame,
    current_col: str,
    horizons: list[tuple[str, str]],
    output_prefix: str,
) -> None:
    from expected_returns_v3 import _add_drift_columns as _legacy

    return _legacy(df, current_col, horizons, output_prefix)
