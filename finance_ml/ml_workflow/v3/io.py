from __future__ import annotations

from typing import Optional, Tuple, Dict

import pandas as pd


def load_expected_returns_data(db_url: Optional[str] = None, schema: str = "public") -> tuple[pd.DataFrame, dict]:
    """Thin adapter to legacy expected_returns_v3.load_expected_returns_data.

    Keeps the old behavior while exposing a stable import path under
    finance_ml.ml_workflow.v3.io for progressive refactoring.
    """
    from expected_returns_v3 import load_expected_returns_data as _legacy

    return _legacy(db_url=db_url, schema=schema)


def load_all_stock_features(
    db_url: Optional[str] = None, schema: str = "public"
) -> tuple[pd.DataFrame, dict]:
    """Thin adapter to legacy expected_returns_v3.load_all_stock_features."""
    from expected_returns_v3 import load_all_stock_features as _legacy

    return _legacy(db_url=db_url, schema=schema)


essentially_same = load_all_stock_features  # alias for backward-compatibility


def load_analytics_table(
    db_url: Optional[str] = None,
    schema: Optional[str] = None,
    earnings_date_filter: str = "2026-01-01",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Thin adapter to legacy expected_returns_v3.load_analytics_table."""
    from expected_returns_v3 import load_analytics_table as _legacy

    return _legacy(
        db_url=db_url,
        schema=schema,
        earnings_date_filter=earnings_date_filter,
        limit=limit,
    )
