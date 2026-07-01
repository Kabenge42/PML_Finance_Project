"""Lightweight logging shim for the GEIB dashboard.

Mirrors the ``logger``/``schema``/``tbl`` helpers the ``feature_factory/*.pyi``
reference stubs import from a Dash-Enterprise ``logger`` module, so the ported
chart logic can keep its ``logger.debug(schema(df))`` / ``tbl(df)`` calls
verbatim.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("geib")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def schema(df: pd.DataFrame) -> str:
    """Return a short one-line description of *df*'s columns and dtypes."""
    try:
        return "schema: " + ", ".join(
            f"{col}:{dtype}" for col, dtype in df.dtypes.astype(str).items()
        )
    except Exception:  # pragma: no cover - defensive
        return "<schema unavailable>"


def tbl(df: pd.DataFrame) -> str:
    """Return a short ``<rows x cols>`` summary of *df*."""
    try:
        return f"<{len(df)} rows x {df.shape[1]} cols>"
    except Exception:  # pragma: no cover - defensive
        return "<tbl unavailable>"
