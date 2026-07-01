"""Shared helpers for the GEIB chart modules.

Small utilities that every analytic chart needs and that were previously copied
verbatim into each module: the styled "empty / no-data" placeholder figure, a
``None``-coalescing helper for dropdown values, and the sorted-sector derivation
used to populate sector dropdowns. Centralising them keeps the charts as the
single concern of *what* to plot rather than re-implementing the same plumbing.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go


def empty_figure(message: str) -> go.Figure:
    """Return a blank figure showing *message* as a centred annotation.

    Used by every chart for the "no data" / "error" placeholder so the panes
    render a readable message instead of an empty axes box.
    """
    fig = go.Figure()
    fig.update_layout(annotations=[{"text": message, "showarrow": False, "font": {"size": 18}}])
    return fig


def coalesce(value: Any, default: Any) -> Any:
    """Return *default* when *value* is ``None``, else *value*.

    Dash passes ``None`` for an un-set / cleared dropdown; this restores the
    chart's documented default without the ``x if x is not None else d`` noise.
    """
    return default if value is None else value


def column_values(df: Optional[pd.DataFrame], column: str) -> list[str]:
    """Return the sorted, de-duplicated, non-empty values of *column* in *df*.

    Used to populate categorical dropdowns (sectors, currencies/``unit`` …) at
    layout-construction time. Defensive against a missing column or unhashable
    values so it can run before any data has loaded.
    """
    try:
        return sorted(v for v in df[column].dropna().unique().tolist() if v)
    except Exception:  # pragma: no cover - defensive (missing column / bad dtype)
        return []


def sector_values(df: Optional[pd.DataFrame]) -> list[str]:
    """Return the sorted, de-duplicated, non-empty ``sector`` values in *df*.

    Thin wrapper over :func:`column_values` kept for the existing call sites.
    """
    return column_values(df, "sector")
