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


def sector_values(df: Optional[pd.DataFrame]) -> list[str]:
    """Return the sorted, de-duplicated, non-empty ``sector`` values in *df*.

    Defensive against a missing column or unhashable values so it can run during
    layout construction before any data has loaded.
    """
    try:
        return sorted(s for s in df["sector"].dropna().unique().tolist() if s)
    except Exception:  # pragma: no cover - defensive (missing column / bad dtype)
        return []
