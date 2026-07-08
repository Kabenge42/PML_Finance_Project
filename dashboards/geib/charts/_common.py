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


def top_label(
        df: Optional[pd.DataFrame],
        value_column: str,
        label_column: str = "name",
) -> Optional[str]:
    """Return the ``label_column`` of the row with the largest ``value_column``.

    Used to pick a data-driven default (e.g. the highest-``market_cap`` name) from
    the globally-filtered frame. Returns ``None`` when *df* is empty or the columns
    are missing / all-NaN, so callers can fall back gracefully.
    """
    try:
        subset = df.dropna(subset=[value_column])
        if len(subset) == 0:
            return None
        return subset.loc[subset[value_column].idxmax(), label_column]
    except Exception:  # pragma: no cover - defensive (missing column / bad dtype)
        return None


def scoped_filter(
        df: Optional[pd.DataFrame],
        column: str,
        current: Any,
        *,
        multi: bool = False,
        include_all: bool = False,
        all_label: str = "All",
        fallback: Any = None,
) -> tuple[list[dict], Any]:
    """Return ``(options, value)`` for a chart-local dropdown scoped to *df*.

    *df* must be the **globally-filtered** frame
    (``filter_data(get_data(), **kwargs)``), so the returned options are exactly
    the ``column`` values still present under the active global filters, and the
    selected ``value`` is reconciled against them. This is the single source of
    truth that keeps every per-chart Stock / Sector / Country / Currency picker
    consistent with the board's global filters: a local dropdown can never offer
    (or stay set to) an entity the global filters have removed.

    Parameters
    ----------
    df
        Globally-filtered data. ``None`` / empty yields an empty option list.
    column
        Coord column the dropdown selects on (``name`` / ``sector`` /
        ``country`` / ``unit`` …). Must match the ``pml.pml_df`` column name.
    current
        The dropdown's current ``value`` (a list when *multi*, else a scalar).
    multi
        Whether the dropdown is multi-select.
    include_all
        Prepend an "all" sentinel option; single-select convenience filters use
        ``all_label`` as their "no constraint" value.
    all_label
        Label/value of the sentinel when *include_all*.
    fallback
        Preferred single-select value when *current* is no longer available (e.g.
        the highest-``market_cap`` name from :func:`top_label`). Used only if it
        is itself available; otherwise the sentinel / first-available rule applies.

    Returns
    -------
    tuple[list[dict], Any]
        ``(options, value)``. For *multi*, ``value`` is the subset of *current*
        still available (an empty selection is preserved as "no local
        constraint"). For single-select, ``value`` is *current* when still valid,
        else *fallback* (when available), else the sentinel (when *include_all*)
        or the first available value.
    """
    available = column_values(df, column)
    options = [{"label": v, "value": v} for v in available]
    if include_all:
        options.insert(0, {"label": all_label, "value": all_label})

    if multi:
        value = [v for v in (current or []) if v in available]
        return options, value

    valid = set(available)
    if include_all:
        valid.add(all_label)
    if current in valid:
        return options, current
    if fallback is not None and fallback in valid:
        return options, fallback
    chosen = all_label if include_all else (available[0] if available else None)
    return options, chosen