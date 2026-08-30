"""Shared helpers for the GEIB chart modules.

Small utilities that every analytic chart needs and that were previously copied
verbatim into each module: the styled "empty / no-data" placeholder figure, a
``None``-coalescing helper for dropdown values, and the sorted-sector derivation
used to populate sector dropdowns. Centralising them keeps the charts as the
single concern of *what* to plot rather than re-implementing the same plumbing.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
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


def cell(row: pd.Series, column: str) -> Any:
    """Return ``row[column]`` as a guaranteed scalar, or ``None`` when missing/NaN.

    Reading a value off a row and testing it directly (``if pd.isna(row[col])``,
    ``pd.notna(value) and ...``) is only safe while the value is a scalar: with
    duplicate column labels ``row[col]`` yields a :class:`pandas.Series`, whose
    ``pd.isna`` result is an *array* and whose boolean context raises
    ``ValueError: The truth value of a Series is ambiguous``. Collapsing to the
    first scalar here makes every downstream truthiness test unambiguous.
    """
    value = row.get(column)
    if isinstance(value, pd.Series):
        value = value.iloc[0] if len(value) else None
    if value is None or pd.isna(value):
        return None
    return value


def finite_cell(row: pd.Series, column: str, default: Any = None) -> Any:
    """Return ``row[column]`` as a finite ``float``, else *default*.

    Scalar-safe (see :func:`cell`) and strict about usability: ``None``, NaN,
    ``±inf`` and non-numeric values all collapse to *default* (``None`` by
    default, so ``is not None`` reads as "usable"; pass ``np.nan`` where the
    caller works in NaN space, e.g. :func:`numpy.isfinite` guards).
    """
    value = cell(row, column)
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


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


def name_options(df: Optional[pd.DataFrame]) -> tuple[list[dict], Optional[str]]:
    """Return ``(options, default)`` for a searchable Stock/Name dropdown.

    Options cover every non-blank ``name`` in *df* (the full universe — the
    dropdowns are searchable, so no cap is applied). The default is the
    highest-``market_cap`` name, falling back to the first name, so a panel
    renders something meaningful before the user picks; callbacks reconcile it
    against the active global filters via :func:`scoped_filter`.
    """
    try:
        names = sorted(t for t in df["name"].dropna().unique().tolist() if str(t).strip())
    except Exception:  # pragma: no cover - defensive (missing column / bad dtype)
        names = []
    options = [{"label": str(t), "value": str(t)} for t in names]
    default = top_label(df, "market_cap") or (names[0] if names else None)
    return options, default


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

# --- categorical colour scope ----------------------------------------------

#: Hue slots available to a categorical encoding. Matches the validated
#: ``theme.COLORWAY`` length. One slot is reserved for the "Other" bucket, so a
#: colour-encoded column may carry at most ``MAX_COLOR_CATEGORIES - 1`` named
#: levels plus the bucket.
MAX_COLOR_CATEGORIES = 8
OTHER_LABEL = "Other"

_SECTOR_RANK: Optional[list[str]] = None


def _sector_rank() -> list[str]:
    """Return sectors ordered by size in the FULL universe, most common first.

    Computed once from the unfiltered frame and cached. That is the whole point:
    ranking within the *filtered* frame would make a name's colour depend on the
    current filter, so narrowing the board to one region could repaint every
    surviving sector. Colour follows the entity, never its rank in the current
    view.
    """
    global _SECTOR_RANK
    if _SECTOR_RANK is None:
        from ..data import get_data
        try:
            counts = get_data()["sector"].value_counts()
            _SECTOR_RANK = [str(x) for x in counts.index]
        except Exception:  # pragma: no cover - defensive, keeps the card rendering
            _SECTOR_RANK = []
    return _SECTOR_RANK


def fold_categories(
        values: pd.Series,
        max_categories: int = MAX_COLOR_CATEGORIES,
        order: Optional[list[str]] = None,
) -> pd.Series:
    """Fold *values* down to at most *max_categories* levels, tail -> ``Other``.

    GICS has **eleven** sectors and the validated colourway has **eight** slots,
    so colouring by raw sector hands at least three sectors a hue that already
    means something else -- Plotly cycles the list without comment. This keeps
    the ``max_categories - 1`` largest levels and buckets the rest.

    Parameters
    ----------
    values
        The categorical column to fold.
    max_categories
        Total levels to emit, counting ``Other``.
    order
        Global level ranking, most important first. Defaults to
        :func:`_sector_rank`. Must NOT be derived from the caller's filtered
        frame -- see that function for why.

    Returns
    -------
    pandas.Series
        Same index, with tail levels replaced by :data:`OTHER_LABEL`.
    """
    ranking = order if order is not None else _sector_rank()
    if not ranking:
        return values
    keep = set(ranking[: max(max_categories - 1, 1)])
    return values.where(values.isin(keep), OTHER_LABEL)


def category_order(values: pd.Series, order: Optional[list[str]] = None) -> list[str]:
    """Return the folded level order for a Plotly ``category_orders`` entry.

    Keeps the global ranking's order so a level's hue is identical on every card
    and under every filter, with ``Other`` last.
    """
    ranking = order if order is not None else _sector_rank()
    present = set(values.dropna().astype(str))
    levels = [c for c in ranking if c in present]
    if OTHER_LABEL in present:
        levels.append(OTHER_LABEL)
    return levels
