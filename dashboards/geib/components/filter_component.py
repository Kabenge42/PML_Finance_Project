"""Global filter panel shared by every GEIB chart.

The categorical filters are driven by a single registry, :data:`COORD_FILTERS`,
listing the ``pymc_role = 'coord'`` columns the board can filter on. The panel
renders a multi-select dropdown for each registry column **present in the loaded
data**, so coords absent from the analytics source (``region``, ``style_class``,
…) simply don't appear — and will appear automatically if the source later
carries them. Identifiers (``isin`` / ``ticker``) and raw date coords are
intentionally excluded (too high-cardinality / not categorical).

Exposes:

* ``build_filter_panel(df)`` — the top control panel (one dropdown per available
  coord + a Market Cap Range range filter) plus a live results count and Reset.
* ``FILTER_CALLBACK_INPUTS`` — the dict of ``Input`` objects each chart spreads
  into its ``@callback`` so the global filters drive every component.
* ``filter_data(df, **kwargs)`` — applies the active global filters generically.
* ``default_filter_values(df)`` — the "all selected" defaults (used by Reset).
* ``ACTIVE_COORD_IDS`` — the coord filter ids present in the data (for Reset).

The coord dropdowns are **cascading/dependent**: the ``sync_filters`` callback
(the single owner of every coord ``value``/``options``, the Market-Cap value, and
``refresh_trigger``) repopulates each downstream coord whenever an upstream one
changes, and restores the "all selected" defaults on Reset.
"""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from ..data import get_data
from ..data import refresh as refresh_data
from ..logger import logger
from ..theme import CONTROLS_ROW_STYLE, control

# --- Coord filter registry -------------------------------------------------
# (column, label) in panel order. Keep this the single source of truth for the
# categorical global filters; everything else (inputs, defaults, reset, the
# applied filtering) is derived from it.
COORD_FILTERS: tuple[tuple[str, str], ...] = (
    ("trading_region", "Trading Region"),
    ("region", "Region"),
    ("country", "Country"),
    ("exchange", "Exchange"),
    ("unit", "Unit"),
    ("sector", "Sector"),
    ("industry", "Industry"),
    ("style_class", "Style Class"),
    ("size_class", "Size Class"),
    ("next_earnings_when", "Next Earnings When"),
    ("next_earnings_status", "Next Earnings Status"),
)

# High-cardinality coords get a searchable dropdown (better than scrolling).
_SEARCHABLE_COORDS = frozenset({"country", "exchange", "unit", "sector", "industry"})

MKTCAP_ID = "global_filter_mktcap"
RESULTS_ID = "global_filter_results"
RESET_ID = "global_filter_reset"


def coord_filter_id(column: str) -> str:
    """DOM id of the global dropdown filtering coord *column*."""
    return f"global_filter_{column}"


# Market-cap range labels -> [lo, hi) bounds in millions (spec mapping).
MKTCAP_OPTIONS = [
    {"label": "0-1000M", "value": "0-1000M"},
    {"label": "1000M-10B", "value": "1000M-10B"},
    {"label": "10B-100B", "value": "10B-100B"},
    {"label": "100B+", "value": "100B+"},
]
MKTCAP_BOUNDS = {
    "0-1000M": (0.0, 1000.0),
    "1000M-10B": (1000.0, 10000.0),
    "10B-100B": (10000.0, 100000.0),
    "100B+": (100000.0, float("inf")),
}
MKTCAP_VALUES = [opt["value"] for opt in MKTCAP_OPTIONS]


def _present_coords(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Registry entries whose column exists in *df* (registry order preserved)."""
    cols = set(df.columns) if df is not None else set()
    return [(column, label) for column, label in COORD_FILTERS if column in cols]


# Resolve the active coords once from the loaded data so the static callback
# wiring (``FILTER_CALLBACK_INPUTS`` and the Reset outputs in ``app.py``) matches
# the panel the layout renders. ``get_data`` is cached; the empty-frame fallback
# carries the declared coord columns so this stays stable without a DB.
_ACTIVE_COORDS = _present_coords(get_data())
ACTIVE_COORD_COLUMNS: list[str] = [column for column, _ in _ACTIVE_COORDS]
ACTIVE_COORD_IDS: list[str] = [coord_filter_id(column) for column in ACTIVE_COORD_COLUMNS]

# Inputs every chart callback spreads in. ``refresh_trigger`` is added by each
# chart separately (mirrors the reference stubs).
FILTER_CALLBACK_INPUTS = {
    **{coord_filter_id(column): Input(coord_filter_id(column), "value")
       for column in ACTIVE_COORD_COLUMNS},
    MKTCAP_ID: Input(MKTCAP_ID, "value"),
}


def _options(df: pd.DataFrame, column: str) -> list[dict]:
    if df is None or column not in df.columns:
        return []
    values = sorted(
        v for v in df[column].dropna().unique().tolist() if str(v).strip()
    )
    return [{"label": str(v), "value": v} for v in values]


def _all_values(df: pd.DataFrame, column: str) -> list:
    return [opt["value"] for opt in _options(df, column)]


def default_filter_values(df: pd.DataFrame) -> dict[str, list]:
    """Return the "everything selected" default for each active global filter."""
    defaults: dict[str, list] = {
        coord_filter_id(column): _all_values(df, column)
        for column in ACTIVE_COORD_COLUMNS
    }
    defaults[MKTCAP_ID] = list(MKTCAP_VALUES)
    return defaults


def _coord_control(df: pd.DataFrame, column: str, label: str) -> html.Div:
    opts = _options(df, column)
    return control(
        f"{label}:",
        dcc.Dropdown(
            id=coord_filter_id(column),
            options=opts,
            value=[opt["value"] for opt in opts],
            multi=True,
            searchable=column in _SEARCHABLE_COORDS,
            style={"minWidth": "200px"},
        ),
    )


def build_filter_panel(df: pd.DataFrame) -> html.Div:
    """Build the top global-filter control panel from the coord registry.

    Renders exactly the coords in :data:`_ACTIVE_COORDS` — the single locked set
    that also backs :data:`FILTER_CALLBACK_INPUTS`, the Reset outputs in
    ``app.py``, and :func:`filter_data`. Deriving the rendered dropdowns from
    this set (rather than re-scanning *df* at request time) keeps the panel,
    the wired callback inputs, and the applied filtering in lockstep, so every
    global filter shown in the panel is guaranteed to drive every chart and KPI
    callback — i.e. the coord filters apply consistently across the whole app.
    """
    coord_controls = [
        _coord_control(df, column, label) for column, label in _ACTIVE_COORDS
    ]
    mktcap_control = control(
        "Market Cap Range:",
        dcc.Dropdown(
            id=MKTCAP_ID,
            options=MKTCAP_OPTIONS,
            value=list(MKTCAP_VALUES),
            multi=True,
            searchable=False,
            style={"minWidth": "220px"},
        ),
    )

    return html.Div(
        className="geib-filter-panel",
        style={"padding": "16px"},
        children=html.Div(
            style=CONTROLS_ROW_STYLE,
            children=[
                *coord_controls,
                mktcap_control,
                html.Button(
                    "Reset Filters",
                    id=RESET_ID,
                    n_clicks=0,
                    className="geib-btn",
                    style={"height": "38px", "padding": "0 16px", "marginRight": "15px"},
                ),
                html.Div(id=RESULTS_ID, className="geib-results", children="0 / 0 rows"),
            ],
        ),
    )


def filter_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Apply the active global filters to *df*.

    Iterates the coord registry generically: each active coord dropdown narrows
    the frame by ``isin`` of the selected values; Market Cap Range maps its
    labels to numeric bounds. Unknown keyword arguments (a chart's own controls,
    ``refresh_trigger``) are ignored. An empty/``None`` selection for a given
    filter is treated as "no constraint" so the board never silently blanks out.
    """
    if df is None or len(df) == 0:
        return df

    out = df

    for column in ACTIVE_COORD_COLUMNS:
        selected = kwargs.get(coord_filter_id(column))
        if selected and column in out.columns:
            out = out[out[column].isin(selected)]

    ranges = kwargs.get(MKTCAP_ID)
    if ranges and "market_cap" in out.columns:
        out = _apply_mktcap(out, ranges)

    return out


def _apply_mktcap(df: pd.DataFrame, ranges: list) -> pd.DataFrame:
    """Narrow *df* to rows whose ``market_cap`` falls in any selected range."""
    if not ranges or "market_cap" not in df.columns:
        return df
    mask = pd.Series(False, index=df.index)
    for label in ranges:
        bounds = MKTCAP_BOUNDS.get(label)
        if bounds is None:
            continue
        lo, hi = bounds
        mask = mask | ((df["market_cap"] >= lo) & (df["market_cap"] < hi))
    return df[mask]


# --- Cascading (dependent) coord filters -----------------------------------
# The coord dropdowns cascade top-down in registry order: changing an upstream
# filter (e.g. ``region``) repopulates every downstream filter (``country``,
# ``sector``, ``industry`` …) to the values still present under the upstream
# selection, and re-selects them all. The Market-Cap range sits above all coords,
# so changing it repopulates the entire coord stack. This keeps the choices a
# user sees consistent with what they have already narrowed to, and guarantees
# the filters apply consistently across every chart and KPI (they all read the
# same coord/mktcap values via ``FILTER_CALLBACK_INPUTS`` + :func:`filter_data`).


def _triggered_index(triggered_ids: set[str]) -> int:
    """Lowest coord position that changed (``-1`` for Market-Cap, the upstream).

    Returns ``len(ACTIVE_COORD_COLUMNS)`` (a no-op sentinel) when nothing
    cascade-relevant fired, so downstream recomputation is skipped.
    """
    if MKTCAP_ID in triggered_ids:
        return -1
    positions = [
        idx
        for idx, column in enumerate(ACTIVE_COORD_COLUMNS)
        if coord_filter_id(column) in triggered_ids
    ]
    return min(positions) if positions else len(ACTIVE_COORD_COLUMNS)


def _cascade(coord_values: dict[str, list], mktcap: list, trigger_idx: int) -> tuple[list, list]:
    """Recompute coord ``options``/``value`` downstream of *trigger_idx*.

    Walks the coord registry in order, narrowing a running frame by each
    upstream coord's current selection. Coords at or above *trigger_idx* keep
    their options/value untouched (``dash.no_update``); coords below are
    repopulated to the running frame's remaining values and re-selected in full.

    Parameters
    ----------
    coord_values
        Current ``{coord_filter_id: selected_values}`` for every active coord.
    mktcap
        Selected Market-Cap range labels (scopes the whole coord stack).
    trigger_idx
        Position of the changed coord (``-1`` = Market-Cap, upstream of all).

    Returns
    -------
    tuple[list, list]
        ``(options_outputs, value_outputs)`` aligned to ``ACTIVE_COORD_COLUMNS``.
    """
    running = _apply_mktcap(get_data(), mktcap)
    options_out: list = []
    values_out: list = []

    for idx, column in enumerate(ACTIVE_COORD_COLUMNS):
        if idx <= trigger_idx:
            # Upstream / the trigger itself: leave the control as the user set it,
            # but still narrow the running frame by its current selection.
            selected = coord_values.get(coord_filter_id(column))
            if selected and column in running.columns:
                running = running[running[column].isin(selected)]
            options_out.append(no_update)
            values_out.append(no_update)
            continue

        # Downstream: repopulate to what remains, and re-select all of it.
        opts = _options(running, column)
        values = [opt["value"] for opt in opts]
        options_out.append(opts)
        values_out.append(values)
        if values and column in running.columns:
            running = running[running[column].isin(values)]

    return options_out, values_out


_COORD_VALUE_OUTPUTS = [Output(coord_filter_id(c), "value") for c in ACTIVE_COORD_COLUMNS]
_COORD_OPTIONS_OUTPUTS = [Output(coord_filter_id(c), "options") for c in ACTIVE_COORD_COLUMNS]


@callback(
    output=[
        *_COORD_OPTIONS_OUTPUTS,
        *_COORD_VALUE_OUTPUTS,
        Output(MKTCAP_ID, "value"),
        Output("refresh_trigger", "data"),
    ],
    inputs=[
        *[Input(coord_filter_id(c), "value") for c in ACTIVE_COORD_COLUMNS],
        Input(MKTCAP_ID, "value"),
        Input(RESET_ID, "n_clicks"),
    ],
    state=State("refresh_trigger", "data"),
    prevent_initial_call=True,
)
def sync_filters(*args):
    """Single owner of the global filter state: Reset + cascading dependencies.

    On Reset, reloads the data and restores every coord + Market-Cap filter to
    "all selected". On any coord/Market-Cap change, cascades the selection down
    the coord registry so downstream dropdowns only offer (and select) values
    consistent with the upstream choice. Centralising both behaviours in one
    callback keeps the coord ``value``/``options`` single-owned (no duplicate
    outputs) and the panel in lockstep with what every chart filters on.
    """
    n = len(ACTIVE_COORD_COLUMNS)
    coord_values = {
        coord_filter_id(column): args[idx] for idx, column in enumerate(ACTIVE_COORD_COLUMNS)
    }
    mktcap = args[n]
    refresh_state = args[n + 2]

    triggered_ids = {entry["prop_id"].split(".")[0] for entry in ctx.triggered}

    if RESET_ID in triggered_ids:
        refresh_data()
        df = get_data()
        defaults = default_filter_values(df)
        logger.info("Filters reset; data reloaded")
        options = [_options(df, column) for column in ACTIVE_COORD_COLUMNS]
        values = [defaults[coord_filter_id(column)] for column in ACTIVE_COORD_COLUMNS]
        return (*options, *values, list(MKTCAP_VALUES), (refresh_state or 0) + 1)

    trigger_idx = _triggered_index(triggered_ids)
    options, values = _cascade(coord_values, mktcap, trigger_idx)
    return (*options, *values, no_update, no_update)