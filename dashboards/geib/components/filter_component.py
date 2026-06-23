"""Global filter panel shared by every GEIB chart.

Exposes:

* ``build_filter_panel(df)`` — the top control panel (Sector / Country /
  Exchange / Market Cap Range) plus a live results count and a Reset button.
* ``FILTER_CALLBACK_INPUTS`` — the dict of ``Input`` objects each chart spreads
  into its ``@callback`` so the global filters drive every component.
* ``filter_data(df, **kwargs)`` — applies the active global filters.
* ``default_filter_values(df)`` — the "all selected" defaults (used by Reset).
"""

from __future__ import annotations

import pandas as pd
from dash import Input, dcc, html

from ..theme import CONTROLS_ROW_STYLE, control

SECTOR_ID = "global_filter_sector"
COUNTRY_ID = "global_filter_country"
EXCHANGE_ID = "global_filter_exchange"
MKTCAP_ID = "global_filter_mktcap"
RESULTS_ID = "global_filter_results"
RESET_ID = "global_filter_reset"

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

# Inputs every chart callback spreads in. ``refresh_trigger`` is added by each
# chart separately (mirrors the reference stubs).
FILTER_CALLBACK_INPUTS = {
    SECTOR_ID: Input(SECTOR_ID, "value"),
    COUNTRY_ID: Input(COUNTRY_ID, "value"),
    EXCHANGE_ID: Input(EXCHANGE_ID, "value"),
    MKTCAP_ID: Input(MKTCAP_ID, "value"),
}


def _options(df: pd.DataFrame, column: str) -> list[dict]:
    if column not in df.columns:
        return []
    values = sorted(
        v for v in df[column].dropna().unique().tolist() if str(v).strip()
    )
    return [{"label": str(v), "value": v} for v in values]


def _all_values(df: pd.DataFrame, column: str) -> list:
    return [opt["value"] for opt in _options(df, column)]


def default_filter_values(df: pd.DataFrame) -> dict[str, list]:
    """Return the "everything selected" default for each global filter."""
    return {
        SECTOR_ID: _all_values(df, "sector"),
        COUNTRY_ID: _all_values(df, "country"),
        EXCHANGE_ID: _all_values(df, "exchange"),
        MKTCAP_ID: list(MKTCAP_VALUES),
    }


def build_filter_panel(df: pd.DataFrame) -> html.Div:
    """Build the top global-filter control panel."""
    sector_opts = _options(df, "sector")
    country_opts = _options(df, "country")
    exchange_opts = _options(df, "exchange")

    return html.Div(
        className="geib-filter-panel",
        style={"padding": "16px"},
        children=html.Div(
            style=CONTROLS_ROW_STYLE,
            children=[
                control(
                    "Sector:",
                    dcc.Dropdown(
                        id=SECTOR_ID,
                        options=sector_opts,
                        value=[opt["value"] for opt in sector_opts],
                        multi=True,
                        searchable=False,
                        style={"minWidth": "220px"},
                    ),
                ),
                control(
                    "Country:",
                    dcc.Dropdown(
                        id=COUNTRY_ID,
                        options=country_opts,
                        value=[opt["value"] for opt in country_opts],
                        multi=True,
                        searchable=True,
                        style={"minWidth": "200px"},
                    ),
                ),
                control(
                    "Exchange:",
                    dcc.Dropdown(
                        id=EXCHANGE_ID,
                        options=exchange_opts,
                        value=[opt["value"] for opt in exchange_opts],
                        multi=True,
                        searchable=True,
                        style={"minWidth": "200px"},
                    ),
                ),
                control(
                    "Market Cap Range:",
                    dcc.Dropdown(
                        id=MKTCAP_ID,
                        options=MKTCAP_OPTIONS,
                        value=list(MKTCAP_VALUES),
                        multi=True,
                        searchable=False,
                        style={"minWidth": "220px"},
                    ),
                ),
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

    Unknown keyword arguments (a chart's own controls, ``refresh_trigger``) are
    ignored. An empty/``None`` selection for a given filter is treated as "no
    constraint" so the board never silently blanks out.
    """
    if df is None or len(df) == 0:
        return df

    out = df

    sectors = kwargs.get(SECTOR_ID)
    if sectors:
        out = out[out["sector"].isin(sectors)]

    countries = kwargs.get(COUNTRY_ID)
    if countries:
        out = out[out["country"].isin(countries)]

    exchanges = kwargs.get(EXCHANGE_ID)
    if exchanges:
        out = out[out["exchange"].isin(exchanges)]

    ranges = kwargs.get(MKTCAP_ID)
    if ranges and "market_cap" in out.columns:
        mask = pd.Series(False, index=out.index)
        for label in ranges:
            bounds = MKTCAP_BOUNDS.get(label)
            if bounds is None:
                continue
            lo, hi = bounds
            mask = mask | ((out["market_cap"] >= lo) & (out["market_cap"] < hi))
        out = out[mask]

    return out
