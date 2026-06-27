"""GEIB dashboard application assembly.

Creates the Dash app, builds the page layout (hero → global filters → KPI cards
→ analytic cards → footer), and registers the app-level callbacks (Reset, the
live results count). Importing the chart modules wires their callbacks into the
global Dash callback registry.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback, dcc, html

from .data import get_data
from .data import refresh as refresh_data
from .logger import logger
from .components import data_cards
from .components.filter_component import (
    ACTIVE_COORD_IDS,
    FILTER_CALLBACK_INPUTS,
    MKTCAP_ID,
    RESET_ID,
    RESULTS_ID,
    build_filter_panel,
    default_filter_values,
    filter_data,
)

# Reset restores every active global filter (coord dropdowns + market cap) to its
# "all selected" default; the order here is the order of the callback outputs.
_RESET_FILTER_IDS = [*ACTIVE_COORD_IDS, MKTCAP_ID]
from .charts import (
    efficient_frontier,
    high_conviction,
    kelly,
    monte_carlo,
    monte_carlo_forecast,
    pt_convergence,
    risk_adj_return,
    sharpe_ratio,
    var_cvar,
)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Global Equity Investment Board",
    update_title=None,
)
server = app.server

# Order of analytic cards down the page.
_CHART_MODULES = [
    pt_convergence,
    risk_adj_return,
    efficient_frontier,
    sharpe_ratio,
    var_cvar,
    kelly,
    monte_carlo,
    monte_carlo_forecast,
    high_conviction,
]


def _header() -> html.Div:
    return html.Div(
        className="geib-header",
        children=[
            html.H1("Global Equity Investment Board", className="geib-header-title"),
            html.P(
                "Kalman-filtered price targets, expected returns, and portfolio "
                "risk analytics across the global equity universe.",
                className="geib-header-subtitle",
            ),
        ],
    )


def serve_layout() -> html.Div:
    """Build the layout fresh on each page load (picks up the latest data)."""
    df = get_data()
    return html.Div(
        className="geib-page",
        children=[
            dcc.Store(id="refresh_trigger", data=0),
            _header(),
            build_filter_panel(df),
            data_cards.component(),
            *[module.component() for module in _CHART_MODULES],
            html.Div("PML Finance Project · Global Equity Investment Board", className="geib-footer"),
        ],
    )


app.layout = serve_layout


@callback(
    output=[
        *[Output(filter_id, "value") for filter_id in _RESET_FILTER_IDS],
        Output("refresh_trigger", "data"),
    ],
    inputs=[Input(RESET_ID, "n_clicks")],
    state=[State("refresh_trigger", "data")],
    prevent_initial_call=True,
)
def reset_filters(_n_clicks, current):
    """Reload data and restore every global filter to "all selected"."""
    refresh_data()
    defaults = default_filter_values(get_data())
    logger.info("Filters reset; data reloaded")
    return (
        *[defaults[filter_id] for filter_id in _RESET_FILTER_IDS],
        (current or 0) + 1,
    )


@callback(
    output=Output(RESULTS_ID, "children"),
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update_results_count(**kwargs):
    """Render the "filtered / total rows" indicator."""
    df = get_data()
    total = len(df)
    filtered = len(filter_data(df, **kwargs))
    return f"{filtered:,} / {total:,} rows"


if __name__ == "__main__":
    app.run(debug=True, port=8050)
