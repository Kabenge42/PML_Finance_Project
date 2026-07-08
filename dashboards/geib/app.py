"""GEIB dashboard application assembly.

Creates the Dash app, builds the page layout (hero → global filters → KPI cards
→ analytic cards → footer), and registers the app-level callbacks (Reset, the
live results count). Importing the chart modules wires their callbacks into the
global Dash callback registry.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, callback, dcc, html

from .charts import (
    beta_capm,
    efficient_frontier,
    high_conviction,
    kalman_structural_forecast,
    kelly,
    monte_carlo,
    monte_carlo_forecast,
    pt_convergence,
    risk_adj_return,
    sharpe_ratio,
    var_cvar,
)
from .components import data_cards
from .components.filter_component import (
    FILTER_CALLBACK_INPUTS,
    RESULTS_ID,
    build_filter_panel,
    filter_data,
)
from .data import get_data

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Global Equity Investment Dashboard",
    update_title=None,
)
server = app.server

# Order of analytic cards down the page.
_CHART_MODULES = [
    kalman_structural_forecast,
    pt_convergence,
    risk_adj_return,
    beta_capm,
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


# Reset + cascading dependent filters are owned by ``sync_filters`` in
# ``components.filter_component`` (registered on import above). This module keeps
# only the live results-count callback below.


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