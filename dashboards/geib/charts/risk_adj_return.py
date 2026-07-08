"""Expected Return vs. Risk-Adjusted Return scatter.

Ported from ``feature_factory/risk_adj_return.pyi``.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import empty_figure, scoped_filter, sector_values
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema, tbl
from ..theme import GRAPH_STYLE, control
from ..theme import card as theme_card

component_id = "return_vs_risk_scatter_plot"

color_by_id = f"{component_id}_color_by"
color_by_options = [
    {"label": "None", "value": "none"},
    {"label": "Sector", "value": "sector"},
    {"label": "Industry", "value": "industry"},
    {"label": "Country", "value": "country"},
]
color_by_default = "sector"

size_by_id = f"{component_id}_size_by"
size_by_options = [
    {"label": "None", "value": "none"},
    {"label": "Signal Strength", "value": "signal_strength"},
    {"label": "Market Cap", "value": "market_cap"},
]
size_by_default = "signal_strength"

sector_filter_id = f"{component_id}_sector_filter"

title = "Expected Return vs. Risk-Adjusted Return"
description = (
    "Scatter plot showing the relationship between expected return "
    "(Kalman-filtered) and risk-adjusted return, with optional coloring by "
    "sector/industry/country and sizing by signal strength or market cap."
)


def component() -> "object":
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    sector_opts = [{"label": s, "value": s} for s in sector_values(get_data())]

    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                className="geib-controls-row",
                children=[
                    control(
                        "Color By:",
                        dcc.Dropdown(
                            id=color_by_id,
                            options=color_by_options,
                            value=color_by_default,
                            searchable=False,
                            style={"minWidth": "200px"},
                        ),
                    ),
                    control(
                        "Size By:",
                        dcc.Dropdown(
                            id=size_by_id,
                            options=size_by_options,
                            value=size_by_default,
                            searchable=False,
                            style={"minWidth": "200px"},
                        ),
                    ),
                    control(
                        "Filter by Sector:",
                        dcc.Dropdown(
                            id=sector_filter_id,
                            options=sector_opts,
                            value=[],
                            multi=True,
                            style={"minWidth": "250px"},
                        ),
                    ),
                ],
            ),
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[dcc.Graph(id=graph_id, style=GRAPH_STYLE)],
            ),
            html.Pre(id=error_id, className="geib-error"),
        ],
    )


def _update_logic(**kwargs) -> go.Figure:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return empty_figure("No data is available to display")

    df = df[
        [
            "expected_return_kalman",
            "risk_adj_return",
            "sector",
            "industry",
            "country",
            "signal_strength",
            "market_cap",
        ]
    ].copy()
    logger.debug(schema(df))

    color_by = kwargs.get(color_by_id) or color_by_default
    size_by = kwargs.get(size_by_id) or size_by_default
    sector_filter = kwargs.get(sector_filter_id)

    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]
    if len(df) == 0:
        return empty_figure("No data matches the selected filters")
    logger.debug(tbl(df))

    scatter_kwargs = dict(x="expected_return_kalman", y="risk_adj_return")
    if color_by != "none":
        scatter_kwargs["color"] = color_by
    if size_by != "none":
        scatter_kwargs["size"] = size_by

    fig = px.scatter(df, **scatter_kwargs)
    if size_by != "none":
        fig.update_traces(marker=dict(sizemin=6))

    fig.update_xaxes(title_text="Expected Return (Kalman)")
    fig.update_yaxes(title_text="Risk-Adjusted Return")
    if color_by != "none":
        fig.update_layout(legend_title_text=color_by.replace("_", " ").title())
    return fig


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(sector_filter_id, "options"),
        Output(sector_filter_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        color_by_id: Input(color_by_id, "value"),
        size_by_id: Input(size_by_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, str, list, list]:
    # Scope the local Sector filter to the globally-filtered universe.
    df_all = filter_data(get_data(), **kwargs)
    sector_opts, sector_val = scoped_filter(
        df_all, "sector", kwargs.get(sector_filter_id), multi=True
    )
    kwargs[sector_filter_id] = sector_val
    try:
        return _update_logic(**kwargs), "", sector_opts, sector_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return empty_figure("An error occurred"), msg, sector_opts, sector_val