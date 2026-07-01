import sys
import os
import traceback
from typing import TypedDict, Any, Tuple
from datetime import datetime, timedelta

from dash import callback, html, dcc, Output, Input
import dash_design_kit as ddk
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from data import get_data
from components.filter_component import filter_data, FILTER_CALLBACK_INPUTS
from logger import logger, schema, tbl


class TestInput(TypedDict):
    options: list[Any]
    default: Any


class ComponentResponse(TypedDict):
    layout: ddk.Card
    test_inputs: dict[str, TestInput]


component_id = "return_vs_risk_scatter_plot"

color_by_id = f"{component_id}_color_by"
color_by_options = [
    {"label": "None", "value": "none"},
    {"label": "Sector", "value": "sector"},
    {"label": "Industry", "value": "industry"},
    {"label": "Country", "value": "country"}
]
color_by_default = "sector"

size_by_id = f"{component_id}_size_by"
size_by_options = [
    {"label": "None", "value": "none"},
    {"label": "Signal Strength", "value": "signal_strength"},
    {"label": "Market Cap", "value": "market_cap"}
]
size_by_default = "signal_strength"

sector_filter_id = f"{component_id}_sector_filter"
sector_options = [
    {"label": "Industrials", "value": "Industrials"},
    {"label": "Information Technology", "value": "Information Technology"},
    {"label": "Consumer Discretionary", "value": "Consumer Discretionary"},
    {"label": "Health Care", "value": "Health Care"},
    {"label": "Materials", "value": "Materials"},
    {"label": "Communication Services", "value": "Communication Services"},
    {"label": "Consumer Staples", "value": "Consumer Staples"},
    {"label": "Energy", "value": "Energy"},
    {"label": "Utilities", "value": "Utilities"}
]
sector_default = [opt["value"] for opt in sector_options]

title = "Expected Return vs. Risk-Adjusted Return"
description = "Scatter plot showing the relationship between expected return (Kalman-filtered) and risk-adjusted return, with optional coloring by sector/industry/country and sizing by signal strength or market cap."


def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px",
                       "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Color By:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=color_by_id,
                                options=color_by_options,
                                value=color_by_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Size By:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=size_by_id,
                                options=size_by_options,
                                value=size_by_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Filter by Sector:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=sector_filter_id,
                                options=sector_options,
                                value=sector_default,
                                multi=True,
                                style={"minWidth": "250px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
            ),
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[
                    ddk.Graph(id=graph_id, style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                ]
            ),
            html.Pre(id=error_id, style={"color": "red", "margin": "10px 0"}),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        color_by_id: {
            "options": [option["value"] for option in color_by_options],
            "default": color_by_default
        },
        size_by_id: {
            "options": [option["value"] for option in size_by_options],
            "default": size_by_default
        },
        sector_filter_id: {
            "options": [option["value"] for option in sector_options],
            "default": sector_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }


def _update_logic(**kwargs) -> go.Figure:
    logger.debug("Updating chart with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{
                "text": "No data is available to display",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    logger.debug("Selecting columns for analysis...")
    df = df[['expected_return_kalman', 'risk_adj_return', 'sector', 'industry', 'country', 'signal_strength',
             'market_cap']].copy()
    logger.debug(schema(df))
    logger.debug(tbl(df))

    color_by = kwargs.get(color_by_id, color_by_default)
    if color_by is None:
        color_by = color_by_default

    size_by = kwargs.get(size_by_id, size_by_default)
    if size_by is None:
        size_by = size_by_default

    sector_filter = kwargs.get(sector_filter_id, sector_default)
    if sector_filter is None or len(sector_filter) == 0:
        sector_filter = sector_default

    logger.debug("Filtering by selected sectors: %s...", sector_filter)
    df = df[df['sector'].isin(sector_filter)]
    logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{
                "text": "No data matches the selected filters",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    logger.debug("Creating scatter plot...")

    if color_by == "none" and size_by == "none":
        fig = px.scatter(
            df,
            x='expected_return_kalman',
            y='risk_adj_return'
        )
    elif color_by == "none":
        fig = px.scatter(
            df,
            x='expected_return_kalman',
            y='risk_adj_return',
            size=size_by
        )
        fig.update_traces(marker=dict(sizemin=6))
    elif size_by == "none":
        fig = px.scatter(
            df,
            x='expected_return_kalman',
            y='risk_adj_return',
            color=color_by
        )
    else:
        fig = px.scatter(
            df,
            x='expected_return_kalman',
            y='risk_adj_return',
            color=color_by,
            size=size_by
        )
        fig.update_traces(marker=dict(sizemin=6))

    fig.update_xaxes(title_text="Expected Return (Kalman)")
    fig.update_yaxes(title_text="Risk-Adjusted Return")

    if color_by != "none":
        legend_title = color_by.replace('_', ' ').title()
        fig.update_layout(legend_title_text=legend_title)

    logger.debug("Done")
    return fig


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        color_by_id: Input(color_by_id, "value"),
        size_by_id: Input(size_by_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure = _update_logic(**kwargs)
        return figure, ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg