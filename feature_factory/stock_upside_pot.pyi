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

component_id = "top_stocks_by_upside"

# Control IDs
n_stocks_control_id = f"{component_id}_n_stocks"
metric_control_id = f"{component_id}_metric"
region_control_id = f"{component_id}_region"

# Control options
n_stocks_options = [
    {"label": "5", "value": 5},
    {"label": "10", "value": 10},
    {"label": "20", "value": 20},
    {"label": "50", "value": 50},
    {"label": "100", "value": 100}
]
n_stocks_default = 10

metric_options = [
    {"label": "Expected Upside", "value": "expected_upside_pct"},
    {"label": "Filtered Upside", "value": "filtered_upside"},
    {"label": "Expected Return (Prob-Weighted)", "value": "expected_return_prob_weighted"}
]
metric_default = "expected_upside_pct"

region_options = [
    {"label": "Asia / Pacific", "value": "Asia / Pacific"},
    {"label": "Europe", "value": "Europe"},
    {"label": "United States and Canada", "value": "United States and Canada"},
    {"label": "Africa / Middle East", "value": "Africa / Middle East"},
    {"label": "Latin America and Caribbean", "value": "Latin America and Caribbean"}
]
region_default = ["Asia / Pacific", "Europe", "United States and Canada", "Africa / Middle East", "Latin America and Caribbean"]

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    title = "Top Stocks by Upside Potential"
    description = "Ranked list of top equities by expected upside potential with customizable metrics and regional filtering"

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Number of Stocks:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=n_stocks_control_id,
                                options=n_stocks_options,
                                value=n_stocks_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Metric:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=metric_control_id,
                                options=metric_options,
                                value=metric_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Regions:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=region_control_id,
                                options=region_options,
                                value=region_default,
                                multi=True,
                                style={"minWidth": "200px"}
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
        n_stocks_control_id: {
            "options": [option["value"] for option in n_stocks_options],
            "default": n_stocks_default
        },
        metric_control_id: {
            "options": [option["value"] for option in metric_options],
            "default": metric_default
        },
        region_control_id: {
            "options": [option["value"] for option in region_options],
            "default": region_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _update_logic(**kwargs) -> go.Figure:
    """Core chart update logic without error handling."""
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
    df = df[['name', 'region', 'expected_upside_pct', 'filtered_upside', 'expected_return_prob_weighted']]
    logger.debug(schema(df))
    logger.debug(tbl(df))

    n_stocks = kwargs.get(n_stocks_control_id, n_stocks_default)
    if n_stocks is None:
        n_stocks = n_stocks_default
    n_stocks = int(n_stocks)

    metric = kwargs.get(metric_control_id, metric_default)
    if metric is None:
        metric = metric_default

    regions = kwargs.get(region_control_id, region_default)
    if regions is None or len(regions) == 0:
        regions = region_default
    if not isinstance(regions, list):
        regions = [regions]

    logger.debug("Filtering by regions: %s...", regions)
    df = df[df['region'].isin(regions)]
    logger.debug(tbl(df))

    logger.debug("Grouping by company and selecting maximum metric value...")
    df = df.groupby('name')[metric].max().reset_index()
    df.columns = ['name', 'value']
    logger.debug(tbl(df))

    logger.debug("Sorting by metric descending and selecting top %d stocks...", n_stocks)
    df = df.sort_values('value', ascending=True).tail(n_stocks)
    logger.debug(tbl(df))

    logger.debug("Creating horizontal bar chart with final data...")
    logger.debug(schema(df))
    logger.debug(tbl(df))

    metric_label_map = {
        "expected_upside_pct": "Expected Upside (%)",
        "filtered_upside": "Filtered Upside (%)",
        "expected_return_prob_weighted": "Expected Return (Prob-Weighted) (%)"
    }
    metric_label = metric_label_map.get(metric, metric)

    fig = px.bar(
        df,
        x='value',
        y='name',
        orientation='h',
        labels={'value': metric_label, 'name': 'Company'},
        hover_data={'value': ':.2f', 'name': True}
    )

    fig.update_layout(
        yaxis_title="Company",
        xaxis_title=metric_label,
        height=max(400, len(df) * 25),
        hovermode='closest'
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" + metric_label + ": %{x:.2f}%<extra></extra>"
    )

    logger.debug("Done")
    return fig

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        n_stocks_control_id: Input(n_stocks_control_id, "value"),
        metric_control_id: Input(metric_control_id, "value"),
        region_control_id: Input(region_control_id, "value"),
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