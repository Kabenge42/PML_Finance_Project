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

component_id = "signal_distribution_by_region"

region_control_1_id = f"{component_id}_region_1"
region_control_2_id = f"{component_id}_region_2"
y_axis_range_control_id = f"{component_id}_y_axis_range"

region_options = [
    {"label": "Asia / Pacific", "value": "Asia / Pacific"},
    {"label": "Europe", "value": "Europe"},
    {"label": "United States and Canada", "value": "United States and Canada"},
    {"label": "Africa / Middle East", "value": "Africa / Middle East"},
    {"label": "Latin America and Caribbean", "value": "Latin America and Caribbean"}
]
region_default_1 = "Asia / Pacific"
region_default_2 = "United States and Canada"

y_axis_range_options = [
    {"label": "Shared", "value": "shared"},
    {"label": "Individual", "value": "individual"}
]
y_axis_range_default = "shared"

signal_order = [
    "Strong Bearish (0/4)",
    "Bearish (1/4)",
    "Neutral (2/4)",
    "Bullish (3/4)",
    "Strong Bullish (4/4)"
]

def component() -> ComponentResponse:
    graph_1_id = f"{component_id}_graph_1"
    error_1_id = f"{component_id}_error_1"
    loading_1_id = f"{component_id}_loading_1"

    graph_2_id = f"{component_id}_graph_2"
    error_2_id = f"{component_id}_error_2"
    loading_2_id = f"{component_id}_loading_2"

    title = "Signal Distribution by Region"
    description = "Compare bullish signal distribution across regions"

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Chart 1 Region:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=region_control_1_id,
                                options=region_options,
                                value=region_default_1,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Chart 2 Region:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=region_control_2_id,
                                options=region_options,
                                value=region_default_2,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Y-Axis Range:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=y_axis_range_control_id,
                                options=y_axis_range_options,
                                value=y_axis_range_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "gap": "20px", "marginBottom": "15px"},
                children=[
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            dcc.Loading(
                                id=loading_1_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_1_id, style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                                ]
                            ),
                            html.Pre(id=error_1_id, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            dcc.Loading(
                                id=loading_2_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_2_id, style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                                ]
                            ),
                            html.Pre(id=error_2_id, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                ]
            ),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        region_control_1_id: {
            "options": [option["value"] for option in region_options],
            "default": region_default_1
        },
        region_control_2_id: {
            "options": [option["value"] for option in region_options],
            "default": region_default_2
        },
        y_axis_range_control_id: {
            "options": [option["value"] for option in y_axis_range_options],
            "default": y_axis_range_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _create_signal_chart(df: pd.DataFrame, region: str, title: str) -> go.Figure:
    """Create a signal distribution bar chart for a given region."""
    logger.debug("Creating signal distribution chart for region: %s...", region)

    df_filtered = df[df['region'] == region].copy()

    if len(df_filtered) == 0:
        logger.debug("No data available for region: %s", region)
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=title,
            annotations=[{
                "text": f"No data available for {region}",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        return empty_fig

    logger.debug("Grouping by signal...")
    signal_counts = df_filtered['signal'].value_counts().reset_index()
    signal_counts.columns = ['signal', 'count']
    logger.debug(tbl(signal_counts))

    signal_counts['signal'] = pd.Categorical(
        signal_counts['signal'],
        categories=signal_order,
        ordered=True
    )
    signal_counts = signal_counts.sort_values('signal')

    logger.debug("Creating bar chart...")
    fig = px.bar(
        signal_counts,
        x='signal',
        y='count',
        labels={'signal': 'Signal', 'count': 'Count'},
        color_discrete_sequence=['#636EFA']
    )

    fig.update_layout(
        title=title,
        xaxis_title="Signal",
        yaxis_title="Count of Stocks",
        hovermode='x unified',
        showlegend=False
    )

    fig.update_xaxes(tickangle=-45)

    logger.debug("Done creating chart for region: %s", region)
    return fig

def _update_logic_chart_1(**kwargs) -> go.Figure:
    """Core chart update logic for Chart 1 (left)."""
    logger.debug("Updating Chart 1 with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Chart 1",
            annotations=[{
                "text": "No data available",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    region_1 = kwargs.get(region_control_1_id, region_default_1)
    if region_1 is None:
        region_1 = region_default_1

    fig = _create_signal_chart(df, region_1, f"Chart 1: {region_1}")

    logger.debug("Done updating Chart 1")
    return fig

def _update_logic_chart_2(**kwargs) -> go.Figure:
    """Core chart update logic for Chart 2 (right)."""
    logger.debug("Updating Chart 2 with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Chart 2",
            annotations=[{
                "text": "No data available",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    region_2 = kwargs.get(region_control_2_id, region_default_2)
    if region_2 is None:
        region_2 = region_default_2

    fig = _create_signal_chart(df, region_2, f"Chart 2: {region_2}")

    logger.debug("Done updating Chart 2")
    return fig

@callback(
    output=[
        Output(f"{component_id}_graph_1", "figure"),
        Output(f"{component_id}_error_1", "children"),
        Output(f"{component_id}_graph_2", "figure"),
        Output(f"{component_id}_error_2", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        region_control_1_id: Input(region_control_1_id, "value"),
        region_control_2_id: Input(region_control_2_id, "value"),
        y_axis_range_control_id: Input(y_axis_range_control_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, go.Figure, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        fig_1 = _update_logic_chart_1(**kwargs)
        fig_2 = _update_logic_chart_2(**kwargs)

        y_axis_range_mode = kwargs.get(y_axis_range_control_id, y_axis_range_default)
        if y_axis_range_mode is None:
            y_axis_range_mode = y_axis_range_default

        if y_axis_range_mode == "shared":
            logger.debug("Applying shared Y-axis range...")
            y_max_1 = fig_1.data[0].y.max() if len(fig_1.data) > 0 and len(fig_1.data[0].y) > 0 else 0
            y_max_2 = fig_2.data[0].y.max() if len(fig_2.data) > 0 and len(fig_2.data[0].y) > 0 else 0
            shared_y_max = max(y_max_1, y_max_2) * 1.1

            fig_1.update_yaxes(range=[0, shared_y_max])
            fig_2.update_yaxes(range=[0, shared_y_max])
            logger.debug("Shared Y-axis range applied: 0 to %.1f", shared_y_max)

        return fig_1, "", fig_2, ""

    except Exception as e:
        error_msg = f"Error updating charts: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg, empty_fig, error_msg