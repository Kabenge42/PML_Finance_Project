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

component_id = "upside_comparison_by_confidence"

confidence_options = [
    {"label": "High", "value": "High"},
    {"label": "Medium", "value": "Medium"},
    {"label": "Low", "value": "Low"}
]
confidence_default_left = "High"
confidence_default_right = "Low"

y_axis_range_options = [
    {"label": "Shared", "value": "shared"},
    {"label": "Individual", "value": "individual"}
]
y_axis_range_default = "shared"

def component() -> ComponentResponse:
    graph_id_left = f"{component_id}_graph_left"
    graph_id_right = f"{component_id}_graph_right"
    error_id_left = f"{component_id}_error_left"
    error_id_right = f"{component_id}_error_right"
    loading_id_left = f"{component_id}_loading_left"
    loading_id_right = f"{component_id}_loading_right"

    confidence_id_left = f"{component_id}_confidence_left"
    confidence_id_right = f"{component_id}_confidence_right"
    y_axis_range_id = f"{component_id}_y_axis_range"

    title = "Expected Upside Comparison by Confidence Level"
    description = "Compare average expected upside across sectors for different confidence levels. Use shared or individual Y-axis ranges for comparison."

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Chart 1 Confidence:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=confidence_id_left,
                                options=confidence_options,
                                value=confidence_default_left,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Chart 2 Confidence:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=confidence_id_right,
                                options=confidence_options,
                                value=confidence_default_right,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Y-Axis Range:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=y_axis_range_id,
                                options=y_axis_range_options,
                                value=y_axis_range_default,
                                style={"minWidth": "200px"},
                                searchable=False
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
                                id=loading_id_left,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_id_left, style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                                ]
                            ),
                            html.Pre(id=error_id_left, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            dcc.Loading(
                                id=loading_id_right,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_id_right, style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                                ]
                            ),
                            html.Pre(id=error_id_right, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                ]
            ),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        confidence_id_left: {
            "options": [option["value"] for option in confidence_options],
            "default": confidence_default_left
        },
        confidence_id_right: {
            "options": [option["value"] for option in confidence_options],
            "default": confidence_default_right
        },
        y_axis_range_id: {
            "options": [option["value"] for option in y_axis_range_options],
            "default": y_axis_range_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _create_upside_chart(df: pd.DataFrame, confidence_level: str, chart_title: str) -> go.Figure:
    """Create a bar chart for expected upside by sector."""
    logger.debug("Filtering by confidence level: %s...", confidence_level)
    df_filtered = df[df['confidence_level'] == confidence_level].copy()
    logger.debug(tbl(df_filtered))

    if len(df_filtered) == 0:
        logger.debug("No data available for confidence level: %s", confidence_level)
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=chart_title,
            annotations=[{
                "text": f"No data available for {confidence_level} confidence level",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        return empty_fig

    logger.debug("Grouping by sector and computing average upside...")
    df_grouped = df_filtered.groupby('sector')['expected_upside_pct'].mean().reset_index()
    df_grouped.columns = ['sector', 'avg_upside']
    logger.debug(tbl(df_grouped))

    logger.debug("Sorting by average upside descending...")
    df_grouped = df_grouped.sort_values('avg_upside', ascending=False)
    logger.debug(tbl(df_grouped))

    logger.debug("Creating bar chart...")
    fig = px.bar(
        df_grouped,
        x='sector',
        y='avg_upside',
        labels={'sector': 'Sector', 'avg_upside': 'Average Expected Upside (%)'},
        title=chart_title
    )

    fig.update_layout(
        xaxis_title="Sector",
        yaxis_title="Average Expected Upside (%)",
        hovermode="x unified"
    )

    logger.debug("Done creating chart")
    return fig

@callback(
    output=[
        Output(f"{component_id}_graph_left", "figure"),
        Output(f"{component_id}_error_left", "children"),
        Output(f"{component_id}_graph_right", "figure"),
        Output(f"{component_id}_error_right", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        f'{component_id}_confidence_left': Input(f"{component_id}_confidence_left", "value"),
        f'{component_id}_confidence_right': Input(f"{component_id}_confidence_right", "value"),
        f'{component_id}_y_axis_range': Input(f"{component_id}_y_axis_range", "value"),
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
        logger.debug("Updating upside comparison charts with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

        df = filter_data(get_data(), **kwargs)

        if len(df) == 0:
            logger.debug("No data available after filtering")
            empty_fig_no_data = go.Figure()
            empty_fig_no_data.update_layout(
                title="No data available",
                annotations=[{"text": "No data is available to display", "showarrow": False, "font": {"size": 20}}]
            )
            return empty_fig_no_data, "", empty_fig_no_data, ""

        confidence_left = kwargs.get(f'{component_id}_confidence_left', confidence_default_left)
        if confidence_left is None:
            confidence_left = confidence_default_left

        confidence_right = kwargs.get(f'{component_id}_confidence_right', confidence_default_right)
        if confidence_right is None:
            confidence_right = confidence_default_right

        y_axis_range = kwargs.get(f'{component_id}_y_axis_range', y_axis_range_default)
        if y_axis_range is None:
            y_axis_range = y_axis_range_default

        logger.debug("Creating left chart for confidence level: %s", confidence_left)
        fig_left = _create_upside_chart(df, confidence_left, f"High Confidence")

        logger.debug("Creating right chart for confidence level: %s", confidence_right)
        fig_right = _create_upside_chart(df, confidence_right, f"Low Confidence")

        if y_axis_range == "shared":
            logger.debug("Applying shared Y-axis range...")
            all_values = []
            if len(fig_left.data) > 0 and fig_left.data[0].y is not None:
                all_values.extend(fig_left.data[0].y)
            if len(fig_right.data) > 0 and fig_right.data[0].y is not None:
                all_values.extend(fig_right.data[0].y)

            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                y_range = y_max - y_min
                y_min_padded = y_min - (y_range * 0.1)
                y_max_padded = y_max + (y_range * 0.1)

                fig_left.update_yaxes(range=[y_min_padded, y_max_padded])
                fig_right.update_yaxes(range=[y_min_padded, y_max_padded])
                logger.debug("Shared Y-axis range applied: [%.2f, %.2f]", y_min_padded, y_max_padded)

        logger.debug("Done updating charts")
        return fig_left, "", fig_right, ""

    except Exception as e:
        error_msg = f"Error updating charts: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg, empty_fig, error_msg