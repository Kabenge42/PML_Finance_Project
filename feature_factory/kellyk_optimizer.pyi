import sys
import os
import traceback
from typing import TypedDict, Any, Tuple
from datetime import datetime, timedelta

from dash import callback, html, dcc, Output, Input
#import dash_design_kit as ddk
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

component_id = "kelly_criterion_position_sizer"

kelly_fraction_id = f"{component_id}_kelly_fraction"
kelly_fraction_options = [
    {"label": "Full Kelly (1.0)", "value": 1.0},
    {"label": "Half Kelly (0.5)", "value": 0.5},
    {"label": "Quarter Kelly (0.25)", "value": 0.25},
    {"label": "Eighth Kelly (0.125)", "value": 0.125}
]
kelly_fraction_default = 0.25

max_position_id = f"{component_id}_max_position"
max_position_options = [
    {"label": "5%", "value": 0.05},
    {"label": "10%", "value": 0.10},
    {"label": "15%", "value": 0.15},
    {"label": "20%", "value": 0.20},
    {"label": "No cap", "value": "no_cap"}
]
max_position_default = 0.10

min_confidence_id = f"{component_id}_min_confidence"
min_confidence_options = [
    {"label": "0.15", "value": 0.15},
    {"label": "0.25", "value": 0.25},
    {"label": "0.35", "value": 0.35},
    {"label": "0.45", "value": 0.45}
]
min_confidence_default = 0.35

adjustment_method_id = f"{component_id}_adjustment_method"
adjustment_method_options = [
    {"label": "None", "value": "none"},
    {"label": "Confidence-weighted", "value": "confidence"},
    {"label": "Achievement-weighted", "value": "achievement"},
    {"label": "Both", "value": "both"}
]
adjustment_method_default = "both"

sector_filter_id = f"{component_id}_sector_filter"
sector_options = [
    {"label": "All sectors", "value": "all"},
    {"label": "Industrials", "value": "Industrials"},
    {"label": "Information Technology", "value": "Information Technology"},
    {"label": "Materials", "value": "Materials"},
    {"label": "Energy", "value": "Energy"},
    {"label": "Health Care", "value": "Health Care"},
    {"label": "Consumer Discretionary", "value": "Consumer Discretionary"},
    {"label": "Financials", "value": "Financials"},
    {"label": "Communication Services", "value": "Communication Services"},
    {"label": "Utilities", "value": "Utilities"}
]
sector_filter_default = ["all"]

color_by_id = f"{component_id}_color_by"
color_by_options = [
    {"label": "None", "value": "none"},
    {"label": "Sector", "value": "sector"},
    {"label": "Confidence Level", "value": "confidence_level"}
]
color_by_default = "none"

scatter_color_id = f"{component_id}_scatter_color"
scatter_color_options = [
    {"label": "None", "value": "none"},
    {"label": "Confidence Level", "value": "confidence_level"}
]
scatter_color_default = "none"

scatter_size_id = f"{component_id}_scatter_size"
scatter_size_options = [
    {"label": "None", "value": "none"},
    {"label": "Achievement Probability", "value": "achievement_probability"}
]
scatter_size_default = "none"

def component() -> ComponentResponse:
    graph_1_id = f"{component_id}_graph_1"
    error_1_id = f"{component_id}_error_1"
    loading_1_id = f"{component_id}_loading_1"

    graph_2_id = f"{component_id}_graph_2"
    error_2_id = f"{component_id}_error_2"
    loading_2_id = f"{component_id}_loading_2"

    title = "Kelly Criterion Position Sizer"
    description = "Calculate optimal position sizing based on expected returns and win probabilities using the Kelly Criterion formula"

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Kelly Fraction:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=kelly_fraction_id,
                                options=kelly_fraction_options,
                                value=kelly_fraction_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Max Position Size:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=max_position_id,
                                options=max_position_options,
                                value=max_position_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Min Confidence Score:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=min_confidence_id,
                                options=min_confidence_options,
                                value=min_confidence_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Adjustment Method:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=adjustment_method_id,
                                options=adjustment_method_options,
                                value=adjustment_method_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Sector Filter:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=sector_filter_id,
                                options=sector_options,
                                value=sector_filter_default,
                                multi=True,
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
                            html.H4("Top 30 Positions by Kelly %", style={"marginBottom": "10px"}),
                            html.Div(
                                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label("Color By:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                                            dcc.Dropdown(
                                                id=color_by_id,
                                                options=color_by_options,
                                                value=color_by_default,
                                                style={"minWidth": "150px"}
                                            )
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                                    ),
                                ],
                            ),
                            dcc.Loading(
                                id=loading_1_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_1_id, style={"minHeight": "500px", "height": "calc(100vh - 700px)"}),
                                ]
                            ),
                            html.Pre(id=error_1_id, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.H4("Kelly % vs Expected Upside", style={"marginBottom": "10px"}),
                            html.Div(
                                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                                children=[
                                    html.Div(
                                        children=[
                                            html.Label("Color By:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                                            dcc.Dropdown(
                                                id=scatter_color_id,
                                                options=scatter_color_options,
                                                value=scatter_color_default,
                                                style={"minWidth": "150px"}
                                            )
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                                    ),
                                    html.Div(
                                        children=[
                                            html.Label("Size By:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                                            dcc.Dropdown(
                                                id=scatter_size_id,
                                                options=scatter_size_options,
                                                value=scatter_size_default,
                                                style={"minWidth": "150px"}
                                            )
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                                    ),
                                ],
                            ),
                            dcc.Loading(
                                id=loading_2_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_2_id, style={"minHeight": "500px", "height": "calc(100vh - 700px)"}),
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
        kelly_fraction_id: {
            "options": [opt["value"] for opt in kelly_fraction_options],
            "default": kelly_fraction_default
        },
        max_position_id: {
            "options": [opt["value"] for opt in max_position_options],
            "default": max_position_default
        },
        min_confidence_id: {
            "options": [opt["value"] for opt in min_confidence_options],
            "default": min_confidence_default
        },
        adjustment_method_id: {
            "options": [opt["value"] for opt in adjustment_method_options],
            "default": adjustment_method_default
        },
        sector_filter_id: {
            "options": [opt["value"] for opt in sector_options],
            "default": sector_filter_default
        },
        color_by_id: {
            "options": [opt["value"] for opt in color_by_options],
            "default": color_by_default
        },
        scatter_color_id: {
            "options": [opt["value"] for opt in scatter_color_options],
            "default": scatter_color_default
        },
        scatter_size_id: {
            "options": [opt["value"] for opt in scatter_size_options],
            "default": scatter_size_default
        },
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _calculate_kelly_metrics(df: pd.DataFrame, kelly_fraction: float, max_position: str, adjustment_method: str) -> pd.DataFrame:
    """Calculate Kelly Criterion metrics for each position."""
    logger.debug("Calculating Kelly Criterion metrics...")

    df = df.copy()

    df['prob_positive_upside'] = pd.to_numeric(df['prob_positive_upside'], errors='coerce')
    df['filtered_upside'] = pd.to_numeric(df['filtered_upside'], errors='coerce')
    df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
    df['achievement_probability'] = pd.to_numeric(df['achievement_probability'], errors='coerce')

    p = df['prob_positive_upside'] / 100.0
    q = 1.0 - p
    b = df['filtered_upside'] / 100.0

    df['kelly_raw'] = np.where(
        b != 0,
        (p * b - q) / b,
        0
    )

    df['kelly_raw'] = df['kelly_raw'].clip(lower=0)

    df['kelly_fractional'] = df['kelly_raw'] * kelly_fraction

    if adjustment_method == "confidence":
        df['kelly_adjusted'] = df['kelly_fractional'] * df['confidence_score']
    elif adjustment_method == "achievement":
        df['kelly_adjusted'] = df['kelly_fractional'] * df['achievement_probability']
    elif adjustment_method == "both":
        df['kelly_adjusted'] = df['kelly_fractional'] * df['confidence_score'] * df['achievement_probability']
    else:
        df['kelly_adjusted'] = df['kelly_fractional']

    if max_position != "no_cap":
        max_position_float = float(max_position)
        df['kelly_adjusted'] = df['kelly_adjusted'].clip(upper=max_position_float)

    total_kelly = df['kelly_adjusted'].sum()
    if total_kelly > 0:
        df['kelly_pct'] = (df['kelly_adjusted'] / total_kelly) * 100.0
    else:
        df['kelly_pct'] = 0

    logger.debug("Kelly metrics calculated:\n  • Mean Kelly %%: %.2f\n  • Max Kelly %%: %.2f", df['kelly_pct'].mean(), df['kelly_pct'].max())

    return df

def _update_chart_1(**kwargs) -> go.Figure:
    """Create bar chart of top 30 positions by Kelly percentage."""
    logger.debug("Updating chart 1 (bar chart) with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data is available to display", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    kelly_fraction = kwargs.get(kelly_fraction_id, kelly_fraction_default)
    if kelly_fraction is None:
        kelly_fraction = kelly_fraction_default

    max_position = kwargs.get(max_position_id, max_position_default)
    if max_position is None:
        max_position = max_position_default

    min_confidence = kwargs.get(min_confidence_id, min_confidence_default)
    if min_confidence is None:
        min_confidence = min_confidence_default

    adjustment_method = kwargs.get(adjustment_method_id, adjustment_method_default)
    if adjustment_method is None:
        adjustment_method = adjustment_method_default

    sector_filter = kwargs.get(sector_filter_id, sector_filter_default)
    if sector_filter is None:
        sector_filter = sector_filter_default

    color_by = kwargs.get(color_by_id, color_by_default)
    if color_by is None:
        color_by = color_by_default

    logger.debug("Filtering by minimum confidence score: %.2f...", min_confidence)
    df = df[df['confidence_score'] >= min_confidence]
    logger.debug(tbl(df))

    if sector_filter and "all" not in sector_filter:
        logger.debug("Filtering by sectors: %s...", ", ".join(sector_filter))
        df = df[df['sector'].isin(sector_filter)]
        logger.debug(tbl(df))

    df = _calculate_kelly_metrics(df, kelly_fraction, max_position, adjustment_method)

    logger.debug("Sorting by Kelly percentage and selecting top 30...")
    df = df.nlargest(30, 'kelly_pct')
    logger.debug(tbl(df))

    logger.debug("Creating bar chart...")
    logger.debug(schema(df))

    if color_by == "sector":
        fig = px.bar(
            df,
            x='ticker',
            y='kelly_pct',
            color='sector',
            labels={"kelly_pct": "Kelly % (Position Size)", "ticker": "Ticker"},
            hover_data={"sector": True, "kelly_pct": ":.2f", "confidence_level": True}
        )
        fig.update_layout(legend_title_text="Sector")
    elif color_by == "confidence_level":
        fig = px.bar(
            df,
            x='ticker',
            y='kelly_pct',
            color='confidence_level',
            labels={"kelly_pct": "Kelly % (Position Size)", "ticker": "Ticker"},
            hover_data={"confidence_level": True, "kelly_pct": ":.2f"}
        )
        fig.update_layout(legend_title_text="Confidence Level")
    else:
        fig = px.bar(
            df,
            x='ticker',
            y='kelly_pct',
            labels={"kelly_pct": "Kelly % (Position Size)", "ticker": "Ticker"},
            hover_data={"kelly_pct": ":.2f", "sector": True, "confidence_level": True}
        )

    fig.update_xaxes(tickangle=-45)
    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Kelly % (Position Size)",
        hovermode="x unified",
        minreducedwidth=400,
        minreducedheight=400
    )

    logger.debug("Done")
    return fig

def _update_chart_2(**kwargs) -> go.Figure:
    """Create scatter chart of Kelly % vs Expected Upside."""
    logger.debug("Updating chart 2 (scatter chart) with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data is available to display", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    kelly_fraction = kwargs.get(kelly_fraction_id, kelly_fraction_default)
    if kelly_fraction is None:
        kelly_fraction = kelly_fraction_default

    max_position = kwargs.get(max_position_id, max_position_default)
    if max_position is None:
        max_position = max_position_default

    min_confidence = kwargs.get(min_confidence_id, min_confidence_default)
    if min_confidence is None:
        min_confidence = min_confidence_default

    adjustment_method = kwargs.get(adjustment_method_id, adjustment_method_default)
    if adjustment_method is None:
        adjustment_method = adjustment_method_default

    sector_filter = kwargs.get(sector_filter_id, sector_filter_default)
    if sector_filter is None:
        sector_filter = sector_filter_default

    scatter_color = kwargs.get(scatter_color_id, scatter_color_default)
    if scatter_color is None:
        scatter_color = scatter_color_default

    scatter_size = kwargs.get(scatter_size_id, scatter_size_default)
    if scatter_size is None:
        scatter_size = scatter_size_default

    logger.debug("Filtering by minimum confidence score: %.2f...", min_confidence)
    df = df[df['confidence_score'] >= min_confidence]
    logger.debug(tbl(df))

    if sector_filter and "all" not in sector_filter:
        logger.debug("Filtering by sectors: %s...", ", ".join(sector_filter))
        df = df[df['sector'].isin(sector_filter)]
        logger.debug(tbl(df))

    df = _calculate_kelly_metrics(df, kelly_fraction, max_position, adjustment_method)

    logger.debug("Creating scatter chart...")
    logger.debug(schema(df))

    if scatter_color == "confidence_level" and scatter_size == "achievement_probability":
        fig = px.scatter(
            df,
            x='filtered_upside',
            y='kelly_pct',
            color='confidence_level',
            size='achievement_probability',
            hover_data={"ticker": True, "filtered_upside": ":.2f", "kelly_pct": ":.2f", "confidence_level": True, "achievement_probability": ":.2f"},
            labels={"filtered_upside": "Expected Upside (%)", "kelly_pct": "Kelly % (Position Size)"}
        )
        fig.update_layout(legend_title_text="Confidence Level")
    elif scatter_color == "confidence_level":
        fig = px.scatter(
            df,
            x='filtered_upside',
            y='kelly_pct',
            color='confidence_level',
            hover_data={"ticker": True, "filtered_upside": ":.2f", "kelly_pct": ":.2f", "confidence_level": True},
            labels={"filtered_upside": "Expected Upside (%)", "kelly_pct": "Kelly % (Position Size)"}
        )
        fig.update_layout(legend_title_text="Confidence Level")
    elif scatter_size == "achievement_probability":
        fig = px.scatter(
            df,
            x='filtered_upside',
            y='kelly_pct',
            size='achievement_probability',
            hover_data={"ticker": True, "filtered_upside": ":.2f", "kelly_pct": ":.2f", "achievement_probability": ":.2f"},
            labels={"filtered_upside": "Expected Upside (%)", "kelly_pct": "Kelly % (Position Size)"}
        )
    else:
        fig = px.scatter(
            df,
            x='filtered_upside',
            y='kelly_pct',
            hover_data={"ticker": True, "filtered_upside": ":.2f", "kelly_pct": ":.2f"},
            labels={"filtered_upside": "Expected Upside (%)", "kelly_pct": "Kelly % (Position Size)"}
        )

    fig.update_traces(marker=dict(sizemin=6))
    fig.update_layout(
        xaxis_title="Expected Upside (%)",
        yaxis_title="Kelly % (Position Size)",
        hovermode="closest"
    )

    logger.debug("Done")
    return fig

@callback(
    output=[
        Output(f"{component_id}_graph_1", "figure"),
        Output(f"{component_id}_error_1", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        kelly_fraction_id: Input(kelly_fraction_id, "value"),
        max_position_id: Input(max_position_id, "value"),
        min_confidence_id: Input(min_confidence_id, "value"),
        adjustment_method_id: Input(adjustment_method_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        color_by_id: Input(color_by_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update_chart_1(**kwargs) -> Tuple[go.Figure, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure = _update_chart_1(**kwargs)
        return figure, ""
    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg

@callback(
    output=[
        Output(f"{component_id}_graph_2", "figure"),
        Output(f"{component_id}_error_2", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        kelly_fraction_id: Input(kelly_fraction_id, "value"),
        max_position_id: Input(max_position_id, "value"),
        min_confidence_id: Input(min_confidence_id, "value"),
        adjustment_method_id: Input(adjustment_method_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        scatter_color_id: Input(scatter_color_id, "value"),
        scatter_size_id: Input(scatter_size_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update_chart_2(**kwargs) -> Tuple[go.Figure, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure = _update_chart_2(**kwargs)
        return figure, ""
    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg