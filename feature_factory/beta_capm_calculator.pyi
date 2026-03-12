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

component_id = "beta_capm_systematic_risk_return"

risk_free_rate_id = f"{component_id}_risk_free_rate"
risk_free_rate_options = [
    {"label": "0%", "value": 0.0},
    {"label": "2%", "value": 2.0},
    {"label": "3%", "value": 3.0},
    {"label": "4%", "value": 4.0},
    {"label": "5%", "value": 5.0}
]
risk_free_rate_default = 3.0

market_return_id = f"{component_id}_market_return"
market_return_options = [
    {"label": "5%", "value": 5.0},
    {"label": "8%", "value": 8.0},
    {"label": "10%", "value": 10.0},
    {"label": "12%", "value": 12.0},
    {"label": "15%", "value": 15.0}
]
market_return_default = 10.0

sector_filter_id = f"{component_id}_sector_filter"

size_encoding_id = f"{component_id}_size_encoding"
size_encoding_options = [
    {"label": "Market Cap", "value": "market_cap"},
    {"label": "None", "value": "none"}
]
size_encoding_default = "market_cap"

confidence_level_id = f"{component_id}_confidence_level"
confidence_level_options = [
    {"label": "High Only", "value": "high_only"},
    {"label": "High or Medium", "value": "high_medium"},
    {"label": "All", "value": "all"}
]
confidence_level_default = "high_medium"

def component() -> ComponentResponse:
    graph_1_id = f"{component_id}_scatter_graph"
    error_1_id = f"{component_id}_scatter_error"
    loading_1_id = f"{component_id}_scatter_loading"

    graph_2_id = f"{component_id}_bar_graph"
    error_2_id = f"{component_id}_bar_error"
    loading_2_id = f"{component_id}_bar_loading"

    title = "Beta & CAPM: Systematic Risk and Expected Return"
    description = "Analyze stock sensitivity to market movements (beta) and expected returns using CAPM. Positive alpha indicates outperformance vs. CAPM prediction."

    df_sample = get_data()
    sectors = sorted(df_sample["sector"].dropna().unique().tolist())
    sector_options = [{"label": s, "value": s} for s in sectors]
    sector_default = sectors

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Risk-Free Rate:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=risk_free_rate_id,
                                options=risk_free_rate_options,
                                value=risk_free_rate_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Market Return:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=market_return_id,
                                options=market_return_options,
                                value=market_return_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Sector:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=sector_filter_id,
                                options=sector_options,
                                value=sector_default,
                                multi=True,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Size Encoding:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=size_encoding_id,
                                options=size_encoding_options,
                                value=size_encoding_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Confidence Level:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=confidence_level_id,
                                options=confidence_level_options,
                                value=confidence_level_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "gap": "10px", "marginBottom": "15px"},
                children=[
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Beta vs Expected Return", style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                            dcc.Loading(
                                id=loading_1_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_1_id, style={"minHeight": "500px"}),
                                ]
                            ),
                            html.Pre(id=error_1_id, style={"color": "red", "margin": "10px 0", "fontSize": "12px"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Alpha (Excess Return)", style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                            dcc.Loading(
                                id=loading_2_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_2_id, style={"minHeight": "500px"}),
                                ]
                            ),
                            html.Pre(id=error_2_id, style={"color": "red", "margin": "10px 0", "fontSize": "12px"}),
                        ]
                    ),
                ]
            ),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        risk_free_rate_id: {
            "options": [opt["value"] for opt in risk_free_rate_options],
            "default": risk_free_rate_default
        },
        market_return_id: {
            "options": [opt["value"] for opt in market_return_options],
            "default": market_return_default
        },
        sector_filter_id: {
            "options": sectors,
            "default": sector_default
        },
        size_encoding_id: {
            "options": [opt["value"] for opt in size_encoding_options],
            "default": size_encoding_default
        },
        confidence_level_id: {
            "options": [opt["value"] for opt in confidence_level_options],
            "default": confidence_level_default
        },
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _calculate_beta(sector: str) -> float:
    """Calculate beta based on sector classification."""
    if sector in ["Information Technology", "Health Care"]:
        return 1.3
    elif sector in ["Utilities", "Consumer Staples"]:
        return 0.7
    elif sector in ["Financials", "Industrials"]:
        return 1.0
    else:
        return 1.1

def _calculate_capm_return(beta: float, rf: float, rm: float) -> float:
    """Calculate expected return using CAPM formula: E(R) = Rf + β(Rm - Rf)."""
    return rf + beta * (rm - rf)

def _update_scatter_logic(**kwargs) -> go.Figure:
    """Core scatter chart update logic."""
    logger.debug("Updating scatter chart with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data is available to display", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    logger.debug("Filtering by confidence level...")
    confidence_level = kwargs.get(confidence_level_id, confidence_level_default)
    if confidence_level == "high_only":
        df = df[df["confidence_level"] == "High"]
    elif confidence_level == "high_medium":
        df = df[df["confidence_level"].isin(["High", "Medium"])]

    logger.debug(tbl(df))

    logger.debug("Filtering by sector...")
    sectors = kwargs.get(sector_filter_id, [])
    if sectors and len(sectors) > 0:
        df = df[df["sector"].isin(sectors)]
    logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available after filtering",
            annotations=[{"text": "No data matches the selected filters", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    logger.debug("Computing beta and CAPM expected return...")
    rf = kwargs.get(risk_free_rate_id, risk_free_rate_default)
    rm = kwargs.get(market_return_id, market_return_default)

    if rf is None:
        rf = risk_free_rate_default
    if rm is None:
        rm = market_return_default

    rf = float(rf)
    rm = float(rm)

    df["beta"] = df["sector"].apply(_calculate_beta)
    df["capm_return"] = df["beta"].apply(lambda b: _calculate_capm_return(b, rf, rm))

    logger.debug(schema(df))
    logger.debug(tbl(df))

    logger.debug("Creating scatter chart...")
    size_encoding = kwargs.get(size_encoding_id, size_encoding_default)

    if size_encoding == "market_cap":
        fig = px.scatter(
            df,
            x="beta",
            y="expected_upside_pct",
            color="sector",
            size="market_cap",
            hover_data={"name": True, "ticker": True, "beta": ":.2f", "expected_upside_pct": ":.2f", "market_cap": ":.0f", "sector": True},
            labels={
                "beta": "Beta (Systematic Risk)",
                "expected_upside_pct": "Expected Return (%)",
                "sector": "Sector"
            }
        )
        fig.update_traces(marker=dict(sizemin=6))
    else:
        fig = px.scatter(
            df,
            x="beta",
            y="expected_upside_pct",
            color="sector",
            hover_data={"name": True, "ticker": True, "beta": ":.2f", "expected_upside_pct": ":.2f", "sector": True},
            labels={
                "beta": "Beta (Systematic Risk)",
                "expected_upside_pct": "Expected Return (%)",
                "sector": "Sector"
            }
        )

    fig.update_layout(
        xaxis_title="Beta (Systematic Risk)",
        yaxis_title="Expected Return (%)",
        legend_title_text="Sector",
        hovermode="closest"
    )

    logger.debug("Done")
    return fig

def _update_bar_logic(**kwargs) -> go.Figure:
    """Core bar chart update logic."""
    logger.debug("Updating bar chart with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data is available to display", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    logger.debug("Filtering by confidence level...")
    confidence_level = kwargs.get(confidence_level_id, confidence_level_default)
    if confidence_level == "high_only":
        df = df[df["confidence_level"] == "High"]
    elif confidence_level == "high_medium":
        df = df[df["confidence_level"].isin(["High", "Medium"])]

    logger.debug(tbl(df))

    logger.debug("Filtering by sector...")
    sectors = kwargs.get(sector_filter_id, [])
    if sectors and len(sectors) > 0:
        df = df[df["sector"].isin(sectors)]
    logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available after filtering",
            annotations=[{"text": "No data matches the selected filters", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    logger.debug("Computing beta, CAPM return, and alpha...")
    rf = kwargs.get(risk_free_rate_id, risk_free_rate_default)
    rm = kwargs.get(market_return_id, market_return_default)

    if rf is None:
        rf = risk_free_rate_default
    if rm is None:
        rm = market_return_default

    rf = float(rf)
    rm = float(rm)

    df["beta"] = df["sector"].apply(_calculate_beta)
    df["capm_return"] = df["beta"].apply(lambda b: _calculate_capm_return(b, rf, rm))
    df["alpha"] = df["expected_upside_pct"] - df["capm_return"]

    logger.debug(schema(df))
    logger.debug(tbl(df))

    logger.debug("Sorting by alpha and selecting top/bottom stocks...")
    df_sorted = df.sort_values("alpha", ascending=False)
    top_n = 15
    top_stocks = df_sorted.head(top_n)
    bottom_stocks = df_sorted.tail(top_n)
    df_display = pd.concat([top_stocks, bottom_stocks]).sort_values("alpha", ascending=True)

    logger.debug("Creating bar chart...")
    df_display["alpha_color"] = df_display["alpha"].apply(lambda x: "Positive" if x > 0 else "Negative")

    fig = px.bar(
        df_display,
        x="alpha",
        y="name",
        color="alpha_color",
        orientation="h",
        hover_data={"ticker": True, "alpha": ":.2f", "expected_upside_pct": ":.2f", "capm_return": ":.2f", "alpha_color": False},
        labels={
            "alpha": "Alpha (Excess Return %)",
            "name": "Company",
            "alpha_color": "Alpha Sign"
        },
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"}
    )

    fig.update_layout(
        xaxis_title="Alpha (Excess Return %)",
        yaxis_title="Company",
        legend_title_text="Alpha Sign",
        height=max(400, len(df_display) * 20),
        hovermode="closest"
    )

    logger.debug("Done")
    return fig

@callback(
    output=[
        Output(f"{component_id}_scatter_graph", "figure"),
        Output(f"{component_id}_scatter_error", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        market_return_id: Input(market_return_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        size_encoding_id: Input(size_encoding_id, "value"),
        confidence_level_id: Input(confidence_level_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update_scatter(**kwargs) -> Tuple[go.Figure, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure = _update_scatter_logic(**kwargs)
        return figure, ""
    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg

@callback(
    output=[
        Output(f"{component_id}_bar_graph", "figure"),
        Output(f"{component_id}_bar_error", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        market_return_id: Input(market_return_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        confidence_level_id: Input(confidence_level_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update_bar(**kwargs) -> Tuple[go.Figure, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure = _update_bar_logic(**kwargs)
        return figure, ""
    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg