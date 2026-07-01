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


component_id = "sharpe_ratio_risk_adjusted_return"


def component() -> ComponentResponse:
    graph_id_1 = f"{component_id}_graph_1"
    graph_id_2 = f"{component_id}_graph_2"
    error_id_1 = f"{component_id}_error_1"
    error_id_2 = f"{component_id}_error_2"
    loading_id_1 = f"{component_id}_loading_1"
    loading_id_2 = f"{component_id}_loading_2"

    risk_free_rate_id = f"{component_id}_risk_free_rate"
    prob_pos_threshold_id = f"{component_id}_prob_pos_threshold"
    min_market_cap_id = f"{component_id}_min_market_cap"
    top_stocks_id = f"{component_id}_top_stocks"
    sector_filter_id = f"{component_id}_sector_filter"

    risk_free_rate_options = [
        {"label": "0%", "value": 0.0},
        {"label": "2%", "value": 0.02},
        {"label": "3%", "value": 0.03},
        {"label": "4%", "value": 0.04},
        {"label": "5%", "value": 0.05}
    ]
    risk_free_rate_default = 0.03

    prob_pos_threshold_options = [
        {"label": "0.5", "value": 0.5},
        {"label": "0.7", "value": 0.7},
        {"label": "0.8", "value": 0.8},
        {"label": "0.9", "value": 0.9},
        {"label": "0.95", "value": 0.95}
    ]
    prob_pos_threshold_default = 0.7

    min_market_cap_options = [
        {"label": "1,000", "value": 1000},
        {"label": "5,000", "value": 5000},
        {"label": "10,000", "value": 10000},
        {"label": "50,000", "value": 50000}
    ]
    min_market_cap_default = 5000

    top_stocks_options = [
        {"label": "10", "value": 10},
        {"label": "20", "value": 20},
        {"label": "50", "value": 50},
        {"label": "100", "value": 100},
        {"label": "All", "value": "all"}
    ]
    top_stocks_default = 50

    title = "Sharpe Ratio & Risk-Adjusted Return Comparison"
    description = "Compares stocks by their risk-adjusted performance using the Sharpe ratio, which measures excess return per unit of risk. Higher values indicate better risk-adjusted returns—use this to identify stocks that deliver strong returns relative to their volatility."

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
                            html.Label("Risk-Free Rate:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
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
                            html.Label("Min Prob Positive:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=prob_pos_threshold_id,
                                options=prob_pos_threshold_options,
                                value=prob_pos_threshold_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Min Market Cap:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=min_market_cap_id,
                                options=min_market_cap_options,
                                value=min_market_cap_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Top Stocks:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=top_stocks_id,
                                options=top_stocks_options,
                                value=top_stocks_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Sector:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=sector_filter_id,
                                options=[],
                                value=[],
                                multi=True,
                                style={"minWidth": "200px"}
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
                            dcc.Loading(
                                id=loading_id_1,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_id_1,
                                              style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                                ]
                            ),
                            html.Pre(id=error_id_1, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            dcc.Loading(
                                id=loading_id_2,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_id_2,
                                              style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                                ]
                            ),
                            html.Pre(id=error_id_2, style={"color": "red", "margin": "10px 0"}),
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
        prob_pos_threshold_id: {
            "options": [opt["value"] for opt in prob_pos_threshold_options],
            "default": prob_pos_threshold_default
        },
        min_market_cap_id: {
            "options": [opt["value"] for opt in min_market_cap_options],
            "default": min_market_cap_default
        },
        top_stocks_id: {
            "options": [opt["value"] for opt in top_stocks_options],
            "default": top_stocks_default
        },
        sector_filter_id: {
            "options": [],
            "default": []
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }


def _update_logic(**kwargs) -> Tuple[go.Figure, go.Figure]:
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
        return empty_fig, empty_fig

    logger.debug("Selecting columns for analysis...")
    df = df[['name', 'sector', 'market_cap', 'expected_return_kalman', 'kalman_variance', 'mc_prob_pos']].copy()
    logger.debug(schema(df))
    logger.debug(tbl(df))

    risk_free_rate = kwargs.get(f'{component_id}_risk_free_rate', 0.03)
    if risk_free_rate is None:
        risk_free_rate = 0.03

    prob_pos_threshold = kwargs.get(f'{component_id}_prob_pos_threshold', 0.7)
    if prob_pos_threshold is None:
        prob_pos_threshold = 0.7

    min_market_cap = kwargs.get(f'{component_id}_min_market_cap', 5000)
    if min_market_cap is None:
        min_market_cap = 5000

    top_stocks = kwargs.get(f'{component_id}_top_stocks', 50)
    if top_stocks is None:
        top_stocks = 50
    elif top_stocks == "all":
        top_stocks = None

    sector_filter = kwargs.get(f'{component_id}_sector_filter', [])
    if sector_filter is None:
        sector_filter = []

    logger.debug("Computing annualized metrics...")
    df['annualized_return'] = df['expected_return_kalman'] * 252 * 100
    df['annualized_volatility'] = np.sqrt(df['kalman_variance']) * np.sqrt(252) * 100
    logger.debug(tbl(df[['name', 'annualized_return', 'annualized_volatility']]))

    logger.debug("Computing Sharpe ratio...")
    df['sharpe_ratio'] = (df['annualized_return'] - (risk_free_rate * 100)) / df['annualized_volatility']
    logger.debug(tbl(df[['name', 'sharpe_ratio']]))

    logger.debug("Filtering by mc_prob_pos threshold: %.2f...", prob_pos_threshold)
    df = df[df['mc_prob_pos'] >= prob_pos_threshold]
    logger.debug(tbl(df))

    logger.debug("Filtering by minimum market cap: %.0f...", min_market_cap)
    df = df[df['market_cap'] >= min_market_cap]
    logger.debug(tbl(df))

    if len(sector_filter) > 0:
        logger.debug("Filtering by sectors: %s...", ", ".join(sector_filter))
        df = df[df['sector'].isin(sector_filter)]
        logger.debug(tbl(df))

    logger.debug("Sorting by Sharpe ratio descending...")
    df = df.sort_values('sharpe_ratio', ascending=False)
    logger.debug(tbl(df))

    if top_stocks is not None:
        logger.debug("Selecting top %d stocks...", top_stocks)
        df = df.head(top_stocks)
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
        return empty_fig, empty_fig

    logger.debug("Creating bar chart with final data...")
    logger.debug(schema(df))
    logger.debug(tbl(df))

    fig1 = px.bar(
        df,
        x='sharpe_ratio',
        y='name',
        color='sector',
        orientation='h',
        title="Sharpe Ratio by Stock"
    )
    fig1.update_layout(
        yaxis_title="Stock",
        xaxis_title="Sharpe Ratio",
        height=max(400, len(df) * 20),
        hovermode='closest'
    )
    fig1.update_yaxes(autorange="reversed")

    logger.debug("Creating scatter chart with final data...")
    fig2 = px.scatter(
        df,
        x='annualized_volatility',
        y='annualized_return',
        size='market_cap',
        color='sharpe_ratio',
        hover_name='name',
        title="Risk-Return Profile",
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        hovermode='closest'
    )
    fig2.update_traces(marker=dict(sizemin=6))

    logger.debug("Done")
    return fig1, fig2


@callback(
    output=[
        Output(f"{component_id}_graph_1", "figure"),
        Output(f"{component_id}_error_1", "children"),
        Output(f"{component_id}_graph_2", "figure"),
        Output(f"{component_id}_error_2", "children"),
        Output(f"{component_id}_sector_filter", "options")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        f'{component_id}_risk_free_rate': Input(f"{component_id}_risk_free_rate", "value"),
        f'{component_id}_prob_pos_threshold': Input(f"{component_id}_prob_pos_threshold", "value"),
        f'{component_id}_min_market_cap': Input(f"{component_id}_min_market_cap", "value"),
        f'{component_id}_top_stocks': Input(f"{component_id}_top_stocks", "value"),
        f'{component_id}_sector_filter': Input(f"{component_id}_sector_filter", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, go.Figure, str, list]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        df_all = filter_data(get_data(), **kwargs)

        if len(df_all) == 0:
            sector_options = []
        else:
            try:
                unique_sectors = df_all['sector'].dropna().replace('', np.nan).dropna().unique().tolist()
            except TypeError:
                unique_sectors = df_all['sector'].dropna().replace('', np.nan).dropna().astype(str).unique().tolist()
            sector_options = [{'label': str(v), 'value': v} for v in sorted(unique_sectors) if
                              v is not None and str(v).strip()]

        fig1, fig2 = _update_logic(**kwargs)
        return fig1, "", fig2, "", sector_options

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg, empty_fig, error_msg, []