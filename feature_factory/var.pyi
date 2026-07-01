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


component_id = "value_at_risk_downside_assessment"

confidence_level_id = f"{component_id}_confidence_level"
confidence_level_options = [
    {"label": "5%", "value": "5"},
    {"label": "10%", "value": "10"},
    {"label": "25%", "value": "25"},
    {"label": "50%", "value": "50"}
]
confidence_level_default = "5"

sort_metric_id = f"{component_id}_sort_metric"
sort_metric_options = [
    {"label": "Highest Reward-to-CVaR", "value": "reward_to_cvar"},
    {"label": "Lowest CVaR", "value": "cvar_lowest"},
    {"label": "Highest Expected Return", "value": "expected_return"}
]
sort_metric_default = "reward_to_cvar"

min_market_cap_id = f"{component_id}_min_market_cap"
min_market_cap_options = [
    {"label": "1,000M", "value": "1000"},
    {"label": "5,000M", "value": "5000"},
    {"label": "10,000M", "value": "10000"},
    {"label": "50,000M", "value": "50000"}
]
min_market_cap_default = "5000"

min_prob_positive_id = f"{component_id}_min_prob_positive"
min_prob_positive_options = [
    {"label": "50%", "value": "0.5"},
    {"label": "70%", "value": "0.7"},
    {"label": "80%", "value": "0.8"},
    {"label": "90%", "value": "0.9"}
]
min_prob_positive_default = "0.7"

num_stocks_id = f"{component_id}_num_stocks"
num_stocks_options = [
    {"label": "20", "value": "20"},
    {"label": "50", "value": "50"},
    {"label": "100", "value": "100"},
    {"label": "All", "value": "all"}
]
num_stocks_default = "50"

sector_filter_id = f"{component_id}_sector_filter"


def component() -> ComponentResponse:
    graph_1_id = f"{component_id}_graph_1"
    error_1_id = f"{component_id}_error_1"
    loading_1_id = f"{component_id}_loading_1"

    graph_2_id = f"{component_id}_graph_2"
    error_2_id = f"{component_id}_error_2"
    loading_2_id = f"{component_id}_loading_2"

    title = "Value at Risk (VaR): Downside Risk Assessment"
    description = "Calculates the maximum expected loss at different confidence levels using Conditional Value at Risk (CVaR). Lower CVaR values indicate less downside risk."

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
                            html.Label("Confidence Level:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=confidence_level_id,
                                options=confidence_level_options,
                                value=confidence_level_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Sort By:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=sort_metric_id,
                                options=sort_metric_options,
                                value=sort_metric_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Min Market Cap (M):",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=min_market_cap_id,
                                options=min_market_cap_options,
                                value=min_market_cap_default,
                                style={"minWidth": "200px"},
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
                                id=min_prob_positive_id,
                                options=min_prob_positive_options,
                                value=min_prob_positive_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Number of Stocks:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=num_stocks_id,
                                options=num_stocks_options,
                                value=num_stocks_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Sectors:",
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
                            html.Label("CVaR by Stock and Sector",
                                       style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                            dcc.Loading(
                                id=loading_1_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_1_id,
                                              style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                                ]
                            ),
                            html.Pre(id=error_1_id, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Risk-Return Profile",
                                       style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                            dcc.Loading(
                                id=loading_2_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_2_id,
                                              style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
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
        confidence_level_id: {
            "options": [option["value"] for option in confidence_level_options],
            "default": confidence_level_default
        },
        sort_metric_id: {
            "options": [option["value"] for option in sort_metric_options],
            "default": sort_metric_default
        },
        min_market_cap_id: {
            "options": [option["value"] for option in min_market_cap_options],
            "default": min_market_cap_default
        },
        min_prob_positive_id: {
            "options": [option["value"] for option in min_prob_positive_options],
            "default": min_prob_positive_default
        },
        num_stocks_id: {
            "options": [option["value"] for option in num_stocks_options],
            "default": num_stocks_default
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
    logger.debug("Updating VaR charts with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

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
    df = df[['name', 'sector', 'market_cap', 'mc_prob_pos', 'cvar_5pct_kalman',
             'expected_return_kalman', 'reward_to_cvar', 'er_p05', 'er_p50', 'er_p95']].copy()
    logger.debug(schema(df))
    logger.debug(tbl(df))

    confidence_level = kwargs.get(confidence_level_id, confidence_level_default)
    if confidence_level is None:
        confidence_level = confidence_level_default
    confidence_level = str(confidence_level)

    sort_metric = kwargs.get(sort_metric_id, sort_metric_default)
    if sort_metric is None:
        sort_metric = sort_metric_default

    min_market_cap = kwargs.get(min_market_cap_id, min_market_cap_default)
    if min_market_cap is None:
        min_market_cap = min_market_cap_default
    min_market_cap = float(min_market_cap)

    min_prob_positive = kwargs.get(min_prob_positive_id, min_prob_positive_default)
    if min_prob_positive is None:
        min_prob_positive = min_prob_positive_default
    min_prob_positive = float(min_prob_positive)

    num_stocks = kwargs.get(num_stocks_id, num_stocks_default)
    if num_stocks is None:
        num_stocks = num_stocks_default
    num_stocks = str(num_stocks)

    sector_filter = kwargs.get(sector_filter_id, [])
    if sector_filter is None:
        sector_filter = []

    logger.debug("Filtering by market cap >= %.0f and probability of positive return >= %.2f...", min_market_cap,
                 min_prob_positive)
    df = df[(df['market_cap'] >= min_market_cap) & (df['mc_prob_pos'] >= min_prob_positive)]
    logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{
                "text": "No stocks match the selected criteria",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, empty_fig

    if sector_filter and len(sector_filter) > 0:
        logger.debug("Filtering by sectors: %s...", sector_filter)
        df = df[df['sector'].isin(sector_filter)]
        logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{
                "text": "No stocks match the selected sectors",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, empty_fig

    logger.debug("Computing CVaR based on confidence level: %s%%...", confidence_level)
    if confidence_level == "5":
        df['cvar_value'] = df['er_p05']
    elif confidence_level == "10":
        df['cvar_value'] = (df['er_p05'] + df['er_p50']) / 2
    elif confidence_level == "25":
        df['cvar_value'] = df['er_p50']
    elif confidence_level == "50":
        df['cvar_value'] = df['er_p95']
    else:
        df['cvar_value'] = df['cvar_5pct_kalman']

    logger.debug("Sorting by metric: %s...", sort_metric)
    if sort_metric == "cvar_lowest":
        df = df.sort_values('cvar_value', ascending=True)
    elif sort_metric == "reward_to_cvar":
        df = df.sort_values('reward_to_cvar', ascending=False)
    elif sort_metric == "expected_return":
        df = df.sort_values('expected_return_kalman', ascending=False)
    logger.debug(tbl(df))

    if num_stocks != "all":
        num_stocks_int = int(num_stocks)
        logger.debug("Limiting to top %d stocks...", num_stocks_int)
        df = df.head(num_stocks_int)
        logger.debug(tbl(df))

    logger.debug("Creating bar chart...")
    fig_1 = px.bar(
        df,
        x='name',
        y='cvar_value',
        color='sector',
        title=f"CVaR at {confidence_level}% Confidence Level",
        labels={'name': 'Stock', 'cvar_value': f'CVaR ({confidence_level}%)', 'sector': 'Sector'},
        hover_data={'name': True, 'sector': True, 'cvar_value': ':.2f', 'market_cap': ':.0f'}
    )
    fig_1.update_xaxes(tickangle=-45)
    fig_1.update_layout(
        minreducedwidth=400,
        minreducedheight=400,
        hovermode='closest'
    )

    logger.debug("Creating scatter chart...")
    fig_2 = px.scatter(
        df,
        x='expected_return_kalman',
        y='cvar_value',
        size='market_cap',
        color='reward_to_cvar',
        hover_data={'name': True, 'sector': True, 'expected_return_kalman': ':.3f', 'cvar_value': ':.2f',
                    'reward_to_cvar': ':.2f', 'market_cap': ':.0f'},
        labels={
            'expected_return_kalman': 'Expected Return (Kalman)',
            'cvar_value': f'CVaR ({confidence_level}%)',
            'reward_to_cvar': 'Reward-to-CVaR',
            'market_cap': 'Market Cap (M)'
        },
        title="Risk-Return Profile"
    )
    fig_2.update_traces(marker=dict(sizemin=6))
    fig_2.update_layout(
        minreducedwidth=400,
        minreducedheight=400,
        hovermode='closest'
    )

    logger.debug("Done")
    return fig_1, fig_2


@callback(
    output=[
        Output(f"{component_id}_graph_1", "figure"),
        Output(f"{component_id}_error_1", "children"),
        Output(f"{component_id}_graph_2", "figure"),
        Output(f"{component_id}_error_2", "children"),
        Output(sector_filter_id, "options")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        confidence_level_id: Input(confidence_level_id, "value"),
        sort_metric_id: Input(sort_metric_id, "value"),
        min_market_cap_id: Input(min_market_cap_id, "value"),
        min_prob_positive_id: Input(min_prob_positive_id, "value"),
        num_stocks_id: Input(num_stocks_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
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
            sector_options = [{'label': str(v), 'value': v} for v in unique_sectors if v is not None and str(v).strip()]

        fig_1, fig_2 = _update_logic(**kwargs)
        return fig_1, "", fig_2, "", sector_options

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg, empty_fig, error_msg, []