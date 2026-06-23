"""Sharpe Ratio & Risk-Adjusted Return Comparison.

Two side-by-side charts: a horizontal Sharpe-by-stock bar and a Viridis
risk-return scatter. Ported from ``feature_factory/sharpe_ratio.pyi``.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ..data import get_data
from ..logger import logger, schema, tbl
from ..theme import GRAPH_STYLE, control
from ..theme import card as theme_card
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data

component_id = "sharpe_ratio_risk_adjusted_return"

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
    {"label": "5%", "value": 0.05},
]
risk_free_rate_default = 0.03

prob_pos_threshold_options = [
    {"label": "0.5", "value": 0.5},
    {"label": "0.7", "value": 0.7},
    {"label": "0.8", "value": 0.8},
    {"label": "0.9", "value": 0.9},
    {"label": "0.95", "value": 0.95},
]
prob_pos_threshold_default = 0.7

min_market_cap_options = [
    {"label": "1,000", "value": 1000},
    {"label": "5,000", "value": 5000},
    {"label": "10,000", "value": 10000},
    {"label": "50,000", "value": 50000},
]
min_market_cap_default = 5000

top_stocks_options = [
    {"label": "10", "value": 10},
    {"label": "20", "value": 20},
    {"label": "50", "value": 50},
    {"label": "100", "value": 100},
    {"label": "All", "value": "all"},
]
top_stocks_default = 50

title = "Sharpe Ratio & Risk-Adjusted Return Comparison"
description = (
    "Compares stocks by their risk-adjusted performance using the Sharpe ratio, "
    "which measures excess return per unit of risk. Higher values indicate "
    "better risk-adjusted returns—use this to identify stocks that deliver "
    "strong returns relative to their volatility."
)


def component() -> "object":
    graph_id_1 = f"{component_id}_graph_1"
    graph_id_2 = f"{component_id}_graph_2"
    error_id_1 = f"{component_id}_error_1"
    error_id_2 = f"{component_id}_error_2"

    df = get_data()
    try:
        sectors = sorted(s for s in df["sector"].dropna().unique().tolist() if s)
    except Exception:  # pragma: no cover - defensive
        sectors = []
    sector_opts = [{"label": s, "value": s} for s in sectors]

    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "rowGap": "10px"},
                children=[
                    control(
                        "Risk-Free Rate:",
                        dcc.Dropdown(id=risk_free_rate_id, options=risk_free_rate_options,
                                     value=risk_free_rate_default, searchable=False,
                                     style={"minWidth": "150px"}),
                    ),
                    control(
                        "Min Prob Positive:",
                        dcc.Dropdown(id=prob_pos_threshold_id, options=prob_pos_threshold_options,
                                     value=prob_pos_threshold_default, searchable=False,
                                     style={"minWidth": "150px"}),
                    ),
                    control(
                        "Min Market Cap:",
                        dcc.Dropdown(id=min_market_cap_id, options=min_market_cap_options,
                                     value=min_market_cap_default, searchable=False,
                                     style={"minWidth": "150px"}),
                    ),
                    control(
                        "Top Stocks:",
                        dcc.Dropdown(id=top_stocks_id, options=top_stocks_options,
                                     value=top_stocks_default, searchable=False,
                                     style={"minWidth": "150px"}),
                    ),
                    control(
                        "Sector:",
                        dcc.Dropdown(id=sector_filter_id, options=sector_opts, value=[],
                                     multi=True, style={"minWidth": "200px"}),
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "gap": "10px"},
                children=[
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            dcc.Loading(type="circle",
                                        children=[dcc.Graph(id=graph_id_1, style=GRAPH_STYLE)]),
                            html.Pre(id=error_id_1, className="geib-error"),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            dcc.Loading(type="circle",
                                        children=[dcc.Graph(id=graph_id_2, style=GRAPH_STYLE)]),
                            html.Pre(id=error_id_2, className="geib-error"),
                        ],
                    ),
                ],
            ),
        ],
    )


def _update_logic(**kwargs) -> Tuple[go.Figure, go.Figure]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        empty = _empty("No data is available to display")
        return empty, empty

    df = df[["name", "sector", "market_cap", "expected_return_kalman", "kalman_variance", "mc_prob_pos"]].copy()
    logger.debug(schema(df))

    risk_free_rate = _coalesce(kwargs.get(risk_free_rate_id), risk_free_rate_default)
    prob_pos_threshold = _coalesce(kwargs.get(prob_pos_threshold_id), prob_pos_threshold_default)
    min_market_cap = _coalesce(kwargs.get(min_market_cap_id), min_market_cap_default)
    top_stocks = kwargs.get(top_stocks_id, top_stocks_default)
    if top_stocks is None:
        top_stocks = top_stocks_default
    elif top_stocks == "all":
        top_stocks = None
    sector_filter = kwargs.get(sector_filter_id) or []

    df["annualized_return"] = df["expected_return_kalman"] * 252 * 100
    df["annualized_volatility"] = np.sqrt(df["kalman_variance"]) * np.sqrt(252) * 100
    df["sharpe_ratio"] = (df["annualized_return"] - (risk_free_rate * 100)) / df["annualized_volatility"]

    df = df[df["mc_prob_pos"] >= prob_pos_threshold]
    df = df[df["market_cap"] >= min_market_cap]
    if len(sector_filter) > 0:
        df = df[df["sector"].isin(sector_filter)]

    df = df.sort_values("sharpe_ratio", ascending=False)
    if top_stocks is not None:
        df = df.head(top_stocks)

    if len(df) == 0:
        empty = _empty("No data matches the selected filters")
        return empty, empty
    logger.debug(tbl(df))

    fig1 = px.bar(df, x="sharpe_ratio", y="name", color="sector", orientation="h",
                  title="Sharpe Ratio by Stock")
    fig1.update_layout(yaxis_title="Stock", xaxis_title="Sharpe Ratio",
                       height=max(400, len(df) * 20), hovermode="closest")
    fig1.update_yaxes(autorange="reversed")

    fig2 = px.scatter(df, x="annualized_volatility", y="annualized_return", size="market_cap",
                      color="sharpe_ratio", hover_name="name", title="Risk-Return Profile",
                      color_continuous_scale="Viridis")
    fig2.update_layout(xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)",
                       hovermode="closest")
    fig2.update_traces(marker=dict(sizemin=6))
    return fig1, fig2


def _coalesce(value, default):
    return default if value is None else value


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(annotations=[{"text": message, "showarrow": False, "font": {"size": 18}}])
    return fig


@callback(
    output=[
        Output(f"{component_id}_graph_1", "figure"),
        Output(f"{component_id}_error_1", "children"),
        Output(f"{component_id}_graph_2", "figure"),
        Output(f"{component_id}_error_2", "children"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        prob_pos_threshold_id: Input(prob_pos_threshold_id, "value"),
        min_market_cap_id: Input(min_market_cap_id, "value"),
        top_stocks_id: Input(top_stocks_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, str, go.Figure, str]:
    try:
        fig1, fig2 = _update_logic(**kwargs)
        return fig1, "", fig2, ""
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        empty = _empty("An error occurred")
        return empty, msg, empty, msg
