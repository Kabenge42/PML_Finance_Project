"""Sharpe Ratio & Risk-Adjusted Return Comparison.

Two side-by-side charts: a horizontal Sharpe-by-stock bar and a Viridis
risk-return scatter. Ported from ``feature_factory/sharpe_ratio.pyi``.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import coalesce, empty_figure, scoped_filter, sector_values
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..components.probability_filter import (
    apply_probability_filter,
    probability_controls,
    probability_inputs,
    register as register_probability_filter,
)
from ..data import get_data
from ..logger import logger, schema, tbl
from ..metrics import annualized_return_pct, quantile_volatility_pct
from ..theme import GRAPH_STYLE, control
from ..theme import card as theme_card

component_id = "sharpe_ratio_risk_adjusted_return"

risk_free_rate_id = f"{component_id}_risk_free_rate"
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

# Probability metric + band (shared control pair). The low handle defaults to the
# 0.7 threshold this card's former "Min Prob Positive" dropdown applied.
prob_pos_threshold_default = 0.7
register_probability_filter(component_id)

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
    sector_opts = [{"label": s, "value": s} for s in sector_values(df)]

    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                className="geib-controls-row",
                children=[
                    control(
                        "Risk-Free Rate:",
                        dcc.Dropdown(id=risk_free_rate_id, options=risk_free_rate_options,
                                     value=risk_free_rate_default, searchable=False,
                                     style={"minWidth": "150px"}),
                    ),
                    *probability_controls(component_id, lo=prob_pos_threshold_default),
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
                className="geib-dual-graph",
                children=[
                    html.Div(
                        className="geib-graph-pane",
                        children=[
                            dcc.Loading(type="circle",
                                        children=[dcc.Graph(id=graph_id_1, style=GRAPH_STYLE)]),
                            html.Pre(id=error_id_1, className="geib-error"),
                        ],
                    ),
                    html.Div(
                        className="geib-graph-pane",
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
        empty = empty_figure("No data is available to display")
        return empty, empty

    # Gate on the selected probability metric *before* the projection below —
    # three of the four selectable metrics are not in that column list.
    df = apply_probability_filter(df, component_id, kwargs)

    df = df[[
        "name", "sector", "market_cap",
        "expected_return_kalman", "er_p05", "er_p95",
    ]].copy()
    logger.debug(schema(df))

    risk_free_rate = coalesce(kwargs.get(risk_free_rate_id), risk_free_rate_default)
    min_market_cap = coalesce(kwargs.get(min_market_cap_id), min_market_cap_default)
    top_stocks = kwargs.get(top_stocks_id, top_stocks_default)
    if top_stocks is None:
        top_stocks = top_stocks_default
    elif top_stocks == "all":
        top_stocks = None
    sector_filter = kwargs.get(sector_filter_id) or []

    df = _annualized_risk_metrics(df, risk_free_rate)

    df = df[df["market_cap"] >= min_market_cap]
    if len(sector_filter) > 0:
        df = df[df["sector"].isin(sector_filter)]

    df = df.sort_values("sharpe_ratio", ascending=False)
    if top_stocks is not None:
        df = df.head(top_stocks)

    if len(df) == 0:
        empty = empty_figure("No data matches the selected filters")
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


def _annualized_risk_metrics(df: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    """Attach horizon-correct annualized return, volatility, and Sharpe ratio.

    Delegates the unit/horizon conversion to :mod:`dashboards.geib.metrics` (the
    single source of truth shared with the other Kalman charts), then forms the
    Sharpe ratio and drops non-finite rows.

    Risk is the asset's *return* dispersion implied by the Monte-Carlo return
    distribution's 5th–95th percentiles (``quantile_volatility_pct``), not the
    posterior sd of expected upside (``expected_upside_sd``). The latter is
    estimation uncertainty about the mean and understates risk by ~1-2 orders of
    magnitude, inflating the Sharpe ratio accordingly. On run 317dbfff4bcf the
    two differ by 4.5x at the median: 4.17pp against 18.62pp.

    Parameters
    ----------
    df
        Frame carrying ``expected_return_kalman`` and the per-name return
        percentiles ``er_p05`` / ``er_p95``.
    risk_free_rate
        Annual risk-free rate as a fraction (e.g. ``0.03`` for 3%).

    Returns
    -------
    pandas.DataFrame
        ``df`` with ``annualized_return``, ``annualized_volatility`` and
        ``sharpe_ratio`` columns (all in percent), with non-finite rows dropped.
    """
    out = df.copy()
    out["annualized_return"] = annualized_return_pct(out["expected_return_kalman"])
    out["annualized_volatility"] = quantile_volatility_pct(
        out["er_p05"], out["er_p95"]
    )
    out["sharpe_ratio"] = (
        out["annualized_return"] - risk_free_rate * 100.0
    ) / out["annualized_volatility"]

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(
        subset=["annualized_return", "annualized_volatility", "sharpe_ratio"]
    )


@callback(
    output=[
        Output(f"{component_id}_graph_1", "figure"),
        Output(f"{component_id}_error_1", "children"),
        Output(f"{component_id}_graph_2", "figure"),
        Output(f"{component_id}_error_2", "children"),
        Output(sector_filter_id, "options"),
        Output(sector_filter_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        **probability_inputs(component_id),
        min_market_cap_id: Input(min_market_cap_id, "value"),
        top_stocks_id: Input(top_stocks_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, str, go.Figure, str, list, list]:
    # Scope the local Sector filter to the globally-filtered universe.
    df_all = filter_data(get_data(), **kwargs)
    sector_opts, sector_val = scoped_filter(
        df_all, "sector", kwargs.get(sector_filter_id), multi=True
    )
    kwargs[sector_filter_id] = sector_val
    try:
        fig1, fig2 = _update_logic(**kwargs)
        return fig1, "", fig2, "", sector_opts, sector_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        empty = empty_figure("An error occurred")
        return empty, msg, empty, msg, sector_opts, sector_val