"""Price Target Convergence Over Time.

Per-ticker scatter comparing original price, original target and the
Kalman-filtered price target. Ported from ``feature_factory/pt_convergence.pyi``.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ..data import get_data
from ..logger import logger, schema, tbl
from ..theme import GRAPH_STYLE, control
from ..theme import card as theme_card
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data

component_id = "price_target_convergence"

metric_control_id = f"{component_id}_metric"
metric_options = [
    {"label": "All Three", "value": "all_three"},
    {"label": "Original vs Target", "value": "original_vs_target"},
    {"label": "Kalman Filtered", "value": "kalman_filtered"},
]
metric_default = "all_three"

sector_control_id = f"{component_id}_sector"
sector_default = "All"

top_n_control_id = f"{component_id}_top_n"
top_n_options = [
    {"label": "Top 10", "value": 10},
    {"label": "Top 20", "value": 20},
    {"label": "Top 50", "value": 50},
]
top_n_default = 20

title = "Price Target Convergence Over Time"
description = (
    "Comparison of original price, original target, and Kalman-filtered price "
    "targets by ticker"
)


def component() -> "object":
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    df = get_data()
    sector_options = [{"label": "All", "value": "All"}]
    try:
        sectors = sorted(s for s in df["sector"].dropna().unique().tolist() if s)
        sector_options.extend({"label": str(s), "value": s} for s in sectors)
    except Exception:  # pragma: no cover - defensive
        pass

    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "rowGap": "10px"},
                children=[
                    control(
                        "Metric Comparison:",
                        dcc.Dropdown(
                            id=metric_control_id,
                            options=metric_options,
                            value=metric_default,
                            style={"minWidth": "200px"},
                        ),
                    ),
                    control(
                        "Sector:",
                        dcc.Dropdown(
                            id=sector_control_id,
                            options=sector_options,
                            value=sector_default,
                            style={"minWidth": "200px"},
                        ),
                    ),
                    control(
                        "Top N Tickers:",
                        dcc.Dropdown(
                            id=top_n_control_id,
                            options=top_n_options,
                            value=top_n_default,
                            searchable=False,
                            style={"minWidth": "200px"},
                        ),
                    ),
                ],
            ),
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[dcc.Graph(id=graph_id, style=GRAPH_STYLE)],
            ),
            html.Pre(id=error_id, className="geib-error"),
        ],
    )


def _update_logic(**kwargs) -> go.Figure:
    df = filter_data(get_data(), **kwargs)

    if df is None or len(df) == 0:
        return _empty("No data is available to display")

    df = df[
        ["ticker", "sector", "market_cap", "original_price", "original_target", "price_target_kalman"]
    ].copy()
    logger.debug(schema(df))

    metric_value = kwargs.get(metric_control_id) or metric_default
    sector_value = kwargs.get(sector_control_id) or sector_default
    top_n_value = kwargs.get(top_n_control_id) or top_n_default

    if sector_value != "All":
        df = df[df["sector"] == sector_value]
    if len(df) == 0:
        return _empty("No data is available for the selected sector")

    top_tickers = df.nlargest(top_n_value, "market_cap")["ticker"].unique()
    df = df[df["ticker"].isin(top_tickers)]
    if len(df) == 0:
        return _empty("No data is available to display")

    df["gap_percentage"] = (
        (df["price_target_kalman"] - df["original_price"]) / df["original_price"] * 100
    ).round(2)
    logger.debug(tbl(df))

    fig = go.Figure()
    tickers = sorted(df["ticker"].unique())
    first = tickers[0] if tickers else None

    def _add(series_key: str, name: str, group: str) -> None:
        for ticker in tickers:
            row = df[df["ticker"] == ticker].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[ticker],
                    y=[row[series_key]],
                    mode="markers",
                    name=name,
                    marker=dict(size=8),
                    legendgroup=group,
                    showlegend=(ticker == first),
                )
            )

    if metric_value in ("all_three", "original_vs_target"):
        _add("original_price", "Original Price", "original_price")
        _add("original_target", "Original Target", "original_target")
    if metric_value in ("all_three", "kalman_filtered"):
        _add("price_target_kalman", "Kalman Filtered Target", "kalman_filtered")

    fig.update_layout(
        xaxis_title="Ticker", yaxis_title="Price", hovermode="closest", height=550
    )
    return fig


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[{"text": message, "showarrow": False, "font": {"size": 18}}]
    )
    return fig


@callback(
    output=[Output(f"{component_id}_graph", "figure"), Output(f"{component_id}_error", "children")],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        metric_control_id: Input(metric_control_id, "value"),
        sector_control_id: Input(sector_control_id, "value"),
        top_n_control_id: Input(top_n_control_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, str]:
    try:
        return _update_logic(**kwargs), ""
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return _empty("An error occurred"), msg
