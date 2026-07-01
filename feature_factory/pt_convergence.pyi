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


component_id = "price_target_convergence"

metric_control_id = f"{component_id}_metric"
metric_options = [
    {"label": "All Three", "value": "all_three"},
    {"label": "Original vs Target", "value": "original_vs_target"},
    {"label": "Kalman Filtered", "value": "kalman_filtered"}
]
metric_default = "all_three"

sector_control_id = f"{component_id}_sector"
sector_default = "All"

top_n_control_id = f"{component_id}_top_n"
top_n_options = [
    {"label": "Top 10", "value": 10},
    {"label": "Top 20", "value": 20},
    {"label": "Top 50", "value": 50}
]
top_n_default = 20


def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    title = "Price Target Convergence Over Time"
    description = "Comparison of original price, original target, and Kalman-filtered price targets by ticker"

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
                            html.Label("Metric Comparison:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=metric_control_id,
                                options=metric_options,
                                value=metric_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Sector:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=sector_control_id,
                                options=[],
                                value=sector_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Top N Tickers:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=top_n_control_id,
                                options=top_n_options,
                                value=top_n_default,
                                style={"minWidth": "200px"},
                                searchable=False
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
        metric_control_id: {
            "options": [option["value"] for option in metric_options],
            "default": metric_default
        },
        sector_control_id: {
            "options": [sector_default],
            "default": sector_default
        },
        top_n_control_id: {
            "options": [option["value"] for option in top_n_options],
            "default": top_n_default
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
    df = df[['ticker', 'sector', 'market_cap', 'original_price', 'original_target', 'price_target_kalman']].copy()
    logger.debug(schema(df))
    logger.debug(tbl(df))

    metric_value = kwargs.get(metric_control_id, metric_default)
    if metric_value is None:
        metric_value = metric_default

    sector_value = kwargs.get(sector_control_id, sector_default)
    if sector_value is None:
        sector_value = sector_default

    top_n_value = kwargs.get(top_n_control_id, top_n_default)
    if top_n_value is None:
        top_n_value = top_n_default

    logger.debug("Filtering by sector: %s...", sector_value)
    if sector_value != "All":
        df = df[df['sector'] == sector_value]
    logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available for selected sector",
            annotations=[{
                "text": "No data is available for the selected sector",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    logger.debug("Getting top %d tickers by market cap...", top_n_value)
    top_tickers = df.nlargest(top_n_value, 'market_cap')['ticker'].unique()
    df = df[df['ticker'].isin(top_tickers)]
    logger.debug(tbl(df))

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

    logger.debug("Computing price target convergence metrics...")
    df['gap_percentage'] = ((df['price_target_kalman'] - df['original_price']) / df['original_price'] * 100).round(2)
    logger.debug(tbl(df))

    logger.debug("Creating line chart with metric: %s...", metric_value)
    logger.debug(schema(df))
    logger.debug(tbl(df))

    fig = go.Figure()

    if metric_value == "all_three":
        for ticker in sorted(df['ticker'].unique()):
            ticker_data = df[df['ticker'] == ticker].iloc[0]
            fig.add_trace(go.Scatter(
                x=[ticker],
                y=[ticker_data['original_price']],
                mode='markers',
                name='Original Price',
                marker=dict(size=8),
                legendgroup='original_price',
                showlegend=ticker == sorted(df['ticker'].unique())[0]
            ))
            fig.add_trace(go.Scatter(
                x=[ticker],
                y=[ticker_data['original_target']],
                mode='markers',
                name='Original Target',
                marker=dict(size=8),
                legendgroup='original_target',
                showlegend=ticker == sorted(df['ticker'].unique())[0]
            ))
            fig.add_trace(go.Scatter(
                x=[ticker],
                y=[ticker_data['price_target_kalman']],
                mode='markers',
                name='Kalman Filtered Target',
                marker=dict(size=8),
                legendgroup='kalman_filtered',
                showlegend=ticker == sorted(df['ticker'].unique())[0]
            ))

    elif metric_value == "original_vs_target":
        for ticker in sorted(df['ticker'].unique()):
            ticker_data = df[df['ticker'] == ticker].iloc[0]
            fig.add_trace(go.Scatter(
                x=[ticker],
                y=[ticker_data['original_price']],
                mode='markers',
                name='Original Price',
                marker=dict(size=8),
                legendgroup='original_price',
                showlegend=ticker == sorted(df['ticker'].unique())[0]
            ))
            fig.add_trace(go.Scatter(
                x=[ticker],
                y=[ticker_data['original_target']],
                mode='markers',
                name='Original Target',
                marker=dict(size=8),
                legendgroup='original_target',
                showlegend=ticker == sorted(df['ticker'].unique())[0]
            ))

    elif metric_value == "kalman_filtered":
        for ticker in sorted(df['ticker'].unique()):
            ticker_data = df[df['ticker'] == ticker].iloc[0]
            fig.add_trace(go.Scatter(
                x=[ticker],
                y=[ticker_data['price_target_kalman']],
                mode='markers',
                name='Kalman Filtered Target',
                marker=dict(size=8),
                legendgroup='kalman_filtered',
                showlegend=True
            ))

    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Price",
        hovermode='closest',
        height=550
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
        metric_control_id: Input(metric_control_id, "value"),
        sector_control_id: Input(sector_control_id, "value"),
        top_n_control_id: Input(top_n_control_id, "value"),
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
        df = get_data()

        logger.debug("Populating sector dropdown options...")
        sector_options = [{"label": "All", "value": "All"}]
        try:
            unique_sectors = sorted(df['sector'].dropna().unique().tolist())
            sector_options.extend([{"label": str(s), "value": s} for s in unique_sectors if s])
        except Exception as e:
            logger.error("Error getting sector options: %s", str(e))

        figure = _update_logic(**kwargs)
        return figure, ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg