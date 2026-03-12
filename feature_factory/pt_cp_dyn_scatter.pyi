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

component_id = "price_target_vs_current_scatter"

price_target_metric_id = f"{component_id}_price_target_metric"
price_target_metric_options = [
    {"label": "Price Target Median", "value": "price_target_median"},
    {"label": "Kalman Estimate", "value": "kalman_estimate"},
    {"label": "Price Target Prob Weighted", "value": "price_target_prob_weighted"}
]
price_target_metric_default = "price_target_median"

size_encoding_id = f"{component_id}_size_encoding"
size_encoding_options = [
    {"label": "Expected Upside %", "value": "expected_upside_pct"},
    {"label": "Market Cap", "value": "market_cap"},
    {"label": "Volume", "value": "volume_shrs"},
    {"label": "None", "value": "none"}
]
size_encoding_default = "expected_upside_pct"

color_encoding_id = f"{component_id}_color_encoding"
color_encoding_options = [
    {"label": "Sector", "value": "sector"},
    {"label": "Confidence Level", "value": "confidence_level"},
    {"label": "Beat Classification", "value": "beat_classification"},
    {"label": "None", "value": "none"}
]
color_encoding_default = "sector"

price_min_id = f"{component_id}_price_min"
price_max_id = f"{component_id}_price_max"

target_min_id = f"{component_id}_target_min"
target_max_id = f"{component_id}_target_max"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    title = "Price Target vs Current Price"
    description = "Scatter plot showing price target vs current price with upside potential. Points above the diagonal indicate upside potential; points below indicate downside risk."

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Price Target Metric:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=price_target_metric_id,
                                options=price_target_metric_options,
                                value=price_target_metric_default,
                                style={"minWidth": "200px"},
                                searchable=False
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
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Color Encoding:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=color_encoding_id,
                                options=color_encoding_options,
                                value=color_encoding_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Last Price Range:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            html.Div([
                                dcc.Input(
                                    id=price_min_id,
                                    type="number",
                                    placeholder="Min",
                                    debounce=True,
                                    style={"width": "100px"}
                                ),
                                html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                                dcc.Input(
                                    id=price_max_id,
                                    type="number",
                                    placeholder="Max",
                                    debounce=True,
                                    style={"width": "100px"}
                                )
                            ], style={
                                "display": "flex",
                                "alignItems": "center",
                                "flexWrap": "wrap"
                            })
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Price Target Range:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            html.Div([
                                dcc.Input(
                                    id=target_min_id,
                                    type="number",
                                    placeholder="Min",
                                    debounce=True,
                                    style={"width": "100px"}
                                ),
                                html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                                dcc.Input(
                                    id=target_max_id,
                                    type="number",
                                    placeholder="Max",
                                    debounce=True,
                                    style={"width": "100px"}
                                )
                            ], style={
                                "display": "flex",
                                "alignItems": "center",
                                "flexWrap": "wrap"
                            })
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
        price_target_metric_id: {
            "options": [option["value"] for option in price_target_metric_options],
            "default": price_target_metric_default
        },
        size_encoding_id: {
            "options": [option["value"] for option in size_encoding_options],
            "default": size_encoding_default
        },
        color_encoding_id: {
            "options": [option["value"] for option in color_encoding_options],
            "default": color_encoding_default
        },
        price_min_id: {
            "options": [0.1, 1, 10, 100],
            "default": ""
        },
        price_max_id: {
            "options": [100, 1000, 10000, 100000],
            "default": ""
        },
        target_min_id: {
            "options": [0.1, 1, 10, 100],
            "default": ""
        },
        target_max_id: {
            "options": [100, 1000, 10000, 100000],
            "default": ""
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
    df = df[['last_price', 'price_target_median', 'kalman_estimate', 'price_target_prob_weighted',
             'expected_upside_pct', 'market_cap', 'volume_shrs', 'sector', 'confidence_level',
             'beat_classification', 'ticker', 'name', 'industry', 'country', 'trading_country',
             'exchange']].copy()
    logger.debug(schema(df))
    logger.debug(tbl(df))

    price_target_metric = kwargs.get(price_target_metric_id, price_target_metric_default)
    if price_target_metric is None:
        price_target_metric = price_target_metric_default

    size_encoding = kwargs.get(size_encoding_id, size_encoding_default)
    if size_encoding is None:
        size_encoding = size_encoding_default

    color_encoding = kwargs.get(color_encoding_id, color_encoding_default)
    if color_encoding is None:
        color_encoding = color_encoding_default

    price_min = kwargs.get(price_min_id)
    price_max = kwargs.get(price_max_id)
    target_min = kwargs.get(target_min_id)
    target_max = kwargs.get(target_max_id)

    logger.debug("Filtering by price range...")
    if price_min is not None and price_min != "":
        price_min = float(price_min)
        df = df[df['last_price'] >= price_min]
    if price_max is not None and price_max != "":
        price_max = float(price_max)
        df = df[df['last_price'] <= price_max]
    logger.debug(tbl(df))

    logger.debug("Filtering by price target range...")
    if target_min is not None and target_min != "":
        target_min = float(target_min)
        df = df[df[price_target_metric] >= target_min]
    if target_max is not None and target_max != "":
        target_max = float(target_max)
        df = df[df[price_target_metric] <= target_max]
    logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available after filtering",
            annotations=[{
                "text": "No data matches the selected filters",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig

    logger.debug("Preparing size column...")
    size_col = None if size_encoding == "none" else size_encoding

    if size_col is not None:
        df['size_normalized'] = df[size_col].copy()
        min_val = df['size_normalized'].min()
        max_val = df['size_normalized'].max()

        if min_val < 0:
            df['size_normalized'] = df['size_normalized'] - min_val
            min_val = 0
            max_val = df['size_normalized'].max()

        if max_val > 0:
            df['size_normalized'] = (df['size_normalized'] / max_val) * 20 + 4
        else:
            df['size_normalized'] = 6

        size_col = 'size_normalized'

    color_col = None if color_encoding == "none" else color_encoding

    logger.debug("Creating scatter plot...")
    logger.debug(schema(df))
    logger.debug(tbl(df))

    hover_dict = {
        'ticker': True, 'name': True, 'sector': True, 'industry': True,
        'country': True, 'trading_country': True, 'exchange': True,
        'last_price': ':.2f', price_target_metric: ':.2f', 'expected_upside_pct': ':.2f'
    }

    if color_col is not None:
        fig = px.scatter(
            df,
            x='last_price',
            y=price_target_metric,
            size=size_col,
            color=color_col,
            hover_data=hover_dict
        )
        fig.update_layout(legend_title_text=color_col.replace('_', ' ').title())
    else:
        fig = px.scatter(
            df,
            x='last_price',
            y=price_target_metric,
            size=size_col,
            hover_data=hover_dict
        )

    max_val = max(df['last_price'].max(), df[price_target_metric].max()) * 1.05
    min_val = min(df['last_price'].min(), df[price_target_metric].min()) * 0.95

    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="gray", dash="dash", width=2),
        name="Fair Value (y=x)"
    )

    fig.update_xaxes(title_text="Last Price")
    fig.update_yaxes(title_text=price_target_metric.replace('_', ' ').title())

    fig.update_layout(
        hovermode='closest',
        minreducedwidth=400,
        minreducedheight=400
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
        price_target_metric_id: Input(price_target_metric_id, "value"),
        size_encoding_id: Input(size_encoding_id, "value"),
        color_encoding_id: Input(color_encoding_id, "value"),
        price_min_id: Input(price_min_id, "value"),
        price_max_id: Input(price_max_id, "value"),
        target_min_id: Input(target_min_id, "value"),
        target_max_id: Input(target_max_id, "value"),
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
        figure = _update_logic(**kwargs)
        return figure, ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg