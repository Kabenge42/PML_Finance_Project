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


component_id = "capm_beta_return_calculator"

risk_free_rate_id = f"{component_id}_risk_free_rate"
risk_free_rate_default = 0.03
risk_free_rate_min = 0.0
risk_free_rate_max = 0.05
risk_free_rate_step = 0.005

market_return_id = f"{component_id}_market_return"
market_return_default = 0.10
market_return_min = 0.05
market_return_max = 0.20
market_return_step = 0.05

color_by_id = f"{component_id}_color_by"
color_by_options = [
    {"label": "Sector", "value": "sector"},
    {"label": "Region", "value": "region"},
    {"label": "Style Class", "value": "style_class"},
    {"label": "Size Class", "value": "size_class"},
    {"label": "Industry", "value": "industry"},
    {"label": "Trading Region", "value": "trading_region"},
    {"label": "None", "value": "none"}
]
color_by_default = "sector"

size_class_id = f"{component_id}_size_class"
size_class_options = [
    {"label": "All", "value": "All"},
    {"label": "Large Cap", "value": "Large Cap"},
    {"label": "Mid Cap", "value": "Mid Cap"},
    {"label": "Small Cap", "value": "Small Cap"}
]
size_class_default = "All"

sector_filter_id = f"{component_id}_sector_filter"

scatter_graph_id = f"{component_id}_scatter_graph"
scatter_error_id = f"{component_id}_scatter_error"
scatter_loading_id = f"{component_id}_scatter_loading"

bar_graph_id = f"{component_id}_bar_graph"
bar_error_id = f"{component_id}_bar_error"
bar_loading_id = f"{component_id}_bar_loading"


def component() -> ComponentResponse:
    title = "CAPM Beta & Return Calculator"
    description = "Calculate stock sensitivity to market movements (beta) and expected returns using CAPM. Identify over/undervalued stocks by comparing actual vs. predicted returns."

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
                            dcc.Slider(
                                id=risk_free_rate_id,
                                min=risk_free_rate_min,
                                max=risk_free_rate_max,
                                step=risk_free_rate_step,
                                value=risk_free_rate_default,
                                marks={i: f"{i * 100:.1f}%" for i in
                                       np.arange(risk_free_rate_min, risk_free_rate_max + risk_free_rate_step,
                                                 risk_free_rate_step * 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px", "minWidth": "200px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Market Return:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Slider(
                                id=market_return_id,
                                min=market_return_min,
                                max=market_return_max,
                                step=market_return_step,
                                value=market_return_default,
                                marks={i: f"{i * 100:.0f}%" for i in
                                       np.arange(market_return_min, market_return_max + market_return_step,
                                                 market_return_step)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px", "minWidth": "200px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Color By:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=color_by_id,
                                options=color_by_options,
                                value=color_by_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Size Class:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=size_class_id,
                                options=size_class_options,
                                value=size_class_default,
                                style={"minWidth": "150px"},
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
                            html.H4("Security Market Line (Beta vs Expected Return)", style={"marginBottom": "10px"}),
                            dcc.Loading(
                                id=scatter_loading_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=scatter_graph_id,
                                              style={"minHeight": "500px", "height": "calc(100vh - 700px)"}),
                                ]
                            ),
                            html.Pre(id=scatter_error_id, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.H4("Alpha Analysis (Actual vs CAPM Return)", style={"marginBottom": "10px"}),
                            dcc.Loading(
                                id=bar_loading_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=bar_graph_id,
                                              style={"minHeight": "500px", "height": "calc(100vh - 700px)"}),
                                ]
                            ),
                            html.Pre(id=bar_error_id, style={"color": "red", "margin": "10px 0"}),
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
            "options": [risk_free_rate_min, risk_free_rate_default, risk_free_rate_max],
            "default": risk_free_rate_default
        },
        market_return_id: {
            "options": [market_return_min, market_return_default, market_return_max],
            "default": market_return_default
        },
        color_by_id: {
            "options": [opt["value"] for opt in color_by_options],
            "default": color_by_default
        },
        size_class_id: {
            "options": [opt["value"] for opt in size_class_options],
            "default": size_class_default
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

    logger.debug("Filtering to stocks with non-null beta...")
    df = df[df["beta"].notna()].copy()
    logger.debug(tbl(df[["ticker", "beta", "expected_return_kalman"]]))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No stocks with beta data available", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    risk_free_rate = kwargs.get(risk_free_rate_id, risk_free_rate_default)
    if risk_free_rate is None:
        risk_free_rate = risk_free_rate_default

    market_return = kwargs.get(market_return_id, market_return_default)
    if market_return == "custom":
        market_return = 0.10
    if market_return is None:
        market_return = market_return_default

    logger.debug("Calculating CAPM expected return with R_f=%.2f%%, E(R_m)=%.2f%%...", risk_free_rate * 100,
                 market_return * 100)
    df["capm_expected_return"] = risk_free_rate + df["beta"] * (market_return - risk_free_rate)
    logger.debug(tbl(df[["ticker", "beta", "capm_expected_return", "expected_return_kalman"]]))

    color_by = kwargs.get(color_by_id, color_by_default)
    if color_by == "none":
        color_by = None

    size_class_filter = kwargs.get(size_class_id, size_class_default)
    if size_class_filter and size_class_filter != "All":
        logger.debug("Filtering by size class: %s...", size_class_filter)
        df = df[df["size_class"] == size_class_filter]
        logger.debug(tbl(df[["ticker", "size_class"]]))

    sector_filter = kwargs.get(sector_filter_id, [])
    if sector_filter and len(sector_filter) > 0:
        logger.debug("Filtering by sectors: %s...", sector_filter)
        df = df[df["sector"].isin(sector_filter)]
        logger.debug(tbl(df[["ticker", "sector"]]))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data matches the selected filters", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    logger.debug("Creating scatter chart...")
    logger.debug(schema(df))

    if color_by:
        fig = px.scatter(
            df,
            x="beta",
            y="expected_return_kalman",
            color=color_by,
            size="market_cap",
            hover_data={"ticker": True, "beta": ":.3f", "expected_return_kalman": ":.3f", "market_cap": ":.0f",
                        color_by: True},
            labels={
                "beta": "Beta (Market Sensitivity)",
                "expected_return_kalman": "Expected Return (Kalman)",
                color_by: color_by.replace("_", " ").title()
            }
        )
        fig.update_layout(legend_title_text=color_by.replace("_", " ").title())
    else:
        fig = px.scatter(
            df,
            x="beta",
            y="expected_return_kalman",
            size="market_cap",
            hover_data={"ticker": True, "beta": ":.3f", "expected_return_kalman": ":.3f", "market_cap": ":.0f"},
            labels={
                "beta": "Beta (Market Sensitivity)",
                "expected_return_kalman": "Expected Return (Kalman)"
            }
        )

    beta_range = [df["beta"].min() * 0.9, df["beta"].max() * 1.1]
    sml_x = np.array(beta_range)
    sml_y = risk_free_rate + sml_x * (market_return - risk_free_rate)

    fig.add_trace(go.Scatter(
        x=sml_x,
        y=sml_y,
        mode="lines",
        name="Security Market Line",
        line=dict(color="gray", dash="dash", width=2),
        hovertemplate="Beta: %{x:.3f}<br>CAPM Return: %{y:.3f}<extra></extra>"
    ))

    fig.update_xaxes(title_text="Beta (Market Sensitivity)")
    fig.update_yaxes(title_text="Expected Return")
    fig.update_layout(
        hovermode="closest",
        height=500
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

    logger.debug("Filtering to stocks with non-null beta...")
    df = df[df["beta"].notna()].copy()

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No stocks with beta data available", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    risk_free_rate = kwargs.get(risk_free_rate_id, risk_free_rate_default)
    if risk_free_rate is None:
        risk_free_rate = risk_free_rate_default

    market_return = kwargs.get(market_return_id, market_return_default)
    if market_return == "custom":
        market_return = 0.10
    if market_return is None:
        market_return = market_return_default

    logger.debug("Calculating CAPM expected return and alpha...")
    df["capm_expected_return"] = risk_free_rate + df["beta"] * (market_return - risk_free_rate)
    df["alpha"] = df["expected_return_kalman"] - df["capm_expected_return"]
    logger.debug(tbl(df[["ticker", "alpha", "expected_return_kalman", "capm_expected_return"]]))

    size_class_filter = kwargs.get(size_class_id, size_class_default)
    if size_class_filter and size_class_filter != "All":
        logger.debug("Filtering by size class: %s...", size_class_filter)
        df = df[df["size_class"] == size_class_filter]

    sector_filter = kwargs.get(sector_filter_id, [])
    if sector_filter and len(sector_filter) > 0:
        logger.debug("Filtering by sectors: %s...", sector_filter)
        df = df[df["sector"].isin(sector_filter)]

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data matches the selected filters", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    logger.debug("Sorting by alpha and selecting top/bottom stocks...")
    df_sorted = df.sort_values("alpha", ascending=False)
    top_n = 15
    top_positive = df_sorted.head(top_n)
    top_negative = df_sorted.tail(top_n)
    df_display = pd.concat([top_positive, top_negative]).sort_values("alpha", ascending=True)
    logger.debug(tbl(df_display[["ticker", "alpha"]]))

    logger.debug("Creating bar chart...")
    logger.debug(schema(df_display))

    df_display["alpha_color"] = df_display["alpha"].apply(lambda x: "Positive" if x > 0 else "Negative")

    fig = px.bar(
        df_display,
        x="alpha",
        y="ticker",
        color="alpha_color",
        orientation="h",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
        hover_data={"ticker": True, "alpha": ":.4f", "alpha_color": False},
        labels={
            "alpha": "Alpha (Actual - CAPM Return)",
            "ticker": "Ticker"
        }
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Zero Alpha",
                  annotation_position="top right")

    fig.update_xaxes(title_text="Alpha (Actual - CAPM Return)")
    fig.update_yaxes(title_text="Ticker")
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        height=500
    )

    logger.debug("Done")
    return fig


@callback(
    output=[
        Output(scatter_graph_id, "figure"),
        Output(scatter_error_id, "children"),
        Output(sector_filter_id, "options")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        market_return_id: Input(market_return_id, "value"),
        color_by_id: Input(color_by_id, "value"),
        size_class_id: Input(size_class_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update_scatter(**kwargs) -> Tuple[go.Figure, str, list]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        df_all = get_data()
        unique_sectors = sorted(df_all["sector"].dropna().unique().tolist())
        sector_options = [{"label": sector, "value": sector} for sector in unique_sectors]

        figure = _update_scatter_logic(**kwargs)
        return figure, "", sector_options

    except Exception as e:
        error_msg = f"Error updating scatter chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        df_all = get_data()
        unique_sectors = sorted(df_all["sector"].dropna().unique().tolist())
        sector_options = [{"label": sector, "value": sector} for sector in unique_sectors]
        return empty_fig, error_msg, sector_options


@callback(
    output=[
        Output(bar_graph_id, "figure"),
        Output(bar_error_id, "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        market_return_id: Input(market_return_id, "value"),
        size_class_id: Input(size_class_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
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
        error_msg = f"Error updating bar chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg