import traceback
from typing import TypedDict, Any, Tuple

import dash_design_kit as ddk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.filter_component import filter_data, FILTER_CALLBACK_INPUTS
from dash import callback, html, dcc, Output, Input
from logger import logger, schema, tbl

from data import get_data


class TestInput(TypedDict):
    options: list[Any]
    default: Any


class ComponentResponse(TypedDict):
    layout: ddk.Card
    test_inputs: dict[str, TestInput]


component_id = "beta_capm_return_calculator"

risk_free_rate_id = f"{component_id}_risk_free_rate"
risk_free_rate_options = [
    {"label": "0%", "value": 0.0},
    {"label": "2%", "value": 0.02},
    {"label": "3%", "value": 0.03},
    {"label": "4%", "value": 0.04},
    {"label": "5%", "value": 0.05}
]
risk_free_rate_default = 0.03

market_proxy_id = f"{component_id}_market_proxy"
market_proxy_options = [
    {"label": "All Assets", "value": "all"},
    {"label": "Sector Average", "value": "sector"},
    {"label": "Region Average", "value": "region"}
]
market_proxy_default = "all"

sector_filter_id = f"{component_id}_sector_filter"

alpha_filter_id = f"{component_id}_alpha_filter"
alpha_filter_options = [
    {"label": "All", "value": "all"},
    {"label": "Positive Only", "value": "positive"},
    {"label": "Negative Only", "value": "negative"}
]
alpha_filter_default = "all"

top_n_id = f"{component_id}_top_n"
top_n_options = [
    {"label": "Top 10", "value": 10},
    {"label": "Top 25", "value": 25},
    {"label": "Top 50", "value": 50},
    {"label": "All", "value": 0}
]
top_n_default = 25

size_by_id = f"{component_id}_size_by"
size_by_options = [
    {"label": "None", "value": "none"},
    {"label": "Market Cap", "value": "market_cap"}
]
size_by_default = "none"


def component() -> ComponentResponse:
    graph1_id = f"{component_id}_scatter_graph"
    graph1_error_id = f"{component_id}_scatter_error"
    graph1_loading_id = f"{component_id}_scatter_loading"

    graph2_id = f"{component_id}_bar_graph"
    graph2_error_id = f"{component_id}_bar_error"
    graph2_loading_id = f"{component_id}_bar_loading"

    title = "Beta & CAPM Expected Return Calculator"
    description = "Calculate systematic risk (beta) and expected returns using the Capital Asset Pricing Model. Beta measures how much an asset moves relative to the market, while CAPM provides the theoretical expected return."

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
                            html.Label("Market Proxy:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=market_proxy_id,
                                options=market_proxy_options,
                                value=market_proxy_default,
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
                                multi=True,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Alpha Filter:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=alpha_filter_id,
                                options=alpha_filter_options,
                                value=alpha_filter_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Top N (Alpha):",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=top_n_id,
                                options=top_n_options,
                                value=top_n_default,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Bubble Size:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=size_by_id,
                                options=size_by_options,
                                value=size_by_default,
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
                            html.H4("Security Market Line (SML)", style={"marginTop": "0", "marginBottom": "10px"}),
                            dcc.Loading(
                                id=graph1_loading_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph1_id,
                                              style={"minHeight": "500px", "height": "calc(100vh - 700px)"}),
                                ]
                            ),
                            html.Pre(id=graph1_error_id,
                                     style={"color": "red", "margin": "10px 0", "fontSize": "12px"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.H4("Alpha Analysis (Top N)", style={"marginTop": "0", "marginBottom": "10px"}),
                            dcc.Loading(
                                id=graph2_loading_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph2_id,
                                              style={"minHeight": "500px", "height": "calc(100vh - 700px)"}),
                                ]
                            ),
                            html.Pre(id=graph2_error_id,
                                     style={"color": "red", "margin": "10px 0", "fontSize": "12px"}),
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
        market_proxy_id: {
            "options": [opt["value"] for opt in market_proxy_options],
            "default": market_proxy_default
        },
        sector_filter_id: {
            "options": [],
            "default": []
        },
        alpha_filter_id: {
            "options": [opt["value"] for opt in alpha_filter_options],
            "default": alpha_filter_default
        },
        top_n_id: {
            "options": [opt["value"] for opt in top_n_options],
            "default": top_n_default
        },
        size_by_id: {
            "options": [opt["value"] for opt in size_by_options],
            "default": size_by_default
        },
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }


def _calculate_beta_and_capm(**kwargs) -> Tuple[pd.DataFrame, float, float]:
    """Calculate beta, CAPM expected returns, and alpha."""
    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        return pd.DataFrame(), 0.0, 0.0

    logger.debug("Calculating beta and CAPM metrics...")

    risk_free_rate = kwargs.get(risk_free_rate_id, risk_free_rate_default)
    if risk_free_rate is None:
        risk_free_rate = risk_free_rate_default
    risk_free_rate = float(risk_free_rate)

    market_proxy = kwargs.get(market_proxy_id, market_proxy_default)
    if market_proxy is None:
        market_proxy = market_proxy_default

    logger.debug("Risk-free rate: %.2f%%, Market proxy: %s", risk_free_rate * 100, market_proxy)

    mean_signal_strength = df["signal_strength"].mean()
    df["beta"] = df["signal_strength"] / mean_signal_strength

    if market_proxy == "all":
        market_return = df["expected_return_kalman"].mean()
        logger.debug("Market return (all assets): %.4f", market_return)
    elif market_proxy == "sector":
        sector_returns = df.groupby("sector")["expected_return_kalman"].mean()
        df = df.merge(sector_returns.rename("sector_market_return"), left_on="sector", right_index=True)
        market_return = df["sector_market_return"].mean()
        logger.debug("Market return (sector average): %.4f", market_return)
    elif market_proxy == "region":
        region_returns = df.groupby("region")["expected_return_kalman"].mean()
        df = df.merge(region_returns.rename("region_market_return"), left_on="region", right_index=True)
        market_return = df["region_market_return"].mean()
        logger.debug("Market return (region average): %.4f", market_return)
    else:
        market_return = df["expected_return_kalman"].mean()

    market_risk_premium = market_return - risk_free_rate

    df["capm_expected_return"] = risk_free_rate + df["beta"] * market_risk_premium
    df["alpha"] = df["expected_return_kalman"] - df["capm_expected_return"]

    logger.debug("Computed metrics:\n  • Mean beta: %.4f\n  • Market risk premium: %.4f\n  • Mean alpha: %.4f",
                 df["beta"].mean(), market_risk_premium, df["alpha"].mean())

    return df, market_return, risk_free_rate


def _update_scatter_chart(**kwargs) -> go.Figure:
    """Create scatter plot for Security Market Line."""
    df, market_return, risk_free_rate = _calculate_beta_and_capm(**kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data is available to display", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    sector_filter = kwargs.get(sector_filter_id, [])
    if sector_filter and len(sector_filter) > 0:
        logger.debug("Filtering by sectors: %s", sector_filter)
        df = df[df["sector"].isin(sector_filter)]
        logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data matches the selected filters", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    size_by = kwargs.get(size_by_id, size_by_default)
    if size_by is None:
        size_by = size_by_default

    logger.debug("Creating scatter plot with size_by: %s", size_by)
    logger.debug(schema(df))
    logger.debug(tbl(df))

    if size_by == "market_cap":
        fig = px.scatter(
            df,
            x="beta",
            y="expected_return_kalman",
            color="sector",
            size="market_cap",
            hover_data={"name": True, "beta": ":.4f", "expected_return_kalman": ":.4f", "market_cap": ":.2f",
                        "sector": True},
            labels={"beta": "Beta", "expected_return_kalman": "Expected Return", "sector": "Sector"}
        )
        fig.update_traces(marker=dict(sizemin=6))
    else:
        fig = px.scatter(
            df,
            x="beta",
            y="expected_return_kalman",
            color="sector",
            hover_data={"name": True, "beta": ":.4f", "expected_return_kalman": ":.4f", "sector": True},
            labels={"beta": "Beta", "expected_return_kalman": "Expected Return", "sector": "Sector"}
        )

    beta_range = [df["beta"].min() * 0.9, df["beta"].max() * 1.1]
    sml_beta = np.linspace(beta_range[0], beta_range[1], 100)
    sml_return = risk_free_rate + sml_beta * (market_return - risk_free_rate)

    fig.add_trace(go.Scatter(
        x=sml_beta,
        y=sml_return,
        mode="lines",
        name="Security Market Line",
        line=dict(color="gray", dash="dash", width=2),
        hovertemplate="Beta: %{x:.4f}<br>SML Return: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Beta (Systematic Risk)",
        yaxis_title="Expected Return",
        legend_title_text="Sector",
        hovermode="closest"
    )

    logger.debug("Done")
    return fig


def _update_bar_chart(**kwargs) -> go.Figure:
    """Create bar plot for alpha analysis."""
    df, _, _ = _calculate_beta_and_capm(**kwargs)

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data is available to display", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    sector_filter = kwargs.get(sector_filter_id, [])
    if sector_filter and len(sector_filter) > 0:
        logger.debug("Filtering by sectors: %s", sector_filter)
        df = df[df["sector"].isin(sector_filter)]
        logger.debug(tbl(df))

    alpha_filter = kwargs.get(alpha_filter_id, alpha_filter_default)
    if alpha_filter is None:
        alpha_filter = alpha_filter_default

    logger.debug("Filtering by alpha: %s", alpha_filter)
    if alpha_filter == "positive":
        df = df[df["alpha"] > 0]
    elif alpha_filter == "negative":
        df = df[df["alpha"] < 0]

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data matches the selected filters", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig

    logger.debug(tbl(df))

    top_n = kwargs.get(top_n_id, top_n_default)
    if top_n is None:
        top_n = top_n_default
    top_n = int(top_n)

    logger.debug("Selecting top %s by absolute alpha...", top_n if top_n > 0 else "all")
    df["abs_alpha"] = df["alpha"].abs()
    if top_n > 0:
        df = df.nlargest(top_n, "abs_alpha")
    df = df.sort_values("alpha", ascending=True)

    logger.debug("Creating bar chart with final data...")
    logger.debug(schema(df))
    logger.debug(tbl(df))

    df["alpha_color"] = df["alpha"].apply(lambda x: "Positive" if x > 0 else "Negative")

    fig = px.bar(
        df,
        x="alpha",
        y="name",
        color="alpha_color",
        orientation="h",
        hover_data={"name": True, "alpha": ":.4f", "alpha_color": False},
        labels={"alpha": "Alpha (Excess Return)", "name": "Company", "alpha_color": "Alpha Type"},
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"}
    )

    fig.update_layout(
        xaxis_title="Alpha (Excess Return)",
        yaxis_title="Company",
        legend_title_text="Alpha Type",
        hovermode="closest",
        height=max(400, len(df) * 20)
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
        market_proxy_id: Input(market_proxy_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        alpha_filter_id: Input(alpha_filter_id, "value"),
        top_n_id: Input(top_n_id, "value"),
        size_by_id: Input(size_by_id, "value"),
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
        figure = _update_scatter_chart(**kwargs)
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
        market_proxy_id: Input(market_proxy_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        alpha_filter_id: Input(alpha_filter_id, "value"),
        top_n_id: Input(top_n_id, "value"),
        size_by_id: Input(size_by_id, "value"),
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
        figure = _update_bar_chart(**kwargs)
        return figure, ""
    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg


@callback(
    output=Output(sector_filter_id, "options"),
    inputs={'refresh_trigger': Input("refresh_trigger", "data"), **FILTER_CALLBACK_INPUTS}
)
def update_sector_options(**kwargs) -> list[dict[str, str]]:
    try:
        df = filter_data(get_data(), **kwargs)
        if len(df) == 0:
            return []

        unique_sectors = df["sector"].dropna().unique().tolist()
        unique_sectors.sort()
        options = [{"label": str(s), "value": s} for s in unique_sectors]
        return options
    except Exception as e:
        logger.error(f"Error updating sector options: {str(e)}")
        return []