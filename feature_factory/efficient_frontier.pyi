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

component_id = "efficient_frontier_optimization"

risk_free_rate_id = f"{component_id}_risk_free_rate"
risk_free_rate_options = [
    {"label": "0%", "value": 0.0},
    {"label": "2%", "value": 0.02},
    {"label": "3%", "value": 0.03},
    {"label": "4%", "value": 0.04},
    {"label": "5%", "value": 0.05}
]
risk_free_rate_default = 0.03

sector_filter_id = f"{component_id}_sector_filter"
sector_options = [
    {"label": "All Sectors", "value": "all"},
    {"label": "Industrials", "value": "Industrials"},
    {"label": "Information Technology", "value": "Information Technology"},
    {"label": "Materials", "value": "Materials"},
    {"label": "Energy", "value": "Energy"},
    {"label": "Health Care", "value": "Health Care"},
    {"label": "Consumer Discretionary", "value": "Consumer Discretionary"},
    {"label": "Financials", "value": "Financials"},
    {"label": "Communication Services", "value": "Communication Services"},
    {"label": "Utilities", "value": "Utilities"}
]
sector_filter_default = ["all"]

confidence_level_id = f"{component_id}_confidence_level"
confidence_level_options = [
    {"label": "All", "value": "all"},
    {"label": "High", "value": "High"},
    {"label": "Medium", "value": "Medium"},
    {"label": "Low", "value": "Low"}
]
confidence_level_default = "High"

num_portfolios_id = f"{component_id}_num_portfolios"
num_portfolios_options = [
    {"label": "100", "value": 100},
    {"label": "500", "value": 500},
    {"label": "1000", "value": 1000},
    {"label": "2000", "value": 2000}
]
num_portfolios_default = 500

highlight_portfolio_id = f"{component_id}_highlight_portfolio"
highlight_portfolio_options = [
    {"label": "Minimum Variance", "value": "min_variance"},
    {"label": "Maximum Sharpe", "value": "max_sharpe"},
    {"label": "All Efficient", "value": "all_efficient"}
]
highlight_portfolio_default = "max_sharpe"

color_by_id = f"{component_id}_color_by"
color_by_options = [
    {"label": "Sharpe Ratio", "value": "sharpe_ratio"},
    {"label": "None", "value": "none"}
]
color_by_default = "sharpe_ratio"

size_by_id = f"{component_id}_size_by"
size_by_options = [
    {"label": "Confidence Score", "value": "confidence_score"},
    {"label": "None", "value": "none"}
]
size_by_default = "confidence_score"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    table_id = f"{component_id}_table"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    title = "Efficient Frontier Optimization"
    description = "Find the optimal balance between risk and return across different stocks. The efficient frontier shows portfolios that maximize return for each level of risk, helping you identify the best investment combinations."

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
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Sectors:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=sector_filter_id,
                                options=sector_options,
                                value=sector_filter_default,
                                multi=True,
                                style={"minWidth": "200px"}
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
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Number of Portfolios:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=num_portfolios_id,
                                options=num_portfolios_options,
                                value=num_portfolios_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Highlight Portfolio:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=highlight_portfolio_id,
                                options=highlight_portfolio_options,
                                value=highlight_portfolio_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Color By:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=color_by_id,
                                options=color_by_options,
                                value=color_by_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Size By:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=size_by_id,
                                options=size_by_options,
                                value=size_by_default,
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
            html.Div(id=table_id, style={"margin": "20px 0"}),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        risk_free_rate_id: {
            "options": [opt["value"] for opt in risk_free_rate_options],
            "default": risk_free_rate_default
        },
        sector_filter_id: {
            "options": [opt["value"] for opt in sector_options],
            "default": sector_filter_default
        },
        confidence_level_id: {
            "options": [opt["value"] for opt in confidence_level_options],
            "default": confidence_level_default
        },
        num_portfolios_id: {
            "options": [opt["value"] for opt in num_portfolios_options],
            "default": num_portfolios_default
        },
        highlight_portfolio_id: {
            "options": [opt["value"] for opt in highlight_portfolio_options],
            "default": highlight_portfolio_default
        },
        color_by_id: {
            "options": [opt["value"] for opt in color_by_options],
            "default": color_by_default
        },
        size_by_id: {
            "options": [opt["value"] for opt in size_by_options],
            "default": size_by_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _compute_portfolio_metrics(weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> Tuple[float, float, float]:
    """Compute portfolio return, risk, and Sharpe ratio."""
    portfolio_return = np.sum(weights * returns)
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_risk = np.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
    return portfolio_return, portfolio_risk, sharpe_ratio

def _generate_random_portfolios(num_portfolios: int, num_assets: int, returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float, confidence_scores: np.ndarray) -> pd.DataFrame:
    """Generate random portfolio combinations."""
    results = []

    np.random.seed(42)
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return, portfolio_risk, sharpe_ratio = _compute_portfolio_metrics(weights, returns, cov_matrix, risk_free_rate)
        avg_confidence = np.average(confidence_scores, weights=weights)

        results.append({
            "return": portfolio_return,
            "risk": portfolio_risk,
            "sharpe_ratio": sharpe_ratio,
            "confidence_score": avg_confidence,
            "weights": weights
        })

    return pd.DataFrame(results)

def _find_efficient_frontier(portfolios_df: pd.DataFrame) -> pd.DataFrame:
    """Identify efficient frontier portfolios."""
    portfolios_df = portfolios_df.sort_values("risk").reset_index(drop=True)
    efficient = []
    max_return = -np.inf

    for idx, row in portfolios_df.iterrows():
        if row["return"] > max_return:
            efficient.append(idx)
            max_return = row["return"]

    return portfolios_df.iloc[efficient].reset_index(drop=True)

def _update_logic(**kwargs) -> Tuple[go.Figure, str, str]:
    """Core chart update logic without error handling."""
    logger.debug("Updating efficient frontier with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        logger.error("No data available after filtering")
        return go.Figure(), "", "<div>No data available</div>"

    logger.debug("Filtering by confidence level...")
    confidence_level = kwargs.get(confidence_level_id, confidence_level_default)
    if confidence_level != "all":
        df = df[df["confidence_level"] == confidence_level]
    logger.debug(tbl(df))

    logger.debug("Filtering by sectors...")
    sectors = kwargs.get(sector_filter_id, sector_filter_default)
    if sectors and "all" not in sectors:
        df = df[df["sector"].isin(sectors)]
    logger.debug(tbl(df))

    if len(df) < 2:
        logger.error("Not enough stocks after filtering for portfolio optimization")
        return go.Figure(), "", "<div>Not enough stocks after filtering (minimum 2 required)</div>"

    logger.debug("Computing correlation matrix and portfolio metrics...")
    returns = df["expected_return_prob_weighted"].values
    confidence_scores = df["confidence_score"].values
    tickers = df["ticker"].values
    names = df["name"].values

    num_assets = len(df)
    cov_matrix = np.eye(num_assets) * 0.01
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            correlation = np.random.uniform(-0.3, 0.8)
            cov_matrix[i, j] = correlation * np.sqrt(0.01 * 0.01)
            cov_matrix[j, i] = cov_matrix[i, j]

    risk_free_rate = kwargs.get(risk_free_rate_id, risk_free_rate_default)
    if risk_free_rate is None:
        risk_free_rate = risk_free_rate_default
    risk_free_rate = float(risk_free_rate)

    num_portfolios = kwargs.get(num_portfolios_id, num_portfolios_default)
    if num_portfolios is None:
        num_portfolios = num_portfolios_default
    num_portfolios = int(num_portfolios)

    logger.debug("Generating %d random portfolios...", num_portfolios)
    portfolios_df = _generate_random_portfolios(num_portfolios, num_assets, returns, cov_matrix, risk_free_rate, confidence_scores)
    logger.debug(schema(portfolios_df))
    logger.debug(tbl(portfolios_df))

    logger.debug("Finding efficient frontier...")
    efficient_df = _find_efficient_frontier(portfolios_df)
    logger.debug(tbl(efficient_df))

    min_variance_idx = portfolios_df["risk"].idxmin()
    min_variance_portfolio = portfolios_df.loc[min_variance_idx]

    max_sharpe_idx = portfolios_df["sharpe_ratio"].idxmax()
    max_sharpe_portfolio = portfolios_df.loc[max_sharpe_idx]

    logger.debug("Creating scatter plot...")
    color_by = kwargs.get(color_by_id, color_by_default)
    size_by = kwargs.get(size_by_id, size_by_default)
    highlight_portfolio = kwargs.get(highlight_portfolio_id, highlight_portfolio_default)

    portfolios_df["portfolio_type"] = "Regular"
    portfolios_df.loc[min_variance_idx, "portfolio_type"] = "Minimum Variance"
    portfolios_df.loc[max_sharpe_idx, "portfolio_type"] = "Maximum Sharpe"

    efficient_df["portfolio_type"] = "Efficient Frontier"

    plot_df = portfolios_df.copy()
    plot_df = plot_df.drop(columns=["weights"])
    if highlight_portfolio == "all_efficient":
        plot_df = pd.concat([portfolios_df, efficient_df], ignore_index=True)

    if color_by == "sharpe_ratio" and size_by == "confidence_score":
        fig = px.scatter(
            plot_df,
            x="risk",
            y="return",
            color="sharpe_ratio",
            size="confidence_score",
            hover_data={"risk": ":.4f", "return": ":.4f", "sharpe_ratio": ":.4f", "confidence_score": ":.4f"},
            color_continuous_scale="Viridis"
        )
        fig.update_traces(marker=dict(sizemin=6))
    elif color_by == "sharpe_ratio":
        fig = px.scatter(
            plot_df,
            x="risk",
            y="return",
            color="sharpe_ratio",
            hover_data={"risk": ":.4f", "return": ":.4f", "sharpe_ratio": ":.4f", "confidence_score": ":.4f"},
            color_continuous_scale="Viridis"
        )
    elif size_by == "confidence_score":
        fig = px.scatter(
            plot_df,
            x="risk",
            y="return",
            size="confidence_score",
            hover_data={"risk": ":.4f", "return": ":.4f", "sharpe_ratio": ":.4f", "confidence_score": ":.4f"}
        )
        fig.update_traces(marker=dict(sizemin=6))
    else:
        fig = px.scatter(
            plot_df,
            x="risk",
            y="return",
            hover_data={"risk": ":.4f", "return": ":.4f", "sharpe_ratio": ":.4f", "confidence_score": ":.4f"}
        )

    if highlight_portfolio == "min_variance":
        fig.add_scatter(
            x=[min_variance_portfolio["risk"]],
            y=[min_variance_portfolio["return"]],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="Minimum Variance",
            hovertemplate="Minimum Variance<br>Risk: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>"
        )
    elif highlight_portfolio == "max_sharpe":
        fig.add_scatter(
            x=[max_sharpe_portfolio["risk"]],
            y=[max_sharpe_portfolio["return"]],
            mode="markers",
            marker=dict(size=15, color="gold", symbol="star"),
            name="Maximum Sharpe",
            hovertemplate="Maximum Sharpe<br>Risk: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>"
        )

    if highlight_portfolio == "all_efficient":
        fig.add_scatter(
            x=efficient_df["risk"],
            y=efficient_df["return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="red", width=2, dash="dash"),
            hovertemplate="Efficient Frontier<br>Risk: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>"
        )

    fig.update_layout(
        xaxis_title="Portfolio Risk (Standard Deviation)",
        yaxis_title="Expected Return",
        hovermode="closest",
        height=550
    )

    logger.debug("Creating top 20 stocks table...")
    selected_weights = max_sharpe_portfolio["weights"]
    stock_data = []
    for i, ticker in enumerate(tickers):
        weight = selected_weights[i] * 100
        if weight > 0.01:
            risk_contribution = weight * np.sqrt(cov_matrix[i, i])
            sharpe = (returns[i] - risk_free_rate) / np.sqrt(cov_matrix[i, i]) if np.sqrt(cov_matrix[i, i]) > 0 else 0
            stock_data.append({
                "Ticker": ticker,
                "Name": names[i],
                "Weight (%)": f"{weight:.2f}",
                "Expected Return": f"{returns[i]:.2f}%",
                "Risk Contribution": f"{risk_contribution:.4f}",
                "Sharpe Ratio": f"{sharpe:.4f}"
            })

    stock_df = pd.DataFrame(stock_data).sort_values("Weight (%)", ascending=False).head(20)

    table_html = html.Div([
        html.H4("Top 20 Stocks in Maximum Sharpe Portfolio"),
        html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in stock_df.columns])
            ),
            html.Tbody([
                html.Tr([html.Td(stock_df.iloc[i][col]) for col in stock_df.columns])
                for i in range(len(stock_df))
            ])
        ], style={"width": "100%", "borderCollapse": "collapse", "border": "1px solid #ddd"})
    ])

    logger.debug("Done")
    return fig, "", table_html

@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(f"{component_id}_table", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        confidence_level_id: Input(confidence_level_id, "value"),
        num_portfolios_id: Input(num_portfolios_id, "value"),
        highlight_portfolio_id: Input(highlight_portfolio_id, "value"),
        color_by_id: Input(color_by_id, "value"),
        size_by_id: Input(size_by_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure, error_msg, table_html = _update_logic(**kwargs)
        return figure, error_msg, table_html

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg, ""