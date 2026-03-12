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

component_id = "monte_carlo_portfolio_simulator"

num_simulations_id = f"{component_id}_num_simulations"
num_simulations_options = [
    {"label": "1,000", "value": 1000},
    {"label": "5,000", "value": 5000},
    {"label": "10,000", "value": 10000},
    {"label": "50,000", "value": 50000}
]
num_simulations_default = 10000

loss_ratio_id = f"{component_id}_loss_ratio"
loss_ratio_options = [
    {"label": "0.25 (25%)", "value": 0.25},
    {"label": "0.5 (50%)", "value": 0.5},
    {"label": "0.75 (75%)", "value": 0.75},
    {"label": "1.0 (100%)", "value": 1.0}
]
loss_ratio_default = 0.5

weighting_id = f"{component_id}_weighting"
weighting_options = [
    {"label": "Equal-weighted", "value": "equal"},
    {"label": "Kelly-weighted", "value": "kelly"},
    {"label": "Market cap proxy", "value": "market_cap"}
]
weighting_default = "equal"

target_return_id = f"{component_id}_target_return"
target_return_options = [
    {"label": "0%", "value": 0.0},
    {"label": "5%", "value": 5.0},
    {"label": "10%", "value": 10.0},
    {"label": "15%", "value": 15.0},
    {"label": "20%", "value": 20.0}
]
target_return_default = 10.0

signal_filter_id = f"{component_id}_signal_filter"
signal_options = [
    {"label": "Strong Bullish (4/4)", "value": "Strong Bullish (4/4)"},
    {"label": "Bullish (3/4)", "value": "Bullish (3/4)"},
    {"label": "Neutral (2/4)", "value": "Neutral (2/4)"},
    {"label": "Bearish (1/4)", "value": "Bearish (1/4)"},
    {"label": "Strong Bearish (0/4)", "value": "Strong Bearish (0/4)"}
]
signal_default = ["Strong Bullish (4/4)", "Bullish (3/4)"]

def component() -> ComponentResponse:
    graph_1_id = f"{component_id}_graph_1"
    error_1_id = f"{component_id}_error_1"
    loading_1_id = f"{component_id}_loading_1"

    graph_2_id = f"{component_id}_graph_2"
    error_2_id = f"{component_id}_error_2"
    loading_2_id = f"{component_id}_loading_2"

    stats_id = f"{component_id}_stats"

    title = "Monte Carlo Portfolio Outcome Simulator"
    description = "Simulate thousands of possible portfolio outcomes based on expected returns and probabilities. See the range of potential results and the likelihood of achieving your target return."

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Simulations:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=num_simulations_id,
                                options=num_simulations_options,
                                value=num_simulations_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Loss Ratio:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=loss_ratio_id,
                                options=loss_ratio_options,
                                value=loss_ratio_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Weighting:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=weighting_id,
                                options=weighting_options,
                                value=weighting_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Target Return:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=target_return_id,
                                options=target_return_options,
                                value=target_return_default,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Signal Filter:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=signal_filter_id,
                                options=signal_options,
                                value=signal_default,
                                multi=True,
                                style={"minWidth": "250px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
            ),
            html.Div(
                id=stats_id,
                style={"backgroundColor": "#f5f5f5", "padding": "15px", "marginBottom": "15px", "borderRadius": "4px", "fontSize": "14px"}
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "gap": "15px", "marginBottom": "15px"},
                children=[
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.H4("Percentile Distribution", style={"marginTop": "0", "marginBottom": "10px"}),
                            dcc.Loading(
                                id=loading_1_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_1_id, style={"minHeight": "450px"}),
                                ]
                            ),
                            html.Pre(id=error_1_id, style={"color": "red", "margin": "10px 0", "fontSize": "12px"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.H4("Return Distribution", style={"marginTop": "0", "marginBottom": "10px"}),
                            dcc.Loading(
                                id=loading_2_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_2_id, style={"minHeight": "450px"}),
                                ]
                            ),
                            html.Pre(id=error_2_id, style={"color": "red", "margin": "10px 0", "fontSize": "12px"}),
                        ]
                    ),
                ]
            ),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        num_simulations_id: {
            "options": [opt["value"] for opt in num_simulations_options],
            "default": num_simulations_default
        },
        loss_ratio_id: {
            "options": [opt["value"] for opt in loss_ratio_options],
            "default": loss_ratio_default
        },
        weighting_id: {
            "options": [opt["value"] for opt in weighting_options],
            "default": weighting_default
        },
        target_return_id: {
            "options": [opt["value"] for opt in target_return_options],
            "default": target_return_default
        },
        signal_filter_id: {
            "options": [opt["value"] for opt in signal_options],
            "default": signal_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _run_monte_carlo_simulation(df: pd.DataFrame, num_simulations: int, loss_ratio: float, weighting: str, target_return: float) -> Tuple[np.ndarray, dict]:
    """Run Monte Carlo simulation and return results."""
    logger.debug("Running Monte Carlo simulation with %d simulations...", num_simulations)

    if len(df) == 0:
        logger.error("No data available for simulation")
        return np.array([]), {}

    df = df.copy()

    df['prob_positive_upside'] = pd.to_numeric(df['prob_positive_upside'], errors='coerce')
    df['filtered_upside'] = pd.to_numeric(df['filtered_upside'], errors='coerce')
    df['achievement_probability'] = pd.to_numeric(df['achievement_probability'], errors='coerce')

    df = df.dropna(subset=['prob_positive_upside', 'filtered_upside', 'achievement_probability'])

    if len(df) == 0:
        logger.error("No valid data after cleaning")
        return np.array([]), {}

    num_stocks = len(df)
    logger.debug("Simulating portfolio with %d stocks", num_stocks)

    if weighting == "equal":
        weights = np.ones(num_stocks) / num_stocks
    elif weighting == "kelly":
        kelly_fractions = []
        for _, row in df.iterrows():
            p = row['prob_positive_upside'] / 100.0
            b = row['filtered_upside'] / 100.0
            if b > 0 and p > 0 and p < 1:
                kelly = (p * b - (1 - p) * loss_ratio * b) / (b * b) if b != 0 else 0
                kelly = max(0, min(kelly, 0.25))
            else:
                kelly = 0
            kelly_fractions.append(kelly)
        kelly_fractions = np.array(kelly_fractions)
        total = kelly_fractions.sum()
        if total > 0:
            weights = kelly_fractions / total
        else:
            weights = np.ones(num_stocks) / num_stocks
    elif weighting == "market_cap":
        weights = np.ones(num_stocks) / num_stocks
    else:
        weights = np.ones(num_stocks) / num_stocks

    prob_wins = (df['prob_positive_upside'].values / 100.0) * df['achievement_probability'].values
    prob_wins = np.clip(prob_wins, 0, 1.0)
    upside_returns = df['filtered_upside'].values / 100.0

    portfolio_returns = np.zeros(num_simulations)

    np.random.seed(42)
    for sim in range(num_simulations):
        outcomes = np.random.random(num_stocks) < prob_wins
        stock_returns = np.where(outcomes, upside_returns, -upside_returns * loss_ratio)
        portfolio_returns[sim] = np.dot(weights, stock_returns) * 100

    percentiles = np.percentile(portfolio_returns, [5, 25, 50, 75, 95])
    var_5 = percentiles[0]
    below_var = portfolio_returns[portfolio_returns <= var_5]
    cvar_5 = below_var.mean() if len(below_var) > 0 else var_5
    prob_positive = (portfolio_returns > 0).sum() / num_simulations * 100
    prob_target = (portfolio_returns > target_return).sum() / num_simulations * 100

    stats = {
        "var_5": var_5,
        "cvar_5": cvar_5,
        "median": percentiles[2],
        "prob_positive": prob_positive,
        "prob_target": prob_target,
        "p5": percentiles[0],
        "p25": percentiles[1],
        "p50": percentiles[2],
        "p75": percentiles[3],
        "p95": percentiles[4]
    }

    logger.debug("Simulation complete. Median return: %.2f%%, VaR(5%%): %.2f%%, Prob(>0%%): %.1f%%, Prob(>%.0f%%): %.1f%%",
                 stats["median"], stats["var_5"], stats["prob_positive"], target_return, stats["prob_target"])

    return portfolio_returns, stats

def _create_percentile_chart(portfolio_returns: np.ndarray) -> go.Figure:
    """Create percentile distribution line chart."""
    if len(portfolio_returns) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")
        return empty_fig

    percentiles = np.arange(0, 101, 1)
    percentile_values = np.percentile(portfolio_returns, percentiles)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=percentile_values,
        mode="lines",
        name="Portfolio Return",
        line=dict(width=2)
    ))

    fig.update_layout(
        xaxis_title="Percentile",
        yaxis_title="Simulated Portfolio Return (%)",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=20, b=60)
    )

    return fig

def _create_distribution_chart(portfolio_returns: np.ndarray) -> go.Figure:
    """Create return distribution histogram."""
    if len(portfolio_returns) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")
        return empty_fig

    min_return = portfolio_returns.min()
    max_return = portfolio_returns.max()

    bucket_width = 10
    bucket_edges = np.arange(np.floor(min_return / bucket_width) * bucket_width,
                             np.ceil(max_return / bucket_width) * bucket_width + bucket_width,
                             bucket_width)

    counts, _ = np.histogram(portfolio_returns, bins=bucket_edges)

    bucket_labels = []
    for i in range(len(bucket_edges) - 1):
        label = f"{bucket_edges[i]:.0f}% to {bucket_edges[i+1]:.0f}%"
        bucket_labels.append(label)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bucket_labels,
        y=counts,
        name="Frequency",
        marker_line_width=0
    ))

    fig.update_layout(
        xaxis_title="Return Bucket",
        yaxis_title="Frequency (Number of Simulations)",
        hovermode="x",
        margin=dict(l=60, r=20, t=20, b=80),
        xaxis=dict(tickangle=-45)
    )

    return fig

def _update_logic(**kwargs) -> Tuple[go.Figure, str, go.Figure, str, str]:
    """Core chart update logic without error handling."""
    logger.debug("Updating Monte Carlo simulator with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        logger.warning("No data available after filtering")
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")
        return empty_fig, "", empty_fig, "", "No data available"

    num_simulations = kwargs.get(num_simulations_id, num_simulations_default)
    if num_simulations is None:
        num_simulations = num_simulations_default
    num_simulations = int(num_simulations)

    loss_ratio = kwargs.get(loss_ratio_id, loss_ratio_default)
    if loss_ratio is None:
        loss_ratio = loss_ratio_default
    loss_ratio = float(loss_ratio)

    weighting = kwargs.get(weighting_id, weighting_default)
    if weighting is None:
        weighting = weighting_default

    target_return = kwargs.get(target_return_id, target_return_default)
    if target_return is None:
        target_return = target_return_default
    target_return = float(target_return)

    signal_filter = kwargs.get(signal_filter_id, signal_default)
    if signal_filter is None or len(signal_filter) == 0:
        signal_filter = signal_default

    logger.debug("Filtering by signal: %s...", signal_filter)
    df = df[df['signal'].isin(signal_filter)]
    logger.debug(tbl(df))

    if len(df) == 0:
        logger.warning("No data available after signal filtering")
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")
        return empty_fig, "", empty_fig, "", "No data available after filtering"

    logger.debug("Running simulation with %d stocks", len(df))
    portfolio_returns, stats = _run_monte_carlo_simulation(df, num_simulations, loss_ratio, weighting, target_return)

    if len(portfolio_returns) == 0:
        logger.error("Simulation failed")
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Simulation failed")
        return empty_fig, "", empty_fig, "", "Simulation failed"

    logger.debug("Creating percentile chart...")
    fig_1 = _create_percentile_chart(portfolio_returns)

    logger.debug("Creating distribution chart...")
    fig_2 = _create_distribution_chart(portfolio_returns)

    stats_html = f"""
    <b>Simulation Results ({num_simulations:,} runs, {len(df)} stocks)</b><br>
    <b>Value at Risk (5th percentile):</b> {stats['var_5']:.2f}%<br>
    <b>Conditional VaR (avg below 5th):</b> {stats['cvar_5']:.2f}%<br>
    <b>Median Return:</b> {stats['median']:.2f}%<br>
    <b>Probability of Positive Return:</b> {stats['prob_positive']:.1f}%<br>
    <b>Probability of Beating {target_return:.0f}% Target:</b> {stats['prob_target']:.1f}%<br>
    <b>Percentiles:</b> 5th: {stats['p5']:.2f}% | 25th: {stats['p25']:.2f}% | 50th: {stats['p50']:.2f}% | 75th: {stats['p75']:.2f}% | 95th: {stats['p95']:.2f}%
    """

    logger.debug("Done")
    return fig_1, "", fig_2, "", stats_html

@callback(
    output=[
        Output(f"{component_id}_graph_1", "figure"),
        Output(f"{component_id}_error_1", "children"),
        Output(f"{component_id}_graph_2", "figure"),
        Output(f"{component_id}_error_2", "children"),
        Output(f"{component_id}_stats", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        num_simulations_id: Input(num_simulations_id, "value"),
        loss_ratio_id: Input(loss_ratio_id, "value"),
        weighting_id: Input(weighting_id, "value"),
        target_return_id: Input(target_return_id, "value"),
        signal_filter_id: Input(signal_filter_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, go.Figure, str, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        fig_1, err_1, fig_2, err_2, stats_html = _update_logic(**kwargs)
        return fig_1, err_1, fig_2, err_2, stats_html

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, error_msg, empty_fig, error_msg, f"<span style='color: red;'>{error_msg}</span>"