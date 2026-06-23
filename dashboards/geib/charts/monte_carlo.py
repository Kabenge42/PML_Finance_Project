"""Monte Carlo Return Distribution Simulation.

Probability-density heatmap + CDF lines from per-stock normal-draw simulations.
Ported from ``feature_factory/monte_carlo_sim.pyi``; the simulation is
vectorised and seeded (``default_rng(42)``) for reproducibility and speed.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema
from ..metrics import return_volatility
from ..theme import DUAL_GRAPH_STYLE, control
from ..theme import card as theme_card

component_id = "monte_carlo_return_distribution"

time_horizon_id = f"{component_id}_time_horizon"
time_horizon_options = [
    {"label": "1 Month", "value": 30},
    {"label": "3 Months", "value": 90},
    {"label": "6 Months", "value": 180},
    {"label": "1 Year", "value": 252},
]
time_horizon_default = 90

num_simulations_id = f"{component_id}_num_simulations"
num_simulations_options = [
    {"label": "1,000", "value": 1000},
    {"label": "5,000", "value": 5000},
    {"label": "10,000", "value": 10000},
    {"label": "50,000", "value": 50000},
]
num_simulations_default = 10000

stock_selection_id = f"{component_id}_stock_selection"
stock_selection_options = [
    {"label": "Top 10 by Market Cap", "value": "top10_mc"},
    {"label": "Top 20 by Market Cap", "value": "top20_mc"},
    {"label": "Top 10 by Expected Return", "value": "top10_er"},
    {"label": "Custom Selection", "value": "custom"},
]
stock_selection_default = "top10_mc"

confidence_interval_id = f"{component_id}_confidence_interval"
confidence_interval_options = [
    {"label": "90%", "value": 0.90},
    {"label": "95%", "value": 0.95},
    {"label": "99%", "value": 0.99},
]
confidence_interval_default = 0.95

title = "Monte Carlo Return Distribution Simulation"
description = (
    "Simulates thousands of possible future return scenarios using Monte Carlo "
    "methods based on each stock's expected return and volatility. View the "
    "probability distribution of outcomes to understand the range of potential "
    "returns and downside risks."
)


def component() -> "object":
    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                className="geib-controls-row",
                children=[
                    control("Time Horizon:", dcc.Dropdown(
                        id=time_horizon_id, options=time_horizon_options,
                        value=time_horizon_default, searchable=False, style={"minWidth": "180px"})),
                    control("Simulations:", dcc.Dropdown(
                        id=num_simulations_id, options=num_simulations_options,
                        value=num_simulations_default, searchable=False, style={"minWidth": "160px"})),
                    control("Stock Selection:", dcc.Dropdown(
                        id=stock_selection_id, options=stock_selection_options,
                        value=stock_selection_default, searchable=False, style={"minWidth": "220px"})),
                    control("Confidence Interval:", dcc.Dropdown(
                        id=confidence_interval_id, options=confidence_interval_options,
                        value=confidence_interval_default, searchable=False, style={"minWidth": "160px"})),
                ],
            ),
            html.Div(
                className="geib-dual-graph",
                children=[
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("Probability Density Heatmap", className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_heatmap_graph", style=DUAL_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_heatmap_error", className="geib-error"),
                    ]),
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("Cumulative Distribution Function", className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_cdf_graph", style=DUAL_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_cdf_error", className="geib-error"),
                    ]),
                ],
            ),
        ],
    )


def _select_stocks(df: pd.DataFrame, selection_type: str) -> list[str]:
    if selection_type == "top20_mc":
        return df.nlargest(20, "market_cap")["name"].tolist()
    if selection_type == "top10_er":
        return df.nlargest(10, "er_mean")["name"].tolist()
    return df.nlargest(10, "market_cap")["name"].tolist()


def _simulate(df: pd.DataFrame, num_simulations: int, time_horizon: int) -> dict[str, np.ndarray]:
    """Return {stock_name: simulated returns (%)} using a seeded RNG."""
    rng = np.random.default_rng(42)
    horizon_years = time_horizon / 252.0

    # Annualised return std (decimal) from the price-target *level* variance, then
    # scaled to the selected sub-horizon. ``geib.metrics.return_volatility``
    # divides sqrt(kalman_variance) by the spot price so the dispersion is a
    # return rather than a currency amount (the original bare sqrt(var) was in
    # price units). Degenerate / missing dispersion is floored so the normal draw
    # always has a positive scale.
    annual_sigma = return_volatility(df["kalman_variance"], df["original_price"]).to_numpy()
    annual_sigma = np.where(np.isfinite(annual_sigma) & (annual_sigma > 0), annual_sigma, 0.01)
    horizon_sigma = annual_sigma * np.sqrt(horizon_years)

    names = df["name"].to_numpy()
    er_mean = df["er_mean"].to_numpy(dtype="float64")
    out: dict[str, np.ndarray] = {}
    for name, mean, std in zip(names, er_mean, horizon_sigma):
        draws = rng.normal(loc=float(mean), scale=float(std), size=num_simulations)
        out[name] = draws * 100.0
    return out


def _update_logic(**kwargs) -> Tuple[go.Figure, go.Figure]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        empty = _empty("No data is available to display")
        return empty, empty

    df = df[["name", "ticker", "market_cap", "er_mean", "kalman_variance",
             "mc_prob_pos", "original_price"]].copy()
    logger.debug(schema(df))

    time_horizon = int(kwargs.get(time_horizon_id) or time_horizon_default)
    num_simulations = int(kwargs.get(num_simulations_id) or num_simulations_default)
    stock_selection = kwargs.get(stock_selection_id) or stock_selection_default
    confidence_interval = float(kwargs.get(confidence_interval_id) or confidence_interval_default)

    selected = _select_stocks(df, stock_selection)
    df = df[df["name"].isin(selected)].copy()
    if len(df) == 0:
        empty = _empty("No stocks available for the selected criteria")
        return empty, empty

    sims = _simulate(df, num_simulations, time_horizon)

    # --- Probability-density heatmap (10% bins from -50% to +200%) ---
    bin_edges = np.arange(-50, 210, 10)
    bin_labels = [f"{int(bin_edges[i])}% to {int(bin_edges[i + 1])}%" for i in range(len(bin_edges) - 1)]
    heat = {}
    for name, returns in sims.items():
        counts, _ = np.histogram(returns, bins=bin_edges)
        heat[name] = counts / max(len(returns), 1)
    stock_order = list(sims.keys())
    z_matrix = np.array([[heat[name][i] for name in stock_order] for i in range(len(bin_labels))])

    fig_heatmap = go.Figure(
        data=go.Heatmap(z=z_matrix, x=stock_order, y=bin_labels, colorscale="Viridis",
                        colorbar={"title": "Probability"})
    )
    fig_heatmap.update_layout(xaxis_title="Stock Name", yaxis_title="Return Range", height=500)

    # --- CDF lines (5% steps) ---
    return_levels = np.arange(-50, 210, 5)
    cdf_rows = []
    for name, returns in sims.items():
        sorted_returns = np.sort(returns)
        for level in return_levels:
            cdf_rows.append({
                "name": name,
                "return_level": level,
                "cumulative_probability": np.searchsorted(sorted_returns, level, side="right") / len(sorted_returns),
            })
    cdf_df = pd.DataFrame(cdf_rows)

    fig_cdf = px.line(
        cdf_df, x="return_level", y="cumulative_probability", color="name",
        labels={"return_level": "Return Level (%)", "cumulative_probability": "Cumulative Probability"},
    )
    fig_cdf.update_layout(xaxis_title="Return Level (%)", yaxis_title="Cumulative Probability",
                          height=500, hovermode="x unified")

    lower_p = (1 - confidence_interval) / 2
    upper_p = 1 - lower_p
    for name in stock_order:
        stock_cdf = cdf_df[cdf_df["name"] == name].sort_values("return_level")
        lower = stock_cdf[stock_cdf["cumulative_probability"] >= lower_p]["return_level"].min()
        upper = stock_cdf[stock_cdf["cumulative_probability"] >= upper_p]["return_level"].min()
        if not np.isnan(lower):
            fig_cdf.add_vline(x=lower, line_dash="dash", line_color="gray", opacity=0.5)
        if not np.isnan(upper):
            fig_cdf.add_vline(x=upper, line_dash="dash", line_color="gray", opacity=0.5)

    return fig_heatmap, fig_cdf


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(annotations=[{"text": message, "showarrow": False, "font": {"size": 18}}])
    return fig


@callback(
    output=[
        Output(f"{component_id}_heatmap_graph", "figure"),
        Output(f"{component_id}_cdf_graph", "figure"),
        Output(f"{component_id}_heatmap_error", "children"),
        Output(f"{component_id}_cdf_error", "children"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        time_horizon_id: Input(time_horizon_id, "value"),
        num_simulations_id: Input(num_simulations_id, "value"),
        stock_selection_id: Input(stock_selection_id, "value"),
        confidence_interval_id: Input(confidence_interval_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, go.Figure, str, str]:
    try:
        fig_heatmap, fig_cdf = _update_logic(**kwargs)
        return fig_heatmap, fig_cdf, "", ""
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        empty = _empty("An error occurred")
        return empty, empty, msg, msg