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


component_id = "monte_carlo_return_distribution"

time_horizon_id = f"{component_id}_time_horizon"
time_horizon_options = [
    {"label": "1 Month", "value": 30},
    {"label": "3 Months", "value": 90},
    {"label": "6 Months", "value": 180},
    {"label": "1 Year", "value": 252}
]
time_horizon_default = 90

num_simulations_id = f"{component_id}_num_simulations"
num_simulations_options = [
    {"label": "1,000", "value": 1000},
    {"label": "5,000", "value": 5000},
    {"label": "10,000", "value": 10000},
    {"label": "50,000", "value": 50000}
]
num_simulations_default = 10000

stock_selection_id = f"{component_id}_stock_selection"
stock_selection_options = [
    {"label": "Top 10 by Market Cap", "value": "top10_mc"},
    {"label": "Top 20 by Market Cap", "value": "top20_mc"},
    {"label": "Top 10 by Expected Return", "value": "top10_er"},
    {"label": "Custom Selection", "value": "custom"}
]
stock_selection_default = "top10_mc"

custom_stocks_id = f"{component_id}_custom_stocks"

confidence_interval_id = f"{component_id}_confidence_interval"
confidence_interval_options = [
    {"label": "90%", "value": 0.90},
    {"label": "95%", "value": 0.95},
    {"label": "99%", "value": 0.99}
]
confidence_interval_default = 0.95


def component() -> ComponentResponse:
    graph_heatmap_id = f"{component_id}_heatmap_graph"
    graph_cdf_id = f"{component_id}_cdf_graph"
    error_heatmap_id = f"{component_id}_heatmap_error"
    error_cdf_id = f"{component_id}_cdf_error"
    loading_heatmap_id = f"{component_id}_heatmap_loading"
    loading_cdf_id = f"{component_id}_cdf_loading"

    title = "Monte Carlo Return Distribution Simulation"
    description = "Simulates thousands of possible future return scenarios using Monte Carlo methods based on each stock's expected return and volatility. View the probability distribution of outcomes to understand the range of potential returns and downside risks."

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
                            html.Label("Time Horizon:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=time_horizon_id,
                                options=time_horizon_options,
                                value=time_horizon_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Simulations:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=num_simulations_id,
                                options=num_simulations_options,
                                value=num_simulations_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Stock Selection:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=stock_selection_id,
                                options=stock_selection_options,
                                value=stock_selection_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Confidence Interval:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=confidence_interval_id,
                                options=confidence_interval_options,
                                value=confidence_interval_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
            ),
            html.Div(
                id=custom_stocks_id,
                style={"display": "none"}
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "gap": "10px", "marginBottom": "15px"},
                children=[
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Probability Density Heatmap",
                                       style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                            dcc.Loading(
                                id=loading_heatmap_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_heatmap_id,
                                              style={"minHeight": "500px", "height": "calc(100vh - 700px)"}),
                                ]
                            ),
                            html.Pre(id=error_heatmap_id, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Cumulative Distribution Function",
                                       style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                            dcc.Loading(
                                id=loading_cdf_id,
                                type="circle",
                                children=[
                                    ddk.Graph(id=graph_cdf_id,
                                              style={"minHeight": "500px", "height": "calc(100vh - 700px)"}),
                                ]
                            ),
                            html.Pre(id=error_cdf_id, style={"color": "red", "margin": "10px 0"}),
                        ]
                    ),
                ]
            ),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        time_horizon_id: {
            "options": [opt["value"] for opt in time_horizon_options],
            "default": time_horizon_default
        },
        num_simulations_id: {
            "options": [opt["value"] for opt in num_simulations_options],
            "default": num_simulations_default
        },
        stock_selection_id: {
            "options": [opt["value"] for opt in stock_selection_options],
            "default": stock_selection_default
        },
        confidence_interval_id: {
            "options": [opt["value"] for opt in confidence_interval_options],
            "default": confidence_interval_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }


def _get_selected_stocks(df: pd.DataFrame, selection_type: str) -> list[str]:
    """Get list of stock names based on selection type."""
    if selection_type == "top10_mc":
        return df.nlargest(10, "market_cap")["name"].tolist()
    elif selection_type == "top20_mc":
        return df.nlargest(20, "market_cap")["name"].tolist()
    elif selection_type == "top10_er":
        return df.nlargest(10, "er_mean")["name"].tolist()
    else:
        return df.nlargest(10, "market_cap")["name"].tolist()


def _simulate_returns(df: pd.DataFrame, num_simulations: int, time_horizon: int) -> pd.DataFrame:
    """Simulate future returns using Monte Carlo method."""
    logger.debug("Simulating %d return scenarios for %d days...", num_simulations, time_horizon)

    simulations = []
    time_horizon_years = time_horizon / 252.0

    for idx, row in df.iterrows():
        stock_name = row["name"]
        er_mean = float(row["er_mean"])
        kalman_var = float(row["kalman_variance"])

        if kalman_var <= 0:
            kalman_var = 0.01

        std_dev = np.sqrt(kalman_var) * np.sqrt(time_horizon_years)

        simulated_returns = np.random.normal(
            loc=er_mean,
            scale=std_dev,
            size=num_simulations
        )

        for ret in simulated_returns:
            simulations.append({
                "name": stock_name,
                "return": ret * 100
            })

    sim_df = pd.DataFrame(simulations)
    logger.debug(tbl(sim_df))
    return sim_df


def _create_return_bins() -> list[float]:
    """Create return bins from -50% to +200% in 10% increments."""
    return list(np.arange(-50, 210, 10))


def _calculate_probability_density(sim_df: pd.DataFrame, stocks: list[str]) -> pd.DataFrame:
    """Calculate probability density for each stock and return bin."""
    logger.debug("Calculating probability density...")

    bins = _create_return_bins()
    density_data = []

    for stock in stocks:
        stock_returns = sim_df[sim_df["name"] == stock]["return"].values

        if len(stock_returns) == 0:
            continue

        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]

            count = np.sum((stock_returns >= bin_start) & (stock_returns < bin_end))
            probability = count / len(stock_returns)

            density_data.append({
                "name": stock,
                "return_bin": f"{bin_start}% to {bin_end}%",
                "return_center": (bin_start + bin_end) / 2,
                "probability": probability
            })

    density_df = pd.DataFrame(density_data)
    logger.debug(tbl(density_df))
    return density_df


def _calculate_cdf(sim_df: pd.DataFrame, stocks: list[str]) -> pd.DataFrame:
    """Calculate cumulative distribution function for each stock."""
    logger.debug("Calculating cumulative distribution function...")

    cdf_data = []
    return_levels = np.arange(-50, 210, 5)

    for stock in stocks:
        stock_returns = sim_df[sim_df["name"] == stock]["return"].values

        if len(stock_returns) == 0:
            continue

        sorted_returns = np.sort(stock_returns)

        for return_level in return_levels:
            cumulative_prob = np.sum(sorted_returns <= return_level) / len(sorted_returns)

            cdf_data.append({
                "name": stock,
                "return_level": return_level,
                "cumulative_probability": cumulative_prob
            })

    cdf_df = pd.DataFrame(cdf_data)
    logger.debug(tbl(cdf_df))
    return cdf_df


def _update_logic(**kwargs) -> Tuple[go.Figure, go.Figure]:
    """Core chart update logic without error handling."""
    logger.debug("Updating Monte Carlo charts with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

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
        return empty_fig, empty_fig

    logger.debug("Selecting columns for analysis...")
    df = df[["name", "ticker", "market_cap", "er_mean", "kalman_variance", "mc_prob_pos"]].copy()
    logger.debug(schema(df))
    logger.debug(tbl(df))

    time_horizon = kwargs.get(time_horizon_id, time_horizon_default)
    if time_horizon is None:
        time_horizon = time_horizon_default
    time_horizon = int(time_horizon)

    num_simulations = kwargs.get(num_simulations_id, num_simulations_default)
    if num_simulations is None:
        num_simulations = num_simulations_default
    num_simulations = int(num_simulations)

    stock_selection = kwargs.get(stock_selection_id, stock_selection_default)
    if stock_selection is None:
        stock_selection = stock_selection_default

    confidence_interval = kwargs.get(confidence_interval_id, confidence_interval_default)
    if confidence_interval is None:
        confidence_interval = confidence_interval_default
    confidence_interval = float(confidence_interval)

    logger.debug("Getting selected stocks...")
    selected_stocks = _get_selected_stocks(df, stock_selection)
    logger.debug("Selected %d stocks: %s", len(selected_stocks), selected_stocks[:5])

    logger.debug("Filtering to selected stocks...")
    df = df[df["name"].isin(selected_stocks)].copy()
    logger.debug(tbl(df))

    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No stocks available",
            annotations=[{
                "text": "No stocks available for the selected criteria",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, empty_fig

    logger.debug("Running Monte Carlo simulations...")
    sim_df = _simulate_returns(df, num_simulations, time_horizon)

    logger.debug("Calculating probability density...")
    density_df = _calculate_probability_density(sim_df, selected_stocks)

    logger.debug("Calculating cumulative distribution...")
    cdf_df = _calculate_cdf(sim_df, selected_stocks)

    logger.debug("Creating heatmap figure...")
    if len(density_df) > 0:
        heatmap_pivot = density_df.pivot_table(
            index="return_bin",
            columns="name",
            values="probability",
            fill_value=0
        )

        bin_order = [f"{int(b)}% to {int(b + 10)}%" for b in np.arange(-50, 200, 10)]
        heatmap_pivot = heatmap_pivot.reindex([b for b in bin_order if b in heatmap_pivot.index])

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale="Viridis",
                colorbar={"title": "Probability"}
            )
        )
        fig_heatmap.update_layout(
            xaxis_title="Stock Name",
            yaxis_title="Return Range",
            height=500
        )
    else:
        fig_heatmap = go.Figure()
        fig_heatmap.update_layout(
            title="No data for heatmap",
            annotations=[{"text": "No simulation data available", "showarrow": False}]
        )

    logger.debug("Creating CDF figure...")
    if len(cdf_df) > 0:
        fig_cdf = px.line(
            cdf_df,
            x="return_level",
            y="cumulative_probability",
            color="name",
            labels={
                "return_level": "Return Level (%)",
                "cumulative_probability": "Cumulative Probability"
            }
        )
        fig_cdf.update_layout(
            xaxis_title="Return Level (%)",
            yaxis_title="Cumulative Probability",
            height=500,
            hovermode="x unified"
        )

        lower_percentile = (1 - confidence_interval) / 2
        upper_percentile = 1 - lower_percentile

        for stock in selected_stocks:
            stock_cdf = cdf_df[cdf_df["name"] == stock].sort_values("return_level")

            if len(stock_cdf) > 0:
                lower_return = stock_cdf[stock_cdf["cumulative_probability"] >= lower_percentile]["return_level"].min()
                upper_return = stock_cdf[stock_cdf["cumulative_probability"] >= upper_percentile]["return_level"].min()

                if not np.isnan(lower_return):
                    fig_cdf.add_vline(
                        x=lower_return,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.5,
                        annotation_text=f"Lower {lower_percentile * 100:.0f}%",
                        annotation_position="top"
                    )

                if not np.isnan(upper_return):
                    fig_cdf.add_vline(
                        x=upper_return,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.5,
                        annotation_text=f"Upper {upper_percentile * 100:.0f}%",
                        annotation_position="top"
                    )
    else:
        fig_cdf = go.Figure()
        fig_cdf.update_layout(
            title="No data for CDF",
            annotations=[{"text": "No simulation data available", "showarrow": False}]
        )

    logger.debug("Done")
    return fig_heatmap, fig_cdf


@callback(
    output=[
        Output(f"{component_id}_heatmap_graph", "figure"),
        Output(f"{component_id}_cdf_graph", "figure"),
        Output(f"{component_id}_heatmap_error", "children"),
        Output(f"{component_id}_cdf_error", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        time_horizon_id: Input(time_horizon_id, "value"),
        num_simulations_id: Input(num_simulations_id, "value"),
        stock_selection_id: Input(stock_selection_id, "value"),
        confidence_interval_id: Input(confidence_interval_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, go.Figure, str, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        fig_heatmap, fig_cdf = _update_logic(**kwargs)
        return fig_heatmap, fig_cdf, "", ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, empty_fig, error_msg, error_msg