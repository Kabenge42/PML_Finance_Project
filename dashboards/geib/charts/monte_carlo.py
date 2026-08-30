"""Monte Carlo Return Distribution Simulator.

Overlaid probability-density **histogram** + **CDF** lines from per-stock
normal-draw simulations, compared across up to five named companies. Each name's
draws are generated from its Kalman expected return and a horizon-correct return
volatility, then binned (histogram) and sorted (CDF). A live *Target Return*
slider draws a red dashed reference line on both panes.

The dispersion is ``er_sd`` — the forward-return sd in raw decimal. v1 used
is the price-target *level* variance (currency-squared), so it is divided by the
spot ``original_price`` to become a *return* before the normal draw. The bare
``sqrt(kalman_variance)``, which was in price units and had to be divided by
spot; ``er_sd`` is already a return, so the conversion is gone.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import coalesce, column_values, empty_figure
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema
from ..theme import ref_line, COLORWAY, STACKED_GRAPH_STYLE, control
from ..theme import card as theme_card

component_id = "monte_carlo_return_distribution"

num_simulations_id = f"{component_id}_simulations"
num_simulations_options = [
    {"label": "1,000", "value": 1000},
    {"label": "5,000", "value": 5000},
    {"label": "10,000", "value": 10000},
]
num_simulations_default = 5000

target_return_id = f"{component_id}_target_return"
target_return_default = 10

distribution_view_id = f"{component_id}_distribution_view"
distribution_view_options = [
    {"label": "Histogram", "value": "histogram"},
    {"label": "CDF", "value": "cdf"},
    {"label": "Both", "value": "both"},
]
distribution_view_default = "both"

sector_filter_id = f"{component_id}_sector_filter"
companies_id = f"{component_id}_companies"

# Max companies compared at once (spec: "Companies (Max 5)").
_MAX_COMPANIES = 5
# Series get a FIXED categorical slot, never a cycled one: `COLORWAY[i]`, not
# `COLORWAY[i % len(COLORWAY)]`. The modulo never wrapped at 5 <= 8 slots, so it
# was a latent hazard rather than a live defect -- but it made raising the cap a
# silent recolouring, where two companies would share a hue with nothing to say
# so. Raising it past the colourway now fails loudly here instead.
assert _MAX_COMPANIES <= 8, (
    "_MAX_COMPANIES exceeds the categorical colourway. Fold the tail into "
    "'Other' or facet into small multiples -- do not cycle hues."
)
# Histogram bins per company (spec: 50 bins).
_NUM_BINS = 50
# Floor for a degenerate / missing return volatility so the normal draw always
# has a positive scale.
_MIN_SIGMA = 0.01

title = "Monte Carlo Return Distribution Simulator"
description = (
    "Simulate thousands of potential return scenarios to understand the full "
    "range of possible outcomes and their probabilities. Use this to assess the "
    "likelihood of achieving target returns or experiencing losses."
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
                    control("Simulations:", dcc.Dropdown(
                        id=num_simulations_id, options=num_simulations_options,
                        value=num_simulations_default, searchable=False,
                        style={"minWidth": "150px"})),
                    control("Distribution View:", dcc.Dropdown(
                        id=distribution_view_id, options=distribution_view_options,
                        value=distribution_view_default, searchable=False,
                        style={"minWidth": "160px"})),
                    control("Sector:", dcc.Dropdown(
                        id=sector_filter_id, options=[], value=None,
                        placeholder="All Sectors", style={"minWidth": "200px"})),
                    control("Companies (Max 5):", dcc.Dropdown(
                        id=companies_id, options=[], value=[], multi=True,
                        placeholder="Select companies...", style={"minWidth": "300px"})),
                    html.Div(
                        className="geib-control",
                        style={"minWidth": "480px", "paddingRight": "20px"},
                        children=[
                            html.Label("Target Return (%):", className="geib-control-label"),
                            html.Div(id=f"{target_return_id}_display",
                                     style={"fontWeight": "bold", "color": COLORWAY[3],
                                            "marginBottom": "6px"}),
                            dcc.Slider(
                                id=target_return_id, min=-10, max=50, step=5,
                                value=target_return_default,
                                marks={i: f"{i}%" for i in range(-10, 51, 5)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                included=False, updatemode="drag"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="geib-dual-graph geib-dual-graph--stacked",
                children=[
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("Return Distribution (Histogram)", className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_heatmap_graph", style=STACKED_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_heatmap_error", className="geib-error"),
                    ]),
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("Cumulative Distribution Function (CDF)", className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_cdf_graph", style=STACKED_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_cdf_error", className="geib-error"),
                    ]),
                ],
            ),
        ],
    )


def _generate_scenarios(df: pd.DataFrame, num_simulations: int) -> list[dict]:
    """Return per-company Monte Carlo return scenarios (decimal return space).

    The scale is ``er_sd``, the pooled standard deviation of the forward-return
    Monte-Carlo draws, floored to :data:`_MIN_SIGMA` when degenerate. It is a raw
    decimal return over the same NTM horizon as ``expected_return_kalman``, so it
    is used as-is -- the v1 code path converted ``kalman_variance`` (a
    currency-squared price-target level variance) into a return here, and that
    conversion no longer exists to get wrong. Draws use a seeded RNG.
    """
    rng = np.random.default_rng(42)
    sigmas = df["er_sd"].to_numpy(dtype="float64")
    names = df["name"].to_numpy()
    means = df["expected_return_kalman"].to_numpy(dtype="float64")

    results: list[dict] = []
    for name, mean, sigma in zip(names, means, sigmas):
        std = float(sigma) if np.isfinite(sigma) and sigma > 0 else _MIN_SIGMA
        results.append({
            "name": name,
            "scenarios": rng.normal(loc=float(mean), scale=std, size=num_simulations),
        })
    logger.debug("Generated scenarios for %d companies", len(results))
    return results


def _histogram_data(scenarios_list: list[dict], num_bins: int = _NUM_BINS) -> pd.DataFrame:
    """Return long-form histogram data (name, return_pct, probability_density)."""
    rows = []
    for item in scenarios_list:
        counts, edges = np.histogram(item["scenarios"], bins=num_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        total = counts.sum()
        density = counts / total if total > 0 else counts
        for center, prob in zip(centers, density):
            rows.append({"name": item["name"], "return_pct": center * 100.0,
                         "probability_density": prob})
    return pd.DataFrame(rows)


def _cdf_data(scenarios_list: list[dict]) -> pd.DataFrame:
    """Return long-form CDF data (name, return_pct, cumulative_probability)."""
    rows = []
    for item in scenarios_list:
        sorted_scenarios = np.sort(item["scenarios"])
        n = len(sorted_scenarios)
        cumulative = np.arange(1, n + 1) / n
        for value, cum in zip(sorted_scenarios, cumulative):
            rows.append({"name": item["name"], "return_pct": value * 100.0,
                         "cumulative_probability": cum})
    return pd.DataFrame(rows)


def _update_logic(**kwargs) -> Tuple[go.Figure, go.Figure]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        empty = empty_figure("No data is available to display")
        return empty, empty

    df = df[["name", "sector", "market_cap", "expected_return_kalman", "er_sd",
             "original_price", "p_upside_pos_cond", "er_p05", "er_p50", "er_p95"]].copy()
    logger.debug(schema(df))

    num_simulations = int(coalesce(kwargs.get(num_simulations_id), num_simulations_default))
    target_return = float(coalesce(kwargs.get(target_return_id), target_return_default)) / 100.0
    distribution_view = coalesce(kwargs.get(distribution_view_id), distribution_view_default)
    sector_filter = kwargs.get(sector_filter_id)
    companies_selected = kwargs.get(companies_id) or []

    if sector_filter:
        df = df[df["sector"] == sector_filter]
        if len(df) == 0:
            empty = empty_figure("No data is available for the selected sector")
            return empty, empty

    df = df.sort_values("market_cap", ascending=False)

    if len(companies_selected) == 0:
        companies_selected = df.head(_MAX_COMPANIES)["name"].tolist()
    else:
        companies_selected = companies_selected[:_MAX_COMPANIES]
    df = df[df["name"].isin(companies_selected)].copy()
    if len(df) == 0:
        empty = empty_figure("Please select at least one company")
        return empty, empty

    scenarios_list = _generate_scenarios(df, num_simulations)

    fig_histogram = go.Figure()
    fig_cdf = go.Figure()

    if distribution_view in ("histogram", "both"):
        df_hist = _histogram_data(scenarios_list)
        for i, item in enumerate(scenarios_list):
            company = df_hist[df_hist["name"] == item["name"]]
            fig_histogram.add_trace(go.Bar(
                x=company["return_pct"], y=company["probability_density"],
                name=item["name"], opacity=0.7,
                marker_color=COLORWAY[i],
            ))
        fig_histogram.update_layout(
            xaxis_title="Return (%)", yaxis_title="Probability Density",
            barmode="overlay", hovermode="x unified", height=500,
        )
        fig_histogram.add_vline(
            # The reader set this target, so it is the one line they are looking
            # for: emphasis, not the status RED it used to borrow.
            x=target_return * 100.0, **ref_line(
                "emphasis",
                annotation_text=f"Target: {target_return * 100:.0f}%",
                annotation_position="top right"),
        )
    else:
        fig_histogram = empty_figure("Select 'Histogram' or 'Both' to view the histogram")

    if distribution_view in ("cdf", "both"):
        df_cdf = _cdf_data(scenarios_list)
        for i, item in enumerate(scenarios_list):
            company = df_cdf[df_cdf["name"] == item["name"]]
            fig_cdf.add_trace(go.Scatter(
                x=company["return_pct"], y=company["cumulative_probability"],
                mode="lines", name=item["name"],
                line=dict(color=COLORWAY[i]),
                hovertemplate="Return: %{x:.2f}%<br>Cumulative Prob: %{y:.2%}<extra></extra>",
            ))
        fig_cdf.update_layout(
            xaxis_title="Return (%)", yaxis_title="Cumulative Probability",
            hovermode="x unified", height=500,
        )
        fig_cdf.add_vline(
            # The reader set this target, so it is the one line they are looking
            # for: emphasis, not the status RED it used to borrow.
            x=target_return * 100.0, **ref_line(
                "emphasis",
                annotation_text=f"Target: {target_return * 100:.0f}%",
                annotation_position="top right"),
        )
    else:
        fig_cdf = empty_figure("Select 'CDF' or 'Both' to view the CDF")

    return fig_histogram, fig_cdf


@callback(
    output=[
        Output(f"{component_id}_heatmap_graph", "figure"),
        Output(f"{component_id}_cdf_graph", "figure"),
        Output(f"{component_id}_heatmap_error", "children"),
        Output(f"{component_id}_cdf_error", "children"),
        Output(sector_filter_id, "options"),
        Output(companies_id, "options"),
        Output(f"{target_return_id}_display", "children"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        num_simulations_id: Input(num_simulations_id, "value"),
        target_return_id: Input(target_return_id, "value"),
        distribution_view_id: Input(distribution_view_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        companies_id: Input(companies_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, go.Figure, str, str, list, list, str]:
    try:
        df_all = filter_data(get_data(), **kwargs)

        sector_options = [{"label": s, "value": s} for s in column_values(df_all, "sector")]

        # Cascade the company list off the chosen sector so the options track the
        # sector filter (spec: "populated from unique name values in filtered data").
        sector_filter = kwargs.get(sector_filter_id)
        df_companies = df_all[df_all["sector"] == sector_filter] if sector_filter else df_all
        company_options = [{"label": c, "value": c} for c in column_values(df_companies, "name")]

        target_value = coalesce(kwargs.get(target_return_id), target_return_default)
        target_display = f"Selected: {target_value}%"

        fig_histogram, fig_cdf = _update_logic(**kwargs)
        return fig_histogram, fig_cdf, "", "", sector_options, company_options, target_display
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        empty = empty_figure("An error occurred")
        return empty, empty, msg, msg, [], [], ""