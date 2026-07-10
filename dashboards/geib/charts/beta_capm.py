"""Beta & CAPM Expected Return Calculator.

Two side-by-side panes driven by a shared CAPM computation:

* **Security Market Line (SML)** — a sector-coloured scatter of each name's
  ``beta`` against its Kalman ``expected_return_kalman``, overlaid with the
  theoretical CAPM line ``rf + beta * (market_return - rf)``.
* **Alpha Analysis** — a horizontal bar chart of the top-N names by absolute
  ``alpha`` (``expected_return_kalman - capm_expected_return``), coloured by the
  sign of alpha.

``beta`` is a relative systematic-risk proxy: each name's ``signal_strength``
divided by the cross-sectional mean ``signal_strength`` (so the average name has
beta ~ 1). The market return is either the universe mean, or a sector / region
average merged back per name, selected by the *Market Proxy* control.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import coalesce, empty_figure, scoped_filter, sector_values
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema
from ..theme import COLORWAY, SUBTLE_TEXT, DUAL_GRAPH_STYLE, control
from ..theme import card as theme_card

component_id = "beta_capm_return_calculator"

risk_free_rate_id = f"{component_id}_risk_free_rate"
risk_free_rate_options = [
    {"label": "0%", "value": 0.0},
    {"label": "1%", "value": 0.01},
    {"label": "2%", "value": 0.02},
    {"label": "3%", "value": 0.03},
    {"label": "4%", "value": 0.04},
    {"label": "5%", "value": 0.05},
]
risk_free_rate_default = 0.03

market_proxy_id = f"{component_id}_market_proxy"
market_proxy_options = [
    {"label": "All Assets", "value": "all"},
    {"label": "Sector Average", "value": "sector"},
    {"label": "Region Average", "value": "region"},
]
market_proxy_default = "all"

sector_filter_id = f"{component_id}_sector_filter"

alpha_filter_id = f"{component_id}_alpha_filter"
alpha_filter_options = [
    {"label": "All", "value": "all"},
    {"label": "Positive Only", "value": "positive"},
    {"label": "Negative Only", "value": "negative"},
]
alpha_filter_default = "all"

top_n_id = f"{component_id}_top_n"
top_n_options = [
    {"label": "Top 10", "value": 10},
    {"label": "Top 25", "value": 25},
    {"label": "Top 50", "value": 50},
    {"label": "All", "value": 0},
]
top_n_default = 25

size_by_id = f"{component_id}_size_by"
size_by_options = [
    {"label": "None", "value": "none"},
    {"label": "Market Cap", "value": "market_cap"},
    {"label": "Enterprise Value", "value": "enterprise_value"},
    {"label": "Signal Strength", "value": "signal_strength"},
    {"label": "Reward to cVAR", "value": "reward_to_cvar"},

]
size_by_default = "none"

# Minimum marker size for the scatter (spec) and per-row height for the bar chart.
_MARKER_SIZE_MIN = 6
_BAR_ROW_PX = 20
_BAR_MIN_HEIGHT = 400

# Alpha bar colours (spec hex, distinct from the board's theme green/red).
_POS_COLOR = "#2ecc71"
_NEG_COLOR = "#e74c3c"

title = "Beta & CAPM Expected Return Calculator"
description = (
    "Calculate systematic risk (beta) and expected returns using the Capital "
    "Asset Pricing Model. Beta measures how much an asset moves relative to the "
    "market, while CAPM provides the theoretical expected return."
)


def component() -> "object":
    sector_opts = [{"label": s, "value": s} for s in sector_values(get_data())]
    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                className="geib-controls-row",
                children=[
                    control("Risk-Free Rate:", dcc.Dropdown(
                        id=risk_free_rate_id, options=risk_free_rate_options,
                        value=risk_free_rate_default, searchable=False,
                        clearable=False, style={"minWidth": "140px"})),
                    control("Market Proxy:", dcc.Dropdown(
                        id=market_proxy_id, options=market_proxy_options,
                        value=market_proxy_default, searchable=False,
                        clearable=False, style={"minWidth": "160px"})),
                    control("Sectors:", dcc.Dropdown(
                        id=sector_filter_id, options=sector_opts, value=[], multi=True,
                        placeholder="All Sectors", style={"minWidth": "200px"})),
                    control("Alpha Filter:", dcc.Dropdown(
                        id=alpha_filter_id, options=alpha_filter_options,
                        value=alpha_filter_default, searchable=False,
                        clearable=False, style={"minWidth": "150px"})),
                    control("Top N (Alpha):", dcc.Dropdown(
                        id=top_n_id, options=top_n_options, value=top_n_default,
                        searchable=False, clearable=False, style={"minWidth": "140px"})),
                    control("Bubble Size:", dcc.Dropdown(
                        id=size_by_id, options=size_by_options, value=size_by_default,
                        searchable=False, clearable=False, style={"minWidth": "150px"})),
                ],
            ),
            html.Div(
                className="geib-dual-graph",
                children=[
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("Security Market Line (SML)", className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_scatter_graph", style=DUAL_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_scatter_error", className="geib-error"),
                    ]),
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("Alpha Analysis (Top N)", className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_bar_graph", style=DUAL_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_bar_error", className="geib-error"),
                    ]),
                ],
            ),
        ],
    )


def _calculate_beta_and_capm(**kwargs) -> Tuple[pd.DataFrame, float, float]:
    """Return ``(df, market_return, risk_free_rate)`` with beta / CAPM / alpha columns.

    ``beta`` is ``signal_strength`` normalised by its cross-sectional mean; the
    market return follows the selected *Market Proxy* (universe mean, or a sector
    / region average merged per name). ``alpha`` is the realised expected return
    minus the CAPM prediction.
    """
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return pd.DataFrame(), 0.0, 0.0

    risk_free_rate = float(coalesce(kwargs.get(risk_free_rate_id), risk_free_rate_default))
    market_proxy = coalesce(kwargs.get(market_proxy_id), market_proxy_default)

    mean_signal = df["signal_strength"].mean()
    if not np.isfinite(mean_signal) or mean_signal == 0:
        mean_signal = 1.0
    df = df.copy()
    df["beta"] = df["signal_strength"] / mean_signal

    if market_proxy == "sector":
        sector_ret = df.groupby("sector")["risk_adj_return"].transform("mean")
        market_return = float(sector_ret.mean())
    elif market_proxy == "region" and "region" in df.columns:
        region_ret = df.groupby("region")["risk_adj_return"].transform("mean")
        market_return = float(region_ret.mean())
    else:
        market_return = float(df["risk_adj_return"].mean())

    market_risk_premium = market_return - risk_free_rate
    df["capm_expected_return"] = risk_free_rate + df["beta"] * market_risk_premium
    df["alpha"] = df["expected_return_kalman"] - df["capm_expected_return"]
    df["abs_alpha"] = df["alpha"].abs()

    logger.debug("Beta/CAPM: mean_beta=%.4f premium=%.4f mean_alpha=%.4f",
                 df["beta"].mean(), market_risk_premium, df["alpha"].mean())
    return df, market_return, risk_free_rate


def _scatter_figure(df: pd.DataFrame, market_return: float, risk_free_rate: float,
                    size_by: str) -> go.Figure:
    """Sector-coloured SML scatter with the CAPM line overlaid."""
    scatter_kwargs = dict(
        x="beta", y="expected_return_kalman", color="sector",
        color_discrete_sequence=COLORWAY,
        hover_data={"name": True, "beta": ":.4f", "expected_return_kalman": ":.4f", "sector": True},
        labels={"beta": "Beta (Systematic Risk)",
                "expected_return_kalman": "Expected Return", "sector": "Sector"},
    )
    if size_by == "market_cap":
        # px.scatter rejects NaN / negative sizes; clip so every name renders.
        df = df.assign(_size=df["market_cap"].fillna(0.0).clip(lower=0.0))
        scatter_kwargs.update(size="_size")
        scatter_kwargs["hover_data"]["market_cap"] = ":.2f"

    fig = px.scatter(df, **scatter_kwargs)
    if size_by == "market_cap":
        fig.update_traces(marker=dict(sizemin=_MARKER_SIZE_MIN),
                          selector=dict(mode="markers"))
    else:
        fig.update_traces(marker=dict(size=max(_MARKER_SIZE_MIN, 8)),
                          selector=dict(mode="markers"))

    beta_lo, beta_hi = df["beta"].min() * 0.9, df["beta"].max() * 1.1
    sml_beta = np.linspace(beta_lo, beta_hi, 100)
    sml_return = risk_free_rate + sml_beta * (market_return - risk_free_rate)
    fig.add_trace(go.Scatter(
        x=sml_beta, y=sml_return, mode="lines", name="Security Market Line",
        line=dict(color=SUBTLE_TEXT, dash="dash", width=2),
        hovertemplate="Beta: %{x:.4f}<br>SML Return: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Beta (Systematic Risk)", yaxis_title="Expected Return",
        legend_title_text="Sector", hovermode="closest",
    )
    return fig


def _bar_figure(df: pd.DataFrame, alpha_filter: str, top_n: int) -> go.Figure:
    """Horizontal top-N alpha bar chart coloured by alpha sign."""
    if alpha_filter == "positive":
        df = df[df["alpha"] > 0]
    elif alpha_filter == "negative":
        df = df[df["alpha"] < 0]
    if len(df) == 0:
        return empty_figure("No names match the selected alpha filter")

    if top_n > 0:
        df = df.nlargest(top_n, "abs_alpha")
    df = df.sort_values("alpha", ascending=True)
    df = df.assign(alpha_color=np.where(df["alpha"] > 0, "Positive", "Negative"))

    fig = px.bar(
        df, x="alpha", y="name", color="alpha_color", orientation="h",
        hover_data={"name": True, "alpha": ":.4f", "alpha_color": False},
        labels={"alpha": "Alpha (Excess Return)", "name": "Company", "alpha_color": "Alpha Type"},
        color_discrete_map={"Positive": _POS_COLOR, "Negative": _NEG_COLOR},
    )
    fig.update_layout(
        xaxis_title="Alpha (Excess Return)", yaxis_title="Company",
        legend_title_text="Alpha Type", hovermode="closest",
        height=max(_BAR_MIN_HEIGHT, len(df) * _BAR_ROW_PX),
    )
    return fig


def _update_logic(**kwargs) -> Tuple[go.Figure, go.Figure]:
    df, market_return, risk_free_rate = _calculate_beta_and_capm(**kwargs)
    if len(df) == 0:
        empty = empty_figure("No data is available to display")
        return empty, empty

    sector_filter = kwargs.get(sector_filter_id) or []
    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]
    if len(df) == 0:
        empty = empty_figure("No data matches the selected sectors")
        return empty, empty
    logger.debug(schema(df))

    size_by = coalesce(kwargs.get(size_by_id), size_by_default)
    alpha_filter = coalesce(kwargs.get(alpha_filter_id), alpha_filter_default)
    top_n = int(coalesce(kwargs.get(top_n_id), top_n_default))

    fig_scatter = _scatter_figure(df, market_return, risk_free_rate, size_by)
    fig_bar = _bar_figure(df, alpha_filter, top_n)
    return fig_scatter, fig_bar


@callback(
    output=[
        Output(f"{component_id}_scatter_graph", "figure"),
        Output(f"{component_id}_bar_graph", "figure"),
        Output(f"{component_id}_scatter_error", "children"),
        Output(f"{component_id}_bar_error", "children"),
        Output(sector_filter_id, "options"),
        Output(sector_filter_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        market_proxy_id: Input(market_proxy_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        alpha_filter_id: Input(alpha_filter_id, "value"),
        top_n_id: Input(top_n_id, "value"),
        size_by_id: Input(size_by_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, go.Figure, str, str, list, list]:
    # Scope the local Sectors filter to the globally-filtered universe.
    df_all = filter_data(get_data(), **kwargs)
    sector_opts, sector_val = scoped_filter(
        df_all, "sector", kwargs.get(sector_filter_id), multi=True
    )
    kwargs[sector_filter_id] = sector_val
    try:
        fig_scatter, fig_bar = _update_logic(**kwargs)
        return fig_scatter, fig_bar, "", "", sector_opts, sector_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        empty = empty_figure("An error occurred")
        return empty, empty, msg, msg, sector_opts, sector_val