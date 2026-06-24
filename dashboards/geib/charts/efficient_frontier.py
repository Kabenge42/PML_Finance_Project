"""Efficient Frontier Optimization (markdown-spec, random-portfolio version).

Mean-variance simulation: annualise per-asset return/vol from the Kalman
outputs, build a covariance matrix from seeded simulated returns, draw N random
Dirichlet portfolios (seed 42), and surface the efficient frontier plus the
minimum-variance and maximum-Sharpe portfolios with their top-5 holdings.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema, tbl
from ..metrics import PRICE_TARGET_HORIZON_YEARS, return_volatility
from ..theme import GOLD, GREEN, RED, control
from ..theme import card as theme_card
from ._common import coalesce, empty_figure, sector_values

component_id = "efficient_frontier_optimization"

risk_free_rate_id = f"{component_id}_risk_free_rate"
risk_free_rate_options = [
    {"label": "0%", "value": 0.0},
    {"label": "2%", "value": 0.02},
    {"label": "3%", "value": 0.03},
    {"label": "4%", "value": 0.04},
    {"label": "5%", "value": 0.05},
]
risk_free_rate_default = 0.03

sector_filter_id = f"{component_id}_sector_filter"

min_market_cap_id = f"{component_id}_min_market_cap"
min_market_cap_options = [
    {"label": "1,000", "value": 1000},
    {"label": "5,000", "value": 5000},
    {"label": "10,000", "value": 10000},
    {"label": "50,000", "value": 50000},
]
min_market_cap_default = 1000

num_portfolios_id = f"{component_id}_num_portfolios"
num_portfolios_options = [
    {"label": "500", "value": 500},
    {"label": "1,000", "value": 1000},
    {"label": "2,000", "value": 2000},
    {"label": "5,000", "value": 5000},
]
num_portfolios_default = 1000

# Universe cap for tractable covariance/Dirichlet maths (logged when applied).
_MAX_ASSETS = 50

title = "Efficient Frontier Optimization"
description = (
    "Visualizes optimal risk-return tradeoff for portfolio combinations using "
    "mean-variance optimization. Adjust parameters to find portfolios that "
    "maximize return for each level of risk."
)


def component() -> "object":
    df = get_data()
    sector_opts = [{"label": s, "value": s} for s in sector_values(df)]

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
                        value=risk_free_rate_default, searchable=False, style={"minWidth": "150px"})),
                    control("Sectors:", dcc.Dropdown(
                        id=sector_filter_id, options=sector_opts, value=[], multi=True,
                        style={"minWidth": "220px"})),
                    control("Min Market Cap ($M):", dcc.Dropdown(
                        id=min_market_cap_id, options=min_market_cap_options,
                        value=min_market_cap_default, searchable=False, style={"minWidth": "170px"})),
                    control("Number of Portfolios:", dcc.Dropdown(
                        id=num_portfolios_id, options=num_portfolios_options,
                        value=num_portfolios_default, searchable=False, style={"minWidth": "170px"})),
                ],
            ),
            dcc.Loading(type="circle", children=[
                dcc.Graph(id=f"{component_id}_graph", style={"height": "600px"})]),
            html.Pre(id=f"{component_id}_error", className="geib-error"),
            html.Div(id=f"{component_id}_table", style={"margin": "20px 0"}),
        ],
    )


def _simulate_portfolios(mu: np.ndarray, cov: np.ndarray, n_portfolios: int, rf: float):
    """Draw Dirichlet weights (seed 42) and return weights, returns, vols, sharpe."""
    rng = np.random.default_rng(42)
    n_assets = len(mu)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    rets = weights @ mu
    vols = np.sqrt(np.einsum("pi,ij,pj->p", weights, cov, weights))
    vols = np.where(vols <= 0, np.nan, vols)
    sharpe = (rets - rf) / vols
    return weights, rets, vols, sharpe


def _efficient_frontier(rets: np.ndarray, vols: np.ndarray):
    """Bin into 50 volatility bins, return (vol, ret) of the max-return per bin."""
    valid = ~np.isnan(vols)
    v, r = vols[valid], rets[valid]
    if len(v) == 0:
        return np.array([]), np.array([])
    bins = np.linspace(v.min(), v.max(), 51)
    idx = np.digitize(v, bins)
    frontier = []
    for b in range(1, 51):
        mask = idx == b
        if mask.any():
            best = np.argmax(r[mask])
            frontier.append((v[mask][best], r[mask][best]))
    frontier.sort()
    if not frontier:
        return np.array([]), np.array([])
    fv, fr = zip(*frontier)
    return np.array(fv), np.array(fr)


def _holdings(tickers: list[str], weights: np.ndarray, top: int = 5) -> str:
    order = np.argsort(weights)[::-1][:top]
    return ", ".join(f"{tickers[i]} ({weights[i] * 100:.1f}%)" for i in order)


def _update_logic(**kwargs) -> Tuple[go.Figure, html.Div]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return empty_figure("No data is available to display"), html.Div()

    df = df[["ticker", "name", "sector", "market_cap", "mc_prob_pos",
             "original_price", "expected_return_kalman", "kalman_variance"]].copy()
    logger.debug(schema(df))

    rf = coalesce(kwargs.get(risk_free_rate_id), risk_free_rate_default)
    sector_filter = kwargs.get(sector_filter_id) or []
    min_market_cap = coalesce(kwargs.get(min_market_cap_id), min_market_cap_default)
    n_portfolios = int(coalesce(kwargs.get(num_portfolios_id), num_portfolios_default))

    df = df[df["mc_prob_pos"] > 0.5]
    df = df[df["market_cap"] >= min_market_cap]
    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]
    df = df.dropna(subset=["expected_return_kalman", "kalman_variance", "original_price"])
    df = df[(df["kalman_variance"] > 0) & (df["original_price"] > 0)]

    if len(df) < 2:
        return empty_figure("Need at least 2 eligible stocks for an efficient frontier"), html.Div()

    if len(df) > _MAX_ASSETS:
        logger.info("Capping efficient-frontier universe from %d to %d assets", len(df), _MAX_ASSETS)
        df = df.nlargest(_MAX_ASSETS, "market_cap")

    df = df.reset_index(drop=True)
    tickers = df["ticker"].tolist()

    # Annualised per-asset return and volatility from the Kalman outputs (both
    # decimal). ``expected_return_kalman`` is already the total NTM upside and
    # ``kalman_variance`` is the price-target *level* variance, so they are scaled
    # by the real price-target horizon (not a daily 252) and the level variance is
    # divided by the spot price to become a return — see ``geib.metrics``.
    mu = (df["expected_return_kalman"] / PRICE_TARGET_HORIZON_YEARS).to_numpy()
    sigma = return_volatility(df["kalman_variance"], df["original_price"]).to_numpy()

    # Covariance from seeded simulated annualised returns (252 draws/asset).
    rng = np.random.default_rng(42)
    sim = rng.normal(loc=mu, scale=sigma, size=(252, len(mu)))
    cov = np.cov(sim, rowvar=False)
    cov = np.atleast_2d(cov)
    logger.debug(tbl(df))

    weights, rets, vols, sharpe = _simulate_portfolios(mu, cov, n_portfolios, rf)
    ret_pct, vol_pct = rets * 100, vols * 100

    fv, fr = _efficient_frontier(rets, vols)

    min_var_idx = int(np.nanargmin(vols))
    max_sharpe_idx = int(np.nanargmax(sharpe))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vol_pct, y=ret_pct, mode="markers", name="All Portfolios",
        marker=dict(size=5, color=sharpe, colorscale="Viridis", opacity=0.6,
                    colorbar=dict(title="Sharpe Ratio")),
        hovertemplate="Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>",
    ))
    if len(fv):
        fig.add_trace(go.Scatter(
            x=fv * 100, y=fr * 100, mode="lines+markers", name="Efficient Frontier",
            line=dict(color=RED, dash="dash"),
            hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[vol_pct[min_var_idx]], y=[ret_pct[min_var_idx]], mode="markers", name="Min Variance",
        marker=dict(size=16, color=GREEN, symbol="star"),
        hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[vol_pct[max_sharpe_idx]], y=[ret_pct[max_sharpe_idx]], mode="markers", name="Max Sharpe Ratio",
        marker=dict(size=16, color=GOLD, symbol="star"),
        hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Portfolio Volatility (Annualized %)",
        yaxis_title="Expected Return (Annualized %)",
        hovermode="closest", height=600,
    )

    table = _build_table(
        ret_pct, vol_pct, sharpe, weights, tickers, min_var_idx, max_sharpe_idx
    )
    return fig, table


def _build_table(ret_pct, vol_pct, sharpe, weights, tickers, min_var_idx, max_sharpe_idx) -> html.Div:
    headers = ["Portfolio", "Expected Return (%)", "Volatility (%)", "Sharpe Ratio", "Top 5 Holdings"]
    rows = []
    for label, idx in (("Minimum Variance", min_var_idx), ("Maximum Sharpe Ratio", max_sharpe_idx)):
        rows.append(html.Tr([
            html.Td(label),
            html.Td(f"{ret_pct[idx]:.2f}"),
            html.Td(f"{vol_pct[idx]:.2f}"),
            html.Td(f"{sharpe[idx]:.2f}"),
            html.Td(_holdings(tickers, weights[idx])),
        ]))
    return html.Div([
        html.Table(
            [html.Thead(html.Tr([html.Th(h) for h in headers])), html.Tbody(rows)],
            className="geib-table",
        )
    ])


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_table", "children"),
        Output(f"{component_id}_error", "children"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        min_market_cap_id: Input(min_market_cap_id, "value"),
        num_portfolios_id: Input(num_portfolios_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, html.Div, str]:
    try:
        fig, table = _update_logic(**kwargs)
        return fig, table, ""
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return empty_figure("An error occurred"), html.Div(), msg