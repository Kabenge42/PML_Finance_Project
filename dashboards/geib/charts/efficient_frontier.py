"""Efficient Frontier Optimization (markdown-spec, random-portfolio version).

Mean-variance simulation: annualise per-asset return/vol from the Kalman
outputs, build a covariance matrix from seeded simulated returns, draw N random
Dirichlet portfolios (seed 42), and surface the efficient frontier plus the
minimum-variance and maximum-Sharpe portfolios.

The maximum-Sharpe draw is treated as the *optimal* book: its weights drive a
holdings-level allocation table that translates each weight into a dollar
(``Investment Amount`` × ``Currency``) allocation and a risk contribution, with
the portfolio's expected return / volatility / Sharpe summarised beneath it.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import coalesce, column_values, empty_figure, scoped_filter, sector_values
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..components.probability_filter import (
    apply_probability_filter,
    probability_controls,
    probability_inputs,
    register as register_probability_filter,
)
from ..data import get_data
from ..logger import logger, schema, tbl
from ..metrics import PRICE_TARGET_HORIZON_YEARS, quantile_return_volatility
from ..theme import GOLD, RED, control
from ..theme import card as theme_card

component_id = "efficient_frontier_optimization"

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

sector_filter_id = f"{component_id}_sector_filter"

# Probability metric + band (shared control pair). The low handle defaults to the
# 0.8 threshold this card's former "Min Probability" dropdown applied.
min_prob_default = 0.8
register_probability_filter(component_id)

portfolio_size_id = f"{component_id}_portfolio_size"
portfolio_size_options = [
    {"label": "5", "value": 5},
    {"label": "10", "value": 10},
    {"label": "15", "value": 15},
    {"label": "20", "value": 20},
    {"label": "25", "value": 25},
    {"label": "30", "value": 30},
]
portfolio_size_default = 10

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

investment_amount_id = f"{component_id}_investment_amount"
investment_amount_default = 100000

currency_id = f"{component_id}_currency"
currency_default = "USD"

# Number-input styling so the free-text Investment Amount field matches the
# react-select dropdowns (which are themed via ``assets/geib.css``).
_INPUT_STYLE = {
    "minWidth": "150px",
    "height": "36px",
    "padding": "0 10px",
    "borderRadius": "6px",
    "border": "1px solid var(--geib-control-border)",
    "backgroundColor": "var(--geib-control-bg)",
    "color": "var(--geib-body)",
}

title = "Portfolio Risk-Return Optimization"
description = (
    "Find the optimal portfolio allocation by plotting risk versus expected "
    "return. Adjust parameters to see which combinations maximize returns for "
    "each level of risk."
)


def component() -> "object":
    df = get_data()
    sector_opts = [{"label": s, "value": s} for s in sector_values(df)]
    currency_opts = [{"label": c, "value": c} for c in column_values(df, "unit_name")]

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
                    *probability_controls(component_id, lo=min_prob_default),
                    control("Portfolio Size:", dcc.Dropdown(
                        id=portfolio_size_id, options=portfolio_size_options,
                        value=portfolio_size_default, searchable=False, style={"minWidth": "150px"})),
                    control("Min Market Cap ($M):", dcc.Dropdown(
                        id=min_market_cap_id, options=min_market_cap_options,
                        value=min_market_cap_default, searchable=False, style={"minWidth": "170px"})),
                    control("Number of Portfolios:", dcc.Dropdown(
                        id=num_portfolios_id, options=num_portfolios_options,
                        value=num_portfolios_default, searchable=False, style={"minWidth": "170px"})),
                    control("Investment Amount:", dcc.Input(
                        id=investment_amount_id, type="number", value=investment_amount_default,
                        min=0, placeholder="Enter amount...", style=_INPUT_STYLE)),
                    control("Currency:", dcc.Dropdown(
                        id=currency_id, options=currency_opts, value=currency_default,
                        searchable=True, style={"minWidth": "150px"})),
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


def _risk_contributions(weights: np.ndarray, cov: np.ndarray, vol: float) -> np.ndarray:
    """Return each asset's contribution to portfolio risk (sums to *vol*).

    The marginal contribution to risk is ``(cov @ w) / vol``; scaling it by the
    asset weight gives the contribution to risk, whose elements sum exactly to
    the portfolio volatility (Euler allocation of risk).

    Parameters
    ----------
    weights
        Portfolio weights (sum to 1).
    cov
        Asset return covariance matrix.
    vol
        Portfolio volatility ``sqrt(w' cov w)``.

    Returns
    -------
    numpy.ndarray
        Per-asset contribution to risk; ``zeros`` when ``vol`` is non-positive.
    """
    if vol <= 0:
        return np.zeros_like(weights)
    marginal = cov @ weights / vol
    return weights * marginal


def _update_logic(**kwargs) -> Tuple[go.Figure, html.Div]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return empty_figure("No data is available to display"), html.Div()

    # Gate on the selected probability metric *before* the projection below —
    # three of the four selectable metrics are not in that column list.
    df = apply_probability_filter(df, component_id, kwargs)

    df = df[["ticker", "name", "sector", "unit_name", "market_cap",
             "expected_return_kalman", "er_p05", "er_p95"]].copy()
    logger.debug(schema(df))

    rf = coalesce(kwargs.get(risk_free_rate_id), risk_free_rate_default)
    sector_filter = kwargs.get(sector_filter_id) or []
    portfolio_size = int(coalesce(kwargs.get(portfolio_size_id), portfolio_size_default))
    min_market_cap = coalesce(kwargs.get(min_market_cap_id), min_market_cap_default)
    n_portfolios = int(coalesce(kwargs.get(num_portfolios_id), num_portfolios_default))
    investment_amount = float(coalesce(kwargs.get(investment_amount_id), investment_amount_default))
    currency = coalesce(kwargs.get(currency_id), currency_default)

    df = df[df["market_cap"] >= min_market_cap]
    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]
    df = df.dropna(subset=["expected_return_kalman", "er_p05", "er_p95"])
    df = df[df["er_p95"] > df["er_p05"]]

    if len(df) < 2:
        return empty_figure("Need at least 2 eligible stocks for an efficient frontier"), html.Div()

    # Portfolio Size: keep the top-N names by market cap for tractable
    # covariance/Dirichlet maths over the requested book size.
    if len(df) > portfolio_size:
        logger.info("Selecting top %d of %d assets by market cap", portfolio_size, len(df))
        df = df.nlargest(portfolio_size, "market_cap")

    df = df.reset_index(drop=True)
    tickers = df["ticker"].tolist()
    names = df["name"].tolist()

    # Per-asset expected return and risk — both decimal and already on the NTM
    # price-target horizon (so no daily-252 annualisation). ``expected_return_kalman``
    # is the total NTM upside. Risk is the asset's *return* dispersion taken from the
    # Monte-Carlo return distribution: the 5th–95th-percentile spread implies, under a
    # normal approximation (``quantile_return_volatility``). ``er_mean`` tracks
    # ``expected_return_kalman``, so the mean and risk stay on one distribution.
    #
    # NOTE: ``expected_upside_sd`` must NOT be used as risk here. It is the posterior
    # variance of the price-target *level* (estimation uncertainty of the mean), not
    # the return volatility; converted to a return and diversified across the book it
    # collapsed portfolio vol to ~0.1% and pushed Sharpe ratios past 150 (e.g. a
    # "Volatility (%) 0.13" with Sharpe 168 against a 24.68% expected return).
    mu = (df["expected_return_kalman"] / PRICE_TARGET_HORIZON_YEARS).to_numpy()
    sigma = quantile_return_volatility(df["er_p05"], df["er_p95"]).to_numpy()

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
        x=vol_pct, y=ret_pct, mode="markers", name="Random Portfolios",
        marker=dict(size=5, color=sharpe, colorscale="Viridis", opacity=0.6,
                    line=dict(width=0.5, color="white"),
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
        marker=dict(size=16, color=RED, symbol="diamond", line=dict(width=2, color="darkred")),
        hovertemplate="Min Variance<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[vol_pct[max_sharpe_idx]], y=[ret_pct[max_sharpe_idx]], mode="markers", name="Max Sharpe Ratio",
        marker=dict(size=18, color=GOLD, symbol="star", line=dict(width=2, color="darkgoldenrod")),
        hovertemplate="Max Sharpe Ratio<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Portfolio Volatility (Annualized)",
        yaxis_title="Expected Return",
        hovermode="closest", height=600,
    )

    table = _build_table(
        mu, cov, rets, vols, sharpe, weights, tickers, names,
        min_var_idx, max_sharpe_idx, rf, investment_amount, currency,
    )
    return fig, table


def _build_table(mu, cov, rets, vols, sharpe, weights, tickers, names,
                 min_var_idx, max_sharpe_idx, rf, investment_amount, currency) -> html.Div:
    """Build the optimal-allocation table + min-var / max-Sharpe comparison.

    The maximum-Sharpe draw is treated as the optimal book: its weights drive a
    holdings-level table (weight, expected return, risk contribution, dollar
    allocation) plus a portfolio summary. A compact comparison table beneath it
    keeps the minimum-variance / maximum-Sharpe headline metrics in view.
    """
    opt_w = weights[max_sharpe_idx]
    opt_ret = rets[max_sharpe_idx]
    opt_vol = vols[max_sharpe_idx]
    opt_sharpe = sharpe[max_sharpe_idx]

    contrib = _risk_contributions(opt_w, cov, opt_vol)
    contrib_share = contrib / opt_vol if opt_vol > 0 else np.zeros_like(contrib)
    dollars = investment_amount * opt_w

    order = np.argsort(opt_w)[::-1]
    alloc_headers = ["Ticker", "Name", "Weight (%)", "Expected Return (%)",
                     "Contribution to Risk (%)", f"Allocation ({currency})"]
    alloc_rows = [
        html.Tr([
            html.Td(tickers[i]),
            html.Td(names[i]),
            html.Td(f"{opt_w[i] * 100:.2f}"),
            html.Td(f"{mu[i] * 100:.2f}"),
            html.Td(f"{contrib_share[i] * 100:.2f}"),
            html.Td(f"{dollars[i]:,.2f}"),
        ])
        for i in order
    ]

    summary = html.Div(
        [
            html.P(f"Portfolio Expected Return: {opt_ret * 100:.2f}%"),
            html.P(f"Portfolio Volatility: {opt_vol * 100:.2f}%"),
            html.P(f"Sharpe Ratio: {opt_sharpe:.4f}"),
            html.P(f"Total Investment: {currency} {investment_amount:,.2f}"),
        ],
        style={"marginTop": "16px", "fontWeight": "bold"},
    )

    cmp_headers = ["Portfolio", "Expected Return (%)", "Volatility (%)", "Sharpe Ratio"]
    cmp_rows = [
        html.Tr([
            html.Td(label),
            html.Td(f"{rets[idx] * 100:.2f}"),
            html.Td(f"{vols[idx] * 100:.2f}"),
            html.Td(f"{sharpe[idx]:.2f}"),
        ])
        for label, idx in (("Minimum Variance", min_var_idx),
                           ("Maximum Sharpe Ratio", max_sharpe_idx))
    ]

    return html.Div([
        html.H5("Optimal Portfolio Allocation", style={"color": "#ffffff"}),
        html.Table(
            [html.Thead(html.Tr([html.Th(h) for h in alloc_headers])), html.Tbody(alloc_rows)],
            className="geib-table",
            style={"width": "100%"},
        ),
        summary,
        html.Table(
            [html.Thead(html.Tr([html.Th(h) for h in cmp_headers])), html.Tbody(cmp_rows)],
            className="geib-table",
            style={"marginTop": "20px"},
        ),
    ])


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_table", "children"),
        Output(f"{component_id}_error", "children"),
        Output(sector_filter_id, "options"),
        Output(sector_filter_id, "value"),
        Output(currency_id, "options"),
        Output(currency_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        risk_free_rate_id: Input(risk_free_rate_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        **probability_inputs(component_id),
        portfolio_size_id: Input(portfolio_size_id, "value"),
        min_market_cap_id: Input(min_market_cap_id, "value"),
        num_portfolios_id: Input(num_portfolios_id, "value"),
        investment_amount_id: Input(investment_amount_id, "value"),
        currency_id: Input(currency_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, html.Div, str, list, list, list, str]:
    # Scope the local Sector + Currency pickers to the globally-filtered universe.
    df_all = filter_data(get_data(), **kwargs)
    sector_opts, sector_val = scoped_filter(
        df_all, "sector", kwargs.get(sector_filter_id), multi=True
    )
    currency_opts, currency_val = scoped_filter(df_all, "unit_name", kwargs.get(currency_id))
    kwargs[sector_filter_id] = sector_val
    kwargs[currency_id] = currency_val
    try:
        fig, table = _update_logic(**kwargs)
        return fig, table, "", sector_opts, sector_val, currency_opts, currency_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return (empty_figure("An error occurred"), html.Div(), msg,
                sector_opts, sector_val, currency_opts, currency_val)