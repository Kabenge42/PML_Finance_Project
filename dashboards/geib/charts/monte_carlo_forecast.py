"""Monte Carlo Return Distribution Forecast (single name).

Simulates thousands of forward price paths for one selected name and renders
the resulting terminal-return distribution, overlaid with the posterior
percentiles already carried on the analytics row (``er_p05`` / ``er_p50`` /
``er_p95``) plus a statistics table.

Distinct from :mod:`dashboards.geib.charts.monte_carlo`, which compares the
*cross-section* of stocks via a probability heatmap and CDF lines. This panel
zooms into a single name.

The simulation reuses the horizon-correct conversion in
:mod:`dashboards.geib.metrics`: ``expected_return_kalman`` is the total
next-twelve-month implied upside and ``kalman_variance`` is the price-target
*level* variance, so they are de-annualised over the real price-target horizon
(not a daily ``* 252``) and the level variance is divided by the spot price to
become a return.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import coalesce, empty_figure, scoped_filter, top_label
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema
from ..metrics import PRICE_TARGET_HORIZON_YEARS, return_volatility
from ..theme import GOLD, GREEN, RED, control
from ..theme import card as theme_card

component_id = "monte_carlo_return_distribution_forecast"

name_id = f"{component_id}_name"

horizon_id = f"{component_id}_horizon"
horizon_options = [
    {"label": "30 days", "value": 30},
    {"label": "60 days", "value": 60},
    {"label": "90 days", "value": 90},
    {"label": "180 days", "value": 180},
    {"label": "270 days", "value": 270},
    {"label": "365 days", "value": 365},
]
horizon_default = 90

simulations_id = f"{component_id}_simulations"
simulations_options = [
    {"label": "1,000", "value": 1000},
    {"label": "5,000", "value": 5000},
    {"label": "10,000", "value": 10000},
    {"label": "50,000", "value": 50000},
]
simulations_default = 10000

distribution_id = f"{component_id}_distribution"
distribution_options = [
    {"label": "Normal", "value": "normal"},
    {"label": "Student-t", "value": "student_t"},
]
distribution_default = "normal"

# Trading days per year — the frequency the daily draws are compounded over.
_TRADING_DAYS = 252
# Student-t degrees of freedom (heavier tails than the normal draw).
_STUDENT_T_DF = 5

title = "Monte Carlo Return Distribution Forecast"
description = (
    "Simulates thousands of forward price paths for a single name from its "
    "Kalman-filtered expected return and volatility, then plots the terminal "
    "return distribution against the posterior 5th / 50th / 95th percentiles. "
    "Use it to weigh likely outcomes versus the tails for one name."
)


def component() -> "object":
    df = get_data()
    try:
        names = sorted(t for t in df["name"].dropna().unique().tolist() if str(t).strip())
    except Exception:  # pragma: no cover - defensive
        names = []
    name_opts = [{"label": str(t), "value": str(t)} for t in names]
    # Default to the largest-market-cap name; the callback reconciles this to the
    # highest-cap name still present under the active global filters.
    name_default = top_label(df, "market_cap") or (names[0] if names else None)

    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                className="geib-controls-row",
                children=[
                    control("Name:", dcc.Dropdown(
                        id=name_id, options=name_opts, value=name_default,
                        searchable=True, style={"minWidth": "200px"})),
                    control("Forecast Horizon:", dcc.Dropdown(
                        id=horizon_id, options=horizon_options, value=horizon_default,
                        searchable=False, style={"minWidth": "160px"})),
                    control("Simulations:", dcc.Dropdown(
                        id=simulations_id, options=simulations_options, value=simulations_default,
                        searchable=False, style={"minWidth": "150px"})),
                    control("Distribution:", dcc.Dropdown(
                        id=distribution_id, options=distribution_options, value=distribution_default,
                        searchable=False, style={"minWidth": "150px"})),
                ],
            ),
            dcc.Loading(type="circle", children=[
                dcc.Graph(id=f"{component_id}_graph", style={"height": "550px"})]),
            html.Pre(id=f"{component_id}_error", className="geib-error"),
            html.Div(id=f"{component_id}_table", style={"margin": "20px 0"}),
        ],
    )


def _simulate_terminal_returns(
    initial_price: float,
    annual_mean: float,
    annual_sigma: float,
    days: int,
    num_simulations: int,
    distribution_type: str,
) -> np.ndarray:
    """Simulate ``num_simulations`` price paths and return their terminal returns.

    Annual inputs (already horizon-correct, see module docstring) are converted
    to per-trading-day drift/scale, compounded over *days* steps from a seeded
    RNG, and reduced to ``(final_price - initial_price) / initial_price``.
    """
    rng = np.random.default_rng(42)
    daily_mean = annual_mean / _TRADING_DAYS
    daily_std = annual_sigma / np.sqrt(_TRADING_DAYS)

    if distribution_type == "student_t":
        # Standardise the t-draw to unit variance before scaling so the chosen
        # daily_std is the realised dispersion (raw t has variance df/(df-2)).
        raw = rng.standard_t(_STUDENT_T_DF, size=(num_simulations, days))
        unit = raw / np.sqrt(_STUDENT_T_DF / (_STUDENT_T_DF - 2))
        returns = unit * daily_std + daily_mean
    else:  # normal
        returns = rng.normal(daily_mean, daily_std, size=(num_simulations, days))

    price_paths = initial_price * np.cumprod(1.0 + returns, axis=1)
    return (price_paths[:, -1] - initial_price) / initial_price


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normal probability density (numpy-only, avoids a scipy dependency)."""
    if sigma <= 0:
        return np.zeros_like(x)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _update_logic(**kwargs) -> Tuple[go.Figure, html.Div]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return empty_figure("No data is available to display"), html.Div()

    name = kwargs.get(name_id)
    if not name:
        return empty_figure("Select a name to run the forecast"), html.Div()

    df = df[df["name"] == name]
    if len(df) == 0:
        return empty_figure(f"No data available for {name}"), html.Div()
    logger.debug(schema(df))

    row = df.iloc[0]
    initial_price = float(row["original_price"])
    if not np.isfinite(initial_price) or initial_price <= 0:
        return empty_figure(f"No usable price for {name}"), html.Div()

    horizon_days = int(coalesce(kwargs.get(horizon_id), horizon_default))
    num_simulations = int(coalesce(kwargs.get(simulations_id), simulations_default))
    distribution_type = coalesce(kwargs.get(distribution_id), distribution_default)

    # Horizon-correct annualised drift and return volatility (see metrics module).
    annual_mean = float(row["expected_return_kalman"]) / PRICE_TARGET_HORIZON_YEARS
    annual_sigma = float(return_volatility(row["kalman_variance"], initial_price))
    if not np.isfinite(annual_sigma) or annual_sigma <= 0:
        annual_sigma = 0.01

    logger.debug("Simulating %d paths x %d days for %s", num_simulations, horizon_days, name)
    terminal = _simulate_terminal_returns(
        initial_price, annual_mean, annual_sigma, horizon_days, num_simulations, distribution_type
    )

    counts, bin_edges = np.histogram(terminal, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bin_centers * 100, y=counts, mode="lines", name="Simulated Distribution",
        line=dict(color="#06B6D4", width=2), fill="tozeroy",
        fillcolor="rgba(6, 182, 212, 0.3)",
    ))
    if distribution_type == "normal":
        x_range = np.linspace(terminal.min(), terminal.max(), 200)
        fig.add_trace(go.Scatter(
            x=x_range * 100, y=_normal_pdf(x_range, terminal.mean(), terminal.std()),
            mode="lines", name="Normal Fit", line=dict(color=GOLD, width=2, dash="dash"),
        ))

    er_p05, er_p50, er_p95 = float(row["er_p05"]), float(row["er_p50"]), float(row["er_p95"])
    sim_median = float(np.median(terminal))
    fig.add_vline(x=er_p05 * 100, line_dash="dash", line_color=RED,
                  annotation_text="5th %ile", annotation_position="top left")
    fig.add_vline(x=er_p50 * 100, line_dash="solid", line_color=GREEN,
                  annotation_text="Median", annotation_position="top right")
    fig.add_vline(x=er_p95 * 100, line_dash="dash", line_color="#06B6D4",
                  annotation_text="95th %ile", annotation_position="bottom right")
    # Median of the actual simulated distribution (distinct from the posterior
    # er_p50 median above), drawn in a contrasting colour/style.
    fig.add_vline(x=sim_median * 100, line_dash="dot", line_color="#A855F7",
                  annotation_text="Sim. Median", annotation_position="bottom left")

    fig.update_layout(
        xaxis_title="Return (%)", yaxis_title="Probability Density",
        hovermode="x unified", height=550, legend=dict(x=0.02, y=0.98),
    )

    table = _build_table(row, terminal, num_simulations, horizon_days, initial_price)
    return fig, table


def _build_table(row, terminal, num_simulations, horizon_days, initial_price) -> html.Div:
    # Median of the actual simulated terminal-return distribution (distinct from
    # the posterior er_p50). The simulated price target is that median return
    # applied to the path's starting price, i.e. the median terminal price level.
    sim_median_return = float(np.median(terminal))
    sim_price_target = initial_price * (1.0 + sim_median_return)
    # Kalman-filtered posterior-mean price target level (analytics row column).
    kalman_price_target = float(row["price_target_kalman"])

    rows = [
        ("Expected Return", f"{float(row['expected_return_kalman']) * 100:.2f}%",
         "Mean implied upside from the Kalman filter"),
        ("Simulated Expected Return", f"{sim_median_return * 100:.2f}%",
         "Median simulated return based on the simulated distribution"),
        ("Initial Price", f"{initial_price:.2f}", "Current price the paths start from"),
        ("Expected Price Target", f"{kalman_price_target:.2f}",
         "Expected price target from the Kalman filter"),
        ("Simulated Price Target", f"{sim_price_target:.2f}",
         "Median simulated terminal price target path"),
        ("Median Return (50th %ile)", f"{float(row['er_p50']) * 100:.2f}%",
         "Median posterior outcome"),
        ("5th Percentile (VaR)", f"{float(row['er_p05']) * 100:.2f}%",
         "Worst 5% of outcomes"),
        ("95th Percentile", f"{float(row['er_p95']) * 100:.2f}%",
         "Best 5% of outcomes"),
        ("Probability of Positive Return", f"{float(row['mc_prob_pos']) * 100:.2f}%",
         "Posterior likelihood of a gain"),
        ("Simulated Std Dev", f"{terminal.std() * 100:.2f}%",
         "Dispersion of the simulated terminal returns"),
        # cvar_5pct_kalman is stored on a percent scale already (unlike the
        # decimal er_p* / expected_return_kalman columns), so it is NOT * 100.
        ("CVaR (5%)", f"{float(row['cvar_5pct_kalman']):.2f}%",
         "Expected loss in the worst 5% tail"),
        ("Reward-to-CVaR", f"{float(row['reward_to_cvar']):.2f}",
         "Risk-adjusted return vs tail loss"),
        ("Simulations Run", f"{num_simulations:,}", "Number of price paths simulated"),
        ("Forecast Horizon (days)", f"{horizon_days}", "Time horizon for the forecast")

    ]
    headers = ["Metric", "Value", "Interpretation"]
    body = [
        html.Tr([html.Td(metric), html.Td(value), html.Td(note)])
        for metric, value, note in rows
    ]
    return html.Div([
        html.H4("Statistical Summary", style={"marginBottom": "10px"}),
        html.Table(
            [html.Thead(html.Tr([html.Th(h) for h in headers])), html.Tbody(body)],
            className="geib-table",
        ),
    ])


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_table", "children"),
        Output(f"{component_id}_error", "children"),
        Output(name_id, "options"),
        Output(name_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        name_id: Input(name_id, "value"),
        horizon_id: Input(horizon_id, "value"),
        simulations_id: Input(simulations_id, "value"),
        distribution_id: Input(distribution_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, html.Div, str, list, str]:
    # Scope the Name dropdown to the globally-filtered universe, defaulting to the
    # highest-market-cap name whenever the current pick is filtered out.
    df_all = filter_data(get_data(), **kwargs)
    name_opts, name_val = scoped_filter(
        df_all, "name", kwargs.get(name_id), fallback=top_label(df_all, "market_cap")
    )
    kwargs[name_id] = name_val
    try:
        fig, table = _update_logic(**kwargs)
        return fig, table, "", name_opts, name_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return empty_figure("An error occurred"), html.Div(), msg, name_opts, name_val