"""Kalman State & Structural Forecast (single name).

Renders one name's Kalman-filtered price-target forecast as a fan chart: the
real price history from the ``price_*_ago`` lookback ladder anchoring the fan at
today's spot, a mean forecast line drifting toward the Kalman price target over
the selected horizon, a widening HDI confidence band, an optional cloud of
posterior predictive draws, the observed analyst target with its real
consensus band (``price_target_low`` / ``price_target_median`` /
``price_target_high``), the current price, and vertical markers for the
upcoming earnings / fiscal events. Names with an all-NaN price ladder fall back
to a flat anchor segment at spot.

Unlike :mod:`dashboards.geib.charts.monte_carlo_forecast` (which collapses the
simulation to a terminal-return *distribution*), this panel keeps the *time*
axis: it shows how the posterior price target and its uncertainty evolve forward
from the snapshot.

The forecast is built from the horizon-correct quantities in
:mod:`dashboards.geib.metrics` rather than the daily ``* 252`` scaling of the
original Plotly stub: ``expected_return_kalman`` is the *total* NTM implied upside
(so it is de-annualised over the real price-target horizon) and the return
dispersion is taken from the posterior ``er_p05`` / ``er_p95`` percentiles
(``er_sd`` is the fallback — the forward-return sd in raw decimal,
which understates return risk; see :func:`~dashboards.geib.metrics.return_volatility`).
"""

from __future__ import annotations

import traceback
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import (
    cell,
    coalesce,
    empty_figure,
    finite_cell,
    name_options,
    scoped_filter,
    top_label,
)
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema
from ..metrics import (
    PRICE_SUFFIXES,
    PRICE_TARGET_HORIZON_YEARS,
    history_ladder,
    quantile_return_volatility,
)
from ..theme import card as theme_card
from ..theme import control

component_id = "kalman_state_structural_forecast"

name_id = f"{component_id}_name"
horizon_id = f"{component_id}_horizon"
draws_id = f"{component_id}_draws"
confidence_id = f"{component_id}_confidence"

horizon_options = [
    {"label": "3 Months", "value": "3m"},
    {"label": "6 Months", "value": "6m"},
    {"label": "9 Months", "value": "9m"},
    {"label": "12 Months", "value": "12m"},
]
horizon_default = "12m"
# Horizon label -> forward calendar days on the forecast axis.
_HORIZON_DAYS = {"3m": 90, "6m": 180, "9m": 270, "12m": 365}

draws_options = [
    {"label": "Show", "value": "show"},
    {"label": "Hide", "value": "hide"},
]
draws_default = "show"

confidence_options = [
    {"label": "90% HDI", "value": "90"},
    {"label": "94% HDI", "value": "94"},
    {"label": "99% HDI", "value": "99"},
]
confidence_default = "94"
# Two-sided normal z-multiplier for each HDI level (z for the (1+level)/2 quantile).
_CONFIDENCE_Z = {"90": 1.6448536269514722, "94": 1.8807936081512509, "99": 2.5758293035489004}

# History window before "today" anchoring the forecast fan. 183 days so the 6m
# ladder point (today - 182d) fits inside the window.
_HISTORY_DAYS = 183
# Analyst price targets are next-twelve-month: the consensus band spans <= 1y.
_TARGET_BAND_DAYS = 365
# Real analyst consensus band fill (green at low alpha, distinct from the blue
# model-HDI _BAND_FILL).
_TARGET_BAND_FILL = "rgba(16,185,129,0.10)"
# Trading days per year — the frequency the daily posterior draws compound over.
_TRADING_DAYS = 252
# Number of posterior predictive paths drawn when "Show" is selected.
_NUM_DRAWS = 50

# --- Spec palette (semantic, readable on the dark board template) -----------
_FORECAST_COLOR = "rgb(0,100,200)"  # blue Kalman mean forecast
_TARGET_COLOR = "#10B981"  # green observed analyst target (theme accent-positive)
_PRICE_COLOR = "#FACC15"  # yellow current-price marker
_BAND_FILL = "rgba(0,100,200,0.2)"  # light-blue HDI band fill
_DRAW_COLOR = "rgba(150,150,150,0.18)"  # faint gray posterior draws
_EVENT_COLOR = "rgb(230,150,0)"  # orange event markers

title = "Kalman State & Structural Forecast"
description = (
    "Kalman-filtered price targets with Monte Carlo probability distributions "
    "and prediction intervals"
)


def component() -> "object":
    name_opts, name_default = name_options(get_data())

    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                className="geib-controls-row",
                children=[
                    control(
                        "Stock:",
                        dcc.Dropdown(
                            id=name_id,
                            options=name_opts,
                            value=name_default,
                            searchable=True,
                            style={"minWidth": "220px"},
                        ),
                    ),
                    control(
                        "Forecast Horizon:",
                        dcc.Dropdown(
                            id=horizon_id,
                            options=horizon_options,
                            value=horizon_default,
                            searchable=False,
                            style={"minWidth": "150px"},
                        ),
                    ),
                    control(
                        "Posterior Draws:",
                        dcc.Dropdown(
                            id=draws_id,
                            options=draws_options,
                            value=draws_default,
                            searchable=False,
                            style={"minWidth": "130px"},
                        ),
                    ),
                    control(
                        "Confidence Interval:",
                        dcc.Dropdown(
                            id=confidence_id,
                            options=confidence_options,
                            value=confidence_default,
                            searchable=False,
                            style={"minWidth": "150px"},
                        ),
                    ),
                ],
            ),
            dcc.Loading(
                type="circle",
                children=[
                    dcc.Graph(
                        id=f"{component_id}_graph",
                        style={"minHeight": "550px", "height": "calc(100vh - 600px)"},
                    )
                ],
            ),
            html.Pre(id=f"{component_id}_error", className="geib-error"),
        ],
    )


def _forecast_mean(spot: float, annual_return: float, days_forward: np.ndarray) -> np.ndarray:
    """Mean forecast level drifting forward from *spot* at today.

    Compounds the (annualised) expected return over the forward horizon so the
    line reaches ``spot * (1 + annual_return)`` at one year — i.e. the Kalman
    price-target level implied by ``expected_return_kalman``. *days_forward* is
    expected to be non-negative (the axis starts at today); any negative values
    are clipped to the spot anchor.
    """
    fwd_years = np.clip(days_forward, 0, None) / 365.0
    return spot * np.power(1.0 + annual_return, fwd_years)


def _posterior_draws(
        spot: float,
        annual_mean: float,
        annual_sigma: float,
        days_forward: np.ndarray,
) -> np.ndarray:
    """Return ``_NUM_DRAWS`` forward price paths from *spot* at today.

    Daily normal return draws (drift/scale de-annualised over ``_TRADING_DAYS``)
    are compounded from *spot*; steps at or before today are zeroed so every path
    fans out from the current price at the snapshot.
    """
    rng = np.random.default_rng(42)
    daily_mean = annual_mean / _TRADING_DAYS
    daily_std = annual_sigma / np.sqrt(_TRADING_DAYS)
    forward_mask = days_forward > 0

    returns = rng.normal(daily_mean, daily_std, size=(_NUM_DRAWS, days_forward.size))
    returns[:, ~forward_mask] = 0.0
    return spot * np.cumprod(1.0 + returns, axis=1)


def _update_logic(**kwargs) -> go.Figure:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return empty_figure("No data is available to display")

    name = kwargs.get(name_id)
    if not name:
        return empty_figure("Select a stock to display the forecast")

    df = df[df["name"] == name]
    if len(df) == 0:
        return empty_figure(f"No data available for {name}")
    logger.debug(schema(df))

    # Most recent row for the name (spec: sort by report date, take the latest).
    if "income_statement_report_date" in df.columns:
        df = df.sort_values("income_statement_report_date")
    row = df.iloc[-1]

    # Scalar-safe row reads (``finite_cell``/``cell``): a raw ``row[column]``
    # inside a boolean test raises "The truth value of a Series is ambiguous"
    # as soon as the value is not the scalar the test assumes.
    spot = finite_cell(row, "original_price")
    if spot is None or spot <= 0:
        return empty_figure(f"No usable price for {name}")

    original_target = finite_cell(row, "original_target")

    horizon = coalesce(kwargs.get(horizon_id), horizon_default)
    show_draws = coalesce(kwargs.get(draws_id), draws_default) == "show"
    confidence = coalesce(kwargs.get(confidence_id), confidence_default)
    z = _CONFIDENCE_Z.get(confidence, _CONFIDENCE_Z[confidence_default])

    # Horizon-correct drift and return dispersion (see metrics module docstring).
    expected_return = finite_cell(row, "expected_return_kalman")
    if expected_return is None:
        return empty_figure(f"No Kalman expected return for {name}")
    annual_return = expected_return / PRICE_TARGET_HORIZON_YEARS
    sigma = float(quantile_return_volatility(
        finite_cell(row, "er_p05", np.nan), finite_cell(row, "er_p95", np.nan)
    ))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(finite_cell(row, "er_sd", np.nan))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.01

    today = pd.Timestamp.now().normalize()
    forecast_end = today + timedelta(days=_HORIZON_DAYS.get(horizon, 365))
    history_start = today - timedelta(days=_HISTORY_DAYS)
    # The simulated fan spans today -> horizon; the pre-today segment is the
    # real price ladder (below), so the fan no longer carries a flat prefix.
    time_points = pd.date_range(start=today, end=forecast_end, freq="D")
    days_forward = (time_points - today).days.to_numpy()

    logger.debug("Building forecast for %s: %d points, sigma=%.4f", name, len(time_points), sigma)

    mean_price = _forecast_mean(spot, annual_return, days_forward)

    fig = go.Figure()

    # --- Real price history (lookback ladder), anchored at today's spot -----
    hist = history_ladder(row, "price", PRICE_SUFFIXES, today_value=spot, asof=today)
    hist = hist[hist["date"] >= history_start]
    if len(hist) < 2:
        # All ladder columns NaN inside the window: fall back to the flat
        # anchor segment the panel drew before the history columns existed.
        hist = pd.DataFrame({"date": [history_start, today], "value": [spot, spot]})
    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["value"],
            mode="lines+markers",
            name="Price History",
            line=dict(color=_PRICE_COLOR, width=2),
            marker=dict(size=6),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
        )
    )

    # --- HDI confidence band (widens with sqrt(time) from today) ------------
    fwd_years = np.clip(days_forward, 0, None) / 365.0
    half_width = z * sigma * np.sqrt(fwd_years) * mean_price
    upper = mean_price + half_width
    lower = mean_price - half_width
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=upper,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=lower,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor=_BAND_FILL,
            name=f"{confidence}% HDI Band",
            hoverinfo="skip",
        )
    )

    # --- Posterior predictive draws -----------------------------------------
    if show_draws:
        draws = _posterior_draws(spot, annual_return, sigma, days_forward)
        for i, path in enumerate(draws):
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=path,
                    mode="lines",
                    line=dict(color=_DRAW_COLOR, width=0.5),
                    name="Posterior Draws",
                    legendgroup="draws",
                    showlegend=(i == 0),
                    hoverinfo="skip",
                )
            )

    # --- Kalman mean forecast line ------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=mean_price,
            mode="lines",
            name="Kalman Forecast",
            line=dict(color=_FORECAST_COLOR, width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Price Target: %{y:.2f}<extra></extra>",
        )
    )

    # --- Real analyst consensus band (low / median / high, NTM) -------------
    target_low = finite_cell(row, "price_target_low")
    target_high = finite_cell(row, "price_target_high")
    target_median = finite_cell(row, "price_target_median")
    band_end = min(forecast_end, today + timedelta(days=_TARGET_BAND_DAYS))
    if target_low is not None and target_high is not None:
        fig.add_trace(
            go.Scatter(
                x=[today, band_end], y=[target_high, target_high], mode="lines",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[today, band_end], y=[target_low, target_low], mode="lines",
                line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
                fillcolor=_TARGET_BAND_FILL, name="Analyst Target Range",
                hoverinfo="skip",
            )
        )
    if target_median is not None:
        fig.add_trace(
            go.Scatter(
                x=[today, band_end], y=[target_median, target_median], mode="lines",
                name="Analyst Median Target",
                line=dict(color=_TARGET_COLOR, dash="dash", width=1.5),
                hovertemplate="Median Target: %{y:.2f}<extra></extra>",
            )
        )

    # --- Observed analyst target (today) ------------------------------------
    if original_target is not None:
        fig.add_trace(
            go.Scatter(
                x=[today],
                y=[original_target],
                mode="markers",
                name="Observed Target",
                marker=dict(size=9, color=_TARGET_COLOR),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Target: %{y:.2f}<extra></extra>",
            )
        )

    # --- Current price (today) ---------------------------------------
    fig.add_trace(
        go.Scatter(
            x=[today],
            y=[spot],
            mode="markers",
            name="Observed Price",
            marker=dict(size=9, color=_PRICE_COLOR),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
        )
    )

    # --- Upcoming event markers ---------------------------------------------
    for column, label in (
            ("expected_report_date", "Expected Report"),
            ("next_income_statement_report_date", "Next Income Statement Report"),
            ("next_fiscal_quarter", "Next Fiscal Quarter"),
            ("next_earnings", "Next Earnings"),
            ("next_fy_end_date", "FY End"),
    ):
        raw_date = cell(row, column)
        if raw_date is None:
            continue
        event_date = pd.to_datetime(raw_date)
        if today <= event_date <= forecast_end:
            # Draw the event marker as an explicit shape + annotation rather than
            # ``add_vline(..., annotation_text=...)``: on this plotly/pandas build
            # the vline-with-annotation path averages the two datetime x-coords and
            # raises "Addition/subtraction of integers ... with Timestamp".
            fig.add_shape(
                type="line", x0=event_date, x1=event_date,
                yref="paper", y0=0.0, y1=1.0,
                line=dict(color=_EVENT_COLOR, dash="dot"),
            )
            fig.add_annotation(
                x=event_date, yref="paper", y=1.0, yanchor="bottom",
                text=label, showarrow=False, font=dict(color=_EVENT_COLOR),
            )

    fig.update_layout(
        title=f"{name} - Kalman State & Structural Forecast",
        xaxis_title="Date",
        yaxis_title="Price Target",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    # Fixed window so the layout is stable whether the ladder is dense or the
    # flat fallback is drawn.
    fig.update_xaxes(range=[history_start, forecast_end])
    logger.debug("Done building forecast for %s", name)
    return fig


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(name_id, "options"),
        Output(name_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        name_id: Input(name_id, "value"),
        horizon_id: Input(horizon_id, "value"),
        draws_id: Input(draws_id, "value"),
        confidence_id: Input(confidence_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, str, list, str]:
    # Scope the Stock dropdown to the globally-filtered universe, defaulting to
    # the highest-market-cap name whenever the current pick is filtered out.
    df_all = filter_data(get_data(), **kwargs)
    name_opts, name_val = scoped_filter(
        df_all, "name", kwargs.get(name_id), fallback=top_label(df_all, "market_cap")
    )
    kwargs[name_id] = name_val
    try:
        return _update_logic(**kwargs), "", name_opts, name_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return empty_figure("An error occurred"), msg, name_opts, name_val