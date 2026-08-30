"""Price Target Convergence Over Time.

A real dated timeline built from the ``price_*_ago`` / ``price_target_*_ago``
lookback ladders in ``analytics.kalman_filtered_price_targets``: the realized
price track and the analyst consensus target track converge to today's spot
(``original_price``), the current consensus (``original_target``) and the
Kalman-filtered target (``price_target_kalman``), with the analyst
low / median / high dispersion drawn at the snapshot.

Two views:

* **Single Name** — the raw price / target ladders for one stock.
* **Universe Average** — every name indexed to spot = 100 first (raw prices are
  not cross-sectionally comparable), then averaged per ladder column.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import (
    coalesce,
    empty_figure,
    finite_cell,
    name_options,
    scoped_filter,
    sector_values,
    top_label,
)
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema, tbl
from ..metrics import PRICE_SUFFIXES, PRICE_TARGET_SUFFIXES, history_ladder
from ..theme import SERIES_COLORS, GOLD, GRAPH_STYLE, SUBTLE_TEXT, control
from ..theme import card as theme_card

component_id = "price_target_convergence"

view_control_id = f"{component_id}_view"
view_options = [
    {"label": "Single Name", "value": "single"},
    {"label": "Universe Average", "value": "aggregate"},
]
view_default = "single"

name_control_id = f"{component_id}_name"

sector_control_id = f"{component_id}_sector"
sector_default = "All"

# Trace palette (semantic, shared with the structural-forecast panel).
# From `theme.SERIES_COLORS` -- kalman_structural_forecast plots the same two
# entities, and both files used to carry their own copy of these literals.
_PRICE_COLOR = SERIES_COLORS["price"]
_TARGET_COLOR = SERIES_COLORS["target"]
_KALMAN_COLOR = GOLD  # cyan accent Kalman target

title = "Price Target Convergence Over Time"
description = (
    "Real price and analyst price-target history from the lookback ladders, "
    "converging to today's spot, the current consensus and the Kalman-filtered "
    "target"
)


def component() -> "object":
    graph_id = f"{component_id}_graph"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    df = get_data()
    sector_options = [{"label": "All", "value": "All"}]
    sector_options.extend({"label": str(s), "value": s} for s in sector_values(df))
    name_opts, name_default = name_options(df)

    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                className="geib-controls-row",
                children=[
                    control(
                        "View:",
                        dcc.Dropdown(
                            id=view_control_id,
                            options=view_options,
                            value=view_default,
                            searchable=False,
                            clearable=False,
                            style={"minWidth": "170px"},
                        ),
                    ),
                    control(
                        "Stock:",
                        dcc.Dropdown(
                            id=name_control_id,
                            options=name_opts,
                            value=name_default,
                            searchable=True,
                            style={"minWidth": "220px"},
                        ),
                    ),
                    control(
                        "Sector:",
                        dcc.Dropdown(
                            id=sector_control_id,
                            options=sector_options,
                            value=sector_default,
                            style={"minWidth": "200px"},
                        ),
                    ),
                ],
            ),
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[dcc.Graph(id=graph_id, style=GRAPH_STYLE)],
            ),
            html.Pre(id=error_id, className="geib-error"),
        ],
    )


def _ladder_trace(ladder: pd.DataFrame, name: str, color: str) -> go.Scatter:
    """Dated ladder polyline; a single surviving point still renders a marker."""
    return go.Scatter(
        x=ladder["date"],
        y=ladder["value"],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=2),
        marker=dict(size=7),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra>" + name + "</extra>",
    )


def _indexed_mean_row(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Cross-sectional mean of each ladder column, indexed to spot = 100.

    Raw prices are not comparable across names, so each column is divided by
    the name's ``original_price`` first; the resulting Series keeps the original
    column names so it feeds straight into
    :func:`~dashboards.geib.metrics.history_ladder`.
    """
    values = {
        col: (df[col] / df["original_price"] * 100.0).mean(skipna=True)
        for col in columns
        if col in df.columns
    }
    return pd.Series(values)


def _single_figure(df: pd.DataFrame, name: str, today: pd.Timestamp) -> go.Figure:
    df = df[df["name"] == name]
    if len(df) == 0:
        return empty_figure(f"No data available for {name}")
    if "income_statement_report_date" in df.columns:
        df = df.sort_values("income_statement_report_date")
    row = df.iloc[-1]

    # Scalar-safe row reads: NaN default keeps the np.isfinite guards below.
    spot = finite_cell(row, "original_price", np.nan)
    target = finite_cell(row, "original_target", np.nan)

    price_hist = history_ladder(
        row, "price", PRICE_SUFFIXES,
        today_value=spot if np.isfinite(spot) else None, asof=today,
    )
    pt_hist = history_ladder(
        row, "price_target", PRICE_TARGET_SUFFIXES,
        today_value=target if np.isfinite(target) else None, asof=today,
    )
    if len(price_hist) == 0 and len(pt_hist) == 0:
        return empty_figure(f"No price/target history available for {name}")
    logger.debug(tbl(price_hist))

    fig = go.Figure()
    if len(price_hist):
        fig.add_trace(_ladder_trace(price_hist, "Price", _PRICE_COLOR))
    if len(pt_hist):
        fig.add_trace(_ladder_trace(pt_hist, "Analyst Target", _TARGET_COLOR))

    # Analyst dispersion at the snapshot (low - high segment + median marker).
    low = finite_cell(row, "price_target_low", np.nan)
    high = finite_cell(row, "price_target_high", np.nan)
    median = finite_cell(row, "price_target_median", np.nan)
    if np.isfinite(low) and np.isfinite(high):
        fig.add_trace(go.Scatter(
            x=[today, today], y=[low, high], mode="lines",
            name="Analyst Target Range",
            line=dict(color=_TARGET_COLOR, width=4), opacity=0.4,
            hovertemplate="Range: %{y:,.2f}<extra>Analyst Target Range</extra>",
        ))
    if np.isfinite(median):
        fig.add_trace(go.Scatter(
            x=[today], y=[median], mode="markers", name="Median Target",
            marker=dict(size=9, color=_TARGET_COLOR, symbol="line-ew-open",
                        line=dict(width=2)),
            hovertemplate="Median Target: %{y:,.2f}<extra></extra>",
        ))

    kalman = finite_cell(row, "price_target_kalman", np.nan)
    if np.isfinite(kalman):
        if np.isfinite(spot):
            # Dotted connector making the spot -> Kalman-target gap explicit.
            fig.add_trace(go.Scatter(
                x=[today, today], y=[spot, kalman], mode="lines",
                line=dict(color=SUBTLE_TEXT, dash="dot", width=1),
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[today], y=[kalman], mode="markers", name="Kalman Target (today)",
            marker=dict(size=11, color=_KALMAN_COLOR, symbol="diamond"),
            hovertemplate="Kalman Target: %{y:,.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=f"{name} - Price vs Analyst Target Convergence",
        yaxis_title="Price",
        xaxis_title="Date",
        hovermode="x unified",
        height=550,
    )
    return fig


def _aggregate_figure(df: pd.DataFrame, today: pd.Timestamp) -> go.Figure:
    usable = df[df["original_price"] > 0]
    if len(usable) == 0:
        return empty_figure("No price/target history available for the current selection")

    price_row = _indexed_mean_row(usable, [f"price_{s}_ago" for s in PRICE_SUFFIXES])
    pt_row = _indexed_mean_row(
        usable, [f"price_target_{s}_ago" for s in PRICE_TARGET_SUFFIXES]
    )
    target_today = (usable["original_target"] / usable["original_price"] * 100.0).mean(skipna=True)
    kalman_today = (
        usable["price_target_kalman"] / usable["original_price"] * 100.0
    ).mean(skipna=True)

    price_hist = history_ladder(price_row, "price", PRICE_SUFFIXES,
                                today_value=100.0, asof=today)
    pt_hist = history_ladder(
        pt_row, "price_target", PRICE_TARGET_SUFFIXES,
        today_value=float(target_today) if np.isfinite(target_today) else None,
        asof=today,
    )
    if len(price_hist) == 0 and len(pt_hist) == 0:
        return empty_figure("No price/target history available for the current selection")

    fig = go.Figure()
    if len(price_hist):
        fig.add_trace(_ladder_trace(price_hist, "Avg Price (Indexed)", _PRICE_COLOR))
    if len(pt_hist):
        fig.add_trace(_ladder_trace(pt_hist, "Avg Analyst Target (Indexed)", _TARGET_COLOR))
    if np.isfinite(kalman_today):
        fig.add_trace(go.Scatter(
            x=[today], y=[float(kalman_today)], mode="markers",
            name="Avg Kalman Target (today)",
            marker=dict(size=11, color=_KALMAN_COLOR, symbol="diamond"),
            hovertemplate="Avg Kalman Target: %{y:,.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Universe Average ({len(usable):,} names) - Convergence, Indexed to Spot = 100",
        yaxis_title="Indexed (Spot = 100)",
        xaxis_title="Date",
        hovermode="x unified",
        height=550,
    )
    return fig


def _update_logic(**kwargs) -> go.Figure:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return empty_figure("No data is available to display")
    logger.debug(schema(df))

    view = coalesce(kwargs.get(view_control_id), view_default)
    sector_value = kwargs.get(sector_control_id) or sector_default
    if sector_value != "All":
        df = df[df["sector"] == sector_value]
    if len(df) == 0:
        return empty_figure("No data is available for the selected sector")

    today = pd.Timestamp.now().normalize()
    if view == "aggregate":
        return _aggregate_figure(df, today)
    name = kwargs.get(name_control_id)
    if not name:
        return empty_figure("Select a stock to display its convergence history")
    return _single_figure(df, name, today)


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_error", "children"),
        Output(sector_control_id, "options"),
        Output(sector_control_id, "value"),
        Output(name_control_id, "options"),
        Output(name_control_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        view_control_id: Input(view_control_id, "value"),
        name_control_id: Input(name_control_id, "value"),
        sector_control_id: Input(sector_control_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, str, list, str, list, str]:
    # Scope the local Sector filter (with its "All" sentinel) to the
    # globally-filtered universe, then the Stock picker to the sector-narrowed
    # frame so it only offers names the chart can actually draw.
    df_all = filter_data(get_data(), **kwargs)
    sector_opts, sector_val = scoped_filter(
        df_all, "sector", kwargs.get(sector_control_id), include_all=True, all_label="All"
    )
    kwargs[sector_control_id] = sector_val
    df_scoped = df_all
    if sector_val != "All" and df_all is not None and "sector" in df_all.columns:
        df_scoped = df_all[df_all["sector"] == sector_val]
    name_opts, name_val = scoped_filter(
        df_scoped, "name", kwargs.get(name_control_id),
        fallback=top_label(df_scoped, "market_cap"),
    )
    kwargs[name_control_id] = name_val
    try:
        return _update_logic(**kwargs), "", sector_opts, sector_val, name_opts, name_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return empty_figure("An error occurred"), msg, sector_opts, sector_val, name_opts, name_val
