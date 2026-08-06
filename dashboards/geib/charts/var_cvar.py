"""Value at Risk (VaR): Downside Risk Assessment.

Grouped VaR bars at the 90% / 95% / 99% confidence levels plus a per-name VaR
analysis table (expected dollar loss for a configurable position size).

All VaR quantities are derived in decimal return space from the exported MC
forward-return distribution and scaled to percent only at the figure / table
boundary (project unit contract, CHANGELOG 0.9.9.7):

* **Parametric** — ``VaR_c = -z_c * sigma`` with ``sigma`` implied by the
  ``er_p05`` / ``er_p95`` posterior return spread (see
  ``quantile_return_volatility``; ``expected_vol_kalman`` is deliberately NOT
  used — it is the std of the posterior *expected-upside* draws, i.e.
  parameter/estimation uncertainty, not forward asset volatility).
* **Historical / Monte Carlo** — anchored on the exported MC 5% quantile:
  ``VaR_95 = min(er_p05, 0)``; the 90% / 99% levels are scaled by the normal
  z-ratios (1.28 / 1.645 and 2.33 / 1.645) because only the 5% quantile is
  exported.

All levels are clipped at zero so a name whose entire return distribution sits
positive shows zero loss rather than a "negative loss".

NOTE: the table's CVaR column is the normal-approximation expected shortfall
``er_mean - 2.0627 * sigma`` — NOT ``cvar_5pct_kalman``, which is the tail mean
of the posterior *upside* draws (estimation uncertainty of the mean, the STARR
denominator input; stored in decimal return units), so it is frequently
positive and cannot be reported as a return-space CVaR.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

from ._common import empty_figure, scoped_filter, sector_values
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..components.probability_filter import (
    apply_probability_filter,
    probability_controls,
    probability_inputs,
    register as register_probability_filter,
)
from ..data import get_data
from ..logger import logger, schema, tbl
from ..metrics import quantile_return_volatility
from ..theme import BACKGROUND_CONTENT, BODY_TEXT, BORDER, COLORWAY, DUAL_GRAPH_STYLE, GOLD, NAVY
from ..theme import card as theme_card
from ..theme import control

component_id = "value_at_risk_downside_assessment"

confidence_level_id = f"{component_id}_confidence_level"
confidence_level_options = [
    {"label": "90%", "value": "90"},
    {"label": "95%", "value": "95"},
    {"label": "99%", "value": "99"},
]
confidence_level_default = "95"

position_size_id = f"{component_id}_position_size"
position_size_options = [
    {"label": "1,000 shares", "value": "1000"},
    {"label": "5,000 shares", "value": "5000"},
    {"label": "10,000 shares", "value": "10000"},
    {"label": "50,000 shares", "value": "50000"},
]
position_size_default = "10000"

top_n_id = f"{component_id}_top_n"
top_n_options = [
    {"label": "Top 10", "value": "10"},
    {"label": "Top 20", "value": "20"},
    {"label": "Top 30", "value": "30"},
    {"label": "Top 50", "value": "50"},
]
top_n_default = "20"

var_method_id = f"{component_id}_var_method"
var_method_options = [
    {"label": "Historical", "value": "historical"},
    {"label": "Parametric", "value": "parametric"},
    {"label": "Monte Carlo", "value": "monte_carlo"},
]
var_method_default = "parametric"

sector_filter_id = f"{component_id}_sector_filter"

# Probability metric + band (shared control pair). Unlike the other cards this
# one carries no pre-existing probability gate, so the band opens at the metric's
# full width: the control is opt-in and the card's default universe is unchanged.
# A downside-risk view should not silently hide names behind a probability floor.
min_prob_default = 0.0
register_probability_filter(component_id)

# One-sided standard-normal quantiles per confidence level, so
# ``VaR_c = -z_c * sigma`` (parametric) and the Historical / Monte-Carlo levels
# scale off the exported 5% quantile by z-ratio.
_Z_SCORES = {"90": 1.28, "95": 1.645, "99": 2.33}
_CONFIDENCE_LEVELS = ("90", "95", "99")

# Expected-shortfall multiplier ``phi(Phi^-1(a)) / a`` of a standard normal at
# a = 5%, so the table's ``CVaR = er_mean - factor * sigma`` (the mean return
# conditional on landing in the worst 5% of outcomes).
_ES_FACTOR_5PCT = 2.0627128

# Names must carry at least this many covering analysts to enter the card —
# thin coverage makes the consensus-derived return distribution unreliable.
_MIN_ANALYSTS = 5

title = "Value at Risk (VaR): Downside Risk Assessment"
description = (
    "Calculate the maximum expected loss at different confidence levels to "
    "understand downside risk exposure. Compare VaR across stocks to identify "
    "positions with the most tail risk."
)


def _table_columns() -> list[dict]:
    pct = Format(precision=2, scheme=Scheme.fixed)
    money = Format(precision=2, scheme=Scheme.fixed).group(True)
    return [
        {"name": "Stock Name", "id": "name"},
        {"name": "Current Price", "id": "original_price", "type": "numeric", "format": money},
        {"name": "VaR (95%)", "id": "var_95_pct", "type": "numeric", "format": pct},
        {"name": "VaR (99%)", "id": "var_99_pct", "type": "numeric", "format": pct},
        {"name": "CVaR", "id": "cvar_pct", "type": "numeric", "format": pct},
        {"name": "Expected Loss ($)", "id": "expected_loss_dollars", "type": "numeric",
         "format": Format(precision=0, scheme=Scheme.fixed).group(True)},
        {"name": "Prob. Positive Return", "id": "prob_pos_pct", "type": "numeric", "format": pct},
    ]


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
                    control("Confidence Level:", dcc.Dropdown(
                        id=confidence_level_id, options=confidence_level_options,
                        value=confidence_level_default, searchable=False, clearable=False,
                        style={"minWidth": "150px"})),
                    control("Position Size:", dcc.Dropdown(
                        id=position_size_id, options=position_size_options,
                        value=position_size_default, searchable=False, clearable=False,
                        style={"minWidth": "170px"})),
                    control("Top N by VaR:", dcc.Dropdown(
                        id=top_n_id, options=top_n_options,
                        value=top_n_default, searchable=False, clearable=False,
                        style={"minWidth": "140px"})),
                    control("VaR Method:", dcc.Dropdown(
                        id=var_method_id, options=var_method_options,
                        value=var_method_default, searchable=False, clearable=False,
                        style={"minWidth": "170px"})),
                    *probability_controls(component_id, lo=min_prob_default),
                    control("Sectors:", dcc.Dropdown(
                        id=sector_filter_id, options=sector_opts, value=[], multi=True,
                        placeholder="All Sectors", style={"minWidth": "200px"})),
                ],
            ),
            html.Label("VaR by Confidence Level", className="geib-graph-label"),
            dcc.Loading(type="circle", children=[
                dcc.Graph(id=f"{component_id}_graph", style=DUAL_GRAPH_STYLE)]),
            html.Label("VaR Analysis Table", className="geib-graph-label"),
            dcc.Loading(type="circle", children=[
                dash_table.DataTable(
                    id=f"{component_id}_table",
                    columns=_table_columns(),
                    data=[],
                    sort_action="native",
                    page_size=25,
                    style_as_list_view=True,
                    style_table={"overflowX": "auto", "width": "100%"},
                    style_header={
                        "backgroundColor": NAVY,
                        "color": "#FFFFFF",
                        "fontWeight": "bold",
                        "borderBottom": f"2px solid {GOLD}",
                        "fontFamily": "monospace",
                    },
                    style_cell={
                        "backgroundColor": BACKGROUND_CONTENT,
                        "color": BODY_TEXT,
                        "fontFamily": "monospace",
                        "fontSize": "12px",
                        "padding": "8px",
                        "borderBottom": f"1px solid {BORDER}",
                        "textAlign": "left",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": col["id"]}, "textAlign": "right"}
                        for col in _table_columns() if col.get("type") == "numeric"
                    ],
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": NAVY},
                    ],
                )
            ]),
            html.Pre(id=f"{component_id}_error", className="geib-error"),
        ],
    )


def _compute_var_frame(df: pd.DataFrame, var_method: str) -> pd.DataFrame:
    """Attach ``var_90`` / ``var_95`` / ``var_99`` + ``cvar`` (decimal, <= 0 loss).

    Every level is clamped to ``[-1, 0]``: a long equity position cannot lose
    more than 100%, and names with degenerate ``er_p05``/``er_p95`` spreads
    would otherwise dominate the worst-VaR sort with impossible losses.
    """
    sigma = quantile_return_volatility(df["er_p05"], df["er_p95"])
    if var_method == "parametric":
        var_95_raw = (-_Z_SCORES["95"] * sigma).clip(upper=0.0)
    else:  # historical / monte_carlo: anchor on the exported MC 5% quantile
        var_95_raw = df["er_p05"].clip(upper=0.0)
    for level in _CONFIDENCE_LEVELS:
        df[f"var_{level}"] = (
            var_95_raw * (_Z_SCORES[level] / _Z_SCORES["95"])
        ).clip(lower=-1.0)
    # Unclamped sort key so severity ordering survives the -100% saturation.
    df["var_95_raw"] = var_95_raw
    # Table CVaR: normal-approx 5% expected shortfall of the return
    # distribution (see module docstring for why not ``cvar_5pct_kalman``).
    df["cvar"] = (df["er_mean"] - _ES_FACTOR_5PCT * sigma).clip(lower=-1.0)
    return df


def _update_logic(**kwargs) -> Tuple[go.Figure, list[dict]]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return empty_figure("No data is available to display"), []

    # Gate on the selected probability metric *before* the projection below —
    # three of the four selectable metrics are not in that column list.
    df = apply_probability_filter(df, component_id, kwargs)

    df = df[["name", "sector", "original_price", "er_mean", "er_p05", "er_p95",
             "mc_prob_pos", "n_analysts", "market_cap"]].copy()
    logger.debug(schema(df))

    confidence_level = str(kwargs.get(confidence_level_id) or confidence_level_default)
    if confidence_level not in _Z_SCORES:
        confidence_level = confidence_level_default
    position_size = float(kwargs.get(position_size_id) or position_size_default)
    top_n = int(kwargs.get(top_n_id) or top_n_default)
    var_method = kwargs.get(var_method_id) or var_method_default
    sector_filter = kwargs.get(sector_filter_id) or []

    df = df[df["n_analysts"].fillna(0) >= _MIN_ANALYSTS]
    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]
    df = df.dropna(subset=["er_p05", "er_p95"])
    if len(df) == 0:
        return empty_figure("No stocks match the selected criteria"), []

    df = _compute_var_frame(df, var_method)
    df["expected_loss_dollars"] = (
        df[f"var_{confidence_level}"].abs() * df["original_price"] * position_size
    )

    # Worst tail loss first (signed return, loss = negative); the unclamped
    # key keeps ordering meaningful among names saturated at -100%.
    df = df.sort_values("var_95_raw", ascending=True).head(top_n)
    logger.debug(tbl(df))

    fig = go.Figure()
    for color, level in zip(COLORWAY, _CONFIDENCE_LEVELS):
        fig.add_trace(go.Bar(
            x=df["name"],
            y=df[f"var_{level}"] * 100.0,
            name=f"{level}%",
            marker_color=color,
            hovertemplate="%{x}<br>VaR (" + level + "%): %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        hovermode="x unified",
        legend_title_text="Confidence Level",
    )
    fig.update_xaxes(title_text="Company", tickangle=-45)
    fig.update_yaxes(title_text="Value at Risk (% Loss)")

    table_df = df.assign(
        var_95_pct=df["var_95"] * 100.0,
        var_99_pct=df["var_99"] * 100.0,
        cvar_pct=df["cvar"] * 100.0,
        prob_pos_pct=df["mc_prob_pos"] * 100.0,
    )
    table_cols = ["name", "original_price", "var_95_pct", "var_99_pct", "cvar_pct",
                  "expected_loss_dollars", "prob_pos_pct"]
    return fig, table_df[table_cols].to_dict("records")


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_table", "data"),
        Output(f"{component_id}_error", "children"),
        Output(sector_filter_id, "options"),
        Output(sector_filter_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        confidence_level_id: Input(confidence_level_id, "value"),
        position_size_id: Input(position_size_id, "value"),
        top_n_id: Input(top_n_id, "value"),
        var_method_id: Input(var_method_id, "value"),
        **probability_inputs(component_id),
        sector_filter_id: Input(sector_filter_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, list, str, list, list]:
    # Scope the local Sectors filter to the globally-filtered universe.
    df_all = filter_data(get_data(), **kwargs)
    sector_opts, sector_val = scoped_filter(
        df_all, "sector", kwargs.get(sector_filter_id), multi=True
    )
    kwargs[sector_filter_id] = sector_val
    try:
        fig, table_data = _update_logic(**kwargs)
        return fig, table_data, "", sector_opts, sector_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return empty_figure("An error occurred"), [], msg, sector_opts, sector_val
