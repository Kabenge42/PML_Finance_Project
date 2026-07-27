"""Kelly Criterion Position Sizing.

Bar of normalised allocations by stock plus a top-stocks table. Ported from
``feature_factory/kellyk_optimizer.pyi``.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import coalesce, empty_figure, scoped_filter, sector_values
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema, tbl
from ..theme import GRAPH_STYLE, control
from ..theme import card as theme_card

component_id = "kelly_criterion_position_sizing"

kelly_multiplier_id = f"{component_id}_kelly_multiplier"
kelly_multiplier_options = [
    {"label": "Quarter-Kelly (0.25)", "value": 0.25},
    {"label": "Half-Kelly (0.5)", "value": 0.5},
    {"label": "Three-Quarter-Kelly (0.75)", "value": 0.75},
    {"label": "Full-Kelly (1.0)", "value": 1.0},
]
kelly_multiplier_default = 0.5

max_position_size_id = f"{component_id}_max_position_size"
# "No Limit" == cap at 100%; since adjusted Kelly fractions are <= 1.0 the
# upper clip never binds, leaving allocations uncapped.
max_position_size_options = [
    {"label": "5%", "value": 0.05},
    {"label": "10%", "value": 0.10},
    {"label": "15%", "value": 0.15},
    {"label": "20%", "value": 0.20},
    {"label": "25%", "value": 0.25},
    {"label": "No Limit", "value": 1.0},
]
max_position_size_default = 0.10

min_prob_id = f"{component_id}_min_prob"
# "No Limit" == 0% floor; ``p_upside_pos_cond >= 0`` admits every name (rows with a
# zero/NaN probability still drop out at the positive-Kelly-fraction filter).
min_prob_options = [
    {"label": "60%", "value": 0.6},
    {"label": "70%", "value": 0.7},
    {"label": "80%", "value": 0.8},
    {"label": "90%", "value": 0.9},
    {"label": "No Limit", "value": 0.0},
]
min_prob_default = 0.7

top_n_id = f"{component_id}_top_n"
# Sentinel value for the "No Limit" Top-N option (keep every qualifying name).
top_n_no_limit = "all"
top_n_options = [
    {"label": "Top 10", "value": 10},
    {"label": "Top 20", "value": 20},
    {"label": "Top 30", "value": 30},
    {"label": "Top 50", "value": 50},
    {"label": "No Limit", "value": top_n_no_limit},
]
top_n_default = 20

sector_filter_id = f"{component_id}_sector_filter"

min_market_cap_id = f"{component_id}_min_market_cap"
min_market_cap_options = [
    {"label": "$1B", "value": 1000},
    {"label": "$5B", "value": 5000},
    {"label": "$10B", "value": 10000},
    {"label": "$50B", "value": 50000},
]
min_market_cap_default = 5000

# Optional CVaR-aware sizing. When a Top-N is chosen, the Kelly universe is
# restricted to the CVaR optimiser's allocated longs — names with a non-zero
# ``cvar_book_weight`` in ``analytics.kalman_filtered_price_targets`` — and the
# chosen Top-N of reward-to-CVaR names governs the book (a 100%-gross long book,
# still subject to the Min Win Probability and Max Position Size controls).
# Defaults to "Off" so the chart is unchanged until the user opts in.
cvar_sizing_id = f"{component_id}_cvar_sizing"
cvar_sizing_off = "off"
cvar_sizing_options = [
    {"label": "Off", "value": cvar_sizing_off},
    {"label": "Top 10", "value": 10},
    {"label": "Top 20", "value": 20},
    {"label": "Top 30", "value": 30},
    {"label": "Top 50", "value": 50},
]
cvar_sizing_default = cvar_sizing_off

title = "Kelly Criterion Position Sizing"
description = (
    "Optimal portfolio allocation using Kelly criterion to maximize long-term "
    "growth while managing risk. Kelly fraction shows the percentage of "
    "portfolio to allocate to each stock."
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
                    control("Kelly Multiplier:", dcc.Dropdown(
                        id=kelly_multiplier_id, options=kelly_multiplier_options,
                        value=kelly_multiplier_default, searchable=False, style={"minWidth": "200px"})),
                    control("Max Position Size:", dcc.Dropdown(
                        id=max_position_size_id, options=max_position_size_options,
                        value=max_position_size_default, searchable=False, style={"minWidth": "160px"})),
                    control("Min Win Probability:", dcc.Dropdown(
                        id=min_prob_id, options=min_prob_options,
                        value=min_prob_default, searchable=False, style={"minWidth": "160px"})),
                    control("Top N Stocks:", dcc.Dropdown(
                        id=top_n_id, options=top_n_options,
                        value=top_n_default, searchable=False, style={"minWidth": "140px"})),
                    control("CVaR-aware sizing:", dcc.Dropdown(
                        id=cvar_sizing_id, options=cvar_sizing_options,
                        value=cvar_sizing_default, searchable=False, clearable=False,
                        style={"minWidth": "170px"})),
                    control("Sector:", dcc.Dropdown(
                        id=sector_filter_id, options=sector_opts,
                        value=[s["value"] for s in sector_opts], multi=True, style={"minWidth": "200px"})),
                    control("Min Market Cap ($M):", dcc.Dropdown(
                        id=min_market_cap_id, options=min_market_cap_options,
                        value=min_market_cap_default, searchable=False, style={"minWidth": "160px"})),
                ],
            ),
            dcc.Loading(type="circle", children=[dcc.Graph(id=f"{component_id}_graph", style=GRAPH_STYLE)]),
            html.Pre(id=f"{component_id}_error", className="geib-error"),
            html.Div(id=f"{component_id}_table", style={"margin": "20px 0"}),
        ],
    )


def _calculate_kelly_fraction(row) -> float:
    p = row["p_upside_pos_cond"]
    q = 1 - p
    if p <= 0 or p >= 1:
        return 0.0
    # cvar_5pct_kalman is stored as a decimal return, same as expected_return_kalman.
    cvar = row["cvar_5pct_kalman"]
    expected_return = row["expected_return_kalman"]
    if cvar == 0 or expected_return <= 0:
        return 0.0
    b = expected_return / abs(cvar)
    if b == 0:
        return 0.0
    return max(0.0, (p * b - q) / b)


def _create_table(df: pd.DataFrame) -> html.Div:
    cols = ["ticker", "name", "allocation_pct", "expected_return_kalman",
            "cvar_5pct_kalman", "p_upside_pos_cond"]
    headers = ["Ticker", "Name", "Allocation (%)", "Expected Return (%)",
               "CVaR 5% (%)", "Win Probability"]
    view = df[cols].copy()
    view["expected_return_kalman"] = (view["expected_return_kalman"] * 100).round(2)
    view["cvar_5pct_kalman"] = (view["cvar_5pct_kalman"] * 100).round(2)
    view["p_upside_pos_cond"] = (view["p_upside_pos_cond"] * 100).round(2)
    view["allocation_pct"] = view["allocation_pct"].round(2)

    body = [
        html.Tr([html.Td(str(view.iloc[i][c])) for c in cols])
        for i in range(len(view))
    ]
    return html.Div([
        html.H5("Top Stocks by Kelly Fraction", style={"color": "#ffffff"}),
        html.Table(
            [html.Thead(html.Tr([html.Th(h) for h in headers])), html.Tbody(body)],
            className="geib-table",
        ),
    ])


def _update_logic(**kwargs) -> Tuple[go.Figure, html.Div]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return empty_figure("No data is available to display"), html.Div()

    df = df[["ticker", "name", "sector", "market_cap", "p_upside_pos_cond",
             "expected_return_kalman", "cvar_5pct_kalman",
             "cvar_book_weight", "reward_to_cvar"]].copy()
    logger.debug(schema(df))

    kelly_multiplier = coalesce(kwargs.get(kelly_multiplier_id), kelly_multiplier_default)
    max_position_size = coalesce(kwargs.get(max_position_size_id), max_position_size_default)
    min_prob = coalesce(kwargs.get(min_prob_id), min_prob_default)
    top_n = coalesce(kwargs.get(top_n_id), top_n_default)
    sector_filter = kwargs.get(sector_filter_id)
    if not sector_filter:
        sector_filter = df["sector"].dropna().unique().tolist()
    min_market_cap = coalesce(kwargs.get(min_market_cap_id), min_market_cap_default)

    # CVaR-aware sizing (optional): restrict to the CVaR optimiser's allocated
    # longs (non-zero ``cvar_book_weight``) and let the chosen Top-N of
    # reward-to-CVaR names govern the book. The Min Win Probability / Max
    # Position Size caps and the 100%-gross normalisation below still apply.
    cvar_sizing = kwargs.get(cvar_sizing_id, cvar_sizing_default)
    cvar_aware = cvar_sizing not in (None, cvar_sizing_off)
    if cvar_aware:
        df = df[df["cvar_book_weight"].fillna(0.0) > 0]
        if len(df) == 0:
            return empty_figure("No names carry a non-zero CVaR book weight"), html.Div()
        top_n = int(cvar_sizing)

    df = df[df["sector"].isin(sector_filter)]
    df = df[df["market_cap"] >= min_market_cap]
    df = df[df["p_upside_pos_cond"] >= min_prob]
    if len(df) == 0:
        return empty_figure("No stocks meet the filtering criteria"), html.Div()

    df["kelly_fraction"] = df.apply(_calculate_kelly_fraction, axis=1)
    df = df[df["kelly_fraction"] > 0]
    if len(df) == 0:
        return empty_figure("No stocks have positive Kelly fraction"), html.Div()

    rank_col = "reward_to_cvar" if cvar_aware else "kelly_fraction"
    df = df.sort_values(rank_col, ascending=False)
    if top_n != top_n_no_limit:
        df = df.head(int(top_n))
    df["kelly_fraction_adjusted"] = df["kelly_fraction"] * kelly_multiplier
    df["kelly_fraction_capped"] = df["kelly_fraction_adjusted"].clip(upper=max_position_size)

    total = df["kelly_fraction_capped"].sum()
    df["allocation_pct"] = (df["kelly_fraction_capped"] / total) * 100 if total > 0 else 0
    logger.debug(tbl(df))

    fig = px.bar(
        df, x="name", y="allocation_pct", color="sector",
        labels={"allocation_pct": "Allocation (%)", "name": "Stock", "sector": "Sector"},
        hover_data={"allocation_pct": ":.2f", "sector": True, "ticker": True},
    )
    fig.update_layout(xaxis_title="Stock", yaxis_title="Allocation (%)",
                      hovermode="x unified", height=550)
    fig.update_xaxes(tickangle=-45)
    return fig, _create_table(df)


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_table", "children"),
        Output(f"{component_id}_error", "children"),
        Output(sector_filter_id, "options"),
        Output(sector_filter_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        kelly_multiplier_id: Input(kelly_multiplier_id, "value"),
        max_position_size_id: Input(max_position_size_id, "value"),
        min_prob_id: Input(min_prob_id, "value"),
        top_n_id: Input(top_n_id, "value"),
        cvar_sizing_id: Input(cvar_sizing_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        min_market_cap_id: Input(min_market_cap_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, html.Div, str, list, list]:
    # Scope the local Sector filter to the globally-filtered universe so it never
    # offers (or stays set to) a sector the global filters have removed.
    df_all = filter_data(get_data(), **kwargs)
    sector_opts, sector_val = scoped_filter(
        df_all, "sector", kwargs.get(sector_filter_id), multi=True
    )
    kwargs[sector_filter_id] = sector_val
    try:
        fig, table = _update_logic(**kwargs)
        return fig, table, "", sector_opts, sector_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return empty_figure("An error occurred"), html.Div(), msg, sector_opts, sector_val