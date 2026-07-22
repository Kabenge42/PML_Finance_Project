"""Value at Risk (VaR): Downside Risk Assessment.

CVaR bar by stock/sector + a risk-return scatter coloured by reward-to-CVaR.
Ported from ``feature_factory/var.pyi``.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import empty_figure, scoped_filter, sector_values
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema, tbl
from ..metrics import quantile_return_volatility
from ..theme import DUAL_GRAPH_STYLE, control
from ..theme import card as theme_card

component_id = "value_at_risk_downside_assessment"

confidence_level_id = f"{component_id}_confidence_level"
confidence_level_options = [
    {"label": "5%", "value": "5"},
    {"label": "10%", "value": "10"},
    {"label": "25%", "value": "25"},
    {"label": "50%", "value": "50"},
]
confidence_level_default = "5"

# Expected-shortfall multipliers ``phi(Phi^-1(a)) / a`` of a standard normal at
# each selectable tail probability *a*, so ``CVaR_a = mu - factor * sigma`` (the
# mean return conditional on landing in the worst a% of outcomes). The pipeline
# does not export return-space CVaR at any level, so every level is derived
# under this normal approximation of the MC return distribution.
_ES_FACTORS = {
    "5": 2.0627128,
    "10": 1.7550151,
    "25": 1.2711055,
    "50": 0.7978846,
}

sort_metric_id = f"{component_id}_sort_metric"
sort_metric_options = [
    {"label": "Highest Reward-to-CVaR", "value": "reward_to_cvar"},
    {"label": "Least Tail Risk (CVaR)", "value": "cvar_lowest"},
    {"label": "Highest Expected Return", "value": "expected_return"},
]
sort_metric_default = "reward_to_cvar"

min_market_cap_id = f"{component_id}_min_market_cap"
min_market_cap_options = [
    {"label": "1,000M", "value": "1000"},
    {"label": "5,000M", "value": "5000"},
    {"label": "10,000M", "value": "10000"},
    {"label": "50,000M", "value": "50000"},
]
min_market_cap_default = "5000"

min_prob_positive_id = f"{component_id}_min_prob_positive"
min_prob_positive_options = [
    {"label": "50%", "value": "0.5"},
    {"label": "70%", "value": "0.7"},
    {"label": "80%", "value": "0.8"},
    {"label": "90%", "value": "0.9"},
]
min_prob_positive_default = "0.7"

num_stocks_id = f"{component_id}_num_stocks"
num_stocks_options = [
    {"label": "20", "value": "20"},
    {"label": "50", "value": "50"},
    {"label": "100", "value": "100"},
    {"label": "All", "value": "all"},
]
num_stocks_default = "50"

sector_filter_id = f"{component_id}_sector_filter"

title = "Value at Risk (VaR): Downside Risk Assessment"
description = (
    "Calculates the expected loss in the worst outcomes at different tail "
    "probabilities using Conditional Value at Risk (CVaR), expressed as a "
    "signed return — less negative means less downside risk."
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
                    control("Confidence Level:", dcc.Dropdown(
                        id=confidence_level_id, options=confidence_level_options,
                        value=confidence_level_default, searchable=False, style={"minWidth": "160px"})),
                    control("Sort By:", dcc.Dropdown(
                        id=sort_metric_id, options=sort_metric_options,
                        value=sort_metric_default, searchable=False, style={"minWidth": "200px"})),
                    control("Min Market Cap (M):", dcc.Dropdown(
                        id=min_market_cap_id, options=min_market_cap_options,
                        value=min_market_cap_default, searchable=False, style={"minWidth": "160px"})),
                    control("Min Prob Positive:", dcc.Dropdown(
                        id=min_prob_positive_id, options=min_prob_positive_options,
                        value=min_prob_positive_default, searchable=False, style={"minWidth": "160px"})),
                    control("Number of Stocks:", dcc.Dropdown(
                        id=num_stocks_id, options=num_stocks_options,
                        value=num_stocks_default, searchable=False, style={"minWidth": "140px"})),
                    control("Sectors:", dcc.Dropdown(
                        id=sector_filter_id, options=sector_opts, value=[], multi=True,
                        style={"minWidth": "200px"})),
                ],
            ),
            html.Div(
                className="geib-dual-graph",
                children=[
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("CVaR by Stock and Sector", className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_graph_1", style=DUAL_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_error_1", className="geib-error"),
                    ]),
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("Risk-Return Profile", className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_graph_2", style=DUAL_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_error_2", className="geib-error"),
                    ]),
                ],
            ),
        ],
    )


def _update_logic(**kwargs) -> Tuple[go.Figure, go.Figure]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        empty = empty_figure("No data is available to display")
        return empty, empty

    df = df[["name", "sector", "market_cap", "p_upside_pos_cond",
             "expected_return_kalman", "reward_to_cvar", "er_p05", "er_p95"]].copy()
    logger.debug(schema(df))

    confidence_level = str(kwargs.get(confidence_level_id) or confidence_level_default)
    sort_metric = kwargs.get(sort_metric_id) or sort_metric_default
    min_market_cap = float(kwargs.get(min_market_cap_id) or min_market_cap_default)
    min_prob_positive = float(kwargs.get(min_prob_positive_id) or min_prob_positive_default)
    num_stocks = str(kwargs.get(num_stocks_id) or num_stocks_default)
    sector_filter = kwargs.get(sector_filter_id) or []

    df = df[(df["market_cap"] >= min_market_cap) & (df["p_upside_pos_cond"] >= min_prob_positive)]
    if len(df) == 0:
        empty = empty_figure("No stocks match the selected criteria")
        return empty, empty
    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]
    if len(df) == 0:
        empty = empty_figure("No stocks match the selected sectors")
        return empty, empty

    # --- CVaR at the selected tail probability (decimal return space) -------
    # Expected shortfall under a normal approximation of the MC *return*
    # distribution: ``CVaR_a = mu - factor * sigma`` with ``mu`` the Kalman
    # expected return and ``sigma`` implied by the ``er_p05``/``er_p95``
    # posterior return spread (see ``quantile_return_volatility``).
    # NOTE: ``cvar_5pct_kalman`` is NOT this quantity — it is the tail mean of
    # the posterior *upside* draws (estimation uncertainty of the mean, the
    # STARR denominator input), so it is frequently positive and cannot be
    # charted as a return-space CVaR.
    sigma = quantile_return_volatility(df["er_p05"], df["er_p95"])
    factor = _ES_FACTORS.get(confidence_level, _ES_FACTORS[confidence_level_default])
    df["cvar_value"] = df["expected_return_kalman"] - factor * sigma

    if sort_metric == "cvar_lowest":
        # ``cvar_value`` is a signed return (loss = negative): least tail risk
        # first means the highest (least negative) values lead.
        df = df.sort_values("cvar_value", ascending=False)
    elif sort_metric == "expected_return":
        df = df.sort_values("expected_return_kalman", ascending=False)
    else:
        df = df.sort_values("reward_to_cvar", ascending=False)

    if num_stocks != "all":
        df = df.head(int(num_stocks))
    logger.debug(tbl(df))

    fig1 = px.bar(
        df, x="name", y="cvar_value", color="sector",
        title=f"CVaR — Expected Return in the Worst {confidence_level}% of Outcomes",
        labels={"name": "Stock", "cvar_value": f"CVaR ({confidence_level}%)", "sector": "Sector"},
        hover_data={"name": True, "sector": True, "cvar_value": ":.2%", "market_cap": ":.0f"},
    )
    fig1.update_xaxes(tickangle=-45)
    fig1.update_layout(hovermode="closest")

    fig2 = px.scatter(
        df, x="expected_return_kalman", y="cvar_value", size="market_cap", color="reward_to_cvar",
        hover_data={"name": True, "sector": True, "expected_return_kalman": ":.3f",
                    "cvar_value": ":.2%", "reward_to_cvar": ":.2f", "market_cap": ":.0f"},
        labels={"expected_return_kalman": "Expected Return (Kalman)",
                "cvar_value": f"CVaR ({confidence_level}%)", "reward_to_cvar": "Reward-to-CVaR",
                "market_cap": "Market Cap (M)"},
        title="Risk-Return Profile",
    )
    fig2.update_traces(marker=dict(sizemin=6))
    fig2.update_layout(hovermode="closest")
    return fig1, fig2


@callback(
    output=[
        Output(f"{component_id}_graph_1", "figure"),
        Output(f"{component_id}_error_1", "children"),
        Output(f"{component_id}_graph_2", "figure"),
        Output(f"{component_id}_error_2", "children"),
        Output(sector_filter_id, "options"),
        Output(sector_filter_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        confidence_level_id: Input(confidence_level_id, "value"),
        sort_metric_id: Input(sort_metric_id, "value"),
        min_market_cap_id: Input(min_market_cap_id, "value"),
        min_prob_positive_id: Input(min_prob_positive_id, "value"),
        num_stocks_id: Input(num_stocks_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, str, go.Figure, str, list, list]:
    # Scope the local Sectors filter to the globally-filtered universe.
    df_all = filter_data(get_data(), **kwargs)
    sector_opts, sector_val = scoped_filter(
        df_all, "sector", kwargs.get(sector_filter_id), multi=True
    )
    kwargs[sector_filter_id] = sector_val
    try:
        fig1, fig2 = _update_logic(**kwargs)
        return fig1, "", fig2, "", sector_opts, sector_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        empty = empty_figure("An error occurred")
        return empty, msg, empty, msg, sector_opts, sector_val