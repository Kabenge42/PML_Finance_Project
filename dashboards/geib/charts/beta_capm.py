"""Beta & CAPM: Market Sensitivity Analysis.

Two side-by-side panes driven by a shared CAPM computation over the real
``beta`` column exported to ``analytics.kalman_filtered_price_targets`` (the
NULL-aware mean of ``beta_{1y,2y,5y}`` — ``feat_avg_beta``):

* **Market Sensitivity** — a scatter of each name's ``beta`` against its
  Kalman ``expected_return_kalman`` (bubble size = ``market_cap``, colour by a
  selectable dimension), overlaid with the theoretical CAPM line
  ``rf + beta * (market_return - rf)``.
* **Jensen's Alpha Analysis** — a horizontal bar chart of the highest- and
  lowest-alpha tickers (``alpha = expected_return_kalman -
  capm_expected_return``), coloured by the sign of alpha.

The risk-free rate and expected market return are dropdown inputs, so the SML
and alpha decomposition reflect the analyst's own capital-market assumptions
rather than a universe-derived proxy. A minimum |alpha| threshold and a beta
band (low / medium / high) narrow both panes. All returns are raw decimals
(0.25 = +25%) per the project unit contract.
"""

from __future__ import annotations

import traceback
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import coalesce, empty_figure, scoped_filter
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema, tbl
from ..theme import ref_line, DIVERGING_POLES, COLORWAY, STACKED_GRAPH_STYLE, SUBTLE_TEXT, control
from ..theme import card as theme_card

component_id = "capm_beta_return_calculator"

risk_free_rate_id = f"{component_id}_risk_free_rate"
risk_free_rate_options = [
    {"label": "0%", "value": 0.0},
    {"label": "2%", "value": 0.02},
    {"label": "3%", "value": 0.03},
    {"label": "4%", "value": 0.04},
    {"label": "5%", "value": 0.05},
]
risk_free_rate_default = 0.03

market_return_id = f"{component_id}_market_return"
market_return_options = [
    {"label": "6%", "value": 0.06},
    {"label": "8%", "value": 0.08},
    {"label": "10%", "value": 0.10},
    {"label": "12%", "value": 0.12},
]
market_return_default = 0.08

min_alpha_id = f"{component_id}_min_alpha"
min_alpha_options = [
    {"label": "0%", "value": 0.0},
    {"label": "2%", "value": 0.02},
    {"label": "5%", "value": 0.05},
    {"label": "10%", "value": 0.10},
]
min_alpha_default = 0.0

beta_range_id = f"{component_id}_beta_range"
beta_range_options = [
    {"label": "All", "value": "all"},
    {"label": "Low (<0.8)", "value": "low"},
    {"label": "Medium (0.8-1.2)", "value": "medium"},
    {"label": "High (>1.2)", "value": "high"},
]
beta_range_default = "all"

# Beta band edges for the Low / Medium / High convenience filter.
_BETA_LOW = 0.8
_BETA_HIGH = 1.2

color_by_id = f"{component_id}_color_by"
color_by_options = [
    {"label": "Sector", "value": "sector"},
    {"label": "Region", "value": "region"},
    {"label": "Style Class", "value": "style_class"},
    {"label": "Size Class", "value": "size_class"},
    {"label": "Industry", "value": "industry"},
    {"label": "Trading Region", "value": "trading_region"},
    {"label": "None", "value": "none"},
]
color_by_default = "sector"

size_class_id = f"{component_id}_size_class"
size_class_options = [
    {"label": "All", "value": "All"},
    {"label": "Large Cap", "value": "Large Cap"},
    {"label": "Mid Cap", "value": "Mid Cap"},
    {"label": "Small Cap", "value": "Small Cap"},
]
size_class_default = "All"

sector_filter_id = f"{component_id}_sector_filter"

# Number of names shown on each side of the alpha bar chart (top + bottom).
_ALPHA_TOP_N = 15

# Minimum marker size so zero / missing market caps still render on the SML.
_MARKER_SIZE_MIN = 6

# Alpha bar colours come from `theme.DIVERGING_POLES`. They were a hardcoded
# green/red pair, "#2ecc71" / "#e74c3c", and the measured case against them is
# narrower than "red and green are colour-blind-unsafe":
#
#   The pair PASSES the adjacent-CVD floor -- dE 10.7 under deuteranopia against
#   a floor of 8. It does not collapse. But the theme's diverging poles separate
#   dE 26.8 on the same surface, 2.5x further apart, and the green also sat
#   OUTSIDE the dark lightness band at L 0.746, which is the check the old pair
#   actually failed.
#
# So this is a real improvement on a pair that was legible rather than broken,
# and it buys consistency besides: the same two hues now serve the continuous
# ramp (DIVERGING_SCALE) and this binary split, so the two views of a signed
# quantity agree.

title = "Beta & CAPM: Market Sensitivity Analysis"
description = (
    "Analyze stock sensitivity to market movements using beta coefficients "
    "and CAPM expected returns. Identify mispriced securities through alpha "
    "analysis."
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
                    control("Risk-Free Rate:", dcc.Dropdown(
                        id=risk_free_rate_id, options=risk_free_rate_options,
                        value=risk_free_rate_default, searchable=False,
                        clearable=False, style={"minWidth": "130px"})),
                    control("Expected Market Return:", dcc.Dropdown(
                        id=market_return_id, options=market_return_options,
                        value=market_return_default, searchable=False,
                        clearable=False, style={"minWidth": "160px"})),
                    control("Min |Alpha| Threshold:", dcc.Dropdown(
                        id=min_alpha_id, options=min_alpha_options,
                        value=min_alpha_default, searchable=False,
                        clearable=False, style={"minWidth": "150px"})),
                    control("Beta Range:", dcc.Dropdown(
                        id=beta_range_id, options=beta_range_options,
                        value=beta_range_default, searchable=False,
                        clearable=False, style={"minWidth": "170px"})),
                    control("Color By:", dcc.Dropdown(
                        id=color_by_id, options=color_by_options,
                        value=color_by_default, searchable=False,
                        clearable=False, style={"minWidth": "160px"})),
                    control("Size Class:", dcc.Dropdown(
                        id=size_class_id, options=size_class_options,
                        value=size_class_default, searchable=False,
                        clearable=False, style={"minWidth": "150px"})),
                    control("Sectors:", dcc.Dropdown(
                        id=sector_filter_id, options=[], value=[], multi=True,
                        placeholder="All Sectors", style={"minWidth": "200px"})),
                ],
            ),
            html.Div(
                className="geib-dual-graph geib-dual-graph--stacked",
                children=[
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("Market Sensitivity (Beta vs Expected Return)",
                                   className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_scatter_graph", style=STACKED_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_scatter_error", className="geib-error"),
                    ]),
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("Jensen's Alpha Analysis",
                                   className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_bar_graph", style=STACKED_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_bar_error", className="geib-error"),
                    ]),
                ],
            ),
        ],
    )


def _prepare_capm_frame(**kwargs) -> Tuple[pd.DataFrame, float, float]:
    """Return ``(df, risk_free_rate, market_return)`` with CAPM / alpha columns.

    Applies the global filters, keeps only rows with a non-null ``beta``,
    narrows by the local Beta Range / Size Class / Sectors controls, computes
    ``capm_expected_return = rf + beta * (market_return - rf)`` plus
    ``alpha = expected_return_kalman - capm_expected_return``, and drops rows
    below the Min |Alpha| threshold.
    """
    df = filter_data(get_data(), **kwargs)
    risk_free_rate = float(coalesce(kwargs.get(risk_free_rate_id), risk_free_rate_default))
    market_return = float(coalesce(kwargs.get(market_return_id), market_return_default))

    if df is None or len(df) == 0:
        return pd.DataFrame(), risk_free_rate, market_return

    logger.debug("Filtering to stocks with non-null beta...")
    df = df[df["beta"].notna()].copy()

    beta_range = coalesce(kwargs.get(beta_range_id), beta_range_default)
    if beta_range == "low":
        df = df[df["beta"] < _BETA_LOW]
    elif beta_range == "medium":
        df = df[(df["beta"] >= _BETA_LOW) & (df["beta"] <= _BETA_HIGH)]
    elif beta_range == "high":
        df = df[df["beta"] > _BETA_HIGH]

    size_class = coalesce(kwargs.get(size_class_id), size_class_default)
    if size_class and size_class != "All":
        df = df[df["size_class"] == size_class]

    sector_filter = kwargs.get(sector_filter_id) or []
    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]
    if len(df) == 0:
        return df, risk_free_rate, market_return

    logger.debug("Calculating CAPM expected return with R_f=%.2f%%, E(R_m)=%.2f%%...",
                 risk_free_rate * 100, market_return * 100)
    df["capm_expected_return"] = risk_free_rate + df["beta"] * (market_return - risk_free_rate)
    df["alpha"] = df["expected_return_kalman"] - df["capm_expected_return"]

    min_alpha = float(coalesce(kwargs.get(min_alpha_id), min_alpha_default))
    if min_alpha > 0.0:
        df = df[df["alpha"].abs() >= min_alpha]
    logger.debug(tbl(df[["ticker", "beta", "capm_expected_return", "alpha"]]))
    return df, risk_free_rate, market_return


def _scatter_figure(df: pd.DataFrame, risk_free_rate: float, market_return: float,
                    color_by: Optional[str]) -> go.Figure:
    """SML scatter (bubble size = market cap) with the CAPM line overlaid."""
    hover_data = {
        "name": True,
        "ticker": True,
        "beta": ":.3f",
        "expected_return_kalman": ":.3f",
        "market_cap": ":.0f",
        "_size": False,
    }
    labels = {
        "beta": "Beta (Market Sensitivity)",
        "expected_return_kalman": "Expected Return",
    }
    scatter_kwargs = dict(
        x="beta", y="expected_return_kalman", size="_size",
        color_discrete_sequence=COLORWAY, hover_data=hover_data, labels=labels,
    )
    if color_by:
        legend_title = color_by.replace("_", " ").title()
        scatter_kwargs["color"] = color_by
        hover_data[color_by] = True
        labels[color_by] = legend_title

    # px.scatter rejects NaN / negative sizes; clip so every name renders.
    df = df.assign(_size=df["market_cap"].fillna(0.0).clip(lower=0.0))
    fig = px.scatter(df, **scatter_kwargs)
    fig.update_traces(marker=dict(sizemin=_MARKER_SIZE_MIN), selector=dict(mode="markers"))

    beta_span = float(df["beta"].max() - df["beta"].min()) or 1.0
    sml_beta = np.array([
        df["beta"].min() - 0.05 * beta_span,
        df["beta"].max() + 0.05 * beta_span,
    ])
    sml_return = risk_free_rate + sml_beta * (market_return - risk_free_rate)
    fig.add_trace(go.Scatter(
        x=sml_beta, y=sml_return, mode="lines", name="Security Market Line",
        line=dict(color=SUBTLE_TEXT, dash="dash", width=2),
        hovertemplate="Beta: %{x:.3f}<br>CAPM Return: %{y:.3f}<extra></extra>",
    ))

    fig.update_xaxes(title_text="Beta (Market Sensitivity)")
    fig.update_yaxes(title_text="Expected Return")
    fig.update_layout(hovermode="closest")
    if color_by:
        fig.update_layout(legend_title_text=color_by.replace("_", " ").title())
    return fig


def _bar_figure(df: pd.DataFrame) -> go.Figure:
    """Horizontal alpha bars: top/bottom ``_ALPHA_TOP_N`` tickers by alpha."""
    df_sorted = df.sort_values("alpha", ascending=False)
    df_display = pd.concat([
        df_sorted.head(_ALPHA_TOP_N),
        df_sorted.tail(_ALPHA_TOP_N),
    ]).drop_duplicates().sort_values("alpha", ascending=True)
    df_display = df_display.assign(
        alpha_color=np.where(df_display["alpha"] > 0, "Positive", "Negative")
    )
    logger.debug(tbl(df_display[["ticker", "alpha"]]))

    fig = px.bar(
        df_display, x="alpha", y="ticker", color="alpha_color", orientation="h",
        color_discrete_map={"Positive": DIVERGING_POLES["positive"],
                            "Negative": DIVERGING_POLES["negative"]},
        hover_data={"ticker": True, "alpha": ":.4f", "alpha_color": False},
        labels={"alpha": "Alpha (Actual - CAPM Return)", "ticker": "Ticker"},
    )
    # Keep the ticker axis in strict ascending-alpha order (px splits the frame
    # into one trace per colour, which would otherwise interleave categories).
    fig.update_yaxes(categoryorder="array",
                     categoryarray=df_display["ticker"].tolist(),
                     title_text="Ticker")
    # "top left", not "top right": the bars are sorted ascending, so the top of
    # the pane is the longest POSITIVE bar and a right-anchored label lands on it.
    # Left of the rule at that height is empty.
    fig.add_vline(x=0, **ref_line("zero", annotation_text="Zero Alpha",
                                  annotation_position="top left"))
    # No legend, deliberately. The rule is that identity must never be carried by
    # colour ALONE -- here the sign is already given by which side of the labelled
    # "Zero Alpha" rule a bar sits on, and colour is the redundant encoding that
    # keeps it readable for a colour-blind reader. A Positive/Negative legend on a
    # chart split about a labelled zero would restate the x-axis.
    fig.update_xaxes(title_text="Jensen's Alpha (Actual - CAPM Return)")
    # Grow the pane with the bar count so every company stays legible.
    fig.update_layout(showlegend=False, hovermode="closest",
                      height=max(400, 20 * len(df_display)))
    return fig


def _update_logic(**kwargs) -> Tuple[go.Figure, go.Figure]:
    df, risk_free_rate, market_return = _prepare_capm_frame(**kwargs)
    if len(df) == 0:
        empty = empty_figure("No stocks with beta data match the selected filters")
        return empty, empty
    logger.debug(schema(df))

    color_by = coalesce(kwargs.get(color_by_id), color_by_default)
    color_column = None if color_by == "none" else color_by

    fig_scatter = _scatter_figure(df, risk_free_rate, market_return, color_column)
    fig_bar = _bar_figure(df)
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
        market_return_id: Input(market_return_id, "value"),
        min_alpha_id: Input(min_alpha_id, "value"),
        beta_range_id: Input(beta_range_id, "value"),
        color_by_id: Input(color_by_id, "value"),
        size_class_id: Input(size_class_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
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