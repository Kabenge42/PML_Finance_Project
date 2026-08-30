"""Piotroski F-Score: Financial Health Trend Analysis.

Two side-by-side panes over the per-fiscal-year Piotroski F-score composites
exported to ``analytics.kalman_filtered_price_targets`` (0.9.9.9):

* **F-Score Trend Over Time** — a line per company across the four exported
  fiscal years (``piotroski_f_score_{neg3fy,neg2fy,neg1fy,fy}``), with
  strong / moderate / weak reference bands. The FY lags are ordinal (there is
  no per-year date column), so the x-axis is derived from ``fy_end_date``'s
  calendar year when available and falls back to ordinal FY labels.
* **F-Score Change vs Analyst Rating** — a scatter of the 1-year F-score
  change against the vendor consensus ``analyst_rating`` (1-5 scale, higher =
  more bullish), bubble size = ``market_cap``, coloured by sector.

The per-year F-score components are analytics-export-only features: they are
barred from the fused Kalman drift matrix (collinear with the median composite
— ``KALMAN_PIOTROSKI_COMPONENT_FEATURES``, CHANGELOG 0.9.9.9), which makes
this card the intended surface for inspecting them.
"""

from __future__ import annotations

import traceback
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ._common import category_order, fold_categories, coalesce, empty_figure, scoped_filter, sector_values
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger, schema, tbl
from ..theme import COLORWAY, STACKED_GRAPH_STYLE, GREEN, RED, SUBTLE_TEXT, control
from ..theme import card as theme_card

component_id = "piotroski_fscore_trend"

companies_id = f"{component_id}_companies"

min_fscore_id = f"{component_id}_min_fscore"
min_fscore_options = [
    {"label": "0 (All)", "value": 0},
    {"label": "3 (Weak+)", "value": 3},
    {"label": "5 (Moderate+)", "value": 5},
    {"label": "7 (Strong+)", "value": 7},
]
min_fscore_default = 0

change_filter_id = f"{component_id}_change_filter"
change_filter_options = [
    {"label": "All", "value": "all"},
    {"label": "Improving (>0)", "value": "improving"},
    {"label": "Deteriorating (<0)", "value": "deteriorating"},
    {"label": "Stable (=0)", "value": "stable"},
]
change_filter_default = "all"

sector_filter_id = f"{component_id}_sector_filter"

highlight_id = f"{component_id}_highlight"
highlight_options = [
    {"label": "All", "value": "all"},
    {"label": "High F-Score (>=7)", "value": "high"},
    {"label": "Low F-Score (<=4)", "value": "low"},
    {"label": "Improving", "value": "improving"},
    {"label": "Deteriorating", "value": "deteriorating"},
]
highlight_default = "all"

# Ordered (column, fiscal-year offset, ordinal label) triples for the melt —
# oldest first so each company's trend line runs left to right.
_FSCORE_PERIODS = [
    ("piotroski_f_score_neg3fy", -3, "FY-3"),
    ("piotroski_f_score_neg2fy", -2, "FY-2"),
    ("piotroski_f_score_neg1fy", -1, "FY-1"),
    ("piotroski_f_score_fy", 0, "FY"),
]

# F-score regime thresholds (Piotroski 0-9 composite).
_STRONG = 7
_MODERATE = 5
_WEAK = 4

# Amber for the moderate reference line (semantic accent; the theme carries no
# orange token — hardcoded like beta_capm's positive/negative pair).
_MODERATE_COLOR = "#F59E0B"

# Companies drawn on the trend pane when no explicit selection is made.
_DEFAULT_TOP_N_COMPANIES = 10

# Minimum bubble size so zero / missing market caps still render.
_MARKER_SIZE_MIN = 6

title = "Piotroski F-Score: Financial Health Trend Analysis"
description = (
    "Track company financial health evolution using Piotroski F-Score (0-9 "
    "scale). Identify improving or deteriorating fundamentals by analyzing "
    "score changes across fiscal years and compare to analyst sentiment."
)


def _top_companies(df: pd.DataFrame, n: int = _DEFAULT_TOP_N_COMPANIES) -> list[str]:
    """Return the *n* largest names by ``market_cap`` (layout / fallback default)."""
    try:
        return (
            df.dropna(subset=["market_cap", "name"])
            .nlargest(n, "market_cap")["name"]
            .astype(str)
            .tolist()
        )
    except Exception:  # pragma: no cover - defensive (missing column / bad dtype)
        return []


def component() -> "object":
    df = get_data()
    names = sorted(v for v in df["name"].dropna().unique().tolist() if str(v).strip())
    company_opts = [{"label": n, "value": n} for n in names]
    sector_opts = [{"label": s, "value": s} for s in sector_values(df)]

    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                className="geib-controls-row",
                children=[
                    control("Companies:", dcc.Dropdown(
                        id=companies_id, options=company_opts,
                        value=_top_companies(df), multi=True,
                        placeholder=f"Top {_DEFAULT_TOP_N_COMPANIES} by market cap",
                        style={"minWidth": "300px"})),
                    control("Min F-Score:", dcc.Dropdown(
                        id=min_fscore_id, options=min_fscore_options,
                        value=min_fscore_default, searchable=False, clearable=False,
                        style={"minWidth": "140px"})),
                    control("F-Score Change:", dcc.Dropdown(
                        id=change_filter_id, options=change_filter_options,
                        value=change_filter_default, searchable=False, clearable=False,
                        style={"minWidth": "170px"})),
                    control("Sectors:", dcc.Dropdown(
                        id=sector_filter_id, options=sector_opts, value=[], multi=True,
                        placeholder="All Sectors", style={"minWidth": "200px"})),
                    control("Highlight:", dcc.Dropdown(
                        id=highlight_id, options=highlight_options,
                        value=highlight_default, searchable=False, clearable=False,
                        style={"minWidth": "180px"})),
                ],
            ),
            html.Div(
                className="geib-dual-graph geib-dual-graph--stacked",
                children=[
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("F-Score Trend Over Time", className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_graph_1", style=STACKED_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_error_1", className="geib-error"),
                    ]),
                    html.Div(className="geib-graph-pane", children=[
                        html.Label("F-Score Change vs Analyst Rating",
                                   className="geib-graph-label"),
                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id=f"{component_id}_graph_2", style=STACKED_GRAPH_STYLE)]),
                        html.Pre(id=f"{component_id}_error_2", className="geib-error"),
                    ]),
                ],
            ),
        ],
    )


def _prepare_frame(**kwargs) -> pd.DataFrame:
    """Return the globally + locally filtered frame with derived F-score columns."""
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return pd.DataFrame()

    cols = ["name", "sector", "market_cap", "analyst_rating", "fy_end_date"]
    cols += [col for col, _, _ in _FSCORE_PERIODS]
    df = df[cols].copy()
    df = df.dropna(subset=["piotroski_f_score_fy"])
    logger.debug(schema(df))

    df["fscore_change"] = df["piotroski_f_score_fy"] - df["piotroski_f_score_neg1fy"]
    df["fscore_momentum"] = (
        (df["piotroski_f_score_fy"] - df["piotroski_f_score_neg3fy"]) / 3.0
    )

    min_fscore = float(coalesce(kwargs.get(min_fscore_id), min_fscore_default))
    if min_fscore > 0:
        df = df[df["piotroski_f_score_fy"] >= min_fscore]

    change_filter = coalesce(kwargs.get(change_filter_id), change_filter_default)
    if change_filter == "improving":
        df = df[df["fscore_change"] > 0]
    elif change_filter == "deteriorating":
        df = df[df["fscore_change"] < 0]
    elif change_filter == "stable":
        df = df[df["fscore_change"] == 0]

    sector_filter = kwargs.get(sector_filter_id) or []
    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]

    highlight = coalesce(kwargs.get(highlight_id), highlight_default)
    if highlight == "high":
        df = df[df["piotroski_f_score_fy"] >= _STRONG]
    elif highlight == "low":
        df = df[df["piotroski_f_score_fy"] <= _WEAK]
    elif highlight == "improving":
        df = df[df["fscore_change"] > 0]
    elif highlight == "deteriorating":
        df = df[df["fscore_change"] < 0]
    return df


def _trend_figure(df: pd.DataFrame, companies: list[str]) -> go.Figure:
    """Per-company F-score lines across the four exported fiscal years."""
    selection = companies or _top_companies(df)
    df_sel = df[df["name"].isin(selection)]
    if len(df_sel) == 0:
        return empty_figure("No companies match the selected criteria")

    frames = []
    for col, offset, label in _FSCORE_PERIODS:
        part = df_sel[["name"]].copy()
        part["fscore"] = df_sel[col]
        part["offset"] = offset
        part["label"] = label
        part["year"] = df_sel["fy_end_date"].dt.year + offset
        frames.append(part)
    long_df = pd.concat(frames, ignore_index=True).dropna(subset=["fscore"])
    if len(long_df) == 0:
        return empty_figure("No F-Score history for the selected companies")

    # Calendar-year x when every row carries a fiscal-year end; ordinal FY
    # labels otherwise (the lags themselves are ordinal, not dated).
    if long_df["year"].notna().all():
        long_df["year"] = long_df["year"].astype(int)
        x_col = "year"
    else:
        x_col = "label"
    long_df = long_df.sort_values(["name", "offset"])

    fig = px.line(
        long_df, x=x_col, y="fscore", color="name", markers=True,
        color_discrete_sequence=COLORWAY,
        labels={x_col: "Fiscal Year", "fscore": "Piotroski F-Score (0-9)",
                "name": "Company"},
        hover_data={"fscore": ":.0f"},
    )
    if x_col == "label":
        fig.update_xaxes(categoryorder="array",
                         categoryarray=[label for _, _, label in _FSCORE_PERIODS])
    for value, color, text in (
        (_STRONG, GREEN, "Strong (7+)"),
        (_MODERATE, _MODERATE_COLOR, "Moderate (5-7)"),
        (_WEAK, RED, "Weak (0-4)"),
    ):
        fig.add_hline(y=value, line_dash="dash", line_color=color,
                      annotation_text=text, annotation_position="top left")
    fig.update_yaxes(range=[-0.3, 9.3], title_text="Piotroski F-Score (0-9)")
    fig.update_layout(hovermode="x unified", legend_title_text="Company")
    return fig


def _rating_figure(df: pd.DataFrame) -> go.Figure:
    """F-score 1y change vs analyst rating (1-5, higher = more bullish)."""
    df = df.dropna(subset=["analyst_rating", "fscore_change"])
    if len(df) == 0:
        return empty_figure("No analyst-rated companies match the selected criteria")
    logger.debug(tbl(df[["name", "fscore_change", "analyst_rating"]]))

    # px.scatter rejects NaN / negative sizes; clip so every name renders.
    df = df.assign(_size=df["market_cap"].fillna(0.0).clip(lower=0.0))
    # Eleven GICS sectors against eight validated hues: Plotly would cycle the
    # list and hand three sectors a colour that already means another one. Fold
    # the tail on the FULL universe's ranking so a sector keeps its hue when the
    # board is filtered.
    df = df.assign(sector=fold_categories(df["sector"]))
    fig = px.scatter(
        df, x="fscore_change", y="analyst_rating", size="_size", color="sector",
        color_discrete_sequence=COLORWAY,
        category_orders={"sector": category_order(df["sector"])},
        hover_data={
            "name": True,
            "fscore_change": ":.0f",
            "analyst_rating": ":.2f",
            "market_cap": ":.0f",
            "sector": True,
            "piotroski_f_score_fy": ":.0f",
            "fscore_momentum": ":.2f",
            "_size": False,
        },
        labels={
            "fscore_change": "F-Score Change (Current - 1Y Ago)",
            "analyst_rating": "Analyst Rating (1=Sell, 5=Buy)",
            "sector": "Sector",
            "market_cap": "Market Cap (M)",
            "piotroski_f_score_fy": "F-Score (FY)",
            "fscore_momentum": "F-Score Momentum (3Y avg)",
        },
    )
    fig.update_traces(marker=dict(sizemin=_MARKER_SIZE_MIN, opacity=0.7),
                      selector=dict(mode="markers"))
    fig.add_vline(x=0, line_dash="dash", line_color=SUBTLE_TEXT,
                  annotation_text="No Change", annotation_position="top right")
    fig.update_yaxes(range=[0.5, 5.5], title_text="Analyst Rating (1=Sell, 5=Buy)")
    fig.update_layout(hovermode="closest", legend_title_text="Sector")
    return fig


def _update_logic(**kwargs) -> Tuple[go.Figure, go.Figure]:
    df = _prepare_frame(**kwargs)
    if len(df) == 0:
        empty = empty_figure("No F-Score data matches the selected filters")
        return empty, empty
    companies = kwargs.get(companies_id) or []
    return _trend_figure(df, companies), _rating_figure(df)


@callback(
    output=[
        Output(f"{component_id}_graph_1", "figure"),
        Output(f"{component_id}_error_1", "children"),
        Output(f"{component_id}_graph_2", "figure"),
        Output(f"{component_id}_error_2", "children"),
        Output(companies_id, "options"),
        Output(companies_id, "value"),
        Output(sector_filter_id, "options"),
        Output(sector_filter_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        companies_id: Input(companies_id, "value"),
        min_fscore_id: Input(min_fscore_id, "value"),
        change_filter_id: Input(change_filter_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        highlight_id: Input(highlight_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[go.Figure, str, go.Figure, str, list, list, list, list]:
    # Scope the local Companies / Sectors pickers to the globally-filtered
    # universe (an empty Companies selection falls back to the top names by
    # market cap inside ``_trend_figure``).
    df_all = filter_data(get_data(), **kwargs)
    company_opts, company_val = scoped_filter(
        df_all, "name", kwargs.get(companies_id), multi=True
    )
    sector_opts, sector_val = scoped_filter(
        df_all, "sector", kwargs.get(sector_filter_id), multi=True
    )
    kwargs[companies_id] = company_val
    kwargs[sector_filter_id] = sector_val
    try:
        fig1, fig2 = _update_logic(**kwargs)
        return fig1, "", fig2, "", company_opts, company_val, sector_opts, sector_val
    except Exception as exc:
        msg = f"Error updating chart: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        empty = empty_figure("An error occurred")
        return empty, msg, empty, msg, company_opts, company_val, sector_opts, sector_val