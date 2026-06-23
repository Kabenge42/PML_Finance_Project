"""High Conviction Opportunities table.

Securities with expected return > 30% and probability of positive return > 95%,
ranked by Kalman-filtered metrics. Spec-only component (no reference stub).
"""

from __future__ import annotations

import traceback
from typing import Tuple

from dash import Input, Output, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

from ..data import get_data
from ..logger import logger
from ..theme import BODY_TEXT, BORDER, NAVY, SLATE_DARK, control
from ..theme import card as theme_card
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data

component_id = "high_conviction_opportunities"

country_id = f"{component_id}_country"
country_options = [
    {"label": c, "value": c}
    for c in ["All", "US", "CN", "JP", "GB", "IN", "DE", "FR", "CA", "AU"]
]
country_default = "US"

sort_by_id = f"{component_id}_sort_by"
sort_by_options = [
    {"label": "Expected Return (Kalman)", "value": "expected_return_kalman"},
    {"label": "Probability of Positive Return", "value": "mc_prob_pos"},
    {"label": "Signal Strength", "value": "signal_strength"},
    {"label": "Reward to CVaR", "value": "reward_to_cvar"},
    {"label": "Price Target (Kalman)", "value": "price_target_kalman"},
]
sort_by_default = ["expected_return_kalman"]

row_limit_id = f"{component_id}_row_limit"
row_limit_options = [
    {"label": "100", "value": 100},
    {"label": "500", "value": 500},
    {"label": "1000", "value": 1000},
    {"label": "All", "value": "all"},
]
row_limit_default = 500

_COLUMNS = [
    ("ticker", "Ticker", None),
    ("name", "Name", None),
    ("country", "Country", None),
    ("industry", "Industry", None),
    ("original_price", "Original Price", 2),
    ("price_target_kalman", "Price Target Kalman", 2),
    ("expected_return_kalman", "Expected Return Kalman", 2),
    ("mc_prob_pos", "Mc Prob Pos", 2),
    ("signal_strength", "Signal Strength", 2),
    ("reward_to_cvar", "Reward To Cvar", 2),
]

title = "High Conviction Opportunities"
description = (
    "Securities with expected return > 30% and probability of positive return "
    "> 95%, ranked by Kalman-filtered metrics"
)


def _table_columns() -> list[dict]:
    columns = []
    for col, label, precision in _COLUMNS:
        spec = {"name": label, "id": col}
        if precision is not None:
            spec["type"] = "numeric"
            spec["format"] = Format(precision=precision, scheme=Scheme.fixed).group(True)
        columns.append(spec)
    return columns


def component() -> "object":
    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "rowGap": "10px"},
                children=[
                    control("Country:", dcc.Dropdown(
                        id=country_id, options=country_options, value=country_default,
                        searchable=False, style={"minWidth": "150px"})),
                    control("Sort By:", dcc.Dropdown(
                        id=sort_by_id, options=sort_by_options, value=sort_by_default,
                        multi=True, style={"minWidth": "300px"})),
                    control("Row Limit:", dcc.Dropdown(
                        id=row_limit_id, options=row_limit_options, value=row_limit_default,
                        searchable=False, style={"minWidth": "140px"})),
                ],
            ),
            dcc.Loading(
                type="circle",
                children=[
                    dash_table.DataTable(
                        id=f"{component_id}_table",
                        columns=_table_columns(),
                        data=[],
                        sort_action="native",
                        page_size=25,
                        style_as_list_view=True,
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": NAVY,
                            "color": "#FFFFFF",
                            "fontWeight": "bold",
                            "borderBottom": "2px solid #D4AF37",
                            "fontFamily": "monospace",
                        },
                        style_cell={
                            "backgroundColor": SLATE_DARK,
                            "color": BODY_TEXT,
                            "fontFamily": "monospace",
                            "fontSize": "12px",
                            "padding": "8px",
                            "borderBottom": f"1px solid {BORDER}",
                            "textAlign": "left",
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": col}, "textAlign": "right"}
                            for col, _, precision in _COLUMNS
                            if precision is not None
                        ],
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": NAVY},
                        ],
                    )
                ],
            ),
            html.Pre(id=f"{component_id}_error", className="geib-error"),
        ],
    )


def _update_logic(**kwargs) -> list[dict]:
    df = filter_data(get_data(), **kwargs)
    if df is None or len(df) == 0:
        return []

    country = kwargs.get(country_id) or country_default
    sort_by = kwargs.get(sort_by_id) or sort_by_default
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    row_limit = kwargs.get(row_limit_id, row_limit_default)

    df = df[(df["expected_return_kalman"] > 0.3) & (df["mc_prob_pos"] > 0.95)]
    if country != "All":
        df = df[df["country"] == country]
    if len(df) == 0:
        return []

    sort_cols = [c for c in sort_by if c in df.columns] or ["expected_return_kalman"]
    df = df.sort_values(sort_cols, ascending=False)

    if row_limit != "all":
        df = df.head(int(row_limit))

    cols = [c for c, _, _ in _COLUMNS]
    return df[cols].to_dict("records")


@callback(
    output=[
        Output(f"{component_id}_table", "data"),
        Output(f"{component_id}_error", "children"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        country_id: Input(country_id, "value"),
        sort_by_id: Input(sort_by_id, "value"),
        row_limit_id: Input(row_limit_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[list, str]:
    try:
        return _update_logic(**kwargs), ""
    except Exception as exc:
        msg = f"Error updating table: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return [], msg
