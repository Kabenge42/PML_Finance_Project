"""High Conviction Opportunities table.

Securities with expected return > 30% whose selected probability metric falls in
a user-chosen band, ranked by Kalman-filtered metrics. The band replaces what was
a hard-coded ``p_upside_pos_cond > 0.67`` gate; it defaults to that same
threshold, so the table is unchanged on load. Spec-only component (no reference
stub).
"""

from __future__ import annotations

import traceback
from typing import Tuple

from dash import Input, Output, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

from ._common import column_values, scoped_filter
from ..components.filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..components.probability_filter import (
    apply_probability_filter,
    probability_controls,
    probability_inputs,
    register as register_probability_filter,
)
from ..data import get_data
from ..logger import logger
from ..theme import BACKGROUND_CONTENT, BODY_TEXT, BORDER, GOLD, NAVY, control
from ..theme import card as theme_card

component_id = "high_conviction_opportunities"

country_id = f"{component_id}_country"
country_default = "US"

sort_by_id = f"{component_id}_sort_by"
sort_by_options = [
    {"label": "Expected Return (Kalman)", "value": "expected_return_kalman"},
    {"label": "P(risk-adj. return > 0)", "value": "p_upside_pos_cond"},
    {"label": "Signal Strength", "value": "signal_strength"},
    {"label": "Reward / Tail Risk (STARR)", "value": "reward_to_cvar"},
    {"label": "Price Target (Kalman)", "value": "price_target_kalman"},
    {"label": "Beta", "value": "beta"},
    # er_mean / er_sd over simulated LOG PRICE-TARGET UPLIFT — a t-statistic on
    # the uplift estimate, not an investment Sharpe ratio. Labelled accordingly
    # so nobody compares the 5-7 book values to a portfolio Sharpe.
    {"label": "Uplift t-stat", "value": "expected_sharpe_ratio"},
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

# Minimum expected return this card screens on (unchanged, not user-tunable).
min_expected_return = 0.3
# Probability metric + band (shared control pair). The low handle reproduces the
# former hard-coded ``p_upside_pos_cond > 0.67`` gate.
min_prob_default = 0.67
register_probability_filter(component_id)

_COLUMNS = [
    ("ticker", "Ticker", None),
    ("name", "Name", None),
    ("country_name", "Country", None),
    ("industry", "Industry", None),
    ("sector", "Sector", None),
    ("exchange_name", "Exchange", None),
    ("unit_name", "Currency", None),
    ("original_price", "Original Price", 2),
    ("price_target_kalman", "Price Target Kalman", 2),
    ("price_target_median", "Price Target Median", 2),
    ("expected_return_kalman", "Expected Return Kalman", 2),
    ("p_upside_pos_cond", "P(risk-adj > 0)", 2),
    ("beta", "Beta", 2),
    ("expected_sharpe_ratio", "Uplift t-stat", 2),
    ("signal_strength", "Signal Strength", 2),
    ("reward_to_cvar", "Reward / Tail Risk", 2),
]

title = "High Conviction Opportunities"
description = (
    "Securities with expected return > 30%, gated by a user-selected probability "
    "metric and band, ranked by Kalman-filtered metrics"
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
    country_opts = [{"label": "All", "value": "All"}]
    country_opts += [{"label": c, "value": c} for c in column_values(get_data(), "country_name")]

    return theme_card(
        title,
        description,
        card_id=component_id,
        children=[
            html.Div(
                className="geib-controls-row",
                children=[
                    control("Country:", dcc.Dropdown(
                        id=country_id, options=country_opts, value=country_default,
                        searchable=True, style={"minWidth": "150px"})),
                    *probability_controls(component_id, lo=min_prob_default),
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

    country_name = kwargs.get(country_id) or country_default
    sort_by = kwargs.get(sort_by_id) or sort_by_default
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    row_limit = kwargs.get(row_limit_id, row_limit_default)

    # The probability band is inclusive at both ends, so a name sitting exactly on
    # the low handle now qualifies (the former gate was a strict ``> 0.67``).
    df = df[df["expected_return_kalman"] > min_expected_return]
    df = apply_probability_filter(df, component_id, kwargs)
    if country_name != "All":
        df = df[df["country_name"] == country_name]
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
        Output(country_id, "options"),
        Output(country_id, "value"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        country_id: Input(country_id, "value"),
        **probability_inputs(component_id),
        sort_by_id: Input(sort_by_id, "value"),
        row_limit_id: Input(row_limit_id, "value"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs) -> Tuple[list, str, list, str]:
    # Scope the local Country filter (with its "All" sentinel) to the
    # globally-filtered universe.
    df_all = filter_data(get_data(), **kwargs)
    country_opts, country_val = scoped_filter(
        df_all, "country_name", kwargs.get(country_id), include_all=True, all_label="All"
    )
    kwargs[country_id] = country_val
    try:
        return _update_logic(**kwargs), "", country_opts, country_val
    except Exception as exc:
        msg = f"Error updating table: {exc}\n{traceback.format_exc()}"
        logger.error(msg)
        return [], msg, country_opts, country_val