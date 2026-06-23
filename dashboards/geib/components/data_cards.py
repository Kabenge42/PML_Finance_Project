"""Five KPI cards summarising the filtered universe.

Cards (left to right): Total Securities, Avg Market Cap (M), Avg Expected
Return, Avg Signal Strength, Avg Reward/CVaR. All cards update on global-filter
changes; they show "No Data" on an empty selection and "Error" on failure.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from ..data import get_data
from ..logger import logger
from .filter_component import FILTER_CALLBACK_INPUTS, filter_data

component_id = "data_cards"

TOTAL_ID = f"{component_id}_total"
AVG_MKTCAP_ID = f"{component_id}_avg_mktcap"
AVG_ER_ID = f"{component_id}_avg_er"
AVG_SIGNAL_ID = f"{component_id}_avg_signal"
AVG_RCVAR_ID = f"{component_id}_avg_rcvar"

_CARDS = [
    (TOTAL_ID, "Total Securities"),
    (AVG_MKTCAP_ID, "Avg Market Cap (M)"),
    (AVG_ER_ID, "Avg Expected Return"),
    (AVG_SIGNAL_ID, "Avg Signal Strength"),
    (AVG_RCVAR_ID, "Avg Reward/CVaR"),
]


def _kpi_card(card_value_id: str, title: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(title, className="geib-kpi-title"),
                    html.H3("—", id=card_value_id, className="geib-kpi-value"),
                ]
            ),
            className="geib-kpi-card",
        ),
        width=True,
    )


def component() -> dbc.Row:
    """Build the single row of five KPI cards."""
    return dbc.Row(
        [_kpi_card(card_id, title) for card_id, title in _CARDS],
        className="g-3",
        style={"marginBottom": "8px"},
    )


@callback(
    output=[
        Output(TOTAL_ID, "children"),
        Output(AVG_MKTCAP_ID, "children"),
        Output(AVG_ER_ID, "children"),
        Output(AVG_SIGNAL_ID, "children"),
        Output(AVG_RCVAR_ID, "children"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs):
    no_data = ("No Data",) * 5
    try:
        df = filter_data(get_data(), **kwargs)
        if df is None or len(df) == 0:
            return no_data

        total = f"{len(df):,}"
        avg_mktcap = f"{df['market_cap'].mean():,.0f}"
        avg_er = f"{df['expected_return_kalman'].mean():.4f}"
        avg_signal = f"{df['signal_strength'].mean():.2f}"
        avg_rcvar = f"{df['reward_to_cvar'].mean():.2f}"
        return total, avg_mktcap, avg_er, avg_signal, avg_rcvar
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error updating KPI cards: %s", exc)
        return ("Error",) * 5
