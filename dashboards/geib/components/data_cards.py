"""Six KPI cards summarising the filtered universe, plus a data-as-of stamp.

Cards (left to right): Total Securities, Avg Market Cap (M), Avg Expected
Return, Avg Signal Strength, Avg Reward/CVaR, Avg Beta. All cards update on
global-filter changes; they show "No Data" on an empty selection and "Error"
on failure. The "Data as of" stamp reads the max ``last_updated`` of the
*unfiltered* frame — it is a data-freshness indicator, not a selection
statistic — and refreshes on Reset (which reloads the data).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, html

from .filter_component import FILTER_CALLBACK_INPUTS, filter_data
from ..data import get_data
from ..logger import logger

component_id = "data_cards"

TOTAL_ID = f"{component_id}_total"
AVG_MKTCAP_ID = f"{component_id}_avg_mktcap"
AVG_ER_ID = f"{component_id}_avg_er"
AVG_SIGNAL_ID = f"{component_id}_avg_signal"
AVG_RCVAR_ID = f"{component_id}_avg_rcvar"
AVG_BETA_ID = f"{component_id}_avg_beta"
ASOF_ID = f"{component_id}_asof"

_CARDS = [
    (TOTAL_ID, "Total Securities"),
    (AVG_MKTCAP_ID, "Avg Market Cap (M)"),
    (AVG_ER_ID, "Avg Expected Return"),
    (AVG_SIGNAL_ID, "Avg Signal Strength"),
    (AVG_RCVAR_ID, "Avg Reward/CVaR"),
    (AVG_BETA_ID, "Avg Beta"),
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


def component() -> html.Div:
    """Build the row of six KPI cards with the data-as-of stamp below it."""
    return html.Div(
        [
            dbc.Row(
                [_kpi_card(card_id, title) for card_id, title in _CARDS],
                className="g-3 geib-kpi-row",
            ),
            html.Div("", id=ASOF_ID, className="geib-kpi-asof"),
        ]
    )


def _asof_stamp() -> str:
    """Return the "Data as of" stamp from the unfiltered frame ("" if unknown)."""
    try:
        df = get_data()
        if "last_updated" not in df.columns:
            return ""
        asof = df["last_updated"].max()
        if pd.isna(asof):
            return ""
        return f"Data as of {pd.Timestamp(asof):%Y-%m-%d}"
    except Exception:  # pragma: no cover - defensive
        return ""


@callback(
    output=[
        Output(TOTAL_ID, "children"),
        Output(AVG_MKTCAP_ID, "children"),
        Output(AVG_ER_ID, "children"),
        Output(AVG_SIGNAL_ID, "children"),
        Output(AVG_RCVAR_ID, "children"),
        Output(AVG_BETA_ID, "children"),
        Output(ASOF_ID, "children"),
    ],
    inputs={
        "refresh_trigger": Input("refresh_trigger", "data"),
        **FILTER_CALLBACK_INPUTS,
    },
)
def update(**kwargs):
    asof = _asof_stamp()
    no_data = ("No Data",) * len(_CARDS) + (asof,)
    try:
        df = filter_data(get_data(), **kwargs)
        if df is None or len(df) == 0:
            return no_data

        total = f"{len(df):,}"
        avg_mktcap = f"{df['market_cap'].mean():,.0f}"
        avg_er = f"{df['expected_return_kalman'].mean():.4f}"
        avg_signal = f"{df['signal_strength'].mean():.2f}"
        avg_rcvar = f"{df['reward_to_cvar'].mean():.2f}"
        beta_mean = df["beta"].mean() if "beta" in df.columns else float("nan")
        avg_beta = "—" if pd.isna(beta_mean) else f"{beta_mean:.2f}"
        return total, avg_mktcap, avg_er, avg_signal, avg_rcvar, avg_beta, asof
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error updating KPI cards: %s", exc)
        return ("Error",) * len(_CARDS) + (asof,)
