import sys
import os
import traceback
from typing import TypedDict, Any, Tuple
from datetime import datetime, timedelta

from dash import callback, html, dcc, Output, Input
import dash_design_kit as ddk
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from data import get_data
from components.filter_component import filter_data, FILTER_CALLBACK_INPUTS
from logger import logger, schema, tbl


class TestInput(TypedDict):
    options: list[Any]
    default: Any


class ComponentResponse(TypedDict):
    layout: ddk.Card
    test_inputs: dict[str, TestInput]


component_id = "kelly_criterion_position_sizing"

kelly_multiplier_id = f"{component_id}_kelly_multiplier"
kelly_multiplier_options = [
    {"label": "Quarter-Kelly (0.25)", "value": 0.25},
    {"label": "Half-Kelly (0.5)", "value": 0.5},
    {"label": "Three-Quarter-Kelly (0.75)", "value": 0.75},
    {"label": "Full-Kelly (1.0)", "value": 1.0}
]
kelly_multiplier_default = 0.5

max_position_size_id = f"{component_id}_max_position_size"
max_position_size_options = [
    {"label": "5%", "value": 0.05},
    {"label": "10%", "value": 0.10},
    {"label": "15%", "value": 0.15},
    {"label": "20%", "value": 0.20},
    {"label": "25%", "value": 0.25}
]
max_position_size_default = 0.10

min_prob_id = f"{component_id}_min_prob"
min_prob_options = [
    {"label": "60%", "value": 0.6},
    {"label": "70%", "value": 0.7},
    {"label": "80%", "value": 0.8},
    {"label": "90%", "value": 0.9}
]
min_prob_default = 0.7

top_n_id = f"{component_id}_top_n"
top_n_options = [
    {"label": "Top 10", "value": 10},
    {"label": "Top 20", "value": 20},
    {"label": "Top 30", "value": 30},
    {"label": "Top 50", "value": 50}
]
top_n_default = 20

sector_filter_id = f"{component_id}_sector_filter"

min_market_cap_id = f"{component_id}_min_market_cap"
min_market_cap_options = [
    {"label": "$1B", "value": 1000},
    {"label": "$5B", "value": 5000},
    {"label": "$10B", "value": 10000},
    {"label": "$50B", "value": 50000}
]
min_market_cap_default = 5000


def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    table_id = f"{component_id}_table"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    title = "Kelly Criterion Position Sizing"
    description = "Optimal portfolio allocation using Kelly criterion to maximize long-term growth while managing risk. Kelly fraction shows the percentage of portfolio to allocate to each stock."

    df = get_data()
    sectors = sorted(df['sector'].dropna().unique().tolist())
    sector_options = [{"label": s, "value": s} for s in sectors]

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px",
                       "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Kelly Multiplier:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=kelly_multiplier_id,
                                options=kelly_multiplier_options,
                                value=kelly_multiplier_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Max Position Size:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=max_position_size_id,
                                options=max_position_size_options,
                                value=max_position_size_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Min Win Probability:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=min_prob_id,
                                options=min_prob_options,
                                value=min_prob_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Top N Stocks:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=top_n_id,
                                options=top_n_options,
                                value=top_n_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Sector:",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=sector_filter_id,
                                options=sector_options,
                                value=[s for s in sectors],
                                multi=True,
                                style={"minWidth": "200px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Min Market Cap ($M):",
                                       style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=min_market_cap_id,
                                options=min_market_cap_options,
                                value=min_market_cap_default,
                                style={"minWidth": "200px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                ],
            ),
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[
                    ddk.Graph(id=graph_id, style={"minHeight": "550px", "height": "calc(100vh - 600px)"}),
                ]
            ),
            html.Pre(id=error_id, style={"color": "red", "margin": "10px 0"}),
            html.Div(id=table_id, style={"margin": "20px 0"}),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        kelly_multiplier_id: {
            "options": [opt["value"] for opt in kelly_multiplier_options],
            "default": kelly_multiplier_default
        },
        max_position_size_id: {
            "options": [opt["value"] for opt in max_position_size_options],
            "default": max_position_size_default
        },
        min_prob_id: {
            "options": [opt["value"] for opt in min_prob_options],
            "default": min_prob_default
        },
        top_n_id: {
            "options": [opt["value"] for opt in top_n_options],
            "default": top_n_default
        },
        sector_filter_id: {
            "options": sectors,
            "default": sectors
        },
        min_market_cap_id: {
            "options": [opt["value"] for opt in min_market_cap_options],
            "default": min_market_cap_default
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }


def _calculate_kelly_fraction(row):
    """Calculate Kelly fraction for a single stock."""
    p = row['mc_prob_pos']
    q = 1 - p

    if p <= 0 or p >= 1:
        return 0

    cvar = row['cvar_5pct_kalman']
    expected_return = row['expected_return_kalman']

    if cvar == 0 or expected_return <= 0:
        return 0

    b = expected_return / abs(cvar)

    kelly = (p * b - q) / b

    return max(0, kelly)


def _create_table_html(df):
    """Convert DataFrame to HTML table."""
    df_table = df[['ticker', 'name', 'allocation_pct', 'expected_return_kalman', 'mc_prob_pos']].copy()
    df_table.columns = ['Ticker', 'Name', 'Allocation (%)', 'Expected Return (%)', 'Win Probability']
    df_table['Expected Return (%)'] = (df_table['Expected Return (%)'] * 100).round(2)
    df_table['Win Probability'] = (df_table['Win Probability'] * 100).round(2)
    df_table['Allocation (%)'] = df_table['Allocation (%)'].round(2)

    rows = []
    for i in range(len(df_table)):
        row_cells = [
            html.Td(str(df_table.iloc[i][col]), style={"padding": "8px", "borderBottom": "1px solid #ddd"})
            for col in df_table.columns
        ]
        rows.append(html.Tr(row_cells))

    table_html = html.Div([
        html.H4("Top Stocks by Kelly Fraction"),
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th(col, style={"padding": "8px", "textAlign": "left", "borderBottom": "2px solid #ddd"})
                    for col in df_table.columns
                ])
            ),
            html.Tbody(rows)
        ], style={"width": "100%", "borderCollapse": "collapse"})
    ])

    return table_html


def _update_logic(**kwargs) -> Tuple[go.Figure, html.Div]:
    """Core chart update logic without error handling."""
    logger.debug("Updating Kelly criterion chart with inputs:\n%s",
                 "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        logger.error("No data available after filtering")
        raise ValueError("No data available after filtering")

    logger.debug("Selecting columns for analysis...")
    df = df[
        ['ticker', 'name', 'sector', 'market_cap', 'mc_prob_pos', 'expected_return_kalman', 'cvar_5pct_kalman']].copy()
    logger.debug(schema(df))
    logger.debug(tbl(df))

    kelly_multiplier = kwargs.get(kelly_multiplier_id, kelly_multiplier_default)
    if kelly_multiplier is None:
        kelly_multiplier = kelly_multiplier_default

    max_position_size = kwargs.get(max_position_size_id, max_position_size_default)
    if max_position_size is None:
        max_position_size = max_position_size_default

    min_prob = kwargs.get(min_prob_id, min_prob_default)
    if min_prob is None:
        min_prob = min_prob_default

    top_n = kwargs.get(top_n_id, top_n_default)
    if top_n is None:
        top_n = top_n_default

    sector_filter = kwargs.get(sector_filter_id, [])
    if sector_filter is None or len(sector_filter) == 0:
        sector_filter = df['sector'].unique().tolist()

    min_market_cap = kwargs.get(min_market_cap_id, min_market_cap_default)
    if min_market_cap is None:
        min_market_cap = min_market_cap_default

    logger.debug("Filtering by sector: %s...", sector_filter)
    df = df[df['sector'].isin(sector_filter)]
    logger.debug(tbl(df))

    logger.debug("Filtering by minimum market cap: %s...", min_market_cap)
    df = df[df['market_cap'] >= min_market_cap]
    logger.debug(tbl(df))

    logger.debug("Filtering by minimum win probability: %s...", min_prob)
    df = df[df['mc_prob_pos'] >= min_prob]
    logger.debug(tbl(df))

    if len(df) == 0:
        raise ValueError("No stocks meet the filtering criteria")

    logger.debug("Calculating Kelly fractions...")
    df['kelly_fraction'] = df.apply(_calculate_kelly_fraction, axis=1)
    logger.debug(tbl(df))

    logger.debug("Filtering by positive Kelly fraction...")
    df = df[df['kelly_fraction'] > 0]
    logger.debug(tbl(df))

    if len(df) == 0:
        raise ValueError("No stocks have positive Kelly fraction")

    logger.debug("Sorting by Kelly fraction descending...")
    df = df.sort_values('kelly_fraction', ascending=False)
    logger.debug(tbl(df))

    logger.debug("Selecting top %d stocks...", top_n)
    df = df.head(top_n)
    logger.debug(tbl(df))

    logger.debug("Applying Kelly multiplier: %s...", kelly_multiplier)
    df['kelly_fraction_adjusted'] = df['kelly_fraction'] * kelly_multiplier
    logger.debug(tbl(df))

    logger.debug("Capping position sizes at %s...", max_position_size)
    df['kelly_fraction_capped'] = df['kelly_fraction_adjusted'].clip(upper=max_position_size)
    logger.debug(tbl(df))

    logger.debug("Normalizing allocations to sum to 100%...")
    total_allocation = df['kelly_fraction_capped'].sum()
    if total_allocation > 0:
        df['allocation_pct'] = (df['kelly_fraction_capped'] / total_allocation) * 100
    else:
        df['allocation_pct'] = 0
    logger.debug(tbl(df))

    logger.debug("Creating bar chart...")
    logger.debug(schema(df))
    logger.debug(tbl(df))

    fig = px.bar(
        df,
        x='name',
        y='allocation_pct',
        color='sector',
        labels={'allocation_pct': 'Allocation (%)', 'name': 'Stock', 'sector': 'Sector'},
        hover_data={'allocation_pct': ':.2f', 'sector': True, 'ticker': True}
    )

    fig.update_layout(
        xaxis_title="Stock",
        yaxis_title="Allocation (%)",
        hovermode='x unified',
        height=550
    )

    fig.update_xaxes(tickangle=-45)

    logger.debug("Creating data table...")
    table_html = _create_table_html(df)

    logger.debug("Done")
    return fig, table_html


@callback(
    output=[
        Output(f"{component_id}_graph", "figure"),
        Output(f"{component_id}_table", "children"),
        Output(f"{component_id}_error", "children")
    ],
    inputs={
        'refresh_trigger': Input("refresh_trigger", "data"),
        kelly_multiplier_id: Input(kelly_multiplier_id, "value"),
        max_position_size_id: Input(max_position_size_id, "value"),
        min_prob_id: Input(min_prob_id, "value"),
        top_n_id: Input(top_n_id, "value"),
        sector_filter_id: Input(sector_filter_id, "value"),
        min_market_cap_id: Input(min_market_cap_id, "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, html.Div, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        fig, table_html = _update_logic(**kwargs)
        return fig, table_html, ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, html.Div(), error_msg