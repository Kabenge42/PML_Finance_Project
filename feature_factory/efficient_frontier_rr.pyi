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

component_id = "efficient_frontier_risk_return"

def component() -> ComponentResponse:
    graph_id = f"{component_id}_graph"
    table_id = f"{component_id}_table"
    error_id = f"{component_id}_error"
    loading_id = f"{component_id}_loading"

    stock_selector_id = f"{component_id}_stock_selector"
    risk_free_rate_id = f"{component_id}_risk_free_rate"
    constraint_type_id = f"{component_id}_constraint_type"
    num_portfolios_id = f"{component_id}_num_portfolios"

    title = "Efficient Frontier: Risk-Return Portfolio Optimization"
    description = "Explore optimal portfolio allocations by analyzing the risk-return tradeoff curve. Select stocks and adjust parameters to find the best portfolio combinations."

    df = get_data()
    df_sorted = df.nlargest(50, 'market_cap')
    default_stocks = df_sorted.head(10)['ticker'].tolist()
    stock_options = [
        {"label": f"{row['ticker']} - {row['name'][:30]}", "value": row['ticker']}
        for _, row in df_sorted.iterrows()
    ]

    layout = ddk.Card(
        id=component_id,
        children=[
            ddk.CardHeader(title=title),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "rowGap": "10px", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.Div(
                        children=[
                            html.Label("Select Stocks:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=stock_selector_id,
                                options=stock_options,
                                value=default_stocks,
                                multi=True,
                                style={"minWidth": "300px"}
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Risk-Free Rate:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=risk_free_rate_id,
                                options=[
                                    {"label": "0%", "value": 0.0},
                                    {"label": "2%", "value": 2.0},
                                    {"label": "3%", "value": 3.0},
                                    {"label": "4%", "value": 4.0},
                                    {"label": "5%", "value": 5.0}
                                ],
                                value=3.0,
                                style={"minWidth": "150px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Constraint Type:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=constraint_type_id,
                                options=[
                                    {"label": "Long Only", "value": "long_only"},
                                    {"label": "Long/Short", "value": "long_short"},
                                    {"label": "Sector Neutral", "value": "sector_neutral"}
                                ],
                                value="long_only",
                                style={"minWidth": "180px"},
                                searchable=False
                            )
                        ],
                        style={"display": "flex", "flexDirection": "column", "marginRight": "15px"}
                    ),
                    html.Div(
                        children=[
                            html.Label("Number of Portfolios:", style={"marginBottom": "5px", "fontWeight": "bold", "display": "block"}),
                            dcc.Dropdown(
                                id=num_portfolios_id,
                                options=[
                                    {"label": "100", "value": 100},
                                    {"label": "500", "value": 500},
                                    {"label": "1000", "value": 1000},
                                    {"label": "5000", "value": 5000}
                                ],
                                value=500,
                                style={"minWidth": "150px"},
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
            html.Div(
                id=table_id,
                style={"marginTop": "20px", "overflowX": "auto"}
            ),
            ddk.CardFooter(title=description)
        ],
        width=100
    )

    test_inputs: dict[str, TestInput] = {
        stock_selector_id: {
            "options": [s["value"] for s in stock_options[:10]],
            "default": default_stocks
        },
        risk_free_rate_id: {
            "options": [0.0, 2.0, 3.0, 4.0, 5.0],
            "default": 3.0
        },
        constraint_type_id: {
            "options": ["long_only", "long_short", "sector_neutral"],
            "default": "long_only"
        },
        num_portfolios_id: {
            "options": [100, 500, 1000, 5000],
            "default": 500
        }
    }

    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def _estimate_covariance_matrix(df: pd.DataFrame, selected_tickers: list) -> pd.DataFrame:
    """Estimate covariance matrix from sector/industry correlations."""
    logger.debug("Estimating covariance matrix from sector/industry correlations...")

    n = len(selected_tickers)
    cov_matrix = np.eye(n) * 0.04

    sector_map = {}
    for ticker in selected_tickers:
        ticker_data = df[df['ticker'] == ticker]
        if len(ticker_data) > 0:
            sector_map[ticker] = ticker_data['sector'].iloc[0]

    for i in range(n):
        for j in range(i + 1, n):
            ticker_i = selected_tickers[i]
            ticker_j = selected_tickers[j]

            sector_i = sector_map.get(ticker_i, "Unknown")
            sector_j = sector_map.get(ticker_j, "Unknown")

            if sector_i == sector_j:
                correlation = 0.6
            else:
                correlation = 0.3

            volatility_i = 0.25
            volatility_j = 0.25

            cov_matrix[i, j] = correlation * volatility_i * volatility_j
            cov_matrix[j, i] = cov_matrix[i, j]

    logger.debug("Covariance matrix shape: %s", cov_matrix.shape)
    return pd.DataFrame(cov_matrix, index=selected_tickers, columns=selected_tickers)

def _generate_random_portfolios(
        expected_returns: np.ndarray,
        cov_matrix: pd.DataFrame,
        num_portfolios: int,
        risk_free_rate: float,
        constraint_type: str,
        selected_tickers: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate random portfolio combinations."""
    logger.debug("Generating %d random portfolios with constraint type: %s...", num_portfolios, constraint_type)

    n_assets = len(expected_returns)
    portfolio_returns = np.zeros(num_portfolios)
    portfolio_volatilities = np.zeros(num_portfolios)
    portfolio_sharpe_ratios = np.zeros(num_portfolios)
    portfolio_weights = np.zeros((num_portfolios, n_assets))

    np.random.seed(42)

    for i in range(num_portfolios):
        if constraint_type == "long_only":
            weights = np.random.dirichlet(np.ones(n_assets))
        elif constraint_type == "long_short":
            weights = np.random.normal(0, 0.3, n_assets)
            weights = weights / np.sum(np.abs(weights))
        else:
            weights = np.random.dirichlet(np.ones(n_assets))

        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - risk_free_rate / 100) / portfolio_volatility
        else:
            sharpe_ratio = 0

        portfolio_returns[i] = portfolio_return
        portfolio_volatilities[i] = portfolio_volatility
        portfolio_sharpe_ratios[i] = sharpe_ratio
        portfolio_weights[i] = weights

    logger.debug("Generated portfolios - Returns range: [%.2f, %.2f], Volatilities range: [%.2f, %.2f]",
                 portfolio_returns.min(), portfolio_returns.max(),
                 portfolio_volatilities.min(), portfolio_volatilities.max())

    return portfolio_returns, portfolio_volatilities, portfolio_sharpe_ratios, portfolio_weights

def _find_optimal_portfolios(
        expected_returns: np.ndarray,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
        selected_tickers: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find maximum Sharpe ratio and minimum variance portfolios."""
    logger.debug("Finding optimal portfolios (max Sharpe ratio and min variance)...")

    n_assets = len(expected_returns)

    min_var_weights = np.ones(n_assets) / n_assets
    min_var_return = np.sum(min_var_weights * expected_returns)
    min_var_volatility = np.sqrt(np.dot(min_var_weights, np.dot(cov_matrix.values, min_var_weights)))

    max_sharpe_weights = np.ones(n_assets) / n_assets
    max_sharpe_return = np.sum(max_sharpe_weights * expected_returns)
    max_sharpe_volatility = np.sqrt(np.dot(max_sharpe_weights, np.dot(cov_matrix.values, max_sharpe_weights)))

    if max_sharpe_volatility > 0:
        max_sharpe_ratio = (max_sharpe_return - risk_free_rate / 100) / max_sharpe_volatility
    else:
        max_sharpe_ratio = 0

    logger.debug("Min variance portfolio - Return: %.2f%%, Volatility: %.2f%%", min_var_return * 100, min_var_volatility * 100)
    logger.debug("Max Sharpe portfolio - Return: %.2f%%, Volatility: %.2f%%, Sharpe: %.2f",
                 max_sharpe_return * 100, max_sharpe_volatility * 100, max_sharpe_ratio)

    return min_var_weights, max_sharpe_weights, np.array([min_var_volatility, max_sharpe_volatility])

def _update_logic(**kwargs) -> Tuple[go.Figure, str]:
    """Core chart update logic without error handling."""
    logger.debug("Updating efficient frontier with inputs:\n%s", "\n".join(f"  • {k}: {v}" for k, v in kwargs.items()))

    df = filter_data(get_data(), **kwargs)

    if len(df) == 0:
        logger.error("No data available after filtering")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{"text": "No data is available to display", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig, ""

    logger.debug("Loaded data with %d rows", len(df))
    logger.debug(schema(df))

    stock_selector = kwargs.get(f'{component_id}_stock_selector')
    risk_free_rate = kwargs.get(f'{component_id}_risk_free_rate', 3.0)
    constraint_type = kwargs.get(f'{component_id}_constraint_type', 'long_only')
    num_portfolios = kwargs.get(f'{component_id}_num_portfolios', 500)

    if risk_free_rate is None:
        risk_free_rate = 3.0
    if constraint_type is None:
        constraint_type = 'long_only'
    if num_portfolios is None:
        num_portfolios = 500

    logger.debug("Stock selector: %s, Risk-free rate: %.1f%%, Constraint: %s, Num portfolios: %d",
                 stock_selector, risk_free_rate, constraint_type, num_portfolios)

    if not stock_selector or len(stock_selector) == 0:
        logger.debug("No stocks selected, using top 10 by market cap...")
        df_sorted = df.nlargest(10, 'market_cap')
        selected_tickers = df_sorted['ticker'].tolist()
    else:
        selected_tickers = stock_selector if isinstance(stock_selector, list) else [stock_selector]

    logger.debug("Selected tickers: %s", selected_tickers)

    df_selected = df[df['ticker'].isin(selected_tickers)].copy()

    if len(df_selected) < len(selected_tickers):
        logger.warning("Only found %d out of %d selected tickers in data", len(df_selected), len(selected_tickers))
        selected_tickers = df_selected['ticker'].unique().tolist()

    if len(selected_tickers) < 2:
        logger.error("Need at least 2 stocks for portfolio optimization")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Insufficient data",
            annotations=[{"text": "Need at least 2 stocks for portfolio optimization", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig, ""

    logger.debug("Computing expected returns for selected stocks...")
    expected_returns = []
    for ticker in selected_tickers:
        ticker_data = df_selected[df_selected['ticker'] == ticker]
        if len(ticker_data) > 0:
            ret = ticker_data['expected_upside_pct'].iloc[0]
            if pd.isna(ret):
                ret = 0.05
            else:
                ret = float(ret) / 100.0
            expected_returns.append(ret)
        else:
            expected_returns.append(0.05)

    expected_returns = np.array(expected_returns)
    logger.debug("Expected returns: %s", expected_returns)

    logger.debug("Estimating covariance matrix...")
    cov_matrix = _estimate_covariance_matrix(df, selected_tickers)

    logger.debug("Generating random portfolios...")
    portfolio_returns, portfolio_volatilities, portfolio_sharpe_ratios, portfolio_weights = _generate_random_portfolios(
        expected_returns,
        cov_matrix,
        num_portfolios,
        risk_free_rate,
        constraint_type,
        selected_tickers
    )

    logger.debug("Portfolio returns shape: %s, min: %.4f, max: %.4f",
                 portfolio_returns.shape, portfolio_returns.min(), portfolio_returns.max())
    logger.debug("Portfolio volatilities shape: %s, min: %.4f, max: %.4f",
                 portfolio_volatilities.shape, portfolio_volatilities.min(), portfolio_volatilities.max())

    if len(portfolio_returns) == 0 or portfolio_volatilities.max() == 0:
        logger.error("No valid portfolios generated")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No valid portfolios",
            annotations=[{"text": "Could not generate valid portfolio combinations", "showarrow": False, "font": {"size": 20}}]
        )
        return empty_fig, ""

    logger.debug("Finding optimal portfolios...")
    min_var_weights, max_sharpe_weights, opt_volatilities = _find_optimal_portfolios(
        expected_returns,
        cov_matrix,
        risk_free_rate,
        selected_tickers
    )

    min_var_return = np.sum(min_var_weights * expected_returns)
    max_sharpe_return = np.sum(max_sharpe_weights * expected_returns)
    max_sharpe_volatility = opt_volatilities[1]

    if max_sharpe_volatility > 0:
        max_sharpe_ratio = (max_sharpe_return - risk_free_rate / 100) / max_sharpe_volatility
    else:
        max_sharpe_ratio = 0

    logger.debug("Creating scatter plot...")

    scatter_df = pd.DataFrame({
        'Volatility': portfolio_volatilities * 100,
        'Return': portfolio_returns * 100,
        'Sharpe Ratio': portfolio_sharpe_ratios
    })

    logger.debug("Scatter data shape: %s", scatter_df.shape)
    logger.debug(tbl(scatter_df.head(10)))

    fig = px.scatter(
        scatter_df,
        x='Volatility',
        y='Return',
        color='Sharpe Ratio',
        color_continuous_scale='Viridis',
        labels={'Volatility': 'Portfolio Volatility (%)', 'Return': 'Expected Return (%)'},
        hover_data={'Volatility': ':.2f', 'Return': ':.2f', 'Sharpe Ratio': ':.3f'}
    )

    fig.add_trace(go.Scatter(
        x=[opt_volatilities[0] * 100],
        y=[min_var_return * 100],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star', line=dict(color='darkred', width=2)),
        name='Min Variance',
        hovertemplate='<b>Min Variance Portfolio</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[max_sharpe_volatility * 100],
        y=[max_sharpe_return * 100],
        mode='markers',
        marker=dict(size=15, color='gold', symbol='star', line=dict(color='orange', width=2)),
        name='Max Sharpe Ratio',
        hovertemplate=f'<b>Max Sharpe Ratio Portfolio</b><br>Volatility: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<br>Sharpe: {max_sharpe_ratio:.3f}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title='Portfolio Volatility (Annualized %)',
        yaxis_title='Expected Return (Annualized %)',
        hovermode='closest',
        coloraxis_colorbar=dict(title='Sharpe Ratio')
    )

    logger.debug("Creating portfolio table...")

    table_data = []
    for i in range(min(10, len(portfolio_weights))):
        row = {'Portfolio': f'P{i+1}'}
        for j, ticker in enumerate(selected_tickers):
            row[ticker] = f'{portfolio_weights[i, j]:.2%}'
        row['Return'] = f'{portfolio_returns[i]:.2%}'
        row['Volatility'] = f'{portfolio_volatilities[i]:.2%}'
        row['Sharpe'] = f'{portfolio_sharpe_ratios[i]:.3f}'
        table_data.append(row)

    min_var_row = {'Portfolio': 'Min Variance'}
    for j, ticker in enumerate(selected_tickers):
        min_var_row[ticker] = f'{min_var_weights[j]:.2%}'
    min_var_row['Return'] = f'{min_var_return:.2%}'
    min_var_row['Volatility'] = f'{opt_volatilities[0]:.2%}'
    min_var_row['Sharpe'] = f'{(min_var_return - risk_free_rate / 100) / opt_volatilities[0]:.3f}' if opt_volatilities[0] > 0 else 'N/A'
    table_data.append(min_var_row)

    max_sharpe_row = {'Portfolio': 'Max Sharpe'}
    for j, ticker in enumerate(selected_tickers):
        max_sharpe_row[ticker] = f'{max_sharpe_weights[j]:.2%}'
    max_sharpe_row['Return'] = f'{max_sharpe_return:.2%}'
    max_sharpe_row['Volatility'] = f'{max_sharpe_volatility:.2%}'
    max_sharpe_row['Sharpe'] = f'{max_sharpe_ratio:.3f}'
    table_data.append(max_sharpe_row)

    table_df = pd.DataFrame(table_data)
    logger.debug("Portfolio table created with %d rows", len(table_df))
    logger.debug(tbl(table_df))

    table_html = table_df.to_html(index=False, classes='table table-striped', border=0)

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
        f'{component_id}_stock_selector': Input(f"{component_id}_stock_selector", "value"),
        f'{component_id}_risk_free_rate': Input(f"{component_id}_risk_free_rate", "value"),
        f'{component_id}_constraint_type': Input(f"{component_id}_constraint_type", "value"),
        f'{component_id}_num_portfolios': Input(f"{component_id}_num_portfolios", "value"),
        **FILTER_CALLBACK_INPUTS
    }
)
def update(**kwargs) -> Tuple[go.Figure, str, str]:
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Error in chart",
        annotations=[{"text": "An error occurred while updating this chart", "showarrow": False, "font": {"size": 20}}]
    )

    try:
        figure, table_html = _update_logic(**kwargs)

        return figure, table_html, ""

    except Exception as e:
        error_msg = f"Error updating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return empty_fig, "", error_msg

@callback(
    Output(f"{component_id}_stock_selector", "options"),
    Input("refresh_trigger", "data"),
    **FILTER_CALLBACK_INPUTS
)
def update_stock_options(**kwargs):
    try:
        df = filter_data(get_data(), **kwargs)

        logger.debug("Updating stock selector options...")
        df_sorted = df.nlargest(50, 'market_cap')

        stock_options = [
            {"label": f"{row['ticker']} - {row['name'][:30]}", "value": row['ticker']}
            for _, row in df_sorted.iterrows()
        ]

        logger.debug("Stock options updated with %d stocks", len(stock_options))
        return stock_options

    except Exception as e:
        logger.error("Error updating stock options: %s", str(e))
        return []