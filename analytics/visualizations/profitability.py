"""
Profitability visualization module for margin analysis.

This module provides visualization functions for analyzing profitability metrics:
- Margin waterfall charts (revenue to net income breakdown)
- DuPont decomposition dashboards
- Profitability quadrant analysis
- Margin trend heatmaps

Feature Categories leveraged (from market_analytics.py):
- Profitability: roe, roa, gross_margin_pct, operating_margin_pct,
  net_margin_pct, ebitda_margin_pct, roic, net_margin_trend_yoy
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.visualizations._shared import (
    PLOTLY_TEMPLATE,
    COLORS,
    create_no_data_figure,
    resolve_column,
)


def create_margin_waterfall_chart(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    show_median: bool = True,
) -> go.Figure:
    """
    Waterfall chart showing revenue to net income margin breakdown.

    Uses: gross_margin_pct, operating_margin_pct, ebitda_margin_pct, net_margin_pct

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with margin columns
    ticker : str, optional
        Specific ticker to highlight. If None, shows median values
    show_median : bool, default True
        Whether to show median line for comparison

    Returns
    -------
    go.Figure
        Plotly waterfall chart figure

    Examples
    --------
    >>> fig = create_margin_waterfall_chart(df, ticker='AAPL')
    >>> fig.show()
    """
    margin_cols = [
        "gross_margin_pct",
        "operating_margin_pct",
        "ebitda_margin_pct",
        "net_margin_pct",
    ]
    available_cols = [col for col in margin_cols if col in df.columns]

    if not available_cols:
        return create_no_data_figure("Margin Waterfall Chart - No Data")

    # Get values for the waterfall
    if ticker and "ticker" in df.columns:
        stock_data = df[df["ticker"] == ticker]
        if len(stock_data) == 0:
            stock_data = df
            title_suffix = "(Median Values)"
        else:
            title_suffix = f"({ticker})"
    else:
        stock_data = df
        title_suffix = "(Median Values)"

    # Calculate margin values
    values = []
    labels = []

    # Start with 100% revenue
    labels.append("Revenue")
    values.append(100)

    # Calculate step-down values
    margin_labels = {
        "gross_margin_pct": "Gross Margin",
        "operating_margin_pct": "Operating Margin",
        "ebitda_margin_pct": "EBITDA Margin",
        "net_margin_pct": "Net Margin",
    }

    prev_value = 100
    waterfall_values = [100]  # Starting point
    waterfall_measures = ["absolute"]
    waterfall_labels = ["Revenue (100%)"]

    for col in available_cols:
        if col in stock_data.columns:
            margin_val = (
                stock_data[col].median() if len(stock_data) > 1 else stock_data[col].iloc[0]
            )
            if pd.notna(margin_val):
                # Calculate the reduction from previous stage
                reduction = prev_value - margin_val
                waterfall_values.append(-reduction)
                waterfall_measures.append("relative")
                waterfall_labels.append(f"{margin_labels.get(col, col)}")
                prev_value = margin_val

    # Add final total
    waterfall_values.append(prev_value)
    waterfall_measures.append("total")
    waterfall_labels.append(f"Net Result ({prev_value:.1f}%)")

    fig = go.Figure(
        go.Waterfall(
            name="Margin Breakdown",
            orientation="v",
            measure=waterfall_measures,
            x=waterfall_labels,
            y=waterfall_values,
            textposition="outside",
            text=[
                f"{v:.1f}%" if m != "relative" else f"-{abs(v):.1f}%"
                for v, m in zip(waterfall_values, waterfall_measures)
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "rgb(46, 184, 46)"}},
            decreasing={"marker": {"color": "rgb(255, 65, 54)"}},
            totals={"marker": {"color": "rgb(55, 128, 191)"}},
        ),
    )

    fig.update_layout(
        title=f"Margin Waterfall Analysis {title_suffix}",
        yaxis_title="Margin (%)",
        showlegend=False,
        height=500,
        template=PLOTLY_TEMPLATE,
    )

    return fig


def create_dupont_decomposition_dashboard(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    DuPont analysis visualization: ROE = Net Margin × Asset Turnover × Leverage.

    Uses: roe, roa, net_margin_pct, debt_to_equity

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with DuPont components
    top_n : int, default 20
        Number of top stocks to display

    Returns
    -------
    go.Figure
        Plotly figure with DuPont decomposition dashboard

    Examples
    --------
    >>> fig = create_dupont_decomposition_dashboard(df)
    >>> fig.show()
    """
    required_cols = ["roe", "net_margin_pct"]
    available_cols = [col for col in required_cols if col in df.columns]

    if len(available_cols) < 2:
        return create_no_data_figure("DuPont Decomposition - Insufficient Data")

    # Prepare data
    plot_df = df.copy()

    # Calculate asset turnover proxy if not available
    at_col = resolve_column(plot_df, "asset_turnover")
    if at_col is not None and at_col != "asset_turnover":
        plot_df["asset_turnover"] = plot_df[at_col]
    if "asset_turnover" not in plot_df.columns:
        if "roa" in plot_df.columns and "net_margin_pct" in plot_df.columns:
            # Asset Turnover ≈ ROA / Net Margin
            plot_df["asset_turnover"] = plot_df["roa"] / plot_df["net_margin_pct"].replace(
                0,
                np.nan,
            )
        else:
            plot_df["asset_turnover"] = 1.0

    # Calculate leverage (equity multiplier)
    if "equity_multiplier" not in plot_df.columns:
        if "debt_to_equity" in plot_df.columns:
            plot_df["equity_multiplier"] = 1 + plot_df["debt_to_equity"]
        else:
            plot_df["equity_multiplier"] = 1.5  # Default assumption

    # Filter valid data
    plot_df = plot_df.dropna(subset=["roe", "net_margin_pct"])

    # Get top performers by ROE
    top_stocks = plot_df.nlargest(top_n, "roe")

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "ROE vs Net Margin",
            "ROE Components Distribution",
            "Leverage vs Profitability",
            "DuPont Factor Contribution",
        ),
        specs=[[{"type": "scatter"}], [{"type": "box"}], [{"type": "scatter"}], [{"type": "bar"}]],
        vertical_spacing=0.08,
    )

    # 1. ROE vs Net Margin scatter
    fig.add_trace(
        go.Scatter(
            x=top_stocks["net_margin_pct"],
            y=top_stocks["roe"],
            mode="markers",
            marker=dict(
                size=10,
                color=top_stocks["equity_multiplier"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Leverage", x=0.45),
            ),
            text=top_stocks.get("ticker", top_stocks.index),
            hovertemplate="<b>%{text}</b><br>Net Margin: %{x:.1f}%<br>ROE: %{y:.1f}%<extra></extra>",
            name="Stocks",
        ),
        row=1,
        col=1,
    )

    # 2. ROE Components box plot
    components = ["net_margin_pct", "asset_turnover", "equity_multiplier"]
    for i, comp in enumerate(components):
        if comp in plot_df.columns:
            fig.add_trace(
                go.Box(
                    y=plot_df[comp].clip(-100, 100),
                    name=comp.replace("_", " ").title(),
                    boxpoints="outliers",
                ),
                row=2,
                col=1,
            )

    # 3. Leverage vs Profitability
    if "roa" in top_stocks.columns:
        fig.add_trace(
            go.Scatter(
                x=top_stocks["equity_multiplier"],
                y=top_stocks["roa"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=top_stocks["roe"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="ROE", x=1.0),
                ),
                text=top_stocks.get("ticker", top_stocks.index),
                hovertemplate="<b>%{text}</b><br>Leverage: %{x:.2f}<br>ROA: %{y:.1f}%<extra></extra>",
                name="ROA vs Leverage",
            ),
            row=3,
            col=1,
        )

    # 4. DuPont Factor Contribution (average)
    avg_margin = plot_df["net_margin_pct"].median()
    avg_turnover = (
        plot_df["asset_turnover"].median() if "asset_turnover" in plot_df.columns else 1.0
    )
    avg_leverage = plot_df["equity_multiplier"].median()

    # Normalize contributions
    total = abs(avg_margin) + abs(avg_turnover) + abs(avg_leverage)
    if total > 0:
        contributions = [
            abs(avg_margin) / total * 100,
            abs(avg_turnover) / total * 100,
            abs(avg_leverage) / total * 100,
        ]
    else:
        contributions = [33.3, 33.3, 33.3]

    fig.add_trace(
        go.Bar(
            x=["Net Margin", "Asset Turnover", "Leverage"],
            y=contributions,
            marker_color=["rgb(55, 128, 191)", "rgb(50, 171, 96)", "rgb(219, 64, 82)"],
            text=[f"{c:.1f}%" for c in contributions],
            textposition="auto",
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title="DuPont ROE Decomposition Dashboard",
        height=1400,
        width=1000,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="Net Margin (%)", row=1, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="ROE (%)", row=1, col=1, tickfont=dict(size=11))
    fig.update_xaxes(title_text="Equity Multiplier", row=3, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="ROA (%)", row=3, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="Contribution (%)", row=4, col=1, tickfont=dict(size=11))

    return fig


def create_profitability_quadrant(
    df: pd.DataFrame,
    x_metric: str = "roe",
    y_metric: str = "roic",
    size_metric: str = "net_margin_pct",
    color_by: str = "industry",
) -> go.Figure:
    """
    Quadrant plot: ROE vs ROIC with margin bubble size.

    Identifies quality vs capital-intensive businesses.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with profitability metrics
    x_metric : str, default 'roe'
        Metric for x-axis
    y_metric : str, default 'roic'
        Metric for y-axis
    size_metric : str, default 'net_margin_pct'
        Metric for bubble size
    color_by : str, default 'industry'
        Column to color bubbles by

    Returns
    -------
    go.Figure
        Plotly scatter figure with quadrant analysis

    Examples
    --------
    >>> fig = create_profitability_quadrant(df)
    >>> fig.show()
    """
    # Check required columns
    if x_metric not in df.columns or y_metric not in df.columns:
        return create_no_data_figure("Profitability Quadrant - Missing Data")

    # Prepare data
    plot_df = df.dropna(subset=[x_metric, y_metric]).copy()

    # Calculate bubble sizes
    if size_metric in plot_df.columns:
        sizes = plot_df[size_metric].clip(1, 50).fillna(10)
    else:
        sizes = 10

    # Calculate quadrant thresholds (medians)
    x_median = plot_df[x_metric].median()
    y_median = plot_df[y_metric].median()

    # Assign quadrants
    plot_df["quadrant"] = "Low-Low"
    plot_df.loc[(plot_df[x_metric] >= x_median) & (plot_df[y_metric] >= y_median), "quadrant"] = (
        "High-High (Quality)"
    )
    plot_df.loc[(plot_df[x_metric] >= x_median) & (plot_df[y_metric] < y_median), "quadrant"] = (
        "High ROE, Low ROIC"
    )
    plot_df.loc[(plot_df[x_metric] < x_median) & (plot_df[y_metric] >= y_median), "quadrant"] = (
        "Low ROE, High ROIC"
    )

    # Create figure
    fig = go.Figure()

    # Add quadrant backgrounds
    x_range = [plot_df[x_metric].min() - 5, plot_df[x_metric].max() + 5]
    y_range = [plot_df[y_metric].min() - 5, plot_df[y_metric].max() + 5]

    # Add quadrant lines
    fig.add_hline(y=y_median, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=x_median, line_dash="dash", line_color="gray", opacity=0.5)

    # Color by industry or quadrant
    if color_by in plot_df.columns:
        color_col = plot_df[color_by]
    else:
        color_col = plot_df["quadrant"]

    # Add scatter points
    # Calculate sizeref properly
    if isinstance(sizes, (int, float)):
        size_values = sizes
        max_size = sizes
    else:
        size_values = sizes.values
        max_size = float(sizes.max())

    fig.add_trace(
        go.Scatter(
            x=plot_df[x_metric],
            y=plot_df[y_metric],
            mode="markers",
            marker=dict(
                size=size_values,
                sizemode="area",
                sizeref=2.0 * max_size / (40.0**2),
                sizemin=4,
                opacity=0.7,
            ),
            text=plot_df.get("ticker", plot_df.index),
            customdata=(
                np.stack(
                    [
                        plot_df.get("name", plot_df.index),
                        (
                            plot_df[size_metric]
                            if size_metric in plot_df.columns
                            else [0] * len(plot_df)
                        ),
                        plot_df["quadrant"],
                    ],
                    axis=-1,
                )
                if size_metric in plot_df.columns
                else None
            ),
            hovertemplate=(
                (
                    "<b>%{text}</b><br>"
                    + f"{x_metric}: "
                    + "%{x:.1f}%<br>"
                    + f"{y_metric}: "
                    + "%{y:.1f}%<br>"
                    + f"{size_metric}: "
                    + "%{customdata[1]:.1f}%<br>"
                    + "Quadrant: %{customdata[2]}<extra></extra>"
                )
                if size_metric in plot_df.columns
                else None
            ),
            name="Stocks",
        ),
    )

    # Add quadrant labels
    fig.add_annotation(
        x=x_range[1],
        y=y_range[1],
        text="Quality Leaders",
        showarrow=False,
        font=dict(size=12, color="green"),
    )
    fig.add_annotation(
        x=x_range[0],
        y=y_range[0],
        text="Underperformers",
        showarrow=False,
        font=dict(size=12, color="red"),
    )
    fig.add_annotation(
        x=x_range[1],
        y=y_range[0],
        text="Leveraged Returns",
        showarrow=False,
        font=dict(size=12, color="orange"),
    )
    fig.add_annotation(
        x=x_range[0],
        y=y_range[1],
        text="Capital Efficient",
        showarrow=False,
        font=dict(size=12, color="blue"),
    )

    fig.update_layout(
        title=f"Profitability Quadrant: {x_metric.upper()} vs {y_metric.upper()}",
        xaxis_title=f"{x_metric.upper()} (%)",
        yaxis_title=f"{y_metric.upper()} (%)",
        height=600,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
    )

    return fig


def create_margin_trend_heatmap(
    df: pd.DataFrame,
    margin_col: str = "net_margin_trend_yoy",
    group_col: str = "industry",
) -> go.Figure:
    """
    Heatmap of margin trends by industry.

    Uses: net_margin_trend_yoy, industry columns

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with margin trend data
    margin_col : str, default 'net_margin_trend_yoy'
        Column containing margin trend values
    group_col : str, default 'industry'
        Column to group by (rows in heatmap)

    Returns
    -------
    go.Figure
        Plotly heatmap figure

    Examples
    --------
    >>> fig = create_margin_trend_heatmap(df)
    >>> fig.show()
    """
    if margin_col not in df.columns or group_col not in df.columns:
        return create_no_data_figure("Margin Trend Heatmap - Missing Data")

    # Calculate statistics by group
    stats = (
        df.groupby(group_col)[margin_col]
        .agg(
            [
                ("mean", "mean"),
                ("median", "median"),
                ("std", "std"),
                ("count", "count"),
                ("positive_pct", lambda x: (x > 0).sum() / len(x) * 100),
            ],
        )
        .reset_index()
    )

    # Filter groups with sufficient data
    stats = stats[stats["count"] >= 5].sort_values("median", ascending=False)

    if len(stats) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for heatmap",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title="Margin Trend Heatmap - Insufficient Data",
            template=PLOTLY_TEMPLATE,
        )
        return fig

    # Create heatmap data
    metrics = ["mean", "median", "std", "positive_pct"]
    metric_labels = ["Mean Trend", "Median Trend", "Volatility", "% Improving"]

    z_data = stats[metrics].values

    # Normalize for better visualization
    z_normalized = np.zeros_like(z_data)
    for i, metric in enumerate(metrics):
        col_data = z_data[:, i]
        if col_data.std() > 0:
            z_normalized[:, i] = (col_data - col_data.mean()) / col_data.std()
        else:
            z_normalized[:, i] = 0

    fig = go.Figure(
        data=go.Heatmap(
            z=z_normalized,
            x=metric_labels,
            y=stats[group_col].tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(z_data, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=(
                "<b>%{y}</b><br>" + "%{x}: %{text}<br>" + "Z-score: %{z:.2f}<extra></extra>"
            ),
        ),
    )

    fig.update_layout(
        title=f"Margin Trend Analysis by {group_col.title()}",
        xaxis_title="Metric",
        yaxis_title=group_col.title(),
        height=max(400, len(stats) * 25 + 100),
        template=PLOTLY_TEMPLATE,
    )

    return fig
