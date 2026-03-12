"""
Technical analysis visualization module.

This module provides visualization functions for technical analysis:
- Momentum ribbon charts (multi-period momentum overlay)
- 52-week range distribution analysis
- Trend strength matrix heatmaps
- Momentum divergence scatter plots

Feature Categories leveraged (from market_analytics.py lines 92-102):
- Momentum & Technical: price_momentum_1m, price_momentum_3m, price_momentum_6m,
  price_momentum_1y, price_momentum_3y, price_momentum_5y, range_52w_position,
  long_term_trend_score, secular_trend_flag
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.visualizations._shared import (
    PLOTLY_TEMPLATE,
    COLORS,
    create_no_data_figure,
)


def create_momentum_ribbon_chart(
    df: pd.DataFrame,
    top_n: int = 30,
    sort_by: str = "price_momentum_1y",
) -> go.Figure:
    """
    Multi-period momentum ribbon: 1m, 3m, 6m, 1y, 3y, 5y overlaid.

    Uses: price_momentum_1m through price_momentum_5y

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with momentum columns
    top_n : int, default 30
        Number of stocks to display
    sort_by : str, default 'price_momentum_1y'
        Column to sort stocks by

    Returns
    -------
    go.Figure
        Plotly figure with momentum ribbon chart

    Examples
    --------
    >>> fig = create_momentum_ribbon_chart(df, top_n=20)
    >>> fig.show()
    """
    momentum_cols = [
        "price_momentum_5d",
        "price_momentum_1m",
        "price_momentum_3m",
        "price_momentum_6m",
        "price_momentum_1y",
        "price_momentum_3y",
        "price_momentum_5y",
    ]
    available_cols = [col for col in momentum_cols if col in df.columns]

    if not available_cols:
        return create_no_data_figure("Momentum Ribbon Chart - No Data")

    # Prepare data
    plot_df = df.dropna(subset=available_cols[:2]).copy()  # Need at least 2 momentum cols

    # Sort and select top N
    if sort_by in plot_df.columns:
        plot_df = plot_df.nlargest(top_n, sort_by)
    else:
        plot_df = plot_df.head(top_n)

    # Get ticker labels
    if "ticker" in plot_df.columns:
        labels = plot_df["ticker"].tolist()
    else:
        labels = [f"Stock {i}" for i in range(len(plot_df))]

    # Create figure
    fig = go.Figure()

    # Color palette for different time periods
    colors = {
        "price_momentum_5d": "rgb(214, 39, 40)",  # Red
        "price_momentum_1m": "rgb(255, 127, 14)",  # Orange
        "price_momentum_3m": "rgb(44, 160, 44)",  # Green
        "price_momentum_6m": "rgb(31, 119, 180)",  # Blue
        "price_momentum_1y": "rgb(148, 103, 189)",  # Purple
        "price_momentum_3y": "rgb(140, 86, 75)",  # Brown
        "price_momentum_5y": "rgb(227, 119, 194)",  # Pink
    }

    period_labels = {
        "price_momentum_5d": "5 Days",
        "price_momentum_1m": "1 Month",
        "price_momentum_3m": "3 Months",
        "price_momentum_6m": "6 Months",
        "price_momentum_1y": "1 Year",
        "price_momentum_3y": "3 Years",
        "price_momentum_5y": "5 Years",
    }

    # Add traces for each momentum period
    for col in available_cols:
        fig.add_trace(
            go.Bar(
                name=period_labels.get(col, col),
                x=labels,
                y=plot_df[col].values,
                marker_color=colors.get(col, "gray"),
                opacity=0.7,
            ),
        )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)

    fig.update_layout(
        title="Momentum Ribbon Chart - Multi-Period Comparison",
        xaxis_title="Stock",
        yaxis_title="Momentum (%)",
        barmode="group",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_tickangle=-45,
        template=PLOTLY_TEMPLATE,
    )

    return fig


def create_52w_range_distribution(df: pd.DataFrame, group_col: str = "industry") -> go.Figure:
    """
    Distribution plot of range_52w_position by sector.

    Identifies overbought/oversold conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with range_52w_position column
    group_col : str, default 'industry'
        Column to group distributions by

    Returns
    -------
    go.Figure
        Plotly figure with distribution analysis

    Examples
    --------
    >>> fig = create_52w_range_distribution(df)
    >>> fig.show()
    """
    range_col = "range_52w_position"

    if range_col not in df.columns:
        return create_no_data_figure("52-Week Range Distribution - No Data")

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Overall Distribution",
            "Distribution by Sector",
            "Overbought/Oversold Analysis",
            "Range Position vs Momentum",
        ),
        specs=[
            [{"type": "histogram"}],
            [{"type": "box"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
        ],
        vertical_spacing=0.08,
    )

    # 1. Overall histogram
    fig.add_trace(
        go.Histogram(
            x=df[range_col].dropna(),
            nbinsx=20,
            name="Distribution",
            marker_color="rgb(55, 128, 191)",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Add overbought/oversold zones
    fig.add_vline(x=0.8, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=0.2, line_dash="dash", line_color="green", row=1, col=1)

    # 2. Box plot by sector
    if group_col in df.columns:
        sectors = df[group_col].dropna().unique()[:10]  # Top 10 sectors
        for sector in sectors:
            sector_data = df[df[group_col] == sector][range_col].dropna()
            if len(sector_data) > 5:
                fig.add_trace(
                    go.Box(
                        y=sector_data,
                        name=sector[:15],
                        boxpoints="outliers",  # Truncate long names
                    ),
                    row=2,
                    col=1,
                )

    # 3. Overbought/Oversold bar chart
    total = len(df[range_col].dropna())
    if total > 0:
        oversold = (df[range_col] < 0.2).sum()
        neutral = ((df[range_col] >= 0.2) & (df[range_col] <= 0.8)).sum()
        overbought = (df[range_col] > 0.8).sum()

        fig.add_trace(
            go.Bar(
                x=["Oversold (<20%)", "Neutral (20-80%)", "Overbought (>80%)"],
                y=[oversold / total * 100, neutral / total * 100, overbought / total * 100],
                marker_color=["green", "gray", "red"],
                text=[f"{oversold}", f"{neutral}", f"{overbought}"],
                textposition="auto",
            ),
            row=3,
            col=1,
        )

    # 4. Range vs Momentum scatter
    momentum_col = "price_momentum_1m"
    if momentum_col in df.columns:
        plot_data = df[[range_col, momentum_col]].dropna()
        fig.add_trace(
            go.Scatter(
                x=plot_data[range_col],
                y=plot_data[momentum_col],
                mode="markers",
                marker=dict(
                    size=5,
                    opacity=0.5,
                    color=plot_data[momentum_col],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Momentum", x=1.0),
                ),
                name="Stocks",
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        title="52-Week Range Position Analysis",
        height=1400,
        width=1000,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="Range Position", row=1, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="Count", row=1, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="Range Position", row=2, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="% of Stocks", row=3, col=1, tickfont=dict(size=11))
    fig.update_xaxes(title_text="52W Range Position", row=4, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="1M Momentum (%)", row=4, col=1, tickfont=dict(size=11))

    return fig


def create_trend_strength_matrix(
    df: pd.DataFrame,
    trend_col: str = "long_term_trend_score",
    flag_col: str = "secular_trend_flag",
    group_col: str = "industry",
) -> go.Figure:
    """
    Matrix heatmap: long_term_trend_score vs secular_trend_flag by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with trend columns
    trend_col : str, default 'long_term_trend_score'
        Column for trend score
    flag_col : str, default 'secular_trend_flag'
        Column for secular trend flag
    group_col : str, default 'industry'
        Column to group by

    Returns
    -------
    go.Figure
        Plotly heatmap figure

    Examples
    --------
    >>> fig = create_trend_strength_matrix(df)
    >>> fig.show()
    """
    if trend_col not in df.columns or group_col not in df.columns:
        return create_no_data_figure("Trend Strength Matrix - Missing Data")

    # Calculate statistics by group
    agg_dict = {trend_col: ["mean", "median", "std", "count"]}

    if flag_col in df.columns:
        agg_dict[flag_col] = ["mean"]  # Percentage in secular uptrend

    stats = df.groupby(group_col).agg(agg_dict).reset_index()
    stats.columns = ["_".join(col).strip("_") for col in stats.columns.values]

    # Rename columns for clarity
    stats = stats.rename(
        columns={
            f"{trend_col}_mean": "avg_trend_score",
            f"{trend_col}_median": "median_trend_score",
            f"{trend_col}_std": "trend_volatility",
            f"{trend_col}_count": "stock_count",
        },
    )

    if f"{flag_col}_mean" in stats.columns:
        stats = stats.rename(columns={f"{flag_col}_mean": "pct_secular_uptrend"})
        stats["pct_secular_uptrend"] = stats["pct_secular_uptrend"] * 100

    # Filter groups with sufficient data
    stats = stats[stats["stock_count"] >= 5].sort_values("avg_trend_score", ascending=False)

    if len(stats) == 0:
        return create_no_data_figure("Trend Strength Matrix - Insufficient Data")

    # Create heatmap data
    metrics = ["avg_trend_score", "median_trend_score", "trend_volatility"]
    if "pct_secular_uptrend" in stats.columns:
        metrics.append("pct_secular_uptrend")

    metric_labels = {
        "avg_trend_score": "Avg Trend",
        "median_trend_score": "Median Trend",
        "trend_volatility": "Volatility",
        "pct_secular_uptrend": "% Uptrend",
    }

    z_data = stats[metrics].values

    # Normalize for visualization
    z_normalized = np.zeros_like(z_data)
    for i in range(z_data.shape[1]):
        col_data = z_data[:, i]
        if col_data.std() > 0:
            z_normalized[:, i] = (col_data - col_data.mean()) / col_data.std()
        else:
            z_normalized[:, i] = 0

    fig = go.Figure(
        data=go.Heatmap(
            z=z_normalized,
            x=[metric_labels.get(m, m) for m in metrics],
            y=stats[group_col].tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(z_data, 1),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=(
                "<b>%{y}</b><br>" + "%{x}: %{text}<br>" + "Z-score: %{z:.2f}<extra></extra>"
            ),
        ),
    )

    fig.update_layout(
        title=f"Trend Strength Matrix by {group_col.title()}",
        xaxis_title="Metric",
        yaxis_title=group_col.title(),
        height=max(400, len(stats) * 25 + 100),
        width=1000,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    return fig


def create_momentum_divergence_scatter(
    df: pd.DataFrame,
    short_term_col: str = "price_momentum_1m",
    long_term_col: str = "price_momentum_1y",
    color_by: str = "industry",
) -> go.Figure:
    """
    Scatter: Short-term momentum (1m) vs Long-term (1y) to spot divergences.

    Divergences indicate potential trend reversals:
    - Positive divergence: Short-term > Long-term (potential breakout)
    - Negative divergence: Short-term < Long-term (potential breakdown)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with momentum columns
    short_term_col : str, default 'price_momentum_1m'
        Column for short-term momentum
    long_term_col : str, default 'price_momentum_1y'
        Column for long-term momentum
    color_by : str, default 'industry'
        Column to color points by

    Returns
    -------
    go.Figure
        Plotly scatter figure with divergence analysis

    Examples
    --------
    >>> fig = create_momentum_divergence_scatter(df)
    >>> fig.show()
    """
    if short_term_col not in df.columns or long_term_col not in df.columns:
        return create_no_data_figure("Momentum Divergence - Missing Data")

    # Prepare data
    plot_df = df.dropna(subset=[short_term_col, long_term_col]).copy()

    # Calculate divergence
    plot_df["divergence"] = plot_df[short_term_col] - plot_df[long_term_col]

    # Classify divergence type
    plot_df["divergence_type"] = "Neutral"
    plot_df.loc[plot_df["divergence"] > 10, "divergence_type"] = "Positive (Breakout)"
    plot_df.loc[plot_df["divergence"] < -10, "divergence_type"] = "Negative (Breakdown)"

    # Create figure
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Short-term vs Long-term Momentum", "Divergence Distribution"),
        vertical_spacing=0.12,
    )

    # 1. Scatter plot
    colors = {"Positive (Breakout)": "green", "Neutral": "gray", "Negative (Breakdown)": "red"}

    for div_type in ["Positive (Breakout)", "Neutral", "Negative (Breakdown)"]:
        mask = plot_df["divergence_type"] == div_type
        subset = plot_df[mask]

        fig.add_trace(
            go.Scatter(
                x=subset[long_term_col],
                y=subset[short_term_col],
                mode="markers",
                marker=dict(size=8, color=colors[div_type], opacity=0.6),
                text=subset.get("ticker", subset.index),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + f"{long_term_col}: "
                    + "%{x:.1f}%<br>"
                    + f"{short_term_col}: "
                    + "%{y:.1f}%<br>"
                    + "<extra></extra>"
                ),
                name=div_type,
            ),
            row=1,
            col=1,
        )

    # Add diagonal line (no divergence)
    min_val = min(plot_df[short_term_col].min(), plot_df[long_term_col].min())
    max_val = max(plot_df[short_term_col].max(), plot_df[long_term_col].max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash", color="white", width=1),
            name="No Divergence",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # 2. Divergence histogram
    fig.add_trace(
        go.Histogram(
            x=plot_df["divergence"],
            nbinsx=30,
            marker_color="rgb(55, 128, 191)",
            opacity=0.7,
            name="Divergence",
        ),
        row=2,
        col=1,
    )

    # Add vertical lines for divergence thresholds
    fig.add_vline(x=10, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_vline(x=-10, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_vline(x=0, line_dash="solid", line_color="white", row=2, col=1)

    fig.update_layout(
        title="Momentum Divergence Analysis",
        height=800,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text=f"Long-term ({long_term_col})", row=1, col=1)
    fig.update_yaxes(title_text=f"Short-term ({short_term_col})", row=1, col=1)
    fig.update_xaxes(title_text="Divergence (Short - Long)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig
