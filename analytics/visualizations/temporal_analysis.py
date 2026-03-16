"""
Temporal and time series visualization module.

This module provides visualization functions for time-based analysis:
- Earnings calendar heatmaps
- Inventory cycle analysis
- FCF trajectory charts
- Dividend streak timelines

Feature Categories leveraged:
- Inventory Temporal (lines 175-180): inventory_days, inventory_turnover_mv,
  inventory_yoy_change, inventory_buildup_flag
- Cash Flow (lines 154-161): fcf_positive_years, fcf_margin, fcf_yield, fcf_growth_yoy
- Dividend Features (lines 162-168): dividend_streak, dividend_yield_ltm, dividend_payout_ratio
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
    resolve_column,
)

def create_earnings_calendar_heatmap(
    df: pd.DataFrame,
    date_col: str = "next_earnings",
    quality_col: str = "earnings_quality_composite",
) -> go.Figure:
    """
    Calendar heatmap of upcoming earnings dates with quality score overlay.

    Uses: next_earnings date column, composite quality scores

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with earnings date and quality columns
    date_col : str, default 'next_earnings'
        Column containing earnings dates
    quality_col : str, default 'earnings_quality_composite'
        Column for quality score overlay

    Returns
    -------
    go.Figure
        Plotly figure with earnings calendar heatmap

    Examples
    --------
    >>> fig = create_earnings_calendar_heatmap(df)
    >>> fig.show()
    """
    if date_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No earnings date column '{date_col}' available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(title="Earnings Calendar - No Data", template=PLOTLY_TEMPLATE)
        return fig

    # Prepare data
    plot_df = df.copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[date_col])

    if len(plot_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid earnings dates found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(title="Earnings Calendar - No Valid Dates", template=PLOTLY_TEMPLATE)
        return fig

    # Extract date components
    plot_df["week"] = plot_df[date_col].dt.isocalendar().week
    plot_df["weekday"] = plot_df[date_col].dt.dayofweek
    plot_df["month"] = plot_df[date_col].dt.month
    plot_df["year"] = plot_df[date_col].dt.year

    # Count earnings per day
    daily_counts = (
        plot_df.groupby([date_col])
        .agg(
            {"ticker": "count", quality_col: "mean" if quality_col in plot_df.columns else "count"}
        )
        .reset_index()
    )
    daily_counts.columns = ["date", "count", "avg_quality"]

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Earnings by Day of Week",
            "Earnings by Month",
            "Daily Earnings Count Timeline",
            "Quality Score Distribution by Week",
        ),
        specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "scatter"}], [{"type": "heatmap"}]],
        vertical_spacing=0.07,
    )

    # 1. Earnings by day of week
    weekday_counts = plot_df.groupby("weekday").size()
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig.add_trace(
        go.Bar(
            x=[weekday_labels[i] for i in weekday_counts.index],
            y=weekday_counts.values,
            marker_color="rgb(55, 128, 191)",
            name="By Weekday",
        ),
        row=1,
        col=1,
    )

    # 2. Earnings by month
    month_counts = plot_df.groupby("month").size()
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    fig.add_trace(
        go.Bar(
            x=[month_labels[i - 1] for i in month_counts.index],
            y=month_counts.values,
            marker_color="rgb(50, 171, 96)",
            name="By Month",
        ),
        row=2,
        col=1,
    )

    # 3. Timeline of daily earnings
    fig.add_trace(
        go.Scatter(
            x=daily_counts["date"],
            y=daily_counts["count"],
            mode="lines+markers",
            marker=dict(
                size=6,
                color=(
                    daily_counts["avg_quality"]
                    if quality_col in plot_df.columns
                    else "rgb(55, 128, 191)"
                ),
                colorscale="RdYlGn",
                showscale=True if quality_col in plot_df.columns else False,
                colorbar=dict(title="Avg Quality", x=0.45, y=0.2, len=0.3),
            ),
            line=dict(color="rgba(55, 128, 191, 0.5)"),
            name="Daily Count",
        ),
        row=3,
        col=1,
    )

    # 4. Weekly heatmap (week vs weekday)
    if quality_col in plot_df.columns:
        heatmap_data = plot_df.pivot_table(
            values=quality_col, index="weekday", columns="week", aggfunc="mean"
        ).fillna(0)

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=[f"W{w}" for w in heatmap_data.columns],
                y=[weekday_labels[i] for i in heatmap_data.index],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Quality", x=1.0, y=0.2, len=0.3),
            ),
            row=4,
            col=1,
        )
    else:
        # Show count heatmap instead
        heatmap_data = plot_df.pivot_table(
            values="ticker", index="weekday", columns="week", aggfunc="count"
        ).fillna(0)

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=[f"W{w}" for w in heatmap_data.columns],
                y=[weekday_labels[i] for i in heatmap_data.index],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Count", x=1.0, y=0.2, len=0.3),
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        title="Earnings Calendar Analysis",
        height=1600,
        width=1000,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="Day of Week", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Earnings Count", row=3, col=1)

    return fig


def create_inventory_cycle_analysis(df: pd.DataFrame, group_col: str = "industry") -> go.Figure:
    """
    Time series of inventory_days and inventory_turnover_itf trends.

    Flags inventory_buildup_flag anomalies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with inventory columns
    group_col : str, default 'industry'
        Column to group analysis by

    Returns
    -------
    go.Figure
        Plotly figure with inventory cycle analysis

    Examples
    --------
    >>> fig = create_inventory_cycle_analysis(df)
    >>> fig.show()
    """
    # Resolve inventory_turnover via alias map (may be inventory_turnover_itf or inventory_turnover)
    turnover_col = resolve_column(df, "inventory_turnover") or "inventory_turnover_itf"
    inventory_cols = ["inventory_days", turnover_col, "inventory_yoy_change"]
    available_cols = [col for col in inventory_cols if col in df.columns]

    if not available_cols:
        return create_no_data_figure("Inventory Cycle Analysis - No Data")

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Inventory Days Distribution",
            "Inventory Turnover by Sector",
            "Inventory YoY Change",
            "Buildup Flag Analysis",
        ),
        specs=[
            [{"type": "histogram"}],
            [{"type": "box"}],
            [{"type": "scatter"}],
            [{"type": "bar"}],
        ],
        vertical_spacing=0.07,
    )

    # 1. Inventory days histogram
    if "inventory_days" in df.columns:
        inv_days = df["inventory_days"].dropna()
        inv_days = inv_days[inv_days.between(0, 365)]  # Filter reasonable values

        fig.add_trace(
            go.Histogram(
                x=inv_days,
                nbinsx=30,
                marker_color="rgb(55, 128, 191)",
                opacity=0.7,
                name="Inventory Days",
            ),
            row=1,
            col=1,
        )

    # 2. Inventory turnover by sector
    if turnover_col in df.columns and group_col in df.columns:
        sectors = df[group_col].dropna().unique()[:8]
        for sector in sectors:
            sector_data = df[df[group_col] == sector][turnover_col].dropna()
            sector_data = sector_data[sector_data.between(0, 50)]  # Filter outliers
            if len(sector_data) > 5:
                fig.add_trace(
                    go.Box(y=sector_data, name=sector[:12], boxpoints="outliers"), row=2, col=1
                )

    # 3. Inventory YoY change scatter
    if "inventory_yoy_change" in df.columns:
        plot_data = df[["inventory_yoy_change"]].dropna()
        if "inventory_days" in df.columns:
            plot_data["inventory_days"] = df["inventory_days"]
            plot_data = plot_data.dropna()

            # Color by buildup flag if available
            if "inventory_buildup_flag" in df.columns:
                plot_data["buildup"] = df["inventory_buildup_flag"].fillna(0)
                colors = plot_data["buildup"].map({0: "green", 1: "red"})
            else:
                colors = "rgb(55, 128, 191)"

            fig.add_trace(
                go.Scatter(
                    x=plot_data["inventory_days"],
                    y=plot_data["inventory_yoy_change"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=colors if isinstance(colors, str) else colors.values,
                        opacity=0.6,
                    ),
                    text=df.get("ticker", df.index),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        + "Inv Days: %{x:.0f}<br>"
                        + "YoY Change: %{y:.1f}%<extra></extra>"
                    ),
                    name="Stocks",
                ),
                row=3,
                col=1,
            )

    # 4. Buildup flag analysis
    if "inventory_buildup_flag" in df.columns and group_col in df.columns:
        buildup_pct = df.groupby(group_col)["inventory_buildup_flag"].mean() * 100
        buildup_pct = buildup_pct.sort_values(ascending=False).head(10)

        fig.add_trace(
            go.Bar(
                x=buildup_pct.index.tolist(),
                y=buildup_pct.values,
                marker_color="rgb(219, 64, 82)",
                text=[f"{v:.1f}%" for v in buildup_pct.values],
                textposition="auto",
                name="Buildup %",
            ),
            row=4,
            col=1,
        )
    elif "inventory_yoy_change" in df.columns and group_col in df.columns:
        # Show average YoY change by sector instead
        avg_change = df.groupby(group_col)["inventory_yoy_change"].mean()
        avg_change = avg_change.sort_values(ascending=False).head(10)

        colors = ["red" if v > 0 else "green" for v in avg_change.values]

        fig.add_trace(
            go.Bar(
                x=avg_change.index.tolist(),
                y=avg_change.values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in avg_change.values],
                textposition="auto",
                name="Avg YoY Change",
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        title="Inventory Cycle Analysis",
        height=1600,
        width=1000,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="Inventory Days", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Turnover Ratio", row=2, col=1)
    fig.update_xaxes(title_text="Inventory Days", row=3, col=1)
    fig.update_yaxes(title_text="YoY Change (%)", row=3, col=1)
    fig.update_xaxes(title_text="Sector", tickangle=-45, row=4, col=1)
    fig.update_yaxes(title_text="% with Buildup", row=4, col=1)

    return fig


def create_fcf_trajectory_chart(df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """
    FCF positive years streak visualization with fcf_growth_yoy overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with FCF columns
    top_n : int, default 30
        Number of top stocks to display

    Returns
    -------
    go.Figure
        Plotly figure with FCF trajectory analysis

    Examples
    --------
    >>> fig = create_fcf_trajectory_chart(df)
    >>> fig.show()
    """
    fcf_cols = ["fcf_positive_years", "fcf_margin", "fcf_yield", "fcf_growth_yoy"]
    available_cols = [col for col in fcf_cols if col in df.columns]

    if "fcf_positive_years" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No FCF positive years data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(title="FCF Trajectory - No Data", template=PLOTLY_TEMPLATE)
        return fig

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "FCF Positive Years Distribution",
            "Top Stocks by FCF Streak",
            "FCF Margin vs FCF Yield",
            "FCF Growth Distribution",
        ),
        specs=[
            [{"type": "histogram"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
            [{"type": "histogram"}],
        ],
        vertical_spacing=0.07,
    )

    # 1. FCF positive years distribution
    fig.add_trace(
        go.Histogram(
            x=df["fcf_positive_years"].dropna(),
            nbinsx=6,
            marker_color="rgb(50, 171, 96)",
            opacity=0.7,
            name="FCF Years",
        ),
        row=1,
        col=1,
    )

    # 2. Top stocks by FCF streak
    top_fcf = df.nlargest(top_n, "fcf_positive_years")

    if "ticker" in top_fcf.columns:
        labels = top_fcf["ticker"].tolist()[:15]  # Show top 15
    else:
        labels = [f"Stock {i}" for i in range(min(15, len(top_fcf)))]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=top_fcf["fcf_positive_years"].head(15).values,
            marker_color="rgb(50, 171, 96)",
            text=top_fcf["fcf_positive_years"].head(15).values,
            textposition="auto",
            name="FCF Streak",
        ),
        row=2,
        col=1,
    )

    # 3. FCF Margin vs FCF Yield scatter
    if "fcf_margin" in df.columns and "fcf_yield" in df.columns:
        plot_data = df[["fcf_margin", "fcf_yield", "fcf_positive_years"]].dropna()
        plot_data = plot_data[
            (plot_data["fcf_margin"].between(-50, 50)) & (plot_data["fcf_yield"].between(-20, 30))
        ]

        fig.add_trace(
            go.Scatter(
                x=plot_data["fcf_margin"],
                y=plot_data["fcf_yield"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=plot_data["fcf_positive_years"],
                    colorscale="Greens",
                    showscale=True,
                    colorbar=dict(title="FCF Years", x=0.45, y=0.2, len=0.3),
                ),
                text=df.loc[plot_data.index].get("ticker", plot_data.index),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + "FCF Margin: %{x:.1f}%<br>"
                    + "FCF Yield: %{y:.1f}%<br>"
                    + "<extra></extra>"
                ),
                name="Stocks",
            ),
            row=3,
            col=1,
        )

        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=3, col=1)

    # 4. FCF Growth distribution
    if "fcf_growth_yoy" in df.columns:
        fcf_growth = df["fcf_growth_yoy"].dropna()
        fcf_growth = fcf_growth[fcf_growth.between(-100, 200)]  # Filter outliers

        fig.add_trace(
            go.Histogram(
                x=fcf_growth,
                nbinsx=30,
                marker_color="rgb(55, 128, 191)",
                opacity=0.7,
                name="FCF Growth",
            ),
            row=4,
            col=1,
        )

        # Add zero line
        fig.add_vline(x=0, line_dash="solid", line_color="black", row=4, col=1)

    fig.update_layout(
        title="Free Cash Flow Trajectory Analysis",
        height=1400,
        width=1000,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="FCF Positive Years", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Stock", tickangle=-45, row=2, col=1)
    fig.update_yaxes(title_text="FCF Years", row=2, col=1)
    fig.update_xaxes(title_text="FCF Margin (%)", row=3, col=1)
    fig.update_yaxes(title_text="FCF Yield (%)", row=3, col=1)
    fig.update_xaxes(title_text="FCF Growth YoY (%)", row=4, col=1)
    fig.update_yaxes(title_text="Count", row=4, col=1)

    return fig


def create_dividend_streak_timeline(df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """
    Timeline chart of dividend_streak with payout sustainability metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with dividend columns
    top_n : int, default 30
        Number of top dividend stocks to display

    Returns
    -------
    go.Figure
        Plotly figure with dividend streak analysis

    Examples
    --------
    >>> fig = create_dividend_streak_timeline(df)
    >>> fig.show()
    """
    dividend_cols = ["dividend_streak", "dividend_yield_ltm", "dividend_payout_ratio"]
    available_cols = [col for col in dividend_cols if col in df.columns]

    if "dividend_streak" not in df.columns:
        # Try alternative column name
        if "dividend_years" in df.columns:
            df = df.copy()
            df["dividend_streak"] = df["dividend_years"]
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="No dividend streak data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(title="Dividend Streak Timeline - No Data", template=PLOTLY_TEMPLATE)
            return fig

    # Filter stocks with dividends
    div_df = df[df["dividend_streak"] > 0].copy()

    if len(div_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No stocks with dividend streaks found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title="Dividend Streak Timeline - No Dividend Stocks", template=PLOTLY_TEMPLATE
        )
        return fig

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Dividend Streak Distribution",
            "Top Dividend Aristocrats",
            "Yield vs Payout Ratio",
            "Dividend Sustainability Matrix",
        ),
        specs=[
            [{"type": "histogram"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
            [{"type": "heatmap"}],
        ],
        vertical_spacing=0.07,
    )

    # 1. Dividend streak distribution
    fig.add_trace(
        go.Histogram(
            x=div_df["dividend_streak"],
            nbinsx=20,
            marker_color="rgb(148, 103, 189)",
            opacity=0.7,
            name="Streak Years",
        ),
        row=1,
        col=1,
    )

    # 2. Top dividend aristocrats
    top_div = div_df.nlargest(min(top_n, 15), "dividend_streak")

    if "ticker" in top_div.columns:
        labels = top_div["ticker"].tolist()
    else:
        labels = [f"Stock {i}" for i in range(len(top_div))]

    # Color by yield if available
    if "dividend_yield_ltm" in top_div.columns:
        colors = top_div["dividend_yield_ltm"].fillna(0)
        colorscale = "Greens"
    else:
        colors = "rgb(148, 103, 189)"
        colorscale = None

    fig.add_trace(
        go.Bar(
            x=labels,
            y=top_div["dividend_streak"].values,
            marker=dict(
                color=colors if isinstance(colors, str) else colors.values,
                colorscale=colorscale,
                showscale=True if colorscale else False,
                colorbar=dict(title="Yield %", x=0.45) if colorscale else None,
            ),
            text=top_div["dividend_streak"].values,
            textposition="auto",
            name="Streak",
        ),
        row=2,
        col=1,
    )

    # 3. Yield vs Payout Ratio scatter
    yield_col = "dividend_yield_ltm" if "dividend_yield_ltm" in div_df.columns else "dividend_yield"

    if yield_col in div_df.columns and "dividend_payout_ratio" in div_df.columns:
        plot_data = div_df[[yield_col, "dividend_payout_ratio", "dividend_streak"]].dropna()
        plot_data = plot_data[
            (plot_data[yield_col].between(0, 15))
            & (plot_data["dividend_payout_ratio"].between(0, 150))
        ]

        fig.add_trace(
            go.Scatter(
                x=plot_data["dividend_payout_ratio"],
                y=plot_data[yield_col],
                mode="markers",
                marker=dict(
                    size=plot_data["dividend_streak"] + 5,
                    sizemode="area",
                    sizeref=2.0 * max(plot_data["dividend_streak"]) / (40.0**2),
                    sizemin=4,
                    color=plot_data["dividend_streak"],
                    colorscale="Purples",
                    showscale=True,
                    colorbar=dict(title="Streak", x=1.0, y=0.7, len=0.3),
                ),
                text=div_df.loc[plot_data.index].get("ticker", plot_data.index),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + "Payout Ratio: %{x:.1f}%<br>"
                    + "Yield: %{y:.2f}%<br>"
                    + "<extra></extra>"
                ),
                name="Stocks",
            ),
            row=3,
            col=1,
        )

        # Add sustainability threshold lines
        fig.add_vline(x=80, line_dash="dash", line_color="orange", row=3, col=1)
        fig.add_annotation(
            x=80,
            y=plot_data[yield_col].max(),
            text="Sustainability Threshold",
            showarrow=False,
            font=dict(size=10, color="orange"),
            row=3,
            col=1,
        )

    # 4. Sustainability matrix by industry
    if "industry" in div_df.columns:
        # Calculate sustainability metrics by industry
        industry_stats = (
            div_df.groupby("industry")
            .agg(
                {
                    "dividend_streak": "mean",
                    yield_col: "mean" if yield_col in div_df.columns else "count",
                    "dividend_payout_ratio": (
                        "mean" if "dividend_payout_ratio" in div_df.columns else "count"
                    ),
                }
            )
            .dropna()
        )

        # Filter industries with enough data
        industry_counts = div_df.groupby("industry").size()
        valid_industries = industry_counts[industry_counts >= 3].index
        industry_stats = industry_stats.loc[industry_stats.index.isin(valid_industries)]

        if len(industry_stats) > 0:
            # Sort by streak
            industry_stats = industry_stats.sort_values("dividend_streak", ascending=False).head(12)

            # Normalize for heatmap
            z_data = industry_stats.values
            z_normalized = np.zeros_like(z_data)
            for i in range(z_data.shape[1]):
                col_data = z_data[:, i]
                if col_data.std() > 0:
                    z_normalized[:, i] = (col_data - col_data.mean()) / col_data.std()
                else:
                    z_normalized[:, i] = 0

            fig.add_trace(
                go.Heatmap(
                    z=z_normalized,
                    x=["Avg Streak", "Avg Yield", "Avg Payout"],
                    y=industry_stats.index.tolist(),
                    colorscale="RdYlGn",
                    zmid=0,
                    text=np.round(z_data, 1),
                    texttemplate="%{text}",
                    textfont={"size": 9},
                    showscale=False,
                ),
                row=4,
                col=1,
            )

    fig.update_layout(
        title="Dividend Streak & Sustainability Analysis",
        height=1400,
        width=1000,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="Dividend Streak (Years)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Stock", tickangle=-45, row=2, col=1)
    fig.update_yaxes(title_text="Streak (Years)", row=2, col=1)
    fig.update_xaxes(title_text="Payout Ratio (%)", row=3, col=1)
    fig.update_yaxes(title_text="Dividend Yield (%)", row=3, col=1)

    return fig
