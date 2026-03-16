"""
Earnings quality visualization module for deep-dive earnings analysis.

This module provides visualization functions for analyzing earnings quality:
- Earnings surprise dashboards (distribution, beat rate, momentum correlation)
- EPS trajectory analysis (improvement counts, streak analysis)
- Earnings quality decomposition (accruals, cash conversion, persistence)
- Beat rate heatmaps by sector
- Earnings consistency matrices
- Revision momentum charts (forward-looking analyst revision signals)
- GAAP-vs-Norm divergence plots (accounting quality guard)
- Enhanced beat probability dashboards (three-layer evidence fusion)

Feature Categories leveraged (from feature_registry.sql vw_features_earnings):
- eps_surprise_pct, eps_positive_years, eps_positive_streak, eps_trajectory_score
- eps_positive_streak, eps_improvement_count, earnings_quality_composite
- earnings_quality_composite, ni_adjustment_ratio, eps_adjustment_ratio
- accounting_quality_score, earnings_quality_impact
- eps_revision_momentum, gaap_adj_eps_gap_pct, revision_trend_short/medium
- posterior_beat_prob, quarterly_beat_streak, historical_beat_rate
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

# Metric display names
METRIC_LABELS = {
    "eps_surprise_pct": "EPS Surprise %",
    "eps_beat_count": "EPS Beat Count",
    "eps_positive_years": "Positive EPS Years",
    "eps_total_reports": "Total Reports",
    "eps_positive_streak": "Positive Streak",
    "eps_trajectory_score": "EPS Trajectory Score",
    "eps_improvement_count": "Improvement Count",
    "earnings_quality_composite": "Quality Composite",
    "ni_adjustment_ratio": "NI Adjustment Ratio",
    "eps_adjustment_ratio": "EPS Adjustment Ratio",
    "accounting_quality_score": "Accounting Quality Score",
    "earnings_quality_impact": "Unusual Items Impact",
}


def create_earnings_surprise_dashboard(
    df: pd.DataFrame,
    group_col: str = "industry",
) -> go.Figure:
    """
    Multi-panel dashboard for earnings surprise analysis.

    Panels:
    - Surprise distribution histogram
    - Beat rate by sector
    - Surprise vs momentum correlation

    Uses: eps_surprise_pct, eps_positive_years (or eps_beat_count), eps_positive_streak (or eps_total_reports)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with earnings columns
    group_col : str, default 'industry'
        Column to group by for sector analysis

    Returns
    -------
    go.Figure
        Plotly figure with multiple panels

    Examples
    --------
    >>> fig = create_earnings_surprise_dashboard(df)
    >>> fig.show()
    """
    has_surprise = "eps_surprise_pct" in df.columns
    beat_col = resolve_column(df, "eps_beat_count") or resolve_column(df, "eps_positive_years")
    total_col = resolve_column(df, "eps_total_reports") or resolve_column(df, "eps_positive_streak")
    has_beat = beat_col is not None and total_col is not None
    has_group = group_col in df.columns

    if not has_surprise and not has_beat:
        return create_no_data_figure("Earnings Surprise Dashboard - No Data")

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "EPS Surprise Distribution",
            "Beat Rate by Sector",
            "Surprise vs Quality Score",
            "Beat Count Distribution",
        ),
        vertical_spacing=0.08,
    )

    # Panel 1: Surprise Distribution
    if has_surprise:
        surprise_data = df["eps_surprise_pct"].dropna()
        if len(surprise_data) > 0:
            fig.add_trace(
                go.Histogram(
                    x=surprise_data,
                    name="EPS Surprise",
                    marker_color="#0A7EA4",
                    opacity=0.7,
                    nbinsx=30,
                ),
                row=1,
                col=1,
            )
            # Add vertical line at 0
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="white",
                row=1,
                col=1,
            )

    # Panel 2: Beat Rate by Sector
    if has_beat and has_group:
        groups = df[group_col].dropna().unique()
        beat_rates = []
        group_names = []

        for group in groups:
            group_data = df[df[group_col] == group]
            total_beats = group_data[beat_col].sum()
            total_reports = group_data[total_col].sum()
            if total_reports > 0:
                beat_rate = (total_beats / total_reports) * 100
                beat_rates.append(beat_rate)
                group_names.append(str(group))

        if beat_rates:
            # Sort by beat rate
            sorted_idx = np.argsort(beat_rates)[::-1]
            beat_rates = [beat_rates[i] for i in sorted_idx]
            group_names = [group_names[i] for i in sorted_idx]

            fig.add_trace(
                go.Bar(
                    x=group_names,
                    y=beat_rates,
                    name="Beat Rate",
                    marker_color="#00A878",
                ),
                row=2,
                col=1,
            )

    # Panel 3: Surprise vs Quality Score
    if has_surprise and "earnings_quality_composite" in df.columns:
        plot_df = df[["eps_surprise_pct", "earnings_quality_composite"]].dropna()
        if len(plot_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["eps_surprise_pct"],
                    y=plot_df["earnings_quality_composite"],
                    mode="markers",
                    name="Surprise vs Quality",
                    marker=dict(
                        size=8,
                        color="#6C63FF",
                        opacity=0.6,
                    ),
                ),
                row=3,
                col=1,
            )

    # Panel 4: Beat Count Distribution
    if has_beat:
        beat_data = df[beat_col].dropna()
        if len(beat_data) > 0:
            fig.add_trace(
                go.Histogram(
                    x=beat_data,
                    name="Beat Count",
                    marker_color="#FF6B6B",
                    opacity=0.7,
                ),
                row=4,
                col=1,
            )

    fig.update_layout(
        title="Earnings Surprise Analysis Dashboard",
        template=PLOTLY_TEMPLATE,
        height=1400,
        width=1000,
        showlegend=False,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    # Update axis labels
    fig.update_xaxes(title_text="Surprise %", row=1, col=1)
    fig.update_xaxes(title_text="Sector", row=2, col=1)
    fig.update_xaxes(title_text="Surprise %", row=3, col=1)
    fig.update_xaxes(title_text="Beat Count", row=4, col=1)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Beat Rate %", row=2, col=1)
    fig.update_yaxes(title_text="Quality Score", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=4, col=1)

    return fig


def create_eps_trajectory_analysis(
    df: pd.DataFrame,
    top_n: int = 30,
) -> go.Figure:
    """
    Trajectory score visualization with improvement counts and streak analysis.

    Uses: eps_trajectory_score, eps_positive_streak, eps_improvement_count

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with trajectory columns
    top_n : int, default 30
        Number of top stocks to display

    Returns
    -------
    go.Figure
        Plotly figure with trajectory analysis

    Examples
    --------
    >>> fig = create_eps_trajectory_analysis(df, top_n=20)
    >>> fig.show()
    """
    required_cols = ["eps_trajectory_score"]
    available_cols = [col for col in required_cols if col in df.columns]

    if not available_cols:
        return create_no_data_figure("EPS Trajectory Analysis - No Data")

    # Sort by trajectory score and get top N
    plot_df = df.dropna(subset=["eps_trajectory_score"]).copy()
    if len(plot_df) == 0:
        return create_no_data_figure("EPS Trajectory Analysis - No Data")

    plot_df = plot_df.nlargest(top_n, "eps_trajectory_score")

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": True}]],
    )

    # Get labels
    if "ticker" in plot_df.columns:
        labels = plot_df["ticker"].tolist()
    elif "name" in plot_df.columns:
        labels = plot_df["name"].tolist()
    else:
        labels = [f"Stock {i}" for i in range(len(plot_df))]

    # Add trajectory score bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=plot_df["eps_trajectory_score"],
            name="Trajectory Score",
            marker_color="#0A7EA4",
            opacity=0.8,
        ),
        secondary_y=False,
    )

    # Add positive streak line if available
    if "eps_positive_streak" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=plot_df["eps_positive_streak"],
                name="Positive Streak",
                mode="lines+markers",
                line=dict(color="#00A878", width=2),
                marker=dict(size=8),
            ),
            secondary_y=True,
        )

    # Add improvement count markers if available
    if "eps_improvement_count" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=plot_df["eps_improvement_count"],
                name="Improvement Count",
                mode="markers",
                marker=dict(
                    size=12,
                    color="#FF6B6B",
                    symbol="diamond",
                ),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=f"EPS Trajectory Analysis (Top {top_n})",
        template=PLOTLY_TEMPLATE,
        height=500,
        xaxis_tickangle=-45,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="Trajectory Score", secondary_y=False)
    fig.update_yaxes(title_text="Streak / Count", secondary_y=True)

    return fig


def create_earnings_quality_decomposition(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> go.Figure:
    """
    Waterfall-style breakdown of earnings quality components.

    Shows: Earnings quality composite, NI adjustment ratio, EPS adjustment ratio,
           accounting quality score, and earnings quality impact from unusual items.

    Uses columns available in mv_all_stock_features:
    - earnings_quality_composite (from calc_net_income_comprehensive)
    - ni_adjustment_ratio (from calc_net_income_comprehensive)
    - eps_adjustment_ratio (from calc_earnings_features / calc_eps_comprehensive)
    - accounting_quality_score (from calc_accounting_quality_features)
    - earnings_quality_impact (from calc_unusual_items_features)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with quality columns
    ticker : str, optional
        Specific ticker to analyze. If None, shows median values

    Returns
    -------
    go.Figure
        Plotly waterfall/bar chart figure

    Examples
    --------
    >>> fig = create_earnings_quality_decomposition(df, ticker='AAPL')
    >>> fig.show()
    """
    # Map to columns that actually exist in mv_all_stock_features
    quality_cols = [
        "earnings_quality_composite",
        "ni_adjustment_ratio",
        "eps_adjustment_ratio",
        "accounting_quality_score",
        "earnings_quality_impact",
    ]
    col_labels = {
        "earnings_quality_composite": "Earnings Quality Composite",
        "ni_adjustment_ratio": "NI Adjustment Ratio",
        "eps_adjustment_ratio": "EPS Adjustment Ratio",
        "accounting_quality_score": "Accounting Quality Score",
        "earnings_quality_impact": "Unusual Items Impact",
    }
    available_cols = [col for col in quality_cols if col in df.columns]

    if not available_cols:
        return create_no_data_figure("Earnings Quality Decomposition - No Data")

    # Get values
    title_suffix = "(Median Values)"
    if ticker and "ticker" in df.columns:
        stock_data = df[df["ticker"] == ticker]
        if len(stock_data) > 0:
            values = {col: stock_data[col].iloc[0] for col in available_cols}
            title_suffix = f"({ticker})"
        else:
            values = {col: df[col].median() for col in available_cols}
    else:
        values = {col: df[col].median() for col in available_cols}

    medians = {col: df[col].median() for col in available_cols}

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[col_labels.get(col, col) for col in available_cols],
            y=[values.get(col, 0) for col in available_cols],
            name="Selected" if not ticker else ticker,
            marker_color="#0A7EA4",
        )
    )

    fig.add_trace(
        go.Bar(
            x=[col_labels.get(col, col) for col in available_cols],
            y=[medians.get(col, 0) for col in available_cols],
            name="Median",
            marker_color="rgba(100, 100, 100, 0.5)",
        )
    )

    # Add quality interpretation annotations
    annotations = []
    for i, col in enumerate(available_cols):
        val = values.get(col, 0)
        if pd.notna(val):
            if col == "earnings_quality_composite":
                quality = "Good" if val > 70 else "Caution" if val > 40 else "Poor"
                color = "#00A878" if val > 70 else "#FFD93D" if val > 40 else "#FF6B6B"
            elif col in ("ni_adjustment_ratio", "eps_adjustment_ratio"):
                abs_val = abs(val) if pd.notna(val) else 0
                quality = "Good" if abs_val < 0.1 else "Caution" if abs_val < 0.3 else "Poor"
                color = "#00A878" if abs_val < 0.1 else "#FFD93D" if abs_val < 0.3 else "#FF6B6B"
            elif col == "accounting_quality_score":
                quality = "Good" if val > 70 else "Caution" if val > 40 else "Poor"
                color = "#00A878" if val > 70 else "#FFD93D" if val > 40 else "#FF6B6B"
            else:  # earnings_quality_impact
                abs_val = abs(val) if pd.notna(val) else 0
                quality = "Good" if abs_val < 0.05 else "Caution" if abs_val < 0.15 else "Poor"
                color = "#00A878" if abs_val < 0.05 else "#FFD93D" if abs_val < 0.15 else "#FF6B6B"

            annotations.append(
                dict(
                    x=i,
                    y=val,
                    text=quality,
                    showarrow=False,
                    yshift=15,
                    font=dict(size=10, color=color),
                )
            )

    fig.update_layout(
        title=f"Earnings Quality Decomposition {title_suffix}",
        template=PLOTLY_TEMPLATE,
        barmode="group",
        height=450,
        annotations=annotations,
        showlegend=True,
    )
    fig.update_yaxes(title_text="Value")

    return fig


def create_beat_rate_heatmap(
    df: pd.DataFrame,
    group_col: str = "industry",
) -> go.Figure:
    """
    Heatmap showing earnings consistency metrics by sector.

    Refactored to use columns available in mv_all_stock_features:
    - eps_positive_years (from calc_eps_comprehensive)
    - eps_positive_streak (from calc_eps_trajectory_features)
    - eps_improvement_count (from calc_eps_trajectory_features)
    - eps_trajectory_score (from calc_eps_trajectory_features)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with earnings consistency columns
    group_col : str, default 'industry'
        Column to group by

    Returns
    -------
    go.Figure
        Plotly heatmap figure

    Examples
    --------
    >>> fig = create_beat_rate_heatmap(df, group_col='sector')
    >>> fig.show()
    """
    consistency_cols = [
        "eps_positive_years",
        "eps_positive_streak",
        "eps_improvement_count",
        "eps_trajectory_score",
    ]
    available_cols = [col for col in consistency_cols if col in df.columns]

    if not available_cols or group_col not in df.columns:
        return create_no_data_figure("Earnings Consistency Heatmap - No Data")

    groups = df[group_col].dropna().unique()
    data_matrix = []

    for group in groups:
        group_data = df[df[group_col] == group]
        row = []
        for col in available_cols:
            median_val = group_data[col].median()
            row.append(median_val if pd.notna(median_val) else 0)
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    col_labels = {
        "eps_positive_years": "Positive EPS Years",
        "eps_positive_streak": "Positive Streak",
        "eps_improvement_count": "Improvement Count",
        "eps_trajectory_score": "Trajectory Score",
    }

    fig = go.Figure(
        data=go.Heatmap(
            z=data_matrix,
            x=[col_labels.get(col, col) for col in available_cols],
            y=list(groups),
            colorscale="RdYlGn",
            text=np.round(data_matrix, 1),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Sector: %{y}<br>Metric: %{x}<br>Value: %{z:.1f}<extra></extra>",
            colorbar=dict(title="Value"),
        )
    )

    fig.update_layout(
        title="Earnings Consistency by Sector",
        xaxis_title="Metric",
        yaxis_title="Sector",
        template=PLOTLY_TEMPLATE,
        height=max(400, len(groups) * 40),
    )

    return fig


def create_earnings_consistency_matrix(
    df: pd.DataFrame,
    group_col: str = "industry",
) -> go.Figure:
    """
    Matrix showing eps_positive_streak vs eps_improvement_count by sector.

    Uses: eps_positive_streak, eps_improvement_count

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with streak columns
    group_col : str, default 'industry'
        Column to group by

    Returns
    -------
    go.Figure
        Plotly scatter/bubble chart figure

    Examples
    --------
    >>> fig = create_earnings_consistency_matrix(df, group_col='sector')
    >>> fig.show()
    """
    if "eps_positive_streak" not in df.columns or "eps_improvement_count" not in df.columns:
        return create_no_data_figure("Earnings Consistency Matrix - No Data")

    fig = go.Figure()

    if group_col in df.columns:
        groups = df[group_col].dropna().unique()

        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group]

            # Calculate group averages
            avg_streak = group_data["eps_positive_streak"].mean()
            avg_improvement = group_data["eps_improvement_count"].mean()
            count = len(group_data)

            fig.add_trace(
                go.Scatter(
                    x=[avg_improvement],
                    y=[avg_streak],
                    mode="markers+text",
                    name=str(group),
                    text=[str(group)],
                    textposition="top center",
                    marker=dict(
                        size=np.sqrt(count) * 5 + 10,
                        color=COLORS[i % len(COLORS)],
                        opacity=0.7,
                    ),
                    hovertemplate=f"{group}<br>Avg Streak: {avg_streak:.1f}<br>Avg Improvement: {avg_improvement:.1f}<br>Count: {count}<extra></extra>",
                )
            )
    else:
        # Plot individual stocks
        plot_df = df[["eps_positive_streak", "eps_improvement_count"]].dropna()
        if len(plot_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["eps_improvement_count"],
                    y=plot_df["eps_positive_streak"],
                    mode="markers",
                    name="Stocks",
                    marker=dict(
                        size=10,
                        color="#0A7EA4",
                        opacity=0.6,
                    ),
                )
            )

    # Add quadrant lines at medians
    if "eps_positive_streak" in df.columns and "eps_improvement_count" in df.columns:
        streak_median = df["eps_positive_streak"].median()
        improvement_median = df["eps_improvement_count"].median()

        fig.add_hline(
            y=streak_median,
            line_dash="dash",
            line_color="rgba(255,255,255,0.5)",
            annotation_text=f"Median Streak: {streak_median:.1f}",
        )
        fig.add_vline(
            x=improvement_median,
            line_dash="dash",
            line_color="rgba(255,255,255,0.5)",
            annotation_text=f"Median Improvement: {improvement_median:.1f}",
        )

    fig.update_layout(
        title="Earnings Consistency Matrix (Streak vs Improvement)",
        xaxis_title="Average Improvement Count",
        yaxis_title="Average Positive Streak",
        template=PLOTLY_TEMPLATE,
        height=500,
        showlegend=True,
    )

    return fig


def create_revision_momentum_chart(
    df: pd.DataFrame,
    top_n: int = 30,
    group_col: str = "sector",
) -> go.Figure:
    """
    Visualize analyst revision momentum scores from forward estimate signals.

    Shows revision momentum score distribution, sector averages, and
    short/medium-term revision trend acceleration for top stocks.

    Uses: eps_revision_momentum, revision_trend_short, revision_trend_medium,
          posterior_beat_prob

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced DataFrame from EarningsBeatProbabilityModel.analyze_dataframe_enhanced
        or any DataFrame containing eps_revision_momentum column.
    top_n : int, default 30
        Number of top/bottom stocks to display in the bar chart.
    group_col : str, default 'sector'
        Column to group by for sector-level aggregation.

    Returns
    -------
    go.Figure
        Plotly figure with revision momentum analysis panels.

    Examples
    --------
    >>> fig = create_revision_momentum_chart(enhanced_df)
    >>> fig.show()
    """
    if "eps_revision_momentum" not in df.columns:
        return create_no_data_figure("Revision Momentum Chart - No Data")

    has_trends = "revision_trend_short" in df.columns and "revision_trend_medium" in df.columns
    has_posterior = "posterior_beat_prob" in df.columns
    has_group = group_col in df.columns

    n_rows = 2 + int(has_trends)
    subplot_titles = [
        "Revision Momentum Score Distribution",
        f"Top/Bottom {top_n} by Revision Momentum",
    ]
    if has_trends:
        subplot_titles.append("Short-Term vs Medium-Term Revision Trend")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=tuple(subplot_titles),
        vertical_spacing=0.10,
    )

    momentum = df["eps_revision_momentum"].dropna()

    # Panel 1: Momentum score distribution with sector overlay
    if has_group and len(momentum) > 0:
        groups = df[group_col].dropna().unique()
        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group]["eps_revision_momentum"].dropna()
            if len(group_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=group_data,
                        name=str(group),
                        marker_color=COLORS[i % len(COLORS)],
                        opacity=0.6,
                        nbinsx=20,
                    ),
                    row=1,
                    col=1,
                )
        fig.update_layout(barmode="overlay")
    elif len(momentum) > 0:
        fig.add_trace(
            go.Histogram(
                x=momentum,
                name="Momentum Score",
                marker_color="#0A7EA4",
                opacity=0.7,
                nbinsx=25,
            ),
            row=1,
            col=1,
        )

    # Add neutral line at 50
    fig.add_vline(
        x=50,
        line_dash="dash",
        line_color="white",
        row=1,
        col=1,
    )

    # Panel 2: Top/bottom stocks bar chart
    plot_df = df.dropna(subset=["eps_revision_momentum"]).copy()
    if len(plot_df) > 0:
        top = plot_df.nlargest(min(top_n, len(plot_df)), "eps_revision_momentum")
        bottom = plot_df.nsmallest(min(top_n, len(plot_df)), "eps_revision_momentum")
        bar_df = pd.concat([top, bottom]).drop_duplicates()
        bar_df = bar_df.sort_values("eps_revision_momentum", ascending=True)

        labels = (
            bar_df["ticker"].tolist()
            if "ticker" in bar_df.columns
            else [f"Stock {i}" for i in range(len(bar_df))]
        )
        scores = bar_df["eps_revision_momentum"].tolist()
        bar_colors = ["#00A878" if s >= 50 else "#E63946" for s in scores]

        fig.add_trace(
            go.Bar(
                y=labels,
                x=scores,
                orientation="h",
                name="Momentum Score",
                marker_color=bar_colors,
            ),
            row=2,
            col=1,
        )
        fig.add_vline(
            x=50,
            line_dash="dash",
            line_color="white",
            row=2,
            col=1,
        )

    # Panel 3: Short vs medium trend scatter
    if has_trends:
        trend_df = df[["revision_trend_short", "revision_trend_medium"]].dropna()
        if len(trend_df) > 0:
            color_vals = (
                df.loc[trend_df.index, "eps_revision_momentum"]
                if "eps_revision_momentum" in df.columns
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=trend_df["revision_trend_medium"],
                    y=trend_df["revision_trend_short"],
                    mode="markers",
                    name="Revision Trends",
                    marker=dict(
                        size=8,
                        color=color_vals,
                        colorscale="RdYlGn",
                        showscale=True,
                        colorbar=dict(title="Momentum"),
                        opacity=0.7,
                    ),
                    text=(df.loc[trend_df.index, "ticker"] if "ticker" in df.columns else None),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Medium Trend: %{x:.2f}<br>"
                        "Short Trend: %{y:.2f}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )
            # Add quadrant lines at 0
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="rgba(255,255,255,0.4)",
                row=3,
                col=1,
            )
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="rgba(255,255,255,0.4)",
                row=3,
                col=1,
            )

    fig.update_layout(
        title="Analyst Revision Momentum Analysis",
        template=PLOTLY_TEMPLATE,
        height=400 * n_rows,
        width=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="Momentum Score (0-100)", row=1, col=1)
    fig.update_xaxes(title_text="Momentum Score", row=2, col=1)
    if has_trends:
        fig.update_xaxes(title_text="Medium-Term Trend (1M - 3M)", row=3, col=1)
        fig.update_yaxes(title_text="Short-Term Trend (1W - 1M)", row=3, col=1)

    return fig


def create_gaap_divergence_plot(
    df: pd.DataFrame,
    group_col: str = "sector",
) -> go.Figure:
    """
    Visualize GAAP-vs-Normalized EPS divergence as an accounting quality guard.

    Shows the spread between GAAP and Normalized forward estimates, highlighting
    stocks with large divergence that may indicate accounting quality concerns.

    Uses: gaap_adj_eps_gap_pct, posterior_beat_prob, eps_revision_momentum

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced DataFrame containing gaap_adj_eps_gap_pct column.
    group_col : str, default 'sector'
        Column to group by for sector-level analysis.

    Returns
    -------
    go.Figure
        Plotly figure with GAAP divergence analysis panels.

    Examples
    --------
    >>> fig = create_gaap_divergence_plot(enhanced_df)
    >>> fig.show()
    """
    if "gaap_adj_eps_gap_pct" not in df.columns:
        return create_no_data_figure("GAAP Divergence Plot - No Data")

    has_posterior = "posterior_beat_prob" in df.columns
    has_momentum = "eps_revision_momentum" in df.columns
    has_group = group_col in df.columns

    n_cols = 1
    n_rows = 4
    subplot_titles = [
        "GAAP-Norm Spread Distribution",
        "Spread by Sector" if has_group else "Spread vs Posterior",
        "Spread vs Posterior Beat Probability" if has_posterior else "Top Divergent Stocks",
        "Spread vs Revision Momentum" if has_momentum else "Top Divergent Stocks",
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=tuple(subplot_titles),
        vertical_spacing=0.08,
    )

    spread = df["gaap_adj_eps_gap_pct"].dropna()

    # Panel 1: Distribution histogram
    if len(spread) > 0:
        fig.add_trace(
            go.Histogram(
                x=spread,
                name="GAAP-Norm Spread %",
                marker_color="#6C63FF",
                opacity=0.7,
                nbinsx=30,
            ),
            row=1,
            col=1,
        )
        # Threshold lines for quality concern
        fig.add_vline(
            x=-20,
            line_dash="dash",
            line_color="#FFD93D",
            annotation_text="Caution (-20%)",
            row=1,
            col=1,
        )
        fig.add_vline(
            x=-35,
            line_dash="dash",
            line_color="#E63946",
            annotation_text="Warning (-35%)",
            row=1,
            col=1,
        )

    # Panel 2: Sector box plot or spread vs posterior
    if has_group:
        groups = df[group_col].dropna().unique()
        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group]["gaap_adj_eps_gap_pct"].dropna()
            if len(group_data) > 0:
                fig.add_trace(
                    go.Box(
                        y=group_data,
                        name=str(group),
                        marker_color=COLORS[i % len(COLORS)],
                    ),
                    row=2,
                    col=1,
                )
    elif has_posterior:
        plot_df = df[["gaap_adj_eps_gap_pct", "posterior_beat_prob"]].dropna()
        if len(plot_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["gaap_adj_eps_gap_pct"],
                    y=plot_df["posterior_beat_prob"],
                    mode="markers",
                    name="Spread vs Posterior",
                    marker=dict(size=8, color="#0A7EA4", opacity=0.6),
                ),
                row=2,
                col=1,
            )

    # Panel 3: Spread vs posterior beat probability
    if has_posterior:
        plot_df = df[["gaap_adj_eps_gap_pct", "posterior_beat_prob"]].dropna()
        if len(plot_df) > 0:
            # Color by absolute spread magnitude
            abs_spread = plot_df["gaap_adj_eps_gap_pct"].abs()
            fig.add_trace(
                go.Scatter(
                    x=plot_df["gaap_adj_eps_gap_pct"],
                    y=plot_df["posterior_beat_prob"],
                    mode="markers",
                    name="Stocks",
                    marker=dict(
                        size=8,
                        color=abs_spread,
                        colorscale="YlOrRd",
                        showscale=True,
                        colorbar=dict(title="|Spread|%", x=0.45),
                        opacity=0.7,
                    ),
                    text=(df.loc[plot_df.index, "ticker"] if "ticker" in df.columns else None),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Spread: %{x:.1f}%<br>"
                        "P(Beat): %{y:.1%}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
    else:
        # Fallback: top divergent stocks bar chart
        plot_df = df.dropna(subset=["gaap_adj_eps_gap_pct"]).copy()
        if len(plot_df) > 0:
            worst = plot_df.nsmallest(20, "gaap_adj_eps_gap_pct")
            labels = (
                worst["ticker"].tolist()
                if "ticker" in worst.columns
                else [f"Stock {i}" for i in range(len(worst))]
            )
            fig.add_trace(
                go.Bar(
                    y=labels,
                    x=worst["gaap_adj_eps_gap_pct"],
                    orientation="h",
                    name="Spread %",
                    marker_color="#E63946",
                ),
                row=3,
                col=1,
            )

    # Panel 4: Spread vs revision momentum
    if has_momentum:
        plot_df = df[["gaap_adj_eps_gap_pct", "eps_revision_momentum"]].dropna()
        if len(plot_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["gaap_adj_eps_gap_pct"],
                    y=plot_df["eps_revision_momentum"],
                    mode="markers",
                    name="Spread vs Momentum",
                    marker=dict(size=8, color="#00A878", opacity=0.6),
                    text=(df.loc[plot_df.index, "ticker"] if "ticker" in df.columns else None),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Spread: %{x:.1f}%<br>"
                        "Momentum: %{y:.0f}<extra></extra>"
                    ),
                ),
                row=4,
                col=1,
            )
    else:
        # Fallback: top divergent stocks
        plot_df = df.dropna(subset=["gaap_adj_eps_gap_pct"]).copy()
        if len(plot_df) > 0:
            worst = plot_df.nsmallest(20, "gaap_adj_eps_gap_pct")
            labels = (
                worst["ticker"].tolist()
                if "ticker" in worst.columns
                else [f"Stock {i}" for i in range(len(worst))]
            )
            fig.add_trace(
                go.Bar(
                    y=labels,
                    x=worst["gaap_adj_eps_gap_pct"],
                    orientation="h",
                    name="Spread %",
                    marker_color="#FF6B6B",
                ),
                row=4,
                col=1,
            )

    fig.update_layout(
        title="GAAP-vs-Normalized EPS Divergence (Accounting Quality Guard)",
        template=PLOTLY_TEMPLATE,
        height=1400,
        width=1000,
        showlegend=False,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="GAAP-Norm Spread %", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    if has_group:
        fig.update_yaxes(title_text="Spread %", row=2, col=1)
    if has_posterior:
        fig.update_xaxes(title_text="GAAP-Norm Spread %", row=3, col=1)
        fig.update_yaxes(title_text="Posterior P(Beat)", row=3, col=1)
    if has_momentum:
        fig.update_xaxes(title_text="GAAP-Norm Spread %", row=4, col=1)
        fig.update_yaxes(title_text="Revision Momentum Score", row=4, col=1)

    return fig


def create_enhanced_beat_probability_dashboard(
    df: pd.DataFrame,
    title: str = "Enhanced Earnings Beat Probability Dashboard",
) -> go.Figure:
    """
    Comprehensive dashboard for the three-layer evidence fusion beat probability model.

    Combines historical beat counting, revision momentum, and GAAP quality guard
    into a unified multi-panel visualization. Designed to work with output from
    EarningsBeatProbabilityModel.analyze_dataframe_enhanced.

    Uses: posterior_beat_prob, historical_beat_rate, eps_revision_momentum,
          gaap_adj_eps_gap_pct, confidence_score, quarterly_beat_streak,
          classification_confidence, data_source

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced DataFrame from analyze_dataframe_enhanced.
    title : str, default 'Enhanced Earnings Beat Probability Dashboard'
        Dashboard title.

    Returns
    -------
    go.Figure
        Plotly figure with multi-panel enhanced beat probability analysis.

    Examples
    --------
    >>> model = EarningsBeatProbabilityModel()
    >>> enhanced_df = model.analyze_dataframe_enhanced(equities_df)
    >>> fig = create_enhanced_beat_probability_dashboard(enhanced_df)
    >>> fig.show()
    """
    required = ["posterior_beat_prob"]
    if not any(col in df.columns for col in required):
        return create_no_data_figure("Enhanced Beat Probability Dashboard - No Data")

    has_momentum = "eps_revision_momentum" in df.columns
    has_spread = "gaap_adj_eps_gap_pct" in df.columns
    has_history = "historical_beat_rate" in df.columns
    has_streak = "quarterly_beat_streak" in df.columns
    has_confidence = "classification_confidence" in df.columns

    fig = make_subplots(
        rows=6,
        cols=1,
        subplot_titles=(
            "Posterior Beat Probability Distribution",
            "Historical vs Posterior Beat Rate",
            "Revision Momentum vs P(Beat)",
            "GAAP Divergence vs P(Beat)",
            "Quarterly Beat Streak Distribution",
            "Classification Confidence Breakdown",
        ),
        specs=[
            [{"type": "histogram"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "bar"}],
            [{"type": "pie"}],
        ],
        vertical_spacing=0.06,
    )

    colors = {
        "primary": "#0A7EA4",
        "secondary": "#00A878",
        "accent": "#6C63FF",
        "warning": "#FFD93D",
        "danger": "#E63946",
    }

    # Panel 1: Posterior distribution
    posterior = df["posterior_beat_prob"].dropna()
    if len(posterior) > 0:
        fig.add_trace(
            go.Histogram(
                x=posterior,
                nbinsx=25,
                name="P(Beat)",
                marker_color=colors["primary"],
                opacity=0.8,
            ),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=0.5,
            line_dash="dash",
            line_color=colors["danger"],
            row=1,
            col=1,
        )

    # Panel 2: Historical vs posterior scatter
    if has_history:
        plot_df = df[["historical_beat_rate", "posterior_beat_prob"]].dropna()
        if len(plot_df) > 0:
            color_vals = (
                df.loc[plot_df.index, "confidence_score"]
                if "confidence_score" in df.columns
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_df["historical_beat_rate"],
                    y=plot_df["posterior_beat_prob"],
                    mode="markers",
                    name="Stocks",
                    marker=dict(
                        size=8,
                        color=color_vals,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Confidence", x=1.0),
                        opacity=0.7,
                    ),
                    text=(df.loc[plot_df.index, "ticker"] if "ticker" in df.columns else None),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Historical: %{x:.1%}<br>"
                        "Posterior: %{y:.1%}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
            # Diagonal reference
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    # Panel 3: Revision momentum vs posterior
    if has_momentum:
        plot_df = df[["eps_revision_momentum", "posterior_beat_prob"]].dropna()
        if len(plot_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["eps_revision_momentum"],
                    y=plot_df["posterior_beat_prob"],
                    mode="markers",
                    name="Momentum vs P(Beat)",
                    marker=dict(
                        size=8,
                        color=colors["secondary"],
                        opacity=0.6,
                    ),
                    text=(df.loc[plot_df.index, "ticker"] if "ticker" in df.columns else None),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Momentum: %{x:.0f}<br>"
                        "P(Beat): %{y:.1%}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )

    # Panel 4: GAAP spread vs posterior
    if has_spread:
        plot_df = df[["gaap_adj_eps_gap_pct", "posterior_beat_prob"]].dropna()
        if len(plot_df) > 0:
            abs_spread = plot_df["gaap_adj_eps_gap_pct"].abs()
            fig.add_trace(
                go.Scatter(
                    x=plot_df["gaap_adj_eps_gap_pct"],
                    y=plot_df["posterior_beat_prob"],
                    mode="markers",
                    name="GAAP Spread vs P(Beat)",
                    marker=dict(
                        size=8,
                        color=abs_spread,
                        colorscale="YlOrRd",
                        showscale=False,
                        opacity=0.7,
                    ),
                    text=(df.loc[plot_df.index, "ticker"] if "ticker" in df.columns else None),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Spread: %{x:.1f}%<br>"
                        "P(Beat): %{y:.1%}<extra></extra>"
                    ),
                ),
                row=4,
                col=1,
            )

    # Panel 5: Quarterly beat streak distribution
    if has_streak:
        streak_data = df["quarterly_beat_streak"].dropna()
        if len(streak_data) > 0:
            streak_counts = streak_data.value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=streak_counts.index.astype(str),
                    y=streak_counts.values,
                    name="Beat Streak",
                    marker_color=colors["accent"],
                ),
                row=5,
                col=1,
            )

    # Panel 6: Classification confidence pie
    if has_confidence:
        conf_counts = df["classification_confidence"].value_counts()
        conf_colors = {
            "High": colors["secondary"],
            "Medium": colors["warning"],
            "Low": colors["danger"],
        }
        fig.add_trace(
            go.Pie(
                labels=conf_counts.index.tolist(),
                values=conf_counts.values.tolist(),
                marker_colors=[conf_colors.get(c, colors["primary"]) for c in conf_counts.index],
                hole=0.4,
            ),
            row=6,
            col=1,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=22)),
        template=PLOTLY_TEMPLATE,
        height=2100,
        width=1000,
        showlegend=False,
        margin=dict(l=80, r=40, t=80, b=60),
    )

    fig.update_xaxes(title_text="P(Beat)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Historical Beat Rate", row=2, col=1)
    fig.update_yaxes(title_text="Posterior P(Beat)", row=2, col=1)
    if has_momentum:
        fig.update_xaxes(title_text="Revision Momentum (0-100)", row=3, col=1)
        fig.update_yaxes(title_text="Posterior P(Beat)", row=3, col=1)
    if has_spread:
        fig.update_xaxes(title_text="GAAP-Norm Spread %", row=4, col=1)
        fig.update_yaxes(title_text="Posterior P(Beat)", row=4, col=1)
    if has_streak:
        fig.update_xaxes(title_text="Consecutive Positive Quarters", row=5, col=1)
        fig.update_yaxes(title_text="Count", row=5, col=1)

    return fig
