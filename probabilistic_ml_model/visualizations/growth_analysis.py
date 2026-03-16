"""
Growth analysis visualization module for comprehensive growth metrics analysis.

This module provides visualization functions for analyzing growth metrics:
- Growth waterfall charts (Revenue → EBITDA → EPS decomposition)
- Growth consistency matrices (YoY, 3Y CAGR, 5Y CAGR by sector)
- Growth vs profitability quadrant analysis
- Growth acceleration charts
- Sustainable growth analysis (SGR = ROE × Retention Rate)

Feature Categories leveraged (from feature_registry.sql vw_features_growth):
- revenue_yoy_growth, revenue_cagr_3y, revenue_cagr_5y
- ebitda_growth_yoy, eps_yoy_growth, eps_cagr_3y
- operating_income_growth, net_income_growth_yoy
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from probabilistic_ml_model.visualizations._shared import (
    PLOTLY_TEMPLATE,
    COLORS,
    create_no_data_figure,
    resolve_column,
)

# Metric display names - aligned with actual MV column names
METRIC_LABELS = {
    "revenue_growth_yoy": "Revenue Growth YoY",
    "revenue_yoy_growth": "Revenue Growth YoY",
    "revenue_cagr_3y": "Revenue 3Y CAGR",
    "revenue_cagr_5y": "Revenue 5Y CAGR",
    "ebitda_growth_yoy": "EBITDA Growth YoY",
    "eps_yoy_growth": "EPS Growth YoY",
    "eps_growth_yoy": "EPS Growth YoY",
    "eps_cagr_3y": "EPS 3Y CAGR",
    "operating_income_growth": "Operating Income Growth",
    "net_income_growth_yoy": "Net Income Growth YoY",
    "roe": "ROE",
    "roic": "ROIC",
    "net_margin_pct": "Net Margin %",
}


def create_growth_waterfall_chart(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> go.Figure:
    """
    Waterfall chart showing Revenue → EBITDA → EPS growth decomposition.

    Uses: revenue_yoy_growth, ebitda_growth_yoy, eps_yoy_growth

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with growth columns
    ticker : str, optional
        Specific ticker to analyze. If None, shows median values

    Returns
    -------
    go.Figure
        Plotly waterfall chart figure

    Examples
    --------
    >>> fig = create_growth_waterfall_chart(df, ticker='AAPL')
    >>> fig.show()
    """
    growth_cols = [
        "revenue_growth_yoy",
        "ebitda_growth_yoy",
        "operating_income_growth",
        "net_income_growth_yoy",
        "eps_yoy_growth",
    ]
    available_cols = []
    for col in growth_cols:
        rc = resolve_column(df, col)
        if rc is not None:
            available_cols.append(rc)

    if not available_cols:
        return create_no_data_figure("Growth Waterfall Chart - No Data")

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

    # Build waterfall data
    labels = [METRIC_LABELS.get(col, col) for col in available_cols]
    vals = [values.get(col, 0) for col in available_cols]

    # Color based on positive/negative
    colors = ["#00A878" if v >= 0 else "#E63946" for v in vals]

    fig = go.Figure(
        go.Waterfall(
            name="Growth",
            orientation="v",
            measure=["relative"] * len(vals),
            x=labels,
            y=vals,
            text=[f"{v:.1f}%" if pd.notna(v) else "N/A" for v in vals],
            textposition="outside",
            connector={"line": {"color": "rgba(255,255,255,0.3)"}},
            increasing={"marker": {"color": "#00A878"}},
            decreasing={"marker": {"color": "#E63946"}},
        )
    )

    fig.update_layout(
        title=f"Growth Decomposition {title_suffix}",
        yaxis_title="Growth Rate (%)",
        template=PLOTLY_TEMPLATE,
        height=500,
        showlegend=False,
    )

    return fig


def create_growth_consistency_matrix(
    df: pd.DataFrame,
    group_col: str = "industry",
) -> go.Figure:
    """
    Heatmap showing growth metrics consistency (YoY, 3Y CAGR, 5Y CAGR) by sector.

    Uses: revenue_yoy_growth, revenue_cagr_3y, revenue_cagr_5y,
          eps_yoy_growth, eps_cagr_3y

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with growth columns
    group_col : str, default 'industry'
        Column to group by

    Returns
    -------
    go.Figure
        Plotly heatmap figure

    Examples
    --------
    >>> fig = create_growth_consistency_matrix(df, group_col='sector')
    >>> fig.show()
    """
    growth_cols = [
        "revenue_growth_yoy",
        "revenue_cagr_3y",
        "revenue_cagr_5y",
        "eps_yoy_growth",
        "eps_cagr_3y",
    ]
    available_cols = []
    for col in growth_cols:
        rc = resolve_column(df, col)
        if rc is not None:
            available_cols.append(rc)

    if not available_cols or group_col not in df.columns:
        return create_no_data_figure("Growth Consistency Matrix - No Data")

    # Calculate median growth by group
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

    fig = go.Figure(
        data=go.Heatmap(
            z=data_matrix,
            x=[METRIC_LABELS.get(col, col) for col in available_cols],
            y=list(groups),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(data_matrix, 1),
            texttemplate="%{text}%",
            textfont={"size": 10},
            hovertemplate="Sector: %{y}<br>Metric: %{x}<br>Growth: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Growth %"),
        )
    )

    fig.update_layout(
        title="Growth Consistency Matrix by Sector",
        xaxis_title="Growth Metric",
        yaxis_title="Sector",
        template=PLOTLY_TEMPLATE,
        height=max(400, len(groups) * 40),
    )

    return fig


def create_growth_vs_profitability_quadrant(
    df: pd.DataFrame,
    growth_metric: str = "revenue_growth_yoy",
    profitability_metric: str = "roe",
    size_metric: str = "net_margin_pct",
    color_by: str = "industry",
) -> go.Figure:
    """
    Scatter plot: Revenue growth vs ROE with margin bubble size.

    Quadrants:
    - Top-Right: High Growth + High Profitability (Stars)
    - Top-Left: Low Growth + High Profitability (Cash Cows)
    - Bottom-Right: High Growth + Low Profitability (Question Marks)
    - Bottom-Left: Low Growth + Low Profitability (Dogs)

    Uses: revenue_yoy_growth, roe, net_margin_pct

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with growth and profitability columns
    growth_metric : str, default 'revenue_yoy_growth'
        Growth metric for X-axis
    profitability_metric : str, default 'roe'
        Profitability metric for Y-axis
    size_metric : str, default 'net_margin_pct'
        Metric for bubble size
    color_by : str, default 'industry'
        Column to color points by

    Returns
    -------
    go.Figure
        Plotly scatter figure with quadrant analysis

    Examples
    --------
    >>> fig = create_growth_vs_profitability_quadrant(df)
    >>> fig.show()
    """
    # Resolve aliased column names
    resolved_growth = resolve_column(df, growth_metric)
    if resolved_growth is not None:
        growth_metric = resolved_growth
    if growth_metric not in df.columns or profitability_metric not in df.columns:
        return create_no_data_figure("Growth vs Profitability Quadrant - No Data")

    plot_df = df[[growth_metric, profitability_metric]].copy()
    if size_metric in df.columns:
        plot_df[size_metric] = df[size_metric]
    if color_by in df.columns:
        plot_df[color_by] = df[color_by]
    if "ticker" in df.columns:
        plot_df["ticker"] = df["ticker"]
    if "name" in df.columns:
        plot_df["name"] = df["name"]

    plot_df = plot_df.dropna(subset=[growth_metric, profitability_metric])

    if len(plot_df) == 0:
        return create_no_data_figure("Growth vs Profitability Quadrant - No Data")

    fig = go.Figure()

    # Calculate bubble sizes
    if size_metric in plot_df.columns:
        sizes = plot_df[size_metric].fillna(0).clip(lower=0)
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1) * 30 + 10
    else:
        sizes = 15

    # Add scatter points by group
    if color_by in plot_df.columns:
        groups = plot_df[color_by].dropna().unique()

        for i, group in enumerate(groups):
            group_data = plot_df[plot_df[color_by] == group]
            group_sizes = sizes[group_data.index] if isinstance(sizes, pd.Series) else sizes

            hover_text = []
            for _, row in group_data.iterrows():
                text = f"{row.get('ticker', 'N/A')}"
                if "name" in row:
                    text += f"<br>{row['name']}"
                text += f"<br>{METRIC_LABELS.get(growth_metric, growth_metric)}: {row[growth_metric]:.1f}%"
                text += f"<br>{METRIC_LABELS.get(profitability_metric, profitability_metric)}: {row[profitability_metric]:.1f}%"
                hover_text.append(text)

            fig.add_trace(
                go.Scatter(
                    x=group_data[growth_metric],
                    y=group_data[profitability_metric],
                    mode="markers",
                    name=str(group),
                    marker=dict(
                        size=group_sizes,
                        color=COLORS[i % len(COLORS)],
                        opacity=0.7,
                    ),
                    text=hover_text,
                    hoverinfo="text",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df[growth_metric],
                y=plot_df[profitability_metric],
                mode="markers",
                name="Stocks",
                marker=dict(size=sizes, color="#0A7EA4", opacity=0.7),
            )
        )

    # Add quadrant lines
    growth_median = plot_df[growth_metric].median()
    profit_median = plot_df[profitability_metric].median()

    fig.add_vline(x=growth_median, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    fig.add_hline(y=profit_median, line_dash="dash", line_color="rgba(255,255,255,0.5)")

    # Add quadrant annotations
    x_range = [plot_df[growth_metric].min(), plot_df[growth_metric].max()]
    y_range = [plot_df[profitability_metric].min(), plot_df[profitability_metric].max()]

    annotations = [
        dict(
            x=x_range[0] + (growth_median - x_range[0]) * 0.3,
            y=y_range[1] - (y_range[1] - profit_median) * 0.15,
            text="Cash Cows<br>(Low Growth, High Profit)",
            showarrow=False,
            font=dict(size=9, color="rgba(255,217,61,0.8)"),
        ),
        dict(
            x=growth_median + (x_range[1] - growth_median) * 0.7,
            y=y_range[1] - (y_range[1] - profit_median) * 0.15,
            text="Stars ⭐<br>(High Growth, High Profit)",
            showarrow=False,
            font=dict(size=9, color="rgba(0,168,120,0.8)"),
        ),
        dict(
            x=x_range[0] + (growth_median - x_range[0]) * 0.3,
            y=y_range[0] + (profit_median - y_range[0]) * 0.15,
            text="Dogs<br>(Low Growth, Low Profit)",
            showarrow=False,
            font=dict(size=9, color="rgba(230,57,70,0.8)"),
        ),
        dict(
            x=growth_median + (x_range[1] - growth_median) * 0.7,
            y=y_range[0] + (profit_median - y_range[0]) * 0.15,
            text="Question Marks<br>(High Growth, Low Profit)",
            showarrow=False,
            font=dict(size=9, color="rgba(255,217,61,0.8)"),
        ),
    ]

    fig.update_layout(
        title="Growth vs Profitability Quadrant (BCG-Style)",
        xaxis_title=METRIC_LABELS.get(growth_metric, growth_metric) + " (%)",
        yaxis_title=METRIC_LABELS.get(profitability_metric, profitability_metric) + " (%)",
        template=PLOTLY_TEMPLATE,
        annotations=annotations,
        height=600,
        showlegend=True,
    )

    return fig


def create_growth_acceleration_chart(
    df: pd.DataFrame,
    top_n: int = 30,
) -> go.Figure:
    """
    Bar chart showing growth acceleration (current YoY vs historical CAGR) ranked.

    Identifies companies with accelerating or decelerating growth.

    Uses: revenue_yoy_growth, revenue_cagr_3y

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with growth columns
    top_n : int, default 30
        Number of top stocks to display

    Returns
    -------
    go.Figure
        Plotly bar chart figure

    Examples
    --------
    >>> fig = create_growth_acceleration_chart(df, top_n=20)
    >>> fig.show()
    """
    yoy_col = resolve_column(df, "revenue_growth_yoy") or "revenue_growth_yoy"
    cagr_col = resolve_column(df, "revenue_growth_3y_cagr") or "revenue_cagr_3y"

    if yoy_col not in df.columns:
        return create_no_data_figure("Growth Acceleration Chart - No Data")

    plot_df = df.dropna(subset=[yoy_col]).copy()
    if len(plot_df) == 0:
        return create_no_data_figure("Growth Acceleration Chart - No Data")

    # Calculate acceleration (YoY - CAGR)
    if cagr_col in plot_df.columns:
        plot_df["acceleration"] = plot_df[yoy_col] - plot_df[cagr_col].fillna(0)
    else:
        plot_df["acceleration"] = plot_df[yoy_col]

    # Sort by acceleration and get top N
    plot_df = plot_df.nlargest(top_n, "acceleration")

    # Get labels
    if "ticker" in plot_df.columns:
        labels = plot_df["ticker"].tolist()
    elif "name" in plot_df.columns:
        labels = plot_df["name"].tolist()
    else:
        labels = [f"Stock {i}" for i in range(len(plot_df))]

    # Color by acceleration direction
    colors = ["#00A878" if a >= 0 else "#E63946" for a in plot_df["acceleration"]]

    fig = go.Figure()

    # Add YoY growth bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=plot_df[yoy_col],
            name="Current YoY Growth",
            marker_color="#0A7EA4",
            opacity=0.8,
        )
    )

    # Add CAGR comparison if available
    if cagr_col in plot_df.columns:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=plot_df[cagr_col],
                name="3Y CAGR",
                marker_color="rgba(100, 100, 100, 0.5)",
            )
        )

    # Add acceleration markers
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=plot_df["acceleration"],
            name="Acceleration",
            mode="markers",
            marker=dict(
                size=12,
                color=colors,
                symbol="diamond",
                line=dict(width=1, color="white"),
            ),
        )
    )

    fig.update_layout(
        title=f"Growth Acceleration Analysis (Top {top_n})",
        xaxis_title="Company",
        yaxis_title="Growth Rate (%)",
        template=PLOTLY_TEMPLATE,
        barmode="group",
        height=500,
        xaxis_tickangle=-45,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_sustainable_growth_analysis(
    df: pd.DataFrame,
    group_col: str = "industry",
) -> go.Figure:
    """
    Multi-panel analysis of sustainable growth rate (SGR = ROE × Retention Rate).

    Uses: roe, revenue_yoy_growth (as proxy for actual growth)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with ROE and growth columns
    group_col : str, default 'industry'
        Column to group by

    Returns
    -------
    go.Figure
        Plotly figure with SGR analysis

    Examples
    --------
    >>> fig = create_sustainable_growth_analysis(df)
    >>> fig.show()
    """
    if "roe" not in df.columns:
        return create_no_data_figure("Sustainable Growth Analysis - No Data")

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("ROE Distribution by Sector", "Growth vs ROE Correlation"),
        vertical_spacing=0.12,
    )

    # Panel 1: ROE distribution by sector
    if group_col in df.columns:
        groups = df[group_col].dropna().unique()

        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group]["roe"].dropna()
            if len(group_data) > 0:
                fig.add_trace(
                    go.Box(
                        y=group_data,
                        name=str(group),
                        marker_color=COLORS[i % len(COLORS)],
                        boxmean=True,
                    ),
                    row=1,
                    col=1,
                )
    else:
        roe_data = df["roe"].dropna()
        fig.add_trace(
            go.Histogram(
                x=roe_data,
                name="ROE",
                marker_color="#0A7EA4",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

    # Panel 2: Growth vs ROE scatter
    growth_col = resolve_column(df, "revenue_growth_yoy") or "revenue_growth_yoy"
    if growth_col in df.columns:
        plot_df = df[["roe", growth_col]].dropna()
        if len(plot_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["roe"],
                    y=plot_df[growth_col],
                    mode="markers",
                    name="Growth vs ROE",
                    marker=dict(
                        size=8,
                        color="#6C63FF",
                        opacity=0.6,
                    ),
                ),
                row=2,
                col=1,
            )

            # Add trend line
            if len(plot_df) > 2:
                z = np.polyfit(plot_df["roe"], plot_df[growth_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(plot_df["roe"].min(), plot_df["roe"].max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=p(x_line),
                        mode="lines",
                        name="Trend",
                        line=dict(color="rgba(255,255,255,0.5)", dash="dash"),
                    ),
                    row=2,
                    col=1,
                )

    fig.update_layout(
        title="Sustainable Growth Analysis (ROE-Based)",
        template=PLOTLY_TEMPLATE,
        height=800,
        width=1000,
        showlegend=True,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="Sector", row=1, col=1, tickfont=dict(size=11))
    fig.update_xaxes(title_text="ROE (%)", row=2, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="ROE (%)", row=1, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="Revenue Growth YoY (%)", row=2, col=1, tickfont=dict(size=11))

    return fig
