"""
Valuation visualization module for comprehensive valuation ratio analysis.

This module provides visualization functions for analyzing valuation metrics:
- Valuation multiples comparison (radar/spider charts)
- Valuation distribution dashboards (violin plots by sector)
- Relative valuation matrix (Z-score heatmaps)
- Valuation vs growth quadrant analysis (PEG-style)
- Historical valuation percentile distributions

Feature Categories leveraged (from feature_registry.sql vw_features_valuation_ratios):
- p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio, peg_ratio, dividend_yield
- forward_pe_premium, valuation_mean_reversion, valuation_compression
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from probabilistic_ml_model.visualizations._shared import (
    PLOTLY_TEMPLATE,
    create_no_data_figure,
    resolve_column,
)
from probabilistic_ml_model.data_utils.feature_catalog import columns_for_viz

# Default valuation metrics — derived from the catalog fallback list,
# trimmed to the core multiples used by the radar/comparison charts.
_CATALOG_VALUATION_COLS = columns_for_viz("valuation")
DEFAULT_VALUATION_METRICS = [
    c
    for c in _CATALOG_VALUATION_COLS
    if c in ("p_e_ratio", "p_b_ratio", "ev_ebitda_ratio", "ev_sales_ratio", "peg_ratio")
] or [
    "p_e_ratio",
    "p_b_ratio",
    "ev_ebitda_ratio",
    "ev_sales_ratio",
    "peg_ratio",
]

# Metric display names
METRIC_LABELS = {
    "p_e_ratio": "P/E Ratio",
    "p_b_ratio": "P/B Ratio",
    "ev_ebitda_ratio": "EV/EBITDA",
    "ev_sales_ratio": "EV/Sales",
    "peg_ratio": "PEG Ratio",
    "dividend_yield": "Dividend Yield",
    "forward_pe_premium": "Forward P/E Premium",
    "valuation_mean_reversion": "Valuation Mean Reversion",
    "valuation_compression": "Valuation Compression",
    "price_to_tangible_book": "Price/Tangible Book",
    "eps_growth_yoy": "EPS Growth YoY",
    "eps_yoy_growth": "EPS Growth YoY",
    "revenue_growth_yoy": "Revenue Growth YoY",
    "revenue_yoy_growth": "Revenue Growth YoY",
}


def create_valuation_multiples_comparison(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    metrics: Optional[list[str]] = None,
) -> go.Figure:
    """
    Spider/radar chart comparing valuation multiples vs sector median.

    Uses: p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio, peg_ratio

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with valuation columns
    ticker : str, optional
        Specific ticker to highlight. If None, shows median values
    metrics : list[str], optional
        List of valuation metrics to include. Defaults to standard multiples

    Returns
    -------
    go.Figure
        Plotly radar chart figure

    Examples
    --------
    >>> fig = create_valuation_multiples_comparison(df, ticker='AAPL')
    >>> fig.show()
    """
    if metrics is None:
        metrics = DEFAULT_VALUATION_METRICS

    available_metrics = [col for col in metrics if col in df.columns]

    if not available_metrics:
        return create_no_data_figure("Valuation Multiples Comparison - No Data")

    # Calculate sector medians for comparison
    sector_medians = {}
    for metric in available_metrics:
        valid_data = df[metric].dropna()
        if len(valid_data) > 0:
            sector_medians[metric] = valid_data.median()
        else:
            sector_medians[metric] = 0

    # Get stock values if ticker specified
    stock_values = {}
    title_suffix = "(Sector Median)"

    if ticker and "ticker" in df.columns:
        stock_data = df[df["ticker"] == ticker]
        if len(stock_data) > 0:
            for metric in available_metrics:
                val = stock_data[metric].iloc[0]
                stock_values[metric] = val if pd.notna(val) else sector_medians[metric]
            title_suffix = f"({ticker} vs Sector Median)"
        else:
            stock_values = sector_medians.copy()
    else:
        stock_values = sector_medians.copy()

    # Normalize values for radar chart (0-100 scale based on percentiles)
    normalized_stock = []
    normalized_median = []
    labels = []

    for metric in available_metrics:
        valid_data = df[metric].dropna()
        if len(valid_data) > 0:
            # Calculate percentile rank (0-100)
            stock_pct = (valid_data < stock_values[metric]).mean() * 100
            median_pct = 50  # Median is always 50th percentile
            normalized_stock.append(stock_pct)
            normalized_median.append(median_pct)
        else:
            normalized_stock.append(50)
            normalized_median.append(50)
        labels.append(METRIC_LABELS.get(metric, metric))

    # Close the radar chart
    normalized_stock.append(normalized_stock[0])
    normalized_median.append(normalized_median[0])
    labels.append(labels[0])

    fig = go.Figure()

    # Add sector median trace
    fig.add_trace(
        go.Scatterpolar(
            r=normalized_median,
            theta=labels,
            fill="toself",
            name="Sector Median",
            line=dict(color="rgba(100, 100, 100, 0.8)"),
            fillcolor="rgba(100, 100, 100, 0.2)",
        )
    )

    # Add stock trace
    trace_name = ticker if ticker else "Selected"
    fig.add_trace(
        go.Scatterpolar(
            r=normalized_stock,
            theta=labels,
            fill="toself",
            name=trace_name,
            line=dict(color="rgba(0, 168, 120, 0.9)"),
            fillcolor="rgba(0, 168, 120, 0.3)",
        )
    )

    fig.update_layout(
        title=f"Valuation Multiples Comparison {title_suffix}",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%ile",
            )
        ),
        showlegend=True,
        template=PLOTLY_TEMPLATE,
    )

    return fig


def create_valuation_distribution_dashboard(
    df: pd.DataFrame,
    group_col: str = "industry",
    metrics: Optional[list[str]] = None,
) -> go.Figure:
    """
    Multi-panel violin plots for valuation metrics by sector.

    Uses: p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with valuation columns
    group_col : str, default 'industry'
        Column to group by (e.g., 'industry', 'sector')
    metrics : list[str], optional
        List of valuation metrics to display

    Returns
    -------
    go.Figure
        Plotly figure with violin plots

    Examples
    --------
    >>> fig = create_valuation_distribution_dashboard(df, group_col='sector')
    >>> fig.show()
    """
    if metrics is None:
        metrics = ["p_e_ratio", "p_b_ratio", "ev_ebitda_ratio", "ev_sales_ratio"]

    available_metrics = [col for col in metrics if col in df.columns]

    if not available_metrics or group_col not in df.columns:
        return create_no_data_figure("Valuation Distribution Dashboard - No Data")

    n_metrics = len(available_metrics)
    n_rows = n_metrics
    n_cols = 1

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[METRIC_LABELS.get(m, m) for m in available_metrics],
    )

    groups = df[group_col].dropna().unique()
    colors = [
        "#0A7EA4",
        "#00A878",
        "#6C63FF",
        "#FF6B6B",
        "#4ECDC4",
        "#FFD93D",
        "#95E1D3",
        "#F38181",
    ]

    for idx, metric in enumerate(available_metrics):
        row = idx + 1
        col = 1

        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group][metric].dropna()
            if len(group_data) > 0:
                fig.add_trace(
                    go.Violin(
                        y=group_data,
                        name=str(group),
                        legendgroup=str(group),
                        showlegend=(idx == 0),
                        line_color=colors[i % len(colors)],
                        box_visible=True,
                        meanline_visible=True,
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title="Valuation Distribution by Sector",
        template=PLOTLY_TEMPLATE,
        height=350 * n_rows,
        width=1000,
        showlegend=True,
        violinmode="group",
        margin=dict(l=80, r=40, t=60, b=60),
    )

    return fig


def create_relative_valuation_matrix(
    df: pd.DataFrame,
    group_col: str = "industry",
    metrics: Optional[list[str]] = None,
) -> go.Figure:
    """
    Heatmap showing Z-scores of valuation metrics by industry.

    Identifies cheap/expensive sectors relative to the overall market.

    Uses: p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio, peg_ratio

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with valuation columns
    group_col : str, default 'industry'
        Column to group by
    metrics : list[str], optional
        List of valuation metrics to include

    Returns
    -------
    go.Figure
        Plotly heatmap figure

    Examples
    --------
    >>> fig = create_relative_valuation_matrix(df, group_col='sector')
    >>> fig.show()
    """
    if metrics is None:
        metrics = DEFAULT_VALUATION_METRICS

    available_metrics = [col for col in metrics if col in df.columns]

    if not available_metrics or group_col not in df.columns:
        return create_no_data_figure("Relative Valuation Matrix - No Data")

    # Calculate Z-scores by group
    groups = df[group_col].dropna().unique()
    z_scores = []

    for group in groups:
        group_z = []
        for metric in available_metrics:
            group_data = df[df[group_col] == group][metric].dropna()
            overall_data = df[metric].dropna()

            if len(group_data) > 0 and len(overall_data) > 1:
                group_median = group_data.median()
                overall_mean = overall_data.mean()
                overall_std = overall_data.std()

                if overall_std > 0:
                    z = (group_median - overall_mean) / overall_std
                else:
                    z = 0
            else:
                z = 0
            group_z.append(z)
        z_scores.append(group_z)

    z_matrix = np.array(z_scores)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=[METRIC_LABELS.get(m, m) for m in available_metrics],
            y=list(groups),
            colorscale="RdYlGn_r",  # Red = expensive, Green = cheap
            zmid=0,
            text=np.round(z_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Sector: %{y}<br>Metric: %{x}<br>Z-Score: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Z-Score<br>(+expensive/-cheap)"),
        )
    )

    fig.update_layout(
        title="Relative Valuation Matrix (Z-Scores by Sector)",
        xaxis_title="Valuation Metric",
        yaxis_title="Sector",
        template=PLOTLY_TEMPLATE,
        height=max(400, len(groups) * 40),
    )

    return fig


def create_valuation_vs_growth_quadrant(
    df: pd.DataFrame,
    valuation_metric: str = "p_e_ratio",
    growth_metric: str = "eps_yoy_growth",
    color_by: str = "industry",
) -> go.Figure:
    """
    Scatter plot with quadrants for PEG-style valuation vs growth analysis.

    Quadrants:
    - Top-Left: Expensive + Slow Growth (Avoid)
    - Top-Right: Expensive + Fast Growth (Growth at Premium)
    - Bottom-Left: Cheap + Slow Growth (Value Trap Risk)
    - Bottom-Right: Cheap + Fast Growth (Opportunity)

    Uses: p_e_ratio (or custom), eps_yoy_growth (or custom)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with valuation and growth columns
    valuation_metric : str, default 'p_e_ratio'
        Valuation metric for Y-axis
    growth_metric : str, default 'eps_yoy_growth'
        Growth metric for X-axis
    color_by : str, default 'industry'
        Column to color points by

    Returns
    -------
    go.Figure
        Plotly scatter figure with quadrant annotations

    Examples
    --------
    >>> fig = create_valuation_vs_growth_quadrant(df, valuation_metric='ev_ebitda_ratio')
    >>> fig.show()
    """
    # Resolve columns via _shared alias map
    resolved_val = resolve_column(df, valuation_metric) or valuation_metric
    resolved_growth = resolve_column(df, growth_metric) or growth_metric
    valuation_metric = resolved_val
    growth_metric = resolved_growth
    if valuation_metric not in df.columns or growth_metric not in df.columns:
        return create_no_data_figure("Valuation vs Growth Quadrant - No Data")

    # Filter valid data
    plot_df = df[[valuation_metric, growth_metric]].copy()
    if color_by in df.columns:
        plot_df[color_by] = df[color_by]
    if "ticker" in df.columns:
        plot_df["ticker"] = df["ticker"]
    if "name" in df.columns:
        plot_df["name"] = df["name"]

    plot_df = plot_df.dropna(subset=[valuation_metric, growth_metric])

    if len(plot_df) == 0:
        return create_no_data_figure("Valuation vs Growth Quadrant - No Data")

    # Calculate medians for quadrant lines
    val_median = plot_df[valuation_metric].median()
    growth_median = plot_df[growth_metric].median()

    fig = go.Figure()

    # Add scatter points by group
    if color_by in plot_df.columns:
        groups = plot_df[color_by].dropna().unique()
        colors = [
            "#0A7EA4",
            "#00A878",
            "#6C63FF",
            "#FF6B6B",
            "#4ECDC4",
            "#FFD93D",
            "#95E1D3",
            "#F38181",
        ]

        for i, group in enumerate(groups):
            group_data = plot_df[plot_df[color_by] == group]

            hover_text = []
            for _, row in group_data.iterrows():
                text = f"{row.get('ticker', 'N/A')}"
                if "name" in row:
                    text += f"<br>{row['name']}"
                text += f"<br>{METRIC_LABELS.get(valuation_metric, valuation_metric)}: {row[valuation_metric]:.1f}"
                text += f"<br>{METRIC_LABELS.get(growth_metric, growth_metric)}: {row[growth_metric]:.1f}%"
                hover_text.append(text)

            fig.add_trace(
                go.Scatter(
                    x=group_data[growth_metric],
                    y=group_data[valuation_metric],
                    mode="markers",
                    name=str(group),
                    marker=dict(
                        size=10,
                        color=colors[i % len(colors)],
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
                y=plot_df[valuation_metric],
                mode="markers",
                name="Stocks",
                marker=dict(size=10, color="#0A7EA4", opacity=0.7),
            )
        )

    # Add quadrant lines
    x_range = [plot_df[growth_metric].min(), plot_df[growth_metric].max()]
    y_range = [plot_df[valuation_metric].min(), plot_df[valuation_metric].max()]

    # Vertical line at growth median
    fig.add_shape(
        type="line",
        x0=growth_median,
        x1=growth_median,
        y0=y_range[0],
        y1=y_range[1],
        line=dict(color="white", width=1, dash="dash"),
    )

    # Horizontal line at valuation median
    fig.add_shape(
        type="line",
        x0=x_range[0],
        x1=x_range[1],
        y0=val_median,
        y1=val_median,
        line=dict(color="white", width=1, dash="dash"),
    )

    # Add quadrant annotations
    annotations = [
        dict(
            x=x_range[0] + (growth_median - x_range[0]) * 0.5,
            y=y_range[1] - (y_range[1] - val_median) * 0.1,
            text="Expensive + Slow<br>(Avoid)",
            showarrow=False,
            font=dict(size=10, color="rgba(255,107,107,0.8)"),
        ),
        dict(
            x=growth_median + (x_range[1] - growth_median) * 0.5,
            y=y_range[1] - (y_range[1] - val_median) * 0.1,
            text="Expensive + Fast<br>(Growth Premium)",
            showarrow=False,
            font=dict(size=10, color="rgba(255,217,61,0.8)"),
        ),
        dict(
            x=x_range[0] + (growth_median - x_range[0]) * 0.5,
            y=y_range[0] + (val_median - y_range[0]) * 0.1,
            text="Cheap + Slow<br>(Value Trap?)",
            showarrow=False,
            font=dict(size=10, color="rgba(255,217,61,0.8)"),
        ),
        dict(
            x=growth_median + (x_range[1] - growth_median) * 0.5,
            y=y_range[0] + (val_median - y_range[0]) * 0.1,
            text="Cheap + Fast<br>(Opportunity)",
            showarrow=False,
            font=dict(size=10, color="rgba(0,168,120,0.8)"),
        ),
    ]

    fig.update_layout(
        title=f"Valuation vs Growth Quadrant Analysis",
        xaxis_title=METRIC_LABELS.get(growth_metric, growth_metric) + " (%)",
        yaxis_title=METRIC_LABELS.get(valuation_metric, valuation_metric),
        template=PLOTLY_TEMPLATE,
        annotations=annotations,
        showlegend=True,
        height=600,
    )

    return fig


def create_historical_valuation_percentile(
    df: pd.DataFrame,
    metric: str = "p_e_ratio",
    group_col: Optional[str] = "industry",
) -> go.Figure:
    """
    Distribution showing where current valuations sit vs historical ranges.

    Creates a histogram with percentile markers showing the distribution
    of the valuation metric across the dataset.

    Uses: p_e_ratio (or custom metric)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with valuation columns
    metric : str, default 'p_e_ratio'
        Valuation metric to analyze
    group_col : str, optional
        Column to break down distribution by

    Returns
    -------
    go.Figure
        Plotly histogram figure with percentile markers

    Examples
    --------
    >>> fig = create_historical_valuation_percentile(df, metric='ev_ebitda_ratio')
    >>> fig.show()
    """
    if metric not in df.columns:
        return create_no_data_figure("Historical Valuation Percentile - No Data")

    valid_data = df[metric].dropna()

    if len(valid_data) == 0:
        return create_no_data_figure("Historical Valuation Percentile - No Data")

    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90]
    pct_values = np.percentile(valid_data, percentiles)

    fig = go.Figure()

    # Add histogram by group if available
    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        colors = [
            "#0A7EA4",
            "#00A878",
            "#6C63FF",
            "#FF6B6B",
            "#4ECDC4",
            "#FFD93D",
            "#95E1D3",
            "#F38181",
        ]

        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group][metric].dropna()
            if len(group_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=group_data,
                        name=str(group),
                        opacity=0.6,
                        marker_color=colors[i % len(colors)],
                    )
                )
    else:
        fig.add_trace(
            go.Histogram(
                x=valid_data,
                name=METRIC_LABELS.get(metric, metric),
                opacity=0.7,
                marker_color="#0A7EA4",
            )
        )

    # Add percentile lines
    percentile_colors = ["#E63946", "#FFD93D", "#00A878", "#FFD93D", "#E63946"]
    for pct, val, color in zip(percentiles, pct_values, percentile_colors):
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            annotation_text=f"P{pct}: {val:.1f}",
            annotation_position="top",
            annotation_font_size=10,
        )

    fig.update_layout(
        title=f"{METRIC_LABELS.get(metric, metric)} Distribution with Percentiles",
        xaxis_title=METRIC_LABELS.get(metric, metric),
        yaxis_title="Count",
        template=PLOTLY_TEMPLATE,
        barmode="overlay",
        showlegend=True,
        height=500,
    )

    return fig
