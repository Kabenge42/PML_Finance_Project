"""
Quality and risk visualization module for comprehensive quality scoring and risk assessment.

This module provides visualization functions for analyzing quality and risk metrics:
- Piotroski F-Score breakdown (9-component analysis)
- Altman Z-Score distribution with distress zones
- Quality vs Risk quadrant analysis
- Beneish M-Score manipulation probability
- Risk tier sunburst charts
- Distress early warning dashboards

Feature Categories leveraged (from feature_registry.sql vw_features_quality_risk):
- piotroski_f_score, altman_z_score, beneish_m_score
- distress_risk_score, accounting_quality_score
- quality_momentum_score, combined_distress_score
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
    "piotroski_f_score": "Piotroski F-Score",
    "altman_z_score": "Altman Z-Score",
    "beneish_m_score": "Beneish M-Score",
    "distress_risk_score": "Distress Risk Score",
    "quality_momentum_score": "Quality Momentum Score",
    "combined_distress_risk_score": "Combined Distress Risk Score",
    "accounting_quality_score": "Accounting Quality Score",
}

# Altman Z-Score zones
ALTMAN_ZONES = {
    "distress": {"max": 1.81, "color": "#E63946", "label": "Distress Zone"},
    "gray": {"min": 1.81, "max": 2.99, "color": "#FFD93D", "label": "Gray Zone"},
    "safe": {"min": 2.99, "color": "#00A878", "label": "Safe Zone"},
}

# Beneish M-Score threshold
BENEISH_THRESHOLD = -2.22  # Above this = likely manipulation


def create_piotroski_fscore_breakdown(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> go.Figure:
    """
    Stacked bar showing F-Score distribution or breakdown for specific stock.

    The Piotroski F-Score ranges from 0-9, with higher scores indicating
    stronger financial health.

    Uses: piotroski_f_score

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with F-Score column
    ticker : str, optional
        Specific ticker to highlight

    Returns
    -------
    go.Figure
        Plotly figure with F-Score analysis

    Examples
    --------
    >>> fig = create_piotroski_fscore_breakdown(df, ticker='AAPL')
    >>> fig.show()
    """
    fscore_col = resolve_column(df, "piotroski_f_score")
    if fscore_col is None:
        return create_no_data_figure(
            "Piotroski F-Score Breakdown — Column not found. "
            "Ensure calc_piotroski_f_score() output is included in your data source.",
        )

    valid_data = df[fscore_col].dropna()
    if len(valid_data) == 0:
        return create_no_data_figure("Piotroski F-Score Breakdown - No Data")

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("F-Score Distribution", "Score Interpretation"),
        vertical_spacing=0.12,
    )

    # Panel 1: Distribution histogram
    score_counts = valid_data.value_counts().sort_index()
    scores = list(range(10))
    counts = [score_counts.get(s, 0) for s in scores]

    # Color by quality zone
    colors = []
    for s in scores:
        if s <= 3:
            colors.append("#E63946")  # Poor
        elif s <= 6:
            colors.append("#FFD93D")  # Average
        else:
            colors.append("#00A878")  # Strong

    fig.add_trace(
        go.Bar(
            x=scores,
            y=counts,
            name="Distribution",
            marker_color=colors,
            text=counts,
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # Highlight specific ticker if provided
    if ticker and "ticker" in df.columns:
        stock_data = df[df["ticker"] == ticker]
        if len(stock_data) > 0:
            stock_score = stock_data[fscore_col].iloc[0]
            if pd.notna(stock_score):
                fig.add_vline(
                    x=stock_score,
                    line_dash="dash",
                    line_color="white",
                    annotation_text=f"{ticker}: {int(stock_score)}",
                    row=1,
                    col=1,
                )

    # Panel 2: Score interpretation guide
    interpretation_data = {
        "Zone": ["Strong (7-9)", "Average (4-6)", "Weak (0-3)"],
        "Count": [
            len(valid_data[valid_data >= 7]),
            len(valid_data[(valid_data >= 4) & (valid_data < 7)]),
            len(valid_data[valid_data < 4]),
        ],
    }

    fig.add_trace(
        go.Bar(
            x=interpretation_data["Zone"],
            y=interpretation_data["Count"],
            name="By Zone",
            marker_color=["#00A878", "#FFD93D", "#E63946"],
            text=interpretation_data["Count"],
            textposition="auto",
        ),
        row=2,
        col=1,
    )

    title = "Piotroski F-Score Analysis"
    if ticker:
        title += f" (Highlighting {ticker})"

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=800,
        width=1000,
        showlegend=False,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="F-Score (0-9)", row=1, col=1)
    fig.update_xaxes(title_text="Quality Zone", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig


def create_altman_zscore_distribution(
    df: pd.DataFrame,
    group_col: str = "industry",
) -> go.Figure:
    """
    Distribution with distress zones (safe/gray/distress) highlighted.

    Altman Z-Score zones:
    - Z < 1.81: Distress Zone (high bankruptcy risk)
    - 1.81 <= Z < 2.99: Gray Zone (uncertain)
    - Z >= 2.99: Safe Zone (low bankruptcy risk)

    Uses: altman_z_score

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with Z-Score column
    group_col : str, default 'industry'
        Column to group by for breakdown

    Returns
    -------
    go.Figure
        Plotly figure with Z-Score distribution

    Examples
    --------
    >>> fig = create_altman_zscore_distribution(df, group_col='sector')
    >>> fig.show()
    """
    if "altman_z_score" not in df.columns:
        return create_no_data_figure("Altman Z-Score Distribution - No Data")

    valid_data = df["altman_z_score"].dropna()
    if len(valid_data) == 0:
        return create_no_data_figure("Altman Z-Score Distribution - No Data")

    fig = go.Figure()

    # Add histogram by group if available
    if group_col in df.columns:
        groups = df[group_col].dropna().unique()

        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group]["altman_z_score"].dropna()
            if len(group_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=group_data,
                        name=str(group),
                        opacity=0.6,
                        marker_color=COLORS[i % len(COLORS)],
                    ),
                )
    else:
        fig.add_trace(
            go.Histogram(
                x=valid_data,
                name="Z-Score",
                opacity=0.7,
                marker_color="#0A7EA4",
            ),
        )

    # Add zone indicators
    y_max = len(valid_data) * 0.3  # Approximate height for annotations

    # Distress zone
    fig.add_vrect(
        x0=valid_data.min() - 0.5,
        x1=1.81,
        fillcolor="rgba(230, 57, 70, 0.2)",
        line_width=0,
        annotation_text="Distress",
        annotation_position="top left",
    )

    # Gray zone
    fig.add_vrect(
        x0=1.81,
        x1=2.99,
        fillcolor="rgba(255, 217, 61, 0.2)",
        line_width=0,
        annotation_text="Gray Zone",
        annotation_position="top",
    )

    # Safe zone
    fig.add_vrect(
        x0=2.99,
        x1=valid_data.max() + 0.5,
        fillcolor="rgba(0, 168, 120, 0.2)",
        line_width=0,
        annotation_text="Safe",
        annotation_position="top right",
    )

    # Add threshold lines
    fig.add_vline(x=1.81, line_dash="dash", line_color="#E63946", line_width=2)
    fig.add_vline(x=2.99, line_dash="dash", line_color="#00A878", line_width=2)

    fig.update_layout(
        title="Altman Z-Score Distribution with Risk Zones",
        xaxis_title="Altman Z-Score",
        yaxis_title="Count",
        template=PLOTLY_TEMPLATE,
        barmode="overlay",
        height=500,
        showlegend=True,
    )

    return fig


def create_quality_risk_quadrant(
    df: pd.DataFrame,
    quality_metric: str = "piotroski_f_score",
    risk_metric: str = "altman_z_score",
    color_by: str = "industry",
) -> go.Figure:
    """
    Scatter plot: Piotroski F-Score vs Altman Z-Score with distress probability overlay.

    Quadrants:
    - Top-Right: High Quality + Low Risk (Best)
    - Top-Left: Low Quality + Low Risk (Improving?)
    - Bottom-Right: High Quality + High Risk (Deteriorating?)
    - Bottom-Left: Low Quality + High Risk (Avoid)

    Uses: piotroski_f_score, altman_z_score

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with quality and risk columns
    quality_metric : str, default 'piotroski_f_score'
        Quality metric for X-axis
    risk_metric : str, default 'altman_z_score'
        Risk metric for Y-axis
    color_by : str, default 'industry'
        Column to color points by

    Returns
    -------
    go.Figure
        Plotly scatter figure with quadrant analysis

    Examples
    --------
    >>> fig = create_quality_risk_quadrant(df)
    >>> fig.show()
    """
    if quality_metric not in df.columns or risk_metric not in df.columns:
        return create_no_data_figure("Quality vs Risk Quadrant - No Data")

    plot_df = df[[quality_metric, risk_metric]].copy()
    if color_by in df.columns:
        plot_df[color_by] = df[color_by]
    if "ticker" in df.columns:
        plot_df["ticker"] = df["ticker"]
    if "name" in df.columns:
        plot_df["name"] = df["name"]

    plot_df = plot_df.dropna(subset=[quality_metric, risk_metric])

    if len(plot_df) == 0:
        return create_no_data_figure("Quality vs Risk Quadrant - No Data")

    fig = go.Figure()

    # Add scatter points by group
    if color_by in plot_df.columns:
        groups = plot_df[color_by].dropna().unique()

        for i, group in enumerate(groups):
            group_data = plot_df[plot_df[color_by] == group]

            hover_text = []
            for _, row in group_data.iterrows():
                text = f"{row.get('ticker', 'N/A')}"
                if "name" in row:
                    text += f"<br>{row['name']}"
                text += f"<br>{METRIC_LABELS.get(quality_metric, quality_metric)}: {row[quality_metric]:.1f}"
                text += f"<br>{METRIC_LABELS.get(risk_metric, risk_metric)}: {row[risk_metric]:.2f}"
                hover_text.append(text)

            fig.add_trace(
                go.Scatter(
                    x=group_data[quality_metric],
                    y=group_data[risk_metric],
                    mode="markers",
                    name=str(group),
                    marker=dict(
                        size=10,
                        color=COLORS[i % len(COLORS)],
                        opacity=0.7,
                    ),
                    text=hover_text,
                    hoverinfo="text",
                ),
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df[quality_metric],
                y=plot_df[risk_metric],
                mode="markers",
                name="Stocks",
                marker=dict(size=10, color="#0A7EA4", opacity=0.7),
            ),
        )

    # Add quadrant lines
    quality_median = plot_df[quality_metric].median()
    risk_threshold = 2.99 if risk_metric == "altman_z_score" else plot_df[risk_metric].median()

    x_range = [plot_df[quality_metric].min(), plot_df[quality_metric].max()]
    y_range = [plot_df[risk_metric].min(), plot_df[risk_metric].max()]

    # Vertical line at quality median
    fig.add_vline(
        x=quality_median,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
    )

    # Horizontal line at risk threshold
    fig.add_hline(
        y=risk_threshold,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
    )

    # Add quadrant annotations
    annotations = [
        dict(
            x=x_range[0] + (quality_median - x_range[0]) * 0.3,
            y=y_range[1] - (y_range[1] - risk_threshold) * 0.2,
            text="Low Quality<br>Low Risk",
            showarrow=False,
            font=dict(size=10, color="rgba(255,217,61,0.8)"),
        ),
        dict(
            x=quality_median + (x_range[1] - quality_median) * 0.7,
            y=y_range[1] - (y_range[1] - risk_threshold) * 0.2,
            text="High Quality<br>Low Risk ✓",
            showarrow=False,
            font=dict(size=10, color="rgba(0,168,120,0.8)"),
        ),
        dict(
            x=x_range[0] + (quality_median - x_range[0]) * 0.3,
            y=y_range[0] + (risk_threshold - y_range[0]) * 0.2,
            text="Low Quality<br>High Risk ✗",
            showarrow=False,
            font=dict(size=10, color="rgba(230,57,70,0.8)"),
        ),
        dict(
            x=quality_median + (x_range[1] - quality_median) * 0.7,
            y=y_range[0] + (risk_threshold - y_range[0]) * 0.2,
            text="High Quality<br>High Risk",
            showarrow=False,
            font=dict(size=10, color="rgba(255,217,61,0.8)"),
        ),
    ]

    fig.update_layout(
        title="Quality vs Risk Quadrant Analysis",
        xaxis_title=METRIC_LABELS.get(quality_metric, quality_metric),
        yaxis_title=METRIC_LABELS.get(risk_metric, risk_metric),
        template=PLOTLY_TEMPLATE,
        annotations=annotations,
        height=600,
        showlegend=True,
    )

    return fig


def create_beneish_mscore_analysis(
    df: pd.DataFrame,
    group_col: str = "industry",
) -> go.Figure:
    """
    Accounting quality analysis visualization.

    Refactored: Uses accounting_quality_score and accruals_quality from
    calc_accounting_quality_features() instead of beneish_m_score which
    is not available in mv_all_stock_features.

    Falls back through: beneish_m_score → accounting_quality_score → accruals_quality

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with accounting quality columns
    group_col : str, default 'industry'
        Column to group by for sector analysis

    Returns
    -------
    go.Figure
        Plotly figure with accounting quality analysis

    Examples
    --------
    >>> fig = create_beneish_mscore_analysis(df)
    >>> fig.show()
    """
    # Try primary column via resolve_column, fall back to accounting_quality_score
    score_col = resolve_column(df, "beneish_m_score")
    if score_col is None:
        score_col = resolve_column(df, "accounting_quality_score")

    if score_col is None:
        return create_no_data_figure("Accounting Quality Analysis - No Data")

    valid_data = df[score_col].dropna()
    if len(valid_data) == 0:
        return create_no_data_figure("Accounting Quality Analysis - No Data")

    # Adapt thresholds and labels based on which column we resolved to
    if score_col == "beneish_m_score":
        threshold = BENEISH_THRESHOLD
        high_label = "Likely Manipulation"
        low_label = "Unlikely Manipulation"
        is_higher_worse = True  # M-Score > threshold = bad
        title = "Beneish M-Score Analysis (Earnings Manipulation Detection)"
        panel1_title = "M-Score Distribution"
        x_label = "M-Score"
    elif score_col == "accounting_quality_score":
        threshold = 50.0
        high_label = "Good Quality (>50)"
        low_label = "Poor Quality (≤50)"
        is_higher_worse = False  # Higher = better for quality score
        title = "Accounting Quality Score Analysis"
        panel1_title = "Quality Score Distribution"
        x_label = "Accounting Quality Score"
    else:  # accruals_quality
        threshold = 0.0
        high_label = "Good Quality (>0)"
        low_label = "Poor Quality (≤0)"
        is_higher_worse = False
        title = "Accruals Quality Analysis"
        panel1_title = "Accruals Quality Distribution"
        x_label = "Accruals Quality"

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(panel1_title, "Quality Risk Summary"),
        vertical_spacing=0.12,
    )

    # Panel 1: Distribution
    fig.add_trace(
        go.Histogram(
            x=valid_data,
            name=score_col.replace("_", " ").title(),
            marker_color="#6C63FF",
            opacity=0.7,
            nbinsx=30,
        ),
        row=1,
        col=1,
    )

    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#E63946",
        annotation_text=f"Threshold: {threshold}",
        row=1,
        col=1,
    )

    # Add zones — color depends on whether higher is worse or better
    if is_higher_worse:
        bad_x0, bad_x1 = threshold, valid_data.max() + 0.5
        good_x0, good_x1 = valid_data.min() - 0.5, threshold
    else:
        good_x0, good_x1 = threshold, valid_data.max() + 0.5
        bad_x0, bad_x1 = valid_data.min() - 0.5, threshold

    fig.add_vrect(
        x0=bad_x0,
        x1=bad_x1,
        fillcolor="rgba(230, 57, 70, 0.2)",
        line_width=0,
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=good_x0,
        x1=good_x1,
        fillcolor="rgba(0, 168, 120, 0.2)",
        line_width=0,
        row=1,
        col=1,
    )

    # Panel 2: Summary
    if is_higher_worse:
        count_bad = len(valid_data[valid_data > threshold])
        count_good = len(valid_data[valid_data <= threshold])
    else:
        count_good = len(valid_data[valid_data > threshold])
        count_bad = len(valid_data[valid_data <= threshold])

    fig.add_trace(
        go.Bar(
            x=[high_label, low_label],
            y=[
                count_good if not is_higher_worse else count_bad,
                count_bad if not is_higher_worse else count_good,
            ],
            marker_color=["#E63946", "#00A878"] if is_higher_worse else ["#00A878", "#E63946"],
            text=[
                count_good if not is_higher_worse else count_bad,
                count_bad if not is_higher_worse else count_good,
            ],
            textposition="auto",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=800,
        width=1000,
        showlegend=False,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text=x_label, row=1, col=1)
    fig.update_xaxes(title_text="Category", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig


def create_risk_tier_sunburst(
    df: pd.DataFrame,
    sector_col: str = "industry",
) -> go.Figure:
    """
    Sunburst chart showing Sector → Industry → Risk Tier hierarchy.

    Uses: distress_risk_score or altman_z_score for risk tier classification

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with sector and risk columns
    sector_col : str, default 'industry'
        Column for sector grouping

    Returns
    -------
    go.Figure
        Plotly sunburst figure

    Examples
    --------
    >>> fig = create_risk_tier_sunburst(df)
    >>> fig.show()
    """
    # Determine risk tier column
    if "distress_risk_score" in df.columns:
        risk_col = "distress_risk_score"
    elif "altman_z_score" in df.columns:
        risk_col = "altman_z_score"
    else:
        return create_no_data_figure("Risk Tier Sunburst - No Data")

    if sector_col not in df.columns:
        return create_no_data_figure("Risk Tier Sunburst - No Data")

    # Create risk tiers
    plot_df = df[[sector_col, risk_col]].dropna().copy()
    if len(plot_df) == 0:
        return create_no_data_figure("Risk Tier Sunburst - No Data")

    # Classify into risk tiers
    if risk_col == "altman_z_score":
        plot_df["risk_tier"] = pd.cut(
            plot_df[risk_col],
            bins=[-np.inf, 1.81, 2.99, np.inf],
            labels=["High Risk", "Medium Risk", "Low Risk"],
        )
    else:
        # For distress_risk_score (higher = more risk)
        plot_df["risk_tier"] = pd.cut(
            plot_df[risk_col],
            bins=[-np.inf, 30, 60, np.inf],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )

    # Build sunburst data
    labels = ["All"]
    parents = [""]
    values = [len(plot_df)]
    colors = ["#0A7EA4"]

    # Add sectors
    sectors = plot_df[sector_col].unique()
    for sector in sectors:
        sector_data = plot_df[plot_df[sector_col] == sector]
        labels.append(str(sector))
        parents.append("All")
        values.append(len(sector_data))
        colors.append("#4ECDC4")

        # Add risk tiers within sector
        for tier in ["Low Risk", "Medium Risk", "High Risk"]:
            tier_data = sector_data[sector_data["risk_tier"] == tier]
            if len(tier_data) > 0:
                labels.append(f"{sector} - {tier}")
                parents.append(str(sector))
                values.append(len(tier_data))
                if tier == "Low Risk":
                    colors.append("#00A878")
                elif tier == "Medium Risk":
                    colors.append("#FFD93D")
                else:
                    colors.append("#E63946")

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            branchvalues="total",
        ),
    )

    fig.update_layout(
        title="Risk Tier Distribution by Sector",
        template=PLOTLY_TEMPLATE,
        height=600,
    )

    return fig


def create_distress_early_warning_dashboard(
    df: pd.DataFrame,
) -> go.Figure:
    """
    Multi-panel dashboard showing companies approaching distress thresholds.

    Uses: altman_z_score, piotroski_f_score, distress_risk_score

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with distress indicators

    Returns
    -------
    go.Figure
        Plotly figure with early warning indicators

    Examples
    --------
    >>> fig = create_distress_early_warning_dashboard(df)
    >>> fig.show()
    """
    has_altman = "altman_z_score" in df.columns
    _fscore_col = resolve_column(df, "piotroski_f_score")
    has_piotroski = _fscore_col is not None
    has_distress = "distress_risk_score" in df.columns

    if not has_altman and not has_piotroski and not has_distress:
        return create_no_data_figure("Distress Early Warning Dashboard - No Data")

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Z-Score Warning Zone",
            "F-Score Warning Zone",
            "Distress Risk Distribution",
            "Combined Risk Summary",
        ),
        vertical_spacing=0.08,
    )

    # Panel 1: Z-Score near distress threshold
    if has_altman:
        z_data = df["altman_z_score"].dropna()
        warning_zone = z_data[(z_data >= 1.5) & (z_data <= 2.5)]
        distress_zone = z_data[z_data < 1.5]
        safe_zone = z_data[z_data > 2.5]

        fig.add_trace(
            go.Histogram(x=distress_zone, name="Distress", marker_color="#E63946", opacity=0.7),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=warning_zone, name="Warning", marker_color="#FFD93D", opacity=0.7),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=safe_zone, name="Safe", marker_color="#00A878", opacity=0.7),
            row=1,
            col=1,
        )

    # Panel 2: F-Score warning (low scores)
    if has_piotroski:
        f_data = df[_fscore_col].dropna()
        weak = f_data[f_data <= 3]
        average = f_data[(f_data > 3) & (f_data <= 6)]
        strong = f_data[f_data > 6]

        fig.add_trace(
            go.Histogram(x=weak, name="Weak (0-3)", marker_color="#E63946", opacity=0.7),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=average, name="Average (4-6)", marker_color="#FFD93D", opacity=0.7),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=strong, name="Strong (7-9)", marker_color="#00A878", opacity=0.7),
            row=2,
            col=1,
        )

    # Panel 3: Distress risk score distribution
    if has_distress:
        distress_data = df["distress_risk_score"].dropna()
        fig.add_trace(
            go.Histogram(
                x=distress_data,
                name="Distress Risk",
                marker_color="#6C63FF",
                opacity=0.7,
                nbinsx=20,
            ),
            row=3,
            col=1,
        )
        # Add warning threshold
        fig.add_vline(x=70, line_dash="dash", line_color="#E63946", row=3, col=1)

    # Panel 4: Combined risk summary
    summary_data = {"Category": [], "Count": []}

    if has_altman:
        z_distress = len(df[df["altman_z_score"] < 1.81])
        summary_data["Category"].append("Z-Score Distress")
        summary_data["Count"].append(z_distress)

    if has_piotroski:
        f_weak = len(df[df[_fscore_col] <= 3])
        summary_data["Category"].append("F-Score Weak")
        summary_data["Count"].append(f_weak)

    if has_distress:
        high_distress = len(df[df["distress_risk_score"] > 70])
        summary_data["Category"].append("High Distress Risk")
        summary_data["Count"].append(high_distress)

    if summary_data["Category"]:
        fig.add_trace(
            go.Bar(
                x=summary_data["Category"],
                y=summary_data["Count"],
                marker_color=["#E63946"] * len(summary_data["Category"]),
                text=summary_data["Count"],
                textposition="auto",
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        title="Distress Early Warning Dashboard",
        template=PLOTLY_TEMPLATE,
        height=1400,
        width=1000,
        showlegend=False,
        barmode="overlay",
        margin=dict(l=80, r=40, t=60, b=60),
    )

    return fig


def create_accounting_anomaly_dashboard(
    df: pd.DataFrame,
    group_col: str = "industry",
) -> go.Figure:
    """
    Multi-panel dashboard for accounting anomaly detection results.

    Panels:
    1. Anomaly score distribution by tier (Clean/Watch/Flag/Alert)
    2. Per-feature flag frequency (horizontal bar)
    3. Sector-relative anomaly heatmap (top sectors)
    4. Mahalanobis distance vs anomaly score scatter

    Uses: accounting_anomaly_score, accounting_anomaly_tier,
          *_anomaly_flag, mahalanobis_distance, sector_relative_anomaly

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with anomaly detection columns
    group_col : str, default 'industry'
        Column for sector grouping

    Returns
    -------
    go.Figure
        Plotly figure with 4-panel anomaly dashboard
    """
    if "accounting_anomaly_score" not in df.columns:
        return create_no_data_figure(
            "Accounting Anomaly Dashboard — Run detect_accounting_anomalies() first",
        )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Anomaly Score Distribution by Tier",
            "Per-Feature Flag Frequency",
            "Sector-Relative Anomaly Scores",
            "Mahalanobis Distance vs Anomaly Score",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # Panel 1: Score distribution by tier
    tier_colors = {
        "Clean": "#00A878",
        "Watch": "#FFD93D",
        "Flag": "#FF8C42",
        "Alert": "#E63946",
    }
    if "accounting_anomaly_tier" in df.columns:
        for tier in ["Clean", "Watch", "Flag", "Alert"]:
            tier_data = df[df["accounting_anomaly_tier"] == tier]["accounting_anomaly_score"]
            if len(tier_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=tier_data,
                        name=tier,
                        marker_color=tier_colors.get(tier, "#0A7EA4"),
                        opacity=0.7,
                    ),
                    row=1,
                    col=1,
                )
    else:
        fig.add_trace(
            go.Histogram(
                x=df["accounting_anomaly_score"].dropna(),
                name="Score",
                marker_color="#0A7EA4",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

    # Panel 2: Per-feature flag frequency
    flag_cols = sorted([c for c in df.columns if c.endswith("_anomaly_flag")])
    if flag_cols:
        flag_counts = {
            c.replace("_anomaly_flag", ""): int(df[c].sum()) for c in flag_cols if df[c].sum() > 0
        }
        if flag_counts:
            sorted_flags = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)
            fig.add_trace(
                go.Bar(
                    y=[f[0] for f in sorted_flags],
                    x=[f[1] for f in sorted_flags],
                    orientation="h",
                    marker_color="#6C63FF",
                    name="Flags",
                ),
                row=1,
                col=2,
            )

    # Panel 3: Sector-relative anomaly scores (top sectors)
    if group_col in df.columns and "accounting_anomaly_score" in df.columns:
        sector_mean = (
            df.groupby(group_col)["accounting_anomaly_score"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        fig.add_trace(
            go.Bar(
                y=sector_mean.index,
                x=sector_mean.values,
                orientation="h",
                marker_color="#FF8C42",
                name="Sector Mean",
            ),
            row=2,
            col=1,
        )

    # Panel 4: Mahalanobis distance vs anomaly score
    if "mahalanobis_distance" in df.columns:
        plot_df = df[["accounting_anomaly_score", "mahalanobis_distance"]].dropna()
        if len(plot_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["accounting_anomaly_score"],
                    y=plot_df["mahalanobis_distance"],
                    mode="markers",
                    marker=dict(size=5, color="#0A7EA4", opacity=0.5),
                    name="Stocks",
                ),
                row=2,
                col=2,
            )

    fig.update_layout(
        title="Accounting Anomaly Detection Dashboard",
        template=PLOTLY_TEMPLATE,
        height=1000,
        width=1200,
        showlegend=True,
        barmode="overlay",
        margin=dict(l=120, r=40, t=80, b=60),
    )

    fig.update_xaxes(title_text="Anomaly Score (0-100)", row=1, col=1)
    fig.update_xaxes(title_text="Stocks Flagged", row=1, col=2)
    fig.update_xaxes(title_text="Mean Anomaly Score", row=2, col=1)
    fig.update_xaxes(title_text="Anomaly Score", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Mahalanobis Distance", row=2, col=2)

    return fig


def create_anomaly_severity_dashboard(
    df: pd.DataFrame,
    group_col: str = "industry",
    top_n: int = 25,
) -> go.Figure:
    """
    Multi-panel dashboard for Bayesian-informed accounting anomaly severity.

    Visualises the enriched columns produced by
    :class:`~analytics.probability_analytics.AccountingAnomalyProbabilityModel`:

    Panels:
    1. Severity Score vs Conditional P(Anomaly) scatter — colour by tier
    2. Top-N stocks by anomaly severity (horizontal bar)
    3. Sector anomaly percentile box-plot
    4. Multi-flag alert distribution (pie) + risk-rank histogram overlay
    5. Conditional probability density by anomaly tier
    6. Severity vs Risk Rank scatter with multi-flag highlights

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by ``AccountingAnomalyProbabilityModel.analyze_dataframe``.
        Expected columns: ``anomaly_severity_score``, ``anomaly_conditional_probability``,
        ``anomaly_risk_rank``, ``sector_anomaly_percentile``, ``multi_flag_alert``,
        ``accounting_anomaly_tier``.
    group_col : str, default 'industry'
        Column used for sector/industry grouping.
    top_n : int, default 25
        Number of top-severity stocks to display in the bar panel.

    Returns
    -------
    go.Figure
        Plotly figure with a 6-panel severity analytics dashboard.
    """
    if "anomaly_severity_score" not in df.columns:
        return create_no_data_figure(
            "Anomaly Severity Dashboard — run AccountingAnomalyProbabilityModel first",
        )

    tier_colors = {
        "Clean": "#00A878",
        "Watch": "#FFD93D",
        "Flag": "#FF8C42",
        "Alert": "#E63946",
    }

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Severity Score vs Conditional P(Anomaly)",
            f"Top {top_n} Stocks by Anomaly Severity",
            "Sector Anomaly Percentile Distribution",
            "Multi-Flag Alert Breakdown",
            "Conditional P(Anomaly) Density by Tier",
            "Severity vs Risk Rank",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.12,
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "pie"}],
            [{"type": "histogram"}, {"type": "scatter"}],
        ],
    )

    has_tier = "accounting_anomaly_tier" in df.columns
    has_cond_prob = "anomaly_conditional_probability" in df.columns
    has_risk_rank = "anomaly_risk_rank" in df.columns
    has_sector_pct = "sector_anomaly_percentile" in df.columns
    has_multi_flag = "multi_flag_alert" in df.columns
    ticker_col = "ticker" if "ticker" in df.columns else None

    # ── Panel 1: Severity vs Conditional Probability scatter ──
    if has_cond_prob:
        if has_tier:
            for tier in ["Clean", "Watch", "Flag", "Alert"]:
                mask = df["accounting_anomaly_tier"] == tier
                sub = df.loc[mask]
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=sub["anomaly_severity_score"],
                        y=sub["anomaly_conditional_probability"],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=tier_colors.get(tier, "#0A7EA4"),
                            opacity=0.6,
                        ),
                        name=tier,
                        hovertemplate=(
                            (
                                "<b>%{customdata[0]}</b><br>"
                                "Severity: %{x:.1f}<br>"
                                "P(Anomaly): %{y:.3f}<extra></extra>"
                            )
                            if ticker_col
                            else None
                        ),
                        customdata=sub[[ticker_col]].values if ticker_col else None,
                    ),
                    row=1,
                    col=1,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df["anomaly_severity_score"],
                    y=df["anomaly_conditional_probability"],
                    mode="markers",
                    marker=dict(size=5, color="#0A7EA4", opacity=0.5),
                    name="Stocks",
                ),
                row=1,
                col=1,
            )
    fig.update_xaxes(title_text="Severity Score", row=1, col=1)
    fig.update_yaxes(title_text="P(Anomaly)", row=1, col=1)

    # ── Panel 2: Top-N severity bar chart ──
    top = df.nlargest(top_n, "anomaly_severity_score")
    labels = top[ticker_col] if ticker_col else top.index.astype(str)
    bar_colors = (
        [tier_colors.get(t, "#0A7EA4") for t in top["accounting_anomaly_tier"]]
        if has_tier
        else "#E63946"
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=top["anomaly_severity_score"],
            orientation="h",
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Severity: %{x:.1f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Severity Score", row=1, col=2)

    # ── Panel 3: Sector anomaly percentile box-plot ──
    if has_sector_pct and group_col in df.columns:
        sector_medians = (
            df.groupby(group_col)["sector_anomaly_percentile"].median().sort_values(ascending=False)
        )
        top_sectors = sector_medians.head(12).index.tolist()
        plot_df = df[df[group_col].isin(top_sectors)]
        for sector in top_sectors:
            sec_data = plot_df.loc[
                plot_df[group_col] == sector,
                "sector_anomaly_percentile",
            ]
            fig.add_trace(
                go.Box(
                    y=sec_data,
                    name=sector[:20],
                    marker_color="#6C63FF",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
    fig.update_yaxes(title_text="Sector Anomaly Percentile", row=2, col=1)

    # ── Panel 4: Multi-flag alert pie ──
    if has_multi_flag:
        alert_counts = df["multi_flag_alert"].value_counts()
        pie_labels = [
            f"Multi-Flag ({int(alert_counts.get(True, 0))})",
            f"Normal ({int(alert_counts.get(False, 0))})",
        ]
        pie_values = [
            int(alert_counts.get(True, 0)),
            int(alert_counts.get(False, 0)),
        ]
        fig.add_trace(
            go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker_colors=["#E63946", "#00A878"],
                textinfo="label+percent",
                hole=0.35,
            ),
            row=2,
            col=2,
        )

    # ── Panel 5: Conditional probability density by tier ──
    if has_cond_prob and has_tier:
        for tier in ["Clean", "Watch", "Flag", "Alert"]:
            tier_data = df.loc[
                df["accounting_anomaly_tier"] == tier,
                "anomaly_conditional_probability",
            ].dropna()
            if len(tier_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=tier_data,
                        name=f"{tier} P(A)",
                        marker_color=tier_colors.get(tier, "#0A7EA4"),
                        opacity=0.6,
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )
    elif has_cond_prob:
        fig.add_trace(
            go.Histogram(
                x=df["anomaly_conditional_probability"].dropna(),
                marker_color="#0A7EA4",
                opacity=0.7,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    fig.update_xaxes(title_text="Conditional P(Anomaly)", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)

    # ── Panel 6: Severity vs Risk Rank with multi-flag highlight ──
    if has_risk_rank:
        if has_multi_flag:
            normal = df[~df["multi_flag_alert"]]
            flagged = df[df["multi_flag_alert"]]
            fig.add_trace(
                go.Scatter(
                    x=normal["anomaly_severity_score"],
                    y=normal["anomaly_risk_rank"],
                    mode="markers",
                    marker=dict(size=4, color="#0A7EA4", opacity=0.3),
                    name="Normal",
                    showlegend=False,
                ),
                row=3,
                col=2,
            )
            if not flagged.empty:
                fig.add_trace(
                    go.Scatter(
                        x=flagged["anomaly_severity_score"],
                        y=flagged["anomaly_risk_rank"],
                        mode="markers",
                        marker=dict(
                            size=7,
                            color="#E63946",
                            symbol="diamond",
                            opacity=0.8,
                        ),
                        name="Multi-Flag",
                        showlegend=False,
                    ),
                    row=3,
                    col=2,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df["anomaly_severity_score"],
                    y=df["anomaly_risk_rank"],
                    mode="markers",
                    marker=dict(size=4, color="#0A7EA4", opacity=0.4),
                    showlegend=False,
                ),
                row=3,
                col=2,
            )
    fig.update_xaxes(title_text="Severity Score", row=3, col=2)
    fig.update_yaxes(title_text="Risk Rank (percentile)", row=3, col=2)

    fig.update_layout(
        title="Accounting Anomaly Severity — Bayesian Analytics Dashboard",
        template=PLOTLY_TEMPLATE,
        height=1400,
        width=1200,
        showlegend=True,
        barmode="overlay",
        margin=dict(l=120, r=40, t=80, b=60),
    )

    return fig
