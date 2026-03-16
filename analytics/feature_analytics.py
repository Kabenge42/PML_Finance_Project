"""
Feature Analytics Module.

This module provides interactive visualizations, probabilistic models,
and statistical analytics for financial feature analysis based on the
feature_analytics.ipynb notebook enhancements.

Functions:
    - load_feature_data_from_db: Load feature data from PostgreSQL database
    - backfill_feature_columns: Normalize and backfill missing columns
    - create_interactive_momentum_dashboard: Interactive Plotly momentum analysis
    - create_interactive_valuation_heatmap: Interactive valuation heatmap by industry
    - create_leverage_liquidity_quadrant: Leverage vs liquidity quadrant analysis
    - monte_carlo_price_target_simulation: Monte Carlo fair value simulation
    - bayesian_earnings_beat_model: Bayesian earnings beat probability
    - analyze_distress_distribution_legacy: Distress risk distribution analysis
    - create_composite_quality_score: Multi-factor quality scoring
    - create_summary_dashboard: KPI summary dashboard
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from analytics.data_utils import (
    load_feature_data_from_db,
    backfill_feature_columns,
    export_to_analytics_db,
    safe_get_column,
    load_feature_categories_from_db,
    _get_fallback_feature_categories,
    compare_registry_with_local,
)

# --- InferenceData schema (ArviZ / xarray bridge) ---
try:
    from analytics.inference_schema import (
        ARVIZ_AVAILABLE,
        build_beat_probability_inference_data,
        build_monte_carlo_inference_data,
        summarize_inference_data,
    )
except ImportError:
    ARVIZ_AVAILABLE = False

from analytics.visualizations.valuation import (
    create_valuation_multiples_comparison,
    create_valuation_distribution_dashboard,
    create_relative_valuation_matrix,
    create_valuation_vs_growth_quadrant,
    create_historical_valuation_percentile,
)
from analytics.visualizations.earnings_quality import (
    create_earnings_surprise_dashboard,
    create_eps_trajectory_analysis,
    create_earnings_quality_decomposition,
    create_beat_rate_heatmap,
    create_earnings_consistency_matrix,
)
from analytics.visualizations.quality_risk import (
    create_piotroski_fscore_breakdown,
    create_altman_zscore_distribution,
    create_quality_risk_quadrant,
    create_beneish_mscore_analysis,
    create_risk_tier_sunburst,
    create_distress_early_warning_dashboard,
)
from analytics.visualizations.growth_analysis import (
    create_growth_waterfall_chart,
    create_growth_consistency_matrix,
    create_growth_vs_profitability_quadrant,
    create_growth_acceleration_chart,
    create_sustainable_growth_analysis,
)

# Import shared constants from canonical source
from analytics.visualizations._shared import PLOTLY_TEMPLATE, COLORS

# Re-export statistical functions from canonical source
from analytics.statistical_analysis import (
    monte_carlo_price_target_simulation,
    bayesian_earnings_beat_model,
    analyze_distress_distribution,
)

px.defaults.template = PLOTLY_TEMPLATE

# Load categories dynamically at module level (with fallback)
FEATURE_CATEGORIES = load_feature_categories_from_db()


# NOTE: load_feature_data_from_db, backfill_feature_columns, safe_get_column,
# load_feature_categories_from_db, _get_fallback_feature_categories, and
# compare_registry_with_local are imported from data_utils.py (canonical source).


def ensure_subplot_data(
    fig: go.Figure, row: int, col: int, has_data: bool, placeholder_text: str = "No data available"
) -> None:
    """
    Add placeholder annotation if subplot has no data.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to modify
    row : int
        Subplot row (1-indexed)
    col : int
        Subplot column (1-indexed)
    has_data : bool
        Whether the subplot has valid data
    placeholder_text : str
        Text to display if no data
    """
    if not has_data:
        # Calculate approximate position for annotation
        x_pos = (col - 0.5) / 2
        y_pos = 1 - (row - 0.5) / 2
        fig.add_annotation(
            text=placeholder_text,
            x=x_pos,
            y=y_pos,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="gray"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=4,
        )


def create_interactive_momentum_dashboard(df: pd.DataFrame) -> Figure:
    """
    Create an interactive momentum analysis dashboard with hover details.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing momentum columns:
        - price_momentum_1m, price_momentum_3m, price_momentum_6m, price_momentum_1y
        - price_momentum_5d, price_momentum_3y, price_momentum_5y, pt_vs_price_momentum
        - range_52w_position
        - ticker, name, industry

    Returns
    -------
    Figure
        Plotly Figure with 4 subplot panels:
        1. Momentum distribution by period (all 8 momentum columns)
        2. 3-Month momentum by industry
        3. 52-Week range position
        4. Short vs medium-term momentum scatter
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Momentum Distribution by Period",
            "3-Month Momentum by Industry",
            "52-Week Range Position",
            "Short vs Medium-Term Momentum",
        ],
        specs=[
            [{"type": "histogram"}, {"type": "box"}],
            [{"type": "histogram"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # All available momentum columns from mv_all_stock_features
    momentum_cols = [
        "price_momentum_5d",
        "price_momentum_1m",
        "price_momentum_3m",
        "price_momentum_6m",
        "price_momentum_1y",
        "price_momentum_3y",
        "price_momentum_5y",
        "pt_vs_price_momentum",
    ]
    colors = [
        "#1abc9c",  # 5-Day - teal
        "#3498db",  # 1-Month - blue
        "#e74c3c",  # 3-Month - red
        "#2ecc71",  # 6-Month - green
        "#9b59b6",  # 1-Year - purple
        "#f39c12",  # 3-Year - orange
        "#e91e63",  # 5-Year - pink
        "#00bcd4",  # PT vs Price - cyan
    ]
    labels = [
        "5-Day",
        "1-Month",
        "3-Month",
        "6-Month",
        "1-Year",
        "3-Year",
        "5-Year",
        "PT vs Price",
    ]

    # Panel 1: Overlaid momentum histograms
    panel1_has_data = False
    for col, color, label in zip(momentum_cols, colors, labels):
        if col in df.columns:
            data = df[col].dropna().clip(-50, 100)
            if len(data) > 0:
                panel1_has_data = True
                fig.add_trace(
                    go.Histogram(x=data, name=label, marker_color=color, opacity=0.6, nbinsx=50),
                    row=1,
                    col=1,
                )

    # Add placeholder if no data for panel 1
    if not panel1_has_data:
        fig.add_trace(
            go.Histogram(x=[0], name="No Data", marker_color="#adb5bd", opacity=0.3),
            row=1,
            col=1,
        )
        fig.add_annotation(
            text="No momentum data available",
            x=0.25,
            y=0.75,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="gray"),
        )

    # Panel 2: Box plot by industry (interactive)
    panel2_has_data = False
    if "industry" in df.columns and "price_momentum_3m" in df.columns:
        valid_data = df.dropna(subset=["industry", "price_momentum_3m"])
        if len(valid_data) > 0:
            panel2_has_data = True
            fig.add_trace(
                go.Box(
                    x=valid_data["industry"],
                    y=valid_data["price_momentum_3m"].clip(-50, 100),
                    marker_color="#3498db",
                    name="3M Momentum",
                    boxpoints="outliers",
                ),
                row=1,
                col=2,
            )

    if not panel2_has_data:
        fig.add_trace(
            go.Box(x=["N/A"], y=[0], marker_color="#adb5bd", name="No Data"),
            row=1,
            col=2,
        )

    # Panel 3: 52-week range position
    panel3_has_data = False
    if "range_52w_position" in df.columns:
        range_data = df["range_52w_position"].dropna()
        if len(range_data) > 0:
            panel3_has_data = True
            fig.add_trace(
                go.Histogram(
                    x=range_data,
                    nbinsx=30,
                    marker_color="#9b59b6",
                    name="52W Position",
                    hovertemplate="Position: %{x:.2f}<br>Count: %{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.add_vline(
                x=range_data.median(),
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"Median: {range_data.median():.2f}",
                row=2,
                col=1,
            )

    if not panel3_has_data:
        fig.add_trace(
            go.Histogram(x=[0.5], name="No Data", marker_color="#adb5bd", opacity=0.3),
            row=2,
            col=1,
        )

    # Panel 4: Scatter with hover details
    panel4_has_data = False
    if "price_momentum_1m" in df.columns and "price_momentum_6m" in df.columns:
        valid_mask = df["price_momentum_1m"].notna() & df["price_momentum_6m"].notna()
        scatter_cols = ["price_momentum_1m", "price_momentum_6m"]
        optional_cols = ["ticker", "name", "industry"]
        available_cols = scatter_cols + [c for c in optional_cols if c in df.columns]

        scatter_df = df.loc[valid_mask, available_cols].copy()
        scatter_df["price_momentum_1m"] = scatter_df["price_momentum_1m"].clip(-50, 100)
        scatter_df["price_momentum_6m"] = scatter_df["price_momentum_6m"].clip(-50, 200)

        if len(scatter_df) > 0:
            panel4_has_data = True
            # Build hover template dynamically based on available columns
            hover_parts = []
            customdata_cols = []
            if "ticker" in scatter_df.columns:
                hover_parts.append("<b>%{text}</b>")
            if "name" in scatter_df.columns:
                hover_parts.append("%{customdata[0]}")
                customdata_cols.append("name")
            if "industry" in scatter_df.columns:
                idx = len(customdata_cols)
                hover_parts.append(f"Industry: %{{customdata[{idx}]}}")
                customdata_cols.append("industry")
            hover_parts.extend(["1M: %{x:.1f}%", "6M: %{y:.1f}%"])

            customdata = (
                np.stack([scatter_df[c] for c in customdata_cols], axis=-1)
                if customdata_cols
                else None
            )

            fig.add_trace(
                go.Scatter(
                    x=scatter_df["price_momentum_1m"],
                    y=scatter_df["price_momentum_6m"],
                    mode="markers",
                    marker=dict(size=5, opacity=0.4, color="#3498db"),
                    text=scatter_df.get("ticker", None),
                    customdata=customdata,
                    hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
                    name="Stocks",
                ),
                row=2,
                col=2,
            )

            # Reference lines for scatter
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=2)
            fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=2)

    if not panel4_has_data:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0], mode="markers", marker=dict(size=10, color="#adb5bd"), name="No Data"
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=800,
        title_text="📈 Interactive Momentum Analysis Dashboard",
        template=PLOTLY_TEMPLATE,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="Momentum (%)", row=1, col=1)
    fig.update_xaxes(title_text="Industry", tickangle=-45, row=1, col=2)
    fig.update_xaxes(title_text="52W Range Position", row=2, col=1)
    fig.update_xaxes(title_text="1-Month Momentum (%)", row=2, col=2)
    fig.update_yaxes(title_text="6-Month Momentum (%)", row=2, col=2)

    return fig


def create_interactive_valuation_heatmap(df: pd.DataFrame) -> Figure:
    """
    Create an interactive valuation heatmap with click-to-filter capability.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing valuation columns:
        - p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio
        - industry

    Returns
    -------
    Figure
        Plotly Figure with heatmap showing median valuation metrics by industry
    """
    valuation_cols = ["p_e_ratio", "p_b_ratio", "ev_ebitda_ratio", "ev_sales_ratio"]
    val_labels = ["P/E", "P/B", "EV/EBITDA", "EV/Sales"]

    # Filter out null industries
    df_filtered = df[df["industry"].notna()] if "industry" in df.columns else df
    sectors = sorted(df_filtered["industry"].dropna().unique())

    # Build heatmap data
    heatmap_data = []
    hover_text = []

    for sector in sectors:
        sector_df = df_filtered[df_filtered["industry"] == sector]
        row_vals = []
        row_hover = []
        # Computes median, IQR, and count for each sector
        for col, label in zip(valuation_cols, val_labels):
            if col in sector_df.columns:
                median_val = sector_df[col].median()
                count = sector_df[col].notna().sum()
                q25 = sector_df[col].quantile(0.25)
                q75 = sector_df[col].quantile(0.75)
            else:
                median_val = 0
                count = 0
                q25 = 0
                q75 = 0
            row_vals.append(median_val if pd.notna(median_val) else 0)
            row_hover.append(
                f"<b>{sector}</b><br>{label}: {median_val:.1f}<br>"
                + f"IQR: [{q25:.1f}, {q75:.1f}]<br>N={count}"
            )
        heatmap_data.append(row_vals)
        hover_text.append(row_hover)

    heatmap_array = np.array(heatmap_data)

    # Normalize for color scale
    min_vals = heatmap_array.min(axis=0)
    max_vals = heatmap_array.max(axis=0)
    heatmap_norm = (heatmap_array - min_vals) / (max_vals - min_vals + 1e-10)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_norm,
            x=val_labels,
            y=sectors,
            text=np.round(heatmap_array, 1),
            texttemplate="%{text}",
            textfont={"size": 10},
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale="RdYlGn_r",
            colorbar=dict(title="Relative<br>Valuation"),
        )
    )

    fig.update_layout(
        title="📊 Median Valuation Metrics by Industry<br><sup>Green=Cheaper, Red=Expensive (Normalized)</sup>",
        height=max(600, len(sectors) * 25),
        template=PLOTLY_TEMPLATE,
        xaxis_title="Valuation Metric",
        yaxis_title="Industry",
    )

    return fig


def create_leverage_liquidity_quadrant(df: pd.DataFrame) -> Figure:
    """
    Interactive quadrant analysis of leverage vs liquidity with distress coloring.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - debt_to_equity, current_ratio, combined_distress_risk_score
        - ticker, name, industry

    Returns
    -------
    Figure
        Plotly Figure with scatter plot showing leverage vs liquidity quadrants
    """
    required_cols = [
        "ticker",
        "name",
        "industry",
        "debt_to_equity",
        "current_ratio",
        "combined_distress_risk_score",
    ]
    available_cols = [c for c in required_cols if c in df.columns]

    plot_df = df[available_cols].dropna().copy()

    if "debt_to_equity" in plot_df.columns:
        plot_df["debt_to_equity"] = plot_df["debt_to_equity"].clip(0, 3)
    if "current_ratio" in plot_df.columns:
        plot_df["current_ratio"] = plot_df["current_ratio"].clip(0, 5)

    fig = px.scatter(
        plot_df,
        x="debt_to_equity",
        y="current_ratio",
        color=(
            "combined_distress_risk_score"
            if "combined_distress_risk_score" in plot_df.columns
            else None
        ),
        color_continuous_scale="RdYlGn",
        hover_data=(
            ["ticker", "name", "industry"]
            if all(c in plot_df.columns for c in ["ticker", "name", "industry"])
            else None
        ),
        title="📉 Leverage vs Liquidity Quadrant Analysis",
        labels={
            "debt_to_equity": "Debt-to-Equity Ratio",
            "current_ratio": "Current Ratio (Liquidity)",
            "combined_distress_risk_score": "Distress Risk Score",
        },
        height=650,
    )

    # Add quadrant lines
    fig.add_hline(
        y=1.5, line_dash="dash", line_color="#2ecc71", annotation_text="Healthy Liquidity (CR=1.5)"
    )
    fig.add_vline(
        x=1.0, line_dash="dash", line_color="#e74c3c", annotation_text="High Leverage (D/E=1)"
    )

    # Add quadrant labels
    fig.add_annotation(
        x=0.3, y=4.5, text="✅ Low Risk", showarrow=False, font=dict(size=14, color="#2ecc71")
    )
    fig.add_annotation(
        x=2.5, y=0.5, text="⚠️ High Risk", showarrow=False, font=dict(size=14, color="#e74c3c")
    )
    fig.add_annotation(
        x=2.5, y=4.5, text="🔄 Mixed", showarrow=False, font=dict(size=12, color="#f39c12")
    )
    fig.add_annotation(
        x=0.3, y=0.5, text="💧 Illiquid", showarrow=False, font=dict(size=12, color="#3498db")
    )

    fig.update_traces(marker=dict(size=6, opacity=0.6))
    fig.update_layout(template=PLOTLY_TEMPLATE)

    return fig


def analyze_distress_distribution_legacy(df: pd.DataFrame) -> Figure:
    """
    Analyze distress risk score distribution with tail risk metrics.

    Uses concepts from MCMC sampling to understand distribution shape.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - combined_distress_risk_score
        - industry

    Returns
    -------
    Figure
        Plotly Figure with 4 panels:
        1. Distress risk score distribution with fitted normal
        2. Empirical CDF
        3. Q-Q plot vs normal
        4. Tail risk by industry
    """
    distress_data = df["combined_distress_risk_score"].dropna()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Distress Risk Score Distribution",
            "Empirical CDF",
            "Q-Q Plot vs Normal",
            "Tail Risk by Industry",
        ],
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # Panel 1: Histogram with fitted distribution
    fig.add_trace(
        go.Histogram(
            x=distress_data,
            nbinsx=50,
            name="Observed",
            marker_color="#3498db",
            opacity=0.7,
            histnorm="probability density",
        ),
        row=1,
        col=1,
    )

    # Fit normal for comparison
    mu, std = distress_data.mean(), distress_data.std()
    x_range = np.linspace(0, 100, 100)
    normal_pdf = stats.norm.pdf(x_range, mu, std)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_pdf,
            mode="lines",
            name="Normal Fit",
            line=dict(color="#e74c3c", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Panel 2: Empirical CDF
    sorted_data = np.sort(distress_data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig.add_trace(
        go.Scatter(x=sorted_data, y=ecdf, mode="lines", name="ECDF", line=dict(color="#00bc8c")),
        row=1,
        col=2,
    )
    # Add risk thresholds
    fig.add_vline(
        x=30, line_dash="dot", line_color="#e74c3c", row=1, col=2, annotation_text="High Risk (<30)"
    )
    fig.add_vline(
        x=70, line_dash="dot", line_color="#2ecc71", row=1, col=2, annotation_text="Low Risk (>70)"
    )

    # Panel 3: Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    empirical_quantiles = np.percentile(distress_data, np.linspace(1, 99, 100))
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=empirical_quantiles,
            mode="markers",
            marker=dict(size=4, color="#9b59b6"),
            name="Q-Q",
        ),
        row=2,
        col=1,
    )
    # Reference line
    fig.add_trace(
        go.Scatter(
            x=[-3, 3],
            y=[mu - 3 * std, mu + 3 * std],
            mode="lines",
            line=dict(dash="dash", color="white"),
            name="Normal Ref",
        ),
        row=2,
        col=1,
    )

    # Panel 4: Tail risk by industry (% below 30)
    if "industry" in df.columns:
        tail_risk = (
            df.groupby("industry")
            .apply(
                lambda x: (x["combined_distress_risk_score"] < 30).mean() * 100,
                include_groups=False,
            )
            .sort_values(ascending=False)
        )

        fig.add_trace(
            go.Bar(
                x=tail_risk.values[:15],
                y=tail_risk.index[:15],
                orientation="h",
                marker_color="#e74c3c",
                name="High Risk %",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=800,
        title_text="📉 Financial Distress Risk Distribution Analysis",
        template=PLOTLY_TEMPLATE,
        showlegend=True,
    )

    # Summary statistics annotation
    var_5 = distress_data.quantile(0.05)
    var_1 = distress_data.quantile(0.01)
    high_risk_pct = (distress_data < 30).mean() * 100

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"<b>Risk Metrics</b><br>"
        + f"VaR(5%): {var_5:.1f}<br>"
        + f"VaR(1%): {var_1:.1f}<br>"
        + f"High Risk (<30): {high_risk_pct:.1f}%",
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="white",
        borderwidth=1,
    )

    return fig


def create_composite_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite quality score combining multiple factors with probabilistic normalization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing quality factor columns:
        - piotroski_f_score, earnings_quality_composite
        - cash_flow_quality_score, combined_distress_risk_score
        - accounting_quality_score, dilution_score
        - beta_stability_score, long_term_trend_score, eps_trajectory_score

    Returns
    -------
    pd.DataFrame
        DataFrame with composite scores:
        - ticker, name, industry, market_cap
        - composite_quality_score (0-100)
        - quality_tier (Low, Below Avg, Above Avg, High)
        Sorted by composite_quality_score descending
    """

    # Define factor weights by category
    factor_weights = {
        "piotroski_f_score": 0.15,
        "earnings_quality_composite": 0.15,
        "cash_flow_quality_score": 0.12,
        "combined_distress_risk_score": 0.12,
        "accounting_quality_score": 0.10,
        "dilution_score": 0.08,
        "beta_stability_score": 0.08,
        "long_term_trend_score": 0.10,
        "eps_trajectory_score": 0.10,
    }

    # Select base columns
    base_cols = [
        "ticker",
        "name",
        "sector",
        "industry",
        "region",
        "country",
        "exchange",
        "market_cap",
        "enterprise_value",
        "last_price",
        "price_target",
        "piotroski_f_score",
        "earnings_quality_composite",
        "cash_flow_quality_score",
        "combined_distress_risk_score",
        "accounting_quality_score",
        "dilution_score",
        "beta_stability_score",
        "long_term_trend_score",
        "eps_trajectory_score",
    ]
    available_base = [c for c in base_cols if c in df.columns]
    result_df = df[available_base].copy()

    # Normalize each factor to 0-100 percentile rank
    for factor, weight in factor_weights.items():
        if factor in df.columns:
            # Percentile rank (0-100)
            result_df[f"{factor}_pctl"] = df[factor].rank(pct=True) * 100
        else:
            result_df[f"{factor}_pctl"] = 50  # Neutral if missing

    # Compute weighted composite score
    composite = np.zeros(len(result_df))
    total_weight = 0

    for factor, weight in factor_weights.items():
        pctl_col = f"{factor}_pctl"
        if pctl_col in result_df.columns:
            valid_mask = result_df[pctl_col].notna()
            composite[valid_mask] += result_df.loc[valid_mask, pctl_col] * weight
            total_weight += weight

    result_df["composite_quality_score"] = (
        composite / total_weight if total_weight > 0 else composite
    )

    # Add probability interpretation
    result_df["quality_tier"] = pd.cut(
        result_df["composite_quality_score"],
        bins=[0, 30, 50, 70, 100],
        labels=["Low", "Below Avg", "Above Avg", "High"],
    )

    # Drop percentile columns for cleaner output
    pctl_cols = [c for c in result_df.columns if c.endswith("_pctl")]
    result_df = result_df.drop(columns=pctl_cols)

    return result_df.sort_values("composite_quality_score", ascending=False).reset_index(drop=True)


def create_summary_dashboard(df: pd.DataFrame) -> Figure:
    """
    Create a KPI summary dashboard using Plotly indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing various financial metrics

    Returns
    -------
    Figure
        Plotly Figure with 8 KPI indicator panels
    """
    fig = make_subplots(
        rows=2,
        cols=4,
        specs=[[{"type": "indicator"}] * 4, [{"type": "indicator"}] * 4],
        subplot_titles=[
            "Total Stocks",
            "Avg P/E",
            "Median Upside",
            "High Quality %",
            "Profitable %",
            "Strong F-Score",
            "Bullish Sentiment",
            "Low Distress %",
        ],
    )

    # Row 1 indicators
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=len(df),
            number={"suffix": "", "font": {"size": 40}},
            title={"text": "Stocks Analyzed"},
        ),
        row=1,
        col=1,
    )

    pe_median = df["p_e_ratio"].median() if "p_e_ratio" in df.columns else 0
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=pe_median,
            number={"suffix": "x", "font": {"size": 40}},
            title={"text": "Median P/E"},
        ),
        row=1,
        col=2,
    )

    upside_median = df["upside_potential"].median() if "upside_potential" in df.columns else 0
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=upside_median,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Median Upside"},
        ),
        row=1,
        col=3,
    )

    high_quality_pct = (
        (df["earnings_quality_composite"] > 70).mean() * 100
        if "earnings_quality_composite" in df.columns
        else 0
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=high_quality_pct,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "High Quality"},
        ),
        row=1,
        col=4,
    )

    # Row 2 indicators
    profitable_pct = (
        (df["net_margin_pct"] > 0).mean() * 100 if "net_margin_pct" in df.columns else 0
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=profitable_pct,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Profitable"},
        ),
        row=2,
        col=1,
    )

    strong_fscore_pct = (
        (df["piotroski_f_score"] >= 7).mean() * 100 if "piotroski_f_score" in df.columns else 0
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=strong_fscore_pct,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Strong F-Score"},
        ),
        row=2,
        col=2,
    )

    bullish_avg = df["analyst_bullish_pct"].mean() if "analyst_bullish_pct" in df.columns else 0
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=bullish_avg,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Avg Bullish %"},
        ),
        row=2,
        col=3,
    )

    low_distress_pct = (
        (df["combined_distress_risk_score"] >= 70).mean() * 100
        if "combined_distress_risk_score" in df.columns
        else 0
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=low_distress_pct,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Low Distress"},
        ),
        row=2,
        col=4,
    )

    fig.update_layout(
        height=400,
        title_text="📊 Feature Analytics Summary Dashboard",
        template=PLOTLY_TEMPLATE,
    )

    return fig


def main():
    """
    Main execution function demonstrating feature analytics workflow.

    This function demonstrates the complete workflow:
    1. Load feature data from database
    2. Backfill missing columns
    3. Generate analytics visualizations
    4. Export results

    The workflow replicates the feature_analytics.ipynb notebook functionality.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 80)
    print("Feature Analytics - Database Loading and Visualization")
    print("=" * 80)
    print()

    # Step 1: Load data from database
    print("Step 1: Loading feature data from database...")
    try:
        df = load_feature_data_from_db(earnings_date_filter="2026-01-01")
        print(f"✓ Loaded {len(df)} stocks with {len(df.columns)} features")
        print(f"  Date range: {df['next_earnings'].min()} to {df['next_earnings'].max()}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nPlease ensure:")
        print("  1. PostgreSQL is running")
        print("  2. DB_URL environment variable is set")
        print("  3. mv_all_stock_features view exists")
        return

    print()

    # Step 2: Backfill missing columns
    print("Step 2: Backfilling missing columns...")
    df = backfill_feature_columns(df)
    print(f"✓ Backfill complete. Total columns: {len(df.columns)}")
    print()

    # Step 3: Generate visualizations
    print("Step 3: Generating analytics visualizations...")

    try:
        # Summary dashboard
        print("  - Creating summary dashboard...")
        fig_summary = create_summary_dashboard(df)
        output_dir = "outputs/analytics"
        os.makedirs(output_dir, exist_ok=True)
        fig_summary.write_html(f"{output_dir}/feature_analytics_summary.html")
        print(f"    ✓ Saved to {output_dir}/feature_analytics_summary.html")

        # Momentum dashboard
        print("  - Creating momentum dashboard...")
        fig_momentum = create_interactive_momentum_dashboard(df)
        fig_momentum.write_html(f"{output_dir}/feature_analytics_momentum.html")
        print(f"    ✓ Saved to {output_dir}/feature_analytics_momentum.html")

        # Valuation heatmap
        if "industry" in df.columns:
            print("  - Creating valuation heatmap...")
            fig_valuation = create_interactive_valuation_heatmap(df)
            fig_valuation.write_html(f"{output_dir}/feature_analytics_valuation.html")
            print(f"    ✓ Saved to {output_dir}/feature_analytics_valuation.html")

        # Leverage-liquidity quadrant
        if all(col in df.columns for col in ["debt_to_equity", "current_ratio"]):
            print("  - Creating leverage-liquidity quadrant...")
            fig_leverage = create_leverage_liquidity_quadrant(df)
            fig_leverage.write_html(f"{output_dir}/feature_analytics_leverage.html")
            print(f"    ✓ Saved to {output_dir}/feature_analytics_leverage.html")

        # Distress distribution
        if "combined_distress_risk_score" in df.columns:
            print("  - Creating distress distribution analysis...")
            fig_distress = analyze_distress_distribution_legacy(df)
            fig_distress.write_html(f"{output_dir}/feature_analytics_distress.html")
            print(f"    ✓ Saved to {output_dir}/feature_analytics_distress.html")

        # --- NEW: Valuation Analysis Visualizations ---
        print("  - Creating valuation analysis visualizations...")

        # Valuation distribution dashboard
        fig_val_dist = create_valuation_distribution_dashboard(df)
        fig_val_dist.write_html(f"{output_dir}/valuation_distribution_dashboard.html")
        print(f"    ✓ Saved valuation_distribution_dashboard.html")

        # Relative valuation matrix
        fig_val_matrix = create_relative_valuation_matrix(df)
        fig_val_matrix.write_html(f"{output_dir}/relative_valuation_matrix.html")
        print(f"    ✓ Saved relative_valuation_matrix.html")

        # Valuation vs growth quadrant
        fig_val_growth = create_valuation_vs_growth_quadrant(df)
        fig_val_growth.write_html(f"{output_dir}/valuation_vs_growth_quadrant.html")
        print(f"    ✓ Saved valuation_vs_growth_quadrant.html")

        # Historical valuation percentile
        fig_val_pct = create_historical_valuation_percentile(df)
        fig_val_pct.write_html(f"{output_dir}/historical_valuation_percentile.html")
        print(f"    ✓ Saved historical_valuation_percentile.html")

        # --- NEW: Earnings Quality Visualizations ---
        print("  - Creating earnings quality visualizations...")

        # Earnings surprise dashboard
        fig_earn_surprise = create_earnings_surprise_dashboard(df)
        fig_earn_surprise.write_html(f"{output_dir}/earnings_surprise_dashboard.html")
        print(f"    ✓ Saved earnings_surprise_dashboard.html")

        # EPS trajectory analysis
        fig_eps_traj = create_eps_trajectory_analysis(df)
        fig_eps_traj.write_html(f"{output_dir}/eps_trajectory_analysis.html")
        print(f"    ✓ Saved eps_trajectory_analysis.html")

        # Beat rate heatmap
        fig_beat_rate = create_beat_rate_heatmap(df)
        fig_beat_rate.write_html(f"{output_dir}/beat_rate_heatmap.html")
        print(f"    ✓ Saved beat_rate_heatmap.html")

        # Earnings consistency matrix
        fig_earn_consist = create_earnings_consistency_matrix(df)
        fig_earn_consist.write_html(f"{output_dir}/earnings_consistency_matrix.html")
        print(f"    ✓ Saved earnings_consistency_matrix.html")

        # --- NEW: Quality & Risk Visualizations ---
        print("  - Creating quality & risk visualizations...")

        # Piotroski F-Score breakdown
        fig_fscore = create_piotroski_fscore_breakdown(df)
        fig_fscore.write_html(f"{output_dir}/piotroski_fscore_breakdown.html")
        print(f"    ✓ Saved piotroski_fscore_breakdown.html")

        # Altman Z-Score distribution
        fig_zscore = create_altman_zscore_distribution(df)
        fig_zscore.write_html(f"{output_dir}/altman_zscore_distribution.html")
        print(f"    ✓ Saved altman_zscore_distribution.html")

        # Quality-Risk quadrant
        fig_qr_quad = create_quality_risk_quadrant(df)
        fig_qr_quad.write_html(f"{output_dir}/quality_risk_quadrant.html")
        print(f"    ✓ Saved quality_risk_quadrant.html")

        # Beneish M-Score analysis
        fig_mscore = create_beneish_mscore_analysis(df)
        fig_mscore.write_html(f"{output_dir}/beneish_mscore_analysis.html")
        print(f"    ✓ Saved beneish_mscore_analysis.html")

        # Risk tier sunburst
        fig_risk_sun = create_risk_tier_sunburst(df)
        fig_risk_sun.write_html(f"{output_dir}/risk_tier_sunburst.html")
        print(f"    ✓ Saved risk_tier_sunburst.html")

        # Distress early warning dashboard
        fig_distress_warn = create_distress_early_warning_dashboard(df)
        fig_distress_warn.write_html(f"{output_dir}/distress_early_warning_dashboard.html")
        print(f"    ✓ Saved distress_early_warning_dashboard.html")

        # --- NEW: Growth Analysis Visualizations ---
        print("  - Creating growth analysis visualizations...")

        # Growth consistency matrix
        fig_growth_consist = create_growth_consistency_matrix(df)
        fig_growth_consist.write_html(f"{output_dir}/growth_consistency_matrix.html")
        print(f"    ✓ Saved growth_consistency_matrix.html")

        # Growth vs profitability quadrant
        fig_growth_prof = create_growth_vs_profitability_quadrant(df)
        fig_growth_prof.write_html(f"{output_dir}/growth_vs_profitability_quadrant.html")
        print(f"    ✓ Saved growth_vs_profitability_quadrant.html")

        # Growth acceleration chart
        fig_growth_accel = create_growth_acceleration_chart(df)
        fig_growth_accel.write_html(f"{output_dir}/growth_acceleration_chart.html")
        print(f"    ✓ Saved growth_acceleration_chart.html")

        # Sustainable growth analysis
        fig_sust_growth = create_sustainable_growth_analysis(df)
        fig_sust_growth.write_html(f"{output_dir}/sustainable_growth_analysis.html")
        print(f"    ✓ Saved sustainable_growth_analysis.html")

    except Exception as e:
        print(f"  ✗ Error generating visualizations: {e}")
        import traceback

        traceback.print_exc()

    print()

    # Step 4: Generate advanced analytics
    print("Step 4: Generating advanced analytics...")

    try:
        # Monte Carlo simulation
        required_mc_cols = ["price_target", "price_target_high", "price_target_low", "last_price"]
        if all(col in df.columns for col in required_mc_cols):
            print("  - Running Monte Carlo price target simulation...")
            mc_results = monte_carlo_price_target_simulation(df, max_stocks=10000)
            if len(mc_results) > 0:
                export_to_analytics_db(mc_results, "monte_carlo_simulation")
                print(
                    f"    ✓ Exported {len(mc_results)} simulations to analytics.monte_carlo_simulation"
                )
                print(f"    Top 5 by risk-reward ratio:")
                top5 = mc_results.nlargest(5, "risk_reward_ratio")[
                    ["ticker", "name", "expected_upside_pct", "risk_reward_ratio"]
                ]
                print(top5.to_string(index=False))

                # Build InferenceData for Monte Carlo simulation
                if ARVIZ_AVAILABLE:
                    try:
                        idata_mc = build_monte_carlo_inference_data(
                            mc_results,
                            df,
                            n_simulations=10000,
                        )
                        mc_summary = summarize_inference_data(idata_mc)
                        print(
                            f"    ✓ InferenceData (MC): {mc_summary.get('n_draws', 0)} simulations"
                        )
                    except Exception as e:
                        logging.warning("InferenceData (MC) build failed: %s", e)

        # Bayesian earnings model
        if "eps_positive_streak" in df.columns:
            print("  - Running Bayesian earnings beat model...")
            bayesian_results = bayesian_earnings_beat_model(df)
            if len(bayesian_results) > 0:
                export_to_analytics_db(bayesian_results, "bayesian_earnings_model")
                print(
                    f"    ✓ Exported {len(bayesian_results)} predictions to analytics.bayesian_earnings_model"
                )

                # Build InferenceData for beat probability
                if ARVIZ_AVAILABLE and "posterior_alpha" in bayesian_results.columns:
                    try:
                        idata_beat = build_beat_probability_inference_data(
                            bayesian_results,
                            df,
                            n_posterior_samples=4000,
                            n_chains=4,
                        )
                        beat_summary = summarize_inference_data(idata_beat)
                        print(
                            f"    ✓ InferenceData (beat): {beat_summary.get('n_chains', 0)} chains × {beat_summary.get('n_draws', 0)} draws"
                        )
                    except Exception as e:
                        logging.warning("InferenceData (beat) build failed: %s", e)

        # Composite quality score
        print("  - Creating composite quality scores...")
        quality_scores = create_composite_quality_score(df)
        export_to_analytics_db(quality_scores, "composite_quality_scores")
        print(f"    ✓ Exported {len(quality_scores)} scores to analytics.composite_quality_scores")
        print(f"    Top 100 highest quality stocks:")
        top10 = quality_scores.head(100)[
            ["ticker", "name", "composite_quality_score", "quality_tier"]
        ]
        print(top10.to_string(index=False))

    except Exception as e:
        print(f"  ✗ Error generating advanced analytics: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 80)
    print("Feature Analytics Complete!")
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
