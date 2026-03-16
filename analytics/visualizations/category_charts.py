"""
Category-specific visualization functions for feature analytics.
This module provides reusable chart functions for various feature categories
including analyst sentiment, earnings quality, growth metrics, cash flow,
dividend features, R&D investment, inventory, goodwill & M&A, and CapEx.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analytics.visualizations._shared import PLOTLY_TEMPLATE, COLORS, resolve_column

# =============================================================================
# Layout defaults & helpers
# =============================================================================
_DEFAULT_HEIGHT = 600
_DEFAULT_WIDTH = 1000
_DEFAULT_MARGIN = dict(l=80, r=40, t=60, b=60)


def _apply_default_layout(
    fig: go.Figure,
    *,
    title: str | None = None,
    height: int = _DEFAULT_HEIGHT,
    width: int = _DEFAULT_WIDTH,
    **extra_layout,
) -> go.Figure:
    """Apply the standard chart layout (template, size, margin) to *fig*."""
    layout_kwargs = dict(
        template=PLOTLY_TEMPLATE,
        height=height,
        width=width,
        margin=_DEFAULT_MARGIN,
    )
    if title is not None:
        layout_kwargs["title"] = title
    layout_kwargs.update(extra_layout)
    fig.update_layout(**layout_kwargs)
    return fig


def _col_or_none(df: pd.DataFrame, col: str) -> str | None:
    """Return *col* if it exists in *df*, otherwise ``None``."""
    return col if col in df.columns else None


def _no_data(title: str) -> go.Figure:
    """Return a placeholder figure when required columns are missing."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16),
    )
    _apply_default_layout(fig, title=title)
    return fig

# =============================================================================
# Analyst Sentiment Visualizations
# =============================================================================


def create_analyst_sentiment_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 30,
) -> go.Figure:
    """
    Create histogram of analyst bullish percentage distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with analyst sentiment features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 30
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly figure with histogram and marginal box plot
    """
    if "analyst_bullish_pct" not in df.columns:
        return _no_data("Analyst Bullish Percentage - No Data")

    fig = px.histogram(
        df,
        x="analyst_bullish_pct",
        color=_col_or_none(df, color_by),
        title="Analyst Bullish Percentage Distribution by Industry",
        nbins=nbins,
        marginal="box",
    )
    _apply_default_layout(fig)
    return fig


def create_analyst_upside_scatter(
    df: pd.DataFrame,
    color_by: str = "industry",
    size_col: str = "market_cap",
    max_upside: float = 500,
) -> go.Figure:
    """
    Create scatter plot of analyst rating vs upside potential.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with analyst sentiment features
    color_by : str, default "industry"
        Column to use for color grouping
    size_col : str, default "market_cap"
        Column to use for marker size
    max_upside : float, default 500
        Maximum y-axis value for upside potential

    Returns
    -------
    go.Figure
        Plotly scatter figure
    """
    if "analyst_rating_normalized" not in df.columns or "upside_potential" not in df.columns:
        return _no_data("Analyst Rating vs Upside Potential - No Data")

    fig = px.scatter(
        df,
        x="analyst_rating_normalized",
        y="upside_potential",
        color=_col_or_none(df, color_by),
        size=_col_or_none(df, size_col),
        hover_data=["ticker", "name"],
        title="Analyst Rating vs Upside Potential",
    )
    _apply_default_layout(fig, yaxis=dict(range=[None, max_upside]))
    return fig


# =============================================================================
# Earnings Quality Visualizations
# =============================================================================


def create_eps_surprise_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 40,
) -> go.Figure:
    """
    Create histogram of EPS surprise distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with earnings quality features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 40
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly figure with histogram and marginal violin plot
    """
    if "eps_surprise_pct" not in df.columns:
        return _no_data("EPS Surprise Distribution - No Data")

    fig = px.histogram(
        df,
        x="eps_surprise_pct",
        color=_col_or_none(df, color_by),
        title="EPS Surprise Distribution by Industry",
        nbins=nbins,
        marginal="violin",
    )
    _apply_default_layout(fig)
    return fig


def create_eps_trajectory_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of EPS trajectory vs GAAP adjustment gap.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with earnings quality features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with color scale
    """
    if "eps_trajectory_score" not in df.columns or "gaap_adj_eps_gap_pct" not in df.columns:
        return _no_data("EPS Trajectory vs GAAP Adjustment Gap - No Data")

    fig = px.scatter(
        df,
        x="eps_trajectory_score",
        y="gaap_adj_eps_gap_pct",
        color=_col_or_none(df, "earnings_quality_score"),
        size=_col_or_none(df, size_col),
        hover_data=[c for c in ["ticker", "name"] if c in df.columns] or None,
        title="EPS Trajectory vs GAAP Adjustment Gap",
        color_continuous_scale="RdYlGn",
    )
    _apply_default_layout(fig)
    return fig


# =============================================================================
# Growth Metrics Visualizations
# =============================================================================

_DEFAULT_GROWTH_COLS = [
    "revenue_growth_yoy",
    "ebitda_growth_yoy",
    "eps_yoy_growth",
    "fcf_growth_yoy",
    "revenue_cagr_5y",
]


def create_growth_correlation_heatmap(
    df: pd.DataFrame,
    growth_cols: Optional[list] = None,
) -> go.Figure:
    """
    Create correlation heatmap for growth metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with growth metrics
    growth_cols : list, optional
        List of growth metric columns. If None, uses default growth columns.

    Returns
    -------
    go.Figure
        Plotly heatmap figure
    """
    if growth_cols is None:
        growth_cols = _DEFAULT_GROWTH_COLS

    # Resolve aliases for each requested column
    resolved = []
    for col in growth_cols:
        rc = resolve_column(df, col)
        if rc is not None:
            resolved.append(rc)
    available_cols = resolved
    if not available_cols:
        return _no_data("Growth Metrics Correlation - No Data")

    growth_corr = df[available_cols].corr()
    fig = px.imshow(
        growth_corr,
        text_auto=".2f",
        title="Growth Metrics Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    _apply_default_layout(fig)
    return fig


def create_revenue_vs_eps_growth_scatter(
    df: pd.DataFrame,
    color_by: str = "industry",
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of revenue growth vs EPS growth.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with growth metrics
    color_by : str, default "industry"
        Column to use for color grouping
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with reference lines
    """
    x_col = resolve_column(df, "revenue_growth_yoy") or "revenue_growth_yoy"
    y_col = "eps_yoy_growth" if "eps_yoy_growth" in df.columns else "eps_growth_yoy"
    if x_col not in df.columns or y_col not in df.columns:
        return _no_data("Revenue Growth vs EPS Growth - No Data")

    hover = [c for c in ["ticker", "name", "revenue_cagr_5y"] if c in df.columns]
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_by if color_by in df.columns else None,
        size=size_col if size_col in df.columns else None,
        hover_data=hover if hover else None,
        title="Revenue Growth vs EPS Growth (YoY)",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


# =============================================================================
# Cash Flow Visualizations
# =============================================================================


def create_fcf_margin_yield_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of FCF margin vs FCF yield.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with cash flow features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with color scale
    """
    if "fcf_margin" not in df.columns or "fcf_yield" not in df.columns:
        return _no_data("FCF Margin vs FCF Yield - No Data")

    fig = px.scatter(
        df,
        x="fcf_margin",
        y="fcf_yield",
        color="fcf_positive_years" if "fcf_positive_years" in df.columns else None,
        size=size_col if size_col in df.columns else None,
        hover_data=[c for c in ["ticker", "name", "self_funding_ratio"] if c in df.columns] or None,
        title="FCF Margin vs FCF Yield (colored by FCF Positive Years)",
        color_continuous_scale="Greens",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_cash_flow_quality_boxplot(
    df: pd.DataFrame,
    group_by: str = "industry",
) -> go.Figure:
    """
    Create box plot of cash flow quality score by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with cash flow features
    group_by : str, default "industry"
        Column to use for grouping

    Returns
    -------
    go.Figure
        Plotly box plot figure
    """
    if "cash_flow_quality_score" not in df.columns:
        return _no_data("Cash Flow Quality Score - No Data")

    fig = px.box(
        df,
        x=group_by if group_by in df.columns else None,
        y="cash_flow_quality_score",
        title="Cash Flow Quality Score by Industry",
        color=group_by if group_by in df.columns else None,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


# =============================================================================
# Dividend Features Visualizations
# =============================================================================


def create_dividend_yield_payout_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of dividend yield vs payout ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with dividend features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with color scale
    """
    if "dividend_payout_ratio" not in df.columns or "dividend_yield_ltm" not in df.columns:
        return _no_data("Dividend Yield vs Payout Ratio - No Data")

    fig = px.scatter(
        df,
        x="dividend_payout_ratio",
        y="dividend_yield_ltm",
        color="dividend_streak" if "dividend_streak" in df.columns else None,
        size=size_col if size_col in df.columns else None,
        hover_data=[c for c in ["ticker", "name", "fcf_dividend_coverage"] if c in df.columns]
        or None,
        title="Dividend Yield vs Payout Ratio (colored by Dividend Streak)",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=600,
        width=1000,
        title_font_size=16,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
        xaxis_title="Dividend Payout Ratio",
        yaxis_title="Dividend Yield (LTM)",
    )
    return fig


def create_shareholder_yield_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 40,
) -> go.Figure:
    """
    Create histogram of total shareholder yield distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with dividend features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 40
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly figure with histogram and marginal box plot
    """
    if "total_shareholder_yield" not in df.columns:
        return _no_data("Total Shareholder Yield - No Data")

    fig = px.histogram(
        df,
        x="total_shareholder_yield",
        color=color_by if color_by in df.columns else None,
        title="Total Shareholder Yield Distribution by Industry",
        nbins=nbins,
        marginal="box",
    )
    fig.update_layout(
        height=600,
        width=1000,
        title_font_size=16,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=80, r=40, t=60, b=60),
        xaxis_title="Total Shareholder Yield (%)",
        yaxis_title="Count",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


# =============================================================================
# R&D Investment Visualizations
# =============================================================================


def create_rnd_intensity_boxplot(
    df: pd.DataFrame,
    group_by: str = "industry",
) -> go.Figure:
    """
    Create box plot of R&D intensity by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with R&D features
    group_by : str, default "industry"
        Column to use for grouping

    Returns
    -------
    go.Figure
        Plotly box plot figure
    """
    if "rnd_intensity_ltm" not in df.columns:
        return _no_data("R&D Intensity - No Data")

    fig = px.box(
        df,
        x=group_by if group_by in df.columns else None,
        y="rnd_intensity_ltm",
        title="R&D Intensity by Industry",
        color=group_by if group_by in df.columns else None,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_rnd_intensity_growth_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of R&D intensity vs YoY R&D growth.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with R&D features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure
    """
    if "rnd_intensity_ltm" not in df.columns or "rnd_yoy_growth" not in df.columns:
        return _no_data("R&D Intensity vs YoY Growth - No Data")

    fig = px.scatter(
        df,
        x="rnd_intensity_ltm",
        y="rnd_yoy_growth",
        color="high_rnd_intensity_flag" if "high_rnd_intensity_flag" in df.columns else None,
        size=size_col if size_col in df.columns else None,
        hover_data=[c for c in ["ticker", "name", "rnd_per_employee"] if c in df.columns] or None,
        title="R&D Intensity vs YoY R&D Growth",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_rnd_per_employee_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 30,
) -> go.Figure:
    """
    Create histogram of R&D per employee distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with R&D features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 30
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly histogram figure
    """
    if "rnd_per_employee" not in df.columns:
        return _no_data("R&D per Employee - No Data")

    df_filtered = df[df["rnd_per_employee"].notna()]
    fig = px.histogram(
        df_filtered,
        x="rnd_per_employee",
        color=_col_or_none(df_filtered, color_by),
        nbins=nbins,
        title="R&D per Employee Distribution by Industry",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


# =============================================================================
# Inventory Visualizations
# =============================================================================


def create_inventory_days_turnover_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of inventory days vs turnover.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with inventory features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure
    """
    if "inventory_days" not in df.columns:
        return _no_data("Inventory Days vs Turnover - No Data")

    # Resolve inventory_turnover column (MV uses inventory_turnover_itf)
    turnover_col = None
    for candidate in ["inventory_turnover_itf", "inventory_turnover", "inventory_turnover_mv"]:
        if candidate in df.columns:
            turnover_col = candidate
            break
    if turnover_col is None:
        return _no_data("Inventory Days vs Turnover - No Data")

    fig = px.scatter(
        df,
        x="inventory_days",
        y=turnover_col,
        color="inventory_buildup_flag" if "inventory_buildup_flag" in df.columns else None,
        size=size_col if size_col in df.columns else None,
        hover_data=[c for c in ["ticker", "name", "inventory_yoy_change"] if c in df.columns]
        or None,
        title="Inventory Days vs Turnover (flagged for buildup)",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


# =============================================================================
# Goodwill & M&A Visualizations
# =============================================================================


def create_goodwill_concentration_boxplot(
    df: pd.DataFrame,
    group_by: str = "industry",
) -> go.Figure:
    """
    Create box plot of goodwill concentration by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with goodwill features
    group_by : str, default "industry"
        Column to use for grouping

    Returns
    -------
    go.Figure
        Plotly box plot figure
    """
    if "goodwill_concentration" not in df.columns:
        return _no_data("Goodwill Concentration - No Data")

    fig = px.box(
        df,
        x=group_by if group_by in df.columns else None,
        y="goodwill_concentration",
        title="Goodwill Concentration by Industry",
        color=group_by if group_by in df.columns else None,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_goodwill_impairment_scatter(
    df: pd.DataFrame,
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of goodwill concentration vs impairment risk score.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with goodwill features
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure
    """
    if "goodwill_concentration" not in df.columns or "impairment_risk_score" not in df.columns:
        return _no_data("Goodwill vs Impairment Risk - No Data")

    fig = px.scatter(
        df,
        x="goodwill_concentration",
        y="impairment_risk_score",
        color="recent_acquisition_flag" if "recent_acquisition_flag" in df.columns else None,
        size=size_col if size_col in df.columns else None,
        hover_data=[c for c in ["ticker", "name", "goodwill_concentration"] if c in df.columns]
        or None,
        title="Goodwill Concentration vs Impairment Risk Score",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_acquisition_activity_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
) -> go.Figure:
    """
    Create histogram of recent acquisition activity by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with M&A features
    color_by : str, default "industry"
        Column to use for color grouping

    Returns
    -------
    go.Figure
        Plotly histogram figure
    """
    if "recent_acquisition_flag" not in df.columns:
        return _no_data("Acquisition Activity - No Data")

    fig = px.histogram(
        df,
        x="recent_acquisition_flag",
        color=color_by if color_by in df.columns else None,
        title="Recent Acquisition Activity by Industry",
        barmode="group",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


# =============================================================================
# CapEx & Investment Visualizations
# =============================================================================


def create_capex_growth_scatter(
    df: pd.DataFrame,
    color_by: str = "industry",
    size_col: str = "market_cap",
) -> go.Figure:
    """
    Create scatter plot of CapEx YoY growth vs 5Y average comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with CapEx features
    color_by : str, default "industry"
        Column to use for color grouping
    size_col : str, default "market_cap"
        Column to use for marker size

    Returns
    -------
    go.Figure
        Plotly scatter figure with reference lines
    """
    if "capex_yoy_growth" not in df.columns or "capex_vs_5y_avg" not in df.columns:
        return _no_data("CapEx Growth vs 5Y Average - No Data")

    fig = px.scatter(
        df,
        x="capex_yoy_growth",
        y="capex_vs_5y_avg",
        color=color_by if color_by in df.columns else None,
        size=size_col if size_col in df.columns else None,
        hover_data=[c for c in ["ticker", "name", "investment_efficiency"] if c in df.columns]
        or None,
        title="CapEx YoY Growth vs 5Y Average Comparison",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="5Y Avg")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_investment_efficiency_boxplot(
    df: pd.DataFrame,
    group_by: str = "industry",
) -> go.Figure:
    """
    Create box plot of investment efficiency by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with investment features
    group_by : str, default "industry"
        Column to use for grouping

    Returns
    -------
    go.Figure
        Plotly box plot figure
    """
    if "investment_efficiency" not in df.columns:
        return _no_data("Investment Efficiency - No Data")

    fig = px.box(
        df,
        x=group_by if group_by in df.columns else None,
        y="investment_efficiency",
        title="Investment Efficiency by Industry",
        color=group_by if group_by in df.columns else None,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_ma_intensity_histogram(
    df: pd.DataFrame,
    color_by: str = "industry",
    nbins: int = 30,
) -> go.Figure:
    """
    Create histogram of M&A intensity score distribution by industry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with M&A features
    color_by : str, default "industry"
        Column to use for color grouping
    nbins : int, default 30
        Number of histogram bins

    Returns
    -------
    go.Figure
        Plotly figure with histogram and marginal box plot
    """
    if "ma_intensity_score" not in df.columns:
        return _no_data("M&A Intensity Score - No Data")

    fig = px.histogram(
        df,
        x="ma_intensity_score",
        color=color_by if color_by in df.columns else None,
        title="M&A Intensity Score Distribution by Industry",
        nbins=nbins,
        marginal="box",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


# =============================================================================
# Multi-Category & Advanced Visualizations
# =============================================================================


def create_valuation_violin_plot(
    df: pd.DataFrame,
    metric: str = "p_e_ratio",
    group_by: str = "industry",
    max_val: Optional[float] = 100.0,
) -> go.Figure:
    """
    Create violin plot for valuation metrics across different groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    metric : str, default "p_e_ratio"
        Valuation metric to plot
    group_by : str, default "industry"
        Grouping column
    max_val : float, optional
        Maximum value to show (to filter outliers)

    Returns
    -------
    go.Figure
        Plotly violin plot figure
    """
    if metric not in df.columns:
        return _no_data(f"{metric.replace('_', ' ').title()} - No Data")

    df_plot = df.copy()
    if max_val is not None and metric in df_plot.columns:
        df_plot = df_plot[df_plot[metric] <= max_val]

    fig = px.violin(
        df_plot,
        x=group_by,
        y=metric,
        color=group_by,
        box=True,
        points="all",
        hover_data=[c for c in ["ticker", "name"] if c in df_plot.columns] or None,
        height=1000,
        width=2000,
        title=f"{metric.replace('_', ' ').title()} Distribution by {group_by.title()}",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def create_quality_risk_radar_chart(
    df: pd.DataFrame,
    ticker: str,
    metrics: Optional[list[str]] = None,
) -> go.Figure:
    """
    Create radar chart for a specific stock's quality and risk metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    ticker : str
        Ticker symbol to highlight
    metrics : list, optional
        List of metrics for radar axes. Defaults to key quality scores.

    Returns
    -------
    go.Figure
        Plotly radar chart figure
    """
    if metrics is None:
        metrics = [
            "piotroski_f_score",
            "distress_risk_score",
            "eps_trajectory_score",
            "earnings_quality_score",
            "cash_flow_quality_score",
        ]

    # Filter for the specific ticker
    stock_data = df[df["ticker"] == ticker]
    if stock_data.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"Ticker {ticker} not found", showarrow=False)
        return fig

    # Prepare values (normalized to 0-100 where needed)
    values = []
    for m in metrics:
        val = stock_data[m].iloc[0] if m in stock_data.columns else 0
        if m == "piotroski_f_score":
            val = (val / 9) * 100
        values.append(val)

    # Close the radar loop
    metrics_label = [m.replace("_", " ").title() for m in metrics]
    metrics_label.append(metrics_label[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=metrics_label,
            fill="toself",
            name=ticker,
            line_color="cyan",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=True,
        title=f"Quality & Risk Radar: {ticker}",
        template=PLOTLY_TEMPLATE,
    )

    return fig


def create_leverage_liquidity_bubble_chart(
    df: pd.DataFrame,
    size_col: str = "market_cap",
    color_by: str = "industry",
) -> go.Figure:
    """
    Create bubble chart of leverage vs liquidity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    size_col : str, default "market_cap"
        Column for bubble size
    color_by : str, default "industry"
        Column for bubble color

    Returns
    -------
    go.Figure
        Plotly bubble chart
    """
    if "current_ratio" not in df.columns or "debt_to_equity" not in df.columns:
        return _no_data("Leverage vs Liquidity - No Data")

    fig = px.scatter(
        df,
        x="current_ratio",
        y="debt_to_equity",
        size=size_col if size_col in df.columns else None,
        color=color_by if color_by in df.columns else None,
        hover_data=["ticker", "name"],
        title="Leverage (D/E) vs Liquidity (Current Ratio)",
        labels={
            "current_ratio": "Current Ratio (Liquidity)",
            "debt_to_equity": "Debt to Equity (Leverage)",
        },
    )

    # Add reference lines for healthy levels
    fig.add_vline(x=1.5, line_dash="dash", line_color="green", annotation_text="Healthy Liquidity")
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="High Leverage")

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_productivity_quadrant(
    df: pd.DataFrame,
    size_col: str = "market_cap",
    color_by: str = "industry",
) -> go.Figure:
    """
    Scatter plot mapping revenue_per_employee against ebitda_per_employee.
    """
    x_col = "revenue_per_employee"
    y_col = "ebitda_per_employee"

    if x_col not in df.columns or y_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Productivity metrics missing", showarrow=False)
        return fig

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col if size_col in df.columns else None,
        color=color_by if color_by in df.columns else None,
        hover_data=[c for c in ["ticker", "name"] if c in df.columns] or None,
        title="Employee Productivity Quadrant",
        labels={
            x_col: "Revenue per Employee",
            y_col: "EBITDA per Employee",
        },
    )

    # Add sector benchmarks (medians)
    if color_by in df.columns:
        x_median = df[x_col].median()
        y_median = df[y_col].median()
        fig.add_vline(
            x=x_median, line_dash="dot", line_color="gray", annotation_text="Market Median Rev"
        )
        fig.add_hline(
            y=y_median, line_dash="dot", line_color="gray", annotation_text="Market Median EBITDA"
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_accounting_quality_breakdown(
    df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """
    Radar chart showing the sub-components of the accounting_quality_score.
    """
    components = [
        "accounting_quality_score",
        "gaap_adj_eps_gap_pct",  # Note: lower is better for quality
        "asset_sale_boost",  # Note: lower is better for quality
        "exceptional_items_frequency",  # Note: lower is better for quality
    ]

    stock_data = df[df["ticker"] == ticker]
    if stock_data.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"Ticker {ticker} not found", showarrow=False)
        return fig

    values = []
    labels = []
    for comp in components:
        if comp in stock_data.columns:
            val = stock_data[comp].iloc[0]
            # Normalize so that higher = better quality
            if comp == "accounting_quality_score":
                score = val
            elif comp == "non_operating_income_share":
                score = max(0, 100 - val)
            elif comp == "gaap_adj_eps_gap_pct":
                score = max(0, 100 - abs(val))
            elif comp == "asset_sale_boost":
                score = max(0, 100 - val)
            elif comp == "exceptional_items_frequency":
                score = max(0, 100 - (val * 10))
            else:
                score = 50

            values.append(score)
            labels.append(comp.replace("_", " ").title())

    if not values:
        fig = go.Figure()
        fig.add_annotation(text="No accounting quality components available", showarrow=False)
        return fig

    # Close radar
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=values, theta=labels, fill="toself", name=ticker, line_color="gold")
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Accounting Quality Breakdown: {ticker}",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def create_valuation_range_visual(
    df: pd.DataFrame,
    ticker: str,
    metric: str = "p_e_ratio",
) -> go.Figure:
    """
    Chart showing the current valuation relative to its historical 3-year and 5-year ranges.
    """
    stock_data = df[df["ticker"] == ticker]
    if stock_data.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"Ticker {ticker} not found", showarrow=False)
        return fig

    current = stock_data[metric].iloc[0] if metric in stock_data.columns else None

    # Use schema-aligned columns: p_e_vs_3y_avg, p_e_vs_5y_avg, p_b_vs_5y_avg
    prefix = metric.replace("_ratio", "")
    vs_3y_col = f"{prefix}_vs_3y_avg"
    vs_5y_col = f"{prefix}_vs_5y_avg"

    vs_3y = stock_data[vs_3y_col].iloc[0] if vs_3y_col in stock_data.columns else None
    vs_5y = stock_data[vs_5y_col].iloc[0] if vs_5y_col in stock_data.columns else None

    if current is None or (vs_3y is None and vs_5y is None):
        return _no_data(f"Historical range data for {metric} not available")

    fig = go.Figure()

    categories = []
    current_vals = []
    avg_vals = []

    if vs_3y is not None:
        # vs_3y_avg is typically current / 3y_avg or a ratio; infer the average
        avg_3y = current / vs_3y if vs_3y != 0 else None
        if avg_3y is not None:
            categories.append("3Y Avg")
            current_vals.append(current)
            avg_vals.append(avg_3y)

    if vs_5y is not None:
        avg_5y = current / vs_5y if vs_5y != 0 else None
        if avg_5y is not None:
            categories.append("5Y Avg")
            current_vals.append(current)
            avg_vals.append(avg_5y)

    if not categories:
        return _no_data(f"Historical range data for {metric} not available")

    fig.add_trace(
        go.Bar(
            name="Historical Avg",
            x=avg_vals,
            y=categories,
            orientation="h",
            marker_color="rgba(100, 100, 255, 0.5)",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Current",
            x=current_vals,
            y=categories,
            orientation="h",
            marker_color="rgba(255, 100, 100, 0.7)",
        )
    )

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} vs Historical Averages: {ticker}",
        template=PLOTLY_TEMPLATE,
        xaxis_title="Valuation Multiple",
        barmode="group",
    )

    return fig


def create_balance_sheet_composition_chart(
    df: pd.DataFrame,
    group_by: str = "industry",
) -> go.Figure:
    """Create stacked bar chart of balance sheet composition by industry."""
    import plotly.express as px

    bs_cols = ["assets_fq", "debt_fq", "wc_fq"]
    available = [c for c in bs_cols if c in df.columns]

    if not available or group_by not in df.columns:
        return _no_data("Balance Sheet Composition — No Data")

    agg_df = df.groupby(group_by)[available].mean().reset_index()

    fig = px.bar(
        agg_df,
        x=group_by,
        y=available,
        title="Balance Sheet Composition by Industry",
        barmode="stack",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_cost_structure_breakdown(
    df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Create cost structure breakdown chart for a specific company."""
    import plotly.graph_objects as go

    cost_cols = ["cogs_pct", "sga_pct", "rnd_pct", "other_opex_pct"]
    available = [c for c in cost_cols if c in df.columns]

    company = df[df["ticker"] == ticker]
    if company.empty or not available:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {ticker}")
        return fig

    values = [company[c].iloc[0] for c in available]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=available,
                values=values,
                hole=0.4,
                title=f"{ticker} Cost Structure",
            )
        ]
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def create_unusual_items_heatmap(
    df: pd.DataFrame,
    max_companies: int = 50,
) -> go.Figure:
    """Create heatmap of unusual items flags across companies."""
    import plotly.express as px

    unusual_cols = [c for c in df.columns if "unusual" in c.lower() or "special" in c.lower()]

    if not unusual_cols:
        return px.imshow([[0]], title="No unusual items columns found")

    subset = df[["ticker"] + unusual_cols].head(max_companies)
    subset = subset.set_index("ticker")

    fig = px.imshow(
        subset.values,
        x=unusual_cols,
        y=subset.index,
        title="Unusual Items Detection Heatmap",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


# Import from sister module for the registry
try:
    from .technical import create_momentum_divergence_scatter
except ImportError:
    # Fallback if technical module is not yet ready or being restructured
    def create_momentum_divergence_scatter(*args, **kwargs):
        return go.Figure().add_annotation(text="Momentum Divergence Scatter Not Available")


# Dictionary mapping views to their primary visualization functions
VIEW_VISUALIZATION_REGISTRY = {
    "vw_features_valuation_ratios": [
        ("valuation_violin", create_valuation_violin_plot),
        ("valuation_range", create_valuation_range_visual),
    ],
    "vw_features_momentum": [
        ("momentum_scatter", create_momentum_divergence_scatter),
    ],
    "vw_features_analyst_sentiment": [
        ("sentiment_histogram", create_analyst_sentiment_histogram),
        ("upside_scatter", create_analyst_upside_scatter),
    ],
    # ... add mappings for other views
}


# ---------------------------------------------------------------------------
# Post-Enhancement 1–12 visualization functions
# ---------------------------------------------------------------------------


def create_volatility_surface_chart(
    df: pd.DataFrame,
    group_by: str = "sector",
) -> go.Figure:
    """Create volatility term-structure chart across 1m/3m/6m/1y horizons."""
    vol_cols = ["volatility_1m", "volatility_3m", "volatility_6m", "volatility_1y"]
    available = [c for c in vol_cols if c in df.columns]

    if not available or group_by not in df.columns:
        return _no_data("Volatility Surface — No Data")

    agg = df.groupby(group_by)[available].mean().reset_index()
    melted = agg.melt(
        id_vars=group_by, value_vars=available, var_name="horizon", value_name="volatility"
    )

    fig = px.line(
        melted,
        x="horizon",
        y="volatility",
        color=group_by,
        title="Volatility Term Structure by Sector",
        markers=True,
    )
    _apply_default_layout(fig, height=600, width=1000)
    return fig


def create_tax_rate_distribution(
    df: pd.DataFrame,
    group_by: str = "sector",
) -> go.Figure:
    """Create box plot of effective tax rates by sector."""
    col = resolve_column(df, "effective_tax_rate_ltm")
    if col is None or group_by not in df.columns:
        return _no_data("Tax Rate Distribution — No Data")

    fig = px.box(
        df,
        x=group_by,
        y=col,
        title="Effective Tax Rate Distribution by Sector",
        color=group_by,
    )
    _apply_default_layout(fig, height=600, width=1000)
    return fig


def create_fcf_estimate_curve(
    df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Create forward FCF estimate curve for a specific company."""
    fcf_cols = [
        "fcf_est_avg_fy1e",
        "fcf_est_avg_fy2e",
        "fcf_est_avg_fy3e",
        "fcf_est_avg_fy4e",
        "fcf_est_avg_fy5e",
    ]
    available = [c for c in fcf_cols if c in df.columns]

    company = df[df["ticker"] == ticker]
    if company.empty or not available:
        return _no_data(f"FCF Estimate Curve — No Data for {ticker}")

    values = [company[c].iloc[0] for c in available]
    labels = ["FY1E", "FY2E", "FY3E", "FY4E", "FY5E"][: len(available)]

    fig = go.Figure(
        data=[go.Scatter(x=labels, y=values, mode="lines+markers", name=ticker, line=dict(width=3))]
    )
    fig.update_layout(
        title=f"{ticker} — Forward FCF Estimate Curve",
        xaxis_title="Forecast Year",
        yaxis_title="FCF Estimate",
    )
    _apply_default_layout(fig, height=500, width=900)
    return fig


def create_opex_efficiency_scatter(
    df: pd.DataFrame,
) -> go.Figure:
    """Scatter plot of operating leverage score vs revenue growth."""
    x_col = resolve_column(df, "operating_leverage_score")
    y_col = resolve_column(df, "revenue_growth_yoy")
    if x_col is None or y_col is None:
        return _no_data("OpEx Efficiency — No Data")

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=resolve_column(df, "sector") or None,
        hover_data=["ticker"] if "ticker" in df.columns else None,
        title="Operating Leverage vs Revenue Growth",
    )
    _apply_default_layout(fig, height=600, width=1000)
    return fig


def create_asset_sale_impact_chart(
    df: pd.DataFrame,
    group_by: str = "sector",
) -> go.Figure:
    """Bar chart showing asset sale frequency and trend by sector."""
    freq_col = resolve_column(df, "asset_sale_frequency")
    if freq_col is None or group_by not in df.columns:
        return _no_data("Asset Sale Impact — No Data")

    agg = df.groupby(group_by)[freq_col].mean().reset_index()
    fig = px.bar(
        agg,
        x=group_by,
        y=freq_col,
        title="Average Asset Sale Frequency by Sector",
        color=group_by,
    )
    _apply_default_layout(fig, height=600, width=1000)
    return fig


def create_share_dilution_scatter(
    df: pd.DataFrame,
) -> go.Figure:
    """Scatter of shares YoY change vs buyback yield."""
    x_col = resolve_column(df, "shares_yoy_change_pct")
    y_col = resolve_column(df, "buyback_yield")
    if x_col is None or y_col is None:
        return _no_data("Share Dilution — No Data")

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=resolve_column(df, "sector") or None,
        hover_data=["ticker"] if "ticker" in df.columns else None,
        title="Share Dilution vs Buyback Yield",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    _apply_default_layout(fig, height=600, width=1000)
    return fig


def create_total_return_comparison(
    df: pd.DataFrame,
    top_n: int = 30,
) -> go.Figure:
    """Horizontal bar chart comparing total return metrics."""
    col = resolve_column(df, "total_return_ytd")
    if col is None or "ticker" not in df.columns:
        return _no_data("Total Return Comparison — No Data")

    sorted_df = df.nlargest(top_n, col)
    fig = px.bar(
        sorted_df,
        x=col,
        y="ticker",
        orientation="h",
        title=f"Top {top_n} — Total Return YTD",
        color=col,
        color_continuous_scale="RdYlGn",
    )
    _apply_default_layout(fig, height=max(400, top_n * 20), width=1000)
    return fig


def create_dividend_yield_history_chart(
    df: pd.DataFrame,
    group_by: str = "sector",
) -> go.Figure:
    """Line chart of historical dividend yield across 5 years by sector."""
    yield_cols = [
        "div_yield_ind",
        "div_yield_1fy_ind",
        "div_yield_2fyind",
        "div_yield_3fyind",
        "div_yield_4fyind",
        "div_yield_5fyind",
    ]
    available = [c for c in yield_cols if c in df.columns]

    if not available or group_by not in df.columns:
        return _no_data("Dividend Yield History — No Data")

    agg = df.groupby(group_by)[available].mean().reset_index()
    melted = agg.melt(
        id_vars=group_by, value_vars=available, var_name="period", value_name="div_yield"
    )

    fig = px.line(
        melted,
        x="period",
        y="div_yield",
        color=group_by,
        title="Historical Dividend Yield by Sector",
        markers=True,
    )
    _apply_default_layout(fig, height=600, width=1000)
    return fig


def create_interest_income_trend(
    df: pd.DataFrame,
    group_by: str = "sector",
) -> go.Figure:
    """Box plot of interest income to revenue ratio by sector."""
    col = resolve_column(df, "interest_income_to_revenue")
    if col is None or group_by not in df.columns:
        return _no_data("Interest Income Trend — No Data")

    fig = px.box(
        df,
        x=group_by,
        y=col,
        title="Interest Income to Revenue by Sector",
        color=group_by,
    )
    _apply_default_layout(fig, height=600, width=1000)
    return fig
