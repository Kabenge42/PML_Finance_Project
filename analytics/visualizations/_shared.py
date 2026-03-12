"""
Shared utilities for visualization modules.

Centralizes common constants, helper functions, and column resolution logic
to avoid duplication across visualization modules.
"""

from __future__ import annotations

import plotly.graph_objects as go

# Dark theme for Plotly (single source of truth)
PLOTLY_TEMPLATE = "plotly_dark"

# Shared color scheme
COLORS = [
    "#0A7EA4",
    "#00A878",
    "#6C63FF",
    "#FF6B6B",
    "#4ECDC4",
    "#FFD93D",
    "#95E1D3",
    "#F38181",
]

# Column alias map: logical_name → [mv_primary, mv_fallback_1, ...]
MV_COLUMN_ALIASES: dict[str, list[str]] = {
    "revenue_growth_yoy": ["revenue_yoy_growth", "revenue_growth_yoy"],
    "revenue_growth_3y_cagr": ["revenue_cagr_3y", "revenue_growth_3y_cagr"],
    "revenue_growth_5y_cagr": ["revenue_cagr_5y", "revenue_growth_5y_cagr"],
    "eps_growth_3y_cagr": ["eps_cagr_3y", "eps_growth_3y_cagr"],
    "net_income_growth": ["net_income_growth_yoy", "net_income_growth"],
    "inventory_turnover": [
        "inventory_turnover_itf",
        "inventory_turnover_mv",
        "inventory_turnover",
    ],
    "beneish_m_score": ["accounting_quality_score", "accruals_quality"],
    "eps_beat_count": ["eps_positive_years"],
    "eps_total_reports": ["eps_positive_streak"],
    "eps_growth_yoy": ["eps_yoy_growth", "eps_growth_yoy"],
    "operating_income_growth": ["operating_income_growth_yoy", "operating_income_growth"],
    "fcf_growth_yoy": ["fcf_growth_yoy", "fcf_yoy_growth"],
    "eps_qoq_growth": ["eps_qoq_growth"],
    "quality_composite_score": ["quality_momentum_score", "quality_composite_score"],
    "ebitda_growth_yoy": ["growth_ebitda_growth_yoy", "ebitda_growth_yoy"],
    "dividend_yield": ["valuation_dividend_yield", "dividend_yield", "dividend_yield_ltm"],
    "rnd_intensity": ["rnd_intensity", "rnd_intensity_ltm"],
    "accounting_quality_score": ["accounting_quality_score", "accounting_quality_score_comp"],
    "earnings_quality_composite": ["earnings_quality_composite", "earnings_quality_score"],
    "asset_turnover": ["asset_turnover", "total_asset_turnover"],
    "piotroski_f_score": ["piotroski_f_score", "f_score"],
}


def resolve_column(df, logical_name: str) -> str | None:
    """Resolve a logical column name to an actual DataFrame column.

    Checks if *logical_name* exists directly in *df*; if not, walks through
    ``MV_COLUMN_ALIASES`` to find the first available fallback.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check columns against.
    logical_name : str
        The logical / expected column name.

    Returns
    -------
    str | None
        The resolved column name present in *df*, or ``None``.
    """
    if logical_name in df.columns:
        return logical_name
    for alias in MV_COLUMN_ALIASES.get(logical_name, []):
        if alias in df.columns:
            return alias
    return None


def create_no_data_figure(title: str) -> go.Figure:
    """Create a placeholder figure when no data is available."""
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
    fig.update_layout(title=title, template=PLOTLY_TEMPLATE)
    return fig
