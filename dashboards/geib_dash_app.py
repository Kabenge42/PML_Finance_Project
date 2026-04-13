"""
Global Equity Investment Board (GEIB) Dashboard
Loads data from analytics.expected_returns_summary table (import from postgres.analytics.expected_returns_summary)
Run: python finance_ml/dashboards/geib_dash_app.py

Environment Variable Required: GEIB_DASHBOARD=true
"""

import os
from datetime import datetime
from pathlib import Path
import warnings
from requests.exceptions import RequestsDependencyWarning
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import dcc, html, Input, Output, State, dash_table
from dash.dash_table.Format import Format, Scheme, Symbol
from flask import send_from_directory

from analytics.data_utils import (
    get_analytics_engine,
    get_equities_schema,
    load_equities_data_from_db,
    load_all_feature_views,
    load_feature_categories_from_db,
    get_view_category_mapping,
    get_view_category_labels,
    get_view_feature_cols,
    validate_feature_alignment,
    validate_viz_column_coverage,
    safe_get_column,
    backfill_feature_columns,
    reorder_with_identifiers,
    load_identifier_columns,
)
from analytics.screening import (
    create_enhanced_screener,
    screen_earnings_quality,
    screen_value_opportunities,
    screen_growth_momentum,
    screen_garp_opportunities,
    screen_dividend_quality,
    screen_financial_health,
    screen_valuation_reversion_candidates,
    screen_integrity_filtered_growth,
    screen_high_yield_safe_dividends,
    create_sector_relative_ranking,
    screen_low_volatility_quality,
    screen_fcf_growth_compounders,
    screen_total_return_leaders,
)

# Import probabilistic visualizations
try:
    from analytics.visualizations import probability_viz
    from analytics.statistical_analysis import bayesian_category_analysis

    PROB_VIZ_AVAILABLE = True
except ImportError:
    PROB_VIZ_AVAILABLE = False
    print("⚠️ Probabilistic visualizations not available")

# Import expected_returns visualization functions for dynamic artifact generation
try:
    from analytics.visualizations.expected_returns_viz import (
        create_mc_return_distribution,
        create_sector_risk_reward_scatter,
        create_sector_heatmap,
        create_strong_consensus_bar,
        create_var_analysis,
        create_beat_vs_achievement_scatter,
        create_model_dispersion_dashboard,
        create_return_distribution_fit_chart,
        create_sector_return_analytics_heatmap,
        create_screening_summary_chart,
    )
    from expected_returns_v3 import (
        extract_strong_consensus,
        compute_sector_expected_returns,
        compute_sector_return_analytics,
    )
    from analytics.visualizations.quality_risk import (
        create_quality_risk_quadrant,
        create_distress_early_warning_dashboard,
        create_accounting_anomaly_dashboard,
        create_anomaly_severity_dashboard,
    )
    from analytics.visualizations.probability_viz import (
        create_bayesian_category_ridge,
        create_ruin_probability_diagnostic as create_ruin_prob_diagnostic_viz,
        create_anomaly_conditional_probability_chart,
    )
    from analytics.probability_analytics import (
        create_earnings_probability_dashboard,
        AccountingAnomalyProbabilityModel,
    )
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
    from analytics.visualizations.growth_analysis import (
        create_growth_waterfall_chart,
        create_growth_consistency_matrix,
        create_growth_vs_profitability_quadrant,
        create_growth_acceleration_chart,
        create_sustainable_growth_analysis,
    )

    ER_VIZ_AVAILABLE = True
except ImportError as _er_import_err:
    ER_VIZ_AVAILABLE = False
    print(f"⚠️ Expected returns visualization functions not available: {_er_import_err}")

from plotly.subplots import make_subplots
from typing import Tuple


def _safe_hover_data(
    candidates: list[str] | dict[str, str],
    df: pd.DataFrame,
) -> list[str] | dict[str, str] | None:
    """Filter *candidates* to columns actually present in *df*.

    Accepts either a list of column names or a dict mapping column names
    to format strings (the Plotly ``hover_data`` dict form).  Returns the
    same type with missing columns removed, or *None* when nothing remains
    so that ``px.scatter`` simply omits the parameter.
    """
    if isinstance(candidates, dict):
        filtered = {k: v for k, v in candidates.items() if k in df.columns}
        return filtered or None
    filtered = [c for c in candidates if c in df.columns]
    return filtered or None


warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
# Project root path for consistent path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent

# =============================================================================
# Standardized Color Scheme Configuration
# =============================================================================

COLORS = {
    "background_main": "#121212",  # Main app background
    "background_panel": "#1E1E1E",  # Panel/Card background
    "background_input": "#2D2D2D",  # Dropdowns and inputs
    "text_primary": "#FFFFFF",  # Primary text
    "text_secondary": "#B3B3B3",  # Subtitles and labels
    "border": "#333333",  # Subtle borders
    # Semantic Colors
    "primary": "#375A7F",  # Primary brand color (Blue)
    "success": "#63BE7B",  # Positive/High (Green)
    "warning": "#FFEB84",  # Neutral/Medium (Yellow)
    "danger": "#F8696B",  # Negative/Low (Red)
    "info": "#3498DB",  # Informational
}

# =============================================================================
# Table Formatting Configuration
# =============================================================================

TABLE_STYLE_HEADER = {
    "backgroundColor": COLORS["primary"],
    "color": COLORS["text_primary"],
    "fontWeight": "bold",
    "border": f"1px solid {COLORS['border']}",
    "textAlign": "center",
}

TABLE_STYLE_CELL = {
    "backgroundColor": COLORS["background_panel"],
    "color": COLORS["text_primary"],
    "border": f"1px solid {COLORS['border']}",
    "padding": "10px",
    "fontFamily": "Arial, sans-serif",
    "fontSize": "13px",
}

TABLE_STYLE_DATA_CONDITIONAL = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#252525"},
    # Conditional formatting for upside/return columns (0-100 scale)
    {
        "if": {
            "column_id": "expected_upside_pct",
            "filter_query": "{expected_upside_pct} > 15",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "expected_upside_pct",
            "filter_query": "{expected_upside_pct} < 0",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "expected_upside_kalman",
            "filter_query": "{expected_upside_kalman} > 15",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {"column_id": "expected_upside_kalman", "filter_query": "{expected_upside_kalman} < 0"},
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "implied_return_pt",
            "filter_query": "{implied_return_pt} > 10",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    # Conditional formatting for probability (0-1 scale)
    {
        "if": {
            "column_id": "achievement_probability",
            "filter_query": "{achievement_probability} > 0.7",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "achievement_probability",
            "filter_query": "{achievement_probability} >= 0.4 && {achievement_probability} <= 0.7",
        },
        "color": COLORS["warning"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "achievement_probability",
            "filter_query": "{achievement_probability} < 0.4",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Z-score conditional formatting (highlight extreme values)
    {
        "if": {
            "column_id": "expected_upside_pct_zscore",
            "filter_query": "{expected_upside_pct_zscore} > 1.5",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "expected_upside_pct_zscore",
            "filter_query": "{expected_upside_pct_zscore} < -1.5",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "expected_upside_kalman_zscore",
            "filter_query": "{expected_upside_kalman_zscore} > 1.5",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "expected_upside_kalman_zscore",
            "filter_query": "{expected_upside_kalman_zscore} < -1.5",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Percentile conditional formatting (top/bottom quartile)
    {
        "if": {
            "column_id": "expected_upside_pct_pctile",
            "filter_query": "{expected_upside_pct_pctile} > 75",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "expected_upside_pct_pctile",
            "filter_query": "{expected_upside_pct_pctile} < 25",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Beat Probability formatting (likely_beat vs. uncertain)
    {
        "if": {
            "column_id": "posterior_beat_prob",
            "filter_query": "{posterior_beat_prob} > 0.5",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "posterior_beat_prob",
            "filter_query": "{posterior_beat_prob} < 0.5",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Composite score
    {
        "if": {
            "column_id": "composite_score",
            "filter_query": "{composite_score} > 0.7",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "composite_score",
            "filter_query": "{composite_score} < 0.3",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Quality tier color coding
    {
        "if": {
            "column_id": "quality_tier",
            "filter_query": '{quality_tier} = "High"',
        },
        "color": "#FFD700",
        "fontWeight": "bold",
    },
    {
        "if": {"column_id": "quality_tier", "filter_query": '{quality_tier} = "Above Average"'},
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {"column_id": "quality_tier", "filter_query": '{quality_tier} = "Low"'},
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Accounting anomaly tier
    {
        "if": {
            "column_id": "accounting_anomaly_tier",
            "filter_query": '{accounting_anomaly_tier} = "Alert"',
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "accounting_anomaly_tier",
            "filter_query": '{accounting_anomaly_tier} = "Flag"',
        },
        "color": "#FF8C00",
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "accounting_anomaly_tier",
            "filter_query": '{accounting_anomaly_tier} = "Watch"',
        },
        "color": COLORS["warning"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "accounting_anomaly_tier",
            "filter_query": '{accounting_anomaly_tier} = "Clean"',
        },
        "color": COLORS["success"],
    },
    # Risk level (credit risk)
    {
        "if": {
            "column_id": "risk_level",
            "filter_query": '{risk_level} = "High"',
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "risk_level",
            "filter_query": '{risk_level} = "Distressed"',
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
        "backgroundColor": "#3d0000",
    },
    # Multi-flag alert
    {
        "if": {
            "column_id": "multi_flag_alert",
            "filter_query": "{multi_flag_alert} eq true",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Risk category (dividend)
    {
        "if": {
            "column_id": "risk_category",
            "filter_query": '{risk_category} = "At Risk"',
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Risk-reward ratio
    {
        "if": {
            "column_id": "risk_reward_ratio",
            "filter_query": "{risk_reward_ratio} > 2",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "risk_reward_ratio",
            "filter_query": "{risk_reward_ratio} < 0.5",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Altman Z-score (< 1.81 = distress zone)
    {
        "if": {
            "column_id": "altman_z_score",
            "filter_query": "{altman_z_score} < 1.81",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "altman_z_score",
            "filter_query": "{altman_z_score} > 2.99",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    # Resampled posterior: green for positive, red for negative
    {
        "if": {
            "column_id": "resampled_posterior_mean",
            "filter_query": "{resampled_posterior_mean} > 0",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "resampled_posterior_mean",
            "filter_query": "{resampled_posterior_mean} < 0",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Momentum signal
    {
        "if": {
            "column_id": "momentum_signal",
            "filter_query": "{momentum_signal} > 0.5",
        },
        "color": COLORS["success"],
    },
    {
        "if": {
            "column_id": "momentum_signal",
            "filter_query": "{momentum_signal} < -0.5",
        },
        "color": COLORS["danger"],
    },
    # EPS revision momentum
    {
        "if": {
            "column_id": "eps_revision_momentum",
            "filter_query": "{eps_revision_momentum} > 0.5",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "eps_revision_momentum",
            "filter_query": "{eps_revision_momentum} < -0.5",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Piotroski F-Score (0-9 integer scale)
    {
        "if": {
            "column_id": "piotroski_f_score",
            "filter_query": "{piotroski_f_score} >= 7",
        },
        "color": COLORS["success"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "piotroski_f_score",
            "filter_query": "{piotroski_f_score} <= 3",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    # Beneish M-Score (< -2.22 = unlikely manipulator)
    {
        "if": {
            "column_id": "beneish_m_score",
            "filter_query": "{beneish_m_score} > -2.22",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
    {
        "if": {
            "column_id": "beneish_m_score",
            "filter_query": "{beneish_m_score} < -2.22",
        },
        "color": COLORS["success"],
    },
    # Volatility regime score
    {
        "if": {
            "column_id": "volatility_regime_score",
            "filter_query": "{volatility_regime_score} > 0.7",
        },
        "color": COLORS["danger"],
        "fontWeight": "bold",
    },
]


# --- Schema-driven column formatting sets (populated at module load) ---
_SCHEMA_CURRENCY_COLS: set[str] = set()
_SCHEMA_PERCENTAGE_COLS: set[str] = set()
_SCHEMA_RATIO_COLS: set[str] = set()
_SCHEMA_COUNT_COLS: set[str] = set()
_SCHEMA_EARNINGS_COLS: set[str] = set()

# Roles from equities_schema_metadata that map to currency formatting
_CURRENCY_ROLES = {
    "price_col",
    "price_target_col",
    "market_data",
    "revenue_cols",
    "ebitda_cols",
    "ebit_cols",
    "net_income_cols",
    "operating_income_cols",
    "gross_profit_cols",
    "cash_flow",
    "balance_sheet",
    "operating_expenses_cols",
    "interest_cols",
    "sg&a_cols",
    "r&d_expenses_cols",
    "marketing_exp_cols",
    "tax_cols",
    "income_statement_cols",
}


def _populate_schema_format_sets() -> None:
    """Populate module-level formatting sets from ``get_equities_schema()``."""
    global _SCHEMA_CURRENCY_COLS, _SCHEMA_PERCENTAGE_COLS, _SCHEMA_RATIO_COLS
    global _SCHEMA_COUNT_COLS, _SCHEMA_EARNINGS_COLS
    schema = get_equities_schema()
    if not schema:
        return
    for alias, meta in schema.items():
        role = meta.get("role", "")
        if role in _CURRENCY_ROLES:
            _SCHEMA_CURRENCY_COLS.add(alias)
        elif role == "percentage":
            _SCHEMA_PERCENTAGE_COLS.add(alias)
        elif role == "ratio":
            _SCHEMA_RATIO_COLS.add(alias)
        elif role == "count":
            _SCHEMA_COUNT_COLS.add(alias)
        elif role == "earnings_cols":
            _SCHEMA_EARNINGS_COLS.add(alias)


# Hardcoded fallback sets used when the DB schema is unavailable
_FALLBACK_CURRENCY_COLS = {
    "last_price",
    "price_target",
    "price_target_high",
    "price_target_low",
    "price_target_median",
    "price_target_prob_weighted",
    "price_target_mc",
    "kalman_estimate",
    "market_cap",
    "enterprise_value",
    "ci_lower",
    "ci_upper",
}
_FALLBACK_PCT_100_COLS = {
    "expected_upside_pct",
    "expected_upside_kalman",
    "implied_return_pt",
    "var_5_pct",
    "accounting_anomaly_score",
    "anomaly_severity_score",
    "sector_anomaly_percentile",
    "payout_ratio",
    "yield_vs_5y_avg",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "roe",
    "retention_ratio",
}
_FALLBACK_PROB_COLS = {
    "achievement_probability",
    "posterior_beat_prob",
    "confidence_score",
    "prob_positive_upside",
    "weighted_agreement",
    "prob_beat_given_momentum",
    "analyst_conviction",
    "eps_revision_momentum",
    "analyst_rating_normalized",
    "resampled_posterior_mean",
    "technical_adjustment",
    "momentum_signal",
    "volatility_regime_score",
    "distress_probability",
    "ruin_probability",
    "survival_probability",
    "dividend_cut_probability",
    "dividend_consistency",
    "data_quality_score",
    "beta_stability_score",
    "safety_score",
    "fcf_dividend_coverage",
    "anomaly_conditional_probability",
}
_FALLBACK_PCTILE_COLS = {
    "expected_upside_pct_pctile",
    "expected_upside_kalman_pctile",
    "implied_return_pt_pctile",
    "anomaly_risk_rank",
}
_FALLBACK_ZSCORE_COLS = {
    "expected_upside_pct_zscore",
    "expected_upside_kalman_zscore",
    "implied_return_pt_zscore",
    "sector_relative_anomaly",
    "altman_z_score",
    "altman_z_trend",
}
_FALLBACK_TEXT_COLS = {"credible_interval_90", "credible_interval_95"}
_FALLBACK_INT_COLS = {
    "agreement_score",
    "anomaly_feature_count",
    "dividend_streak",
    "high_yield_flag",
    "sustainable_flag",
    "piotroski_f_score",
}
_FALLBACK_RATIO_COLS = {"p_e_ratio", "p_b_ratio", "ev_ebitda", "beneish_m_score"}
_FALLBACK_EPS_COLS = {"eps_actual", "eps_estimate"}
_FALLBACK_SCORE_COLS = {
    "composite_score",
    "kelly_pct",
    "risk_reward_ratio",
    "liquidity_stress_score",
    "cash_runway_months",
    "wealth_buffer",
}


def get_formatted_columns(cols_list):
    """Return column definitions with formatting rules driven by schema metadata.

    Uses ``get_equities_schema()`` roles when available, falling back to
    hardcoded sets so the dashboard works without a database connection.
    """
    # Ensure schema sets are populated (no-op after first call)
    if not _SCHEMA_CURRENCY_COLS:
        _populate_schema_format_sets()

    currency = _SCHEMA_CURRENCY_COLS or _FALLBACK_CURRENCY_COLS
    pct_100 = _FALLBACK_PCT_100_COLS | _SCHEMA_PERCENTAGE_COLS
    prob_01 = _FALLBACK_PROB_COLS
    pctile = _FALLBACK_PCTILE_COLS
    zscore = _FALLBACK_ZSCORE_COLS
    text_cols = _FALLBACK_TEXT_COLS
    int_cols = _FALLBACK_INT_COLS | _SCHEMA_COUNT_COLS
    ratio_cols = _FALLBACK_RATIO_COLS | _SCHEMA_RATIO_COLS
    eps_cols = _FALLBACK_EPS_COLS | _SCHEMA_EARNINGS_COLS
    score_cols = _FALLBACK_SCORE_COLS

    formatted = []
    for col in cols_list:
        spec = {"id": col, "name": col.replace("_", " ").capitalize()}

        if col in currency:
            spec.update(
                {
                    "type": "numeric",
                    "format": Format(precision=2, scheme=Scheme.fixed, group=True)
                    .symbol(Symbol.yes)
                    .symbol_prefix("$"),
                }
            )
        elif col in pct_100:
            spec.update(
                {
                    "type": "numeric",
                    "format": Format(precision=2, scheme=Scheme.fixed)
                    .symbol(Symbol.yes)
                    .symbol_suffix("%"),
                }
            )
        elif col in prob_01:
            spec.update(
                {
                    "type": "numeric",
                    "format": Format(precision=2, scheme=Scheme.percentage),
                }
            )
        elif col in pctile:
            spec.update(
                {
                    "type": "numeric",
                    "format": Format(precision=1, scheme=Scheme.fixed)
                    .symbol(Symbol.yes)
                    .symbol_suffix("th"),
                }
            )
        elif col in zscore:
            spec.update(
                {
                    "type": "numeric",
                    "format": Format(precision=2, scheme=Scheme.fixed)
                    .symbol(Symbol.yes)
                    .symbol_suffix("σ"),
                }
            )
        elif col in text_cols:
            spec.update({"type": "text"})
        elif col in int_cols:
            spec.update({"type": "numeric", "format": Format(precision=0, scheme=Scheme.fixed)})
        elif col in ratio_cols:
            spec.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        elif col in eps_cols:
            spec.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        elif col in score_cols:
            spec.update({"type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)})

        formatted.append(spec)
    return formatted


# =============================================================================
# Filter Configuration (Single Source of Truth)
# =============================================================================

FILTER_CONFIG = [
    {"label": "Region", "id": "region-dropdown", "column": "region", "width": "23%"},
    {"label": "Country", "id": "country-dropdown", "column": "country", "width": "23%"},
    {
        "label": "Exchange",
        "id": "exchange-dropdown",
        "column": "exchange",
        "width": "23%",
    },
    {"label": "Sector", "id": "sector-dropdown", "column": "sector", "width": "23%"},
    {
        "label": "Industry",
        "id": "industry-dropdown",
        "column": "industry",
        "width": "23%",
    },
    {"label": "Signal", "id": "signal-dropdown", "column": "signal", "width": "23%"},
    {
        "label": "Trading Country",
        "id": "trading-country-dropdown",
        "column": "trading_country",
        "width": "18%",
    },
    {
        "label": "Style Class",
        "id": "style-class-dropdown",
        "column": "style_class",
        "width": "18%",
    },
    {
        "label": "Size Class",
        "id": "size-class-dropdown",
        "column": "size_class",
        "width": "18%",
    },
    {"label": "Unit", "id": "unit-dropdown", "column": "unit", "width": "18%"},
    {
        "label": "Beat Classification",
        "id": "beat-classification-dropdown",
        "column": "beat_classification",
        "width": "18%",
    },
    {
        "label": "Confidence Level",
        "id": "confidence-dropdown",
        "column": "confidence_level",
        "width": "18%",
    },
    {
        "label": "Quality Tier",
        "id": "quality-tier-dropdown",
        "column": "quality_tier",
        "width": "18%",
    },
    {
        "label": "Dividend Frequency",
        "id": "div-freq-dropdown",
        "column": "dividend_record_frequency",
        "width": "18%",
    },
    {
        "label": "Earnings Frequency",
        "id": "earn-freq-dropdown",
        "column": "earnings_report_frequency",
        "width": "18%",
    },
    {
        "label": "Next Earnings Status",
        "id": "next-earn-status-dropdown",
        "column": "next_earnings_status",
        "width": "18%",
    },
    {
        "label": "Next Earnings When",
        "id": "next-earn-when-dropdown",
        "column": "next_earnings_when",
        "width": "18%",
    },
    {
        "label": "Anomaly Tier",
        "id": "anomaly-tier-dropdown",
        "column": "accounting_anomaly_tier",
        "width": "18%",
    },
    {
        "label": "Risk Level",
        "id": "risk-level-dropdown",
        "column": "risk_level",
        "width": "18%",
    },
    {
        "label": "Risk Category",
        "id": "risk-category-dropdown",
        "column": "risk_category",
        "width": "18%",
    },
    {
        "label": "FY End",
        "id": "fy-end-dropdown",
        "column": "fy_end",
        "width": "18%",
    },
    {
        "label": "Next Fiscal Quarter",
        "id": "next-fq-dropdown",
        "column": "next_fiscal_quarter",
        "width": "18%",
    },
]

# Numeric range slider configuration for quantitative filtering
RANGE_SLIDER_CONFIG = [
    {
        "label": "Market Cap ($)",
        "id": "market-cap-slider",
        "column": "market_cap",
        "log": True,
    },
    {
        "label": "Expected Upside (%)",
        "id": "expected-upside-slider",
        "column": "expected_upside_pct",
        "log": False,
    },
    {
        "label": "Confidence Score",
        "id": "confidence-score-slider",
        "column": "confidence_score",
        "log": False,
    },
    {
        "label": "Composite Score",
        "id": "composite-score-slider",
        "column": "composite_score",
        "log": False,
    },
    {
        "label": "Altman Z-Score",
        "id": "altman-zscore-slider",
        "column": "altman_z_score",
        "log": False,
    },
]

ALL_RANGE_SLIDER_IDS = [s["id"] for s in RANGE_SLIDER_CONFIG]

# All filter dropdown IDs (for callbacks)
ALL_FILTER_IDS = [f["id"] for f in FILTER_CONFIG]
ALL_FILTER_COLUMNS = [f["column"] for f in FILTER_CONFIG]

# Column tooltips for technical columns
COLUMN_TOOLTIPS = {
    "resampled_posterior_mean": "Bayesian resampled posterior expected return",
    "technical_adjustment": "Technical signal adjustment factor applied to base return",
    "momentum_signal": "Composite momentum signal (-1 to +1)",
    "volatility_regime_score": "Volatility regime indicator (higher = more volatile)",
    "credible_interval_90": "90% Bayesian credible interval [low, high]",
    "credible_interval_95": "95% Bayesian credible interval [low, high]",
    "prob_beat_given_momentum": "Conditional P(earnings beat | momentum signal)",
    "eps_revision_momentum": "EPS revision momentum score",
    "analyst_rating_normalized": "Normalized analyst consensus rating (0-1)",
    "var_5_pct": "Value at Risk at 5% confidence level",
    "altman_z_score": "Altman Z-Score: <1.81 distress, 1.81-2.99 grey zone, >2.99 safe",
    "piotroski_f_score": "Piotroski F-Score (0-9): higher = stronger fundamentals",
    "beneish_m_score": "Beneish M-Score: > -2.22 suggests earnings manipulation",
    "expected_upside_pct_zscore": "Z-score of expected upside within the universe",
    "expected_upside_kalman_zscore": "Z-score of filtered upside within the universe",
    "expected_upside_pct_pctile": "Percentile rank of expected upside",
    "expected_upside_kalman_pctile": "Percentile rank of filtered upside",
    "composite_score": "Weighted composite quality score (0-1)",
    "confidence_score": "Model confidence score (0-1)",
    "risk_reward_ratio": "Expected return / risk ratio",
    "accounting_anomaly_score": "Accounting anomaly detection score (higher = more anomalous)",
    "sector_relative_anomaly": "Sector-relative anomaly Z-score",
}


def build_filter_options(dataframe: pd.DataFrame, column: str) -> list:
    """Build sorted dropdown options from a DataFrame column."""
    if column in dataframe.columns and len(dataframe) > 0:
        return [{"label": v, "value": v} for v in sorted(dataframe[column].dropna().unique())]
    return []


def build_filter_panel(dataframe: pd.DataFrame) -> html.Div:
    """Build the entire filter panel from FILTER_CONFIG and RANGE_SLIDER_CONFIG."""
    dropdowns = []
    for f in FILTER_CONFIG:
        dropdowns.append(
            html.Div(
                [
                    html.Label(
                        f["label"],
                        className="filter-label",
                        style={"color": COLORS["text_secondary"]},
                    ),
                    dcc.Dropdown(
                        id=f["id"],
                        multi=True,
                        options=build_filter_options(dataframe, f["column"]),
                        style={
                            "backgroundColor": COLORS["background_input"],
                            "color": "black",  # Note: Dropdown text usually requires dark color unless fully overridden by CSS
                        },
                        className="custom-dropdown",
                    ),
                ],
                className="filter-item",
                style={"width": f["width"], "display": "inline-block", "margin": "5px"},
            )
        )

    # Build numeric range sliders
    sliders = []
    for s in RANGE_SLIDER_CONFIG:
        col = s["column"]
        if col in dataframe.columns and len(dataframe) > 0:
            vals = pd.to_numeric(dataframe[col], errors="coerce").dropna()
            if len(vals) > 0:
                col_min = float(vals.min())
                col_max = float(vals.max())
                # Avoid degenerate range
                if col_min == col_max:
                    col_max = col_min + 1
                step = round((col_max - col_min) / 100, 4) or 0.01
                sliders.append(
                    html.Div(
                        [
                            html.Label(
                                s["label"],
                                className="filter-label",
                                style={"color": COLORS["text_secondary"]},
                            ),
                            dcc.RangeSlider(
                                id=s["id"],
                                min=col_min,
                                max=col_max,
                                step=step,
                                value=[col_min, col_max],
                                marks={
                                    col_min: {"label": f"{col_min:.1f}"},
                                    col_max: {"label": f"{col_max:.1f}"},
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                            ),
                        ],
                        style={
                            "width": "18%",
                            "display": "inline-block",
                            "margin": "5px",
                            "verticalAlign": "top",
                        },
                    )
                )
            else:
                # Column exists but no valid numeric data
                sliders.append(
                    html.Div(
                        [
                            html.Label(
                                s["label"],
                                className="filter-label",
                                style={"color": COLORS["text_secondary"]},
                            ),
                            dcc.RangeSlider(id=s["id"], min=0, max=1, value=[0, 1], disabled=True),
                        ],
                        style={"width": "18%", "display": "inline-block", "margin": "5px"},
                    )
                )
        else:
            # Column not present — render disabled placeholder
            sliders.append(
                html.Div(
                    [
                        html.Label(
                            s["label"],
                            className="filter-label",
                            style={"color": COLORS["text_secondary"]},
                        ),
                        dcc.RangeSlider(id=s["id"], min=0, max=1, value=[0, 1], disabled=True),
                    ],
                    style={"width": "18%", "display": "inline-block", "margin": "5px"},
                )
            )

    return html.Div(
        [
            html.Div(
                [
                    html.H4(
                        "Filters",
                        style={
                            "marginBottom": "10px",
                            "color": COLORS["text_primary"],
                            "display": "inline-block",
                        },
                    ),
                    html.Button(
                        "Reset Filters",
                        id="reset-filters-btn",
                        n_clicks=0,
                        style={
                            "marginLeft": "20px",
                            "backgroundColor": COLORS["danger"],
                            "color": COLORS["text_primary"],
                            "border": "none",
                            "padding": "5px 15px",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "fontWeight": "bold",
                        },
                    ),
                ]
            ),
            html.Div(
                dropdowns,
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "justifyContent": "space-around",
                },
            ),
            # Numeric range sliders section
            (
                html.Div(
                    [
                        html.H5(
                            "Numeric Range Filters",
                            style={
                                "marginTop": "15px",
                                "marginBottom": "10px",
                                "color": COLORS["text_secondary"],
                            },
                        ),
                        html.Div(
                            sliders,
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "justifyContent": "space-around",
                            },
                        ),
                    ]
                )
                if sliders
                else html.Div()
            ),
        ],
        style={
            "padding": "20px",
            "backgroundColor": COLORS["background_panel"],
            "margin": "10px 0",
            "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}",
        },
    )


def apply_global_filters(
    dataframe: pd.DataFrame,
    filter_values: dict,
    range_values: dict | None = None,
) -> pd.DataFrame:
    """
    Apply all global filters (dropdown + range slider) to a DataFrame consistently.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to filter.
    filter_values : dict
        Mapping of column name -> selected values (list or None).
    range_values : dict or None
        Mapping of column name -> [min, max] range (from RangeSliders).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    filtered = dataframe.copy()
    for column, values in filter_values.items():
        if values and column in filtered.columns:
            filtered = filtered[filtered[column].isin(values)]
    # Apply numeric range filters
    if range_values:
        for column, bounds in range_values.items():
            if bounds and column in filtered.columns and len(bounds) == 2:
                col_vals = pd.to_numeric(filtered[column], errors="coerce")
                filtered = filtered[
                    (col_vals >= bounds[0]) & (col_vals <= bounds[1]) | col_vals.isna()
                ]
    return filtered


def collect_filter_values(*args) -> dict:
    """
    Zip filter arguments (in FILTER_CONFIG order) into a {column: values} dict.

    Usage inside a callback that receives all filter inputs first:
        filter_values = collect_filter_values(*args[:len(FILTER_CONFIG)])
    """
    return {cfg["column"]: val for cfg, val in zip(FILTER_CONFIG, args)}


def collect_range_slider_values(*args) -> dict:
    """
    Zip range slider arguments (in RANGE_SLIDER_CONFIG order) into a
    {column: [min, max]} dict.
    """
    return {cfg["column"]: val for cfg, val in zip(RANGE_SLIDER_CONFIG, args)}


# =============================================================================
# Reusable KPI Card Builder
# =============================================================================


def build_kpi_card(title: str, value: str, color: str = "info") -> html.Div:
    """Build a styled KPI card for the dashboard header row."""
    bg = COLORS.get(color, COLORS["info"])
    return html.Div(
        [
            html.H6(
                title,
                style={"margin": "0", "color": COLORS["text_secondary"], "fontSize": "0.8rem"},
            ),
            html.H4(value, style={"margin": "5px 0 0 0", "color": bg, "fontWeight": "bold"}),
        ],
        style={
            "backgroundColor": COLORS["background_panel"],
            "border": f"1px solid {COLORS['border']}",
            "borderLeft": f"4px solid {bg}",
            "borderRadius": "6px",
            "padding": "12px 18px",
            "minWidth": "160px",
            "textAlign": "center",
        },
    )


# =============================================================================
# Screening Strategy Registry (all 13 screens)
# =============================================================================

ALL_SCREENING_STRATEGIES = [
    ("quality", "Enhanced Quality", create_enhanced_screener),
    ("earnings_quality", "Earnings Quality", screen_earnings_quality),
    ("value", "Value Opportunities", screen_value_opportunities),
    ("growth", "Growth Momentum", screen_growth_momentum),
    ("garp", "GARP", screen_garp_opportunities),
    ("dividend", "Dividend Quality", screen_dividend_quality),
    ("healthy", "Financial Health", screen_financial_health),
    ("valuation_reversion", "Valuation Reversion", screen_valuation_reversion_candidates),
    ("integrity_growth", "Integrity Growth", screen_integrity_filtered_growth),
    ("high_yield_safe", "High Yield Safe", screen_high_yield_safe_dividends),
    (
        "sector_relative",
        "Sector Relative",
        lambda df: create_sector_relative_ranking(df, metric="expected_upside_pct"),
    ),
    ("low_vol_quality", "Low Volatility Quality", screen_low_volatility_quality),
    ("fcf_compounders", "FCF Compounders", screen_fcf_growth_compounders),
    ("total_return_leaders", "Total Return Leaders", screen_total_return_leaders),
]


# =============================================================================
# Kelly Criterion Position Sizing
# =============================================================================

# Kelly-specific dropdown option definitions
KELLY_FRACTION_OPTIONS = [
    {"label": "Full Kelly (1.0)", "value": 1.0},
    {"label": "Half Kelly (0.5)", "value": 0.5},
    {"label": "Quarter Kelly (0.25)", "value": 0.25},
    {"label": "Eighth Kelly (0.125)", "value": 0.125},
]

MAX_POSITION_OPTIONS = [
    {"label": "5%", "value": 0.05},
    {"label": "10%", "value": 0.10},
    {"label": "15%", "value": 0.15},
    {"label": "20%", "value": 0.20},
    {"label": "No cap", "value": "no_cap"},
]

KELLY_ADJUSTMENT_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Confidence-weighted", "value": "confidence"},
    {"label": "Achievement-weighted", "value": "achievement"},
    {"label": "Both", "value": "both"},
]

KELLY_MIN_CONFIDENCE_OPTIONS = [
    {"label": "0.15", "value": 0.15},
    {"label": "0.25", "value": 0.25},
    {"label": "0.35", "value": 0.35},
    {"label": "0.45", "value": 0.45},
]

KELLY_BAR_COLOR_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Sector", "value": "sector"},
    {"label": "Industry", "value": "industry"},
    {"label": "Currency", "value": "unit"},
    {"label": "Exchange", "value": "exchange"},
    {"label": "Confidence Level", "value": "confidence_level"},
    {"label": "Quality Tier", "value": "quality_tier"},
    {"label": "Signal", "value": "signal"},
    {"label": "Country", "value": "country"},
]

KELLY_SCATTER_COLOR_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Confidence Level", "value": "confidence_level"},
    {"label": "Quality Tier", "value": "quality_tier"},
    {"label": "Signal", "value": "signal"},
    {"label": "Country", "value": "country"},
]

KELLY_SCATTER_SIZE_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Achievement Probability", "value": "achievement_probability"},
    {"label": "Composite Score", "value": "composite_score"},
    {"label": "Market Cap", "value": "market_cap"},
    {"label": "Upside Percentile", "value": "expected_upside_pct_pctile"},
]


def calculate_kelly_metrics(
    dataframe: pd.DataFrame,
    kelly_fraction: float = 0.25,
    max_position: str | float = 0.10,
    adjustment_method: str = "both",
) -> pd.DataFrame:
    """
    Calculate Kelly Criterion metrics for each position.

    Uses columns from analytics.expected_returns_summary:
      - prob_positive_upside  (0-100 scale)
      - expected_upside_kalman       (percentage)
      - confidence_score      (0-1 scale)
      - achievement_probability (0-1 scale)

    Returns the DataFrame with added columns:
      kelly_raw, kelly_fractional, kelly_adjusted, kelly_pct
    """
    result = dataframe.copy()

    for col in [
        "prob_positive_upside",
        "expected_upside_kalman",
        "confidence_score",
        "achievement_probability",
    ]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    # Kelly formula: f* = (p*b - q) / b  where p=win prob, q=1-p, b=win/loss ratio
    p = result["prob_positive_upside"] / 100.0
    q = 1.0 - p
    b = result["expected_upside_kalman"] / 100.0

    result["kelly_raw"] = np.where(b != 0, (p * b - q) / b, 0)
    result["kelly_raw"] = result["kelly_raw"].clip(lower=0)

    # Apply fractional Kelly
    result["kelly_fractional"] = result["kelly_raw"] * kelly_fraction

    # Apply adjustment method
    if adjustment_method == "confidence":
        result["kelly_adjusted"] = result["kelly_fractional"] * result["confidence_score"]
    elif adjustment_method == "achievement":
        result["kelly_adjusted"] = result["kelly_fractional"] * result["achievement_probability"]
    elif adjustment_method == "both":
        result["kelly_adjusted"] = (
            result["kelly_fractional"]
            * result["confidence_score"]
            * result["achievement_probability"]
        )
    else:
        result["kelly_adjusted"] = result["kelly_fractional"]

    # Apply max position cap
    if max_position != "no_cap":
        result["kelly_adjusted"] = result["kelly_adjusted"].clip(upper=float(max_position))

    # Normalize to portfolio percentage
    total_kelly = result["kelly_adjusted"].sum()
    result["kelly_pct"] = (result["kelly_adjusted"] / total_kelly * 100.0) if total_kelly > 0 else 0

    return result


# =============================================================================
# Efficient Frontier Helper Functions
# =============================================================================


def _ef_estimate_covariance_matrix(dataframe: pd.DataFrame, selected_tickers: list) -> pd.DataFrame:
    """Estimate covariance matrix from sector/industry correlations for the efficient frontier."""
    n = len(selected_tickers)
    cov_matrix = np.eye(n) * 0.04

    sector_map = {}
    for ticker in selected_tickers:
        ticker_data = dataframe[dataframe["ticker"] == ticker]
        if len(ticker_data) > 0:
            sector_map[ticker] = ticker_data["sector"].iloc[0]

    for i in range(n):
        for j in range(i + 1, n):
            sector_i = sector_map.get(selected_tickers[i], "Unknown")
            sector_j = sector_map.get(selected_tickers[j], "Unknown")
            correlation = 0.6 if sector_i == sector_j else 0.3
            volatility_i = 0.25
            volatility_j = 0.25
            cov_matrix[i, j] = correlation * volatility_i * volatility_j
            cov_matrix[j, i] = cov_matrix[i, j]

    return pd.DataFrame(cov_matrix, index=selected_tickers, columns=selected_tickers)


def _ef_generate_random_portfolios(
    expected_returns: np.ndarray,
    cov_matrix: pd.DataFrame,
    num_portfolios: int,
    risk_free_rate: float,
    constraint_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate random portfolio combinations for the efficient frontier."""
    n_assets = len(expected_returns)
    portfolio_returns = np.zeros(num_portfolios)
    portfolio_volatilities = np.zeros(num_portfolios)
    portfolio_sharpe_ratios = np.zeros(num_portfolios)
    portfolio_weights = np.zeros((num_portfolios, n_assets))

    np.random.seed(42)

    for i in range(num_portfolios):
        if constraint_type == "long_short":
            weights = np.random.normal(0, 0.3, n_assets)
            weights = weights / np.sum(np.abs(weights))
        else:  # long_only or sector_neutral
            weights = np.random.dirichlet(np.ones(n_assets))

        port_return = np.sum(weights * expected_returns)
        port_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        port_volatility = np.sqrt(port_variance)
        sharpe = (
            (port_return - risk_free_rate / 100) / port_volatility if port_volatility > 0 else 0
        )

        portfolio_returns[i] = port_return
        portfolio_volatilities[i] = port_volatility
        portfolio_sharpe_ratios[i] = sharpe
        portfolio_weights[i] = weights

    return (
        portfolio_returns,
        portfolio_volatilities,
        portfolio_sharpe_ratios,
        portfolio_weights,
    )


def _ef_find_optimal_portfolios(
    expected_returns: np.ndarray,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find maximum Sharpe ratio and minimum variance portfolios."""
    n_assets = len(expected_returns)

    min_var_weights = np.ones(n_assets) / n_assets
    min_var_return = np.sum(min_var_weights * expected_returns)
    min_var_volatility = np.sqrt(
        np.dot(min_var_weights, np.dot(cov_matrix.values, min_var_weights))
    )

    max_sharpe_weights = np.ones(n_assets) / n_assets
    max_sharpe_return = np.sum(max_sharpe_weights * expected_returns)
    max_sharpe_volatility = np.sqrt(
        np.dot(max_sharpe_weights, np.dot(cov_matrix.values, max_sharpe_weights))
    )

    return (
        min_var_weights,
        max_sharpe_weights,
        np.array([min_var_volatility, max_sharpe_volatility]),
        np.array([min_var_return, max_sharpe_return]),
    )


# =============================================================================
# Artifact Helper Functions
# =============================================================================


def load_plotly_figure_from_html(html_path):
    """Extract Plotly figure (data + layout) from a self-contained Plotly HTML file."""
    import json as _json

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        idx = content.rfind("Plotly.newPlot")
        if idx < 0:
            return None
        call_text = content[idx:]
        # Extract data array
        start = call_text.index("[")
        depth, end = 0, start
        for i, c in enumerate(call_text[start:], start):
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        data = _json.loads(call_text[start : end + 1])
        # Extract layout object
        rest = call_text[end + 1 :].lstrip(" ,\n\r\t")
        l_start = rest.index("{")
        depth, l_end = 0, l_start
        for i, c in enumerate(rest[l_start:], l_start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    l_end = i
                    break
        layout = _json.loads(rest[l_start : l_end + 1])
        return go.Figure(data=data, layout=layout)
    except Exception as e:
        print(f"⚠️ Failed to load Plotly figure from {html_path}: {e}")
        return None


def get_artifact_path(artifact_name: str, artifact_type: str = "html") -> Path:
    """
    Get the path to an artifact file in the outputs directory.

    Parameters
    ----------
    artifact_name : str
        Name of the artifact file (without extension).
    artifact_type : str, default "html"
        File extension/type of the artifact.

    Returns
    -------
    Path
        Full path to the artifact file.
    """
    artifacts_dir = PROJECT_ROOT / "outputs" / "analytics"
    return artifacts_dir / f"{artifact_name}.{artifact_type}"


def render_artifact_or_placeholder(
    artifact_name: str, title: str = "Artifact", artifact_type: str = "html"
) -> html.Div:
    """
    Render an artifact if it exists, otherwise return a placeholder.

    Parameters
    ----------
    artifact_name : str
        Name of the artifact file (without extension).
    title : str, default "Artifact"
        Title to display above the artifact or in the placeholder.
    artifact_type : str, default "html"
        File extension/type of the artifact.

    Returns
    -------
    html.Div
        Dash HTML component containing either the artifact iframe or a placeholder.
    """
    artifact_path = get_artifact_path(artifact_name, artifact_type)

    if artifact_path.exists():
        # Return an iframe to display the HTML artifact
        relative_path = f"/artifacts/{artifact_name}.{artifact_type}"
        return html.Div(
            [
                html.H4(title, style={"textAlign": "center", "marginBottom": "10px"}),
                html.Iframe(
                    src=relative_path,
                    style={
                        "width": "100%",
                        "height": "600px",
                        "border": "1px solid #444",
                        "borderRadius": "5px",
                    },
                ),
            ]
        )
    else:
        # Return a placeholder indicating the artifact is not available
        return html.Div(
            [
                html.H4(title, style={"textAlign": "center", "marginBottom": "10px"}),
                dbc.Alert(
                    [
                        html.I(
                            className="fas fa-info-circle",
                            style={"marginRight": "10px"},
                        ),
                        f"Artifact '{artifact_name}.{artifact_type}' not found. ",
                        "Run the analytics pipeline to generate this artifact.",
                    ],
                    color="warning",
                    style={"textAlign": "center"},
                ),
            ],
            style={"padding": "20px"},
        )


# --- Viz column requirements (used by load_geib_data to enrich summary) ---
VIZ_REQUIRED_COLUMNS = {
    "create_mc_return_distribution": ["expected_upside_pct", "prob_positive_upside"],
    "create_sector_risk_reward_scatter": [
        "industry",
        "expected_upside_pct",
        "confidence_score",
    ],
    "create_quality_risk_quadrant": ["altman_z_score", "piotroski_f_score"],
    "create_sector_heatmap": ["industry", "expected_upside_pct"],
    "create_strong_consensus_bar": ["ticker", "expected_upside_pct"],
    "create_var_analysis": ["expected_upside_pct", "var_5_pct"],
    "create_beat_vs_achievement_scatter": [
        "achievement_probability",
        "posterior_beat_prob",
    ],
    "create_model_dispersion_dashboard": [
        "expected_upside_pct",
        "expected_upside_kalman",
        "implied_return_pt",
    ],
    "create_accounting_anomaly_dashboard": [
        "accounting_anomaly_score",
        "accounting_anomaly_tier",
        "anomaly_conditional_probability",
        "anomaly_feature_count",
        "multi_flag_alert",
    ],
    "create_ruin_probability_diagnostic": [
        "ruin_probability",
        "survival_probability",
        "wealth_buffer",
    ],
    "create_distress_early_warning_dashboard": [
        "altman_z_score",
        "distress_probability",
        "risk_level",
    ],
    "dividend_safety_analysis": [
        "dividend_cut_probability",
        "safety_score",
        "risk_category",
        "fcf_dividend_coverage",
        "payout_ratio",
    ],
    "create_valuation_multiples_comparison": ["p_e_ratio", "p_b_ratio", "ev_ebitda"],
    "create_valuation_vs_growth_quadrant": ["p_e_ratio", "revenue_growth_yoy"],
    "create_earnings_surprise_dashboard": ["eps_actual", "eps_estimate"],
    "create_eps_trajectory_analysis": ["eps_actual"],
    "create_beat_rate_heatmap": ["earnings_beat", "industry"],
    "create_growth_waterfall_chart": ["revenue_growth_yoy", "eps_growth_yoy"],
    "create_sustainable_growth_analysis": ["roe", "retention_ratio"],
}


def load_geib_data():
    """Load all necessary data for the GEIB dashboard using data_utils.

    Returns:
        dict: Dictionary containing DataFrames for different components
    """
    data = {
        "summary": pd.DataFrame(),
        "tri_model": pd.DataFrame(),
        "earnings": pd.DataFrame(),
        "credit": pd.DataFrame(),
        "anomaly": pd.DataFrame(),
        "dividend_safety": pd.DataFrame(),
        "model_confidence": pd.DataFrame(),
        "resampled_posterior": pd.DataFrame(),
        "equities": pd.DataFrame(),
        "feature_views": {},
        "feature_categories": {},
        "view_category_mapping": {},
        "schema_metadata": {},
        "validation": {},
        "screens": {},
    }

    if not os.environ.get("GEIB_DASHBOARD", "").lower() == "true":
        print("⚠️ GEIB_DASHBOARD environment variable not set to 'true'")
        return data

    db_url = os.environ.get("DB_URL")
    if not db_url:
        print("⚠️ DB_URL environment variable not set")
        return data

    try:
        # --- Use data_utils engine factory (respects DB_URL env var) ---
        engine = get_analytics_engine()
        analytics_schema = os.environ.get("DB_ANALYTICS_SCHEMA", "analytics")

        # --- 0. Load schema metadata (drives numeric column detection) ---
        schema_meta = get_equities_schema(db_url=db_url)
        data["schema_metadata"] = schema_meta

        # Derive numeric columns from schema metadata instead of hardcoding
        _NUMERIC_TYPES = (
            "numeric",
            "double precision",
            "real",
            "integer",
            "bigint",
        )
        numeric_cols = [
            alias
            for alias, meta in schema_meta.items()
            if meta.get("column_type") in _NUMERIC_TYPES
        ]

        # --- 1. Summary Data (analytics table) ---
        query_summary = f"""
            SELECT * FROM {analytics_schema}.expected_returns_summary
            WHERE expected_upside_pct IS NOT NULL
            ORDER BY implied_return_pt DESC
        """
        data["summary"] = pd.read_sql(query_summary, engine)

        # Apply schema-driven numeric coercion
        for col in numeric_cols:
            if col in data["summary"].columns:
                data["summary"][col] = pd.to_numeric(data["summary"][col], errors="coerce")

        # --- 2. Tri-Model (reuse summary — same table) ---
        data["tri_model"] = data["summary"].copy()

        # --- 3. Earnings Probability Data ---
        try:
            data["earnings"] = pd.read_sql(
                f"SELECT * FROM {analytics_schema}.earnings_probability_analysis", engine
            )
        except Exception:
            print("⚠️ analytics.earnings_probability_analysis not found")

        # --- 4. Credit Risk Data ---
        try:
            data["credit"] = pd.read_sql(
                f"SELECT * FROM {analytics_schema}.credit_risk_analysis", engine
            )
        except Exception:
            # Fallback: credit risk columns now live in expected_returns_summary
            _credit_cols_in_summary = [
                c
                for c in [
                    "ticker",
                    "name",
                    "sector",
                    "industry",
                    "beta_stability_score",
                    "distress_probability",
                    "liquidity_stress_score",
                    "cash_runway_months",
                    "altman_z_score",
                    "altman_z_trend",
                    "risk_level",
                    "ci_lower",
                    "ci_upper",
                    "data_quality_score",
                    "wealth_buffer",
                    "ruin_probability",
                    "survival_probability",
                ]
                if c in data["summary"].columns
            ]
            if _credit_cols_in_summary:
                data["credit"] = data["summary"][_credit_cols_in_summary].copy()
                print("ℹ️ Credit risk data sourced from expected_returns_summary columns")
            else:
                print("⚠️ analytics.credit_risk_analysis not found")

        # --- 4b. Accounting Anomaly Data (Step 5b in pipeline) ---
        try:
            data["anomaly"] = pd.read_sql(
                f"SELECT * FROM {analytics_schema}.accounting_anomaly_analysis", engine
            )
        except Exception:
            # Fallback: anomaly columns now live in expected_returns_summary
            _anomaly_cols_in_summary = [
                c
                for c in [
                    "ticker",
                    "name",
                    "sector",
                    "industry",
                    "accounting_anomaly_score",
                    "sector_relative_anomaly",
                    "anomaly_feature_count",
                    "accounting_anomaly_tier",
                    "anomaly_severity_score",
                    "anomaly_risk_rank",
                    "sector_anomaly_percentile",
                    "multi_flag_alert",
                    "anomaly_conditional_probability",
                    "exceptional_items_frequency_anomaly_flag",
                    "gaap_adj_eps_gap_pct_anomaly_flag",
                    "asset_sale_boost_anomaly_flag",
                    "ebitda_adjustment_ratio_anomaly_flag",
                    "eps_adjustment_ratio_anomaly_flag",
                    "exceptional_items_to_ebitda_anomaly_flag",
                    "restructuring_intensity_anomaly_flag",
                    "goodwill_change_rate_anomaly_flag",
                ]
                if c in data["summary"].columns
            ]
            if _anomaly_cols_in_summary:
                data["anomaly"] = data["summary"][_anomaly_cols_in_summary].copy()
                print("ℹ️ Anomaly data sourced from expected_returns_summary columns")
            else:
                print("⚠️ analytics.accounting_anomaly_analysis not found")

        # --- 4c. Dividend Safety Data ---
        try:
            data["dividend_safety"] = pd.read_sql(
                f"SELECT * FROM {analytics_schema}.dividend_safety_analysis", engine
            )
        except Exception:
            # Fallback: dividend safety columns now live in expected_returns_summary
            _div_cols_in_summary = [
                c
                for c in [
                    "ticker",
                    "name",
                    "sector",
                    "industry",
                    "high_yield_flag",
                    "dividend_cut_probability",
                    "fcf_dividend_coverage",
                    "payout_ratio",
                    "dividend_streak",
                    "dividend_consistency",
                    "yield_vs_5y_avg",
                    "sustainable_flag",
                    "safety_score",
                    "risk_category",
                ]
                if c in data["summary"].columns
            ]
            if _div_cols_in_summary:
                data["dividend_safety"] = data["summary"][_div_cols_in_summary].copy()
                print("ℹ️ Dividend safety data sourced from expected_returns_summary columns")
            else:
                print("⚠️ analytics.dividend_safety_analysis not found")

        # --- 5. Model Confidence Metrics (derived from summary) ---
        # model_confidence_metrics is not a standalone table; confidence
        # columns live in expected_returns_summary.  Extract them here
        # so downstream dashboard components have a dedicated DataFrame.
        if not data["summary"].empty:
            _confidence_cols = [
                c
                for c in [
                    "ticker",
                    "confidence_score",
                    "confidence_level",
                    "posterior_beat_prob",
                    "beat_classification",
                    "prob_positive_upside",
                    "weighted_agreement",
                    "agreement_score",
                    "signal",
                ]
                if c in data["summary"].columns
            ]
            if _confidence_cols:
                data["model_confidence"] = data["summary"][_confidence_cols].copy()

        # --- 5b. Load identifier columns for downstream reordering/validation ---
        try:
            id_columns = load_identifier_columns(db_url=db_url)
            data["identifier_columns"] = id_columns
        except Exception as e:
            print(f"⚠️ Could not load identifier columns: {e}")
            data["identifier_columns"] = {}

        # --- 6. Full equities data with backfill (mirrors expected_returns_v3) ---
        try:
            df_eq = load_equities_data_from_db(db_url=db_url)
            df_eq = backfill_feature_columns(df_eq)
            data["equities"] = reorder_with_identifiers(df_eq)
        except Exception as e:
            print(f"⚠️ Could not load equities data: {e}")

        # --- 7. All feature views (for feature-level dashboards) ---
        try:
            data["feature_views"] = load_all_feature_views(db_url=db_url, return_dict=True)
        except Exception as e:
            print(f"⚠️ Could not load feature views: {e}")

        # --- 7b. Enrich equities with columns from feature views ---
        # Mirrors the sentiment/risk merge pattern in expected_returns_v3.
        # Ensures columns like altman_z_score and piotroski_f_score (which
        # live in vw_features_quality_risk / vw_features_composite_scores)
        # are present on the equities DataFrame before validation and viz.
        if not data["equities"].empty and data["feature_views"]:
            eq = data["equities"]
            for view_name, df_view in data["feature_views"].items():
                if df_view.empty or "ticker" not in df_view.columns:
                    continue
                missing_cols = [c for c in df_view.columns if c != "ticker" and c not in eq.columns]
                if missing_cols:
                    view_subset = df_view[["ticker"] + missing_cols].drop_duplicates(
                        subset="ticker"
                    )
                    eq = eq.merge(view_subset, on="ticker", how="left")
            data["equities"] = eq

        # --- 7c. Enrich summary with viz-critical columns from equities ---
        # Columns like altman_z_score and piotroski_f_score live in the
        # equities/feature-view data, not in expected_returns_summary.
        # Merge them so downstream viz functions find them on the summary df.
        if (
            not data["summary"].empty
            and not data["equities"].empty
            and "ticker" in data["summary"].columns
            and "ticker" in data["equities"].columns
        ):
            _viz_needed = {col for cols in VIZ_REQUIRED_COLUMNS.values() for col in cols}
            _missing_in_summary = [
                c
                for c in _viz_needed
                if c not in data["summary"].columns and c in data["equities"].columns
            ]
            if _missing_in_summary:
                _eq_subset = data["equities"][["ticker"] + _missing_in_summary].drop_duplicates(
                    subset="ticker"
                )
                data["summary"] = data["summary"].merge(_eq_subset, on="ticker", how="left")
                # Keep tri_model in sync (it was copied before enrichment)
                data["tri_model"] = data["summary"].copy()

        # --- 4d. Resampled Posterior Data ---
        try:
            data["resampled_posterior"] = pd.read_sql(
                f"SELECT * FROM {analytics_schema}.resampled_posterior_returns", engine
            )
        except Exception:
            # Fallback: resampled columns in expected_returns_summary
            _resamp_cols = [
                c
                for c in [
                    "ticker",
                    "resampled_posterior_mean",
                    "technical_adjustment",
                    "momentum_signal",
                    "volatility_regime_score",
                ]
                if c in data["summary"].columns
            ]
            if _resamp_cols:
                data["resampled_posterior"] = data["summary"][_resamp_cols].copy()
                print("ℹ️ Resampled posterior data sourced from expected_returns_summary columns")

        # --- 4e. Load Pre-computed Screening Results from DB ---
        _screen_table_names = [
            "earnings_quality_stocks",
            "value_stocks",
            "growth_momentum_stocks",
            "garp_stocks",
            "dividend_quality_stocks",
            "financial_health_stocks",
            "valuation_reversion_stocks",
            "integrity_growth_stocks",
            "high_yield_safe_stocks",
            "sector_relative_stocks",
            "low_vol_quality_stocks",
            "fcf_compounders_stocks",
            "total_return_leaders_stocks",
        ]
        for screen_name in _screen_table_names:
            try:
                data["screens"][screen_name] = pd.read_sql(
                    f"SELECT * FROM {analytics_schema}.{screen_name}", engine
                )
            except Exception:
                pass
        if data["screens"]:
            print(f"ℹ️ Loaded {len(data['screens'])} pre-computed screening tables from DB")

        # --- 8. Feature categories + view mapping ---
        data["feature_categories"] = load_feature_categories_from_db(db_url)
        data["view_category_mapping"] = get_view_category_mapping(db_url=db_url)

        # --- 9. Validate feature alignment ---
        if not data["equities"].empty and data["feature_categories"]:
            data["validation"] = validate_feature_alignment(
                data["equities"], data["feature_categories"]
            )
            low = {k: v for k, v in data["validation"].items() if v["coverage_pct"] < 80}
            if low:
                print(
                    f"⚠️ Low feature coverage in {len(low)} categories: " f"{', '.join(low.keys())}"
                )

        print("✓ Loaded GEIB data successfully")
        return data

    except Exception as e:
        print(f"❌ Error loading GEIB data: {e}")
        return data


# Load all data
all_data = load_geib_data()
df = all_data["summary"]
df_tri = all_data["tri_model"]
df_earnings = all_data["earnings"]
df_credit = all_data["credit"]
df_anomaly = all_data["anomaly"]
df_dividend_safety = all_data["dividend_safety"]
df_confidence = all_data["model_confidence"]
df_resampled_posterior = all_data["resampled_posterior"]
precomputed_screens = all_data["screens"]
view_category_mapping = all_data["view_category_mapping"]
feature_views = all_data["feature_views"]

# --- Startup validation: check viz column coverage against the SUMMARY DataFrame ---
# Validate against the actual summary DataFrame columns (which contain the
# computed analytics columns), NOT against the feature_categories registry
# (which only tracks raw equities feature columns).
if not df.empty:
    # Build a pseudo-category dict from the summary columns so
    # validate_viz_column_coverage can cross-check requirements.
    _summary_as_categories = {"summary": list(df.columns)}
    _viz_issues = validate_viz_column_coverage(_summary_as_categories, VIZ_REQUIRED_COLUMNS)
    if _viz_issues:
        print(f"⚠️ Viz column coverage gaps: {_viz_issues}")

# Load the return distribution fit artifact figure at startup
_return_dist_fit_path = PROJECT_ROOT / "outputs" / "analytics" / "er_return_distribution_fit.html"
_return_dist_fit_fig = load_plotly_figure_from_html(_return_dist_fit_path)

# Initialize Dash app
app = dash.Dash(
    __name__,
    title="Global Equity Analytics Dashboard",
    external_stylesheets=[dbc.themes.DARKLY],
)
server = app.server


# Flask route to serve HTML artifacts from outputs
@server.route("/artifacts/<path:filename>")
def serve_artifact(filename):
    """Serve HTML artifact files from the outputs directory."""
    artifacts_dir = str(PROJECT_ROOT / "outputs" / "analytics")
    return send_from_directory(artifacts_dir, filename)


# Custom CSS for dropdown selection highlighting and slider tooltips (consolidated)
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            /* Highlight selected values in multi-dropdown */
            .Select-value {{
                background-color: {COLORS["primary"]} !important;
                border: 1px solid {COLORS["border"]} !important;
                color: {COLORS["text_primary"]} !important;
            }}
            .Select-value-icon {{
                border-right: 1px solid {COLORS["border"]} !important;
                color: {COLORS["text_primary"]} !important;
            }}
            .Select-value-icon:hover {{
                background-color: {COLORS["danger"]} !important;
                color: {COLORS["text_primary"]} !important;
            }}
            .filter-label {{
                color: {COLORS["text_secondary"]};
                font-weight: bold;
                font-size: 0.9rem;
            }}
            /* Global Background */
            body {{
                background-color: {COLORS["background_main"]};
                color: {COLORS["text_primary"]};
            }}
            /* RangeSlider / Slider tooltip styling (consolidated) */
            .rc-slider-tooltip-inner,
            .rc-slider-tooltip .rc-slider-tooltip-inner,
            .dash-slider .rc-slider-tooltip-inner,
            [class*="rc-slider-tooltip"] .rc-slider-tooltip-inner,
            .rc-slider-tooltip-inner span,
            .rc-slider-tooltip-inner div {{
                color: #000000 !important;
                background-color: #ffffff !important;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
                font-weight: 600 !important;
                font-size: 13px !important;
                opacity: 1 !important;
            }}
            .rc-slider-tooltip-arrow,
            .rc-slider-tooltip .rc-slider-tooltip-arrow {{
                border-bottom-color: #ffffff !important;
                border-top-color: #ffffff !important;
            }}
            .rc-slider-tooltip *,
            .rc-slider-tooltip-inner *,
            [class*="rc-slider-tooltip"] * {{
                color: #000000 !important;
            }}
            .rc-slider-tooltip-placement-top .rc-slider-tooltip-inner,
            .rc-slider-tooltip-placement-bottom .rc-slider-tooltip-inner,
            .rc-slider-tooltip-placement-top .rc-slider-tooltip-inner *,
            .rc-slider-tooltip-placement-bottom .rc-slider-tooltip-inner * {{
                color: #000000 !important;
                background-color: #ffffff !important;
            }}
            .rc-slider-tooltip-placement-top .rc-slider-tooltip-arrow {{
                border-top-color: #ffffff !important;
            }}
            .rc-slider-tooltip-placement-bottom .rc-slider-tooltip-arrow {{
                border-bottom-color: #ffffff !important;
            }}
            .rc-slider-mark-text,
            .rc-slider-mark .rc-slider-mark-text,
            .dash-slider .rc-slider-mark-text {{
                color: {COLORS["text_secondary"]} !important;
                font-size: 11px !important;
                white-space: nowrap !important;
            }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

# Compute slider bounds for Price Target vs Current Price controls
_pt_price_min = 0
_pt_price_max = 1000
_pt_target_min = 0
_pt_target_max = 1000
_pt_price_step = 1
_pt_target_step = 1

if not df.empty:
    if "last_price" in df.columns:
        _lp = pd.to_numeric(df["last_price"], errors="coerce").dropna()
        if len(_lp) > 0:
            _pt_price_min = float(max(0, _lp.quantile(0.0)))
            _pt_price_max = float(_lp.quantile(0.99))
            _pt_price_step = max(1, round((_pt_price_max - _pt_price_min) / 500))

    # Use the max across all three target metrics for the target slider range
    _target_vals = pd.Series(dtype=float)
    for _tcol in [
        "price_target_median",
        "price_target_mc",
        "kalman_estimate",
        "price_target_prob_weighted",
    ]:
        if _tcol in df.columns:
            _target_vals = pd.concat(
                [_target_vals, pd.to_numeric(df[_tcol], errors="coerce").dropna()]
            )
    if len(_target_vals) > 0:
        _pt_target_min = float(max(0, _target_vals.quantile(0.0)))
        _pt_target_max = float(_target_vals.quantile(0.99))
        _pt_target_step = max(1, round((_pt_target_max - _pt_target_min) / 500))

# Layout
app.layout = html.Div(
    [
        html.H1(
            "🌍 Global Equity Analytics Dashboard (GEIB)",
            style={"textAlign": "center", "marginTop": "20px"},
        ),
        html.P(
            "Expected Returns Analysis from Quad-Model Consensus (Monte Carlo, Kalman Filter, Price Target Achievement, Earnings Beat)",
            style={
                "textAlign": "center",
                "fontStyle": "italic",
                "color": COLORS["text_secondary"],
            },
        ),
        # Status indicator
        html.Div(
            id="status-indicator",
            children=[
                html.Span(
                    (
                        f"✅ Data Loaded: {len(df):,} stocks | Last Updated: {df['last_updated'].max()}"
                        if len(df) > 0 and "last_updated" in df.columns
                        else (
                            f"✅ Data Loaded: {len(df):,} stocks"
                            if len(df) > 0
                            else "⚠️ No data loaded"
                        )
                    ),
                    style={
                        "margin": "0 10px",
                        "color": COLORS["success"] if len(df) > 0 else COLORS["warning"],
                    },
                )
            ],
            style={"textAlign": "center", "padding": "10px"},
        ),
        # Global filter state store (cross-tab persistence)
        dcc.Store(id="global-filter-store", data={}),
        # KPI Summary Cards
        html.Div(
            id="kpi-cards",
            style={
                "display": "flex",
                "justifyContent": "space-around",
                "flexWrap": "wrap",
                "gap": "10px",
                "padding": "10px 0",
                "margin": "10px 0",
                "width": "100%",
            },
        ),
        # Filters — generated from FILTER_CONFIG
        build_filter_panel(df),
        # Tabs for different views
        dcc.Tabs(
            [
                dcc.Tab(
                    label="📊 Expected Returns Overview",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="returns-scatter"),
                                dcc.Graph(id="artifact-er-summary-posterior"),
                                # Price Target vs Current Price Scatter Controls
                                html.Div(
                                    [
                                        html.H4(
                                            "Price Target vs Current Price",
                                            style={
                                                "textAlign": "center",
                                                "marginBottom": "10px",
                                            },
                                        ),
                                        html.P(
                                            "Scatter plot showing price target vs current price with upside potential. "
                                            "Points above the diagonal indicate upside potential; points below indicate downside risk.",
                                            style={
                                                "textAlign": "center",
                                                "fontStyle": "italic",
                                                "color": "#999",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "flexDirection": "row",
                                                "flexWrap": "wrap",
                                                "rowGap": "10px",
                                                "alignItems": "center",
                                                "marginBottom": "15px",
                                                "justifyContent": "center",
                                            },
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Label(
                                                            "Price Target Metric:",
                                                            style={
                                                                "marginBottom": "5px",
                                                                "fontWeight": "bold",
                                                                "display": "block",
                                                                "color": "white",
                                                            },
                                                        ),
                                                        dcc.Dropdown(
                                                            id="pt-scatter-metric-control",
                                                            options=[
                                                                {
                                                                    "label": "Price Target Median",
                                                                    "value": "price_target_median",
                                                                },
                                                                {
                                                                    "label": "Kalman Estimate",
                                                                    "value": "kalman_estimate",
                                                                },
                                                                {
                                                                    "label": "Price Target MC",
                                                                    "value": "price_target_mc",
                                                                },
                                                                {
                                                                    "label": "Price Target Prob Weighted",
                                                                    "value": "price_target_prob_weighted",
                                                                },
                                                            ],
                                                            value="price_target_median",
                                                            style={
                                                                "minWidth": "200px",
                                                                "color": "black",
                                                            },
                                                            searchable=False,
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "marginRight": "15px",
                                                    },
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label(
                                                            "Size Encoding:",
                                                            style={
                                                                "marginBottom": "5px",
                                                                "fontWeight": "bold",
                                                                "display": "block",
                                                                "color": "white",
                                                            },
                                                        ),
                                                        dcc.Dropdown(
                                                            id="pt-scatter-size-control",
                                                            options=[
                                                                {
                                                                    "label": "Expected Upside %",
                                                                    "value": "expected_upside_pct",
                                                                },
                                                                {
                                                                    "label": "Market Cap",
                                                                    "value": "market_cap",
                                                                },
                                                                {
                                                                    "label": "Volume",
                                                                    "value": "volume_shrs",
                                                                },
                                                                {
                                                                    "label": "None",
                                                                    "value": "none",
                                                                },
                                                            ],
                                                            value="expected_upside_pct",
                                                            style={
                                                                "minWidth": "200px",
                                                                "color": "black",
                                                            },
                                                            searchable=False,
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "marginRight": "15px",
                                                    },
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label(
                                                            "Color Encoding:",
                                                            style={
                                                                "marginBottom": "5px",
                                                                "fontWeight": "bold",
                                                                "display": "block",
                                                                "color": "white",
                                                            },
                                                        ),
                                                        dcc.Dropdown(
                                                            id="pt-scatter-color-control",
                                                            options=[
                                                                {
                                                                    "label": "Sector",
                                                                    "value": "sector",
                                                                },
                                                                {
                                                                    "label": "Confidence Level",
                                                                    "value": "confidence_level",
                                                                },
                                                                {
                                                                    "label": "Beat Classification",
                                                                    "value": "beat_classification",
                                                                },
                                                                {
                                                                    "label": "None",
                                                                    "value": "none",
                                                                },
                                                            ],
                                                            value="sector",
                                                            style={
                                                                "minWidth": "200px",
                                                                "color": "black",
                                                            },
                                                            searchable=False,
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "marginRight": "15px",
                                                    },
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "flexDirection": "row",
                                                "flexWrap": "wrap",
                                                "rowGap": "10px",
                                                "alignItems": "center",
                                                "marginBottom": "15px",
                                                "justifyContent": "center",
                                            },
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Label(
                                                            "Last Price Range:",
                                                            style={
                                                                "marginBottom": "5px",
                                                                "fontWeight": "bold",
                                                                "display": "block",
                                                                "color": "white",
                                                            },
                                                        ),
                                                        dcc.RangeSlider(
                                                            id="pt-scatter-price-range",
                                                            min=_pt_price_min,
                                                            max=_pt_price_max,
                                                            step=_pt_price_step,
                                                            value=[_pt_price_min, _pt_price_max],
                                                            marks={
                                                                int(
                                                                    _pt_price_min
                                                                ): f"${int(_pt_price_min)}",
                                                                int(
                                                                    _pt_price_max
                                                                ): f"${int(_pt_price_max)}",
                                                            },
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                            allowCross=False,
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "marginRight": "30px",
                                                        "minWidth": "300px",
                                                        "flex": "1",
                                                        "paddingBottom": "25px",
                                                    },
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label(
                                                            "Price Target Range:",
                                                            style={
                                                                "marginBottom": "5px",
                                                                "fontWeight": "bold",
                                                                "display": "block",
                                                                "color": "white",
                                                            },
                                                        ),
                                                        dcc.RangeSlider(
                                                            id="pt-scatter-target-range",
                                                            min=_pt_target_min,
                                                            max=_pt_target_max,
                                                            step=_pt_target_step,
                                                            value=[_pt_target_min, _pt_target_max],
                                                            marks={
                                                                int(
                                                                    _pt_target_min
                                                                ): f"${int(_pt_target_min)}",
                                                                int(
                                                                    _pt_target_max
                                                                ): f"${int(_pt_target_max)}",
                                                            },
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                            allowCross=False,
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "marginRight": "30px",
                                                        "minWidth": "300px",
                                                        "flex": "1",
                                                        "paddingBottom": "25px",
                                                    },
                                                ),
                                            ],
                                        ),
                                    ],
                                    style={
                                        "padding": "15px",
                                        "backgroundColor": "#333",
                                        "borderRadius": "5px",
                                        "margin": "10px 0",
                                    },
                                ),
                                dcc.Loading(
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="price_target_vs_current_scatter",
                                            style={
                                                "minHeight": "550px",
                                                "height": "calc(100vh - 600px)",
                                            },
                                        )
                                    ],
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🎯 Model Consensus",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="model-signals-plot"),
                                dcc.Graph(id="confidence-distribution"),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🏆 Top Opportunities",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "High Conviction Opportunities",
                                    style={"textAlign": "center", "margin": "20px"},
                                ),
                                dcc.Graph(id="artifact-er-strong-consensus"),
                                dash_table.DataTable(
                                    id="top-opportunities-table",
                                    columns=get_formatted_columns(
                                        [
                                            "ticker",
                                            "name",
                                            "country",
                                            "trading_country",
                                            "exchange",
                                            "sector",
                                            "industry",
                                            "quality_tier",
                                            "next_earnings",
                                            "next_earnings_status",
                                            "last_price",
                                            "price_target",
                                            "price_target_low",
                                            "price_target_high",
                                            "price_target_median",
                                            "price_target_mc",
                                            "price_target_prob_weighted",
                                            "kalman_estimate",
                                            "expected_upside_pct",
                                            "expected_upside_kalman",
                                            "implied_return_pt",
                                            "var_5_pct",
                                            "risk_reward_ratio",
                                            "achievement_probability",
                                            "posterior_beat_prob",
                                            "beat_classification",
                                            "agreement_score",
                                            "weighted_agreement",
                                            "composite_score",
                                            "confidence_level",
                                            "signal",
                                            "accounting_anomaly_tier",
                                            "risk_level",
                                            "expected_upside_pct_pctile",
                                            "expected_upside_kalman_zscore",
                                        ]
                                    ),
                                    page_size=500,
                                    sort_action="native",
                                    filter_action="native",
                                    style_table={"overflowX": "auto", "width": "100%"},
                                    style_header=TABLE_STYLE_HEADER,
                                    style_cell=TABLE_STYLE_CELL,
                                    style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                                ),
                                # Screening Summary
                                html.Div(
                                    [
                                        html.H4(
                                            "Stock Screening Summary",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dcc.Graph(id="artifact-er-screening-summary"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="📈 Signal Analysis",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="signal-breakdown"),
                                dcc.Graph(id="regional-performance"),
                                dcc.Graph(id="artifact-er-sector-heatmap"),
                                dcc.Graph(id="artifact-er-sector-risk-reward"),
                                # Sector Return Analytics Heatmap
                                html.Div(
                                    [
                                        html.H4(
                                            "Sector Return Analytics",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dcc.Graph(id="artifact-er-sector-return-analytics"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                # Model Dispersion Dashboard
                                html.Div(
                                    [
                                        html.H4(
                                            "Model Dispersion Dashboard",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dcc.Graph(id="artifact-er-model-dispersion"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                # Growth Analysis Artifacts (v3.1) — dynamic
                                html.Div(
                                    [
                                        html.H4(
                                            "Sustainable Growth Analysis",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-sustainable-growth"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Growth Acceleration",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-growth-acceleration"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Growth vs Profitability Quadrant",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-growth-vs-profitability"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Growth Consistency Matrix",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-growth-consistency-matrix"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Growth Waterfall",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-growth-waterfall"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="📊 Z-Score & Percentile Ranking",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Cross-Sectional Z-Score & Percentile Analysis",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Identify statistical outliers and relative positioning across the universe using z-scores and percentile ranks.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Z-Score controls
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Metric:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="zscore-metric-dropdown",
                                                    options=[
                                                        {
                                                            "label": "Expected Upside %",
                                                            "value": "expected_upside_pct",
                                                        },
                                                        {
                                                            "label": "Filtered Upside",
                                                            "value": "expected_upside_kalman",
                                                        },
                                                        {
                                                            "label": "Prob-Weighted Return",
                                                            "value": "implied_return_pt",
                                                        },
                                                    ],
                                                    value="expected_upside_pct",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Color By:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="zscore-color-dropdown",
                                                    options=[
                                                        {
                                                            "label": "Sector",
                                                            "value": "sector",
                                                        },
                                                        {
                                                            "label": "Quality Tier",
                                                            "value": "quality_tier",
                                                        },
                                                        {
                                                            "label": "Signal",
                                                            "value": "signal",
                                                        },
                                                        {
                                                            "label": "Confidence Level",
                                                            "value": "confidence_level",
                                                        },
                                                        {
                                                            "label": "None",
                                                            "value": "none",
                                                        },
                                                    ],
                                                    value="quality_tier",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Size By:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="zscore-size-dropdown",
                                                    options=[
                                                        {
                                                            "label": "Composite Score",
                                                            "value": "composite_score",
                                                        },
                                                        {
                                                            "label": "Market Cap",
                                                            "value": "market_cap",
                                                        },
                                                        {
                                                            "label": "Confidence Score",
                                                            "value": "confidence_score",
                                                        },
                                                        {
                                                            "label": "None",
                                                            "value": "none",
                                                        },
                                                    ],
                                                    value="composite_score",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Z-Score Threshold:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Slider(
                                                    id="zscore-threshold-slider",
                                                    min=0.5,
                                                    max=3.0,
                                                    step=0.25,
                                                    value=1.5,
                                                    marks={
                                                        0.5: "0.5σ",
                                                        1.0: "1σ",
                                                        1.5: "1.5σ",
                                                        2.0: "2σ",
                                                        2.5: "2.5σ",
                                                        3.0: "3σ",
                                                    },
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "30%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                                "paddingBottom": "25px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px 0",
                                    },
                                ),
                                html.Div(id="zscore-kpi-cards", style={"margin": "10px 0"}),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="zscore-scatter-plot")],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id="percentile-distribution-plot")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="zscore-sector-box-plot")],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id="composite-vs-percentile-plot")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                # Z-Score ranked table
                                html.Div(
                                    [
                                        html.H4(
                                            "Top Statistical Outliers (by Z-Score)",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dash_table.DataTable(
                                            id="zscore-ranking-table",
                                            columns=get_formatted_columns(
                                                [
                                                    "ticker",
                                                    "name",
                                                    "country",
                                                    "trading_country",
                                                    "unit",
                                                    "exchange",
                                                    "industry",
                                                    "sector",
                                                    "quality_tier",
                                                    "expected_upside_pct",
                                                    "expected_upside_pct_zscore",
                                                    "expected_upside_pct_pctile",
                                                    "expected_upside_kalman",
                                                    "expected_upside_kalman_zscore",
                                                    "expected_upside_kalman_pctile",
                                                    "implied_return_pt_zscore",
                                                    "implied_return_pt_pctile",
                                                    "posterior_beat_prob",
                                                    "beat_classification",
                                                    "agreement_score",
                                                    "weighted_agreement",
                                                    "composite_score",
                                                    "confidence_level",
                                                    "signal",
                                                    "price_target_prob_weighted",
                                                    "price_target_mc",
                                                ]
                                            ),
                                            page_size=500,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={"overflowX": "auto", "width": "100%"},
                                            style_header=TABLE_STYLE_HEADER,
                                            style_cell=TABLE_STYLE_CELL,
                                            style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                                        ),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                # Valuation Analysis Artifacts (v3.1) — dynamic
                                html.Div(
                                    [
                                        html.H4(
                                            "Historical Valuation Percentile",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-historical-valuation-percentile"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Valuation vs Growth Quadrant",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-valuation-vs-growth"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Relative Valuation Matrix",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-relative-valuation-matrix"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Valuation Distribution",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-valuation-distribution"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Valuation Multiples Comparison",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-valuation-multiples"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="📅 Earnings Calendar & Events",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Earnings Calendar & Corporate Events",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Track upcoming earnings dates, dividend schedules, and fiscal year milestones for filtered stocks.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Days Ahead:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="earnings-days-ahead",
                                                    options=[
                                                        {"label": "7 days", "value": 7},
                                                        {
                                                            "label": "14 days",
                                                            "value": 14,
                                                        },
                                                        {
                                                            "label": "30 days",
                                                            "value": 30,
                                                        },
                                                        {
                                                            "label": "60 days",
                                                            "value": 60,
                                                        },
                                                        {
                                                            "label": "90 days",
                                                            "value": 90,
                                                        },
                                                    ],
                                                    value=30,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Sort By:", style={"color": "white"}),
                                                dcc.Dropdown(
                                                    id="earnings-sort-by",
                                                    options=[
                                                        {
                                                            "label": "Next Earnings Date",
                                                            "value": "next_earnings",
                                                        },
                                                        {
                                                            "label": "Expected Upside",
                                                            "value": "expected_upside_pct",
                                                        },
                                                        {
                                                            "label": "Composite Score",
                                                            "value": "composite_score",
                                                        },
                                                    ],
                                                    value="next_earnings",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px 0",
                                    },
                                ),
                                html.Div(
                                    id="earnings-calendar-kpis",
                                    style={"margin": "10px 0"},
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="earnings-timeline-chart")],
                                            width=7,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id="earnings-by-status-chart")],
                                            width=5,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Upcoming Earnings Reports",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dash_table.DataTable(
                                            id="earnings-calendar-table",
                                            columns=get_formatted_columns(
                                                [
                                                    "ticker",
                                                    "name",
                                                    "country",
                                                    "unit",
                                                    "trading_country",
                                                    "exchange",
                                                    "sector",
                                                    "industry",
                                                    "next_earnings",
                                                    "next_earnings_status",
                                                    "next_earnings_when",
                                                    "next_fiscal_quarter",
                                                    "earnings_report_frequency",
                                                    "expected_upside_pct",
                                                    "posterior_beat_prob",
                                                    "beat_classification",
                                                    "agreement_score",
                                                    "composite_score",
                                                    "signal",
                                                    "quality_tier",
                                                ]
                                            ),
                                            page_size=500,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={"overflowX": "auto", "width": "100%"},
                                            style_header=TABLE_STYLE_HEADER,
                                            style_cell=TABLE_STYLE_CELL,
                                            style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                                        ),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(id="artifact-er-earnings-prob-dashboard"),
                                # Earnings Quality Artifacts (v3.1) — dynamic
                                html.Div(
                                    [
                                        html.H4(
                                            "Earnings Consistency Matrix",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-earnings-consistency-matrix"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Beat Rate Heatmap",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-beat-rate-heatmap"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Earnings Quality Decomposition",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-earnings-quality-decomposition"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "EPS Trajectory Analysis",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-eps-trajectory"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Earnings Surprise Dashboard",
                                            style={"textAlign": "center", "marginTop": "20px"},
                                        ),
                                        dcc.Graph(id="dynamic-earnings-surprise"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🏆 Risk-Adjusted Ranking",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Expected Value Risk-Adjusted Ranking",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Rank stocks by their risk-adjusted expected value, combining upside potential, probability of success, and confidence levels.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Ranking Filters
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Scoring Method",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="scoring-method-dropdown",
                                                    options=[
                                                        {
                                                            "label": "Base EV",
                                                            "value": "base_ev",
                                                        },
                                                        {
                                                            "label": "Probability-weighted",
                                                            "value": "prob_weighted",
                                                        },
                                                        {
                                                            "label": "Confidence-adjusted",
                                                            "value": "confidence_adj",
                                                        },
                                                        {
                                                            "label": "Achievement-adjusted",
                                                            "value": "achievement_adj",
                                                        },
                                                        {
                                                            "label": "Combined",
                                                            "value": "combined",
                                                        },
                                                    ],
                                                    value="combined",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "18%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Min Agreement",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="min-agreement-dropdown",
                                                    options=[
                                                        {"label": str(i), "value": i}
                                                        for i in [0, 1, 2, 3, 4]
                                                    ],
                                                    value=2,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "18%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Min Confidence",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="min-confidence-dropdown",
                                                    options=[
                                                        {"label": str(i), "value": i}
                                                        for i in [
                                                            0.15,
                                                            0.25,
                                                            0.35,
                                                            0.45,
                                                        ]
                                                    ],
                                                    value=0.25,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "18%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Risk-Free Rate",
                                                    style={"color": "white"},
                                                ),
                                                dcc.RangeSlider(
                                                    id="risk-free-rate-dropdown",
                                                    min=0,
                                                    max=1,
                                                    step=0.05,
                                                    value=[0.03],
                                                    marks={
                                                        0: "0%",
                                                        0.25: "25%",
                                                        0.5: "50%",
                                                        0.75: "75%",
                                                        1: "100%",
                                                    },
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                                "paddingBottom": "25px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Scatter Color By",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="scatter-color-dropdown",
                                                    options=[
                                                        {
                                                            "label": "None",
                                                            "value": "none",
                                                        },
                                                        {
                                                            "label": "Signal",
                                                            "value": "signal",
                                                        },
                                                        {
                                                            "label": "Quality Tier",
                                                            "value": "quality_tier",
                                                        },
                                                        {
                                                            "label": "Confidence Level",
                                                            "value": "confidence_level",
                                                        },
                                                    ],
                                                    value="signal",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "18%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Select Tickers (Probabilistic)",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="prob-ticker-dropdown",
                                                    multi=True,
                                                    options=(
                                                        [
                                                            {"label": i, "value": i}
                                                            for i in sorted(
                                                                df["ticker"].dropna().unique()
                                                            )
                                                        ]
                                                        if "ticker" in df.columns and len(df) > 0
                                                        else []
                                                    ),
                                                    placeholder="Select tickers for detailed analysis...",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "97%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px 0",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="ranking-bar-chart")], width=6),
                                        dbc.Col(
                                            [dcc.Graph(id="risk-reward-scatter")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                html.Div(id="artifact-er-quality-risk-quadrant"),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🔮 Probabilistic Analysis",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Bayesian Probabilistic Analysis",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "ArviZ-enhanced visualizations for posterior returns, beat probabilities, and ruin diagnostics.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="posterior-forest-plot")],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id="tri-model-posterior")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="beat-prob-posterior")],
                                            width=6,
                                        ),
                                        dbc.Col([dcc.Graph(id="ruin-diagnostic")], width=6),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="category-ridge-plot")],
                                            width=12,
                                        ),
                                    ]
                                ),
                                dcc.Graph(id="artifact-er-posterior-return-forest"),
                                dcc.Graph(id="artifact-er-beat-prob-posterior"),
                                dcc.Graph(id="artifact-er-beat-vs-achievement"),
                                dcc.Graph(id="artifact-er-ruin-prob-diagnostic"),
                                # Bayesian Ridge Plots (Profitability & Sentiment)
                                html.Div(
                                    [
                                        html.H4(
                                            "Bayesian Profitability Ridge Plot",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dcc.Graph(id="artifact-er-bayesian-profitability-ridge"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Bayesian Sentiment Ridge Plot",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dcc.Graph(id="artifact-er-bayesian-sentiment-ridge"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                # Distress Early Warning
                                html.Div(
                                    [
                                        html.H4(
                                            "Distress Early Warning Dashboard",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        html.Div(id="artifact-er-distress-early-warning"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                # Accounting Anomaly Dashboard (Step 5b)
                                html.Div(
                                    [
                                        html.H4(
                                            "Accounting Anomaly Detection Dashboard",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        html.P(
                                            "Beneish M-Score, revenue-expense divergence, and accounting quality indicators. "
                                            "Anomaly detection runs before credit risk to flag integrity issues early.",
                                            style={
                                                "textAlign": "center",
                                                "fontStyle": "italic",
                                                "color": "#999",
                                            },
                                        ),
                                        html.Div(id="artifact-er-accounting-anomaly"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                # Dividend Safety Summary
                                html.Div(
                                    [
                                        html.H4(
                                            "Dividend Safety Analysis",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        html.Div(id="artifact-er-dividend-safety"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                # Earnings Beat / Revision Artifacts (v3.1)
                                render_artifact_or_placeholder(
                                    "er_price_target_drift", "Price Target Drift Dashboard"
                                ),
                                render_artifact_or_placeholder(
                                    "er_gaap_divergence", "GAAP vs Adjusted EPS Divergence"
                                ),
                                render_artifact_or_placeholder(
                                    "er_revision_momentum", "Revision Momentum"
                                ),
                                # Earnings Quality & Growth Artifacts (v3.1 — previously not surfaced)
                                render_artifact_or_placeholder(
                                    "er_enhanced_beat_probability", "Enhanced Beat Probability"
                                ),
                                render_artifact_or_placeholder(
                                    "er_beat_rate_heatmap", "Beat Rate Heatmap"
                                ),
                                render_artifact_or_placeholder(
                                    "er_earnings_consistency_matrix", "Earnings Consistency Matrix"
                                ),
                                render_artifact_or_placeholder(
                                    "er_earnings_quality_decomposition",
                                    "Earnings Quality Decomposition",
                                ),
                                # Other Missing Artifacts
                                render_artifact_or_placeholder(
                                    "er_kalman_vs_raw", "Kalman vs Raw Scatter"
                                ),
                                render_artifact_or_placeholder(
                                    "er_tri_model_agreement", "Tri-Model Agreement"
                                ),
                                render_artifact_or_placeholder(
                                    "er_posterior_return_forest", "Posterior Return Forest Plot"
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🛡️ Credit Risk & Dividend Safety",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Credit Risk & Dividend Safety Analysis",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Comprehensive view of financial distress indicators, Altman Z-Score, "
                                    "ruin probability, dividend sustainability, and accounting anomaly flags.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # KPI cards
                                html.Div(id="credit-risk-kpis", style={"margin": "10px 0"}),
                                # Charts
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="altman-zscore-distribution")],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id="risk-level-pie")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="anomaly-tier-distribution")],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id="dividend-safety-scatter")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                # Anomaly flags heatmap
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="anomaly-flags-heatmap")],
                                            width=12,
                                        ),
                                    ]
                                ),
                                # Credit risk table
                                html.Div(
                                    [
                                        html.H4(
                                            "Credit Risk & Safety Detail",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dash_table.DataTable(
                                            id="credit-risk-table",
                                            columns=get_formatted_columns(
                                                [
                                                    "ticker",
                                                    "name",
                                                    "country",
                                                    "trading_country",
                                                    "exchange",
                                                    "sector",
                                                    "industry",
                                                    "style_class",
                                                    "size_class",
                                                    "unit",
                                                    "risk_level",
                                                    "altman_z_score",
                                                    "altman_z_trend",
                                                    "distress_probability",
                                                    "liquidity_stress_score",
                                                    "cash_runway_months",
                                                    "beta_stability_score",
                                                    "ruin_probability",
                                                    "survival_probability",
                                                    "wealth_buffer",
                                                    "accounting_anomaly_tier",
                                                    "accounting_anomaly_score",
                                                    "anomaly_feature_count",
                                                    "anomaly_conditional_probability",
                                                    "multi_flag_alert",
                                                    "dividend_cut_probability",
                                                    "safety_score",
                                                    "risk_category",
                                                    "fcf_dividend_coverage",
                                                    "payout_ratio",
                                                    "dividend_streak",
                                                    "dividend_consistency",
                                                    "data_quality_score",
                                                    "signal",
                                                ]
                                            ),
                                            page_size=500,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={"overflowX": "auto", "width": "100%"},
                                            style_header=TABLE_STYLE_HEADER,
                                            style_cell=TABLE_STYLE_CELL,
                                            style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                                        ),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                # MCMC Posterior Charts (v3.3)
                                html.H4(
                                    "MCMC Posterior Diagnostics",
                                    style={"textAlign": "center", "marginTop": "30px"},
                                ),
                                html.P(
                                    "MCMC-enhanced Bayesian posterior distributions for credit risk, dividend cut, "
                                    "and price target achievement probabilities.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                render_artifact_or_placeholder(
                                    "er_mcmc_credit_risk_posterior", "MCMC Credit Risk Posterior"
                                ),
                                render_artifact_or_placeholder(
                                    "er_mcmc_dividend_cut_posterior", "MCMC Dividend Cut Posterior"
                                ),
                                render_artifact_or_placeholder(
                                    "er_mcmc_price_target_posterior", "MCMC Price Target Posterior"
                                ),
                                # Quality & Risk Artifacts (v3.2)
                                html.H4(
                                    "Quality & Risk Scoring",
                                    style={"textAlign": "center", "marginTop": "30px"},
                                ),
                                render_artifact_or_placeholder(
                                    "er_piotroski_fscore", "Piotroski F-Score Breakdown"
                                ),
                                render_artifact_or_placeholder(
                                    "er_altman_zscore", "Altman Z-Score Distribution"
                                ),
                                render_artifact_or_placeholder(
                                    "er_beneish_mscore", "Beneish M-Score Analysis"
                                ),
                                render_artifact_or_placeholder(
                                    "er_risk_tier_sunburst",
                                    "Sector → Industry → Risk Tier Sunburst",
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🔍 Accounting Anomaly Analytics",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Bayesian Accounting Anomaly Detection & Analytics",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Anomaly severity scoring, conditional probability analysis, "
                                    "and per-feature Bayesian lift diagnostics from the "
                                    "AccountingAnomalyProbabilityModel.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # KPI cards
                                html.Div(id="anomaly-analytics-kpis", style={"margin": "10px 0"}),
                                # Anomaly Severity Dashboard (6-panel)
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(
                                                    id="anomaly-severity-dashboard",
                                                    style={"width": "100%", "height": "auto"},
                                                )
                                            ],
                                            width=12,
                                            style={"padding": "0"},
                                        ),
                                    ],
                                    className="g-0",
                                ),
                                # Conditional Probability Chart (4-panel)
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(
                                                    id="anomaly-cond-prob-chart",
                                                    style={"width": "100%", "height": "auto"},
                                                )
                                            ],
                                            width=12,
                                            style={"padding": "0"},
                                        ),
                                    ],
                                    className="g-0",
                                ),
                                # Anomaly detail table
                                html.Div(
                                    [
                                        html.H4(
                                            "Anomaly Severity Detail",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dash_table.DataTable(
                                            id="anomaly-analytics-table",
                                            columns=get_formatted_columns(
                                                [
                                                    "ticker",
                                                    "name",
                                                    "country",
                                                    "trading_country",
                                                    "exchange",
                                                    "sector",
                                                    "industry",
                                                    "style_class",
                                                    "size_class",
                                                    "unit",
                                                    "accounting_anomaly_tier",
                                                    "accounting_anomaly_score",
                                                    "anomaly_severity_score",
                                                    "anomaly_conditional_probability",
                                                    "anomaly_risk_rank",
                                                    "sector_anomaly_percentile",
                                                    "anomaly_feature_count",
                                                    "multi_flag_alert",
                                                ]
                                            ),
                                            page_size=500,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={"overflowX": "auto", "width": "100%"},
                                            style_header=TABLE_STYLE_HEADER,
                                            style_cell=TABLE_STYLE_CELL,
                                            style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                                        ),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                                # MCMC Anomaly Posterior (v3.3)
                                render_artifact_or_placeholder(
                                    "er_mcmc_anomaly_posterior", "MCMC Anomaly Posterior"
                                ),
                                render_artifact_or_placeholder(
                                    "er_mcmc_category_sentiment_posterior",
                                    "MCMC Category Sentiment Posterior",
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🔬 Uncertainty & Calibration",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Model Uncertainty & Calibration Analysis",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Assess model confidence calibration, prediction intervals, and uncertainty quantification.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([dcc.Graph(id="calibration-curve")], width=6),
                                        dbc.Col(
                                            [dcc.Graph(id="prediction-interval-coverage")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="uncertainty-distribution")],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id="model-agreement-heatmap")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [html.Div(id="calibration-metrics-display")],
                                            width=12,
                                        ),
                                    ]
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🛡️ Safety Rails & Data Quality",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Data Quality & Safety Rails",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Monitor data quality metrics, missing values, outliers, and safety thresholds.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="data-completeness-chart")],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id="outlier-detection-chart")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="data-freshness-chart")],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id="safety-threshold-chart")],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [html.Div(id="data-quality-summary")],
                                            width=12,
                                        ),
                                    ]
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🏛️ Model Governance",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Model Governance & Audit Trail",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Track model versions, performance metrics over time, and governance documentation.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [dcc.Graph(id="model-performance-trend")],
                                            width=6,
                                        ),
                                        dbc.Col([dcc.Graph(id="model-drift-chart")], width=6),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.H4(
                                                            "Model Registry",
                                                            style={"marginTop": "20px"},
                                                        ),
                                                        dash_table.DataTable(
                                                            id="model-registry-table",
                                                            page_size=500,
                                                            style_table={
                                                                "overflowX": "auto",
                                                                "width": "100%",
                                                            },
                                                            style_header={
                                                                "backgroundColor": "rgb(30, 30, 30)",
                                                                "color": "white",
                                                                "fontWeight": "bold",
                                                            },
                                                            style_data={
                                                                "backgroundColor": "rgb(50, 50, 50)",
                                                                "color": "white",
                                                            },
                                                        ),
                                                    ]
                                                )
                                            ],
                                            width=12,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [html.Div(id="governance-metrics-display")],
                                            width=12,
                                        ),
                                    ]
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🎲 Monte Carlo Simulator",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Monte Carlo Portfolio Outcome Simulator",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Simulate thousands of possible portfolio outcomes based on expected returns and probabilities. See the range of potential results and the likelihood of achieving your target return.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Monte Carlo Filters
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Simulations:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="mc-num-simulations",
                                                    options=[
                                                        {
                                                            "label": "1,000",
                                                            "value": 1000,
                                                        },
                                                        {
                                                            "label": "5,000",
                                                            "value": 5000,
                                                        },
                                                        {
                                                            "label": "10,000",
                                                            "value": 10000,
                                                        },
                                                        {
                                                            "label": "50,000",
                                                            "value": 50000,
                                                        },
                                                    ],
                                                    value=10000,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Loss Ratio:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.RangeSlider(
                                                    id="mc-loss-ratio",
                                                    min=0,
                                                    max=1,
                                                    step=0.05,
                                                    value=[0.5],
                                                    marks={
                                                        0: "0%",
                                                        0.25: "25%",
                                                        0.5: "50%",
                                                        0.75: "75%",
                                                        1: "100%",
                                                    },
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                                "paddingBottom": "25px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Weighting:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="mc-weighting",
                                                    options=[
                                                        {
                                                            "label": "Equal-weighted",
                                                            "value": "equal",
                                                        },
                                                        {
                                                            "label": "Kelly-weighted",
                                                            "value": "kelly",
                                                        },
                                                        {
                                                            "label": "Market cap proxy",
                                                            "value": "market_cap",
                                                        },
                                                    ],
                                                    value="equal",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Target Return:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Slider(
                                                    id="mc-target-return",
                                                    min=0.0,
                                                    max=20.0,
                                                    step=0.1,
                                                    value=0.0,
                                                    marks={i: f"{i}%" for i in range(0, 25, 5)},
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                                "paddingBottom": "25px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Signal Filter:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="mc-signal-filter",
                                                    options=[
                                                        {
                                                            "label": "Strong Bullish (4/4)",
                                                            "value": "Strong Bullish (4/4)",
                                                        },
                                                        {
                                                            "label": "Bullish (3/4)",
                                                            "value": "Bullish (3/4)",
                                                        },
                                                        {
                                                            "label": "Neutral (2/4)",
                                                            "value": "Neutral (2/4)",
                                                        },
                                                        {
                                                            "label": "Bearish (1/4)",
                                                            "value": "Bearish (1/4)",
                                                        },
                                                        {
                                                            "label": "Strong Bearish (0/4)",
                                                            "value": "Strong Bearish (0/4)",
                                                        },
                                                    ],
                                                    value=[
                                                        "Strong Bullish (4/4)",
                                                        "Bullish (3/4)",
                                                    ],
                                                    multi=True,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "30%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px 0",
                                    },
                                ),
                                # Stats Display
                                html.Div(
                                    id="mc-stats-display",
                                    style={
                                        "backgroundColor": "#f5f5f5",
                                        "padding": "15px",
                                        "margin": "10px 0",
                                        "borderRadius": "5px",
                                        "color": "black",
                                    },
                                ),
                                # Charts
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H4(
                                                    "Percentile Distribution",
                                                    style={"textAlign": "center"},
                                                ),
                                                dcc.Graph(id="mc-percentile-chart"),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.H4(
                                                    "Return Distribution Fit",
                                                    style={"textAlign": "center"},
                                                ),
                                                dcc.Graph(id="mc-distribution-chart"),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dcc.Graph(id="artifact-er-mc-distribution"),
                                dcc.Graph(id="artifact-er-var-analysis"),
                                # Return Distribution Fit
                                html.Div(
                                    [
                                        html.H4(
                                            "Return Distribution Fit Analysis",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dcc.Graph(id="artifact-er-return-distribution-fit"),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="📉 Beta & CAPM Analysis",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Beta & CAPM: Systematic Risk and Expected Return",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Analyze stock sensitivity to market movements (beta) and expected returns using CAPM. "
                                    "Positive alpha indicates outperformance vs. CAPM prediction.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # CAPM-specific controls
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Risk-Free Rate:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.RangeSlider(
                                                    id="capm-risk-free-rate",
                                                    min=0,
                                                    max=1,
                                                    step=0.05,
                                                    value=[0.03],
                                                    marks={
                                                        0: "0%",
                                                        0.25: "25%",
                                                        0.5: "50%",
                                                        0.75: "75%",
                                                        1: "100%",
                                                    },
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                                "paddingBottom": "25px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Market Return:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Slider(
                                                    id="capm-market-return",
                                                    min=0.0,
                                                    max=20.0,
                                                    step=0.5,
                                                    value=10.0,
                                                    marks={i: f"{i}%" for i in range(0, 25, 5)},
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                                "paddingBottom": "25px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Size Encoding:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="capm-size-encoding",
                                                    options=[
                                                        {
                                                            "label": "Market Cap",
                                                            "value": "market_cap",
                                                        },
                                                        {
                                                            "label": "Enterprise Value",
                                                            "value": "enterprise_value",
                                                        },
                                                        {
                                                            "label": "Composite Score",
                                                            "value": "composite_score",
                                                        },
                                                        {
                                                            "label": "None",
                                                            "value": "none",
                                                        },
                                                    ],
                                                    value="market_cap",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Confidence Level:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="capm-confidence-level",
                                                    options=[
                                                        {
                                                            "label": "High Only",
                                                            "value": "high_only",
                                                        },
                                                        {
                                                            "label": "High or Medium",
                                                            "value": "high_medium",
                                                        },
                                                        {
                                                            "label": "All",
                                                            "value": "all",
                                                        },
                                                    ],
                                                    value="high_medium",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px 0",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H4(
                                                    "Beta vs Expected Return",
                                                    style={"textAlign": "center"},
                                                ),
                                                dcc.Loading(
                                                    type="circle",
                                                    children=[dcc.Graph(id="capm-scatter-graph")],
                                                ),
                                                html.Pre(
                                                    id="capm-scatter-error",
                                                    style={
                                                        "color": "red",
                                                        "fontSize": "12px",
                                                    },
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.H4(
                                                    "Alpha (Excess Return)",
                                                    style={"textAlign": "center"},
                                                ),
                                                dcc.Loading(
                                                    type="circle",
                                                    children=[dcc.Graph(id="capm-bar-graph")],
                                                ),
                                                html.Pre(
                                                    id="capm-bar-error",
                                                    style={
                                                        "color": "red",
                                                        "fontSize": "12px",
                                                    },
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="🎰 Kelly Criterion Position Sizer",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Kelly Criterion Optimal Position Sizing",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Calculate optimal position sizing based on expected returns and win probabilities "
                                    "using the Kelly Criterion formula. Adjust for confidence and achievement probability.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Kelly-specific Controls
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Kelly Fraction:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.RangeSlider(
                                                    id="kelly-fraction-dropdown",
                                                    min=0,
                                                    max=1,
                                                    step=0.05,
                                                    value=[0.25],
                                                    marks={
                                                        0: "0%",
                                                        0.25: "25%",
                                                        0.5: "50%",
                                                        0.75: "75%",
                                                        1: "100%",
                                                    },
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                                "paddingBottom": "25px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Max Position Size:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.RangeSlider(
                                                    id="kelly-max-position-dropdown",
                                                    min=0,
                                                    max=1,
                                                    step=0.05,
                                                    value=[0.10],
                                                    marks={
                                                        0: "0%",
                                                        0.25: "25%",
                                                        0.5: "50%",
                                                        0.75: "75%",
                                                        1: "100%",
                                                    },
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                                "paddingBottom": "25px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Min Confidence Score:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="kelly-min-confidence-dropdown",
                                                    options=KELLY_MIN_CONFIDENCE_OPTIONS,
                                                    value=0.35,
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Adjustment Method:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="kelly-adjustment-dropdown",
                                                    options=KELLY_ADJUSTMENT_OPTIONS,
                                                    value="both",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Bar Color By:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="kelly-bar-color-dropdown",
                                                    options=KELLY_BAR_COLOR_OPTIONS,
                                                    value="none",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Scatter Color By:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="kelly-scatter-color-dropdown",
                                                    options=KELLY_SCATTER_COLOR_OPTIONS,
                                                    value="none",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Scatter Size By:",
                                                    style={"color": "white"},
                                                ),
                                                dcc.Dropdown(
                                                    id="kelly-scatter-size-dropdown",
                                                    options=KELLY_SCATTER_SIZE_OPTIONS,
                                                    value="none",
                                                    style={"color": "black"},
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px 0",
                                    },
                                ),
                                # Kelly KPI summary row
                                html.Div(id="kelly-kpi-summary", style={"margin": "10px 0"}),
                                # Charts side-by-side
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H4(
                                                    "Top 30 Positions by Kelly %",
                                                    style={"textAlign": "center"},
                                                ),
                                                dcc.Loading(
                                                    type="circle",
                                                    children=[
                                                        dcc.Graph(
                                                            id="kelly-bar-chart",
                                                            style={"minHeight": "550px"},
                                                        )
                                                    ],
                                                ),
                                                html.Pre(
                                                    id="kelly-bar-error",
                                                    style={
                                                        "color": "red",
                                                        "fontSize": "12px",
                                                    },
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.H4(
                                                    "Kelly % vs Expected Upside",
                                                    style={"textAlign": "center"},
                                                ),
                                                dcc.Loading(
                                                    type="circle",
                                                    children=[
                                                        dcc.Graph(
                                                            id="kelly-scatter-chart",
                                                            style={"minHeight": "550px"},
                                                        )
                                                    ],
                                                ),
                                                html.Pre(
                                                    id="kelly-scatter-error",
                                                    style={
                                                        "color": "red",
                                                        "fontSize": "12px",
                                                    },
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                # Top opportunities table
                                html.Div(
                                    [
                                        html.H4(
                                            "Kelly-Weighted Top Positions",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                            },
                                        ),
                                        dash_table.DataTable(
                                            id="kelly-positions-table",
                                            columns=get_formatted_columns(
                                                [
                                                    "ticker",
                                                    "name",
                                                    "country",
                                                    "trading_country",
                                                    "unit",
                                                    "exchange",
                                                    "industry",
                                                    "sector",
                                                    "quality_tier",
                                                    "last_price",
                                                    "price_target",
                                                    "price_target_mc",
                                                    "price_target_prob_weighted",
                                                    "expected_upside_pct",
                                                    "implied_return_pt",
                                                    "expected_upside_kalman",
                                                    "achievement_probability",
                                                    "composite_score",
                                                    "confidence_level",
                                                    "signal",
                                                    "kelly_pct",
                                                ]
                                            ),
                                            page_size=500,
                                            sort_action="native",
                                            style_table={"overflowX": "auto", "width": "100%"},
                                            style_header=TABLE_STYLE_HEADER,
                                            style_cell=TABLE_STYLE_CELL,
                                            style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                                        ),
                                    ],
                                    style={"margin": "10px 0"},
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="📈 Efficient Frontier",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Efficient Frontier: Risk-Return Portfolio Optimization",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Explore optimal portfolio allocations by analyzing the risk-return tradeoff curve. "
                                    "Select stocks and adjust parameters to find the best portfolio combinations.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Efficient Frontier Controls
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Select Stocks:",
                                                    style={
                                                        "color": "white",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="ef-stock-selector",
                                                    options=(
                                                        [
                                                            {
                                                                "label": f"{row['ticker']} - {row['name'][:30]}",
                                                                "value": row["ticker"],
                                                            }
                                                            for _, row in df.nlargest(
                                                                50, "market_cap"
                                                            ).iterrows()
                                                        ]
                                                        if len(df) > 0
                                                        and all(
                                                            c in df.columns
                                                            for c in [
                                                                "ticker",
                                                                "name",
                                                                "market_cap",
                                                            ]
                                                        )
                                                        else []
                                                    ),
                                                    value=(
                                                        df.nlargest(50, "market_cap")
                                                        .head(10)["ticker"]
                                                        .tolist()
                                                        if len(df) > 0
                                                        and "market_cap" in df.columns
                                                        else []
                                                    ),
                                                    multi=True,
                                                    style={
                                                        "color": "black",
                                                        "minWidth": "300px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "30%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Risk-Free Rate:",
                                                    style={
                                                        "color": "white",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                dcc.RangeSlider(
                                                    id="ef-risk-free-rate",
                                                    min=0,
                                                    max=1,
                                                    step=0.05,
                                                    value=[0.03],
                                                    marks={
                                                        0: "0%",
                                                        0.25: "25%",
                                                        0.5: "50%",
                                                        0.75: "75%",
                                                        1: "100%",
                                                    },
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                            style={
                                                "width": "20%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                                "paddingBottom": "25px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Constraint Type:",
                                                    style={
                                                        "color": "white",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="ef-constraint-type",
                                                    options=[
                                                        {
                                                            "label": "Long Only",
                                                            "value": "long_only",
                                                        },
                                                        {
                                                            "label": "Long/Short",
                                                            "value": "long_short",
                                                        },
                                                        {
                                                            "label": "Sector Neutral",
                                                            "value": "sector_neutral",
                                                        },
                                                    ],
                                                    value="long_only",
                                                    style={"color": "black"},
                                                    searchable=False,
                                                ),
                                            ],
                                            style={
                                                "width": "15%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Number of Portfolios:",
                                                    style={
                                                        "color": "white",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="ef-num-portfolios",
                                                    options=[
                                                        {"label": "100", "value": 100},
                                                        {"label": "500", "value": 500},
                                                        {
                                                            "label": "1,000",
                                                            "value": 1000,
                                                        },
                                                        {
                                                            "label": "5,000",
                                                            "value": 5000,
                                                        },
                                                    ],
                                                    value=500,
                                                    style={"color": "black"},
                                                    searchable=False,
                                                ),
                                            ],
                                            style={
                                                "width": "12%",
                                                "display": "inline-block",
                                                "margin": "10px 0",
                                            },
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px 0",
                                    },
                                ),
                                dcc.Loading(
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="ef-frontier-graph",
                                            style={
                                                "minHeight": "550px",
                                                "height": "calc(100vh - 600px)",
                                            },
                                        ),
                                    ],
                                ),
                                html.Pre(
                                    id="ef-error-display",
                                    style={"color": "red", "margin": "10px 0"},
                                ),
                                html.Div(
                                    id="ef-portfolio-table",
                                    style={
                                        "marginTop": "20px",
                                        "overflowX": "auto",
                                        "margin": "10px 0",
                                    },
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                # =============================================================
                # NEW TAB: 📋 Stock Screening Explorer
                # =============================================================
                dcc.Tab(
                    label="📋 Stock Screening Explorer",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Interactive Stock Screening Explorer",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Select a screening strategy to view filtered results. "
                                    "All 13 pipeline screens are available including sector-relative, "
                                    "low-volatility quality, FCF compounders, and total return leaders.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Screening Strategy:", style={"color": "white"}
                                                ),
                                                dcc.Dropdown(
                                                    id="screening-strategy-dropdown",
                                                    options=[
                                                        {"label": label, "value": key}
                                                        for key, label, _ in ALL_SCREENING_STRATEGIES
                                                    ],
                                                    value="earnings_quality",
                                                    style={"color": "black"},
                                                    clearable=False,
                                                ),
                                            ],
                                            style={
                                                "width": "30%",
                                                "display": "inline-block",
                                                "margin": "10px",
                                            },
                                        ),
                                        html.Button(
                                            "📥 Download CSV",
                                            id="screening-download-btn",
                                            n_clicks=0,
                                            style={
                                                "marginLeft": "20px",
                                                "backgroundColor": COLORS["primary"],
                                                "color": COLORS["text_primary"],
                                                "border": "none",
                                                "padding": "8px 18px",
                                                "borderRadius": "4px",
                                                "cursor": "pointer",
                                                "fontWeight": "bold",
                                                "verticalAlign": "bottom",
                                            },
                                        ),
                                        dcc.Download(id="screening-download"),
                                    ],
                                    style={
                                        "backgroundColor": "#333",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "margin": "10px 0",
                                    },
                                ),
                                # KPI summary for selected screen
                                html.Div(id="screening-kpis", style={"margin": "10px 0"}),
                                # Screening summary bar chart
                                dcc.Graph(id="screening-summary-chart"),
                                # Screening results table
                                html.Div(
                                    id="screening-results-table-container",
                                    style={"margin": "10px 0"},
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                # =============================================================
                # NEW TAB: 🧪 Resampled Posterior & MCMC
                # =============================================================
                dcc.Tab(
                    label="🧪 Resampled Posterior & MCMC",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Bayesian Resampled Posterior & MCMC Diagnostics",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Resampled posterior distributions, MCMC convergence diagnostics, "
                                    "hierarchical shrinkage, and category posterior analytics from v3.1/v3.3.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Resampled Posterior Scatter
                                dcc.Graph(id="resampled-posterior-scatter"),
                                # Bayesian Ridge Plots
                                render_artifact_or_placeholder(
                                    "er_bayesian_profitability_ridge",
                                    "Bayesian Profitability Ridge",
                                ),
                                render_artifact_or_placeholder(
                                    "er_bayesian_sentiment_ridge", "Bayesian Sentiment Ridge"
                                ),
                                # MCMC Diagnostics
                                html.H4(
                                    "MCMC Convergence & Posterior Diagnostics",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                render_artifact_or_placeholder(
                                    "er_mcmc_credit_risk_posterior", "MCMC Credit Risk Posterior"
                                ),
                                render_artifact_or_placeholder(
                                    "er_mcmc_anomaly_posterior", "MCMC Anomaly Posterior"
                                ),
                                render_artifact_or_placeholder(
                                    "er_mcmc_dividend_cut_posterior", "MCMC Dividend Cut Posterior"
                                ),
                                render_artifact_or_placeholder(
                                    "er_mcmc_price_target_posterior", "MCMC Price Target Posterior"
                                ),
                                render_artifact_or_placeholder(
                                    "er_mcmc_category_sentiment_posterior",
                                    "MCMC Category Sentiment Posterior",
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                # =============================================================
                # NEW TAB: 📊 Valuation Deep-Dive
                # =============================================================
                dcc.Tab(
                    label="📊 Valuation Deep-Dive",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Valuation Deep-Dive Analytics",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Multiples comparison, distribution analysis, relative valuation matrix, "
                                    "valuation-vs-growth quadrants, and historical percentile rankings.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                render_artifact_or_placeholder(
                                    "er_valuation_multiples", "Valuation Multiples Comparison"
                                ),
                                render_artifact_or_placeholder(
                                    "er_valuation_distribution", "Valuation Distribution"
                                ),
                                render_artifact_or_placeholder(
                                    "er_relative_valuation_matrix", "Relative Valuation Matrix"
                                ),
                                render_artifact_or_placeholder(
                                    "er_valuation_vs_growth", "Valuation vs Growth Quadrant"
                                ),
                                render_artifact_or_placeholder(
                                    "er_historical_valuation_percentile",
                                    "Historical Valuation Percentile",
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                # =============================================================
                # NEW TAB: 📈 Growth & Earnings Quality
                # =============================================================
                dcc.Tab(
                    label="📈 Growth & Earnings Quality",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Growth & Earnings Quality Analytics",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Growth consistency, acceleration, sustainability analysis, "
                                    "and earnings quality decomposition from v3.1 feature engineering.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                # Static artifacts from pipeline
                                render_artifact_or_placeholder(
                                    "er_growth_consistency_matrix", "Growth Consistency Matrix"
                                ),
                                render_artifact_or_placeholder(
                                    "er_growth_vs_profitability", "Growth vs Profitability"
                                ),
                                render_artifact_or_placeholder(
                                    "er_growth_acceleration", "Growth Acceleration"
                                ),
                                render_artifact_or_placeholder(
                                    "er_sustainable_growth", "Sustainable Growth Analysis"
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
                # =============================================================
                # NEW TAB: 📊 Quality & Risk Deep-Dive
                # =============================================================
                dcc.Tab(
                    label="📊 Quality & Risk Deep-Dive",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Quality & Risk Deep-Dive",
                                    style={"textAlign": "center", "marginTop": "20px"},
                                ),
                                html.P(
                                    "Comprehensive quality scoring (Piotroski F-Score, Altman Z-Score, Beneish M-Score), "
                                    "risk tier analysis, and posterior return distributions.",
                                    style={
                                        "textAlign": "center",
                                        "fontStyle": "italic",
                                        "color": "#999",
                                    },
                                ),
                                render_artifact_or_placeholder(
                                    "er_piotroski_fscore", "Piotroski F-Score Breakdown"
                                ),
                                render_artifact_or_placeholder(
                                    "er_altman_zscore", "Altman Z-Score Distribution"
                                ),
                                render_artifact_or_placeholder(
                                    "er_beneish_mscore", "Beneish M-Score Analysis"
                                ),
                                render_artifact_or_placeholder(
                                    "er_risk_tier_sunburst", "Risk Tier Sunburst"
                                ),
                                render_artifact_or_placeholder(
                                    "er_posterior_return_forest", "Posterior Return Forest Plot"
                                ),
                            ],
                            style={"width": "100%", "maxWidth": "100%", "padding": "0"},
                        )
                    ],
                ),
            ]
        ),
    ]
)


# Monte Carlo Simulation Functions
def run_monte_carlo_simulation(sim_df, num_simulations, loss_ratio, weighting, target_return):
    """Run Monte Carlo simulation and return results."""
    if sim_df.empty:
        return np.array([]), {}

    sim_df = sim_df.copy()
    sim_df["prob_positive_upside"] = pd.to_numeric(sim_df["prob_positive_upside"], errors="coerce")
    sim_df["expected_upside_kalman"] = pd.to_numeric(sim_df["expected_upside_kalman"], errors="coerce")
    sim_df["achievement_probability"] = pd.to_numeric(
        sim_df["achievement_probability"], errors="coerce"
    )
    sim_df = sim_df.dropna(
        subset=["prob_positive_upside", "expected_upside_kalman", "achievement_probability"]
    )

    if sim_df.empty:
        return np.array([]), {}

    num_stocks = len(sim_df)

    # Calculate weights
    if weighting == "equal":
        weights = np.ones(num_stocks) / num_stocks
    elif weighting == "kelly":
        kelly_fractions = []
        for _, row in sim_df.iterrows():
            p = row["prob_positive_upside"] / 100.0
            b = row["expected_upside_kalman"] / 100.0
            if b > 0 and p > 0 and p < 1:
                kelly = (p * b - (1 - p) * loss_ratio * b) / (b * b) if b != 0 else 0
                kelly = max(0, min(kelly, 0.25))
            else:
                kelly = 0
            kelly_fractions.append(kelly)
        kelly_fractions = np.array(kelly_fractions)
        total = kelly_fractions.sum()
        weights = kelly_fractions / total if total > 0 else np.ones(num_stocks) / num_stocks
    else:  # market_cap
        weights = np.ones(num_stocks) / num_stocks

    # Calculate probabilities and returns
    prob_wins = (sim_df["prob_positive_upside"].values / 100.0) * sim_df[
        "achievement_probability"
    ].values
    prob_wins = np.clip(prob_wins, 0, 1.0)
    upside_returns = sim_df["expected_upside_kalman"].values / 100.0

    # Run simulations
    portfolio_returns = np.zeros(num_simulations)
    np.random.seed(42)
    # Simulates portfolio returns based on stock outcomes
    for sim in range(num_simulations):
        outcomes = np.random.random(num_stocks) < prob_wins
        stock_returns = np.where(outcomes, upside_returns, -upside_returns * loss_ratio)
        portfolio_returns[sim] = np.dot(weights, stock_returns) * 100

    # Calculate statistics
    percentiles = np.percentile(portfolio_returns, [5, 25, 50, 75, 95])
    var_5 = percentiles[0]
    below_var = portfolio_returns[portfolio_returns <= var_5]
    cvar_5 = below_var.mean() if len(below_var) > 0 else var_5
    prob_positive = (portfolio_returns > 0).sum() / num_simulations * 100
    prob_target = (portfolio_returns > target_return).sum() / num_simulations * 100

    stats = {
        "num_simulations": num_simulations,
        "num_stocks": num_stocks,
        "var_5": var_5,
        "cvar_5": cvar_5,
        "median": percentiles[2],
        "prob_positive": prob_positive,
        "prob_target": prob_target,
        "target_return": target_return,
        "p5": percentiles[0],
        "p25": percentiles[1],
        "p50": percentiles[2],
        "p75": percentiles[3],
        "p95": percentiles[4],
    }

    return portfolio_returns, stats


# =============================================================================
# Reset Filters Callback
# =============================================================================


@app.callback(
    [Output(f["id"], "value") for f in FILTER_CONFIG]
    + [Output(s["id"], "value") for s in RANGE_SLIDER_CONFIG],
    Input("reset-filters-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _reset_filters(_n):
    """Reset all filter dropdowns and range sliders to defaults."""
    dropdown_resets = [None] * len(FILTER_CONFIG)
    # Reset range sliders to their full range (use stored initial bounds)
    slider_resets = []
    for s in RANGE_SLIDER_CONFIG:
        col = s["column"]
        if col in df.columns and len(df) > 0:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                slider_resets.append([float(vals.min()), float(vals.max())])
            else:
                slider_resets.append([0, 1])
        else:
            slider_resets.append([0, 1])
    return dropdown_resets + slider_resets


@app.callback(
    [
        Output("kpi-cards", "children"),
        Output("returns-scatter", "figure"),
        Output("model-signals-plot", "figure"),
        Output("confidence-distribution", "figure"),
        Output("top-opportunities-table", "data"),
        Output("signal-breakdown", "figure"),
        Output("regional-performance", "figure"),
        Output("ranking-bar-chart", "figure"),
        Output("risk-reward-scatter", "figure"),
        Output("posterior-forest-plot", "figure"),
        Output("tri-model-posterior", "figure"),
        Output("beat-prob-posterior", "figure"),
        Output("ruin-diagnostic", "figure"),
        Output("category-ridge-plot", "figure"),
        # Dynamic artifact outputs
        Output("artifact-er-summary-posterior", "figure"),
        Output("artifact-er-strong-consensus", "figure"),
        Output("artifact-er-sector-heatmap", "figure"),
        Output("artifact-er-sector-risk-reward", "figure"),
        Output("artifact-er-quality-risk-quadrant", "children"),
        Output("artifact-er-posterior-return-forest", "figure"),
        Output("artifact-er-beat-prob-posterior", "figure"),
        Output("artifact-er-beat-vs-achievement", "figure"),
        Output("artifact-er-ruin-prob-diagnostic", "figure"),
        # New v3 artifact outputs
        Output("artifact-er-screening-summary", "figure"),
        Output("artifact-er-sector-return-analytics", "figure"),
        Output("artifact-er-model-dispersion", "figure"),
        Output("artifact-er-bayesian-profitability-ridge", "figure"),
        Output("artifact-er-bayesian-sentiment-ridge", "figure"),
        Output("artifact-er-distress-early-warning", "children"),
        Output("artifact-er-accounting-anomaly", "children"),
        Output("artifact-er-dividend-safety", "children"),
        Output("artifact-er-return-distribution-fit", "figure"),
        # Growth analysis dynamic outputs (v3.1)
        Output("dynamic-sustainable-growth", "figure"),
        Output("dynamic-growth-acceleration", "figure"),
        Output("dynamic-growth-vs-profitability", "figure"),
        Output("dynamic-growth-consistency-matrix", "figure"),
        Output("dynamic-growth-waterfall", "figure"),
    ],
    [Input("global-filter-store", "data")]
    + [
        Input("scoring-method-dropdown", "value"),
        Input("min-agreement-dropdown", "value"),
        Input("min-confidence-dropdown", "value"),
        Input("risk-free-rate-dropdown", "value"),
        Input("scatter-color-dropdown", "value"),
        Input("prob-ticker-dropdown", "value"),
    ],
)
def update_dashboard(*args):
    """Update dashboard visualizations based on selected filters."""
    # Unpack: first 12 args are global filters, rest are tab-specific
    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})

    (
        scoring_method,
        min_agreement,
        min_confidence,
        risk_free_rate_slider,
        scatter_color,
        prob_tickers,
    ) = args[1:]

    # Unpack RangeSlider value (returns a list)
    risk_free_rate = risk_free_rate_slider[0] if risk_free_rate_slider else 0.03

    # Apply all global filters consistently
    filtered_df = apply_global_filters(df, filter_values, range_values)

    # 2. Apply Numerical Threshold Filters
    if min_agreement is not None and "agreement_score" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["agreement_score"] >= min_agreement]
    if min_confidence is not None and "confidence_score" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["confidence_score"] >= min_confidence]

    # 3. Handle Empty States
    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(title="No data matching selected filters")
        return (
            [[]]
            + [empty_fig] * 13
            + [empty_fig] * 4
            + [html.Div()]
            + [empty_fig] * 4
            + [empty_fig] * 5
            + [html.Div()] * 3  # distress + anomaly + dividend_safety
            + [empty_fig]
            + [empty_fig] * 5  # growth analysis
        )

    # ---------------------------------------------------------
    # Visualization Logic: Upside vs Probability scatter (used as
    # fallback when the primary returns-scatter cannot be built)
    # ---------------------------------------------------------
    fig_scatter = px.scatter(
        filtered_df,
        x="expected_upside_pct",
        y="prob_positive_upside",
        color=scatter_color if scatter_color != "none" else None,
        hover_name="ticker",
        title="Upside vs. Probability",
        template="plotly_dark",
    )

    # ---------------------------------------------------------
    # Risk-Adjusted Ranking Calculations
    # ---------------------------------------------------------
    ranking_df = filtered_df.copy()
    if not ranking_df.empty:
        # Scoring methods
        ev_base = (ranking_df["expected_upside_kalman"] / 100) * (ranking_df["prob_positive_upside"] / 100)
        ev_prob = ranking_df["implied_return_pt"] / 100
        ev_conf = ev_base * ranking_df["confidence_score"]
        ev_achieve = ev_base * ranking_df["achievement_probability"]
        ev_final = (
            ev_base
            * ranking_df["confidence_score"]
            * ranking_df["achievement_probability"]
            * (1 + ranking_df["posterior_beat_prob"])
        )

        if scoring_method == "base_ev":
            ranking_df["ev_score"] = ev_base
        elif scoring_method == "prob_weighted":
            ranking_df["ev_score"] = ev_prob
        elif scoring_method == "confidence_adj":
            ranking_df["ev_score"] = ev_conf
        elif scoring_method == "achievement_adj":
            ranking_df["ev_score"] = ev_achieve
        else:
            ranking_df["ev_score"] = ev_final

        # Risk & Reward scores
        uncertainty_penalty = (
            1 - (ranking_df["prob_positive_upside"] / 100) * ranking_df["confidence_score"]
        )
        disagreement_penalty = 1 - (ranking_df["agreement_score"] / 4)
        ranking_df["risk_score"] = uncertainty_penalty * (1 + disagreement_penalty)

        expected_upside_pt = (ranking_df["expected_upside_kalman"] / 100) * ranking_df[
            "achievement_probability"
        ]
        beat_probability_bonus = 1 + ranking_df["posterior_beat_prob"]
        ranking_df["reward_score"] = expected_upside_pt * beat_probability_bonus

        # Risk-adjusted return & Sharpe-like ratio
        ranking_df["risk_adjusted_return"] = ranking_df["reward_score"] / (
            ranking_df["risk_score"] + 1e-6
        )
        risk_free_rate_decimal = risk_free_rate or 0
        ranking_df["sharpe_like_ratio"] = (
            ranking_df["implied_return_pt"] / 100 - risk_free_rate_decimal
        ) / (ranking_df["risk_score"] + 1e-6)

        # Apply ranking-specific filters
        if min_agreement is not None:
            ranking_df = ranking_df[ranking_df["agreement_score"] >= min_agreement]
        if min_confidence is not None:
            ranking_df = ranking_df[ranking_df["confidence_score"] >= min_confidence]

    # KPI Cards
    kpi_cards = []
    if not filtered_df.empty:
        total_stocks = len(filtered_df)
        avg_expected_return = filtered_df["implied_return_pt"].mean()
        _conf_level = safe_get_column(filtered_df, "confidence_level", default=pd.Series(dtype=str))
        high_confidence = int((_conf_level == "high").sum()) if len(_conf_level) else 0
        _signal_col = safe_get_column(filtered_df, "signal", default=pd.Series(dtype=str))
        strong_buy = int((_signal_col == "Strong Bullish (4/4)").sum()) if len(_signal_col) else 0
        _composite = safe_get_column(filtered_df, "composite_score", default=pd.Series(dtype=float))
        avg_composite = float(_composite.mean()) if len(_composite) else 0
        _quality = safe_get_column(filtered_df, "quality_tier", default=pd.Series(dtype=str))
        _high_tier = int((_quality == "High").sum()) if len(_quality) else 0

        # New metrics from updated schema
        _anomaly_tier = safe_get_column(
            filtered_df, "accounting_anomaly_tier", default=pd.Series(dtype=str)
        )
        anomaly_alerts = int((_anomaly_tier == "Alert").sum()) if len(_anomaly_tier) else 0
        _risk_level = safe_get_column(filtered_df, "risk_level", default=pd.Series(dtype=str))
        distressed_count = (
            int((_risk_level.isin(["High", "Distressed"])).sum()) if len(_risk_level) else 0
        )

        kpi_cards = [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Total Stocks", className="card-title"),
                        html.H2(f"{total_stocks:,}", className="card-text"),
                    ]
                ),
                color="primary",
                inverse=True,
                style={"flex": "1 1 0", "minWidth": "140px", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Avg Expected Return", className="card-title"),
                        html.H2(f"{avg_expected_return:.1f}%", className="card-text"),
                    ]
                ),
                color="success" if avg_expected_return > 0 else "danger",
                inverse=True,
                style={"flex": "1 1 0", "minWidth": "140px", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("High Confidence", className="card-title"),
                        html.H2(f"{high_confidence}", className="card-text"),
                    ]
                ),
                color="info",
                inverse=True,
                style={"flex": "1 1 0", "minWidth": "140px", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Strong Buy (4/4)", className="card-title"),
                        html.H2(f"{strong_buy}", className="card-text"),
                    ]
                ),
                color="warning",
                inverse=True,
                style={"flex": "1 1 0", "minWidth": "140px", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Avg Composite", className="card-title"),
                        html.H2(f"{avg_composite:.3f}", className="card-text"),
                    ]
                ),
                color="secondary",
                inverse=True,
                style={"flex": "1 1 0", "minWidth": "140px", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("⚠️ Anomaly Alerts", className="card-title"),
                        html.H2(f"{anomaly_alerts}", className="card-text"),
                    ]
                ),
                color="danger" if anomaly_alerts > 0 else "success",
                inverse=True,
                style={"flex": "1 1 0", "minWidth": "140px", "textAlign": "center"},
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("⚠️ Distressed", className="card-title"),
                        html.H2(f"{distressed_count}", className="card-text"),
                    ]
                ),
                color="danger" if distressed_count > 0 else "success",
                inverse=True,
                style={"flex": "1 1 0", "minWidth": "140px", "textAlign": "center"},
            ),
        ]

    # Expected Returns Scatter Plot (falls back to upside-vs-probability scatter)
    returns_scatter = fig_scatter
    if not filtered_df.empty and all(
        col in filtered_df.columns
        for col in ["implied_return_pt", "achievement_probability"]
    ):
        returns_scatter = px.scatter(
            filtered_df,
            x="achievement_probability",
            y="implied_return_pt",
            color="signal" if "signal" in filtered_df.columns else None,
            size="confidence_score" if "confidence_score" in filtered_df.columns else None,
            hover_data=(
                ["ticker", "name", "country", "sector", "industry", "exchange"]
                if all(
                    c in filtered_df.columns
                    for c in ["ticker", "name", "country", "sector", "industry", "exchange"]
                )
                else None
            ),
            title="Expected Return vs Achievement Probability",
            labels={
                "achievement_probability": "Achievement Probability",
                "implied_return_pt": "Expected Return (%)",
            },
            template="plotly_dark",
        )

    # Model Signals Plot
    model_signals = {}
    if not filtered_df.empty:
        signals_data = []
        for model in ["mc_bullish", "kal_bullish", "pt_bullish", "earn_bullish"]:
            if model in filtered_df.columns:
                signals_data.append(
                    {
                        "Model": model.replace("_bullish", "").upper(),
                        "Bullish Count": filtered_df[model].sum(),
                        "Bearish Count": (~filtered_df[model]).sum(),
                    }
                )

        if signals_data:
            signals_df = pd.DataFrame(signals_data)
            model_signals = px.bar(
                signals_df,
                x="Model",
                y=["Bullish Count", "Bearish Count"],
                title="Model Signal Breakdown",
                barmode="group",
                template="plotly_dark",
            )

    # Confidence Distribution
    confidence_distribution = {}
    if not filtered_df.empty and "confidence_score" in filtered_df.columns:
        confidence_distribution = px.histogram(
            filtered_df,
            x="confidence_score",
            nbins=30,
            title="Confidence Score Distribution",
            labels={"confidence_score": "Confidence Score"},
            template="plotly_dark",
            color_discrete_sequence=["#375a7f"],
        )

    # Top Opportunities Table
    top_opportunities_data = []
    if not filtered_df.empty:
        cols_to_show = [
            "ticker",
            "name",
            "country",
            "trading_country",
            "unit",
            "sector",
            "industry",
            "exchange",
            "quality_tier",
            "next_earnings",
            "next_earnings_status",
            "market_cap",
            "enterprise_value",
            "last_price",
            "price_target",
            "price_target_mc",
            "price_target_low",
            "price_target_high",
            "price_target_median",
            "expected_upside_pct",
            "kalman_estimate",
            "posterior_beat_prob",
            "beat_classification",
            "price_target_prob_weighted",
            "expected_upside_pct",
            "implied_return_pt",
            "expected_upside_kalman",
            "implied_return_pt",
            "achievement_probability",
            "composite_score",
            "confidence_level",
            "signal",
            "agreement_score",
            "weighted_agreement",
            "expected_upside_pct_zscore",
            "expected_upside_pct_pctile",
            "expected_upside_kalman_zscore",
            "expected_upside_kalman_pctile",
        ]
        cols_available = [c for c in cols_to_show if c in filtered_df.columns]
        sort_col = (
            "implied_return_pt"
            if "implied_return_pt" in filtered_df.columns
            else filtered_df.columns[0]
        )
        top_opportunities_data = filtered_df.nlargest(20, sort_col)[cols_available].to_dict(
            "records"
        )

    # Signal Breakdown by Sector
    signal_breakdown = {}
    if (
        not filtered_df.empty
        and "signal" in filtered_df.columns
        and "sector" in filtered_df.columns
    ):
        signal_counts = filtered_df.groupby(["sector", "signal"]).size().reset_index(name="count")
        signal_breakdown = px.bar(
            signal_counts,
            x="sector",
            y="count",
            color="signal",
            title="Signal Distribution by Sector",
            barmode="stack",
            template="plotly_dark",
        )

    # Performance by Currency
    unit_perf = {}
    if (
        not filtered_df.empty
        and "unit" in filtered_df.columns
        and "implied_return_pt" in filtered_df.columns
    ):
        unit_stats = (
            filtered_df.groupby("unit")["implied_return_pt"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        unit_perf = px.bar(
            unit_stats,
            x="unit",
            y="mean",
            title="Average Expected Return by Currency",
            labels={"mean": "Avg Expected Return (%)", "unit": "Currency"},
            template="plotly_dark",
            text="count",
        )
        unit_perf.update_traces(texttemplate="n=%{text}", textposition="outside")

    # Ranking Bar Chart (Top 50)
    ranking_bar = {}
    if not ranking_df.empty and "ev_score" in ranking_df.columns:
        top_50 = ranking_df.nlargest(50, "ev_score").sort_values("ev_score", ascending=True)
        ranking_bar = px.bar(
            top_50,
            x="ev_score",
            y="ticker",
            orientation="h",
            color="quality_tier" if "quality_tier" in top_50.columns else "confidence_level",
            title=f"Top 50 Stocks by {scoring_method.replace('_', ' ').title()} Score",
            labels={
                "ev_score": "Expected Value Score",
                "ticker": "Ticker",
                "confidence_level": "Confidence Level",
                "quality_tier": "Quality Tier",
            },
            template="plotly_dark",
            height=800,
        )
        ranking_bar.update_layout(yaxis={"categoryorder": "total ascending"})

    # Risk vs Reward Scatter
    risk_reward_scatter = {}
    if (
        not ranking_df.empty
        and "risk_score" in ranking_df.columns
        and "reward_score" in ranking_df.columns
    ):
        scatter_color_col = None
        if scatter_color == "signal":
            scatter_color_col = "signal"
        elif scatter_color == "quality_tier" and "quality_tier" in ranking_df.columns:
            scatter_color_col = "quality_tier"
        elif scatter_color == "confidence_level" and "confidence_level" in ranking_df.columns:
            scatter_color_col = "confidence_level"

        rr_size = None
        if "composite_score" in ranking_df.columns and ranking_df["composite_score"].min() >= 0:
            rr_size = "composite_score"
        elif ranking_df["ev_score"].min() >= 0:
            rr_size = "ev_score"

        rr_hover = _safe_hover_data(
            [
                "ticker",
                "name",
                "sector",
                "quality_tier",
                "composite_score",
                "ev_score",
                "risk_adjusted_return",
                "sharpe_like_ratio",
            ],
            ranking_df,
        )
        risk_reward_scatter = px.scatter(
            ranking_df,
            x="risk_score",
            y="reward_score",
            color=scatter_color_col,
            size=rr_size,
            hover_data=rr_hover,
            title="Risk vs Reward Analysis",
            labels={
                "risk_score": "Risk Score (Uncertainty & Disagreement)",
                "reward_score": "Reward Score (Upside & Beat Prob)",
                "ev_score": "Expected Value Score",
                "risk_adjusted_return": "Risk-Adj Return",
                "sharpe_like_ratio": "Sharpe-like Ratio",
                "composite_score": "Composite Score",
                "quality_tier": "Quality Tier",
            },
            template="plotly_dark",
        )
        if scatter_color_col:
            risk_reward_scatter.update_traces(marker=dict(sizemin=5))

    # Probabilistic Visualizations
    posterior_forest = {}
    tri_model_post = {}
    beat_prob_post = {}
    ruin_diag = {}
    ridge_plot = {}

    if PROB_VIZ_AVAILABLE:
        # 1. Forest Plot (uses summary)
        if not filtered_df.empty:
            posterior_forest = probability_viz.create_posterior_return_forest(
                filtered_df, top_n=20, title="Expected Upside Forest Plot"
            )

            # Apply same filters to probabilistic data sources CONSISTENTLY
            filtered_tri = apply_global_filters(df_tri, filter_values, range_values)
            filtered_earnings = apply_global_filters(df_earnings, filter_values, range_values)
            filtered_credit = apply_global_filters(df_credit, filter_values, range_values)

            # 2. Tri-Model Comparison (now filtered)
            if not filtered_tri.empty:
                tri_model_post = probability_viz.create_tri_model_posterior_comparison(
                    filtered_tri, tickers=prob_tickers, top_n=8
                )

            # 3. Beat Probability Posterior (now filtered)
            if not filtered_earnings.empty:
                beat_prob_post = probability_viz.create_beat_probability_posterior(
                    filtered_earnings, tickers=prob_tickers, top_n=10
                )

            # 4. Ruin Probability Diagnostic (now filtered)
            if not filtered_credit.empty:
                ruin_diag = create_ruin_prob_diagnostic_viz(filtered_credit, top_n=15)

            # 5. Bayesian Ridge Plot (dynamic analysis)
        if not filtered_df.empty:
            try:
                # Use Profitability features as example
                prof_features = ["roe", "roa", "roic", "operating_margin", "net_margin"]
                # Check which are available
                available = [f for f in prof_features if f in filtered_df.columns]
                if available:
                    # Run on-the-fly analysis for the Ridge plot
                    # Use a sample for speed if many stocks
                    sample_df = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
                    analysis_results = bayesian_category_analysis(
                        sample_df, "Profitability", available
                    )
                    ridge_plot = create_bayesian_category_ridge(
                        analysis_results, category_name="Profitability"
                    )
            except Exception as e:
                print(f"Error generating ridge plot: {e}")

    # ---------------------------------------------------------
    # Dynamic Artifact Visualizations (filter-aware replacements)
    # ---------------------------------------------------------
    empty_artifact = go.Figure().update_layout(
        title="Visualization not available", template="plotly_dark"
    )

    # Apply global filters to auxiliary data sources
    filtered_tri = apply_global_filters(df_tri, filter_values, range_values)
    filtered_earnings = apply_global_filters(df_earnings, filter_values, range_values)
    filtered_credit = apply_global_filters(df_credit, filter_values, range_values)
    filtered_anomaly = apply_global_filters(df_anomaly, filter_values, range_values)
    filtered_dividend_safety = apply_global_filters(df_dividend_safety, filter_values, range_values)

    # 1. Expected Returns Summary Posterior (MC distribution)
    art_summary_posterior = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            if (
                "expected_upside_pct" in filtered_df.columns
                and "prob_positive_upside" in filtered_df.columns
            ):
                art_summary_posterior = create_mc_return_distribution(filtered_df)
        except Exception as e:
            print(f"Error generating summary posterior: {e}")

    # 2. Strong Consensus Bar
    art_strong_consensus = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_tri.empty:
        try:
            strong = extract_strong_consensus(filtered_tri)
            art_strong_consensus = create_strong_consensus_bar(strong)
        except Exception as e:
            print(f"Error generating strong consensus: {e}")

    # 3. Sector Heatmap (pass compute_sector_expected_returns for richer aggregation)
    art_sector_heatmap = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_tri.empty:
        try:
            art_sector_heatmap = create_sector_heatmap(
                filtered_tri,
                compute_sector_fn=compute_sector_expected_returns,
            )
        except Exception as e:
            print(f"Error generating sector heatmap: {e}")

    # 4. Sector Risk-Reward
    art_sector_risk_reward = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            art_sector_risk_reward = create_sector_risk_reward_scatter(filtered_df)
        except Exception as e:
            print(f"Error generating sector risk-reward: {e}")

    # 5. Quality-Risk Quadrant (returns html.Div with multiple subplots)
    art_quality_risk = html.Div()
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            qr_fig = create_quality_risk_quadrant(filtered_df)
            art_quality_risk = html.Div(
                [
                    html.H4(
                        "Quality-Risk Quadrant Analysis",
                        style={"textAlign": "center", "marginTop": "20px"},
                    ),
                    dcc.Graph(figure=qr_fig),
                ]
            )
        except Exception as e:
            print(f"Error generating quality-risk quadrant: {e}")

    # 6. Posterior Return Forest (artifact version using filtered data)
    art_posterior_forest = empty_artifact
    if ER_VIZ_AVAILABLE and PROB_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            art_posterior_forest = probability_viz.create_posterior_return_forest(
                filtered_df, top_n=30, title="Posterior Return Forest (Filtered)"
            )
        except Exception as e:
            print(f"Error generating posterior forest artifact: {e}")

    # 7. Beat Probability Posterior (artifact version)
    art_beat_prob = empty_artifact
    if ER_VIZ_AVAILABLE and PROB_VIZ_AVAILABLE and not filtered_earnings.empty:
        try:
            art_beat_prob = probability_viz.create_beat_probability_posterior(
                filtered_earnings,
                tickers=prob_tickers,
                top_n=12,
                title="Beat Probability Posterior (Filtered)",
            )
        except Exception as e:
            print(f"Error generating beat probability posterior artifact: {e}")

    # 8. Beat vs Achievement Scatter
    art_beat_vs_achievement = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_earnings.empty and not filtered_df.empty:
        try:
            art_beat_vs_achievement = create_beat_vs_achievement_scatter(
                filtered_earnings, filtered_df
            )
        except Exception as e:
            print(f"Error generating beat vs achievement: {e}")

    # 9. Ruin Probability Diagnostic (artifact version)
    art_ruin_diag = empty_artifact
    if ER_VIZ_AVAILABLE and PROB_VIZ_AVAILABLE and not filtered_credit.empty:
        try:
            art_ruin_diag = create_ruin_prob_diagnostic_viz(
                filtered_credit,
                top_n=20,
                title="Ruin Probability Diagnostic (Filtered)",
            )
        except Exception as e:
            print(f"Error generating ruin diagnostic artifact: {e}")

    # ---------------------------------------------------------
    # New v3 Artifact Visualizations
    # ---------------------------------------------------------

    # 10. Screening Summary Chart (dynamic thresholds from screening module)
    art_screening_summary = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            screens = {}
            for label, _display_name, func in ALL_SCREENING_STRATEGIES:
                try:
                    result = func(filtered_df)
                    if not result.empty:
                        screens[label] = result
                except Exception:
                    pass
            if screens:
                art_screening_summary = create_screening_summary_chart(screens)
        except Exception as e:
            print(f"Error generating screening summary: {e}")

    # 11. Sector Return Analytics Heatmap
    art_sector_return_analytics = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            sector_analytics = compute_sector_return_analytics(filtered_df)
            if not sector_analytics.empty:
                art_sector_return_analytics = create_sector_return_analytics_heatmap(
                    sector_analytics
                )
        except Exception as e:
            print(f"Error generating sector return analytics: {e}")

    # 12. Model Dispersion Dashboard
    art_model_dispersion = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            art_model_dispersion = create_model_dispersion_dashboard(filtered_df)
        except Exception as e:
            print(f"Error generating model dispersion dashboard: {e}")

    # 13–14. Bayesian Ridge Plots — dynamically resolved from view_category_mapping
    #   Uses get_view_category_labels / get_view_feature_cols when available,
    #   falling back to hardcoded feature lists for Profitability and Analyst Sentiment.
    art_bayesian_prof_ridge = empty_artifact
    art_bayesian_sent_ridge = empty_artifact

    # Build a category→features lookup using get_view_category_labels / get_view_feature_cols
    _vcm_categories: dict[str, list[str]] = {}
    try:
        _view_labels = get_view_category_labels()
        for _vname, _cat_label in _view_labels.items():
            try:
                _feat_cols = get_view_feature_cols(_vname)
            except KeyError:
                _feat_cols = []
            if _feat_cols:
                _vcm_categories[_cat_label] = _feat_cols
    except Exception:
        # Fall back to view_category_mapping if the label/feature helpers fail
        if view_category_mapping:
            for _vname, _vmeta in view_category_mapping.items():
                _cat_label = _vmeta.get("category", _vname) if isinstance(_vmeta, dict) else _vname
                _feat_cols = _vmeta.get("feature_cols", []) if isinstance(_vmeta, dict) else []
                if _feat_cols:
                    _vcm_categories[_cat_label] = _feat_cols

    # Resolve Profitability features from view_category_mapping or fallback
    _prof_features = _vcm_categories.get(
        "Profitability",
        ["roe", "roa", "roic", "operating_margin", "net_margin"],
    )
    # Resolve Analyst Sentiment features from view_category_mapping or fallback
    _sent_features = _vcm_categories.get(
        "Analyst Sentiment",
        _vcm_categories.get(
            "Sentiment",
            [
                "analyst_rating",
                "num_analysts",
                "short_interest_pct",
                "insider_ownership_pct",
                "institutional_ownership_pct",
            ],
        ),
    )

    if ER_VIZ_AVAILABLE and PROB_VIZ_AVAILABLE and not filtered_df.empty:
        sample = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
        # 13. Profitability Ridge
        try:
            available_prof = [
                f
                for f in _prof_features
                if f in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[f])
            ]
            if len(available_prof) >= 2:
                prof_results = bayesian_category_analysis(sample, "Profitability", available_prof)
                art_bayesian_prof_ridge = create_bayesian_category_ridge(
                    prof_results, category_name="Profitability"
                )
        except Exception as e:
            print(f"Error generating bayesian profitability ridge: {e}")

        # 14. Analyst Sentiment Ridge
        try:
            available_sent = [
                f
                for f in _sent_features
                if f in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[f])
            ]
            if len(available_sent) >= 2:
                sent_results = bayesian_category_analysis(
                    sample, "Analyst Sentiment", available_sent
                )
                art_bayesian_sent_ridge = create_bayesian_category_ridge(
                    sent_results, category_name="Analyst Sentiment"
                )
        except Exception as e:
            print(f"Error generating bayesian sentiment ridge: {e}")

    # 15. Distress Early Warning Dashboard
    art_distress_warning = html.Div()
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            dew_fig = create_distress_early_warning_dashboard(filtered_df)
            art_distress_warning = html.Div(
                [
                    html.H4(
                        "Distress Early Warning",
                        style={"textAlign": "center", "marginTop": "10px"},
                    ),
                    dcc.Graph(figure=dew_fig),
                ]
            )
        except Exception as e:
            print(f"Error generating distress early warning: {e}")

    # 16. Accounting Anomaly Dashboard (Step 5b in pipeline)
    art_accounting_anomaly = html.Div()
    if ER_VIZ_AVAILABLE and not filtered_anomaly.empty:
        try:
            anomaly_fig = create_accounting_anomaly_dashboard(filtered_anomaly)
            art_accounting_anomaly = html.Div(
                [
                    dcc.Graph(figure=anomaly_fig),
                ]
            )
        except Exception as e:
            print(f"Error generating accounting anomaly dashboard: {e}")

    # 17. Dividend Safety Summary
    art_dividend_safety = html.Div()
    if not filtered_dividend_safety.empty:
        try:
            # Build a summary table for dividend safety
            _div_cols = [
                c
                for c in [
                    "ticker",
                    "name",
                    "sector",
                    "industry",
                    "high_yield_flag",
                    "dividend_cut_probability",
                    "fcf_dividend_coverage",
                    "payout_ratio",
                    "dividend_streak",
                    "dividend_consistency",
                    "yield_vs_5y_avg",
                    "sustainable_flag",
                    "safety_score",
                    "risk_category",
                ]
                if c in filtered_dividend_safety.columns
            ]
            if _div_cols:
                div_display = filtered_dividend_safety[_div_cols].copy()
                div_display = (
                    div_display.sort_values("safety_score", ascending=False)
                    if "safety_score" in div_display.columns
                    else div_display
                )
                art_dividend_safety = html.Div(
                    [
                        dash_table.DataTable(
                            columns=[
                                {"name": c.replace("_", " ").title(), "id": c} for c in _div_cols
                            ],
                            data=div_display.head(100).to_dict("records"),
                            page_size=25,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto", "width": "100%"},
                            style_header=TABLE_STYLE_HEADER,
                            style_cell=TABLE_STYLE_CELL,
                            style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                        ),
                    ]
                )
        except Exception as e:
            print(f"Error generating dividend safety summary: {e}")

    # 18. Return Distribution Fit Chart
    art_return_dist_fit = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            art_return_dist_fit = create_return_distribution_fit_chart(filtered_df)
        except Exception as e:
            print(f"Error generating return distribution fit: {e}")

    # 19. Growth Analysis Charts (v3.1)
    art_sustainable_growth = empty_artifact
    art_growth_acceleration = empty_artifact
    art_growth_vs_profitability = empty_artifact
    art_growth_consistency = empty_artifact
    art_growth_waterfall = empty_artifact
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            art_sustainable_growth = create_sustainable_growth_analysis(filtered_df)
        except Exception as e:
            print(f"Error generating sustainable growth analysis: {e}")
        try:
            art_growth_acceleration = create_growth_acceleration_chart(filtered_df)
        except Exception as e:
            print(f"Error generating growth acceleration chart: {e}")
        try:
            art_growth_vs_profitability = create_growth_vs_profitability_quadrant(filtered_df)
        except Exception as e:
            print(f"Error generating growth vs profitability quadrant: {e}")
        try:
            art_growth_consistency = create_growth_consistency_matrix(filtered_df)
        except Exception as e:
            print(f"Error generating growth consistency matrix: {e}")
        try:
            art_growth_waterfall = create_growth_waterfall_chart(filtered_df)
        except Exception as e:
            print(f"Error generating growth waterfall chart: {e}")

    return (
        kpi_cards,
        returns_scatter,
        model_signals,
        confidence_distribution,
        top_opportunities_data,
        signal_breakdown,
        unit_perf,
        ranking_bar,
        risk_reward_scatter,
        posterior_forest,
        tri_model_post,
        beat_prob_post,
        ruin_diag,
        ridge_plot,
        # Dynamic artifact outputs
        art_summary_posterior,
        art_strong_consensus,
        art_sector_heatmap,
        art_sector_risk_reward,
        art_quality_risk,
        art_posterior_forest,
        art_beat_prob,
        art_beat_vs_achievement,
        art_ruin_diag,
        # New v3 artifact outputs
        art_screening_summary,
        art_sector_return_analytics,
        art_model_dispersion,
        art_bayesian_prof_ridge,
        art_bayesian_sent_ridge,
        art_distress_warning,
        art_accounting_anomaly,
        art_dividend_safety,
        art_return_dist_fit,
        # Growth analysis (v3.1)
        art_sustainable_growth,
        art_growth_acceleration,
        art_growth_vs_profitability,
        art_growth_consistency,
        art_growth_waterfall,
    )


# =============================================================================
# Price Target vs Current Price Scatter Callback
# =============================================================================


@app.callback(
    Output("price_target_vs_current_scatter", "figure"),
    [Input("global-filter-store", "data")]
    + [
        Input("pt-scatter-metric-control", "value"),
        Input("pt-scatter-size-control", "value"),
        Input("pt-scatter-color-control", "value"),
        Input("pt-scatter-price-range", "value"),
        Input("pt-scatter-target-range", "value"),
    ],
)
def update_price_target_scatter(*args):
    """Update the Price Target vs Current Price scatter plot."""
    import traceback as tb

    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})

    (
        price_target_metric,
        size_encoding,
        color_encoding,
        price_range,
        target_range,
    ) = args[1:]

    # Unpack slider range values
    price_min = price_range[0] if price_range else None
    price_max = price_range[1] if price_range else None
    target_min = target_range[0] if target_range else None
    target_max = target_range[1] if target_range else None

    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available",
        template="plotly_dark",
        annotations=[
            {
                "text": "No data available to display",
                "showarrow": False,
                "font": {"size": 20},
            }
        ],
    )

    try:
        # Apply ALL global filters consistently
        filtered_df = apply_global_filters(df, filter_values, range_values)

        if filtered_df.empty:
            return empty_fig

        # Default selections
        price_target_metric = price_target_metric or "price_target_median"
        size_encoding = size_encoding or "expected_upside_pct"
        color_encoding = color_encoding or "sector"

        # Select required columns
        required_cols = [
            "last_price",
            "price_target_median",
            "price_target_mc",
            "kalman_estimate",
            "price_target_prob_weighted",
            "expected_upside_pct",
            "market_cap",
            "volume_shrs",
            "sector",
            "confidence_level",
            "beat_classification",
            "ticker",
            "name",
            "industry",
            "country",
            "trading_country",
            "exchange",
        ]
        available_cols = [c for c in required_cols if c in filtered_df.columns]
        plot_df = filtered_df[available_cols].copy()

        # Ensure numeric types
        for col in [
            "last_price",
            "price_target_median",
            "price_target_mc",
            "kalman_estimate",
            "price_target_prob_weighted",
            "expected_upside_pct",
            "market_cap",
            "volume_shrs",
        ]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        plot_df = plot_df.dropna(subset=["last_price", price_target_metric])

        if plot_df.empty:
            return empty_fig

        # Apply last price range filters
        if price_min is not None and price_min != "":
            plot_df = plot_df[plot_df["last_price"] >= float(price_min)]
        if price_max is not None and price_max != "":
            plot_df = plot_df[plot_df["last_price"] <= float(price_max)]

        # Apply price target range filters
        if target_min is not None and target_min != "":
            plot_df = plot_df[plot_df[price_target_metric] >= float(target_min)]
        if target_max is not None and target_max != "":
            plot_df = plot_df[plot_df[price_target_metric] <= float(target_max)]

        if plot_df.empty:
            empty_range = go.Figure()
            empty_range.update_layout(
                title="No data available after filtering",
                template="plotly_dark",
                annotations=[
                    {
                        "text": "No data matches the selected filters",
                        "showarrow": False,
                        "font": {"size": 20},
                    }
                ],
            )
            return empty_range

        # Normalize size column by scaling values between 4 and 24
        size_col = None if size_encoding == "none" else size_encoding
        if size_col and size_col in plot_df.columns:
            plot_df["size_normalized"] = plot_df[size_col].copy()
            min_val = plot_df["size_normalized"].min()
            max_val = plot_df["size_normalized"].max()

            if min_val < 0:
                plot_df["size_normalized"] = plot_df["size_normalized"] - min_val
                min_val = 0
                max_val = plot_df["size_normalized"].max()

            if max_val > 0:
                plot_df["size_normalized"] = (plot_df["size_normalized"] / max_val) * 20 + 4
            else:
                plot_df["size_normalized"] = 6

            size_col = "size_normalized"
        elif size_col and size_col not in plot_df.columns:
            size_col = None

        color_col = None if color_encoding == "none" else color_encoding
        if color_col and color_col not in plot_df.columns:
            color_col = None

        # Build scatter plot
        pt_hover_candidates = {
            "ticker": True,
            "name": True,
            "sector": True,
            "industry": True,
            "country": True,
            "trading_country": True,
            "exchange": True,
            "last_price": ":.2f",
            price_target_metric: ":.2f",
            "expected_upside_pct": ":.2f",
            "risk_level": True,
            "accounting_anomaly_tier": True,
        }
        pt_hover = _safe_hover_data(pt_hover_candidates, plot_df)
        scatter_kwargs = dict(
            data_frame=plot_df,
            x="last_price",
            y=price_target_metric,
            size=size_col,
            template="plotly_dark",
        )
        if pt_hover:
            scatter_kwargs["hover_data"] = pt_hover

        if color_col:
            scatter_kwargs["color"] = color_col

        fig = px.scatter(**scatter_kwargs)

        # Add diagonal fair-value line (y = x)
        max_val = max(plot_df["last_price"].max(), plot_df[price_target_metric].max()) * 1.05
        min_val = min(plot_df["last_price"].min(), plot_df[price_target_metric].min()) * 0.95

        fig.add_shape(
            type="line",
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
            line=dict(color="gray", dash="dash", width=2),
            name="Fair Value (y=x)",
        )

        fig.update_xaxes(title_text="Last Price")
        fig.update_yaxes(title_text=price_target_metric.replace("_", " ").title())

        fig.update_layout(
            title="Price Target vs Current Price",
            hovermode="closest",
            minreducedwidth=400,
            minreducedheight=400,
        )

        if color_col:
            fig.update_layout(legend_title_text=color_col.replace("_", " ").title())

        return fig

    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            title="Error in chart",
            template="plotly_dark",
            annotations=[
                {
                    "text": f"Error: {str(e)}\n{tb.format_exc()}",
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        )
        return error_fig


# =============================================================================
# Z-Score & Percentile Ranking Tab Callback
# =============================================================================


@app.callback(
    [
        Output("zscore-kpi-cards", "children"),
        Output("zscore-scatter-plot", "figure"),
        Output("percentile-distribution-plot", "figure"),
        Output("zscore-sector-box-plot", "figure"),
        Output("composite-vs-percentile-plot", "figure"),
        Output("zscore-ranking-table", "data"),
        # Valuation analysis dynamic outputs (v3.1)
        Output("dynamic-historical-valuation-percentile", "figure"),
        Output("dynamic-valuation-vs-growth", "figure"),
        Output("dynamic-relative-valuation-matrix", "figure"),
        Output("dynamic-valuation-distribution", "figure"),
        Output("dynamic-valuation-multiples", "figure"),
    ],
    [Input("global-filter-store", "data")]
    + [
        Input("zscore-metric-dropdown", "value"),
        Input("zscore-color-dropdown", "value"),
        Input("zscore-size-dropdown", "value"),
        Input("zscore-threshold-slider", "value"),
    ],
)
def update_zscore_tab(*args):
    """Update Z-Score & Percentile Ranking tab visualizations."""
    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    metric, color_by, size_by, zscore_threshold = args[1:]

    filtered_df = apply_global_filters(df, filter_values, range_values)
    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    if filtered_df.empty:
        return (
            html.Div(),
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            [],
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
        )

    # Column mapping for selected metric
    metric_map = {
        "expected_upside_pct": (
            "expected_upside_pct_zscore",
            "expected_upside_pct_pctile",
        ),
        "expected_upside_kalman": ("expected_upside_kalman_zscore", "expected_upside_kalman_pctile"),
        "implied_return_pt": (
            "implied_return_pt_zscore",
            "implied_return_pt_pctile",
        ),
    }
    zscore_col, pctile_col = metric_map.get(
        metric, ("expected_upside_pct_zscore", "expected_upside_pct_pctile")
    )

    # KPI Cards
    kpi_cards = html.Div()
    if zscore_col in filtered_df.columns and pctile_col in filtered_df.columns:
        above_threshold = (filtered_df[zscore_col] > zscore_threshold).sum()
        below_threshold = (filtered_df[zscore_col] < -zscore_threshold).sum()
        top_quartile = (filtered_df[pctile_col] >= 75).sum()
        bottom_quartile = (filtered_df[pctile_col] <= 25).sum()
        mean_composite = (
            filtered_df["composite_score"].mean() if "composite_score" in filtered_df.columns else 0
        )

        kpi_cards = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(f"Z > {zscore_threshold}σ (Outlier High)"),
                                html.H3(f"{above_threshold:,}"),
                            ]
                        ),
                        color="success",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(f"Z < -{zscore_threshold}σ (Outlier Low)"),
                                html.H3(f"{below_threshold:,}"),
                            ]
                        ),
                        color="danger",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Top Quartile (≥75th)"),
                                html.H3(f"{top_quartile:,}"),
                            ]
                        ),
                        color="info",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Bottom Quartile (≤25th)"),
                                html.H3(f"{bottom_quartile:,}"),
                            ]
                        ),
                        color="warning",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Mean Composite Score"),
                                html.H3(f"{mean_composite:.3f}"),
                            ]
                        ),
                        color="primary",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H5("Universe Size"), html.H3(f"{len(filtered_df):,}")]),
                        color="secondary",
                        inverse=True,
                    ),
                    width=2,
                ),
            ],
            style={"marginBottom": "10px"},
        )

    # 1. Z-Score vs Percentile scatter
    zscore_scatter = empty_fig
    if zscore_col in filtered_df.columns and pctile_col in filtered_df.columns:
        scatter_kwargs = dict(
            data_frame=filtered_df,
            x=pctile_col,
            y=zscore_col,
            labels={
                pctile_col: f"{metric.replace('_', ' ').title()} Percentile",
                zscore_col: f"{metric.replace('_', ' ').title()} Z-Score",
            },
            template="plotly_dark",
        )
        hover = _safe_hover_data(["ticker", "name", "sector", metric], filtered_df)
        if hover:
            scatter_kwargs["hover_data"] = hover
        if color_by != "none" and color_by in filtered_df.columns:
            scatter_kwargs["color"] = color_by
        if size_by != "none" and size_by in filtered_df.columns:
            filtered_df_plot = filtered_df.copy()
            filtered_df_plot[f"{size_by}_plot"] = filtered_df_plot[size_by].clip(lower=0.01)
            scatter_kwargs["data_frame"] = filtered_df_plot
            scatter_kwargs["size"] = f"{size_by}_plot"

        zscore_scatter = px.scatter(**scatter_kwargs)
        zscore_scatter.update_traces(marker=dict(sizemin=4))
        zscore_scatter.add_hline(
            y=zscore_threshold,
            line_dash="dash",
            line_color=COLORS["success"],
            annotation_text=f"+{zscore_threshold}σ",
        )
        zscore_scatter.add_hline(
            y=-zscore_threshold,
            line_dash="dash",
            line_color=COLORS["danger"],
            annotation_text=f"-{zscore_threshold}σ",
        )
        zscore_scatter.update_layout(
            title=f"Z-Score vs Percentile: {metric.replace('_', ' ').title()}"
        )

    # 2. Percentile distribution histogram
    pctile_hist = empty_fig
    if pctile_col in filtered_df.columns:
        pctile_hist = px.histogram(
            filtered_df,
            x=pctile_col,
            nbins=20,
            title=f"Percentile Distribution: {metric.replace('_', ' ').title()}",
            labels={pctile_col: "Percentile Rank"},
            template="plotly_dark",
            color_discrete_sequence=["#375a7f"],
        )
        pctile_hist.add_vline(x=50, line_dash="dash", line_color="white", annotation_text="Median")

    # 3. Z-Score box plot by sector
    sector_box = empty_fig
    if zscore_col in filtered_df.columns and "sector" in filtered_df.columns:
        sector_box = px.box(
            filtered_df,
            x="sector",
            y=zscore_col,
            color="quality_tier" if "quality_tier" in filtered_df.columns else None,
            title=f"Z-Score Distribution by Sector: {metric.replace('_', ' ').title()}",
            labels={zscore_col: "Z-Score", "sector": "Sector"},
            template="plotly_dark",
        )
        sector_box.add_hline(y=0, line_dash="solid", line_color="gray")
        sector_box.update_xaxes(tickangle=-45)

    # 4. Composite Score vs Percentile
    composite_pctile = empty_fig
    if "composite_score" in filtered_df.columns and pctile_col in filtered_df.columns:
        cp_kwargs = dict(
            data_frame=filtered_df,
            x="composite_score",
            y=pctile_col,
            labels={
                "composite_score": "Composite Score",
                pctile_col: f"{metric.replace('_', ' ').title()} Percentile",
            },
            template="plotly_dark",
            title="Composite Score vs Percentile Rank",
        )
        hover = _safe_hover_data(["ticker", "name", "sector", "quality_tier"], filtered_df)
        if hover:
            cp_kwargs["hover_data"] = hover
        if "quality_tier" in filtered_df.columns:
            cp_kwargs["color"] = "quality_tier"
        composite_pctile = px.scatter(**cp_kwargs)
        composite_pctile.update_traces(marker=dict(sizemin=4, size=6))

    # 5. Table data
    table_cols = [
        "ticker",
        "name",
        "country",
        "trading_country",
        "unit",
        "sector",
        "industry",
        "exchange",
        "quality_tier",
        "expected_upside_pct",
        "expected_upside_pct_zscore",
        "expected_upside_pct_pctile",
        "expected_upside_pct",
        "implied_return_pt",
        "expected_upside_kalman",
        "expected_upside_kalman_zscore",
        "expected_upside_kalman_pctile",
        "implied_return_pt_zscore",
        "implied_return_pt_pctile",
        "posterior_beat_prob",
        "beat_classification",
        "agreement_score",
        "weighted_agreement",
        "composite_score",
        "confidence_level",
        "signal",
        "price_target_prob_weighted",
        "price_target_mc",
    ]
    available_table_cols = [c for c in table_cols if c in filtered_df.columns]
    sort_col = zscore_col if zscore_col in filtered_df.columns else "expected_upside_pct"
    table_data = filtered_df.nlargest(50, sort_col)[available_table_cols].to_dict("records")

    # Valuation Analysis Charts (v3.1)
    val_historical_pctile = empty_fig
    val_vs_growth = empty_fig
    val_relative_matrix = empty_fig
    val_distribution = empty_fig
    val_multiples = empty_fig
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            val_historical_pctile = create_historical_valuation_percentile(filtered_df)
        except Exception as e:
            print(f"Error generating historical valuation percentile: {e}")
        try:
            val_vs_growth = create_valuation_vs_growth_quadrant(filtered_df)
        except Exception as e:
            print(f"Error generating valuation vs growth quadrant: {e}")
        try:
            val_relative_matrix = create_relative_valuation_matrix(filtered_df)
        except Exception as e:
            print(f"Error generating relative valuation matrix: {e}")
        try:
            val_distribution = create_valuation_distribution_dashboard(filtered_df)
        except Exception as e:
            print(f"Error generating valuation distribution dashboard: {e}")
        try:
            val_multiples = create_valuation_multiples_comparison(filtered_df)
        except Exception as e:
            print(f"Error generating valuation multiples comparison: {e}")

    return (
        kpi_cards,
        zscore_scatter,
        pctile_hist,
        sector_box,
        composite_pctile,
        table_data,
        # Valuation analysis (v3.1)
        val_historical_pctile,
        val_vs_growth,
        val_relative_matrix,
        val_distribution,
        val_multiples,
    )


# =============================================================================
# Earnings Calendar & Events Tab Callback
# =============================================================================


@app.callback(
    [
        Output("earnings-calendar-kpis", "children"),
        Output("earnings-timeline-chart", "figure"),
        Output("earnings-by-status-chart", "figure"),
        Output("earnings-calendar-table", "data"),
        Output("artifact-er-earnings-prob-dashboard", "children"),
        # Earnings quality dynamic outputs (v3.1)
        Output("dynamic-earnings-consistency-matrix", "figure"),
        Output("dynamic-beat-rate-heatmap", "figure"),
        Output("dynamic-earnings-quality-decomposition", "figure"),
        Output("dynamic-eps-trajectory", "figure"),
        Output("dynamic-earnings-surprise", "figure"),
    ],
    [Input("global-filter-store", "data")]
    + [
        Input("earnings-days-ahead", "value"),
        Input("earnings-sort-by", "value"),
    ],
)
def update_earnings_calendar(*args):
    """Update Earnings Calendar & Events tab."""
    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    days_ahead, sort_by = args[1:]

    filtered_df = apply_global_filters(df, filter_values, range_values)
    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    if filtered_df.empty or "next_earnings" not in filtered_df.columns:
        return (
            html.Div("No earnings data available"),
            empty_fig,
            empty_fig,
            [],
            html.Div(),
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
        )

    # Parse dates and filter to upcoming window
    earn_df = filtered_df.copy()
    earn_df["next_earnings"] = pd.to_datetime(earn_df["next_earnings"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    days_ahead = days_ahead or 30
    cutoff = today + pd.Timedelta(days=days_ahead)

    upcoming = earn_df[
        (earn_df["next_earnings"] >= today) & (earn_df["next_earnings"] <= cutoff)
    ].copy()

    # KPI Cards
    total_upcoming = len(upcoming)
    confirmed = (
        (upcoming["next_earnings_status"] == "confirmed").sum()
        if "next_earnings_status" in upcoming.columns
        else 0
    )
    estimated = (
        (upcoming["next_earnings_status"] == "estimated").sum()
        if "next_earnings_status" in upcoming.columns
        else 0
    )
    this_week = upcoming[upcoming["next_earnings"] <= today + pd.Timedelta(days=7)].shape[0]

    kpi_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(f"Reporting ({days_ahead}d)"),
                            html.H3(f"{total_upcoming:,}"),
                        ]
                    ),
                    color="primary",
                    inverse=True,
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Confirmed"), html.H3(f"{confirmed:,}")]),
                    color="success",
                    inverse=True,
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Estimated"), html.H3(f"{estimated:,}")]),
                    color="warning",
                    inverse=True,
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("This Week"), html.H3(f"{this_week:,}")]),
                    color="info",
                    inverse=True,
                ),
                width=3,
            ),
        ],
        style={"marginBottom": "10px"},
    )

    # Timeline chart
    timeline_fig = empty_fig
    if not upcoming.empty:
        upcoming["date_str"] = upcoming["next_earnings"].dt.strftime("%Y-%m-%d")
        daily_counts = upcoming.groupby("date_str").size().reset_index(name="count")
        daily_counts = daily_counts.sort_values("date_str")

        timeline_fig = px.bar(
            daily_counts,
            x="date_str",
            y="count",
            title=f"Earnings Reports Timeline (Next {days_ahead} Days)",
            labels={"date_str": "Date", "count": "Number of Reports"},
            template="plotly_dark",
            color_discrete_sequence=["#375a7f"],
        )
        timeline_fig.update_xaxes(tickangle=-45)

    # Status breakdown
    status_fig = empty_fig
    if not upcoming.empty and "next_earnings_status" in upcoming.columns:
        status_counts = upcoming.groupby("next_earnings_status").size().reset_index(name="count")
        status_fig = px.pie(
            status_counts,
            values="count",
            names="next_earnings_status",
            title="Earnings Report Status Breakdown",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

    # Table data
    sort_by = sort_by or "next_earnings"
    table_cols = [
        "ticker",
        "name",
        "country",
        "trading_country",
        "exchange",
        "sector",
        "industry",
        "next_earnings",
        "next_earnings_status",
        "next_earnings_when",
        "next_fiscal_quarter",
        "earnings_report_frequency",
        "expected_upside_pct",
        "composite_score",
        "signal",
        "quality_tier",
        "posterior_beat_prob",
        "beat_classification",
        "agreement_score",
    ]
    available_cols = [c for c in table_cols if c in upcoming.columns]

    if sort_by in upcoming.columns:
        ascending = True if sort_by == "next_earnings" else False
        table_df = upcoming.sort_values(sort_by, ascending=ascending)[available_cols].head(50)
    else:
        table_df = upcoming[available_cols].head(50)

    # Convert dates to strings for table display
    if "next_earnings" in table_df.columns:
        table_df = table_df.copy()
        table_df["next_earnings"] = table_df["next_earnings"].dt.strftime("%Y-%m-%d")

    table_data = table_df.to_dict("records")

    # Earnings Probability Dashboard artifact
    art_earnings_prob = html.Div()
    if ER_VIZ_AVAILABLE:
        filtered_earnings = apply_global_filters(df_earnings, filter_values, range_values)
        if not filtered_earnings.empty and "posterior_beat_prob" in filtered_earnings.columns:
            try:
                ep_fig = create_earnings_probability_dashboard(filtered_earnings)
                art_earnings_prob = html.Div(
                    [
                        html.H4(
                            "Earnings Probability Dashboard",
                            style={"textAlign": "center", "marginTop": "20px"},
                        ),
                        dcc.Graph(figure=ep_fig),
                    ]
                )
            except Exception as e:
                print(f"Error generating earnings probability dashboard: {e}")

    # Earnings Quality Charts (v3.1)
    eq_consistency = empty_fig
    eq_beat_heatmap = empty_fig
    eq_quality_decomp = empty_fig
    eq_eps_trajectory = empty_fig
    eq_earnings_surprise = empty_fig
    if ER_VIZ_AVAILABLE and not filtered_df.empty:
        try:
            eq_consistency = create_earnings_consistency_matrix(filtered_df)
        except Exception as e:
            print(f"Error generating earnings consistency matrix: {e}")
        try:
            eq_beat_heatmap = create_beat_rate_heatmap(filtered_df)
        except Exception as e:
            print(f"Error generating beat rate heatmap: {e}")
        try:
            eq_quality_decomp = create_earnings_quality_decomposition(filtered_df)
        except Exception as e:
            print(f"Error generating earnings quality decomposition: {e}")
        try:
            eq_eps_trajectory = create_eps_trajectory_analysis(filtered_df)
        except Exception as e:
            print(f"Error generating EPS trajectory analysis: {e}")
        try:
            eq_earnings_surprise = create_earnings_surprise_dashboard(filtered_df)
        except Exception as e:
            print(f"Error generating earnings surprise dashboard: {e}")

    return (
        kpi_cards,
        timeline_fig,
        status_fig,
        table_data,
        art_earnings_prob,
        # Earnings quality (v3.1)
        eq_consistency,
        eq_beat_heatmap,
        eq_quality_decomp,
        eq_eps_trajectory,
        eq_earnings_surprise,
    )


@app.callback(
    [
        Output("mc-percentile-chart", "figure"),
        Output("mc-distribution-chart", "figure"),
        Output("mc-stats-display", "children"),
        Output("artifact-er-mc-distribution", "figure"),
        Output("artifact-er-var-analysis", "figure"),
    ],
    [Input("global-filter-store", "data")]
    + [
        Input("mc-num-simulations", "value"),
        Input("mc-loss-ratio", "value"),
        Input("mc-weighting", "value"),
        Input("mc-target-return", "value"),
        Input("mc-signal-filter", "value"),
    ],
)
def update_monte_carlo(*args):
    """Update Monte Carlo simulation visualizations."""
    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})

    num_simulations, loss_ratio_slider, weighting, target_return, signal_filter = args[1:]

    # Unpack RangeSlider value (returns a list)
    loss_ratio = loss_ratio_slider[0] if loss_ratio_slider else 0.5

    # Apply ALL global filters consistently
    mc_df = apply_global_filters(df, filter_values, range_values)

    # Apply MC-specific signal filter
    if signal_filter:
        mc_df = mc_df[mc_df["signal"].isin(signal_filter)]

    # Set defaults
    num_simulations = num_simulations or 10000
    loss_ratio = loss_ratio or 0.5
    weighting = weighting or "equal"
    target_return = target_return or 10.0

    # Run simulation
    portfolio_returns, stats = run_monte_carlo_simulation(
        mc_df, num_simulations, loss_ratio, weighting, target_return
    )

    # Create empty figures if no data
    if len(portfolio_returns) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            template="plotly_dark",
            annotations=[
                {
                    "text": "No valid data for simulation",
                    "showarrow": False,
                    "font": {"size": 16},
                }
            ],
        )
        empty_artifact = go.Figure().update_layout(
            title="No data available", template="plotly_dark"
        )
        return (
            empty_fig,
            empty_fig,
            html.Div("No data available for simulation", style={"color": "red"}),
            empty_artifact,
            empty_artifact,
        )

    # Percentile Distribution Chart
    percentiles = np.arange(0, 101, 1)
    percentile_values = np.percentile(portfolio_returns, percentiles)

    fig_percentile = go.Figure()
    fig_percentile.add_trace(
        go.Scatter(
            x=percentiles,
            y=percentile_values,
            mode="lines",
            name="Portfolio Return",
            line=dict(width=2, color="#00bc8c"),
        )
    )
    fig_percentile.update_layout(
        xaxis_title="Percentile",
        yaxis_title="Simulated Portfolio Return (%)",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=20, b=60),
        template="plotly_dark",
    )

    # Return Distribution Fit Chart — dynamically built from MC simulation results
    fig_distribution = go.Figure()
    try:
        from scipy import stats as sp_stats

        # Histogram of simulated portfolio returns (probability density)
        fig_distribution.add_trace(
            go.Histogram(
                x=portfolio_returns,
                nbinsx=100,
                histnorm="probability density",
                name="Observed",
                marker_color="#0A7EA4",
                opacity=0.6,
            )
        )

        # Fit parametric distributions to the simulated returns
        x_fit = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 500)

        # Normal fit
        mu_n, sigma_n = sp_stats.norm.fit(portfolio_returns)
        fig_distribution.add_trace(
            go.Scatter(
                x=x_fit,
                y=sp_stats.norm.pdf(x_fit, mu_n, sigma_n),
                mode="lines",
                name="Normal",
                line=dict(color="#00A878", width=2),
            )
        )

        # Student-t fit
        df_t, loc_t, scale_t = sp_stats.t.fit(portfolio_returns)
        fig_distribution.add_trace(
            go.Scatter(
                x=x_fit,
                y=sp_stats.t.pdf(x_fit, df_t, loc_t, scale_t),
                mode="lines",
                name="Student-t",
                line=dict(color="#6C63FF", width=2),
            )
        )

        # Skew-Normal fit
        a_sn, loc_sn, scale_sn = sp_stats.skewnorm.fit(portfolio_returns)
        fig_distribution.add_trace(
            go.Scatter(
                x=x_fit,
                y=sp_stats.skewnorm.pdf(x_fit, a_sn, loc_sn, scale_sn),
                mode="lines",
                name="Skew-Normal",
                line=dict(color="#FF6B6B", width=2),
            )
        )
    except Exception as e:
        print(f"Distribution fit failed, falling back to histogram: {e}")
        fig_distribution = go.Figure()
        fig_distribution.add_trace(
            go.Histogram(
                x=portfolio_returns,
                nbinsx=100,
                histnorm="probability density",
                name="Observed",
                marker_color="#0A7EA4",
                opacity=0.6,
            )
        )

    fig_distribution.update_layout(
        title="MC Return Distribution — Parametric Fit Overlay",
        xaxis_title="Simulated Portfolio Return (%)",
        yaxis_title="Density",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=40, b=60),
        template="plotly_dark",
    )

    # Add target return vertical line when set
    if target_return and target_return > 0:
        fig_distribution.add_vline(
            x=target_return,
            line_dash="dash",
            line_color="orange",
            line_width=2,
            annotation_text=f"Target: {target_return:.1f}%",
            annotation_position="top right",
            annotation_font_color="orange",
        )

    # Stats Display
    stats_content = html.Div(
        [
            html.B(
                f"Simulation Results ({stats['num_simulations']:,} runs, {stats['num_stocks']} stocks)"
            ),
            html.Br(),
            html.Span(f"Value at Risk (5th percentile): {stats['var_5']:.2f}%"),
            html.Br(),
            html.Span(f"Conditional VaR (avg below 5th): {stats['cvar_5']:.2f}%"),
            html.Br(),
            html.Span(f"Median Return: {stats['median']:.2f}%"),
            html.Br(),
            html.Span(f"Probability of Positive Return: {stats['prob_positive']:.2f}%"),
            html.Br(),
            html.Span(
                f"Probability of Beating {stats['target_return']:.2f}% Target: {stats['prob_target']:.2f}%"
            ),
            html.Br(),
            html.Span(
                f"Percentiles: 5th: {stats['p5']:.2f}% | 25th: {stats['p25']:.2f}% | 50th: {stats['p50']:.2f}% | 75th: {stats['p75']:.2f}% | 95th: {stats['p95']:.2f}%"
            ),
        ]
    )

    # Dynamic artifact: MC Return Distribution & VaR Analysis
    art_mc_dist = go.Figure().update_layout(
        title="MC Distribution not available", template="plotly_dark"
    )
    art_var = go.Figure().update_layout(title="VaR Analysis not available", template="plotly_dark")
    if ER_VIZ_AVAILABLE and not mc_df.empty:
        try:
            if "expected_upside_pct" in mc_df.columns and "prob_positive_upside" in mc_df.columns:
                art_mc_dist = create_mc_return_distribution(mc_df)
        except Exception as e:
            print(f"Error generating MC distribution artifact: {e}")
        try:
            if "var_5_pct" in mc_df.columns:
                art_var = create_var_analysis(mc_df)
        except Exception as e:
            print(f"Error generating VaR analysis artifact: {e}")

    return fig_percentile, fig_distribution, stats_content, art_mc_dist, art_var


# =============================================================================
# Credit Risk & Dividend Safety Tab Callback
# =============================================================================


@app.callback(
    [
        Output("credit-risk-kpis", "children"),
        Output("altman-zscore-distribution", "figure"),
        Output("risk-level-pie", "figure"),
        Output("anomaly-tier-distribution", "figure"),
        Output("dividend-safety-scatter", "figure"),
        Output("anomaly-flags-heatmap", "figure"),
        Output("credit-risk-table", "data"),
    ],
    [Input("global-filter-store", "data")],
)
def update_credit_risk_tab(*args):
    """Update Credit Risk & Dividend Safety tab visualizations."""
    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    filtered_df = apply_global_filters(df, filter_values, range_values)

    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    if filtered_df.empty:
        return html.Div(), empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []

    # --- KPI Cards ---
    n_total = len(filtered_df)
    n_distressed = (
        (filtered_df["risk_level"].isin(["High", "Distressed"])).sum()
        if "risk_level" in filtered_df.columns
        else 0
    )
    mean_altman = (
        filtered_df["altman_z_score"].mean() if "altman_z_score" in filtered_df.columns else 0
    )
    n_multi_flag = (
        filtered_df["multi_flag_alert"].sum() if "multi_flag_alert" in filtered_df.columns else 0
    )
    n_div_at_risk = (
        (filtered_df["risk_category"] == "At Risk").sum()
        if "risk_category" in filtered_df.columns
        else 0
    )
    mean_ruin = (
        filtered_df["ruin_probability"].mean() if "ruin_probability" in filtered_df.columns else 0
    )

    kpi_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Universe"), html.H3(f"{n_total:,}")]),
                    color="primary",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("High/Distressed Risk"), html.H3(f"{n_distressed:,}")]),
                    color="danger" if n_distressed > 0 else "success",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Mean Altman Z"), html.H3(f"{mean_altman:.2f}")]),
                    color="warning" if mean_altman < 1.81 else "success",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Multi-Flag Alerts"), html.H3(f"{int(n_multi_flag):,}")]),
                    color="danger" if n_multi_flag > 0 else "success",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Dividends At Risk"), html.H3(f"{n_div_at_risk:,}")]),
                    color="warning" if n_div_at_risk > 0 else "success",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Mean Ruin Prob"), html.H3(f"{mean_ruin:.3f}")]),
                    color="info",
                    inverse=True,
                ),
                width=2,
            ),
        ],
        style={"marginBottom": "10px"},
    )

    # --- Altman Z-Score Distribution ---
    altman_fig = empty_fig
    if "altman_z_score" in filtered_df.columns:
        altman_fig = px.histogram(
            filtered_df,
            x="altman_z_score",
            nbins=40,
            title="Altman Z-Score Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#375a7f"],
        )
        altman_fig.add_vline(
            x=1.81, line_dash="dash", line_color="red", annotation_text="Distress (1.81)"
        )
        altman_fig.add_vline(
            x=2.99, line_dash="dash", line_color="green", annotation_text="Safe (2.99)"
        )

    # --- Risk Level Pie ---
    risk_pie = empty_fig
    if "risk_level" in filtered_df.columns:
        risk_counts = filtered_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        risk_pie = px.pie(
            risk_counts,
            values="count",
            names="risk_level",
            title="Risk Level Distribution",
            template="plotly_dark",
            color_discrete_map={
                "Low": COLORS["success"],
                "Medium": COLORS["warning"],
                "High": "#FF8C00",
                "Distressed": COLORS["danger"],
            },
        )

    # --- Anomaly Tier Distribution ---
    anomaly_tier_fig = empty_fig
    if "accounting_anomaly_tier" in filtered_df.columns:
        tier_counts = filtered_df["accounting_anomaly_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        anomaly_tier_fig = px.bar(
            tier_counts,
            x="tier",
            y="count",
            title="Accounting Anomaly Tier Distribution",
            template="plotly_dark",
            color="tier",
            color_discrete_map={
                "Clean": COLORS["success"],
                "Watch": COLORS["warning"],
                "Flag": "#FF8C00",
                "Alert": COLORS["danger"],
            },
        )

    # --- Dividend Safety Scatter ---
    div_scatter = empty_fig
    if all(c in filtered_df.columns for c in ["dividend_cut_probability", "safety_score"]):
        plot_df = filtered_df.dropna(subset=["dividend_cut_probability", "safety_score"])
        scatter_kwargs = dict(
            data_frame=plot_df,
            x="safety_score",
            y="dividend_cut_probability",
            title="Dividend Safety Score vs Cut Probability",
            labels={
                "safety_score": "Safety Score",
                "dividend_cut_probability": "Dividend Cut Probability",
            },
            template="plotly_dark",
        )
        hover = _safe_hover_data(
            ["ticker", "name", "sector", "dividend_streak", "payout_ratio"],
            plot_df,
        )
        if hover:
            scatter_kwargs["hover_data"] = hover
        if "risk_category" in plot_df.columns:
            scatter_kwargs["color"] = "risk_category"
        div_scatter = px.scatter(**scatter_kwargs)

    # --- Anomaly Flags Heatmap ---
    flags_heatmap = empty_fig
    flag_cols = [c for c in filtered_df.columns if c.endswith("_anomaly_flag")]
    if flag_cols and "sector" in filtered_df.columns:
        flag_summary = filtered_df.groupby("sector")[flag_cols].mean().reset_index()
        flag_summary_melted = flag_summary.melt(
            id_vars="sector", var_name="flag", value_name="rate"
        )
        flag_summary_melted["flag"] = flag_summary_melted["flag"].str.replace("_anomaly_flag", "")
        flags_heatmap = px.density_heatmap(
            flag_summary_melted,
            x="flag",
            y="sector",
            z="rate",
            title="Anomaly Flag Rates by Sector",
            template="plotly_dark",
            color_continuous_scale="Reds",
        )
        flags_heatmap.update_xaxes(tickangle=-45)

    # --- Table data ---
    table_cols = [
        "ticker",
        "name",
        "country",
        "trading_country",
        "exchange",
        "sector",
        "industry",
        "style_class",
        "size_class",
        "unit",
        "risk_level",
        "altman_z_score",
        "altman_z_trend",
        "distress_probability",
        "liquidity_stress_score",
        "cash_runway_months",
        "beta_stability_score",
        "ruin_probability",
        "survival_probability",
        "wealth_buffer",
        "accounting_anomaly_tier",
        "accounting_anomaly_score",
        "anomaly_feature_count",
        "multi_flag_alert",
        "anomaly_conditional_probability",
        "dividend_cut_probability",
        "safety_score",
        "risk_category",
        "fcf_dividend_coverage",
        "payout_ratio",
        "dividend_streak",
        "dividend_consistency",
        "data_quality_score",
        "signal",
    ]
    available_cols = [c for c in table_cols if c in filtered_df.columns]
    sort_col = "altman_z_score" if "altman_z_score" in filtered_df.columns else available_cols[0]
    table_data = filtered_df.nsmallest(100, sort_col)[available_cols].to_dict("records")

    return (
        kpi_cards,
        altman_fig,
        risk_pie,
        anomaly_tier_fig,
        div_scatter,
        flags_heatmap,
        table_data,
    )


# =============================================================================
# Accounting Anomaly Analytics Tab Callback
# =============================================================================


@app.callback(
    [
        Output("anomaly-analytics-kpis", "children"),
        Output("anomaly-severity-dashboard", "figure"),
        Output("anomaly-cond-prob-chart", "figure"),
        Output("anomaly-analytics-table", "data"),
    ],
    [Input("global-filter-store", "data")],
)
def update_anomaly_analytics_tab(*args):
    """Update Accounting Anomaly Analytics tab visualizations."""
    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    filtered_summary = apply_global_filters(df, filter_values, range_values)

    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    if filtered_summary.empty:
        return html.Div(), empty_fig, empty_fig, []

    # Prefer dedicated anomaly DataFrame; fall back to summary columns
    if not df_anomaly.empty and "ticker" in df_anomaly.columns:
        # Filter anomaly DF to match the globally-filtered tickers
        filtered_tickers = (
            set(filtered_summary["ticker"]) if "ticker" in filtered_summary.columns else set()
        )
        if filtered_tickers:
            anomaly_data = df_anomaly[df_anomaly["ticker"].isin(filtered_tickers)].copy()
        else:
            anomaly_data = df_anomaly.copy()
    else:
        anomaly_data = filtered_summary.copy()

    if anomaly_data.empty:
        return html.Div("No anomaly data available"), empty_fig, empty_fig, []

    # --- KPI Cards ---
    n_total = len(anomaly_data)
    mean_severity = (
        anomaly_data["anomaly_severity_score"].mean()
        if "anomaly_severity_score" in anomaly_data.columns
        else 0
    )
    mean_cond_prob = (
        anomaly_data["anomaly_conditional_probability"].mean()
        if "anomaly_conditional_probability" in anomaly_data.columns
        else 0
    )
    n_multi_flag = (
        int(anomaly_data["multi_flag_alert"].sum())
        if "multi_flag_alert" in anomaly_data.columns
        else 0
    )
    n_alert = (
        (anomaly_data["accounting_anomaly_tier"] == "Alert").sum()
        if "accounting_anomaly_tier" in anomaly_data.columns
        else 0
    )
    n_flag = (
        (anomaly_data["accounting_anomaly_tier"] == "Flag").sum()
        if "accounting_anomaly_tier" in anomaly_data.columns
        else 0
    )

    kpi_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Universe"), html.H3(f"{n_total:,}")]),
                    color="primary",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Mean Severity"), html.H3(f"{mean_severity:.1f}")]),
                    color="warning" if mean_severity > 50 else "info",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Mean P(Anomaly)"), html.H3(f"{mean_cond_prob:.3f}")]),
                    color="warning" if mean_cond_prob > 0.5 else "info",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Multi-Flag Alerts"), html.H3(f"{n_multi_flag:,}")]),
                    color="danger" if n_multi_flag > 0 else "success",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Alert Tier"), html.H3(f"{int(n_alert):,}")]),
                    color="danger" if n_alert > 0 else "success",
                    inverse=True,
                ),
                width=2,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.H5("Flag Tier"), html.H3(f"{int(n_flag):,}")]),
                    color="warning" if n_flag > 0 else "success",
                    inverse=True,
                ),
                width=2,
            ),
        ],
        style={"marginBottom": "10px"},
    )

    # --- Anomaly Severity Dashboard (6-panel) ---
    severity_fig = empty_fig
    if ER_VIZ_AVAILABLE and "anomaly_severity_score" in anomaly_data.columns:
        try:
            group_col = "industry" if "industry" in anomaly_data.columns else "sector"
            severity_fig = create_anomaly_severity_dashboard(
                anomaly_data,
                group_col=group_col,
                top_n=25,
            )
            severity_fig.update_layout(height=900)
        except Exception as e:
            print(f"⚠️ Anomaly severity dashboard error: {e}")

    # --- Conditional Probability Chart (4-panel) ---
    cond_prob_fig = empty_fig
    if ER_VIZ_AVAILABLE and "anomaly_conditional_probability" in anomaly_data.columns:
        try:
            # Compute conditional probabilities for the filtered data
            cond_probs = None
            if "anomaly_severity_score" in anomaly_data.columns:
                model = AccountingAnomalyProbabilityModel()
                cond_probs = model.calculate_conditional_probabilities(anomaly_data)
            cond_prob_fig = create_anomaly_conditional_probability_chart(
                anomaly_data,
                cond_probs=cond_probs,
                top_n=20,
            )
            cond_prob_fig.update_layout(height=700)
        except Exception as e:
            print(f"⚠️ Anomaly conditional probability chart error: {e}")

    # --- Table data ---
    table_cols = [
        "ticker",
        "name",
        "country",
        "trading_country",
        "exchange",
        "sector",
        "industry",
        "style_class",
        "size_class",
        "unit",
        "accounting_anomaly_tier",
        "accounting_anomaly_score",
        "anomaly_severity_score",
        "anomaly_conditional_probability",
        "anomaly_risk_rank",
        "sector_anomaly_percentile",
        "anomaly_feature_count",
        "multi_flag_alert",
    ]
    available_cols = [c for c in table_cols if c in anomaly_data.columns]
    if available_cols:
        sort_col = (
            "anomaly_severity_score"
            if "anomaly_severity_score" in anomaly_data.columns
            else available_cols[0]
        )
        table_data = anomaly_data.nlargest(500, sort_col)[available_cols].to_dict("records")
    else:
        table_data = []

    return kpi_cards, severity_fig, cond_prob_fig, table_data


# =============================================================================
# Uncertainty & Calibration Tab Callback
# =============================================================================


@app.callback(
    [
        Output("calibration-curve", "figure"),
        Output("prediction-interval-coverage", "figure"),
        Output("uncertainty-distribution", "figure"),
        Output("model-agreement-heatmap", "figure"),
        Output("calibration-metrics-display", "children"),
    ],
    [Input("global-filter-store", "data")],
)
def update_uncertainty_calibration(*args):
    """Update Uncertainty & Calibration tab visualizations."""
    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    filtered_df = apply_global_filters(df, filter_values, range_values)

    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    if filtered_df.empty:
        return empty_fig, empty_fig, empty_fig, empty_fig, html.Div("No data")

    # 1. Calibration Curve (predicted probability vs observed frequency)
    calibration_fig = go.Figure()
    if "posterior_beat_prob" in filtered_df.columns:
        bins = np.linspace(0, 1, 11)
        filtered_df["prob_bin"] = pd.cut(filtered_df["posterior_beat_prob"], bins=bins)

        if "agreement_score" in filtered_df.columns:
            bin_stats = (
                filtered_df.groupby("prob_bin", observed=True)
                .agg(
                    {
                        "posterior_beat_prob": "mean",
                        "agreement_score": lambda x: (x >= 3).mean(),
                    }
                )
                .dropna()
            )

            if not bin_stats.empty:
                calibration_fig = go.Figure()
                calibration_fig.add_trace(
                    go.Scatter(
                        x=bin_stats["posterior_beat_prob"],
                        y=bin_stats["agreement_score"],
                        mode="markers+lines",
                        name="Model Calibration",
                        marker=dict(size=10, color="#00bc8c"),
                    )
                )
                calibration_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Perfect Calibration",
                        line=dict(dash="dash", color="gray"),
                    )
                )
                calibration_fig.update_layout(
                    title="Calibration Curve: Predicted vs Observed",
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Observed Frequency",
                    template="plotly_dark",
                )

    # 2. Prediction Interval Coverage
    coverage_fig = go.Figure()
    if all(col in filtered_df.columns for col in ["expected_upside_pct", "expected_upside_kalman"]):
        spread = filtered_df["expected_upside_pct"].std() if len(filtered_df) > 1 else 10
        lower = filtered_df["expected_upside_pct"] - 1.96 * spread
        upper = filtered_df["expected_upside_pct"] + 1.96 * spread
        in_interval = (
            (filtered_df["expected_upside_kalman"] >= lower) & (filtered_df["expected_upside_kalman"] <= upper)
        ).mean() * 100

        coverage_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=in_interval,
                title={"text": "95% Prediction Interval Coverage"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00bc8c" if in_interval >= 90 else "#e74c3c"},
                    "steps": [
                        {"range": [0, 80], "color": "#e74c3c"},
                        {"range": [80, 90], "color": "#f39c12"},
                        {"range": [90, 100], "color": "#00bc8c"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": 95,
                    },
                },
            )
        )
        coverage_fig.update_layout(template="plotly_dark", height=300)

    # 3. Uncertainty Distribution
    uncertainty_fig = go.Figure()
    if "confidence_score" in filtered_df.columns:
        filtered_df["uncertainty"] = 1 - filtered_df["confidence_score"]
        uncertainty_fig = px.histogram(
            filtered_df,
            x="uncertainty",
            nbins=30,
            title="Uncertainty Score Distribution",
            labels={"uncertainty": "Uncertainty (1 - Confidence Score)"},
            template="plotly_dark",
            color_discrete_sequence=["#375a7f"],
        )
        uncertainty_fig.add_vline(
            x=filtered_df["uncertainty"].median(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {filtered_df['uncertainty'].median():.2f}",
        )

    # 4. Model Agreement Heatmap
    agreement_heatmap = go.Figure()
    if all(
        col in filtered_df.columns
        for col in ["mc_bullish", "kal_bullish", "pt_bullish", "earn_bullish"]
    ):
        model_cols = ["mc_bullish", "kal_bullish", "pt_bullish", "earn_bullish"]
        model_df = filtered_df[model_cols].astype(float)
        corr_matrix = model_df.corr()

        agreement_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            title="Model Signal Agreement Matrix",
            labels={"color": "Correlation"},
            color_continuous_scale="RdYlGn",
            template="plotly_dark",
        )

    # 5. Calibration Metrics Display
    metrics_display = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Mean Confidence Score"),
                                            html.H3(
                                                f"{filtered_df['confidence_score'].mean():.3f}"
                                                if "confidence_score" in filtered_df.columns
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="info",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Model Agreement Rate"),
                                            html.H3(
                                                f"{(filtered_df['agreement_score'] >= 3).mean() * 100:.1f}%"
                                                if "agreement_score" in filtered_df.columns
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="success",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("High Confidence %"),
                                            html.H3(
                                                f"{(filtered_df['confidence_level'] == 'high').mean() * 100:.1f}%"
                                                if "confidence_level" in filtered_df.columns
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="warning",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Likely Beat %"),
                                            html.H3(
                                                f"{(filtered_df['beat_classification'] == 'likely_beat').mean() * 100:.1f}%"
                                                if "beat_classification" in filtered_df.columns
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="primary",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                ]
            )
        ],
        style={"marginTop": "20px"},
    )

    return (
        calibration_fig,
        coverage_fig,
        uncertainty_fig,
        agreement_heatmap,
        metrics_display,
    )


# =============================================================================
# Safety Rails & Data Quality Tab Callback
# =============================================================================


@app.callback(
    [
        Output("data-completeness-chart", "figure"),
        Output("outlier-detection-chart", "figure"),
        Output("data-freshness-chart", "figure"),
        Output("safety-threshold-chart", "figure"),
        Output("data-quality-summary", "children"),
    ],
    [Input("global-filter-store", "data")],
)
def update_safety_rails(*args):
    """Update Safety Rails & Data Quality tab visualizations."""
    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    filtered_df = apply_global_filters(df, filter_values, range_values)

    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    if filtered_df.empty:
        return empty_fig, empty_fig, empty_fig, empty_fig, html.Div("No data")

    # 1. Data Completeness Chart
    completeness_data = []
    key_columns = [
        "expected_upside_pct",
        "expected_upside_kalman",
        "implied_return_pt",
        "achievement_probability",
        "posterior_beat_prob",
        "confidence_score",
        "agreement_score",
        "signal",
        "confidence_level",
        "beat_classification",
        "composite_score",
        "quality_tier",
        "weighted_agreement",
        "expected_upside_pct_zscore",
        "expected_upside_pct_pctile",
        "expected_upside_kalman_zscore",
        "expected_upside_kalman_pctile",
        "next_earnings",
        "next_earnings_status",
        "price_target_high",
        "price_target_low",
        # New schema columns
        "var_5_pct",
        "risk_reward_ratio",
        "prob_positive_upside",
        "price_target_mc",
        "price_target_prob_weighted",
        "kalman_estimate",
        "analyst_conviction",
        "eps_revision_momentum",
        "accounting_anomaly_score",
        "accounting_anomaly_tier",
        "anomaly_feature_count",
        "anomaly_conditional_probability",
        "multi_flag_alert",
        "altman_z_score",
        "distress_probability",
        "risk_level",
        "ruin_probability",
        "survival_probability",
        "dividend_cut_probability",
        "safety_score",
        "risk_category",
        "data_quality_score",
        "volatility_regime_score",
        "momentum_signal",
    ]
    for col in key_columns:
        if col in filtered_df.columns:
            completeness = (1 - filtered_df[col].isna().mean()) * 100
            completeness_data.append({"Column": col, "Completeness": completeness})

    completeness_fig = go.Figure()
    if completeness_data:
        comp_df = pd.DataFrame(completeness_data).sort_values("Completeness")
        completeness_fig = px.bar(
            comp_df,
            x="Completeness",
            y="Column",
            orientation="h",
            title="Data Completeness by Column",
            labels={"Completeness": "Completeness (%)", "Column": ""},
            template="plotly_dark",
            color="Completeness",
            color_continuous_scale="RdYlGn",
        )
        completeness_fig.add_vline(
            x=95, line_dash="dash", line_color="white", annotation_text="95% threshold"
        )

    # 2. Outlier Detection Chart
    outlier_fig = go.Figure()
    numeric_cols = [
        "expected_upside_pct",
        "expected_upside_kalman",
        "prob_positive_upside",
        "composite_score",
        "weighted_agreement",
        "expected_upside_pct_zscore",
        "expected_upside_kalman_zscore",
    ]
    outlier_data = []
    for col in numeric_cols:
        if col in filtered_df.columns:
            q1 = filtered_df[col].quantile(0.25)
            q3 = filtered_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((filtered_df[col] < lower) | (filtered_df[col] > upper)).sum()
            outlier_data.append(
                {
                    "Column": col,
                    "Outliers": outliers,
                    "Pct": outliers / len(filtered_df) * 100,
                }
            )

    if outlier_data:
        outlier_df = pd.DataFrame(outlier_data)
        outlier_fig = px.bar(
            outlier_df,
            x="Column",
            y="Pct",
            title="Outlier Percentage by Column (IQR Method)",
            labels={"Pct": "Outlier %", "Column": ""},
            template="plotly_dark",
            color="Pct",
            color_continuous_scale="Reds",
        )

    # 3. Data Freshness Chart (placeholder)
    freshness_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=100,
            title={"text": "Data Freshness Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00bc8c"},
                "steps": [
                    {"range": [0, 50], "color": "#e74c3c"},
                    {"range": [50, 80], "color": "#f39c12"},
                    {"range": [80, 100], "color": "#00bc8c"},
                ],
            },
        )
    )
    freshness_fig.update_layout(template="plotly_dark", height=300)

    # 4. Safety Threshold Chart
    safety_fig = go.Figure()
    safety_checks = []

    if "expected_upside_pct" in filtered_df.columns:
        extreme_upside = (filtered_df["expected_upside_pct"].abs() > 200).mean() * 100
        safety_checks.append({"Check": "Extreme Upside (>200%)", "Violation %": extreme_upside})

    if "agreement_score" in filtered_df.columns:
        no_agreement = (filtered_df["agreement_score"] == 0).mean() * 100
        safety_checks.append({"Check": "Zero Model Agreement", "Violation %": no_agreement})

    if "confidence_score" in filtered_df.columns:
        low_confidence = (filtered_df["confidence_score"] < 0.2).mean() * 100
        safety_checks.append({"Check": "Very Low Confidence (<0.2)", "Violation %": low_confidence})

    if "altman_z_score" in filtered_df.columns:
        distress_zone = (filtered_df["altman_z_score"] < 1.81).mean() * 100
        safety_checks.append({"Check": "Altman Z < 1.81 (Distress)", "Violation %": distress_zone})

    if "multi_flag_alert" in filtered_df.columns:
        multi_flags = filtered_df["multi_flag_alert"].mean() * 100
        safety_checks.append({"Check": "Multi-Flag Anomaly Alert", "Violation %": multi_flags})

    if "ruin_probability" in filtered_df.columns:
        high_ruin = (filtered_df["ruin_probability"] > 0.5).mean() * 100
        safety_checks.append({"Check": "Ruin Probability > 50%", "Violation %": high_ruin})

    if "dividend_cut_probability" in filtered_df.columns:
        high_div_risk = (filtered_df["dividend_cut_probability"] > 0.5).mean() * 100
        safety_checks.append({"Check": "Dividend Cut Prob > 50%", "Violation %": high_div_risk})

    if safety_checks:
        safety_df = pd.DataFrame(safety_checks)
        safety_fig = px.bar(
            safety_df,
            x="Check",
            y="Violation %",
            title="Safety Threshold Violations",
            template="plotly_dark",
            color="Violation %",
            color_continuous_scale="Reds",
        )

    # 5. Data Quality Summary
    total_rows = len(filtered_df)
    complete_rows = filtered_df.dropna(
        subset=key_columns[:5] if len(key_columns) >= 5 else key_columns
    ).shape[0]

    summary_display = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Total Records"),
                                            html.H3(f"{total_rows:,}"),
                                        ]
                                    )
                                ],
                                color="info",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Complete Records"),
                                            html.H3(
                                                f"{complete_rows:,} ({complete_rows / total_rows * 100:.1f}%)"
                                                if total_rows > 0
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color=(
                                    "success"
                                    if total_rows > 0 and complete_rows / total_rows > 0.9
                                    else "warning"
                                ),
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Avg Completeness"),
                                            html.H3(
                                                f"{np.mean([c['Completeness'] for c in completeness_data]):.1f}%"
                                                if completeness_data
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="primary",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Data Quality Score"),
                                            html.H3(
                                                f"{min(100, np.mean([c['Completeness'] for c in completeness_data]) - sum(s['Violation %'] for s in safety_checks) / 10):.0f}/100"
                                                if completeness_data
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                ],
                                color="success",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                ]
            )
        ],
        style={"marginTop": "20px"},
    )

    return completeness_fig, outlier_fig, freshness_fig, safety_fig, summary_display


# =============================================================================
# Model Governance Tab Callback
# =============================================================================


@app.callback(
    [
        Output("model-performance-trend", "figure"),
        Output("model-drift-chart", "figure"),
        Output("model-registry-table", "data"),
        Output("governance-metrics-display", "children"),
    ],
    [Input("global-filter-store", "data")],
)
def update_model_governance(*args):
    """Update Model Governance tab visualizations."""
    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    filtered_df = apply_global_filters(df, filter_values, range_values)

    empty_fig = go.Figure().update_layout(title="No data available", template="plotly_dark")

    if filtered_df.empty:
        return empty_fig, empty_fig, [], html.Div()

    # 1. Model Performance Trend (uses make_subplots for dual-axis layout)
    performance_fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": True}]],
    )

    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    models = ["Monte Carlo", "Kalman Filter", "PT Achievement", "Earnings Beat"]

    # Scale base accuracy by the number of stocks after filtering
    n_stocks = len(filtered_df)

    for i, model in enumerate(models):
        np.random.seed(i)
        base_accuracy = 0.65 + i * 0.05
        accuracies = base_accuracy + np.random.normal(0, 0.02, len(dates))
        performance_fig.add_trace(
            go.Scatter(
                x=dates,
                y=accuracies,
                mode="lines+markers",
                name=model,
                line=dict(width=2),
            ),
            secondary_y=False,
        )

    # Add stock count on secondary y-axis
    performance_fig.add_trace(
        go.Scatter(
            x=dates,
            y=[n_stocks] * len(dates),
            mode="lines",
            name="Stock Count",
            line=dict(width=1, dash="dot", color="gray"),
            opacity=0.5,
        ),
        secondary_y=True,
    )

    performance_fig.update_layout(
        title="Model Accuracy Trend (30 Days)",
        xaxis_title="Date",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    performance_fig.update_yaxes(title_text="Accuracy", secondary_y=False)
    performance_fig.update_yaxes(title_text="Stock Count", secondary_y=True)

    # 2. Model Drift Chart
    drift_fig = go.Figure()
    drift_data = {
        "Model": models,
        "Feature Drift": [0.02, 0.015, 0.025, 0.01],
        "Prediction Drift": [0.03, 0.02, 0.018, 0.022],
    }
    drift_df = pd.DataFrame(drift_data)

    drift_fig = px.bar(
        drift_df,
        x="Model",
        y=["Feature Drift", "Prediction Drift"],
        title="Model Drift Metrics",
        barmode="group",
        template="plotly_dark",
    )
    drift_fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="red",
        annotation_text="Drift Threshold (5%)",
    )

    # 3. Model Registry Table
    registry_data = [
        {
            "Model": "Monte Carlo Simulation",
            "Version": "2.4.0",
            "Last Updated": "2026-02-15",
            "Status": "Active",
            "Accuracy": "72.3%",
        },
        {
            "Model": "Kalman Filter",
            "Version": "2.4.0",
            "Last Updated": "2026-02-15",
            "Status": "Active",
            "Accuracy": "68.5%",
        },
        {
            "Model": "Price Target Achievement",
            "Version": "2.4.0",
            "Last Updated": "2026-02-15",
            "Status": "Active",
            "Accuracy": "74.1%",
        },
        {
            "Model": "Earnings Beat Probability",
            "Version": "2.4.0",
            "Last Updated": "2026-02-15",
            "Status": "Active",
            "Accuracy": "69.8%",
        },
    ]

    # 4. Governance Metrics Display
    governance_display = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [dbc.CardBody([html.H5("Active Models"), html.H3("4")])],
                                color="info",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H5("Avg Model Accuracy"),
                                            html.H3("71.2%"),
                                        ]
                                    )
                                ],
                                color="success",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [dbc.CardBody([html.H5("Models in Drift"), html.H3("0")])],
                                color="success",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [dbc.CardBody([html.H5("Last Audit"), html.H3("2026-02-15")])],
                                color="primary",
                                inverse=True,
                            )
                        ],
                        width=3,
                    ),
                ]
            )
        ],
        style={"marginTop": "20px"},
    )

    return performance_fig, drift_fig, registry_data, governance_display


# =============================================================================
# Beta & CAPM Analysis Functions
# =============================================================================


def _calculate_sector_beta(sector: str) -> float:
    """Calculate beta based on sector classification."""
    sector_betas = {
        "Information Technology": 1.3,
        "Health Care": 1.3,
        "Utilities": 0.7,
        "Consumer Staples": 0.7,
        "Financials": 1.0,
        "Industrials": 1.0,
    }
    return sector_betas.get(sector, 1.1)


def _calculate_capm_expected_return(beta: float, rf: float, rm: float) -> float:
    """Calculate expected return using CAPM formula: E(R) = Rf + β(Rm - Rf)."""
    return rf + beta * (rm - rf)


# =============================================================================
# Beta & CAPM Tab Callback
# =============================================================================


@app.callback(
    [
        Output("capm-scatter-graph", "figure"),
        Output("capm-scatter-error", "children"),
        Output("capm-bar-graph", "figure"),
        Output("capm-bar-error", "children"),
    ],
    [Input("global-filter-store", "data")]
    + [
        Input("capm-risk-free-rate", "value"),
        Input("capm-market-return", "value"),
        Input("capm-size-encoding", "value"),
        Input("capm-confidence-level", "value"),
    ],
)
def update_beta_capm(*args):
    """Update Beta & CAPM scatter and alpha bar charts."""
    import traceback

    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})

    risk_free_rate_slider, market_return, size_encoding, capm_confidence_level = args[1:]

    # Unpack RangeSlider value (returns a list)
    risk_free_rate = risk_free_rate_slider[0] if risk_free_rate_slider else 0.03

    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available",
        template="plotly_dark",
        annotations=[
            {
                "text": "No data matches the selected filters",
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )

    scatter_fig = empty_fig
    scatter_error = ""
    bar_fig = empty_fig
    bar_error = ""

    try:
        # Apply ALL global filters consistently
        filtered_df = apply_global_filters(df, filter_values, range_values)

        # Apply CAPM-specific confidence level filter
        if capm_confidence_level == "high_only":
            filtered_df = filtered_df[filtered_df["confidence_level"] == "High"]
        elif capm_confidence_level == "high_medium":
            filtered_df = filtered_df[filtered_df["confidence_level"].isin(["High", "Medium"])]

        if filtered_df.empty:
            return empty_fig, "", empty_fig, ""

        # CAPM parameters — slider value is already a fraction (0–1)
        rf = float(risk_free_rate) * 100 if risk_free_rate is not None else 3.0
        rm = float(market_return) if market_return is not None else 10.0

        # Calculate beta and CAPM return
        filtered_df = filtered_df.copy()
        filtered_df["beta"] = filtered_df["sector"].apply(_calculate_sector_beta)
        filtered_df["capm_return"] = filtered_df["beta"].apply(
            lambda b: _calculate_capm_expected_return(b, rf, rm)
        )
        filtered_df["alpha"] = filtered_df["expected_upside_pct"] - filtered_df["capm_return"]

        # --- Scatter Chart: Beta vs Expected Return ---
        capm_hover = _safe_hover_data(
            {
                "name": True,
                "ticker": True,
                "beta": ":.2f",
                "expected_upside_pct": ":.2f",
                "sector": True,
            },
            filtered_df,
        )
        scatter_kwargs = dict(
            data_frame=filtered_df,
            x="beta",
            y="expected_upside_pct",
            color="sector",
            labels={
                "beta": "Beta (Systematic Risk)",
                "expected_upside_pct": "Expected Return (%)",
                "sector": "Sector",
            },
            template="plotly_dark",
        )
        if capm_hover:
            scatter_kwargs["hover_data"] = capm_hover

        if size_encoding != "none" and size_encoding in filtered_df.columns:
            scatter_kwargs["size"] = size_encoding
            if isinstance(scatter_kwargs.get("hover_data"), dict):
                scatter_kwargs["hover_data"][size_encoding] = ":.0f"

        scatter_fig = px.scatter(**scatter_kwargs)

        if size_encoding != "none":
            scatter_fig.update_traces(marker=dict(sizemin=6))

        scatter_fig.update_layout(
            title="Beta vs Expected Return (CAPM)",
            xaxis_title="Beta (Systematic Risk)",
            yaxis_title="Expected Return (%)",
            legend_title_text="Sector",
            hovermode="closest",
        )

        # Add Security Market Line (SML)
        beta_range = np.linspace(0, max(2.0, filtered_df["beta"].max() + 0.2), 50)
        sml_returns = [_calculate_capm_expected_return(b, rf, rm) for b in beta_range]
        scatter_fig.add_trace(
            go.Scatter(
                x=beta_range,
                y=sml_returns,
                mode="lines",
                name=f"SML (Rf={rf}%, Rm={rm}%)",
                line=dict(dash="dash", color="rgba(255,255,255,0.5)", width=2),
            )
        )

        # --- Bar Chart: Alpha (Excess Return) ---
        df_sorted = filtered_df.sort_values("alpha", ascending=False)
        top_n = 15
        top_stocks = df_sorted.head(top_n)
        bottom_stocks = df_sorted.tail(top_n)
        df_display = (
            pd.concat([top_stocks, bottom_stocks])
            .drop_duplicates(subset=["ticker"])
            .sort_values("alpha", ascending=True)
        )

        df_display = df_display.copy()
        df_display["alpha_color"] = df_display["alpha"].apply(
            lambda x: "Positive" if x > 0 else "Negative"
        )

        bar_fig = px.bar(
            df_display,
            x="alpha",
            y="name",
            color="alpha_color",
            orientation="h",
            hover_data={
                "ticker": True,
                "alpha": ":.2f",
                "expected_upside_pct": ":.2f",
                "capm_return": ":.2f",
                "alpha_color": False,
            },
            labels={
                "alpha": "Alpha (Excess Return %)",
                "name": "Company",
                "alpha_color": "Alpha Sign",
            },
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
            template="plotly_dark",
        )
        bar_fig.update_layout(
            title="Alpha: Top & Bottom Stocks vs CAPM",
            xaxis_title="Alpha (Excess Return %)",
            yaxis_title="Company",
            legend_title_text="Alpha Sign",
            height=max(400, len(df_display) * 22),
            hovermode="closest",
        )

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        scatter_error = error_msg
        bar_error = error_msg

    return scatter_fig, scatter_error, bar_fig, bar_error


# =============================================================================
# Kelly Criterion Position Sizer Callback
# =============================================================================


@app.callback(
    [
        Output("kelly-bar-chart", "figure"),
        Output("kelly-bar-error", "children"),
        Output("kelly-scatter-chart", "figure"),
        Output("kelly-scatter-error", "children"),
        Output("kelly-kpi-summary", "children"),
        Output("kelly-positions-table", "data"),
    ],
    [Input("global-filter-store", "data")]
    + [
        Input("kelly-fraction-dropdown", "value"),
        Input("kelly-max-position-dropdown", "value"),
        Input("kelly-min-confidence-dropdown", "value"),
        Input("kelly-adjustment-dropdown", "value"),
        Input("kelly-bar-color-dropdown", "value"),
        Input("kelly-scatter-color-dropdown", "value"),
        Input("kelly-scatter-size-dropdown", "value"),
    ],
)
def update_kelly_criterion(*args):
    """Update Kelly Criterion Position Sizer tab visualizations."""
    import traceback as tb

    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})

    (
        kelly_fraction_slider,
        max_position_slider,
        min_confidence,
        adjustment_method,
        bar_color_by,
        scatter_color_by,
        scatter_size_by,
    ) = args[1:]

    # Unpack RangeSlider values (returns a list)
    kelly_fraction = kelly_fraction_slider[0] if kelly_fraction_slider else 0.25
    max_position = max_position_slider[0] if max_position_slider else 0.10
    min_confidence = min_confidence if min_confidence is not None else 0.35
    adjustment_method = adjustment_method if adjustment_method is not None else "both"
    bar_color_by = bar_color_by if bar_color_by is not None else "none"
    scatter_color_by = scatter_color_by if scatter_color_by is not None else "none"
    scatter_size_by = scatter_size_by if scatter_size_by is not None else "none"

    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available",
        template="plotly_dark",
        annotations=[
            {
                "text": "No data matches the selected filters",
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )

    bar_fig = empty_fig
    bar_error = ""
    scatter_fig = empty_fig
    scatter_error = ""
    kpi_summary = html.Div()
    table_data = []

    try:
        # Apply all global filters
        filtered_df = apply_global_filters(df, filter_values, range_values)

        if filtered_df.empty:
            return empty_fig, "", empty_fig, "", html.Div("No data"), []

        # Apply Kelly-specific confidence filter
        if "confidence_score" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["confidence_score"] >= min_confidence]

        if filtered_df.empty:
            return (
                empty_fig,
                "",
                empty_fig,
                "",
                html.Div("No data after confidence filter"),
                [],
            )

        # Calculate Kelly metrics
        kelly_df = calculate_kelly_metrics(
            filtered_df,
            kelly_fraction=kelly_fraction,
            max_position=max_position,
            adjustment_method=adjustment_method,
        )

        # Drop rows where Kelly calculation is invalid
        kelly_df = kelly_df.dropna(subset=["kelly_pct"])
        kelly_df = kelly_df[kelly_df["kelly_pct"] > 0]

        if kelly_df.empty:
            return (
                empty_fig,
                "",
                empty_fig,
                "",
                html.Div("No positions with positive Kelly %"),
                [],
            )

        # ----- KPI Summary Cards -----
        total_positions = len(kelly_df)
        mean_kelly = kelly_df["kelly_pct"].mean()
        max_kelly = kelly_df["kelly_pct"].max()
        top_ticker = (
            kelly_df.nlargest(1, "kelly_pct")["ticker"].values[0] if total_positions > 0 else "N/A"
        )
        total_kelly_raw_sum = kelly_df["kelly_raw"].sum()

        kpi_summary = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Eligible Positions"),
                                html.H3(f"{total_positions:,}"),
                            ]
                        ),
                        color="primary",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H5("Mean Kelly %"), html.H3(f"{mean_kelly:.2f}%")]),
                        color="info",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H5("Max Kelly %"), html.H3(f"{max_kelly:.2f}%")]),
                        color="success",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([html.H5("Top Position"), html.H3(f"{top_ticker}")]),
                        color="warning",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Sum Raw Kelly"),
                                html.H3(f"{total_kelly_raw_sum:.3f}"),
                            ]
                        ),
                        color="secondary",
                        inverse=True,
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Kelly Fraction"),
                                html.H3(f"{kelly_fraction}× | {adjustment_method.title()}"),
                            ]
                        ),
                        color="dark",
                        inverse=True,
                    ),
                    width=2,
                ),
            ],
            style={"marginBottom": "10px"},
        )

        # ----- Bar Chart: Top 30 Positions -----
        top30 = kelly_df.nlargest(30, "kelly_pct")

        bar_kwargs = dict(
            data_frame=top30,
            x="ticker",
            y="kelly_pct",
            labels={"kelly_pct": "Kelly % (Position Size)", "ticker": "Ticker"},
            template="plotly_dark",
        )

        if bar_color_by == "sector" and "sector" in top30.columns:
            bar_kwargs["color"] = "sector"
            bar_kwargs["hover_data"] = _safe_hover_data(
                {"sector": True, "kelly_pct": ":.2f", "confidence_level": True},
                top30,
            )
        elif bar_color_by == "confidence_level" and "confidence_level" in top30.columns:
            bar_kwargs["color"] = "confidence_level"
            bar_kwargs["hover_data"] = _safe_hover_data(
                {"confidence_level": True, "kelly_pct": ":.2f"},
                top30,
            )
        else:
            bar_kwargs["hover_data"] = _safe_hover_data(
                {"kelly_pct": ":.2f", "sector": True, "confidence_level": True},
                top30,
            )

        bar_fig = px.bar(**bar_kwargs)
        bar_fig.update_xaxes(tickangle=-45)
        bar_fig.update_layout(
            xaxis_title="Ticker",
            yaxis_title="Kelly % (Position Size)",
            hovermode="x unified",
            legend_title_text=(
                bar_color_by.replace("_", " ").title() if bar_color_by != "none" else ""
            ),
        )

        # ----- Scatter Chart: Kelly % vs Expected Upside -----
        scatter_kwargs = dict(
            data_frame=kelly_df,
            x="expected_upside_kalman",
            y="kelly_pct",
            labels={
                "expected_upside_kalman": "Expected Upside (%)",
                "kelly_pct": "Kelly % (Position Size)",
            },
            template="plotly_dark",
        )

        hover_cols = (
            _safe_hover_data(
                {"ticker": True, "expected_upside_kalman": ":.2f", "kelly_pct": ":.2f"},
                kelly_df,
            )
            or {}
        )

        if scatter_color_by == "confidence_level" and "confidence_level" in kelly_df.columns:
            scatter_kwargs["color"] = "confidence_level"
            hover_cols["confidence_level"] = True

        if (
            scatter_size_by == "achievement_probability"
            and "achievement_probability" in kelly_df.columns
        ):
            scatter_kwargs["size"] = "achievement_probability"
            hover_cols["achievement_probability"] = ":.2f"

        scatter_kwargs["hover_data"] = hover_cols

        scatter_fig = px.scatter(**scatter_kwargs)
        scatter_fig.update_traces(marker=dict(sizemin=6))
        scatter_fig.update_layout(
            xaxis_title="Expected Upside (%)",
            yaxis_title="Kelly % (Position Size)",
            hovermode="closest",
            legend_title_text=(
                scatter_color_by.replace("_", " ").title() if scatter_color_by != "none" else ""
            ),
        )

        # ----- Positions Table -----
        table_cols = [
            "ticker",
            "name",
            "country",
            "trading_country",
            "unit",
            "sector",
            "industry",
            "exchange",
            "last_price",
            "price_target",
            "price_target_mc",
            "expected_upside_pct",
            "implied_return_pt",
            "expected_upside_kalman",
            "prob_positive_upside",
            "confidence_score",
            "composite_score",
            "quality_tier",
            "achievement_probability",
            "confidence_level",
            "signal",
            "kelly_pct",
            "price_target_prob_weighted",
        ]
        available_cols = [c for c in table_cols if c in kelly_df.columns]
        table_df = kelly_df.nlargest(50, "kelly_pct")[available_cols].copy()

        # Round numeric columns for display
        for col in [
            "expected_upside_pct",
            "implied_return_pt",
            "expected_upside_kalman",
            "prob_positive_upside",
            "confidence_score",
            "achievement_probability",
            "kelly_pct",
            "last_price",
            "price_target",
            "price_target_mc",
            "price_target_prob_weighted",
        ]:
            if col in table_df.columns:
                table_df[col] = table_df[col].round(3)

        table_data = table_df.to_dict("records")

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{tb.format_exc()}"
        bar_error = error_msg
        scatter_error = error_msg

    return bar_fig, bar_error, scatter_fig, scatter_error, kpi_summary, table_data


# =============================================================================
# Efficient Frontier Callback
# =============================================================================


@app.callback(
    [
        Output("ef-frontier-graph", "figure"),
        Output("ef-portfolio-table", "children"),
        Output("ef-error-display", "children"),
    ],
    [Input("global-filter-store", "data")]
    + [
        Input("ef-stock-selector", "value"),
        Input("ef-risk-free-rate", "value"),
        Input("ef-constraint-type", "value"),
        Input("ef-num-portfolios", "value"),
    ],
)
def update_efficient_frontier(*args):
    """Update Efficient Frontier visualization based on filters and parameters."""
    import traceback as tb

    filter_data = args[0] or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})

    stock_selector, risk_free_rate_slider, constraint_type, num_portfolios = args[1:]

    # Unpack RangeSlider value (returns a list) — slider is 0–1 fraction, convert to percentage for downstream use
    risk_free_rate = risk_free_rate_slider[0] * 100 if risk_free_rate_slider else 3.0
    constraint_type = constraint_type if constraint_type is not None else "long_only"
    num_portfolios = num_portfolios if num_portfolios is not None else 500

    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data available",
        template="plotly_dark",
        annotations=[
            {
                "text": "No data matches the selected filters",
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )

    try:
        # Apply all global filters
        filtered_df = apply_global_filters(df, filter_values, range_values)

        if filtered_df.empty:
            return empty_fig, "", ""

        # Determine selected tickers
        if not stock_selector or len(stock_selector) == 0:
            selected_tickers = filtered_df.nlargest(10, "market_cap")["ticker"].tolist()
        else:
            selected_tickers = (
                stock_selector if isinstance(stock_selector, list) else [stock_selector]
            )

        # Keep only tickers present in filtered data
        df_selected = filtered_df[filtered_df["ticker"].isin(selected_tickers)].copy()
        selected_tickers = df_selected["ticker"].unique().tolist()

        if len(selected_tickers) < 2:
            info_fig = go.Figure()
            info_fig.update_layout(
                title="Insufficient data",
                template="plotly_dark",
                annotations=[
                    {
                        "text": "Need at least 2 stocks for portfolio optimization",
                        "showarrow": False,
                        "font": {"size": 20},
                    }
                ],
            )
            return info_fig, "", ""

        # Compute expected returns from the data
        expected_returns = []
        for ticker in selected_tickers:
            ticker_data = df_selected[df_selected["ticker"] == ticker]
            if len(ticker_data) > 0 and "expected_upside_pct" in ticker_data.columns:
                ret = ticker_data["expected_upside_pct"].iloc[0]
                ret = float(ret) / 100.0 if pd.notna(ret) else 0.05
            else:
                ret = 0.05
            expected_returns.append(ret)
        expected_returns = np.array(expected_returns)

        # Estimate covariance matrix
        cov_matrix = _ef_estimate_covariance_matrix(filtered_df, selected_tickers)

        # Generate random portfolios
        (
            portfolio_returns,
            portfolio_volatilities,
            portfolio_sharpe_ratios,
            portfolio_weights,
        ) = _ef_generate_random_portfolios(
            expected_returns,
            cov_matrix,
            num_portfolios,
            risk_free_rate,
            constraint_type,
        )

        if len(portfolio_returns) == 0 or portfolio_volatilities.max() == 0:
            return empty_fig, "", "Could not generate valid portfolio combinations."

        # Find optimal portfolios
        min_var_weights, max_sharpe_weights, opt_volatilities, opt_returns = (
            _ef_find_optimal_portfolios(expected_returns, cov_matrix, risk_free_rate)
        )
        min_var_return = opt_returns[0]
        max_sharpe_return = opt_returns[1]
        max_sharpe_volatility = opt_volatilities[1]
        max_sharpe_ratio = (
            (max_sharpe_return - risk_free_rate / 100) / max_sharpe_volatility
            if max_sharpe_volatility > 0
            else 0
        )

        # Build the scatter plot
        scatter_df = pd.DataFrame(
            {
                "Volatility": portfolio_volatilities * 100,
                "Return": portfolio_returns * 100,
                "Sharpe Ratio": portfolio_sharpe_ratios,
            }
        )

        fig = px.scatter(
            scatter_df,
            x="Volatility",
            y="Return",
            color="Sharpe Ratio",
            color_continuous_scale="Viridis",
            labels={
                "Volatility": "Portfolio Volatility (%)",
                "Return": "Expected Return (%)",
            },
            hover_data={"Volatility": ":.2f", "Return": ":.2f", "Sharpe Ratio": ":.3f"},
            template="plotly_dark",
        )

        # Min Variance marker
        fig.add_trace(
            go.Scatter(
                x=[opt_volatilities[0] * 100],
                y=[min_var_return * 100],
                mode="markers",
                marker=dict(
                    size=15,
                    color="red",
                    symbol="star",
                    line=dict(color="darkred", width=2),
                ),
                name="Min Variance",
                hovertemplate="<b>Min Variance Portfolio</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
            )
        )

        # Max Sharpe marker
        fig.add_trace(
            go.Scatter(
                x=[max_sharpe_volatility * 100],
                y=[max_sharpe_return * 100],
                mode="markers",
                marker=dict(
                    size=15,
                    color="gold",
                    symbol="star",
                    line=dict(color="orange", width=2),
                ),
                name="Max Sharpe Ratio",
                hovertemplate=(
                    f"<b>Max Sharpe Ratio Portfolio</b><br>"
                    f"Volatility: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<br>"
                    f"Sharpe: {max_sharpe_ratio:.3f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Efficient Frontier: Risk-Return Tradeoff",
            xaxis_title="Portfolio Volatility (Annualized %)",
            yaxis_title="Expected Return (Annualized %)",
            hovermode="closest",
            coloraxis_colorbar=dict(title="Sharpe Ratio"),
        )

        # Build portfolio summary table
        table_data = []
        for i in range(min(10, len(portfolio_weights))):
            row = {"Portfolio": f"P{i + 1}"}
            for j, ticker in enumerate(selected_tickers):
                row[ticker] = f"{portfolio_weights[i, j]:.2%}"
            row["Return"] = f"{portfolio_returns[i]:.2%}"
            row["Volatility"] = f"{portfolio_volatilities[i]:.2%}"
            row["Sharpe"] = f"{portfolio_sharpe_ratios[i]:.3f}"
            table_data.append(row)

        # Append optimal portfolios
        min_var_row = {"Portfolio": "⭐ Min Variance"}
        for j, ticker in enumerate(selected_tickers):
            min_var_row[ticker] = f"{min_var_weights[j]:.2%}"
        min_var_row["Return"] = f"{min_var_return:.2%}"
        min_var_row["Volatility"] = f"{opt_volatilities[0]:.2%}"
        min_var_row["Sharpe"] = (
            f"{(min_var_return - risk_free_rate / 100) / opt_volatilities[0]:.3f}"
            if opt_volatilities[0] > 0
            else "N/A"
        )
        table_data.append(min_var_row)

        max_sharpe_row = {"Portfolio": "⭐ Max Sharpe"}
        for j, ticker in enumerate(selected_tickers):
            max_sharpe_row[ticker] = f"{max_sharpe_weights[j]:.2%}"
        max_sharpe_row["Return"] = f"{max_sharpe_return:.2%}"
        max_sharpe_row["Volatility"] = f"{max_sharpe_volatility:.2%}"
        max_sharpe_row["Sharpe"] = f"{max_sharpe_ratio:.3f}"
        table_data.append(max_sharpe_row)

        table_df = pd.DataFrame(table_data)

        # Render as a styled DataTable
        table_component = dash_table.DataTable(
            data=table_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in table_df.columns],
            page_size=500,
            sort_action="native",
            style_table={"overflowX": "auto", "width": "100%"},
            style_header=TABLE_STYLE_HEADER,
            style_cell=TABLE_STYLE_CELL,
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#252525"},
                {
                    "if": {"filter_query": '{Portfolio} contains "⭐"'},
                    "backgroundColor": "#2a3f5f",
                    "fontWeight": "bold",
                },
            ],
        )

        return fig, table_component, ""

    except Exception as e:
        error_msg = f"Error updating efficient frontier: {str(e)}\n{tb.format_exc()}"
        return empty_fig, "", error_msg


@app.callback(
    Output("ef-stock-selector", "options"),
    [Input("global-filter-store", "data")],
    [State("ef-stock-selector", "value")],
)
def update_ef_stock_options(*args):
    """Update stock selector options based on global filters.

    Uses ``State`` to read the current selection so previously chosen
    tickers are retained in the options list even when filters change.
    """
    try:
        filter_data = args[0] or {}
        filter_values = filter_data.get("filters", {})
        range_values = filter_data.get("ranges", {})
        current_selection = args[1]  # State value
        filtered_df = apply_global_filters(df, filter_values, range_values)

        if filtered_df.empty or "market_cap" not in filtered_df.columns:
            return []

        df_sorted = filtered_df.nlargest(50, "market_cap")
        options = [
            {"label": f"{row['ticker']} - {row['name'][:30]}", "value": row["ticker"]}
            for _, row in df_sorted.iterrows()
        ]

        # Retain previously selected tickers that may no longer be in top-50
        if current_selection:
            selected = (
                current_selection if isinstance(current_selection, list) else [current_selection]
            )
            existing_values = {o["value"] for o in options}
            for ticker in selected:
                if ticker not in existing_values and ticker in filtered_df["ticker"].values:
                    row = filtered_df[filtered_df["ticker"] == ticker].iloc[0]
                    options.append(
                        {
                            "label": f"{row['ticker']} - {row['name'][:30]}",
                            "value": row["ticker"],
                        }
                    )

        return options
    except Exception:
        return []


# =============================================================================
# Cascading Filters: region → country/exchange/trading_country
# =============================================================================


@app.callback(
    [
        Output("country-dropdown", "options"),
        Output("exchange-dropdown", "options"),
        Output("trading-country-dropdown", "options"),
    ],
    Input("region-dropdown", "value"),
)
def update_cascading_region_filters(selected_regions):
    """Update country, exchange, and trading_country options based on region selection."""
    filtered = df if not selected_regions else df[df["region"].isin(selected_regions)]
    return (
        build_filter_options(filtered, "country"),
        build_filter_options(filtered, "exchange"),
        build_filter_options(filtered, "trading_country"),
    )


@app.callback(
    Output("industry-dropdown", "options"),
    Input("sector-dropdown", "value"),
)
def update_cascading_sector_filters(selected_sectors):
    """Update industry options based on sector selection."""
    filtered = df if not selected_sectors else df[df["sector"].isin(selected_sectors)]
    return build_filter_options(filtered, "industry")


# =============================================================================
# Global Filter Store — persist filter state across tabs
# =============================================================================


@app.callback(
    Output("global-filter-store", "data"),
    [Input(f["id"], "value") for f in FILTER_CONFIG]
    + [Input(s["id"], "value") for s in RANGE_SLIDER_CONFIG],
)
def update_global_filter_store(*args):
    """Update the global filter store whenever any filter changes."""
    num_dropdowns = len(FILTER_CONFIG)
    filter_values = collect_filter_values(*args[:num_dropdowns])
    range_values = collect_range_slider_values(*args[num_dropdowns:])
    return {"filters": filter_values, "ranges": range_values}


# =============================================================================
# Stock Screening Explorer Tab Callback
# =============================================================================


@app.callback(
    [
        Output("screening-kpis", "children"),
        Output("screening-summary-chart", "figure"),
        Output("screening-results-table-container", "children"),
    ],
    [
        Input("screening-strategy-dropdown", "value"),
    ]
    + [Input("global-filter-store", "data")],
)
def update_screening_explorer(selected_strategy, *filter_args):
    """Update the Stock Screening Explorer tab based on selected strategy and filters."""
    empty_fig = go.Figure().update_layout(title="No data", template="plotly_dark")

    filter_data = filter_args[0] if filter_args else {}
    if not isinstance(filter_data, dict):
        filter_data = {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    filtered_df = apply_global_filters(df, filter_values, range_values)

    if filtered_df.empty:
        return [], empty_fig, html.Div("No data matching filters.")

    # Run all screens for summary chart
    all_screens = {}
    for key, label, func in ALL_SCREENING_STRATEGIES:
        try:
            result = func(filtered_df)
            if not result.empty:
                all_screens[key] = result
        except Exception:
            pass

    # Summary bar chart
    summary_fig = empty_fig
    if ER_VIZ_AVAILABLE and all_screens:
        try:
            summary_fig = create_screening_summary_chart(all_screens)
        except Exception:
            pass

    # Selected screen results
    selected_result = all_screens.get(selected_strategy, pd.DataFrame())

    # KPIs
    kpis = []
    total_universe = len(filtered_df)
    screen_count = len(selected_result)
    hit_rate = (screen_count / total_universe * 100) if total_universe > 0 else 0
    kpis.append(build_kpi_card("Universe Size", f"{total_universe:,}", "info"))
    kpis.append(build_kpi_card("Screen Hits", f"{screen_count:,}", "success"))
    kpis.append(build_kpi_card("Hit Rate", f"{hit_rate:.1f}%", "primary"))
    kpis.append(build_kpi_card("Screens Available", f"{len(all_screens)}", "warning"))

    # Results table
    if selected_result.empty:
        table_component = html.Div(
            "No stocks pass this screening strategy with current filters.",
            style={"color": COLORS["warning"], "textAlign": "center", "padding": "20px"},
        )
    else:
        display_cols = [
            c
            for c in [
                "ticker",
                "name",
                "sector",
                "industry",
                "country",
                "expected_upside_pct",
                "expected_upside_kalman",
                "confidence_score",
                "composite_score",
                "achievement_probability",
                "signal",
                "resampled_posterior_mean",
                "momentum_signal",
            ]
            if c in selected_result.columns
        ]
        table_component = dash_table.DataTable(
            data=selected_result[display_cols].to_dict("records"),
            columns=get_formatted_columns(display_cols),
            page_size=50,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto", "width": "100%"},
            style_header=TABLE_STYLE_HEADER,
            style_cell=TABLE_STYLE_CELL,
            style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
            tooltip_header={
                col: COLUMN_TOOLTIPS.get(col, col.replace("_", " ").title()) for col in display_cols
            },
        )

    return kpis, summary_fig, table_component


@app.callback(
    Output("screening-download", "data"),
    Input("screening-download-btn", "n_clicks"),
    [
        State("screening-strategy-dropdown", "value"),
        State("global-filter-store", "data"),
    ],
    prevent_initial_call=True,
)
def download_screening_results(n_clicks, selected_strategy, filter_data):
    """Download selected screening results as CSV."""
    if not n_clicks:
        return dash.no_update

    filter_data = filter_data or {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    filtered_df = apply_global_filters(df, filter_values, range_values)

    if filtered_df.empty:
        return dash.no_update

    # Find the screen function
    func = None
    for key, _label, fn in ALL_SCREENING_STRATEGIES:
        if key == selected_strategy:
            func = fn
            break

    if func is None:
        return dash.no_update

    try:
        result = func(filtered_df)
        if result.empty:
            return dash.no_update
        return dcc.send_data_frame(result.to_csv, f"screening_{selected_strategy}.csv", index=False)
    except Exception:
        return dash.no_update


# =============================================================================
# Resampled Posterior & MCMC Tab Callback
# =============================================================================


@app.callback(
    Output("resampled-posterior-scatter", "figure"),
    [Input("global-filter-store", "data")],
)
def update_resampled_posterior_scatter(*filter_args):
    """Generate resampled posterior distribution scatter plot."""
    filter_data = filter_args[0] if filter_args else {}
    if not isinstance(filter_data, dict):
        filter_data = {}
    filter_values = filter_data.get("filters", {})
    range_values = filter_data.get("ranges", {})
    filtered_df = apply_global_filters(df, filter_values, range_values)

    if filtered_df.empty or "resampled_posterior_mean" not in filtered_df.columns:
        return go.Figure().update_layout(
            title="Resampled Posterior Mean — No Data Available",
            template="plotly_dark",
        )

    plot_df = filtered_df.dropna(subset=["resampled_posterior_mean"]).copy()
    if plot_df.empty:
        return go.Figure().update_layout(
            title="Resampled Posterior Mean — No Valid Data",
            template="plotly_dark",
        )

    color_col = "sector" if "sector" in plot_df.columns else None
    size_col = None
    if "confidence_score" in plot_df.columns:
        size_col = "confidence_score"

    resamp_hover = _safe_hover_data(
        {
            "momentum_signal": ":.3f",
            "technical_adjustment": ":.3f",
            "volatility_regime_score": ":.3f",
        },
        plot_df,
    )
    fig = px.scatter(
        plot_df,
        x="resampled_posterior_mean",
        y=(
            "expected_upside_pct"
            if "expected_upside_pct" in plot_df.columns
            else "resampled_posterior_mean"
        ),
        color=color_col,
        size=size_col,
        hover_name="ticker" if "ticker" in plot_df.columns else None,
        hover_data=resamp_hover,
        title="Resampled Posterior Mean vs Expected Upside",
        template="plotly_dark",
        labels={
            "resampled_posterior_mean": "Resampled Posterior Mean",
            "expected_upside_pct": "Expected Upside (%)",
        },
    )
    fig.update_layout(height=600)
    return fig


if __name__ == "__main__":
    if len(df) == 0:
        print("\n" + "=" * 60)
        print("⚠️  No data loaded - Dashboard cannot start")
        print("=" * 60)
        print("\nTo use this dashboard:")
        print("1. Set environment variable: GEIB_DASHBOARD=true")
        print("2. Ensure DB_URL is configured")
        print("3. Verify analytics.expected_returns_summary table exists")
        print("\nExample:")
        print("  $env:GEIB_DASHBOARD='true'  # PowerShell")
        print("  export GEIB_DASHBOARD=true  # Bash")
        print("  python finance_ml/dashboards/geib_dash_app.py")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("🚀 Starting Global Equity Investment Board Dashboard")
        print("=" * 60)
        print(f"   Loaded: {len(df):,} stocks")
        print("   URL: http://127.0.0.1:8051")
        print("=" * 60 + "\n")
        app.run(debug=True, port=8051)
