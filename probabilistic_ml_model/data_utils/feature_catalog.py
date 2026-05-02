"""
Schema-driven feature catalog for SQL ↔ Python model alignment.

Bridges the PostgreSQL schema metadata tables with the Python model
pipeline, eliminating hardcoded column lists and providing a single
source of truth for which columns each model needs and which
``vw_features_*`` views provide them.

Data sources:
    - ``calculated_features_registry`` — feature → category → source_function
    - ``feature_registry_metadata``    — function → category → description
    - ``information_schema.columns``   — view → column discovery
    - ``equities_schema_metadata``     — column → role → alias

Canonical registries (imported by ``inference_schema`` and ``data_utils``):
    - ``FEATURE_VIEW_REGISTRY``     — view name → category label (17 views)
    - ``DEFAULT_IDENTIFIER_COLUMNS`` — ordered fallback identifier column list
    - ``IDENTIFIER_COLUMNS_SET``     — frozenset for fast membership tests
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    create_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    SQLAlchemyError = Exception  # type: ignore[assignment,misc]


# ═══════════════════════════════════════════════════════════════════════════════
# Canonical View ↔ Category Registry (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_VIEW_REGISTRY: dict[str, str] = {
    "vw_features_analyst_sentiment": "Analyst Sentiment",
    "vw_features_balance_sheet": "Balance Sheet",
    "vw_features_cashflow": "Cash Flow",
    "vw_features_composite_scores": "Composite Scores",
    "vw_features_cost_structure": "Cost Structure",
    "vw_features_dividends": "Dividend Features",
    "vw_features_earnings": "Earnings Quality",
    "vw_features_employment": "Employment",
    "vw_features_growth": "Growth Metrics",
    "vw_features_leverage_liquidity": "Leverage & Liquidity",
    "vw_features_momentum": "Technical Analysis",
    "vw_features_profitability": "Profitability",
    "vw_features_quality_risk": "Quality & Risk",
    "vw_features_technical_analysis": "Technical Analysis",
    "vw_features_temporal": "Inventory Temporal",
    "vw_features_unusual_items": "Unusual Items",
    "vw_features_valuation_ratios": "Valuation Ratios",
}

#: Ordered list of view names (convenience for iteration).
VW_FEATURES_VIEWS: list[str] = list(FEATURE_VIEW_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Canonical Identifier Columns (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_IDENTIFIER_COLUMNS: list[str] = [
    "isin",
    "ticker",
    "name",
    "description",
    "region",
    "country",
    "trading_country",
    "exchange",
    "sector",
    "industry",
    "dividend_record_frequency",
    "earnings_report_frequency",
    "fy_end",
    "next_earnings_report",
    "next_earnings_status",
    "next_earnings_when",
    "next_fiscal_quarter",
    "reporting_interval",
    "size_class",
    "style_class",
    "unit",
    "dividend_record_announce_date",
    "dividend_record_ex_date",
    "dividend_record_payable_date",
    "dividend_record_record_date",
    "fy_end_date",
    "income_statement_report_date",
    "last_updated",
    "next_earnings",
    "next_fy_end_date",
    "next_income_statement_report_date",
    "reference_date",
]

#: Frozenset for fast membership tests (used by inference_schema builders).
IDENTIFIER_COLUMNS_SET: frozenset[str] = frozenset(DEFAULT_IDENTIFIER_COLUMNS)


# ═══════════════════════════════════════════════════════════════════════════════
# Model ↔ Feature Requirements Registry
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ModelFeatureRequirement:
    """Declares which feature categories a probability model consumes.

    Instead of hardcoding column lists in each ``run_*`` function, each
    model declares the *categories* (from ``calculated_features_registry``)
    it needs.  The catalog resolves categories → actual column names at
    runtime from the database.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier (e.g. ``"PriceTargetAchievement"``).
    required_categories : tuple[str, ...]
        Category names from ``calculated_features_registry.category``
        (e.g. ``("Analyst Sentiment", "Beta & Risk")``).
    optional_categories : tuple[str, ...]
        Categories that improve the model but aren't mandatory.
    fallback_columns : tuple[str, ...]
        Hardcoded column names used when the DB is unreachable.
    """

    model_name: str
    required_categories: tuple[str, ...] = ()
    optional_categories: tuple[str, ...] = ()
    fallback_columns: tuple[str, ...] = ()


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization ↔ Feature Requirements Registry
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VisualizationFeatureRequirement:
    """Declares which columns a visualization module needs.

    Similar to ``ModelFeatureRequirement`` but oriented toward visualization
    functions that need specific columns to render charts.

    Attributes
    ----------
    viz_name : str
        Human-readable visualization identifier.
    required_categories : tuple[str, ...]
        Category names from ``calculated_features_registry.category``.
    fallback_columns : tuple[str, ...]
        Hardcoded column names used when the DB is unreachable.
    column_aliases : dict[str, list[str]]
        Logical column name → list of fallback aliases.  Used by
        ``resolve_column_from_catalog`` to find the best available column.
    """

    viz_name: str
    required_categories: tuple[str, ...] = ()
    fallback_columns: tuple[str, ...] = ()
    column_aliases: dict[str, list[str]] = field(default_factory=dict)


VIZ_REQUIREMENTS: dict[str, VisualizationFeatureRequirement] = {
    "quality_risk": VisualizationFeatureRequirement(
        viz_name="QualityRisk",
        required_categories=("Quality & Risk",),
        fallback_columns=(
            "piotroski_f_score",
            "altman_z_score",
            "beneish_m_score",
            "distress_risk_score",
            "quality_momentum_score",
            "combined_distress_score",
            "accounting_quality_score",
        ),
        column_aliases={
            "piotroski_f_score": ["piotroski_f_score", "f_score"],
            "accounting_quality_score": [
                "accounting_quality_score",
                "accounting_quality_score_comp",
            ],
        },
    ),
    "valuation": VisualizationFeatureRequirement(
        viz_name="Valuation",
        required_categories=("Valuation Ratios",),
        fallback_columns=(
            "p_e_ratio",
            "p_b_ratio",
            "ev_ebitda_ratio",
            "ev_sales_ratio",
            "peg_ratio",
            "dividend_yield",
            "forward_pe_premium",
            "valuation_mean_reversion",
            "valuation_compression",
        ),
        column_aliases={
            "dividend_yield": [
                "valuation_dividend_yield",
                "dividend_yield",
                "dividend_yield_ltm",
            ],
        },
    ),
    "growth_analysis": VisualizationFeatureRequirement(
        viz_name="GrowthAnalysis",
        required_categories=("Growth Metrics",),
        fallback_columns=(
            "revenue_growth_yoy",
            "revenue_cagr_3y",
            "revenue_cagr_5y",
            "ebitda_growth_yoy",
            "eps_yoy_growth",
            "eps_cagr_3y",
            "operating_income_growth",
            "net_income_growth_yoy",
        ),
        column_aliases={
            "revenue_growth_yoy": ["revenue_yoy_growth", "revenue_growth_yoy"],
            "revenue_growth_3y_cagr": ["revenue_cagr_3y", "revenue_growth_3y_cagr"],
            "revenue_growth_5y_cagr": ["revenue_cagr_5y", "revenue_growth_5y_cagr"],
            "eps_growth_3y_cagr": ["eps_cagr_3y", "eps_growth_3y_cagr"],
            "eps_growth_yoy": ["eps_yoy_growth", "eps_growth_yoy"],
            "net_income_growth": ["net_income_growth_yoy", "net_income_growth"],
            "operating_income_growth": [
                "operating_income_growth_yoy",
                "operating_income_growth",
            ],
            "ebitda_growth_yoy": ["growth_ebitda_growth_yoy", "ebitda_growth_yoy"],
            "fcf_growth_yoy": ["fcf_growth_yoy", "fcf_yoy_growth"],
        },
    ),
    "earnings_quality": VisualizationFeatureRequirement(
        viz_name="EarningsQuality",
        required_categories=("Earnings Quality",),
        fallback_columns=(
            "eps_surprise_pct",
            "eps_positive_years",
            "eps_positive_streak",
            "eps_trajectory_score",
            "eps_improvement_count",
            "earnings_quality_composite",
            "ni_adjustment_ratio",
            "eps_adjustment_ratio",
            "accounting_quality_score",
            "earnings_quality_impact",
            "eps_revision_momentum",
            "gaap_adj_eps_gap_pct",
        ),
        column_aliases={
            "eps_beat_count": ["eps_positive_years"],
            "eps_total_reports": ["eps_positive_streak"],
            "earnings_quality_composite": [
                "earnings_quality_composite",
                "earnings_quality_score",
            ],
            "accounting_quality_score": [
                "accounting_quality_score",
                "accounting_quality_score_comp",
            ],
        },
    ),
    "expected_returns": VisualizationFeatureRequirement(
        viz_name="ExpectedReturns",
        required_categories=("Analyst Sentiment", "Price Target Dynamics"),
        fallback_columns=(
            "implied_return_mc",
            "implied_return_kalman",
            "implied_return_pt",
            "prob_positive_upside",
            "agreement_score",
            "weighted_agreement",
        ),
    ),
    "probability": VisualizationFeatureRequirement(
        viz_name="Probability",
        required_categories=("Quality & Risk", "Earnings Quality"),
        fallback_columns=(
            "implied_return_pt",
            "posterior_beat_prob",
            "achievement_probability",
            "ruin_probability",
            "accounting_anomaly_score",
            "anomaly_conditional_probability",
        ),
    ),
}


# Declare requirements for each model — replaces inline column lists
MODEL_REQUIREMENTS: dict[str, ModelFeatureRequirement] = {
    "price_target_achievement": ModelFeatureRequirement(
        model_name="PriceTargetAchievement",
        required_categories=("Analyst Sentiment", "Price Target Dynamics"),
        optional_categories=(
            "Quality & Risk",
            "Earnings Quality",
            "Balance Sheet",
            "Technical Analysis",
            "Profitablility",
            "Cash Flow",
            "Profitablility",
        ),
        fallback_columns=(
            "expected_upside_pt",
            "analyst_conviction",
            "eps_revision_momentum",
            "analyst_rating_normalized",
            "price_target_spread_pct",
            "pt_momentum_1m",
            "pt_consensus_convergence",
            "pt_acceleration_short",
            "analyst_coverage_trend",
            "analyst_bullish_pct",
            "beta_1y",
            "beta_stability_score",
            "distress_risk_score",
            "balance_sheet_strength",
            "debt_maturity_risk",
        ),
    ),
    "credit_risk": ModelFeatureRequirement(
        model_name="CreditRisk",
        required_categories=(
            "Financial Distress",
            "Leverage & Liquidity",
            "Working Capital",
            "Balance Sheet",
            "Quality & Risk",
            "Cash Flow",
        ),
        optional_categories=(
            "Earnings Quality",
            "Profitablility",
        ),
        fallback_columns=(
            "altman_z_score",
            "altman_z_trend",
            "liquidity_stress_score",
            "cash_runway_months",
            "accumulated_deficit_flag",
            "combined_distress_risk_score",
            "wc_deteriorating_flag",
            "interest_coverage",
            "quick_ratio",
            "beta_stability_score",
            "balance_sheet_strength",
            "days_working_capital",
            "debt_maturity_risk",
            "current_ratio",
            "wc_fq_deep",
            "debt_deleveraging",
            "wc_to_revenue",
            "asset_turnover",
            "working_capital_turns",
            "debt_to_equity_trend",
            "intangibles_growth_flag",
            "asset_quality_score",
            "inventory_turnover",
            "wc_volatility",
            "wc_improvement_flag_deep",
            "wc_fy_deep",
            "wc_ltm_deep",
            "negative_wc_flag",
            "receivables_days",
            "wc_change_qoq_deep",
            "wc_change_yoy_deep",
            "wc_efficiency_score",
            "wc_positive_quarters",
            "cash_to_assets_pct",
            "working_capital_ratio",
            "wc_improving_flag",
            "cash_ratio",
            "debt_to_assets",
            "equity_ratio",
            "wc_to_assets",
            "debt_to_equity",
            "cash_change_qoq",
            "debt_3y_cagr",
            "debt_4q_trend",
            "debt_yoy_change",
            "adequate_cash_buffer",
            "cash_vs_5y_avg",
            "retained_earnings_vs_5y",
            "distress_risk_score",
            "retained_earnings_growth",
            "beta_trend",
        ),
    ),
    "dividend_safety": ModelFeatureRequirement(
        model_name="DividendSafety",
        required_categories=("Dividends",),
        optional_categories=("Leverage & Liquidity", "Balance Sheet"),
        fallback_columns=(
            "interest_coverage",
            "debt_to_equity",
            "cash_ratio",
            "working_capital_ratio",
            "balance_sheet_strength",
            "cash_runway_months",
            "retained_earnings_growth",
            "debt_3y_cagr",
        ),
    ),
    "earnings_beat": ModelFeatureRequirement(
        model_name="EarningsBeat",
        required_categories=(
            "Earnings",
            "Earnings Quality",
            "EPS Trajectory",
            "Growth Metrics",
            "Analyst Sentiment",
        ),
        optional_categories=("Quality & Risk", "Balance Sheet", "Efficiency Ratios"),
        fallback_columns=(
            "accounting_quality_score",
            "quality_issues_count_5y",
            "balance_sheet_strength",
            "cash_runway_months",
            "retained_earnings_growth",
            "debt_3y_cagr",
        ),
    ),
    "accounting_anomaly": ModelFeatureRequirement(
        model_name="AccountingAnomaly",
        required_categories=(
            "Earnings Quality",
            "Cash Flow",
            "Profitablility",
            "Unusual Items",
            "Quality & Risk",
            "Balance Sheet",
        ),
        optional_categories=("Quality & Risk",),
        fallback_columns=(
            "exceptional_items_frequency",
            "gaap_adj_eps_gap_pct",
            "asset_sale_boost",
            "ebitda_adjustment_ratio",
            "eps_adjustment_ratio",
            "exceptional_items_to_ebitda",
            "restructuring_intensity",
            "goodwill_change_rate",
            "eps_adj_ltm",
            "eps_adjustment_ratio_comp",
            "eps_adjustment_spread_ltm",
            "eps_adjustment_spread_fy",
            "eps_adjustment_pct",
            "net_income_adjustment_ratio_ltm",
            "net_income_adjustment_ratio_fy",
            "net_income_adjustment_pct",
            "ebitda_adjustment_pct_ltm",
            "ebitda_adjustment_pct_fy",
            "ebit_adjustment_pct_ltm",
            "ebit_adjustment_pct_fy",
            "forward_eps_gaap_adj_spread",
            "gaap_vs_norm_revision_spread",
            "gaap_revision_momentum",
            "gaap_revision_1m",
            "gaap_revision_3m",
            "gaap_revision_6m",
            "gaap_revision_1y",
            "discontinued_ops_impact",
            "earnings_quality_warning",
            "revision_quality_divergence",
            "eps_growth_accel",
            "eps_surprise_pct",
            "revenue_surprise_pct",
            "working_capital_trend",
            "accum",
        ),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Canonical Fallback Feature Categories (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

#: Hardcoded fallback mapping of category → feature aliases.
#: Used when the database (``calculated_features_registry``) is unreachable.
#: Imported by ``data_utils.load_feature_categories_from_db`` as the
#: canonical fallback so there is exactly one place to maintain this list.
FALLBACK_FEATURE_CATEGORIES: dict[str, list[str]] = {
    "Valuation Ratios": [
        "p_e_ratio",
        "p_b_ratio",
        "ev_ebitda_ratio",
        "ev_sales_ratio",
        "valuation_dividend_yield",
        "peg_ratio",
        "price_to_tangible_book",
        "tangible_book_value_ltm",
    ],
    "Valuation Timeseries": [
        "ev_sales_trend_1y",
        "ev_ebitda_momentum",
        "p_e_momentum_yoy",
        "p_e_momentum_qoq",
        "ev_sales_vs_3y_avg",
        "ev_ebitda_vs_3y_avg",
        "p_e_vs_3y_avg",
        "p_e_vs_5y_avg",
        "p_b_vs_5y_avg",
        "valuation_mean_reversion",
        "valuation_compression",
        "forward_pe_premium",
    ],
    "Technical Analysis": [
        "price_momentum_1m",
        "price_momentum_3m",
        "price_momentum_6m",
        "price_momentum_1y",
        "price_momentum_3y",
        "price_momentum_5y",
        "price_momentum_5d",
        "range_52w_position",
        "long_term_trend_score",
        "secular_trend_flag",
        "beta_momentum",
        "volatility_regime",
    ],
    "Technical Analysis": [
        "ema_slope_20d",
        "ema_trend_consistency",
        "price_vs_ema_100d",
        "near_52w_high_flag",
        "near_52w_low_flag",
        "volume_momentum_score",
        "breakout_signal",
        "volatility_compression",
        "volatility_term_structure",
    ],
    "Profitability": [
        "roe",
        "roa",
        "gross_margin_pct",
        "operating_margin_pct",
        "net_margin_pct",
        "ebitda_margin_pct",
        "roic",
        "rnd_intensity",
        "equity_multiplier",
        "gross_margin_trend_yoy",
        "operating_margin_trend",
        "net_margin_trend_yoy",
        "ebitda_margin_trend",
        "margin_expansion_flag",
        "margin_stability_score",
    ],
    "Earnings Quality": [
        "eps_surprise_pct",
        "revenue_surprise_pct",
        "eps_adjustment_ratio",
        "gaap_adj_eps_gap_pct",
        "ebitda_adjustment_ratio",
        "eps_quarterly_trend",
        "eps_yoy_growth",
        "earnings_quality_score",
        "earnings_quality_warning",
        "earnings_quality_composite",
        "gaap_revision_momentum",
        "revision_quality_divergence",
    ],
    "EPS Trajectory": [
        "eps_qoq_growth",
        "eps_yoy_quarterly",
        "eps_positive_streak",
        "eps_cagr_3y",
        "eps_cagr_5y",
        "eps_growth_accel",
        "eps_vs_5y_avg",
        "eps_trajectory_score",
        "eps_stability",
    ],
    "Growth Metrics": [
        "revenue_growth_yoy",
        "growth_ebitda_growth_yoy",
        "operating_income_growth",
        "fcf_growth",
        "revenue_cagr_5y",
        "forward_revenue_growth",
        "revenue_vs_5y_avg",
        "revenue_acceleration",
    ],
    "Quality & Risk": [
        "piotroski_f_score",
        "altman_z_score",
        "altman_z_score_fy",
        "altman_z_trend",
        "has_goodwill_impairment",
        "has_asset_writedown",
        "has_restructuring",
        "goodwill_to_assets_pct",
        "intangible_intensity",
        "exceptional_items_to_ebitda",
        "current_ratio",
        "quick_ratio",
        "accounting_quality_score",
        "beta_stability_score",
    ],
    "Financial Distress": [
        "combined_distress_score",
        "liquidity_stress_score",
        "working_capital_trend",
        "cash_runway_months",
        "wc_deteriorating_flag",
        "accumulated_deficit_flag",
        "adequate_cash_buffer",
    ],
    "Accounting Quality": [
        "goodwill_change_rate",
        "restructuring_intensity",
        "exceptional_items_frequency",
        "merger_impact_ratio",
        "asset_sale_boost",
        "accounting_quality_score",
    ],
    "Leverage & Liquidity": [
        "debt_to_equity",
        "debt_to_assets",
        "equity_ratio",
        "interest_coverage",
        "cash_ratio",
        "working_capital_ratio",
        "debt_deleveraging",
        "debt_to_equity_trend",
    ],
    "Efficiency Ratios": [
        "asset_turnover",
        "inventory_turnover",
        "receivables_days",
        "working_capital_turns",
        "asset_turnover",
        "inventory_turnover",
        "receivables_days",
        "working_capital_turns",
        "cost_efficiency_score",
        "wc_efficiency_score",
    ],
    "Balance Sheet": [
        "assets_fq",
        "assets_yoy_growth",
        "assets_3y_cagr",
        "asset_quality_score",
        "balance_sheet_strength",
        "cash_to_assets_pct",
        "cash_vs_5y_avg",
        "inventory_yoy_change",
        "receivables_change_yoy",
        "retained_earnings_vs_5y",
    ],
    "Analyst Sentiment": [
        "analyst_bullish_pct",
        "analyst_neutral_pct",
        "analyst_bearish_pct",
        "analyst_conviction",
        "expected_upside_pt",
        "price_target_spread_pct",
        "price_target_revision_1m",
        "price_target_revision_3m",
        "analyst_rating_normalized",
        "analyst_coverage_quality",
        "eps_revision_momentum",
    ],
    "Price Target Dynamics": [
        "pt_momentum_1w",
        "pt_momentum_1m",
        "pt_momentum_3m",
        "pt_momentum_6m",
        "pt_momentum_1y",
        "pt_consensus_convergence",
        "analyst_coverage_change_1m",
        "analyst_coverage_change_3m",
        "analyst_coverage_trend",
    ],
    "Dividend Reliability": [
        "dividend_streak",
        "dividend_yield_ltm",
        "dividend_yield_ntm",
        "dividend_payout_ratio",
        "fcf_dividend_coverage",
        "buyback_yield",
        "total_shareholder_yield",
        "dividend_consistency",
        "dividend_yield_vs_5y_avg",
        "sustainable_dividend_flag",
    ],
    "Cash Flow": [
        "cfo_to_net_income",
        "fcf_to_net_income",
        "fcf_margin",
        "cfo_growth_yoy",
        "fcf_positive_ratio",
        "self_funding_ratio",
        "fcf_positive_years",
        "fcf_yield",
        "cash_flow_quality_score",
        "acquisition_pause_flag",
        "acquisition_to_fcf",
        "acquisitions_ltm_total",
        "acquisitions_vs_5y_avg",
        "acquisitions_yoy_growth",
        "capex_3y_trend",
        "capex_acceleration",
        "capex_cut_flag",
        "capex_qoq_growth",
        "capex_volatility",
        "capex_vs_5y_avg",
        "capex_yoy_growth",
        "cash_flow_quality_score",
        "cff_share_of_cf",
        "cfi_share_of_cf",
        "cfo_share_of_cf",
        "fcf_4q_improvement",
        "fcf_always_positive",
        "fcf_positive_years",
        "investment_efficiency",
        "ma_intensity_score",
        "organic_vs_inorganic",
        "overinvestment_flag",
        "self_funding_flag",
        "serial_acquirer_flag",
        "sustainable_ma_flag",
        "total_investment_to_cfo",
        "underinvestment_flag",
    ],
    "Employee Productivity": [
        "revenue_per_employee",
        "profit_per_employee",
        "ebitda_per_employee",
        "assets_per_employee",
        "fte_growth_1y_pct",
        "fte_growth_3y_pct",
        "workforce_stability",
        "productivity_trend",
    ],
    "Composite Scores": [
        "piotroski_f_score",
        "dilution_score",
        "quality_momentum_score",
        "earnings_quality_composite",
    ],
    "Volatility Surface": [
        "volatility_1m",
        "volatility_3m",
        "volatility_6m",
        "volatility_1y",
        "volatility_trend_short",
        "volatility_trend_long",
        "vol_ratio_3m_1y",
        "vol_hump",
        "beta_2y",
        "beta_term_structure",
        "beta_convexity",
        "realized_vs_implied_proxy",
        "beta_short_term_shift",
    ],
    "Tax Rate": [
        "effective_tax_rate_ltm",
        "effective_tax_rate_fy",
        "tax_rate_yoy_change",
        "tax_rate_qoq_change",
        "tax_rate_stability",
        "low_tax_flag",
        "tax_rate_trend_4q",
    ],
    "OpEx Temporal": [
        "opex_fq",
        "opex_ltm",
        "opex_fy",
        "opex_qoq_growth",
        "opex_yoy_growth",
        "opex_vs_revenue_trend",
        "sga_qoq_growth",
        "sga_yoy_growth",
        "operating_leverage_score",
    ],
    "Asset Sales": [
        "asset_sale_gain_loss_ltm",
        "asset_sale_frequency",
        "asset_sale_trend",
    ],
    "FCF Estimates": [
        "fcf_est_avg_fy1e",
        "fcf_est_avg_fy2e",
        "fcf_est_avg_fy3e",
        "fcf_est_avg_fy4e",
        "fcf_est_avg_fy5e",
        "fcf_est_cagr_5y",
        "fcf_est_trend",
    ],
    "Dividend History": [
        "div_yield_2fyind",
        "div_yield_3fyind",
        "div_yield_4fyind",
        "div_yield_5fyind",
        "div_yield_5y_trend",
        "div_yield_stability",
    ],
    "Interest Income Temporal": [
        "interest_income_fq",
        "interest_income_fy",
        "interest_income_qoq_growth",
        "interest_income_yoy_growth",
        "interest_income_to_revenue_trend",
    ],
    "Share Dilution": [
        "shares_yoy_change_pct",
        "net_buyback_flag",
        "shrs_out_1fy",
    ],
    "Forward Consensus": [
        "pe_ntm",
        "pe_est_fy1",
        "pe_forward_discount",
        "eps_gaap_vs_norm_ntm",
        "eps_gaap_vs_norm_fy1e",
        "forward_adjustment_trend",
        "ebitda_est_ntm",
        "ebitda_est_fy1e",
        "ev_ebitda_est_fy1",
        "ebitda_forward_growth",
        "earnings_revision_divergence",
        "forward_pe_vs_sector_proxy",
    ],
    "Employment Dynamics": [
        "fte_acceleration",
        "fte_growth_2y_pct",
        "headcount_vs_revenue",
        "hiring_intensity",
        "layoff_risk_flag",
        "productivity_trend",
        "rapid_hiring_flag",
        "sustainable_growth_flag",
        "workforce_efficiency_gain",
        "workforce_volatility",
    ],
    "GAAP vs Adjusted": [
        "earnings_quality_score",
        "earnings_quality_warning",
        "ebit_adjustment_pct_1fqfq",
        "ebit_adjustment_pct_1fy",
        "ebit_adjustment_pct_2fqfq",
        "ebit_adjustment_pct_2fy",
        "ebit_adjustment_pct_3fqfq",
        "ebit_adjustment_pct_3fy",
        "ebit_adjustment_pct_4fqfq",
        "ebit_adjustment_pct_4fy",
        "ebit_adjustment_pct_fq",
        "ebit_adjustment_pct_fy",
        "ebit_adjustment_pct_ltm",
        "ebitda_adjustment_pct_1fqfq",
        "ebitda_adjustment_pct_1fy",
        "ebitda_adjustment_pct_2fqfq",
        "ebitda_adjustment_pct_2fy",
        "ebitda_adjustment_pct_3fqfq",
        "ebitda_adjustment_pct_3fy",
        "ebitda_adjustment_pct_4fqfq",
        "ebitda_adjustment_pct_4fy",
        "ebitda_adjustment_pct_fq",
        "ebitda_adjustment_pct_fy",
        "ebitda_adjustment_pct_ltm",
        "eps_adjustment_pct",
        "eps_adjustment_spread_1fqfq",
        "eps_adjustment_spread_1fy",
        "eps_adjustment_spread_2fqfq",
        "eps_adjustment_spread_2fy",
        "eps_adjustment_spread_3fqfq",
        "eps_adjustment_spread_3fy",
        "eps_adjustment_spread_4fqfq",
        "eps_adjustment_spread_4fy",
        "eps_adjustment_spread_fq",
        "eps_adjustment_spread_fy",
        "eps_adjustment_spread_ltm",
        "forward_eps_gaap_adj_spread",
        "net_income_adjustment_pct",
        "net_income_adjustment_ratio_1fqfq",
        "net_income_adjustment_ratio_1fy",
        "net_income_adjustment_ratio_2fqfq",
        "net_income_adjustment_ratio_2fy",
        "net_income_adjustment_ratio_3fqfq",
        "net_income_adjustment_ratio_3fy",
        "net_income_adjustment_ratio_4fqfq",
        "net_income_adjustment_ratio_4fy",
        "net_income_adjustment_ratio_5yavgfq",
        "net_income_adjustment_ratio_fq",
        "net_income_adjustment_ratio_fy",
        "net_income_adjustment_ratio_ltm",
    ],
    "Interest Income": [
        "interest_coverage_ratio",
        "interest_expense_ltm",
        "interest_expense_to_revenue",
        "interest_income_ltm",
        "interest_income_to_revenue",
        "net_interest_income",
        "net_interest_margin_proxy",
        "inv_income_ltm",
        "interest_income_fq",
        "interest_income_fy",
        "interest_income_qoq_growth",
        "interest_income_yoy_growth",
        "interest_income_to_revenue_trend",
        "inv_income_trend_3y",
        "inv_income_positive_quarters",
        "financial_company_proxy",
    ],
    "Revenue Forecasting": [
        "revenue_avg_med_diff_pct",
        "revenue_consensus_strength",
        "revenue_est_avg_fy1e",
        "revenue_est_avg_ntm",
        "revenue_est_med_fy1e",
        "revenue_est_med_ntm",
        "revenue_revision_trend",
        "revenue_vs_current",
        "consensus_revenue_growth",
        "ebit_estimate_spread",
        "ebitda_est_vs_actual",
        "estimate_confidence_score",
        "forward_ebitda_margin",
        "forward_revenue_multiple",
        "revenue_acceleration",
        "revenue_beat_potential",
        "revenue_est_revision_trend",
        "revenue_est_spread",
        "revenue_estimate_count",
        "revenue_guidance_gap",
        "revenue_1fqfq",
        "revenue_1fy",
        "revenue_2fqfq",
        "revenue_2fy",
        "revenue_2y_growth",
        "revenue_3fqfq",
        "revenue_3fy",
        "revenue_3y_growth",
        "revenue_4fqfq",
        "revenue_4fy",
        "revenue_4q_avg",
        "revenue_4q_trend",
        "revenue_4y_growth",
        "revenue_5y_avg",
        "revenue_accelerating_flag",
        "revenue_cagr_3y",
        "revenue_cagr_4y",
        "revenue_fq",
        "revenue_fq_vs_4q_avg",
        "revenue_fy",
        "revenue_growth_flag",
        "revenue_ltm",
        "revenue_positive_qoq_streak",
        "revenue_qoq_2q",
        "revenue_qoq_3q",
        "revenue_qoq_4q",
        "revenue_qoq_growth",
        "revenue_stability_score",
        "revenue_yoy_quarterly",
    ],
    "Temporal Patterns": [
        "fiscal_quarter",
        "fiscal_month",
        "fiscal_year",
        "days_to_earnings",
        "earnings_report_recency",
        "reporting_lag",
        "fiscal_year_progress",
        "days_since_last_report",
        "days_to_fy_end",
        "earnings_season_flag",
        "fiscal_quarter_progress",
        "is_fy_end_month",
        "is_quarter_end_month",
        "post_earnings_window",
        "pre_earnings_window",
        "reporting_freshness_score",
    ],
    "Direct Reference": [
        "current_fiscal_quarter",
        "dividend_record_currency",
        "dividend_record_amount",
        "dividend_per_share_ltm",
        "market_cap_country_r",
        "rel_volume",
        "one_day_pct",
        "total_return_ytd",
        "total_return_5y",
        "total_return_10y",
        "tot_return_pct_cagr_3y",
        "tot_return_pct_cagr_10y",
        "total_revenues_cagr_5y_fy",
        "analyst_rating",
        "price_target_count",
        "num_strong_buys_ratings",
        "num_buys_ratings",
        "num_hold_ratings",
        "num_sell_ratings",
        "num_strong_sell_ratings",
        "eps_norm_est_num_fy1e",
        "eps_gaap_est_avg_ntm",
        "eps_gaap_est_avg_fy1e",
        "eps_norm_est_avg_ntm",
        "eps_norm_est_avg_fy1e",
    ],
}


def get_fallback_feature_categories() -> dict[str, list[str]]:
    """Return the canonical fallback feature categories.

    This is the single source of truth for hardcoded category → feature
    mappings used when the database is unreachable.
    """
    return dict(FALLBACK_FEATURE_CATEGORIES)


def get_features_for_category(category: str) -> list[str]:
    """Return fallback feature names for a single category.

    Parameters
    ----------
    category : str
        Category name (e.g. ``"Quality & Risk"``).

    Returns
    -------
    list[str]
        Feature aliases, or empty list if category is unknown.
    """
    return list(FALLBACK_FEATURE_CATEGORIES.get(category, []))


# ═══════════════════════════════════════════════════════════════════════════════
# Column Alias Resolution (visualization-oriented)
# ═══════════════════════════════════════════════════════════════════════════════


# Merged column alias map — single source of truth for logical → actual column resolution.
# Populated from VIZ_REQUIREMENTS.column_aliases at module load time.
_COLUMN_ALIASES: dict[str, list[str]] = {}
for _vr in VIZ_REQUIREMENTS.values():
    for _logical, _aliases in _vr.column_aliases.items():
        if _logical not in _COLUMN_ALIASES:
            _COLUMN_ALIASES[_logical] = list(_aliases)
        else:
            # Merge new aliases that aren't already present
            for _a in _aliases:
                if _a not in _COLUMN_ALIASES[_logical]:
                    _COLUMN_ALIASES[_logical].append(_a)


def get_column_aliases() -> dict[str, list[str]]:
    """Return the merged column alias map derived from visualization requirements.

    Each key is a logical column name; values are candidate aliases
    ordered by preference.  The catalog builds this from all registered
    ``VisualizationFeatureRequirement.column_aliases``.
    """
    return dict(_COLUMN_ALIASES)


def resolve_column_from_catalog(
    df: pd.DataFrame,
    logical_name: str,
) -> str | None:
    """Resolve a logical column name using the catalog alias map.

    Checks *logical_name* directly first, then walks through aliases
    from ``_COLUMN_ALIASES``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check columns against.
    logical_name : str
        Logical / expected column name.

    Returns
    -------
    str | None
        The resolved column name present in *df*, or ``None``.
    """
    if logical_name in df.columns:
        return logical_name
    for alias in _COLUMN_ALIASES.get(logical_name, []):
        if alias in df.columns:
            return alias
    return None


def columns_for_viz(
    viz_key: str, catalog: FeatureViewCatalog | None = None
) -> list[str]:
    """Resolve the column names a visualization module needs.

    When a loaded ``FeatureViewCatalog`` is provided, columns are resolved
    from the database categories.  Otherwise falls back to
    ``VisualizationFeatureRequirement.fallback_columns``.

    Parameters
    ----------
    viz_key : str
        Key into ``VIZ_REQUIREMENTS``.
    catalog : FeatureViewCatalog or None
        Optional pre-loaded catalog.

    Returns
    -------
    list[str]
        Ordered list of column names.
    """
    req = VIZ_REQUIREMENTS.get(viz_key)
    if req is None:
        return []

    if catalog is not None and catalog._loaded:
        columns: list[str] = []
        for cat in req.required_categories:
            columns.extend(catalog.category_columns.get(cat, []))
        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for c in columns:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique or list(req.fallback_columns)

    return list(req.fallback_columns)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature View Catalog
# ═══════════════════════════════════════════════════════════════════════════════

# Valid SQL identifier pattern — used to reject unsafe schema names.
_VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _validate_identifier(name: str, label: str = "identifier") -> str:
    """Validate that *name* is a safe SQL identifier (no injection risk).

    Raises
    ------
    ValueError
        If *name* doesn't match the allowed pattern.
    """
    if not _VALID_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid SQL {label}: {name!r} — must match {_VALID_IDENTIFIER_RE.pattern}"
        )
    return name


@dataclass
class FeatureViewCatalog:
    """Runtime-resolved mapping from categories → column names.

    Populated once at pipeline startup from ``calculated_features_registry``
    and cached for the process lifetime.

    Attributes
    ----------
    category_columns : dict[str, list[str]]
        category name → list of feature column aliases.
    view_columns : dict[str, list[str]]
        view name → list of feature column names (from information_schema).
    """

    category_columns: dict[str, list[str]] = field(default_factory=dict)
    view_columns: dict[str, list[str]] = field(default_factory=dict)
    _loaded: bool = field(default=False, repr=False, compare=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def load_from_db(
            self,
            db_url: Optional[str] = None,
            schema: Optional[str] = None,
    ) -> None:
        """Populate the catalog from SQL metadata tables."""
        if create_engine is None or text is None:
            logger.warning("SQLAlchemy not available — catalog using fallbacks only")
            return

        db_url = db_url or os.environ.get("DB_URL")
        schema = schema or os.environ.get("DB_SCHEMA")
        if not db_url:
            logger.warning("DB_URL not configured — catalog using fallbacks only")
            return
        if not schema:
            logger.warning("DB_SCHEMA not configured — catalog using fallbacks only")
            return

        # Reject anything that isn't a plain SQL identifier
        _validate_identifier(schema, "schema")

        engine = create_engine(db_url)
        try:
            # Build new dicts locally, then swap atomically
            new_category_columns: dict[str, list[str]] = {}
            new_view_columns: dict[str, list[str]] = {}

            # 1) Feature categories from calculated_features_registry
            registry_df = pd.read_sql(
                text(
                    "SELECT feature_alias, category "
                    "FROM calculated_features_registry "

                ),
                engine,
            )
            for category, group in registry_df.groupby("category"):
                new_category_columns[str(category)] = (
                    group["feature_alias"].tolist()
                )

            # 2) View columns from information_schema
            with engine.connect() as conn:
                # Identifier columns from the dedicated vw_identifier_columns view
                id_cols_result = conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = :schema "
                        "AND table_name = 'vw_identifier_columns'"
                    ),
                    {"schema": schema},
                )
                id_cols = {row[0] for row in id_cols_result}

                views_result = conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.views "
                        "WHERE table_schema = :schema "
                        "AND table_name LIKE 'vw_features_%'"
                    ),
                    {"schema": schema},
                )
                view_names = [row[0] for row in views_result]

                for vn in view_names:
                    cols_result = conn.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_schema = :schema AND table_name = :view "
                            "ORDER BY ordinal_position"
                        ),
                        {"schema": schema, "view": vn},
                    )
                    new_view_columns[vn] = [
                        row[0] for row in cols_result if row[0] not in id_cols
                    ]

            # Atomic swap under lock
            with self._lock:
                self.category_columns = new_category_columns
                self.view_columns = new_view_columns
                self._loaded = True

            logger.info(
                "FeatureViewCatalog loaded: %d categories, %d views, %d total features",
                len(self.category_columns),
                len(self.view_columns),
                sum(len(v) for v in self.category_columns.values()),
            )

        except SQLAlchemyError as e:
            logger.warning("FeatureViewCatalog DB load failed: %s — using fallbacks", e)
            if not self.category_columns:
                self.category_columns = dict(FALLBACK_FEATURE_CATEGORIES)
        finally:
            engine.dispose()

    def columns_for_model(self, model_key: str) -> list[str]:
        """Resolve all column names a model needs from its category requirements.

        Falls back to ``ModelFeatureRequirement.fallback_columns`` when the
        catalog hasn't been loaded from the database.
        """
        req = MODEL_REQUIREMENTS.get(model_key)
        if req is None:
            return []

        if not self._loaded:
            return list(req.fallback_columns)

        columns: list[str] = []
        for cat in (*req.required_categories, *req.optional_categories):
            columns.extend(self.category_columns.get(cat, []))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for c in columns:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique or list(req.fallback_columns)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema-Aware Auto-Enrichment (replaces repeated _merge_missing_columns calls)
# ═══════════════════════════════════════════════════════════════════════════════


def auto_enrich_for_model(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame | None,
    model_key: str,
    catalog: FeatureViewCatalog,
) -> pd.DataFrame:
    """Merge all columns a model needs from *source_df* into *target_df*.

    Replaces the 6+ repetitive merge blocks in the ``run_*`` functions
    with a single schema-driven call.

    Parameters
    ----------
    target_df : pd.DataFrame
        Primary model input DataFrame.
    source_df : pd.DataFrame or None
        Feature superset (e.g. from ``load_all_feature_views``).
    model_key : str
        Key into ``MODEL_REQUIREMENTS`` (e.g. ``"price_target_achievement"``).
    catalog : FeatureViewCatalog
        Pre-loaded catalog with column → category mappings.

    Returns
    -------
    pd.DataFrame
        Enriched copy of *target_df*.
    """
    if source_df is None or source_df.empty:
        return target_df
    if "isin" not in target_df.columns or "isin" not in source_df.columns:
        return target_df

    # v3.10: Deduplicate on isin (unique per security) instead of ticker
    target_df = target_df.drop_duplicates(subset="isin")

    needed = catalog.columns_for_model(model_key)
    missing = [
        c for c in needed if c not in target_df.columns and c in source_df.columns
    ]

    if not missing:
        return target_df

    subset = source_df[["isin"] + missing].drop_duplicates(subset="isin")
    result = target_df.merge(subset, on="isin", how="left")

    req = MODEL_REQUIREMENTS.get(model_key)
    label = req.model_name if req else model_key
    logger.info(
        "%s: auto-enriched %d columns from feature views (of %d needed)",
        label,
        len(missing),
        len(needed),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton instance (initialized lazily in pipeline startup)
# ═══════════════════════════════════════════════════════════════════════════════

_catalog_instance: FeatureViewCatalog | None = None


def get_feature_catalog(
    db_url: Optional[str] = None,
    schema: str = "public",
) -> FeatureViewCatalog:
    """Get or create the global FeatureViewCatalog singleton."""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = FeatureViewCatalog()
        _catalog_instance.load_from_db(db_url=db_url, schema=schema)
    return _catalog_instance


def reset_feature_catalog() -> None:
    """Reset the singleton (useful in tests)."""
    global _catalog_instance
    _catalog_instance = None
