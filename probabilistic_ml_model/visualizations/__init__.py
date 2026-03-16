"""
Visualization modules for the probabilistic_ml_model package.

Submodules
----------
- ``expected_returns_viz`` — Pipeline charts (MC distribution, sector heatmaps, etc.)
- ``probability_viz`` — Probabilistic ArviZ-backed visualizations
- ``arviz_diagnostics`` — ArviZ diagnostic panels
- ``earnings_quality`` — Earnings quality charts
- ``quality_risk`` — Piotroski F-Score, Altman Z-Score, anomaly dashboards
- ``valuation`` — Valuation multiples, distribution, relative matrix
- ``growth_analysis`` — Growth waterfall, consistency, acceleration charts
"""

from __future__ import annotations

import importlib
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Shared helpers available at package level
# ---------------------------------------------------------------------------
from probabilistic_ml_model.visualizations._shared import (  # noqa: E402
    PLOTLY_TEMPLATE,
    COLORS,
    MV_COLUMN_ALIASES,
    resolve_column,
    create_no_data_figure,
)

_shared_exports: list[str] = [
    "PLOTLY_TEMPLATE",
    "COLORS",
    "MV_COLUMN_ALIASES",
    "resolve_column",
    "create_no_data_figure",
]

# ---------------------------------------------------------------------------
# Registry: (module_path, [names_to_import])
# ---------------------------------------------------------------------------
_IMPORT_REGISTRY: Sequence[tuple[str, Sequence[str]]] = (
    # Quality risk visualization functions
    (
        ".quality_risk",
        [
            "create_piotroski_fscore_breakdown",
            "create_altman_zscore_distribution",
            "create_quality_risk_quadrant",
            "create_beneish_mscore_analysis",
            "create_risk_tier_sunburst",
            "create_distress_early_warning_dashboard",
            "create_accounting_anomaly_dashboard",
            "create_anomaly_severity_dashboard",
        ],
    ),
    # Valuation visualization functions
    (
        ".valuation",
        [
            "create_valuation_multiples_comparison",
            "create_valuation_distribution_dashboard",
            "create_relative_valuation_matrix",
            "create_valuation_vs_growth_quadrant",
            "create_historical_valuation_percentile",
        ],
    ),
    # Growth analysis visualization functions
    (
        ".growth_analysis",
        [
            "create_growth_waterfall_chart",
            "create_growth_consistency_matrix",
            "create_growth_vs_profitability_quadrant",
            "create_growth_acceleration_chart",
            "create_sustainable_growth_analysis",
        ],
    ),
    # Probabilistic ArviZ-backed visualization functions
    (
        ".probability_viz",
        [
            "create_posterior_return_forest",
            "create_beat_probability_posterior",
            "create_ruin_probability_diagnostic",
            "create_mcse_convergence_panel",
            "create_bayesian_category_ridge",
            "create_tri_model_posterior_comparison",
            "create_feature_view_posterior_panel",
            "create_anomaly_conditional_probability_chart",
            "create_mcmc_anomaly_posterior_chart",
            "create_mcmc_credit_risk_chart",
            "create_mcmc_dividend_cut_chart",
            "create_mcmc_price_target_chart",
            "create_mcmc_category_posterior_chart",
        ],
    ),
    # Expected returns pipeline visualization functions
    (
        ".expected_returns_viz",
        [
            "create_mc_return_distribution",
            "create_sector_risk_reward_scatter",
            "create_kalman_vs_raw_scatter",
            "create_tri_model_agreement_histogram",
            "create_strong_consensus_bar",
            "create_sector_heatmap",
            "create_var_analysis",
            "create_beat_vs_achievement_scatter",
            "create_model_dispersion_dashboard",
            "create_return_distribution_fit_chart",
            "create_sector_return_analytics_heatmap",
            "create_screening_summary_chart",
            "create_price_target_drift_dashboard",
        ],
    ),
    # ArviZ-backed diagnostic visualizations
    (
        ".arviz_diagnostics",
        [
            "build_screening_inference_data",
            "create_screening_posterior_ridge",
            "create_productivity_frontier_posterior",
            "build_resampled_posterior_idata",
            "create_resampled_posterior_diagnostics",
            "create_resampled_sector_forest",
            "build_alignment_inference_data",
            "create_model_alignment_arviz_panel",
            "create_agreement_posterior_by_sector",
            "create_hierarchical_shrinkage_diagnostic",
            "create_multi_level_mcmc_comparison",
            "create_mcmc_convergence_panel_arviz",
            "build_category_analytics_idata",
            "create_category_posterior_diagnostics",
            "create_cross_category_summary",
        ],
    ),
)

# ---------------------------------------------------------------------------
# Dynamic import loop
# ---------------------------------------------------------------------------
_current_module = sys.modules[__name__]
_dynamic_exports: list[str] = []

for _module_path, _names in _IMPORT_REGISTRY:
    try:
        _mod = importlib.import_module(_module_path, package=__name__)
        for _name in _names:
            _obj = getattr(_mod, _name)
            setattr(_current_module, _name, _obj)
            _dynamic_exports.append(_name)
    except (ImportError, AttributeError):
        continue

__all__ = _shared_exports + _dynamic_exports
