"""
Analytics module for feature engineering analysis.
This module provides interactive visualizations, probabilistic models,
and statistical analytics for financial feature analysis.
Modules:
- feature_analytics: Core visualization dashboards
- data_utils: Data loading, preprocessing, and standardized exports
- statistical_analysis: Bayesian, MCMC, and advanced statistics
- screening: Stock screening and filtering
- optimized_ops: Performance-optimized operations
- visualizations: Additional visualization modules
  - profitability: Margin and profitability charts
  - technical: Technical analysis charts
  - temporal_analysis: Time series analysis
Export Framework:
The new ExportConfig class centralizes all export settings (database, CSV, JSON)
for consistent handling across the analytics pipeline. All export functions
accept ExportConfig for unified configuration:
  - export_to_db(): PostgreSQL analytics schema
  - export_to_csv(): Comma-separated values to outputs
  - export_to_json(): JSON format with configurable orientation/indentation
"""

from analytics.data_utils import (  # noqa: F401
    ANALYTICS_EXPORT_TABLES,
    VW_FEATURES_VIEWS,
    ExportConfig,
    _get_fallback_feature_categories,
    aggregate_probability_results,
    backfill_feature_columns,
    compare_registry_with_local,
    compute_metric_statistics,
    export_to_csv,
    export_to_db,
    export_to_json,
    export_view_analytics_results,
    get_identifier_cols_set,
    get_view_category_labels,
    get_view_category_mapping,
    get_view_feature_cols,
    load_all_feature_views,
    load_feature_categories_from_db,
    load_feature_data_from_db,
    load_identifier_columns,
    reorder_with_identifiers,
    safe_get_column,
    validate_feature_alignment,
)

from analytics.feature_analytics import (  # noqa: F401
    FEATURE_CATEGORIES,
    PLOTLY_TEMPLATE,
    create_composite_quality_score,
    create_interactive_momentum_dashboard,
    create_interactive_valuation_heatmap,
    create_leverage_liquidity_quadrant,
    create_summary_dashboard,
    ensure_subplot_data,
)

from analytics.optimized_ops import (  # noqa: F401
    dataframe_hash,
    fast_monte_carlo_simulation,
    fast_ruin_probability,
    get_optimization_status,
    load_feature_data_from_db_cached,
    vectorized_percentile_rank,
    vectorized_zscore,
)

from analytics.screening import (  # noqa: F401
    create_enhanced_screener,
    create_sector_relative_ranking,
    rank_stocks_by_composite_score,
    screen_dividend_quality,
    screen_earnings_quality,
    screen_fcf_growth_compounders,
    screen_financial_health,
    screen_garp_opportunities,
    screen_growth_momentum,
    screen_high_yield_safe_dividends,
    screen_integrity_filtered_growth,
    screen_low_volatility_quality,
    screen_total_return_leaders,
    screen_valuation_reversion_candidates,
    screen_value_opportunities,
)

from analytics.statistical_analysis import (  # noqa: F401
    BayesianTechnicalResampler,
    ResampledReturnDistribution,
    analyze_accounting_anomalies,
    analyze_distress_distribution,
    analyze_employee_productivity_frontier,
    analyze_reporting_lag_sentiment,
    bayesian_category_analysis,
    bayesian_earnings_beat_model,
    calculate_conditional_probabilities,
    calculate_ruin_probability,
    detect_accounting_anomalies,
    export_probability_view_results,
    fit_distributions_by_category,
    fit_gaussian_copula,
    hierarchical_mcmc_by_sector,
    hierarchical_mcmc_multi_level,
    kalman_filter_price_target,
    kalman_momentum_filter,
    mcmc_student_t,
    metropolis_hastings_sampler,
    monte_carlo_price_target_simulation,
    parallel_mcmc_chains,
    resampled_posterior_returns,
    run_all_views_probability_analytics,
    run_category_probability_analytics,
)

# ---------------------------------------------------------------------------
# Stub helper for optional-dependency fallbacks
# ---------------------------------------------------------------------------


def _unavailable_stub(module_label: str):
    """Return a function that raises ImportError with a clear message."""

    def _stub(*args, **kwargs):
        raise ImportError(f"{module_label} is not available")

    return _stub


# ---------------------------------------------------------------------------
# InferenceData schema (ArviZ / xarray bridge) — optional dependency
# ---------------------------------------------------------------------------
_INFERENCE_LABEL = "inference_schema module (ArviZ)"

_INFERENCE_BUILDER_NAMES = [
    "build_beat_probability_inference_data",
    "build_credit_risk_inference_data",
    "build_accounting_anomaly_inference_data",
    "build_monte_carlo_inference_data",
    "build_category_analysis_inference_data",
    "build_feature_view_inference_data",
    "build_resampled_technical_inference_data",
]

_INFERENCE_LOADER_NAMES = [
    "load_equity_coordinates_from_db",
    "load_feature_coordinates_from_db",
    "load_identifier_coordinates_from_db",
    "load_equities_schema_metadata_from_db",
    "load_feature_registry_metadata_from_db",
    "load_feature_view_spec_from_db",
    "load_mv_equities_spec_from_db",
    "summarize_inference_data",
]

try:
    from analytics.inference_schema import (  # noqa: F401
        ARVIZ_AVAILABLE,
        FEATURE_VIEW_REGISTRY,
        EquitiesMaterializedViewSpec,
        EquitiesSchemaMetadata,
        EquityCoordinates,
        FeatureCoordinates,
        FeatureRegistryMetadata,
        FeatureViewSpec,
        IdentifierCoordinates,
        build_accounting_anomaly_inference_data,
        build_beat_probability_inference_data,
        build_category_analysis_inference_data,
        build_credit_risk_inference_data,
        build_feature_view_inference_data,
        build_monte_carlo_inference_data,
        build_resampled_technical_inference_data,
        load_equities_schema_metadata_from_db,
        load_equity_coordinates_from_db,
        load_feature_coordinates_from_db,
        load_feature_registry_metadata_from_db,
        load_feature_view_spec_from_db,
        load_identifier_coordinates_from_db,
        load_mv_equities_spec_from_db,
        summarize_inference_data,
    )
except ImportError:
    ARVIZ_AVAILABLE = False

    # Coordinate & metadata classes → None
    EquityCoordinates = None
    FeatureCoordinates = None
    IdentifierCoordinates = None
    EquitiesSchemaMetadata = None
    FeatureRegistryMetadata = None
    FeatureViewSpec = None
    EquitiesMaterializedViewSpec = None

    FEATURE_VIEW_REGISTRY = {}

    # Generate stub functions for every builder and loader
    _ns = globals()
    for _name in _INFERENCE_BUILDER_NAMES + _INFERENCE_LOADER_NAMES:
        _ns[_name] = _unavailable_stub(_INFERENCE_LABEL)
    del _ns, _name

# ---------------------------------------------------------------------------
# Probability Analytics — optional dependency
# ---------------------------------------------------------------------------
_PROBABILITY_LABEL = "probability_analytics module"

try:
    from analytics.probability_analytics import (  # noqa: F401
        CreditRiskProbabilityModel,
        EarningsBeatProbabilityModel,
    )
except ImportError:
    EarningsBeatProbabilityModel = None
    CreditRiskProbabilityModel = None
