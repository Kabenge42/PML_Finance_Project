"""
Statistical functions for the probabilistic_ml_model package.

Submodules
----------
- ``statistical_analysis`` — Bayesian, MCMC, Kalman, Copula, distribution fitting
- ``ensemble`` — Tri-/quad-model alignment, expected returns summary builders
- ``screening`` — Stock screening strategies
- ``probability_analytics`` — Probability models (earnings, credit, dividend, anomaly)
"""

from probabilistic_ml_model.statistical_functions.statistical_models import (
    bayesian_category_analysis,
    bayesian_earnings_beat_model,
    metropolis_hastings_sampler,
    mcmc_student_t,
    hierarchical_mcmc_by_sector,
    hierarchical_mcmc_multi_level,
    parallel_mcmc_chains,
    kalman_momentum_filter,
    kalman_filter_price_target,
    fit_gaussian_copula,
    fit_distributions_by_category,
    monte_carlo_price_target_simulation,
    calculate_ruin_probability,
    calculate_conditional_probabilities,
    resampled_posterior_returns,
    run_category_probability_analytics,
    analyze_employee_productivity_frontier,
    analyze_reporting_lag_sentiment,
    detect_accounting_anomalies,
    analyze_distress_distribution,
    BayesianTechnicalResampler,
)

from probabilistic_ml_model.statistical_functions.ensemble_models import (
    build_tri_model_alignment,
    build_quad_model_alignment,
    build_expected_returns_summary,
    extract_strong_consensus,
)

from probabilistic_ml_model.statistical_functions.screening import (
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

from probabilistic_ml_model.statistical_functions.probability_models import (
    AccountingAnomalyProbabilityModel,
    CategoryProbabilityAnalyzer,
    CreditRiskProbabilityModel,
    DividendCutProbabilityModel,
    EarningsBeatProbabilityModel,
    EPSStreakAnalyzer,
    PriceTargetAchievementModel,
    ResampledBeatProbabilityModel,
    create_earnings_probability_dashboard,
    export_probability_analytics_results,
)

__all__ = [
    # statistical_models
    "bayesian_category_analysis",
    "bayesian_earnings_beat_model",
    "metropolis_hastings_sampler",
    "mcmc_student_t",
    "hierarchical_mcmc_by_sector",
    "hierarchical_mcmc_multi_level",
    "parallel_mcmc_chains",
    "kalman_momentum_filter",
    "kalman_filter_price_target",
    "fit_gaussian_copula",
    "fit_distributions_by_category",
    "monte_carlo_price_target_simulation",
    "calculate_ruin_probability",
    "calculate_conditional_probabilities",
    "resampled_posterior_returns",
    "run_category_probability_analytics",
    "analyze_employee_productivity_frontier",
    "analyze_reporting_lag_sentiment",
    "detect_accounting_anomalies",
    "analyze_distress_distribution",
    "BayesianTechnicalResampler",
    # ensemble_models
    "build_tri_model_alignment",
    "build_quad_model_alignment",
    "build_expected_returns_summary",
    "extract_strong_consensus",
    # screening
    "create_enhanced_screener",
    "create_sector_relative_ranking",
    "rank_stocks_by_composite_score",
    "screen_dividend_quality",
    "screen_earnings_quality",
    "screen_fcf_growth_compounders",
    "screen_financial_health",
    "screen_garp_opportunities",
    "screen_growth_momentum",
    "screen_high_yield_safe_dividends",
    "screen_integrity_filtered_growth",
    "screen_low_volatility_quality",
    "screen_total_return_leaders",
    "screen_valuation_reversion_candidates",
    "screen_value_opportunities",
    # probability_models
    "AccountingAnomalyProbabilityModel",
    "CategoryProbabilityAnalyzer",
    "CreditRiskProbabilityModel",
    "DividendCutProbabilityModel",
    "EarningsBeatProbabilityModel",
    "EPSStreakAnalyzer",
    "PriceTargetAchievementModel",
    "ResampledBeatProbabilityModel",
    "create_earnings_probability_dashboard",
    "export_probability_analytics_results",
]
