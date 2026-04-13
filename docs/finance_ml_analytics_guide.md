# Probabilistic ML Model Guide

## Overview

This guide provides an overview of the probabilistic machine-learning module and its key features.

### Module Organization

```
PML_Finance_Project/
├── analytics/                      # Feature analytics, screening, statistics, visualizations
│   ├── __init__.py                 # Package exports & optional-dep stubs
│   ├── data_utils.py               # Data loading, preprocessing, export framework
│   ├── statistical_analysis.py     # Bayesian, MCMC, Kalman, Copula
│   ├── screening.py                # 15 stock screeners
│   ├── feature_analytics.py        # Visualization dashboards
│   ├── probability_analytics.py    # Probability models
│   ├── inference_schema.py         # ArviZ / xarray InferenceData bridge
│   ├── optimized_ops.py            # Performance optimizations
│   └── visualizations/             # Modular visualization sub-package (12 modules)
├── probabilistic_ml_model/         # Probabilistic ML models & pipeline
│   ├── __init__.py                 # Package init & optional-dep stubs
│   ├── config.py                   # Pipeline configuration
│   ├── pipeline_runners.py         # Orchestration for model execution
│   ├── utils.py                    # Shared utilities
│   ├── logging_config.py           # Logging setup
│   ├── optimized_ops.py            # Performance optimizations
│   ├── pml_models/                 # Model implementations
│   │   ├── BaselineProbabilityModel.py
│   │   ├── ProbabilisticLinearRegressionModel.py
│   │   ├── KalmanFilterModel.py
│   │   ├── DCF_PriceTargetModel.py
│   │   ├── EarningsBeatModel.py
│   │   ├── DividendSafetyModel.py
│   │   ├── PriceTargetModel.py
│   │   ├── AccountingAnomalyModel.py
│   │   ├── CreditRiskModel.py
│   │   └── MonteCarloSimulation.py
│   ├── data_utils/                 # Data loading & inference schema
│   ├── statistical_functions/      # Screening, statistics, ensemble, probability analytics
│   └── visualizations/             # Model-specific visualizations (8 modules, ArviZ 1.0)

expected_returns_v3.py          # Automated expected returns pipeline v3.1 (4528 lines) [EXPANDED]

## Module Descriptions

### 1. `data_utils.py`

**Purpose**: Data loading, preprocessing, validation, and export framework

**Key Functions**:

- `load_feature_data_from_db()` - Load data from PostgreSQL materialized view
- `backfill_feature_columns()` - Fill missing columns with calculated values
- `compute_metric_statistics()` - Calculate comprehensive statistics for features
- `validate_feature_alignment()` - Check feature coverage by category
- `safe_get_column()` - Safely retrieve columns with fallback options

**Export Framework** (New):

- `ExportConfig` — Centralized export configuration dataclass (database, CSV, JSON settings)
- `export_to_db()` — Export DataFrame to PostgreSQL analytics schema
- `export_to_csv()` — Export to CSV in `outputs/analytics/views/`
- `export_to_json()` — Export to JSON with configurable orientation/indentation
- `reorder_with_identifiers()` — Reorder DataFrame columns with identifiers first
- `load_identifier_columns()` — Load identifier column names from DB
- `get_identifier_cols_set()` — Get identifier columns as a set
- `load_feature_categories_from_db()` — Load feature categories from `calculated_features_registry`
- `compare_registry_with_local()` — Compare DB registry with local fallback categories

**Example Usage**:

```python
from finance_ml.analytics.data_utils import (
    load_feature_data_from_db,
    backfill_feature_columns,
    compute_metric_statistics,
    ExportConfig,
    export_to_db,
    export_to_csv,
)

# Load data
df = load_feature_data_from_db(earnings_date_filter="2026-01-01")

# Backfill missing columns
df = backfill_feature_columns(df)

# Get statistics
stats = compute_metric_statistics(df['p_e_ratio'])
print(f"Mean P/E: {stats['mean']:.2f}")

# Export framework
config = ExportConfig(db_url="postgresql+psycopg2://...", schema="analytics")
export_to_db(df, table_name="expected_returns_summary", config=config)
export_to_csv(df, filename="expected_returns_summary", config=config)
```

---

### 2. `statistical_analysis.py`

**Purpose**: Advanced statistical analysis including Bayesian methods, MCMC, Monte Carlo simulations, and resampled
posteriors

**Key Functions**:

- `bayesian_category_analysis()` - Bayesian parameter estimation with conjugate priors
- `metropolis_hastings_sampler()` - MCMC sampling for posterior distributions
- `mcmc_student_t()` - MCMC with Student's t distribution (for heavy tails)
- `hierarchical_mcmc_by_sector()` - Hierarchical Bayesian modeling by sector
- `fit_distributions_by_category()` - Fit and select best distribution using AIC
- `calculate_ruin_probability()` - Investor's ruin probability (Gambler's Ruin)
- `calculate_conditional_probabilities()` - P(Distress | Feature) analysis
- `monte_carlo_price_target_simulation()` — Monte Carlo price target simulation
- `bayesian_earnings_beat_model()` — Bayesian earnings beat model
- `analyze_distress_distribution()` — Financial distress distribution analysis
- `analyze_employee_productivity_frontier()` — Employee productivity frontier analysis
- `detect_accounting_anomalies()` / `analyze_accounting_anomalies()` — Accounting anomaly detection
- `analyze_reporting_lag_sentiment()` — Reporting lag sentiment analysis
- `run_category_probability_analytics()` / `run_all_views_probability_analytics()` — Per-category/view analytics
- `export_probability_view_results()` — Export probability view results
- `BayesianTechnicalResampler` — Resampled Bayesian technical returns class
- `ResampledReturnDistribution` — Resampled return distribution dataclass
- `resampled_posterior_returns()` — Compute resampled posterior returns

**Example Usage**:

```python
from probabilistic_ml_model.statistical_functions.statistical_models import (
    bayesian_category_analysis,
    calculate_ruin_probability,
    calculate_conditional_probabilities
)

# Bayesian analysis
results = bayesian_category_analysis(
    df,
    'Profitability',
    ['roe', 'roa', 'roic']
)
print(f"ROE posterior mean: {results['roe']['posterior_mean']:.2f}")

# Ruin probability
ruin_df = calculate_ruin_probability(df)
high_risk = ruin_df[ruin_df['ruin_probability'] > 0.6]
print(f"High risk stocks: {len(high_risk)}")

# Conditional probabilities
cond_probs = calculate_conditional_probabilities(df, FEATURE_CATEGORIES)
top_predictors = cond_probs.nlargest(10, 'separation')
```

---

### 3. `screening.py`

**Purpose**: Multi-factor stock screening and quality scoring

**Key Functions**:

- `create_enhanced_screener()` - Multi-factor quality and momentum screening
- `screen_earnings_quality()` - Filter by earnings quality metrics
- `screen_value_opportunities()` - Find undervalued stocks
- `screen_growth_momentum()` - Identify growth stocks
- `screen_dividend_quality()` - Quality dividend stock screening
- `screen_financial_health()` - Filter financially healthy companies
- `rank_stocks_by_composite_score()` - Composite quality ranking
- `create_sector_relative_ranking()` - Sector-relative performance ranking
- `screen_valuation_reversion_candidates()` — Valuation reversion candidate screening
- `screen_integrity_filtered_growth()` — Growth stocks filtered by accounting integrity
- `screen_garp_opportunities()` — Growth at a Reasonable Price (GARP) screening
- `screen_high_yield_safe_dividends()` — High yield with dividend safety screening
- `screen_low_volatility_quality()` — Low volatility + quality factor screening
- `screen_fcf_growth_compounders()` — FCF growth compounders screening
- `screen_total_return_leaders()` — Total return leaders screening

**Example Usage**:

```python
from probabilistic_ml_model.statistical_functions.screening import (
    create_enhanced_screener,
    screen_value_opportunities,
    screen_growth_momentum
)

# Quality screening
quality_stocks = create_enhanced_screener(
    df,
    min_fscore=7,
    min_fcf_positive_years=4,
    require_deleveraging=True
)

# Value screening
value_stocks = screen_value_opportunities(
    df,
    max_pe_ratio=20,
    min_upside_potential=25
)

# Growth screening
growth_stocks = screen_growth_momentum(
    df,
    min_revenue_growth=15,
    min_eps_growth=10
)
```

---

### 4. `feature_analytics.py` (Existing)

**Purpose**: Interactive visualization dashboards

**Key Functions**:

- `create_interactive_momentum_dashboard()` - Momentum analysis dashboard
- `create_interactive_valuation_heatmap()` - Valuation by industry heatmap
- `create_leverage_liquidity_quadrant()` - Leverage vs liquidity analysis
- `monte_carlo_price_target_simulation()` - Monte Carlo price target simulation
- `bayesian_earnings_beat_model()` - Bayesian earnings beat probability
- `analyze_distress_distribution()` - Financial distress distribution
- `create_composite_quality_score()` - Composite quality scoring
- `create_summary_dashboard()` - KPI summary dashboard

**Example Usage**:

```python
from finance_ml.analytics.feature_analytics import (
    create_interactive_momentum_dashboard,
    monte_carlo_price_target_simulation,
    create_summary_dashboard
)

# Create dashboards
momentum_fig = create_interactive_momentum_dashboard(df)
momentum_fig.write_html("outputs/momentum.html")

# Monte Carlo simulation
mc_results = monte_carlo_price_target_simulation(df, n_simulations=10000)
top_opportunities = mc_results.nlargest(20, 'risk_reward_ratio')

# Summary dashboard
summary_fig = create_summary_dashboard(df)
summary_fig.show()
```

---

### 5. `visualizations/profitability.py` (New)

**Purpose**: Margin and profitability analysis visualizations

**Key Functions**:

- `create_margin_waterfall_chart()` - Revenue to net income margin breakdown
- `create_dupont_decomposition_dashboard()` - ROE = Net Margin × Asset Turnover × Leverage
- `create_profitability_quadrant()` - ROE vs ROIC with margin bubble size
- `create_margin_trend_heatmap()` - Margin trends by industry

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.profitability import (
    create_margin_waterfall_chart,
    create_dupont_decomposition_dashboard,
    create_profitability_quadrant
)

# DuPont analysis dashboard
dupont_fig = create_dupont_decomposition_dashboard(df)
dupont_fig.write_html("outputs/dupont_analysis.html")

# Profitability quadrant (ROE vs ROIC)
quadrant_fig = create_profitability_quadrant(df)
quadrant_fig.show()

# Margin waterfall for a specific stock
waterfall_fig = create_margin_waterfall_chart(df[df['ticker'] == 'AAPL'])
```

---

### 6. `visualizations/technical.py` (New)

**Purpose**: Technical analysis and momentum visualizations

**Key Functions**:

- `create_momentum_ribbon_chart()` - Multi-period momentum overlay (1m-5y)
- `create_52w_range_distribution()` - Overbought/oversold analysis by sector
- `create_trend_strength_matrix()` - Trend score heatmap by industry
- `create_momentum_divergence_scatter()` - Short vs long-term momentum divergence

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.technical import (
    create_momentum_ribbon_chart,
    create_52w_range_distribution,
    create_momentum_divergence_scatter
)

# Momentum ribbon chart
ribbon_fig = create_momentum_ribbon_chart(df)
ribbon_fig.write_html("outputs/momentum_ribbon.html")

# 52-week range distribution by sector
range_fig = create_52w_range_distribution(df)
range_fig.show()

# Identify momentum divergences
divergence_fig = create_momentum_divergence_scatter(df)
```

---

### 7. `visualizations/temporal_analysis.py` (New)

**Purpose**: Time series and temporal pattern visualizations

**Key Functions**:

- `create_earnings_calendar_heatmap()` - Earnings dates with quality overlay
- `create_inventory_cycle_analysis()` - Inventory days and turnover trends
- `create_fcf_trajectory_chart()` - FCF positive years visualization
- `create_dividend_streak_timeline()` - Dividend sustainability analysis

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.temporal_analysis import (
    create_earnings_calendar_heatmap,
    create_fcf_trajectory_chart,
    create_dividend_streak_timeline
)

# Earnings calendar with quality scores
calendar_fig = create_earnings_calendar_heatmap(df)
calendar_fig.write_html("outputs/earnings_calendar.html")

# FCF trajectory analysis
fcf_fig = create_fcf_trajectory_chart(df)
fcf_fig.show()

# Dividend streak timeline
dividend_fig = create_dividend_streak_timeline(df)
```

---

### 8. `visualizations/valuation.py` (New)

**Purpose**: Comprehensive valuation ratio analysis and visualization

**Key Functions**:

- `create_valuation_multiples_comparison()` - Spider/radar chart comparing P/E, P/B, EV/EBITDA vs sector median
- `create_valuation_distribution_dashboard()` - Multi-panel violin plots for valuation metrics by sector
- `create_relative_valuation_matrix()` - Heatmap of Z-scores identifying cheap/expensive sectors
- `create_valuation_vs_growth_quadrant()` - PEG-style scatter with quadrants (cheap+growing, expensive+slow)
- `create_historical_valuation_percentile()` - Distribution showing current valuations vs historical ranges

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.valuation import (
    create_valuation_multiples_comparison,
    create_valuation_vs_growth_quadrant,
    create_relative_valuation_matrix
)

# Radar chart for specific stock vs sector
radar_fig = create_valuation_multiples_comparison(df, ticker='AAPL')
radar_fig.write_html("outputs/valuation_radar.html")

# PEG-style quadrant analysis
quadrant_fig = create_valuation_vs_growth_quadrant(df)
quadrant_fig.show()

# Sector valuation heatmap
matrix_fig = create_relative_valuation_matrix(df, group_col='industry')
```

---

### 9. `visualizations/earnings_quality.py` (New)

**Purpose**: Deep-dive earnings quality and predictability analysis

**Key Functions**:

- `create_earnings_surprise_dashboard()` - Multi-panel: surprise distribution, beat rate by sector
- `create_eps_trajectory_analysis()` - Trajectory score with improvement counts and streak analysis
- `create_earnings_quality_decomposition()` - Waterfall: accruals ratio, cash conversion, persistence
- `create_beat_rate_heatmap()` - Historical beat rates by sector
- `create_earnings_consistency_matrix()` - eps_positive_streak vs eps_improvement_count by sector

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.earnings_quality import (
    create_earnings_surprise_dashboard,
    create_eps_trajectory_analysis,
    create_earnings_quality_decomposition
)

# Earnings surprise analysis
surprise_fig = create_earnings_surprise_dashboard(df)
surprise_fig.write_html("outputs/earnings_surprise.html")

# EPS trajectory for top performers
trajectory_fig = create_eps_trajectory_analysis(df, top_n=30)
trajectory_fig.show()

# Quality decomposition for specific stock
quality_fig = create_earnings_quality_decomposition(df, ticker='MSFT')
```

---

### 10. `visualizations/quality_risk.py` (New)

**Purpose**: Comprehensive quality scoring and risk assessment visualization

**Key Functions**:

- `create_piotroski_fscore_breakdown()` - F-Score distribution with pass/fail indicators
- `create_altman_zscore_distribution()` - Distribution with distress zones (safe/gray/distress)
- `create_quality_risk_quadrant()` - Piotroski F-Score vs Altman Z-Score scatter
- `create_beneish_mscore_analysis()` - M-Score with manipulation probability zones
- `create_risk_tier_sunburst()` - Sector → Industry → Risk Tier hierarchy
- `create_distress_early_warning_dashboard()` - Companies approaching distress thresholds

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.quality_risk import (
    create_piotroski_fscore_breakdown,
    create_altman_zscore_distribution,
    create_quality_risk_quadrant
)

# F-Score analysis
fscore_fig = create_piotroski_fscore_breakdown(df, ticker='AAPL')
fscore_fig.write_html("outputs/fscore_analysis.html")

# Z-Score distribution with risk zones
zscore_fig = create_altman_zscore_distribution(df, group_col='industry')
zscore_fig.show()

# Quality vs Risk quadrant
quadrant_fig = create_quality_risk_quadrant(df)
```

---

### 11. `visualizations/growth_analysis.py` (New)

**Purpose**: Comprehensive growth metrics analysis similar to profitability.py structure

**Key Functions**:

- `create_growth_waterfall_chart()` - Revenue → EBITDA → EPS growth decomposition
- `create_growth_consistency_matrix()` - Growth metrics consistency (YoY, 3Y CAGR, 5Y CAGR) by sector
- `create_growth_vs_profitability_quadrant()` - BCG-style: Revenue growth vs ROE with margin bubble
- `create_growth_acceleration_chart()` - Growth acceleration (current vs historical) ranked
- `create_sustainable_growth_analysis()` - SGR = ROE × Retention Rate analysis by sector

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.growth_analysis import (
    create_growth_waterfall_chart,
    create_growth_vs_profitability_quadrant,
    create_growth_acceleration_chart
)

# Growth decomposition waterfall
waterfall_fig = create_growth_waterfall_chart(df, ticker='GOOGL')
waterfall_fig.write_html("outputs/growth_waterfall.html")

# BCG-style growth vs profitability
bcg_fig = create_growth_vs_profitability_quadrant(df)
bcg_fig.show()

# Growth acceleration analysis
accel_fig = create_growth_acceleration_chart(df, top_n=25)
```

---

### 12. `visualizations/category_charts.py`

**Purpose**: Category-specific charts covering 50+ functions across financial analysis domains

**Key Functions by Category**:

- **Analyst Sentiment**: `create_analyst_sentiment_histogram()`, `create_analyst_upside_scatter()`
- **Earnings Quality**: `create_eps_surprise_histogram()`, `create_eps_trajectory_scatter()`
- **Growth Metrics**: `create_growth_correlation_heatmap()`, `create_revenue_vs_eps_growth_scatter()`
- **Cash Flow**: `create_fcf_margin_yield_scatter()`, `create_cash_flow_quality_boxplot()`
- **Dividend Features**: `create_dividend_yield_payout_scatter()`, `create_shareholder_yield_histogram()`
- **R&D Investment**: `create_rnd_intensity_boxplot()`, `create_rnd_intensity_growth_scatter()`,
  `create_rnd_per_employee_histogram()`
- **Inventory**: `create_inventory_days_turnover_scatter()`
- **Goodwill & M&A**: `create_goodwill_concentration_boxplot()`, `create_goodwill_impairment_scatter()`,
  `create_acquisition_activity_histogram()`
- **CapEx & Investment**: `create_capex_growth_scatter()`, `create_investment_efficiency_boxplot()`,
  `create_ma_intensity_histogram()`
- **Advanced/Multi-Category**: `create_valuation_violin_plot()`, `create_quality_risk_radar_chart()`,
  `create_leverage_liquidity_bubble_chart()`
- **Post-v2.2.0**: `create_productivity_quadrant()`, `create_accounting_quality_breakdown()`,
  `create_valuation_range_visual()`, `create_balance_sheet_composition_chart()`, `create_cost_structure_breakdown()`,
  `create_unusual_items_heatmap()`
- **Post-Enhancement 1–12**: `create_volatility_surface_chart()`, `create_tax_rate_distribution()`,
  `create_fcf_estimate_curve()`, `create_opex_efficiency_scatter()`, `create_asset_sale_impact_chart()`,
  `create_share_dilution_scatter()`, `create_total_return_comparison()`, `create_dividend_yield_history_chart()`,
  `create_interest_income_trend()`

---

### 13. `probability_analytics.py`

**Purpose**: Probabilistic models for earnings beat, credit risk, dividend safety, price target achievement, and
accounting anomaly detection.

**Key Classes**:

- `EarningsBeatProbabilityModel` — Bayesian earnings beat probability with Beta-Binomial conjugate prior
- `CreditRiskProbabilityModel` — Credit risk probability using Altman Z-Score and financial health metrics
- `DividendCutProbabilityModel` — Dividend cut/safety probability estimation
- `PriceTargetAchievementModel` — Price target achievement probability
- `EPSStreakAnalyzer` — EPS streak and trajectory analysis
- `ModelConfidenceEstimator` — Model confidence calibration
- `CategoryProbabilityAnalyzer` — Per-category Bayesian probability analytics
- `ResampledBeatProbabilityModel` — Resampled beat probability with bootstrap posteriors
- `AccountingAnomalyProbabilityModel` — Accounting anomaly detection probability

**Key Result Dataclasses**:

- `BeatProbabilityResult`, `BeatProbabilityEstimate`, `ResampledBeatEstimate`
- `CreditRiskResult`, `DividendSafetyResult`, `PriceTargetResult`
- `EPSStreakResult`, `ModelConfidenceResult`, `AccountingAnomalyResult`
- `PriorParameters`, `ReportedEPSHistory`, `ForwardEstimateSignals`

**Key Functions**:

- `create_earnings_probability_dashboard()` — Interactive dashboard for earnings probability
- `create_confidence_calibration_chart()` — Confidence calibration visualization
- `create_eps_streak_analysis_chart()` — EPS streak analysis chart
- `create_view_probability_dashboard()` — Per-view probability dashboard
- `export_probability_analytics_results()` — Export results to DB/CSV/JSON
- `compute_beta_confidence_score()` — Shared Beta confidence utility

**Example Usage**:

```python
from probabilistic_ml_model.statistical_functions.probability_models import (
    EarningsBeatProbabilityModel,
    CreditRiskProbabilityModel,
    CategoryProbabilityAnalyzer,
    create_earnings_probability_dashboard,
)

# Earnings beat probability
beat_model = EarningsBeatProbabilityModel()
beat_results = beat_model.predict(df)
high_beat = beat_results[beat_results['beat_probability'] > 0.7]

# Credit risk
credit_model = CreditRiskProbabilityModel()
credit_results = credit_model.predict(df)

# Category-level Bayesian analysis
analyzer = CategoryProbabilityAnalyzer()
category_results = analyzer.analyze(df, categories=FEATURE_CATEGORIES)

# Dashboard
fig = create_earnings_probability_dashboard(beat_results)
fig.write_html("outputs/analytics/earnings_probability.html")
```

---

### 14. `inference_schema.py`

**Purpose**: ArviZ / xarray InferenceData bridge for structured Bayesian posterior storage and diagnostics.

**Key Coordinate Classes**:

- `EquityCoordinates` — Ticker, name, sector, industry coordinates
- `FeatureCoordinates` — Feature name, category, function coordinates
- `IdentifierCoordinates` — Full identifier column coordinates (31 columns)

**Key Metadata Classes**:

- `EquitiesSchemaMetadata` — Column metadata from equities schema (id, categorical, date, numeric)
- `FeatureRegistryMetadata` — Feature registry metadata (function names, categories)
- `FeatureViewSpec` — Feature view specification for xarray Dataset construction
- `EquitiesMaterializedViewSpec` — MV equities specification

**Shared Constants & Internal Helpers** (reduce duplication across builders):

- `_IDENTIFIER_COLS` — Frozen set of 9 identifier column names shared by `build_feature_view_inference_data()` and
  `EquitiesMaterializedViewSpec.from_dataframe()`
- `_safe_column_values()` — Column extraction with default factory fallback
- `_build_posterior_samples_beta()` / `_build_posterior_samples_normal()` — Generalized posterior sampling (Beta or
  Normal) across chains
- `_build_xarray_coords()` — Shared coordinate dict assembly (chain × draw × equity)
- `_moment_matched_beta_params()` — Score → (α, β) via moment matching with configurable concentration
- `_build_arviz_or_xarray()` — ArviZ-vs-xarray dispatch with automatic fallback to `xr.Dataset`
- `_build_observed_beat()`, `_build_beat_constant_data()` — Beat-specific observed/constant data extraction
- `_build_credit_observed_data()`, `_build_credit_constant_data()` — Credit risk observed/constant data extraction
- `_build_anomaly_observed_data()`, `_build_anomaly_constant_data()` — Anomaly observed/constant data extraction
- `_resolve_price_target_inputs()` — Monte Carlo price target input resolution with observed_df refinement
- `_extract_category_posterior_params()`, `_build_category_constant_data()` — Category analysis helpers

**Key Builder Functions**:

- `build_beat_probability_inference_data()` — InferenceData for earnings beat posteriors
- `build_credit_risk_inference_data()` — InferenceData for credit risk posteriors
- `build_accounting_anomaly_inference_data()` — InferenceData for anomaly posteriors
- `build_monte_carlo_inference_data()` — InferenceData for Monte Carlo simulations
- `build_category_analysis_inference_data()` — InferenceData for per-category Bayesian analysis
- `build_feature_view_inference_data()` — InferenceData for any feature view
- `build_resampled_technical_inference_data()` — InferenceData for resampled technical returns

**Key Loader Functions**:

- `load_equity_coordinates_from_db()`, `load_feature_coordinates_from_db()`
- `load_identifier_coordinates_from_db()`
- `load_equities_schema_metadata_from_db()`, `load_feature_registry_metadata_from_db()`
- `load_feature_view_spec_from_db()`, `load_mv_equities_spec_from_db()`
- `summarize_inference_data()` — Human-readable summary of InferenceData contents

**Example Usage**:

```python
from probabilistic_ml_model.data_utils.inference_schema import (
    build_beat_probability_inference_data,
    build_monte_carlo_inference_data,
    load_identifier_coordinates_from_db,
    summarize_inference_data,
)

# Build InferenceData for beat probability posteriors
idata = build_beat_probability_inference_data(beat_results_df=beat, observed_beat=, n_posterior_samples=4000)
print(summarize_inference_data(idata))

# Load identifier coordinates
id_coords = load_identifier_coordinates_from_db()
```

---

### 15. `visualizations/probability_viz.py`

**Purpose**: Probabilistic ArviZ-backed visualizations (1389 lines, 8 public functions).

**Key Functions**:

- `create_posterior_return_forest()` — Forest plot of posterior return distributions (top N stocks)
- `create_beat_probability_posterior()` — Beat probability density/bar chart
- `create_ruin_probability_diagnostic()` — Ruin probability diagnostic with risk tiers
- `create_mcse_convergence_panel()` — MCSE convergence diagnostics panel
- `create_bayesian_category_ridge()` — Ridge plot for Bayesian category analysis
- `create_tri_model_posterior_comparison()` — Tri-model posterior comparison (MC, Kalman, PT)
- `create_feature_view_posterior_panel()` — Feature view posterior panel (features + equities)
- `create_anomaly_conditional_probability_chart()` — Anomaly conditional probability visualization

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.probability_viz import (
    create_posterior_return_forest,
    create_beat_probability_posterior,
    create_ruin_probability_diagnostic,
)

# Forest plot from InferenceData or DataFrame
fig = create_posterior_return_forest(idata, top_n=30, credible_interval=0.94)
fig.write_html("outputs/analytics/posterior_forest.html")

# Beat probability posterior
fig = create_beat_probability_posterior(beat_df, top_n=12)
fig.write_html("outputs/analytics/beat_posterior.html")
```

---

### 16. `visualizations/expected_returns_viz.py`

**Purpose**: Expected returns pipeline-specific visualizations (796 lines, 14 public functions).

**Key Functions**:

- `create_mc_return_distribution()` — Monte Carlo return distribution histogram
- `create_sector_risk_reward_scatter()` — Sector risk-reward scatter plot
- `create_kalman_vs_raw_scatter()` — Kalman-filtered vs raw price target scatter
- `create_tri_model_agreement_histogram()` — Tri-model agreement distribution
- `create_strong_consensus_bar()` — Strong consensus stocks bar chart
- `create_sector_heatmap()` — Sector expected returns heatmap
- `create_var_analysis()` — Value-at-Risk analysis visualization
- `create_beat_vs_achievement_scatter()` — Beat probability vs price target achievement
- `create_model_dispersion_dashboard()` — Multi-panel model dispersion dashboard
- `create_return_distribution_fit_chart()` — Return distribution fit (normal, t, skewnorm)
- `create_sector_return_analytics_heatmap()` — Sector return analytics heatmap
- `create_screening_summary_chart()` — Screening results summary
- `create_price_target_drift_dashboard()` — Historical price target drift analysis

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.expected_returns_viz import (
    create_mc_return_distribution,
    create_sector_risk_reward_scatter,
    create_model_dispersion_dashboard,
)

fig = create_mc_return_distribution(mc_results)
fig.write_html("outputs/analytics/er_mc_distribution.html")

fig = create_sector_risk_reward_scatter(mc_results)
fig.write_html("outputs/analytics/er_sector_risk_reward.html")

fig = create_model_dispersion_dashboard(summary)
fig.write_html("outputs/analytics/er_model_dispersion.html")
```

---

### 17. `visualizations/arviz_diagnostics.py`

**Purpose**: ArviZ-backed diagnostic visualizations for the expected returns pipeline (898 lines, 15 public functions).

**Key Functions**:

- `build_screening_inference_data()` — Build InferenceData from screening results
- `create_screening_posterior_ridge()` — Ridge plot of screening posterior distributions
- `create_productivity_frontier_posterior()` — Productivity frontier posterior by quantile
- `build_resampled_posterior_idata()` — Build InferenceData from resampled posteriors
- `create_resampled_posterior_diagnostics()` — Resampled posterior diagnostic panel
- `create_resampled_sector_forest()` — Sector-level resampled forest plot
- `build_alignment_inference_data()` — Build InferenceData from tri-model alignment
- `create_model_alignment_arviz_panel()` — Model alignment ArviZ diagnostic panel
- `create_agreement_posterior_by_sector()` — Agreement posterior by sector
- `create_hierarchical_shrinkage_diagnostic()` — Hierarchical shrinkage diagnostic
- `create_multi_level_mcmc_comparison()` — Multi-level MCMC comparison
- `create_mcmc_convergence_panel_arviz()` — MCMC convergence panel (trace, R-hat, ESS)
- `build_category_analytics_idata()` — Build InferenceData from category analytics
- `create_category_posterior_diagnostics()` — Category posterior diagnostic panel
- `create_cross_category_summary()` — Cross-category summary visualization

**Example Usage**:

```python
from probabilistic_ml_model.visualizations.arviz_diagnostics import (
    create_screening_posterior_ridge,
    create_mcmc_convergence_panel_arviz,
    create_resampled_posterior_diagnostics,
)

fig = create_screening_posterior_ridge(screens)
fig.write_html("outputs/analytics/screening_posterior.html")

create_mcmc_convergence_panel_arviz(mcmc_result, output_dir=Path("outputs/analytics"))
```

---

### 18. `expected_returns_v3.py` — Expected Returns Analytics Pipeline v3.1

**Purpose**: Automated end-to-end expected returns analysis pipeline integrating all analytics modules.

**Pipeline Configuration** (`PipelineConfig` dataclass):

- `mc_simulations` (default: 50,000) — Monte Carlo simulations per stock
- `mc_max_stocks` (default: 10,000) — Maximum stocks to simulate
- `mcmc_chains` (default: 6) — Parallel MCMC chains
- `mcmc_samples` (default: 50,000) — MCMC posterior samples per chain
- `beat_threshold` (default: 0.6) — Quad-model beat classification threshold
- `output_dir` (default: "outputs/analytics") — Output directory
- `log_file` / `log_level` — Logging configuration
- `PipelineConfig.from_env()` — Build config from environment variables

**10-Step Pipeline** (`main()` function):

| Step | Description                                                        | Key Functions                                                                         |
|------|--------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 1    | Data loading (equities MV + feature views + all stock features MV) | `load_expected_returns_data()`, `load_all_stock_features()`, `load_analytics_table()` |
| 1b   | Historical target drift enrichment                                 | `_enrich_with_historical_target_drift()`                                              |
| 2    | Monte Carlo simulation                                             | `run_monte_carlo_analysis()`                                                          |
| 3    | Price Target Achievement model                                     | `run_price_target_achievement()`                                                      |
| 4    | Kalman filter                                                      | `run_kalman_filter()`                                                                 |
| 5    | Earnings Beat analysis                                             | `run_earnings_beat_analysis()`                                                        |
| 5b   | Accounting Anomaly Detection                                       | `run_accounting_anomaly_analysis()`                                                   |
| 5c   | Credit Risk & Dividend Safety                                      | `run_credit_risk_analysis()`, `run_dividend_safety_analysis()`                        |
| 5d   | Stock Screening (15 screeners)                                     | `run_stock_screening()`                                                               |
| 5e   | Resampled Bayesian posterior returns                               | `run_resampled_posterior_analysis()`                                                  |
| 6    | Tri-model & quad-model alignment                                   | `build_tri_model_alignment()`, `build_quad_model_alignment()`                         |
| 7    | Expected returns summary + MCMC                                    | `build_expected_returns_summary()`, `run_parallel_mcmc_return_analysis()`             |
| 7b   | Per-category Bayesian probability analytics                        | `run_category_probability_analysis()`                                                 |
| 8    | Build InferenceData (ArviZ)                                        | `build_*_inference_data()` functions                                                  |
| 9    | Generate visualizations                                            | 30+ visualization functions                                                           |
| 10   | Export results (deduplicated)                                      | `export_expected_returns_results()`                                                   |

**Key Analytical Functions**:

- `compute_model_detailed_statistics()` — Per-model statistics with sector breakdown
- `compute_sector_expected_returns()` — Sector-level expected return aggregation
- `compute_sector_return_analytics()` — Comprehensive sector analytics (mean, median, std, Sharpe, skew, kurtosis)
- `compute_return_zscore_ranks()` — Z-score ranking across models
- `compute_cross_model_correlation()` — MC vs Kalman correlation analysis
- `compute_cross_model_diagnostics()` — Multi-model diagnostic metrics
- `compute_return_distribution_analytics()` — Distribution fitting (normal, t, skewnorm)
- `compute_derived_price_target()` — Derived price targets from expected upside
- `extract_strong_consensus()` — Filter strong consensus stocks
- `filter_quality_stocks()` — Quality filtering of summary results
- `reconcile_feature_categories()` — Reconcile categories with available DataFrame columns

**Example Usage**:

```python
from expected_returns_v3 import main, PipelineConfig

# Run with defaults
main()

# Run with custom config
config = PipelineConfig(
    mc_simulations=25_000,
    mcmc_chains=4,
    mcmc_samples=10_000,
    output_dir="outputs/custom_run",
)
main(config)

# From environment variables
import os
os.environ["ER_MC_SIMULATIONS"] = "100000"
config = PipelineConfig.from_env()
main(config)
```

---

### 19. Enhanced Statistical Methods (in `statistical_analysis.py`)

**Purpose**: Advanced time series filtering, dependency modeling, and parallel MCMC

**New Functions**:

- `kalman_filter_price_target()` - Kalman filter for smoothing price targets
- `kalman_momentum_filter()` - Smooth noisy momentum indicators
- `fit_gaussian_copula()` - Dependency structure modeling with tail dependence
- `parallel_mcmc_chains()` - Multi-chain MCMC with Gelman-Rubin diagnostic

**Example Usage**:

```python
from probabilistic_ml_model.statistical_functions.statistical_models import (
    kalman_filter_price_target,
    fit_gaussian_copula,
    parallel_mcmc_chains
)

# Kalman filter for price targets
kalman_results = kalman_filter_price_target(df)
smoothed_targets = kalman_results['kalman_estimate']

# Copula dependency modeling
copula_result = fit_gaussian_copula(df, features=['roe', 'roa', 'debt_to_equity', 'current_ratio'])
print(f"Tail dependence: {copula_result['tail_dependence']}")

# Parallel MCMC with convergence diagnostics
mcmc_result = parallel_mcmc_chains(data=df['roe'].dropna().values, n_chains=4, n_samples=10000)
print(f"R-hat convergence: {mcmc_result['r_hat']:.3f}")
print(f"Converged: {mcmc_result['converged']}")
```

---

### 20. `optimized_ops.py`

**Purpose**: Performance-optimized operations with caching and vectorization

**Key Functions**:

- `dataframe_hash()` - Generate hash for DataFrame caching
- `load_feature_data_from_db_cached()` - Cached database queries
- `fast_monte_carlo_simulation()` - Numba-accelerated Monte Carlo
- `fast_ruin_probability()` - Vectorized ruin probability calculation
- `vectorized_zscore()` - Efficient z-score computation
- `vectorized_percentile_rank()` - Fast percentile ranking
- `get_optimization_status()` - Check optimization feature availability

**Example Usage**:

```python
from probabilistic_ml_model.optimized_ops import (
    load_feature_data_from_db_cached,
    fast_monte_carlo_simulation,
    vectorized_zscore,
    get_optimization_status
)

# Check available optimizations
status = get_optimization_status()
print(f"Numba available: {status['numba_available']}")
print(f"Joblib available: {status['joblib_available']}")

# Cached data loading (subsequent calls use cache)
df = load_feature_data_from_db_cached(earnings_date_filter="2026-01-01")

# Fast Monte Carlo simulation
expected_upside, upside_std, var_5, prob_positive = fast_monte_carlo_simulation(
    pt_low=df['price_target_low'].values,
    pt_median=df['price_target_median'].values,
    pt_high=df['price_target_high'].values,
    last_price=df['last_price'].values,
    n_simulations=10000
)

# Vectorized z-score calculation
z_scores = vectorized_zscore(df['p_e_ratio'].values)

# Backfill feature columns
df = backfill_feature_columns(df)
```

## Future Enhancements

### Planned Improvements

1. **API Development**
    - REST API for screening functions
    - WebSocket for real-time updates
    - GraphQL interface

2. **Machine Learning Integration**
    - ML-based quality classification
    - Price target regression models
    - Feature importance analysis

3. **Real-time Data Integration**
    - Live market data feeds
    - Streaming analytics
    - Alert system for screening triggers

---

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'finance_ml.analytics'`
**Solution**: Ensure you're running from the project root and the package is installed:

```bash
pip install -e .
```

**Issue**: Database connection errors
**Solution**: Set environment variables:

```bash
export DB_URL="postgresql+psycopg2://user:pass@host:5432/db"
export DB_EQUITIES_SCHEMA="public"
```

**Issue**: Missing features in DataFrame
**Solution**: Use `backfill_feature_columns()` to create derived features:

```python
from finance_ml.analytics.data_utils import backfill_feature_columns

df = backfill_feature_columns(df)
```

---

## Contributing

### Adding New Screening Functions

1. Add function to `screening.py`
2. Follow existing naming conventions
3. Include comprehensive docstring
4. Add example usage
5. Write unit tests

### Adding New Statistical Methods

1. Add function to `statistical_analysis.py`
2. Include mathematical documentation
3. Provide references to papers/methods
4. Add validation tests

---

## References

### Original Code

- `market_analytics.py` - Original notebook (5208 lines)
- `feature_analytics.ipynb` - Jupyter notebook version

### Core Refactored Modules

- `finance_ml/analytics/data_utils.py` - Data loading, preprocessing, and export framework
- `finance_ml/analytics/statistical_analysis.py` - Bayesian, MCMC, Kalman, Copula, resampled posteriors
- `finance_ml/analytics/screening.py` - Stock screening functions (15 screeners)
- `finance_ml/analytics/feature_analytics.py` - Interactive visualizations
- `finance_ml/analytics/probability_analytics.py` - Probabilistic models (earnings beat, credit risk, anomaly)
- `finance_ml/analytics/inference_schema.py` - ArviZ / xarray InferenceData bridge
- `finance_ml/analytics/optimized_ops.py` - Performance optimizations

### Visualization Modules (New in v2.0)

- `finance_ml/analytics/visualizations/profitability.py` - Margin analysis
- `finance_ml/analytics/visualizations/technical.py` - Technical analysis
- `finance_ml/analytics/visualizations/temporal_analysis.py` - Time series

### Visualization Modules (New in v2.1)

- `finance_ml/analytics/visualizations/valuation.py` - Valuation ratio analysis
- `finance_ml/analytics/visualizations/earnings_quality.py` - Earnings quality charts
- `finance_ml/analytics/visualizations/quality_risk.py` - Quality & risk assessment
- `finance_ml/analytics/visualizations/growth_analysis.py` - Growth metrics analysis

### Visualization Modules (New in v3.1)

- `finance_ml/analytics/visualizations/probability_viz.py` - Probabilistic ArviZ-backed visualizations (1,389 lines)
- `finance_ml/analytics/visualizations/expected_returns_viz.py` - Expected returns pipeline charts (796 lines)
- `finance_ml/analytics/visualizations/arviz_diagnostics.py` - ArviZ diagnostic visualizations (898 lines)

### Expected Returns Pipeline

- `expected_returns_v3.py` - Automated expected returns pipeline v3.1 (4,528 lines)

### Test Files

- `tests/test_screening.py` - Screening function tests
- `tests/test_data_utils.py` - Data utility tests
- `tests/test_statistical_analysis.py` - Statistical method tests
- `tests/test_market_analytics_integration.py` - Integration tests
- `tests/test_visualizations.py` - Visualization tests (31 tests)
- `tests/test_visualizations_valuation.py` - Valuation visualization tests (19 tests)
- `tests/test_visualizations_earnings_quality.py` - Earnings quality tests (17 tests)
- `tests/test_visualizations_quality_risk.py` - Quality & risk tests (17 tests)
- `tests/test_visualizations_growth_analysis.py` - Growth analysis tests (15 tests)
- `tests/test_enhanced_statistics.py` - Enhanced statistics tests

### Main Script

- `market_analytics.py` - Main demonstration script (1074 lines, updated with new visualizations)

### Jupyter Notebooks (Updated in v2.1)

- `feature_analytics.ipynb` - Feature analytics notebook (209 cells, updated with new visualizations)
- `financial_market_statistical_analysis.ipynb` - Statistical analysis notebook (900 lines, updated with new
  visualizations)

### Documentation

- `README.md` - Project overview
- `docs/code_guidelines.md` - Coding standards
- `docs/improvement_plan/market_analysis_refactoring_guide.md` - This document

---

## Integration Summary (v2.1)

The following scripts and notebooks have been updated to integrate the new visualization modules:

### Scripts Updated

| Script                                      | Changes                                                                                 |
|---------------------------------------------|-----------------------------------------------------------------------------------------|
| `finance_ml/analytics/feature_analytics.py` | Added imports for 21 new visualization functions; main() generates 18 additional charts |
| `market_analytics.py`                       | Added imports for 21 new visualization functions; generates 21 additional charts        |

### Notebooks Updated

| Notebook                                      | Changes                                                            |
|-----------------------------------------------|--------------------------------------------------------------------|
| `feature_analytics.ipynb`                     | Added 37 new import lines; 22 new visualization cells              |
| `financial_market_statistical_analysis.ipynb` | Added 37 new import lines; 22 new visualization cells in Section 6 |

### New Visualizations Available

#### Valuation Analysis (5 functions)

- `create_valuation_multiples_comparison()` - Spider/radar chart vs sector median
- `create_valuation_distribution_dashboard()` - Multi-panel violin plots
- `create_relative_valuation_matrix()` - Z-score heatmap by industry
- `create_valuation_vs_growth_quadrant()` - PEG-style scatter analysis
- `create_historical_valuation_percentile()` - Distribution with percentile markers

#### Earnings Quality (5 functions)

- `create_earnings_surprise_dashboard()` - Multi-panel surprise analysis
- `create_eps_trajectory_analysis()` - Trajectory score visualization
- `create_earnings_quality_decomposition()` - Waterfall decomposition
- `create_beat_rate_heatmap()` - Beat rates by sector
- `create_earnings_consistency_matrix()` - Streak vs improvement matrix

#### Quality & Risk (6 functions)

- `create_piotroski_fscore_breakdown()` - F-Score distribution
- `create_altman_zscore_distribution()` - Z-Score with distress zones
- `create_quality_risk_quadrant()` - F-Score vs Z-Score scatter
- `create_beneish_mscore_analysis()` - M-Score manipulation analysis
- `create_risk_tier_sunburst()` - Hierarchical risk visualization
- `create_distress_early_warning_dashboard()` - Early warning system

#### Growth Analysis (5 functions)

- `create_growth_waterfall_chart()` - Growth decomposition
- `create_growth_consistency_matrix()` - Consistency by sector
- `create_growth_vs_profitability_quadrant()` - BCG-style analysis
- `create_growth_acceleration_chart()` - Acceleration ranking
- `create_sustainable_growth_analysis()` - SGR analysis

---

## Contact & Support

For questions or issues with the refactored code:

1. Check this guide first
2. Review module docstrings
3. Examine example usage in `market_analytics.py`
4. Create an issue in the project repository

---

**Last Updated**: 2026-03-07
**Version**: 3.1.0
**Status**: Production Ready (Expected Returns Pipeline v3.1 + ArviZ Diagnostics)

---

## Changelog (v3.1.0)

### New Modules

- Addition of `inference_schema.py` — ArviZ/xarray InferenceData bridge (1,588 lines, refactored with Extract Function
  strategy)
- Addition of `visualizations/probability_viz.py` — 8 probabilistic visualization functions (1,389 lines)
- Addition of `visualizations/expected_returns_viz.py` — 14 pipeline visualization functions (796 lines)
- Addition of `visualizations/arviz_diagnostics.py` — 15 ArviZ diagnostic functions (898 lines)

### Expanded Modules

- Expansion of `expected_returns_v3.py` to 4,528 lines with 10-step pipeline
- Addition of `PipelineConfig` dataclass with environment variable support (`PipelineConfig.from_env()`)
- Addition of 6 new screeners: GARP, high yield, low volatility, FCF compounders, total return, integrity-filtered
  growth
- Addition of `ExportConfig` and unified export framework in `data_utils.py`
- Addition of `BayesianTechnicalResampler` and `ResampledReturnDistribution` in `statistical_analysis.py`
- Addition of `AccountingAnomalyProbabilityModel` and `ResampledBeatProbabilityModel` in `probability_analytics.py`

---

## Changelog (v2.3.0)

### DRY Identifier Columns Refactoring (`feature_registry.sql`)

All 17 `vw_features_*` views and the `mv_all_stock_features` materialized view have been refactored
to inherit identifier columns from `vw_identifier_columns` via `id.*` instead of hardcoding 9 columns
(`isin`, `ticker`, `name`, `industry`, `sector`, `trading_country`, `region`, `country`, `exchange`).

| What changed                                   | Before                                                                        | After                                                                                      |
|------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Identifier columns in 17 `vw_features_*` views | Hardcoded 9 columns (`id.isin, id.ticker, ...`)                               | `id.*` — inherits all 31 columns from `vw_identifier_columns`                              |
| Identifier columns in `mv_all_stock_features`  | Hardcoded 9 identifier columns + separate `e.` selects for dates/categoricals | `id.*` — all identifier, categorical, and date columns from single source                  |
| Single source of truth                         | `vw_identifier_columns` defined but not fully utilized                        | `vw_identifier_columns` is the **sole** source for all identifier/categorical/date columns |
| Adding a new identifier column                 | Required editing 17 views + 1 MV                                              | Edit only `vw_identifier_columns` — all views inherit automatically                        |

### Materialized View Duplicate Column Removal

Duplicate date columns that overlap with `vw_identifier_columns` have been removed from the
`mv_all_stock_features` materialized view:

- `e."FY End Date"` → already provided as `fy_end_date` via `id.*`
- `e."Next FY End Date"` → already provided as `next_fy_end_date` via `id.*`
- `e."Next Earnings"` → already provided as `next_earnings` via `id.*`
- `e."Income Statement Report Date"` → already provided as `income_statement_report_date` via `id.*`
- `e."Next Income Statement Report Date"` → already provided as `next_income_statement_report_date` via `id.*`

Non-overlapping equities columns (`market_cap`, `enterprise_value`, `last_price`, price targets,
`volume_shrs`, `shares_outstanding`) are retained as explicit selects from `e.`.

### Python Analytics Alignment

- `probability_analytics.py`: Replaced hardcoded 5-column identifier list in
  `ViewProbabilityAnalyzer.analyze_view()` with `load_identifier_columns()` from `data_utils`.
- All other analytics modules (`data_utils.py`, `statistical_analysis.py`, `screening.py`,
  `feature_analytics.py`, `optimized_ops.py`, `market_analytics.py`) already use the dynamic
  `load_identifier_columns()` / `get_identifier_cols_set()` utilities and required no changes.

### Files Updated

| File                                                         | Changes                                                             |
|--------------------------------------------------------------|---------------------------------------------------------------------|
| `feature_registry.sql`                                       | 17 views + 1 MV refactored to use `id.*`; section header updated    |
| `finance_ml/analytics/probability_analytics.py`              | Replaced hardcoded identifier list with `load_identifier_columns()` |
| `docs/improvement_plan/market_analysis_refactoring_guide.md` | Updated version to 2.3.0; added this changelog                      |

---

## Changelog (v2.2.0)

### Column Name Alignment (MV Schema Sync)

All visualization modules and dependent scripts/notebooks have been updated to use the correct
materialized view (`mv_all_stock_features`) column names:

| Old Name                                                        | New Name                                                                        | Affected Modules                                        |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------|
| `revenue_growth_yoy`                                            | `revenue_yoy_growth`                                                            | `growth_analysis.py`, `market_analytics.py`, notebooks  |
| `revenue_growth_3y_cagr`                                        | `revenue_cagr_3y`                                                               | `growth_analysis.py`                                    |
| `revenue_growth_5y_cagr`                                        | `revenue_cagr_5y`                                                               | `growth_analysis.py`                                    |
| `eps_growth_3y_cagr`                                            | `eps_cagr_3y`                                                                   | `growth_analysis.py`                                    |
| `net_income_growth`                                             | `net_income_growth_yoy`                                                         | `growth_analysis.py`                                    |
| `inventory_turnover_mv`                                         | `inventory_turnover_itf`                                                        | `temporal_analysis.py`, `category_charts.py`, notebooks |
| `beneish_m_score`                                               | `accounting_quality_score` (fallback)                                           | `quality_risk.py`                                       |
| `eps_beat_count` / `eps_total_reports`                          | `eps_positive_years` / `eps_positive_streak`                                    | `earnings_quality.py`                                   |
| `accruals_ratio`, `cash_earnings_ratio`, `earnings_persistence` | `earnings_quality_composite`, `ni_adjustment_ratio`, `accounting_quality_score` | `earnings_quality.py`                                   |

### Earnings Quality Decomposition Remap

`create_earnings_quality_decomposition()` now uses columns actually present in the MV:

- `earnings_quality_composite`, `ni_adjustment_ratio`, `eps_adjustment_ratio`,
  `accounting_quality_score`, `earnings_quality_impact`

### Beat Rate & Consistency Remap

`create_beat_rate_heatmap()` and `create_earnings_consistency_matrix()` now use:

- `eps_positive_years`, `eps_positive_streak`, `eps_improvement_count`, `eps_trajectory_score`

### Beneish M-Score Fallback

`create_beneish_mscore_analysis()` now falls back through:
`beneish_m_score` → `accounting_quality_score` → `accruals_quality`
with adaptive thresholds and labels per resolved column.

### Shared Utilities (`_shared.py`)

New shared module `finance_ml/analytics/visualizations/_shared.py` provides:

- `PLOTLY_TEMPLATE`, `COLORS` — centralized constants
- `MV_COLUMN_ALIASES` — canonical alias map for MV column resolution
- `resolve_column(df, logical_name)` — resolve logical column names to actual DataFrame columns
- `create_no_data_figure(title)` — DRY replacement for per-module `_create_no_data_figure()`

### Data Guard Clauses (`category_charts.py`)

All 23+ functions in `category_charts.py` now include column-existence checks
and return graceful "No Data" placeholder figures instead of raising `ValueError`.

### Files Updated

| File                                                       | Changes                                                      |
|------------------------------------------------------------|--------------------------------------------------------------|
| `finance_ml/analytics/visualizations/earnings_quality.py`  | Remapped decomposition, beat rate, consistency to MV columns |
| `finance_ml/analytics/visualizations/growth_analysis.py`   | Fixed all growth metric column names                         |
| `finance_ml/analytics/visualizations/quality_risk.py`      | Added M-Score fallback chain                                 |
| `finance_ml/analytics/visualizations/temporal_analysis.py` | Fixed inventory turnover column                              |
| `finance_ml/analytics/visualizations/category_charts.py`   | Added data guard clauses to all functions                    |
| `finance_ml/analytics/visualizations/profitability.py`     | Added `total_asset_turnover` fallback for DuPont             |
| `finance_ml/analytics/visualizations/_shared.py`           | New shared utilities module                                  |
| `finance_ml/analytics/visualizations/__init__.py`          | Exports shared utilities                                     |
| `market_analytics.py`                                      | Updated metric references                                    |
| `feature_analytics.ipynb`                                  | Updated metric references                                    |
| `financial_market_statistical_analysis.ipynb`              | Updated metric references                                    |
