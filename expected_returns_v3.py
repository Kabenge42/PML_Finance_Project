"""
Expected Returns Analytics Module (v3.4)

Automated pipeline for expected returns analysis using the v3.4+ analytics platform:

**Core Models:**
- **Monte Carlo Simulation** — Probabilistic upside/downside distributions with historical target drift
- **Price Target Achievement** — Probability-weighted expected returns with analyst sentiment & risk adjustment
- **Kalman Filtered Targets** — Noise-reduced price target signals with momentum-informed priors
- **Earnings Beat Analysis** — Three-layer Bayesian earnings beat probability with quality filters
- **Credit Risk Analysis** — Bayesian distress estimation with debt trajectory & balance sheet strength
- **Dividend Safety Analysis** — Dividend cut probability with FCF coverage & leverage signals
- **Accounting Anomaly Detection** — Multi-layered statistical anomaly detection with Mahalanobis distance

**Cross-Model Analytics:**
- **Tri-Model Alignment** — MC vs Kalman vs Achievement model consensus
- **Quad-Model Agreement** — MC + Kalman + Achievement + Earnings Beat signals
- **Cross-Model Dispersion** — Kendall concordance & inter-model divergence metrics
- **Hierarchical MCMC** — Bayesian shrinkage-based sector/category posteriors

**Statistical Analysis:**
- **Bayesian Category Analysis** — Per-feature-category posterior estimation
- **Gaussian Copula Dependency** — Tail dependence & joint distribution modeling
- **Parallel MCMC Chains** — Gelman-Rubin convergence diagnostics
- **Resampled Posterior Returns** — Bayesian technical resampling from historical snapshots
- **Student-t MCMC** — Heavy-tail robust posterior inference
- **Distribution Fitting** — AIC-based best-fit selection (Normal, Student-t, Skew-normal, Laplace)

**Probability Analytics:**
- **Category-Level Distributions** — Per-category credible intervals & posterior means
- **Conditional Probability Analysis** — Feature-level P(anomaly | conditions)
- **Risk Metrics** — VaR, CVaR, downside deviation, gain/loss ratio

**Stock Screening:**
- **Quality Screening** — Composite score-based tiers with dynamic thresholds
- **Earnings Quality** — EPS consistency, GAAP divergence, revision momentum
- **Value Opportunities** — Valuation reversion candidates
- **Growth Momentum** — Revenue/EPS acceleration with profitability filters
- **GARP** — Growth at a reasonable price
- **Dividend Quality** — Yield safety with coverage & streak metrics
- **Financial Health** — Altman Z-score, Piotroski F-score, distress risk
- **Integrity-Filtered Growth** — Accounting quality & growth alignment
- **High-Yield Safe Dividends** — Sustainable yield with leverage constraints
- **Low-Volatility Quality** — Beta stability with profitability
- **FCF Compounders** — Free cash flow growth consistency
- **Total Return Leaders** — Price appreciation + dividend yield
- **Sector-Relative Ranking** — Percentile-based composite scores

**Advanced Analytics (v3.4):**
- **Productivity Frontier Analysis** — Employee efficiency vs revenue-per-employee
- **Reporting Lag Sentiment** — "Bad news travels slow" hypothesis testing
- **MCMC-Enhanced Probability Models** — Posterior distributions for anomaly, credit, dividend, price target
- **Multi-Level Hierarchical MCMC** — Cross-category shrinkage (region, country, sector, industry, style, size)
- **Feature View Posterior Panels** — Per-view InferenceData with ArviZ diagnostics

**Data Sources (v3.4 — Equities MV + Feature Views):**
- `public.mv_equities` — Core equities data via `load_equities_data_from_db`
- `public.vw_features_*` — 17 feature views via `load_all_feature_views`:
  - `vw_features_analyst_sentiment`
  - `vw_features_balance_sheet`
  - `vw_features_cash_flow`
  - `vw_features_debt_leverage`
  - `vw_features_dividends`
  - `vw_features_earnings_quality`
  - `vw_features_eps_estimates`
  - `vw_features_growth`
  - `vw_features_income_statement`
  - `vw_features_liquidity`
  - `vw_features_momentum`
  - `vw_features_operational_efficiency`
  - `vw_features_price_target_dynamics`
  - `vw_features_profitability`
  - `vw_features_quality_risk`
  - `vw_features_valuation`
  - `vw_features_working_capital`
- `public.mv_all_stock_features` — Full feature superset via `load_feature_data_from_db`
- `public.equities_schema_metadata` — Dynamic column discovery via `get_equities_schema`
- `public.calculated_features_registry` — Feature categories via `load_feature_categories_from_db`

**Migration from v3.3:**
- Added debt trajectory features (`debt_3y_cagr`, `debt_4q_trend`, `debt_yoy_change`) to credit risk model
- Added cash buffer features (`adequate_cash_buffer`, `cash_vs_5y_avg`) to credit risk model
- Added retained earnings growth to credit & dividend models
- Added beta stability & trend to price target achievement model
- Added accounting quality features to earnings beat model
- Enriched dividend safety with leverage/liquidity columns (`interest_coverage`, `debt_to_equity`, `cash_ratio`, etc.)
- Enhanced price target model with distress risk & balance sheet strength
- Improved anomaly detection with per-feature conditional probabilities
- Extended historical target drift with price-vs-target convergence signal
- Added multi-level hierarchical MCMC across 9 category columns (region, country, exchange, sector, industry, style, size, unit, trading_country)
- Integrated MCMC posterior visualizations for anomaly, credit, dividend, price target models

**Migration from v3.0:**
- Replaced `load_expected_returns_data` → `load_equities_data_from_db` (`data_utils`)
- Replaced `load_all_stock_features` → `load_all_feature_views` (`data_utils`)
- Replaced `load_analytics_table` → `load_feature_data_from_db` (`data_utils`)
- Replaced hardcoded column lists → dynamic `get_equities_schema`
- Retained hardcoded fallbacks for offline/no-DB environments

**ArviZ Integration:**
- `InferenceData` schema for Monte Carlo, Earnings Beat, Credit Risk, Feature Views
- Per-model `EquityCoordinates` with ticker/sector/industry/country dimensions
- MCMC convergence diagnostics (R-hat, ESS, MCSE)
- Posterior ridge plots, forest plots, trace plots via `arviz_diagnostics`

**Visualization Categories:**
- **Expected Returns:** MC distribution, Kalman vs raw, tri-model agreement, sector heatmap, risk-reward scatter
- **Earnings Quality:** Revision momentum, GAAP divergence, beat probability, surprise dashboard, consistency matrix
- **Quality & Risk:** Piotroski F-score, Altman Z-score, Beneish M-score, distress early warning, risk tier sunburst
- **Accounting Anomaly:** Dashboard, severity, conditional probability, MCMC posterior
- **Credit Risk:** Ruin probability, MCMC distress posterior
- **Dividend Safety:** MCMC dividend cut posterior
- **Price Target:** MCMC achievement posterior, drift dashboard
- **Valuation:** Multiples comparison, distribution dashboard, relative matrix, growth quadrant, historical percentile
- **Growth:** Waterfall, consistency matrix, profitability quadrant, acceleration, sustainability
- **Bayesian Analytics:** Category ridge, posterior forest, resampled diagnostics, hierarchical shrinkage
- **Screening:** Posterior ridge, productivity frontier, model alignment panel

**Usage:**
Data sources (v3.2 — Equities MV + Feature Views):
    - public.mv_equities              (equities data via load_equities_data_from_db)
    - public.vw_features_*            (17 feature views via load_all_feature_views)
    - equities_schema_metadata        (dynamic column discovery via get_equities_schema)

Migration from v3.0:
    - Replaced load_expected_returns_data → load_equities_data_from_db (data_utils)
    - Replaced load_all_stock_features → load_all_feature_views (data_utils)
    - Replaced load_analytics_table → load_all_feature_views (data_utils)
    - Replaced hardcoded Model Runners column lists → dynamic get_equities_schema
    - Retained hardcoded fallbacks for offline/no-DB environments

Usage:
    python expected_returns_v3.py
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats as sp_stats
from requests.exceptions import RequestsDependencyWarning

# --- Data utilities ---
from analytics.data_utils import (
    ExportConfig,
    aggregate_probability_results,
    backfill_feature_columns,
    compute_metric_statistics,
    export_to_db,
    get_equities_schema,
    get_identifier_cols_set,
    get_view_category_mapping,
    load_all_feature_views,
    load_feature_data_from_db,
    load_equities_data_from_db,
    load_feature_categories_from_db,
    load_identifier_columns,
    reorder_with_identifiers,
    validate_feature_alignment,
    )

# --- Optimised operations ---
from analytics.optimized_ops import (
    get_optimization_status,
    vectorized_percentile_rank,
    vectorized_zscore,
    )

# --- Probability models ---
from analytics.probability_analytics import (
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

# --- Screening (quality filtering of results) ---
from analytics.screening import (
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

# --- Statistical analysis ---
from analytics.statistical_analysis import (
    bayesian_category_analysis,
    bayesian_earnings_beat_model,
    calculate_conditional_probabilities,
    calculate_ruin_probability,
    fit_distributions_by_category,
    fit_gaussian_copula,
    hierarchical_mcmc_by_sector,
    hierarchical_mcmc_multi_level,
    kalman_filter_price_target,
    kalman_momentum_filter,
    mcmc_student_t,
    monte_carlo_price_target_simulation,
    parallel_mcmc_chains,
    resampled_posterior_returns,
    run_category_probability_analytics,
    analyze_employee_productivity_frontier,
    analyze_reporting_lag_sentiment,
    )

# --- InferenceData schema (ArviZ / xarray bridge) ---
try:
    from analytics.inference_schema import (
        ARVIZ_AVAILABLE,
        EquityCoordinates,
        IdentifierCoordinates,
        EquitiesSchemaMetadata,
        FeatureRegistryMetadata,
        FeatureViewSpec,
        EquitiesMaterializedViewSpec,
        FEATURE_VIEW_REGISTRY,
        build_beat_probability_inference_data,
        build_credit_risk_inference_data,
        build_monte_carlo_inference_data,
        build_feature_view_inference_data,
        load_identifier_coordinates_from_db,
        load_equities_schema_metadata_from_db,
        load_feature_registry_metadata_from_db,
        load_feature_view_spec_from_db,
        load_mv_equities_spec_from_db,
        summarize_inference_data,
        )
except Exception as _inference_err:
    logging.getLogger(__name__).warning(
            "inference_schema import failed: %s", _inference_err, exc_info=True,
            )
    ARVIZ_AVAILABLE = False
    EquityCoordinates = None  # type: ignore[assignment,misc]
    IdentifierCoordinates = None  # type: ignore[assignment,misc]
    EquitiesSchemaMetadata = None  # type: ignore[assignment,misc]
    FeatureRegistryMetadata = None  # type: ignore[assignment,misc]
    FeatureViewSpec = None  # type: ignore[assignment,misc]
    EquitiesMaterializedViewSpec = None  # type: ignore[assignment,misc]
    FEATURE_VIEW_REGISTRY = {}  # type: ignore[misc]
    build_beat_probability_inference_data = None  # type: ignore[assignment]
    build_credit_risk_inference_data = None  # type: ignore[assignment]
    build_monte_carlo_inference_data = None  # type: ignore[assignment]
    build_feature_view_inference_data = None  # type: ignore[assignment]
    load_identifier_coordinates_from_db = None  # type: ignore[assignment]
    load_equities_schema_metadata_from_db = None  # type: ignore[assignment]
    load_feature_registry_metadata_from_db = None  # type: ignore[assignment]
    load_feature_view_spec_from_db = None  # type: ignore[assignment]
    load_mv_equities_spec_from_db = None  # type: ignore[assignment]
    summarize_inference_data = None  # type: ignore[assignment]

# --- Probabilistic visualizations (ArviZ-backed) ---
# --- Other visualizations ---
from analytics.visualizations import (
    PLOTLY_TEMPLATE,
    MV_COLUMN_ALIASES,
    resolve_column,
    )

# --- Quality & Risk charts ---
from analytics.visualizations.quality_risk import (
    create_accounting_anomaly_dashboard,
    create_anomaly_severity_dashboard,
    create_altman_zscore_distribution,
    create_beneish_mscore_analysis,
    create_distress_early_warning_dashboard,
    create_piotroski_fscore_breakdown,
    create_quality_risk_quadrant,
    create_risk_tier_sunburst,
    )

# --- Earnings Quality charts ---
from analytics.visualizations.earnings_quality import (
    create_enhanced_beat_probability_dashboard as create_enhanced_beat_prob_dash,
    create_gaap_divergence_plot,
    create_revision_momentum_chart,
    create_earnings_surprise_dashboard,
    create_eps_trajectory_analysis,
    create_earnings_quality_decomposition,
    create_beat_rate_heatmap,
    create_earnings_consistency_matrix,
    )

# --- Expected Returns Pipeline charts ---
from analytics.visualizations.expected_returns_viz import (
    create_beat_vs_achievement_scatter,
    create_kalman_vs_raw_scatter,
    create_mc_return_distribution,
    create_model_dispersion_dashboard,
    create_price_target_drift_dashboard,
    create_return_distribution_fit_chart,
    create_screening_summary_chart,
    create_sector_heatmap,
    create_sector_return_analytics_heatmap,
    create_sector_risk_reward_scatter,
    create_strong_consensus_bar,
    create_tri_model_agreement_histogram,
    create_var_analysis,
    )

# --- Valuation Analysis charts ---
from analytics.visualizations.valuation import (
    create_valuation_multiples_comparison,
    create_valuation_distribution_dashboard,
    create_relative_valuation_matrix,
    create_valuation_vs_growth_quadrant,
    create_historical_valuation_percentile,
    )

# --- Growth Analysis charts ---
from analytics.visualizations.growth_analysis import (
    create_growth_waterfall_chart,
    create_growth_consistency_matrix,
    create_growth_vs_profitability_quadrant,
    create_growth_acceleration_chart,
    create_sustainable_growth_analysis,
    )

# --- Probabilistic & Bayesian Analysis (ArviZ-backed) ---
from analytics.visualizations.probability_viz import (
    create_bayesian_category_ridge,
    create_beat_probability_posterior,
    create_posterior_return_forest,
    create_ruin_probability_diagnostic,
    create_tri_model_posterior_comparison,
    create_feature_view_posterior_panel,
    create_mcse_convergence_panel,
    create_anomaly_conditional_probability_chart,
    # MCMC-enhanced probability model visualizations (v3.3)
    create_mcmc_anomaly_posterior_chart,
    create_mcmc_credit_risk_chart,
    create_mcmc_dividend_cut_chart,
    create_mcmc_price_target_chart,
    create_mcmc_category_posterior_chart,
    )

# --- ArviZ diagnostic visualizations ---
try:
    from analytics.visualizations.arviz_diagnostics import (
        ARVIZ_AVAILABLE as _ARVIZ_DIAG_AVAILABLE,
        create_screening_posterior_ridge,
        create_productivity_frontier_posterior,
        create_resampled_posterior_diagnostics,
        create_resampled_sector_forest,
        create_model_alignment_arviz_panel,
        create_agreement_posterior_by_sector,
        create_hierarchical_shrinkage_diagnostic,
        create_multi_level_mcmc_comparison,
        create_mcmc_convergence_panel_arviz,
        create_category_posterior_diagnostics,
        create_cross_category_summary,
        )
except Exception as _arviz_diag_err:
    logging.getLogger(__name__).warning(
            "arviz_diagnostics import failed: %s", _arviz_diag_err, exc_info=True,
            )
    _ARVIZ_DIAG_AVAILABLE = False

from logging_config import configure_logging
from ml_workflow.core.utils import safe_divide

px.defaults.template = PLOTLY_TEMPLATE

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """
    Centralized configuration for the expected returns analytics pipeline.

    All hardcoded magic numbers are surfaced here so they can be overridden
    from CLI arguments, environment variables, or test fixtures.

    Parameters
    ----------
    mc_simulations : int
        Number of Monte Carlo simulations per stock.
    mc_max_stocks : int
        Maximum number of stocks to simulate.
    mcmc_chains : int
        Number of parallel MCMC chains.
    mcmc_samples : int
        Number of MCMC posterior samples per chain.
    beat_threshold : float
        Probability threshold for quad-model "beat bullish" classification.
    output_dir : str
        Output directory for analytics artifacts.
    log_file : str | None
        Log file path. None disables file logging.
    log_level : int
        Logging level (e.g. logging.INFO).
    """

    mc_simulations: int = 50_000
    mc_max_stocks: int = 10_000
    mcmc_chains: int = 6
    mcmc_samples: int = 50_000
    beat_threshold: float = 0.6
    output_dir: str = "outputs"
    log_file: str | None = "logs/expected_returns_pipeline.log"
    log_level: int = logging.INFO
    # v3.5: MCMC-specific settings surfaced for per-model configuration
    mcmc_burn_in: int = 2000
    use_mcmc: bool = True
    use_student_t: bool = False
    anomaly_z_threshold: float | None = None

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build config from environment variables with sensible defaults."""
        return cls(
                mc_simulations=int(os.environ.get("ER_MC_SIMULATIONS", 50_000)),
                mc_max_stocks=int(os.environ.get("ER_MC_MAX_STOCKS", 10_000)),
                mcmc_chains=int(os.environ.get("ER_MCMC_CHAINS", 6)),
                mcmc_samples=int(os.environ.get("ER_MCMC_SAMPLES", 50_000)),
                output_dir=os.environ.get("ER_OUTPUT_DIR", "outputs"),
                log_file=os.environ.get(
                        "ER_LOG_FILE", "logs/expected_returns_pipeline.log",
                        ),
                mcmc_burn_in=int(os.environ.get("ER_MCMC_BURN_IN", 2000)),
                use_mcmc=os.environ.get("ER_USE_MCMC", "true").lower() == "true",
                use_student_t=os.environ.get("ER_USE_STUDENT_T", "false").lower() == "true",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _log_and_print(msg: str = "", level: int = logging.INFO) -> None:
    """Log a message and print it to stdout for pipeline visibility."""
    logger.log(level, msg)
    print(msg)


def _has_required_columns(df: pd.DataFrame, columns: list[str], context: str) -> bool:
    """Check that all required columns exist in a DataFrame, log missing ones."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning("%s: missing expected columns %s", context, missing)
        return False
    return True


def _write_viz(
        fig,
        output_dir: Path,
        filename: str,
        *,
        fmt: str = "html",
        dpi: int = 150,
        ) -> None:
    """
    Write a visualization figure to disk and log success.

    Centralises the repetitive create → write → log pattern used throughout
    Step 9 of the pipeline.

    Parameters
    ----------
    fig
        A Plotly ``Figure`` (for *html*) or Matplotlib ``Figure`` (for *png*).
    output_dir : Path
        Directory to write the file into.
    filename : str
        File name (e.g. ``"er_mc_distribution.html"``).
    fmt : str, default "html"
        ``"html"`` calls ``fig.write_html``; ``"png"`` calls ``fig.savefig``.
    dpi : int, default 150
        Resolution for PNG output (ignored for HTML).
    """
    dest = output_dir / filename
    if fmt == "html":
        fig.write_html(dest)
    else:
        fig.savefig(dest, dpi=dpi, bbox_inches="tight")
    _log_and_print(f"   ✓ {filename}")


def _enrich_dataframe(
        target: pd.DataFrame,
        source: pd.DataFrame,
        needed_cols: set[str],
        label: str,
        ) -> pd.DataFrame:
    """
    Merge missing columns from *source* into *target* on ``ticker``.

    Used by Step 8b to enrich ``df`` and ``beat`` with viz-critical columns
    from the richest available feature source.

    Parameters
    ----------
    target : pd.DataFrame
        DataFrame to enrich (e.g. ``df`` or ``beat``).
    source : pd.DataFrame
        DataFrame containing the missing columns.
    needed_cols : set[str]
        Set of column names that should be present.
    label : str
        Human-readable label for logging (e.g. ``"df (mv_equities)"``).

    Returns
    -------
    pd.DataFrame
        Enriched copy of *target*.
    """
    if (
            target.empty
            or source.empty
            or "ticker" not in target.columns
            or "ticker" not in source.columns
    ):
        return target

    missing = [c for c in needed_cols if c not in target.columns and c in source.columns]
    if not missing:
        return target

    src_subset = source[["ticker"] + missing].drop_duplicates(subset="ticker")
    target = target.merge(src_subset, on="ticker", how="left")
    _log_and_print(f"  ✓ Enriched {label} with {len(missing)} viz-critical columns")
    return target


def _log_anomaly_diagnostics(anomaly_results: pd.DataFrame) -> None:
    """
    Log detailed accounting anomaly diagnostics.

    Extracted from Step 5b of ``main()`` to reduce its line count and
    improve readability.  All output goes through ``_log_and_print``.
    """
    # ── Tier distribution ──
    if "accounting_anomaly_tier" in anomaly_results.columns:
        tier_counts = anomaly_results["accounting_anomaly_tier"].value_counts()
        _log_and_print("  Anomaly tier distribution:")
        for tier_label in ["Clean", "Watch", "Flag", "Alert"]:
            count = tier_counts.get(tier_label, 0)
            pct = count / len(anomaly_results) * 100 if len(anomaly_results) > 0 else 0
            _log_and_print(f"    {tier_label}: {count:,} ({pct:.1f}%)")

    # ── Score statistics ──
    score_stats = compute_metric_statistics(anomaly_results["accounting_anomaly_score"])
    if score_stats:
        _log_and_print(
                f"  Anomaly score — mean: {score_stats['mean']:.1f}, "
                f"median: {score_stats['median']:.1f}, "
                f"std: {score_stats['std']:.1f}, "
                f"max: {score_stats['max']:.1f}",
                )

    # ── Anomaly feature count ──
    if "anomaly_feature_count" in anomaly_results.columns:
        flagged = (anomaly_results["anomaly_feature_count"] > 0).sum()
        multi_flagged = (anomaly_results["anomaly_feature_count"] >= 10).sum()
        _log_and_print(
                f"  Stocks with ≥1 flagged feature: {flagged:,}, ≥10 flags: {multi_flagged:,}",
                )

    # ── Per-feature flag summary ──
    flag_cols = [c for c in anomaly_results.columns if c.endswith("_anomaly_flag")]
    if flag_cols:
        _log_and_print("  Per-feature anomaly flags:")
        for fc in sorted(flag_cols):
            feat_name = fc.replace("_anomaly_flag", "")
            n_flagged = anomaly_results[fc].sum() if fc in anomaly_results.columns else 0
            if n_flagged > 0:
                _log_and_print(f"    {feat_name}: {int(n_flagged):,} stocks flagged")

    # ── Mahalanobis distance ──
    if "mahalanobis_distance" in anomaly_results.columns:
        mahal = anomaly_results["mahalanobis_distance"].dropna()
        if len(mahal) > 0:
            _log_and_print(
                    f"  Mahalanobis distance — computed: {len(mahal):,}, "
                    f"mean: {mahal.mean():.2f}, p95: {mahal.quantile(0.95):.2f}",
                    )

    # ── Sector-relative anomaly ──
    if "sector_relative_anomaly" in anomaly_results.columns:
        sra = anomaly_results["sector_relative_anomaly"].dropna()
        sector_outliers = (sra.abs() > 2.0).sum()
        _log_and_print(
                f"  Sector-relative outliers (|z| > 2): {sector_outliers:,}",
                )

    # ── Severity score & conditional probability ──
    if "anomaly_severity_score" in anomaly_results.columns:
        sev = anomaly_results["anomaly_severity_score"].dropna()
        if len(sev) > 0:
            _log_and_print(
                    f"  Severity score — mean: {sev.mean():.2f}, "
                    f"median: {sev.median():.2f}, max: {sev.max():.2f}",
                    )

    if "anomaly_conditional_probability" in anomaly_results.columns:
        cond_p = anomaly_results["anomaly_conditional_probability"].dropna()
        if len(cond_p) > 0:
            _log_and_print(
                    f"  Conditional P(anomaly) — mean: {cond_p.mean():.3f}, "
                    f"median: {cond_p.median():.3f}, max: {cond_p.max():.3f}",
                    )

    if "multi_flag_alert" in anomaly_results.columns:
        n_alerts = anomaly_results["multi_flag_alert"].sum()
        _log_and_print(f"  Multi-flag alerts: {int(n_alerts):,}")

    if "anomaly_risk_rank" in anomaly_results.columns:
        rank_data = anomaly_results["anomaly_risk_rank"].dropna()
        if len(rank_data) > 0:
            top_risk = (rank_data >= 90).sum()
            _log_and_print(f"  Stocks in top 10% risk rank: {int(top_risk):,}")

    # ── Benford's Law test ──
    if "benford_chi2_pvalue" in anomaly_results.columns:
        bp = anomaly_results["benford_chi2_pvalue"].dropna()
        if len(bp) > 0:
            p_val = bp.iloc[0]
            verdict = "⚠️ suspicious" if p_val < 0.05 else "✓ consistent"
            _log_and_print(
                    f"  Benford's Law chi² p-value: {p_val:.4f} ({verdict})",
                    )

    # ── Distribution fit summary ──
    dist_cols = [c for c in anomaly_results.columns if c.endswith("_dist_name")]
    if dist_cols:
        _log_and_print("  Best-fit distributions per feature:")
        for dc in sorted(dist_cols):
            feat_name = dc.replace("_dist_name", "")
            dist_name = anomaly_results[dc].mode().iloc[0] if len(anomaly_results[dc].dropna()) > 0 else "n/a"
            pval_col = f"{feat_name}_dist_pvalue"
            pval = anomaly_results[pval_col].mean() if pval_col in anomaly_results.columns else float("nan")
            _log_and_print(
                    f"    {feat_name}: {dist_name} (mean KS p={pval:.3f})",
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Categories — loaded dynamically from calculated_features_registry
# with fallback to data_utils._get_fallback_feature_categories()
# ═══════════════════════════════════════════════════════════════════════════════
_feature_categories_cache: Optional[dict[str, list[str]]] = None


def get_feature_categories() -> dict[str, list[str]]:
    """
    Load feature categories from database, with hardcoded fallback.

    Delegates to ``data_utils.load_feature_categories_from_db()`` and caches
    the result for the lifetime of the process.  Replaces the former ~150-line
    inline ``FEATURE_CATEGORIES`` dict, keeping a single source of truth in
    the ``calculated_features_registry`` table (or its ``data_utils`` fallback).

    Returns
    -------
    dict[str, list[str]]
        Mapping of category name → list of feature column names.
    """
    global _feature_categories_cache
    if _feature_categories_cache is None:
        _feature_categories_cache = load_feature_categories_from_db()
        logger.info(
                "Loaded %d feature categories (%d total features)",
                len(_feature_categories_cache),
                sum(len(v) for v in _feature_categories_cache.values()),
                )
    return _feature_categories_cache


# Backward-compatible module-level name — callers that reference
# FEATURE_CATEGORIES directly still work via the lazy proxy.
class _LazyFeatureCategories:
    """Dict-like proxy that loads feature categories on first access."""

    def __getattr__(self, name: str):
        return getattr(get_feature_categories(), name)

    def __getitem__(self, key: str):
        return get_feature_categories()[key]

    def __iter__(self):
        return iter(get_feature_categories())

    def __len__(self):
        return len(get_feature_categories())

    def __contains__(self, item):
        return item in get_feature_categories()

    def __repr__(self):
        return repr(get_feature_categories())

    def items(self):
        return get_feature_categories().items()

    def keys(self):
        return get_feature_categories().keys()

    def values(self):
        return get_feature_categories().values()

    def get(self, key, default=None):
        return get_feature_categories().get(key, default)


FEATURE_CATEGORIES = _LazyFeatureCategories()  # type: ignore[assignment]


def reconcile_feature_categories(
        categories: dict[str, list[str]], df_columns: set[str],
        ) -> dict[str, list[str]]:
    """
    Reconcile feature categories against actual DataFrame columns.

    Removes features that don't exist in the DataFrame and drops
    categories that end up empty. Logs mismatches for debugging.

    Parameters
    ----------
    categories : dict[str, list[str]]
        Category name -> list of expected feature aliases.
    df_columns : set[str]
        Actual column names present in the DataFrame.

    Returns
    -------
    dict[str, list[str]]
        Reconciled categories with only existing features.
    """
    reconciled = {}
    for cat, features in categories.items():
        matched = [f for f in features if f in df_columns]
        unmatched = [f for f in features if f not in df_columns]
        if unmatched:
            logging.debug(
                    "Category '%s': %d/%d features missing from DataFrame: %s",
                    cat,
                    len(unmatched),
                    len(features),
                    unmatched[:5],
                    )
        if matched:
            reconciled[cat] = matched
        else:
            logging.debug("Category '%s' dropped — no matching columns", cat)
    return reconciled


# ═══════════════════════════════════════════════════════════════════════════════
# Shared constants & helpers for data loading
# ═══════════════════════════════════════════════════════════════════════════════

# Momentum columns eligible for Kalman smoothing (used in both MV loaders)
_KALMAN_MOMENTUM_COLS = [
    "price_momentum_1m",
    "price_momentum_3m",
    "price_momentum_6m",
    "price_momentum_1y",
    "price_momentum_5d",
    ]

# Module-level cache for equities-schema-derived column lists
_schema_column_cache: Optional[dict[str, list[str]]] = None


def _get_schema_columns() -> dict[str, list[str]]:
    """
    Derive column lists dynamically from ``get_equities_schema()``.

    Falls back to hardcoded defaults when the database is unavailable.

    Returns
    -------
    dict[str, list[str]]
        Mapping of column-group name → list of column aliases.
    """
    global _schema_column_cache
    if _schema_column_cache is not None:
        return _schema_column_cache

    schema = get_equities_schema()

    if schema:
        # Build role → list[alias] index
        role_cols: dict[str, list[str]] = {}
        for alias, meta in schema.items():
            role_cols.setdefault(meta["role"], []).append(alias)

        mc_required = [
            c
            for c in [
                "price_target",
                "price_target_high",
                "price_target_low",
                "last_price",
                ]
            if c in schema
            ]
        kalman_required = [c for c in ["last_price", "price_target"] if c in schema]
        historical_prices = sorted(
                (c for c, m in schema.items() if m["role"] == "historical_price"),
                )
        historical_targets = sorted(
                (c for c, m in schema.items() if m["role"] == "historical_price_target"),
                )
        historical_targets_high = sorted(
                (c for c, m in schema.items() if m["role"] == "historical_price_target_high"),
                )
        historical_targets_low = sorted(
                (c for c, m in schema.items() if m["role"] == "historical_price_target_low"),
                )
        historical_targets_median = sorted(
                (c
                 for c, m in schema.items()
                 if m["role"] == "historical_price_target_median"),
                )

        _schema_column_cache = {
            "mc_required": mc_required or _MC_REQUIRED_COLS_FALLBACK,
            "kalman_required": kalman_required or _KALMAN_REQUIRED_COLS_FALLBACK,
            "historical_prices": historical_prices or _HISTORICAL_PRICE_COLS_FALLBACK,
            "historical_targets": historical_targets
                                  or _HISTORICAL_PRICE_TARGET_COLS_FALLBACK,
            "historical_targets_high": historical_targets_high
                                       or _HISTORICAL_PRICE_TARGET_HIGH_COLS_FALLBACK,
            "historical_targets_low": historical_targets_low
                                      or _HISTORICAL_PRICE_TARGET_LOW_COLS_FALLBACK,
            "historical_targets_median": historical_targets_median
                                         or _HISTORICAL_PRICE_TARGET_MEDIAN_COLS_FALLBACK,
            }
        logger.info(
                "Column lists derived from equities schema (%d total columns)",
                sum(len(v) for v in _schema_column_cache.values()),
                )
    else:
        logger.info("Equities schema unavailable — using fallback column lists")
        _schema_column_cache = {
            "mc_required": _MC_REQUIRED_COLS_FALLBACK,
            "kalman_required": _KALMAN_REQUIRED_COLS_FALLBACK,
            "historical_prices": _HISTORICAL_PRICE_COLS_FALLBACK,
            "historical_targets": _HISTORICAL_PRICE_TARGET_COLS_FALLBACK,
            "historical_targets_high": _HISTORICAL_PRICE_TARGET_HIGH_COLS_FALLBACK,
            "historical_targets_low": _HISTORICAL_PRICE_TARGET_LOW_COLS_FALLBACK,
            "historical_targets_median": _HISTORICAL_PRICE_TARGET_MEDIAN_COLS_FALLBACK,
            }

    return _schema_column_cache


def _apply_backfill_and_kalman(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply column backfill and Kalman momentum smoothing.

    Shared post-processing step used by both ``load_equities_data``
    and ``load_all_feature_data``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from a materialized view.

    Returns
    -------
    pd.DataFrame
        DataFrame with backfilled columns and Kalman-filtered momentum.
    """
    df = backfill_feature_columns(df)

    momentum_cols = [c for c in _KALMAN_MOMENTUM_COLS if c in df.columns]
    if momentum_cols:
        df = kalman_momentum_filter(df, momentum_cols=momentum_cols)
        logger.info("Kalman momentum filter applied to %d columns", len(momentum_cols))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Per-Model Detailed Statistics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_model_detailed_statistics(
        df: pd.DataFrame,
        model_name: str,
        key_columns: list[str],
        group_col: str = "industry",
        ) -> dict:
    """
    Compute granular statistics for a model's output DataFrame.

    Uses ``compute_metric_statistics`` for each key column and adds
    distribution shape metrics (skewness, kurtosis), inter-model
    consistency indicators, and sector-level breakdowns.
    """
    if df.empty:
        logger.warning(
                "compute_model_detailed_statistics: %s — empty DataFrame", model_name,
                )
        return {}

    results = {}
    for col in key_columns:
        if col not in df.columns:
            continue

        base_stats = compute_metric_statistics(df[col])
        if base_stats is None:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        shape_stats = {}
        if len(series) > 3:
            shape_stats["skewness"] = float(series.skew())
            shape_stats["kurtosis"] = float(series.kurtosis())
            shape_stats["iqr"] = float(base_stats["q75"] - base_stats["q25"])
            shape_stats["coefficient_of_variation"] = (
                float(series.std() / series.mean()) if series.mean() != 0 else None
            )
            mean, std = series.mean(), series.std()
            if std > 0:
                shape_stats["pct_beyond_2std"] = float(
                        ((series < mean - 2 * std) | (series > mean + 2 * std)).sum()
                        / len(series)
                        * 100,
                        )

        sector_breakdown = {}
        if group_col in df.columns:
            for sector, group in df.groupby(group_col):
                sector_stats = compute_metric_statistics(group[col])
                if sector_stats:
                    sector_breakdown[str(sector)] = {
                        "count": sector_stats["count"],
                        "mean": sector_stats["mean"],
                        "median": sector_stats["median"],
                        "std": sector_stats["std"],
                        }

        results[col] = {
            "global": base_stats,
            "distribution_shape": shape_stats,
            "sector_breakdown": sector_breakdown,
            }

    logger.info(
            "%s: computed detailed statistics for %d / %d columns",
            model_name,
            len(results),
            len(key_columns),
            )
    return results


def print_model_statistics(
        stats: dict,
        model_name: str,
        show_sectors: bool = True,
        top_n_sectors: int = 20,
        ) -> None:
    """Pretty-print the detailed statistics from compute_model_detailed_statistics."""
    if not stats:
        return

    print(f"\n  📊 {model_name} — Detailed Statistics:")
    for col, info in stats.items():
        g = info["global"]
        s = info.get("distribution_shape", {})
        print(f"    ▸ {col}:")
        print(
                f"        Count: {g['count']:,}  |  Mean: {g['mean']:.2f}  |  "
                f"Median: {g['median']:.2f}  |  Std: {g['std']:.2f}",
                )
        print(
                f"        Min: {g['min']:.2f}  |  Max: {g['max']:.2f}  |  "
                f"IQR: [{g['q25']:.2f}, {g['q75']:.2f}]",
                )
        print(
                f"        Positive: {g['positive_pct']:.1f}%  |  "
                f"Missing: {g['missing_pct']:.1f}%",
                )
        if s:
            print(
                    f"        Skew: {s.get('skewness') or 0:.3f}  |  "
                    f"Kurtosis: {s.get('kurtosis') or 0:.3f}  |  "
                    f"CV: {s.get('coefficient_of_variation') or 0:.3f}",
                    )
            if "pct_beyond_2std" in s:
                print(f"        Outliers (>2σ): {s['pct_beyond_2std']:.1f}%")

        if show_sectors and info.get("sector_breakdown"):
            sorted_sectors = sorted(
                    info["sector_breakdown"].items(),
                    key=lambda x: x[1]["mean"],
                    reverse=True,
                    )[:top_n_sectors]
            print(f"        Top {top_n_sectors} sectors by mean:")
            for sector, sinfo in sorted_sectors:
                print(
                        f"          {sector}: mean={sinfo['mean']:.2f}, "
                        f"median={sinfo['median']:.2f}, n={sinfo['count']}",
                        )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading — Equities & Feature Views Backend (v3.2)
# ═══════════════════════════════════════════════════════════════════════════════


def load_expected_returns_data(
        db_url: Optional[str] = None,
        schema: str = "public",
        ) -> tuple[pd.DataFrame, "IdentifierCoordinates | None"]:
    """
    Load equities data with full identifier coordinates.

    v3.2 migration: delegates to ``data_utils.load_equities_data_from_db``
    instead of the former ``_load_materialized_view('mv_expected_returns')``.
    Post-processing (backfill + Kalman momentum) is applied identically.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL. Falls back to DB_URL env var.
    schema : str, default "public"
        Schema containing the materialized view.

    Returns
    -------
    tuple[pd.DataFrame, IdentifierCoordinates | None]
        Feature DataFrame and identifier coordinates (None on failure).
    """
    try:
        df = load_equities_data_from_db(db_url=db_url, schema=schema)
    except (ImportError, ValueError) as e:
        logger.warning("Failed to load equities data: %s", e)
        return pd.DataFrame(), None

    id_coords = None
    if df is not None and not df.empty:
        df = _apply_backfill_and_kalman(df)

        # Build identifier coordinates for downstream use
        try:
            if IdentifierCoordinates is not None:
                id_coords = IdentifierCoordinates.from_dataframe(df)
        except (ValueError, KeyError):
            id_coords = None

        # Validate feature coverage against expected categories
        global _feature_categories_cache
        feature_categories = get_feature_categories()
        # Reconcile registry categories against actual DataFrame columns
        feature_categories = reconcile_feature_categories(
                feature_categories, set(df.columns),
                )
        _feature_categories_cache = feature_categories

        validation = validate_feature_alignment(df, feature_categories)
        low_coverage = {k: v for k, v in validation.items() if v["coverage_pct"] < 80}
        if low_coverage:
            logger.warning(
                    "Low feature coverage in %d categories: %s",
                    len(low_coverage),
                    {k: f"{v['coverage_pct']:.0f}%" for k, v in low_coverage.items()},
                    )
        else:
            logger.info("All feature categories have ≥80%% coverage")

        logger.info(
                "Loaded expected returns data: %d stocks × %d features",
                len(df),
                len(df.columns),
                )
    else:
        logger.warning("No data loaded from mv_equities")
        df = pd.DataFrame()
    return df, id_coords


def load_all_stock_features(
        db_url: Optional[str] = None,
        schema: str = "public",
        ) -> tuple[pd.DataFrame, dict[str, "FeatureViewSpec"]]:
    """
    Load full features with per-view specs for InferenceData construction.

    v3.2 migration: delegates to ``data_utils.load_all_feature_views``
    (merges all 17 ``vw_features_*`` views) to ensure full feature coverage,
    replacing the former ``_load_materialized_view('mv_all_stock_features')``.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL. Falls back to DB_URL env var.
    schema : str, default "public"
        Schema containing the feature views.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, FeatureViewSpec]]
        Full feature DataFrame and per-view specifications.
    """
    try:
        df = load_all_feature_views(db_url=db_url, schema=schema)
    except (ImportError, ValueError) as e:
        logger.warning("Failed to load feature views: %s", e)
        return pd.DataFrame(), {}

    view_specs: dict[str, FeatureViewSpec] = {}
    if df is not None and not df.empty:
        df = _apply_backfill_and_kalman(df)

        # Build view specs from registry
        if FEATURE_VIEW_REGISTRY is not None:
            for view_name, category in FEATURE_VIEW_REGISTRY.items():
                try:
                    spec = load_feature_view_spec_from_db(
                            view_name, db_url=db_url, schema=schema,
                            )
                    if spec is not None:
                        view_specs[view_name] = spec
                except Exception:
                    pass

        logger.info(
                "Loaded all stock features: %d stocks × %d features, %d view specs",
                len(df),
                len(df.columns),
                len(view_specs),
                )
    else:
        logger.warning("No data loaded from feature views")
        df = pd.DataFrame()
    return df, view_specs


def load_analytics_table(
        db_url: Optional[str] = None,
        schema: Optional[str] = None,
        earnings_date_filter: str = "2026-01-01",
        limit: Optional[int] = None,
        ) -> pd.DataFrame:
    """
    Load feature data from ``mv_all_stock_features`` via :func:`load_feature_data_from_db`.

    Delegates to ``data_utils.load_feature_data_from_db`` which queries the
    ``mv_all_stock_features`` materialized view with optional earnings-date
    filtering.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL. Falls back to ``DB_URL`` env var.
    schema : str, optional
        Database schema name. Falls back to ``DB_EQUITIES_SCHEMA`` env var
        or defaults to ``'public'``.
    earnings_date_filter : str, default "2026-01-01"
        Filter stocks with ``next_earnings >= this date`` (ISO format).
    limit : int, optional
        Maximum number of rows to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature data from ``mv_all_stock_features``,
        or empty DataFrame on failure.
    """
    try:
        df = load_feature_data_from_db(
                db_url=db_url,
                schema=schema,
                earnings_date_filter=earnings_date_filter,
                limit=limit,
                )
    except (ImportError, ValueError) as e:
        logger.warning("Failed to load mv_all_stock_features: %s", e)
        return pd.DataFrame()

    if df is not None and not df.empty:
        logger.info(
                "Loaded mv_all_stock_features: %d rows × %d columns",
                len(df),
                len(df.columns),
                )
        return df

    logger.warning("No data loaded from mv_all_stock_features")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Model Runners — column lists derived from get_equities_schema()
#    with hardcoded fallbacks when the database is unavailable.
# ═══════════════════════════════════════════════════════════════════════════════

# --- Fallback column definitions (used when equities schema DB is unreachable) ---
_MC_REQUIRED_COLS_FALLBACK = [
    "price_target",
    "price_target_high",
    "price_target_low",
    "last_price",
    ]
_KALMAN_REQUIRED_COLS_FALLBACK = ["last_price", "price_target"]

_HISTORICAL_PRICE_COLS_FALLBACK = [
    "price_5d_ago",
    "price_1w_ago",
    "price_1m_ago",
    "price_3m_ago",
    "price_6m_ago",
    "price_1y_ago",
    "price_3y_ago",
    "price_5y_ago",
    "price_qtd_ago",
    ]

_HISTORICAL_PRICE_TARGET_COLS_FALLBACK = [
    "price_target_1w_ago",
    "price_target_1m_ago",
    "price_target_3m_ago",
    "price_target_6m_ago",
    "price_target_mtd_ago",
    "price_target_qtd_ago",
    "price_target_1y_ago",
    ]

_HISTORICAL_PRICE_TARGET_HIGH_COLS_FALLBACK = [
    "price_target_high_1w_ago",
    "price_target_high_1m_ago",
    "price_target_high_6m_ago",
    "price_target_high_mtd_ago",
    "price_target_high_3m_ago",
    "price_target_high_qtd_ago",
    "price_target_high_1y_ago",
    "price_target_high_ytd_ago",
    ]

_HISTORICAL_PRICE_TARGET_LOW_COLS_FALLBACK = [
    "price_target_low_1w_ago",
    "price_target_low_1m_ago",
    "price_target_low_3m_ago",
    "price_target_low_6m_ago",
    "price_target_low_mtd_ago",
    "price_target_low_qtd_ago",
    "price_target_low_ytd_ago",
    "price_target_low_1y_ago",
    ]

_HISTORICAL_PRICE_TARGET_MEDIAN_COLS_FALLBACK = [
    "price_target_median_1w_ago",
    "price_target_median_1m_ago",
    "price_target_median_3m_ago",
    "price_target_median_6m_ago",
    "price_target_median_mtd_ago",
    "price_target_median_qtd_ago",
    "price_target_median_ytd_ago",
    "price_target_median_1y_ago",
    ]

# Backward-compatible module-level names so existing
# ``from expected_returns_v3 import _HISTORICAL_PRICE_COLS, …`` keeps working.
# At runtime the dynamic ``_get_schema_columns()`` is used instead.
_MC_REQUIRED_COLS = _MC_REQUIRED_COLS_FALLBACK
_KALMAN_REQUIRED_COLS = _KALMAN_REQUIRED_COLS_FALLBACK
_HISTORICAL_PRICE_COLS = _HISTORICAL_PRICE_COLS_FALLBACK
_HISTORICAL_PRICE_TARGET_COLS = _HISTORICAL_PRICE_TARGET_COLS_FALLBACK
_HISTORICAL_PRICE_TARGET_HIGH_COLS = _HISTORICAL_PRICE_TARGET_HIGH_COLS_FALLBACK
_HISTORICAL_PRICE_TARGET_LOW_COLS = _HISTORICAL_PRICE_TARGET_LOW_COLS_FALLBACK
_HISTORICAL_PRICE_TARGET_MEDIAN_COLS = _HISTORICAL_PRICE_TARGET_MEDIAN_COLS_FALLBACK

# All historical columns combined (for validation / feature coverage checks)
ALL_HISTORICAL_PRICE_TARGET_COLS = (
        _HISTORICAL_PRICE_COLS
        + _HISTORICAL_PRICE_TARGET_COLS
        + _HISTORICAL_PRICE_TARGET_HIGH_COLS
        + _HISTORICAL_PRICE_TARGET_LOW_COLS
        + _HISTORICAL_PRICE_TARGET_MEDIAN_COLS
)


def _resolve_available_historical_cols(
        df: pd.DataFrame,
        ) -> dict[str, list[str]]:
    """
    Identify which historical price/target columns are present in the DataFrame.

    Uses dynamic column lists from ``get_equities_schema()`` when available,
    falling back to hardcoded defaults.

    Returns a dict keyed by category name with lists of available column names.
    """
    schema_cols = _get_schema_columns()
    return {
        "historical_prices": [
            c for c in schema_cols["historical_prices"] if c in df.columns
            ],
        "historical_targets": [
            c for c in schema_cols["historical_targets"] if c in df.columns
            ],
        "historical_targets_high": [
            c for c in schema_cols["historical_targets_high"] if c in df.columns
            ],
        "historical_targets_low": [
            c for c in schema_cols["historical_targets_low"] if c in df.columns
            ],
        "historical_targets_median": [
            c for c in schema_cols["historical_targets_median"] if c in df.columns
            ],
        }


def _log_historical_coverage(available: dict[str, list[str]]) -> None:
    """Log how many historical price/target columns were found."""
    total_found = sum(len(v) for v in available.values())
    total_possible = len(ALL_HISTORICAL_PRICE_TARGET_COLS)
    logger.info(
            "Historical price/target coverage: %d / %d columns available (%s)",
            total_found,
            total_possible,
            ", ".join(f"{k}={len(v)}" for k, v in available.items()),
            )


def run_monte_carlo_analysis(
        df: pd.DataFrame,
        n_simulations: int = 25_000,
        max_stocks: int = 10_000,
        use_historical_targets: bool = True,
        ) -> pd.DataFrame:
    """
    Run Monte Carlo price target simulation on the feature DataFrame.

    v3.0: Increased default n_simulations to 25,000 for tighter confidence
    intervals on the triangular distribution sampling.

    v3.1: When ``use_historical_targets=True`` and historical price target
    columns are present, the simulation is enriched with:
    - Historical price target drift (consensus movement over time)
    - Historical price target spread evolution (high/low band changes)
    - Historical median target convergence signals

    These are passed as auxiliary columns so the downstream
    ``monte_carlo_price_target_simulation`` can optionally use them
    for informed drift and volatility priors.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with price target columns.
    n_simulations : int
        Number of triangular distribution samples per stock.
    max_stocks : int
        Cap on number of stocks to simulate.
    use_historical_targets : bool, default True
        Whether to compute and attach historical target drift columns
        to the simulation input.

    Returns
    -------
    pd.DataFrame
        Monte Carlo results with ``expected_upside_pct``, ``var_5_pct``,
        ``prob_positive_upside``, ``risk_reward_ratio``, etc.
        When historical targets are used, also includes
        ``pt_drift_1m``, ``pt_drift_3m``, ``pt_spread_change_1m``,
        ``historical_price_anchor``, and ``pt_median_drift_1m``.
    """
    mc_cols = _get_schema_columns()["mc_required"]
    missing = [c for c in mc_cols if c not in df.columns]
    if missing:
        logger.warning("MC simulation skipped — missing columns: %s", missing)
        return pd.DataFrame()

    sim_df = df.copy()

    # Enrich with historical target drift metrics when columns are available
    hist_available = _resolve_available_historical_cols(sim_df)
    if use_historical_targets:
        _log_historical_coverage(hist_available)
        sim_df = _enrich_with_historical_target_drift(sim_df, hist_available)

    mc = monte_carlo_price_target_simulation(
            sim_df,
            n_simulations=n_simulations,
            max_stocks=max_stocks,
            )
    logger.info("Monte Carlo simulation: %d stocks processed", len(mc))
    return mc


def run_price_target_achievement(
        df: pd.DataFrame,
        use_historical_targets: bool = True,
        feature_df: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
    """
    Estimate probability of reaching consensus price targets.

    Uses ``PriceTargetAchievementModel`` from probability_analytics.

    v3.1: When ``use_historical_targets=True``, historical price target
    columns are used to compute target drift and spread evolution,
    which refine the achievement probability via momentum-adjusted
    base probabilities and analyst conviction signals.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with price target and analyst sentiment columns.
    use_historical_targets : bool, default True
        Whether to enrich input with historical target drift columns.
    feature_df : pd.DataFrame or None, optional
        Full feature DataFrame (e.g. from ``load_all_feature_views``)
        containing analyst sentiment columns (``upside_potential``,
        ``analyst_conviction``, ``eps_revision_momentum``, etc.).
        When provided and these columns are missing from ``df``,
        they are merged in to ensure the model is properly calibrated.

    Returns
    -------
    pd.DataFrame
        Price target achievement results with ``achievement_probability``,
        ``expected_return_prob_weighted``, ``confidence_level``, etc.
    """
    pt_df = df.copy()

    # Ensure analyst sentiment features are present for model calibration.
    # These columns are computed by calc_sentiment_features() and
    # calc_price_target_dynamics() and surfaced via vw_features_analyst_sentiment.
    # If the primary DataFrame (mv_equities) lacks them, merge from the
    # feature views superset to avoid the model falling back to neutral defaults.
    _SENTIMENT_COLS = [
        "upside_potential",
        "analyst_conviction",
        "eps_revision_momentum",
        "analyst_rating_normalized",
        "price_target_spread_pct",
        "pt_momentum_1m",
        "pt_consensus_convergence",
        "pt_acceleration_short",
        "analyst_coverage_trend",
        "analyst_bullish_pct",
        ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing_sentiment = [
            c
            for c in _SENTIMENT_COLS
            if c not in pt_df.columns and c in feature_df.columns
            ]
        if missing_sentiment:
            sentiment_subset = feature_df[
                ["ticker"] + missing_sentiment
                ].drop_duplicates(subset="ticker")
            pt_df = pt_df.merge(sentiment_subset, on="ticker", how="left")
            logger.info(
                    "Price target achievement: merged %d sentiment columns from feature views",
                    len(missing_sentiment),
                    )

    # NEW: Enrich with risk/financial health columns for PriceTargetAchievementModel (v3.4)
    _PT_RISK_COLS = [
        "beta_1y", "beta_stability_score", "distress_risk_score",
        "balance_sheet_strength", "debt_maturity_risk",
        ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing_risk = [c for c in _PT_RISK_COLS if c not in pt_df.columns and c in feature_df.columns]
        if missing_risk:
            risk_subset = feature_df[["ticker"] + missing_risk].drop_duplicates(subset="ticker")
            pt_df = pt_df.merge(risk_subset, on="ticker", how="left")
            logger.info("Price target: merged %d risk columns", len(missing_risk))

    hist_available = _resolve_available_historical_cols(pt_df)
    if use_historical_targets:
        _log_historical_coverage(hist_available)
        pt_df = _enrich_with_historical_target_drift(pt_df, hist_available)

    model = PriceTargetAchievementModel()
    pt = model.analyze_dataframe(pt_df)
    logger.info("Price target achievement: %d stocks processed", len(pt))
    return pt


def run_kalman_filter(
        df: pd.DataFrame,
        use_historical_targets: bool = True,
        ) -> pd.DataFrame:
    """
    Apply Kalman filter to smooth noisy analyst price targets.

    Delegates to ``statistical_analysis.kalman_filter_price_target``.

    v3.1: When ``use_historical_targets=True``, historical price and
    price target columns are used to initialise the Kalman state with
    a more informed prior (anchored to recent historical price levels
    and target drift trajectories), reducing filter warm-up artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with ``last_price`` and ``price_target``.
    use_historical_targets : bool, default True
        Whether to enrich input with historical target drift columns.

    Returns
    -------
    pd.DataFrame
        Kalman-filtered results with ``filtered_upside``,
        ``kalman_estimate``, ``kalman_variance``, etc.
        When historical targets are used, also includes
        ``pt_drift_1m``, ``pt_drift_3m``, ``pt_spread_change_1m``,
        ``historical_price_anchor``, and ``pt_median_drift_1m``.
    """
    kal_cols = _get_schema_columns()["kalman_required"]
    missing = [c for c in kal_cols if c not in df.columns]
    if missing:
        logger.warning("Kalman filter skipped — missing columns: %s", missing)
        return pd.DataFrame()

    kal_df = df.copy()

    hist_available = _resolve_available_historical_cols(kal_df)
    if use_historical_targets:
        _log_historical_coverage(hist_available)
        kal_df = _enrich_with_historical_target_drift(kal_df, hist_available)

    kal = kalman_filter_price_target(kal_df)

    # v3.5: Winsorize filtered_upside at 1st/99th percentile to prevent
    # extreme outliers inflating the mean (observed 116.6% vs MC 27.3%).
    if not kal.empty and "filtered_upside" in kal.columns:
        lower, upper = kal["filtered_upside"].quantile([0.01, 0.99])
        kal["filtered_upside"] = kal["filtered_upside"].clip(lower, upper)

    logger.info("Kalman filter: %d stocks processed", len(kal))
    return kal


# ═══════════════════════════════════════════════════════════════════════════════
# Historical Price Target Drift Enrichment
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_pct_change(
        current: pd.Series,
        previous: pd.Series,
        ) -> pd.Series:
    """Compute ``((current - previous) / |previous|) * 100``, replacing ±inf with NaN."""
    prev = pd.to_numeric(previous, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = ((current - prev) / prev.abs()) * 100.0
    return result.replace([np.inf, -np.inf], np.nan)


def _compute_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Compute numeric spread ``high − low``."""
    return pd.to_numeric(high, errors="coerce") - pd.to_numeric(low, errors="coerce")


def _enrich_with_historical_target_drift(
        df: pd.DataFrame,
        hist_available: dict[str, list[str]],
        ) -> pd.DataFrame:
    """
    Compute derived drift and spread-evolution columns from historical
    price and price target data.

    Adds the following columns when the requisite inputs exist:

    - ``pt_drift_1m``   — % change in consensus target vs 1 month ago
    - ``pt_drift_3m``   — % change in consensus target vs 3 months ago
    - ``pt_drift_6m``   — % change in consensus target vs 6 months ago
    - ``pt_drift_1y``   — % change in consensus target vs 1 year ago
    - ``pt_spread_change_1m``  — change in (high − low) target spread vs 1m ago
    - ``pt_spread_change_3m``  — change in (high − low) target spread vs 3m ago
    - ``historical_price_anchor`` — best available recent historical price
                                    (5d → 1w → 1m fallback chain)
    - ``pt_median_drift_1m``  — % change in median target vs 1 month ago
    - ``pt_median_drift_3m``  — % change in median target vs 3 months ago
    - ``price_vs_historical_1m`` — % change in last_price vs price_1m_ago
    - ``price_vs_historical_3m`` — % change in last_price vs price_3m_ago
    - ``target_vs_price_convergence_1m`` — whether target drift and price
      movement are converging (positive) or diverging (negative)

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (mutated in place for performance; caller
        should pass a ``.copy()`` if the original must be preserved).
    hist_available : dict[str, list[str]]
        Output of ``_resolve_available_historical_cols``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional derived columns appended.
    """
    hist_keys = [
        "historical_targets",
        "historical_targets_high",
        "historical_targets_low",
        "historical_targets_median",
        "historical_prices",
        ]
    if not any(hist_available[k] for k in hist_keys):
        logger.debug(
                "No historical price/target columns found — skipping drift enrichment",
                )
        return df

    # --- Consensus target drift ---
    _add_drift_columns(
            df,
            current_col="price_target",
            horizons=[
                ("1m", "price_target_1m_ago"),
                ("3m", "price_target_3m_ago"),
                ("6m", "price_target_6m_ago"),
                ("1y", "price_target_1y_ago"),
                ],
            output_prefix="pt_drift",
            )

    # --- Spread evolution (high − low band width change) ---
    current_high = df.get("price_target_high")
    current_low = df.get("price_target_low")
    if current_high is not None and current_low is not None:
        current_spread = _compute_spread(current_high, current_low)
        for horizon, high_col, low_col in [
            ("1m", "price_target_high_1m_ago", "price_target_low_1m_ago"),
            ("3m", "price_target_high_3m_ago", "price_target_low_3m_ago"),
            ]:
            if high_col in df.columns and low_col in df.columns:
                prev_spread = _compute_spread(df[high_col], df[low_col])
                df[f"pt_spread_change_{horizon}"] = current_spread - prev_spread

    # --- Median target drift ---
    _add_drift_columns(
            df,
            current_col="price_target_median",
            horizons=[
                ("1m", "price_target_median_1m_ago"),
                ("3m", "price_target_median_3m_ago"),
                ],
            output_prefix="pt_median_drift",
            )

    # --- Historical price anchor (best-available recent price) ---
    anchor_chain = ["price_5d_ago", "price_1w_ago", "price_1m_ago"]
    anchor = pd.Series(np.nan, index=df.index, dtype=float)
    for col in anchor_chain:
        if col in df.columns and anchor.isna().any():
            anchor = anchor.fillna(pd.to_numeric(df[col], errors="coerce"))
    if anchor.notna().any():
        df["historical_price_anchor"] = anchor

    # --- Price momentum vs historical levels ---
    _add_drift_columns(
            df,
            current_col="last_price",
            horizons=[
                ("1m", "price_1m_ago"),
                ("3m", "price_3m_ago"),
                ],
            output_prefix="price_vs_historical",
            )

    # --- Target-vs-price convergence signal ---
    if "pt_drift_1m" in df.columns and "price_vs_historical_1m" in df.columns:
        df["target_vs_price_convergence_1m"] = (
                df["pt_drift_1m"] - df["price_vs_historical_1m"]
        )

    _DERIVED_PREFIXES = (
        "pt_drift_",
        "pt_spread_change_",
        "pt_median_drift_",
        "historical_price_anchor",
        "price_vs_historical_",
        "target_vs_price_convergence_",
        )
    n_derived = sum(1 for c in df.columns if c.startswith(_DERIVED_PREFIXES))
    logger.info(
            "Historical target drift enrichment: %d derived columns added", n_derived,
            )
    return df


def _add_drift_columns(
        df: pd.DataFrame,
        current_col: str,
        horizons: list[tuple[str, str]],
        output_prefix: str,
        ) -> None:
    """
    Add percentage-drift columns to *df* for each horizon where both
    the *current_col* and the historical column exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to mutate in place.
    current_col : str
        Column name for the current value (e.g. ``"price_target"``).
    horizons : list[tuple[str, str]]
        Pairs of ``(horizon_label, historical_column_name)``.
    output_prefix : str
        Prefix for the output column, e.g. ``"pt_drift"`` → ``"pt_drift_1m"``.
    """
    current = df.get(current_col)
    if current is None:
        return
    for horizon, hist_col in horizons:
        if hist_col in df.columns:
            df[f"{output_prefix}_{horizon}"] = _safe_pct_change(current, df[hist_col])


def run_earnings_beat_analysis(
        df: pd.DataFrame,
        feature_df: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
    """
    Run enhanced three-layer Bayesian earnings beat probability model.

    Uses ``EarningsBeatProbabilityModel.analyze_dataframe_enhanced()``
    which fuses historical EPS, revision momentum, and GAAP quality layers.
    Enriches results with EPS streak analysis via ``EPSStreakAnalyzer``,
    resampled technical priors via ``ResampledBeatProbabilityModel``,
    and classical Bayesian beat estimates via ``bayesian_earnings_beat_model``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with earnings columns.
    feature_df : pd.DataFrame or None, optional
        Full feature DataFrame for merging missing quality columns.
    """
    beat_df = df.copy()

    # NEW: Enrich with quality columns for EarningsBeatProbabilityModel (v3.4)
    _BEAT_QUALITY_COLS = [
        "accounting_quality_score", "quality_issues_count_5y", "balance_sheet_strength",
        ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing_q = [c for c in _BEAT_QUALITY_COLS if c not in beat_df.columns and c in feature_df.columns]
        if missing_q:
            q_subset = feature_df[["ticker"] + missing_q].drop_duplicates(subset="ticker")
            beat_df = beat_df.merge(q_subset, on="ticker", how="left")
            logger.info("Earnings beat: merged %d quality columns", len(missing_q))

    model = EarningsBeatProbabilityModel()
    sector_col = "sector" if "sector" in beat_df.columns else "industry"
    beat = model.analyze_dataframe_enhanced(beat_df, sector_col=sector_col)
    logger.info("Earnings beat analysis: %d stocks processed", len(beat))

    # --- EPS streak analysis (Markov-chain continuation probabilities) ---
    try:
        streak_analyzer = EPSStreakAnalyzer()
        streak_df = streak_analyzer.analyze_dataframe(df)
        if not streak_df.empty and "ticker" in streak_df.columns:
            streak_cols = [
                c for c in streak_df.columns if c != "ticker" and c not in beat.columns
                ]
            if streak_cols:
                beat = beat.merge(
                        streak_df[["ticker"] + streak_cols],
                        on="ticker",
                        how="left",
                        )
                logger.info("EPS streak enrichment: %d columns added", len(streak_cols))
    except Exception as e:
        logger.warning("EPS streak analysis failed: %s", e)

    # --- Resampled technical priors ---
    try:
        resampled_model = ResampledBeatProbabilityModel(base_model=model)
        resampled_df = resampled_model.analyze_dataframe(df)
        if not resampled_df.empty and "ticker" in resampled_df.columns:
            resamp_cols = [
                c
                for c in resampled_df.columns
                if c != "ticker" and c not in beat.columns
                ]
            if resamp_cols:
                beat = beat.merge(
                        resampled_df[["ticker"] + resamp_cols],
                        on="ticker",
                        how="left",
                        )
                logger.info(
                        "Resampled beat enrichment: %d columns added", len(resamp_cols),
                        )
    except Exception as e:
        logger.warning("Resampled beat probability failed: %s", e)

    # --- Classical Bayesian earnings beat model ---
    try:
        bayesian_beat = bayesian_earnings_beat_model(df)
        if not bayesian_beat.empty and "ticker" in bayesian_beat.columns:
            bay_cols = [
                c
                for c in bayesian_beat.columns
                if c != "ticker" and c not in beat.columns
                ]
            if bay_cols:
                beat = beat.merge(
                        bayesian_beat[["ticker"] + bay_cols],
                        on="ticker",
                        how="left",
                        )
                logger.info("Bayesian beat enrichment: %d columns added", len(bay_cols))
    except Exception as e:
        logger.warning("Bayesian earnings beat model failed: %s", e)

    return beat


def run_credit_risk_analysis(
        df: pd.DataFrame,
        feature_df: pd.DataFrame | None = None,
        *,
        n_mcmc_samples: int = 5000,
        burn_in: int = 1000,
        ) -> pd.DataFrame:
    """
    Run credit risk and ruin probability analysis.

    Uses ``CreditRiskProbabilityModel`` for Bayesian distress estimation
    and ``calculate_ruin_probability`` for analytical ruin estimates
    (modified Gambler's Ruin framework).

    Accounting anomaly detection has been extracted to
    ``run_accounting_anomaly_analysis`` (Step 5e).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with financial health columns.
    feature_df : pd.DataFrame or None, optional
        Full feature DataFrame for merging missing columns.
    """
    credit_df = df.copy()

    # Ensure quality/risk and leverage features are present for model calibration.
    _CREDIT_RISK_COLS = [
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
        # NEW: Debt trajectory (v3.4)
        "debt_3y_cagr",
        "debt_4q_trend",
        "debt_yoy_change",
        # NEW: Cash buffer
        "adequate_cash_buffer",
        "cash_vs_5y_avg",
        # NEW: Working capital deep
        "retained_earnings_vs_5y",
        # NEW: Quality & Risk
        "distress_risk_score",
        "retained_earnings_growth",
        "beta_trend",
        ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing_risk = [
            c
            for c in _CREDIT_RISK_COLS
            if c not in credit_df.columns and c in feature_df.columns
            ]
        if missing_risk:
            risk_subset = feature_df[["ticker"] + missing_risk].drop_duplicates(
                    subset="ticker",
                    )
            credit_df = credit_df.merge(risk_subset, on="ticker", how="left")
            logger.info(
                    "Credit risk analysis: merged %d risk columns from feature views",
                    len(missing_risk),
                    )

    credit_model = CreditRiskProbabilityModel(
            n_mcmc_samples=n_mcmc_samples,
            burn_in=burn_in,
            )
    credit = credit_model.analyze_dataframe(credit_df)

    # --- Hierarchical sector-level MCMC enrichment ---
    try:
        if "altman_z_score" in credit_df.columns:
            z_data = credit_df["altman_z_score"].dropna()
            if len(z_data) > 50:
                sector_mcmc = hierarchical_mcmc_by_sector(
                        credit_df, "altman_z_score",
                        )
                sector_mean_map = {
                    s: v.get("posterior_mean", np.nan)
                    for s, v in sector_mcmc.items()
                    if isinstance(v, dict)
                    }
                sector_col = "industry" if "industry" in credit.columns else "sector"
                if sector_col in credit.columns:
                    credit["sector_z_posterior_mean"] = credit[sector_col].map(
                            sector_mean_map,
                            )
                    logger.info(
                            "Hierarchical MCMC credit risk: %d sectors enriched",
                            len(sector_mean_map),
                            )
    except Exception as e:
        logger.warning("Hierarchical MCMC for credit risk failed: %s", e)

    # --- Ruin probability (Gambler's Ruin framework) ---
    try:
        ruin = calculate_ruin_probability(credit_df)
        if not ruin.empty and not credit.empty and "ticker" in ruin.columns:
            ruin_cols = [
                c
                for c in ruin.columns
                if c != "ticker" and c not in credit.columns
                ]
            if ruin_cols:
                credit = credit.merge(
                        ruin[["ticker"] + ruin_cols],
                        on="ticker",
                        how="left",
                        )
                logger.info(
                        "Ruin probability enrichment: %d columns added", len(ruin_cols),
                        )
    except Exception as e:
        logger.warning("Ruin probability calculation failed: %s", e)

    logger.info("Credit risk analysis: %d stocks processed", len(credit))
    return credit


def run_dividend_safety_analysis(
        df: pd.DataFrame,
        feature_df: pd.DataFrame | None = None,
        *,
        n_mcmc_samples: int = 5000,
        burn_in: int = 1000,
        ) -> pd.DataFrame:
    """
    Run dividend cut probability analysis.

    Uses ``DividendCutProbabilityModel`` to estimate probability of
    dividend reduction based on FCF coverage, payout ratio, streak,
    and leverage/liquidity signals.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with dividend columns.
    feature_df : pd.DataFrame or None, optional
        Full feature DataFrame for merging missing leverage/liquidity columns.

    Returns
    -------
    pd.DataFrame
        Dividend safety results with ``dividend_cut_probability``,
        ``safety_score``, ``risk_category``.
    """
    div_df = df.copy()

    # NEW: Enrich with leverage/liquidity columns for DividendCutProbabilityModel (v3.4)
    _DIV_LEVERAGE_COLS = [
        "interest_coverage", "debt_to_equity", "cash_ratio",
        "working_capital_ratio", "balance_sheet_strength",
        "cash_runway_months", "retained_earnings_growth", "debt_3y_cagr",
        ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing = [c for c in _DIV_LEVERAGE_COLS if c not in div_df.columns and c in feature_df.columns]
        if missing:
            subset = feature_df[["ticker"] + missing].drop_duplicates(subset="ticker")
            div_df = div_df.merge(subset, on="ticker", how="left")
            logger.info("Dividend safety: merged %d leverage columns", len(missing))

    model = DividendCutProbabilityModel(
            n_mcmc_samples=n_mcmc_samples,
            burn_in=burn_in,
            )
    div_safety = model.analyze_dataframe(div_df)
    logger.info("Dividend safety analysis: %d stocks processed", len(div_safety))
    return div_safety


def run_accounting_anomaly_analysis(
        df: pd.DataFrame,
        feature_df: pd.DataFrame | None = None,
        *,
        severity_anomaly_weight: float = 0.7,
        severity_feature_weight: float = 0.3,
        multi_flag_threshold: int = 10,
        anomaly_z_threshold: float | None = None,
        tier_bins: list[float] | None = None,
        tier_labels: list[str] | None = None,
        n_mcmc_samples: int = 5000,
        burn_in: int = 1000,
        ) -> pd.DataFrame:
    """
    Run standalone accounting anomaly detection and analytics.

    Delegates to :class:`AccountingAnomalyProbabilityModel` which wraps
    ``detect_accounting_anomalies`` for multi-layered statistical detection
    (robust z-scores, distribution fitting, Mahalanobis distance, Benford's
    Law) and then computes extended analytics including:

    - ``anomaly_severity_score`` — weighted combination of anomaly score
      and feature count
    - ``anomaly_risk_rank`` — universe-level percentile rank
    - ``sector_anomaly_percentile`` — within-sector rank
    - ``multi_flag_alert`` — boolean threshold flag
    - ``anomaly_conditional_probability`` — Bayesian-informed per-row
      conditional P(anomaly) via separation-weighted feature contributions

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with quality/risk and earnings columns.
    feature_df : pd.DataFrame or None, optional
        Full feature DataFrame for merging missing accounting columns.
    severity_anomaly_weight : float, default 0.7
        Weight for anomaly_score in severity computation.
    severity_feature_weight : float, default 0.3
        Weight for feature_count in severity computation.
    multi_flag_threshold : int, default 3
        Minimum flagged features to trigger multi_flag_alert.
    anomaly_z_threshold : float or None
        Robust z-score threshold for flagging anomalies. None = auto-derived.
    tier_bins : list[float] or None
        Bin edges for anomaly tier classification. None = auto-derived.
    tier_labels : list[str] or None
        Labels for the tier bins. None = ['Clean', 'Watch', 'Flag', 'Alert'].

    Returns
    -------
    pd.DataFrame
        DataFrame with anomaly scores, tiers, per-feature flags,
        Mahalanobis distance, Benford's Law test, sector-relative scoring,
        severity scores, risk ranks, and conditional anomaly probabilities.
    """
    anomaly_df = df.copy()

    # Merge accounting quality features from feature views if missing
    _ACCOUNTING_COLS = [
        "exceptional_items_frequency",
        "gaap_adj_eps_gap_pct",
        "asset_sale_boost",
        "ebitda_adjustment_ratio",
        "eps_adjustment_ratio",
        "exceptional_items_to_ebitda",
        "restructuring_intensity",
        "goodwill_change_rate",
        # ── EPS adjustment features ──
        "eps_adj_ltm",
        "eps_adjustment_ratio_comp",
        "eps_adjustment_spread_ltm",
        "eps_adjustment_spread_fy",
        "eps_adjustment_pct",
        # ── Net income adjustment features ──
        "net_income_adjustment_ratio_ltm",
        "net_income_adjustment_ratio_fy",
        "net_income_adjustment_pct",
        # ── EBITDA / EBIT adjustment features ──
        "ebitda_adjustment_pct_ltm",
        "ebitda_adjustment_pct_fy",
        "ebit_adjustment_pct_ltm",
        "ebit_adjustment_pct_fy",
        # ── GAAP vs non-GAAP spread & revision features ──
        "forward_eps_gaap_adj_spread",
        "gaap_vs_norm_revision_spread",
        "gaap_revision_momentum",
        "gaap_revision_1m",
        "gaap_revision_3m",
        "gaap_revision_6m",
        "gaap_revision_1y",
        # ── Earnings quality & discontinuities ──
        "discontinued_ops_impact",
        "earnings_quality_warning",
        "revision_quality_divergence",
        # ── Surprise & growth acceleration ──
        "eps_growth_accel",
        "eps_surprise_pct",
        "revenue_surprise_pct",
        ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing_acct = [
            c for c in _ACCOUNTING_COLS
            if c not in anomaly_df.columns and c in feature_df.columns
            ]
        if missing_acct:
            acct_subset = feature_df[["ticker"] + missing_acct].drop_duplicates(
                    subset="ticker",
                    )
            anomaly_df = anomaly_df.merge(acct_subset, on="ticker", how="left")
            logger.info(
                    "Accounting anomaly analysis: merged %d columns from feature views",
                    len(missing_acct),
                    )

    # Use the new model class with configurable parameters
    model = AccountingAnomalyProbabilityModel(
            anomaly_z_threshold=anomaly_z_threshold,
            tier_bins=tier_bins,
            tier_labels=tier_labels,
            severity_anomaly_weight=severity_anomaly_weight,
            severity_feature_weight=severity_feature_weight,
            multi_flag_threshold=multi_flag_threshold,
            n_mcmc_samples=n_mcmc_samples,
            burn_in=burn_in,
            )
    result = model.analyze_dataframe(anomaly_df)

    # --- Student-t MCMC for anomaly score posterior ---
    try:
        if "accounting_anomaly_score" in result.columns:
            anomaly_scores = result["accounting_anomaly_score"].dropna().values
            if len(anomaly_scores) > 50:
                mu_samples, df_samples = mcmc_student_t(anomaly_scores)
                result["anomaly_posterior_location"] = mu_samples.mean()
                logger.info(
                        "MCMC anomaly posterior: location=%.3f",
                        mu_samples.mean(),
                        )
    except Exception as e:
        logger.warning("MCMC anomaly posterior failed: %s", e)

    logger.info("Accounting anomaly analysis: %d stocks processed", len(result))
    return result


def run_category_probability_analysis(
        df: pd.DataFrame,
        categories: Optional[dict[str, list[str]]] = None,
        *,
        use_mcmc: bool = True,
        n_mcmc_samples: int = 5000,
        burn_in: int = 1000,
        ) -> dict[str, dict]:
    """
    Run per-category Bayesian probability analytics.

    Computes Bayesian posterior estimation, distribution fitting,
    and conditional probability analysis for each feature category.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    categories : dict, optional
        Feature categories to analyze. Defaults to FEATURE_CATEGORIES.

    Returns
    -------
    dict[str, dict]
        Per-category analytics results.
    """
    cats = categories or FEATURE_CATEGORIES
    results = {}

    for cat_name, features in cats.items():
        available = [f for f in features if f in df.columns]
        # Filter out non-numeric columns to avoid reduction errors
        available = [f for f in available if pd.api.types.is_numeric_dtype(df[f])]
        if len(available) < 2:
            continue

        try:
            cat_results = run_category_probability_analytics(
                    df,
                    cat_name,
                    available,
                    n_simulations=10_000,
                    )

            # --- CategoryProbabilityAnalyzer: Bayesian view-level analysis ---
            try:
                analyzer = CategoryProbabilityAnalyzer(
                        category_name=cat_name,
                        use_mcmc=use_mcmc,
                        n_mcmc_samples=n_mcmc_samples,
                        burn_in=burn_in,
                        )
                view_result = analyzer.analyze_view(df, feature_cols=available)
                if view_result is not None:
                    cat_results["category_probability_analysis"] = view_result
            except Exception as e:
                logger.debug(
                        "CategoryProbabilityAnalyzer skipped for %s: %s", cat_name, e,
                        )

            # --- Distribution fitting per category ---
            try:
                dist_results = fit_distributions_by_category(
                        df,
                        cat_name,
                        available,
                        )
                if dist_results:
                    cat_results["distribution_fits"] = dist_results
            except Exception as e:
                logger.debug("Distribution fitting skipped for %s: %s", cat_name, e)

            # --- Conditional probability analysis ---
            try:
                cond_probs = calculate_conditional_probabilities(
                        df,
                        {cat_name: available},
                        )
                if cond_probs is not None and not (
                        isinstance(cond_probs, pd.DataFrame) and cond_probs.empty
                ):
                    cat_results["conditional_probabilities"] = cond_probs
            except Exception as e:
                logger.debug(
                        "Conditional probabilities skipped for %s: %s", cat_name, e,
                        )

            results[cat_name] = cat_results
            logger.info(
                    "Category analytics: %s — %d features analyzed",
                    cat_name,
                    cat_results.get("features_analyzed", 0),
                    )
        except Exception as e:
            logger.warning("Category analytics failed for %s: %s", cat_name, e)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. Screening Runners (v3.0 — NEW)
# ═══════════════════════════════════════════════════════════════════════════════


def run_stock_screening(
        df_all: pd.DataFrame,
        ) -> dict[str, pd.DataFrame]:
    """
    Run all stock screening strategies on the full feature set.

    Uses ``mv_all_stock_features`` as input (broader feature coverage
    than ``mv_expected_returns``).

    Parameters
    ----------
    df_all : pd.DataFrame
        Full feature DataFrame from ``mv_all_stock_features``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Screening results keyed by strategy name.
    """
    screens: dict[str, pd.DataFrame] = {}

    # Quality screening (dynamic thresholds from data distributions)
    try:
        screens["quality"] = create_enhanced_screener(df_all)
        logger.info("Quality screen: %d stocks", len(screens["quality"]))
        if screens["quality"].empty:
            logger.warning(
                    "Quality screen returned 0 stocks — thresholds may be too restrictive "
                    "for current data. Consider relaxing create_enhanced_screener params.",
                    )
    except Exception as e:
        logger.warning("Quality screening failed: %s", e)

    # Earnings quality
    try:
        screens["earnings_quality"] = screen_earnings_quality(df_all)
        logger.info(
                "Earnings quality screen: %d stocks", len(screens["earnings_quality"]),
                )
    except Exception as e:
        logger.warning("Earnings quality screening failed: %s", e)

    # Value opportunities
    try:
        screens["value"] = screen_value_opportunities(df_all)
        logger.info("Value screen: %d stocks", len(screens["value"]))
    except Exception as e:
        logger.warning("Value screening failed: %s", e)

    # Growth momentum
    try:
        screens["growth"] = screen_growth_momentum(df_all)
        logger.info("Growth screen: %d stocks", len(screens["growth"]))
    except Exception as e:
        logger.warning("Growth screening failed: %s", e)

    # GARP (Growth at a Reasonable Price)
    try:
        screens["garp"] = screen_garp_opportunities(df_all)
        logger.info("GARP screen: %d stocks", len(screens["garp"]))
    except Exception as e:
        logger.warning("GARP screening failed: %s", e)

    # Dividend quality
    try:
        screens["dividend"] = screen_dividend_quality(df_all)
        logger.info("Dividend screen: %d stocks", len(screens["dividend"]))
        if 0 < len(screens["dividend"]) < 50:
            logger.warning(
                    "Dividend screen returned only %d stocks (%.1f%% of universe) — "
                    "filter criteria may be too restrictive for a global equity universe.",
                    len(screens["dividend"]),
                    100.0 * len(screens["dividend"]) / max(len(df_all), 1),
                    )
    except Exception as e:
        logger.warning("Dividend screening failed: %s", e)

    # Financial health
    try:
        screens["healthy"] = screen_financial_health(df_all)
        logger.info("Financial health screen: %d stocks", len(screens["healthy"]))
    except Exception as e:
        logger.warning("Financial health screening failed: %s", e)

    # Valuation reversion candidates
    try:
        screens["valuation_reversion"] = screen_valuation_reversion_candidates(df_all)
        logger.info(
                "Valuation reversion screen: %d stocks", len(screens["valuation_reversion"]),
                )
    except Exception as e:
        logger.warning("Valuation reversion screening failed: %s", e)

    # Integrity-filtered growth
    try:
        screens["integrity_growth"] = screen_integrity_filtered_growth(df_all)
        logger.info(
                "Integrity growth screen: %d stocks", len(screens["integrity_growth"]),
                )
    except Exception as e:
        logger.warning("Integrity growth screening failed: %s", e)

    # High-yield safe dividends
    try:
        screens["high_yield_safe"] = screen_high_yield_safe_dividends(df_all)
        logger.info(
                "High-yield safe dividend screen: %d stocks",
                len(screens["high_yield_safe"]),
                )
    except Exception as e:
        logger.warning("High-yield safe dividend screening failed: %s", e)

    # Sector-relative ranking (composite score)
    try:
        screens["sector_relative"] = create_sector_relative_ranking(
                df_all,
                metric="composite_score"
                if "composite_score" in df_all.columns
                else "upside_potential",
                )
        logger.info(
                "Sector-relative ranking: %d stocks",
                len(screens["sector_relative"]),
                )
    except Exception as e:
        logger.warning("Sector-relative ranking failed: %s", e)

    # Low-volatility quality (Enhancement 2+3)
    try:
        screens["low_vol_quality"] = screen_low_volatility_quality(df_all)
        logger.info(
                "Low-volatility quality screen: %d stocks",
                len(screens["low_vol_quality"]),
                )
    except Exception as e:
        logger.warning("Low-volatility quality screening failed: %s", e)

    # FCF growth compounders (Enhancement 4+5+9+12)
    try:
        screens["fcf_compounders"] = screen_fcf_growth_compounders(df_all)
        logger.info(
                "FCF compounders screen: %d stocks",
                len(screens["fcf_compounders"]),
                )
    except Exception as e:
        logger.warning("FCF compounders screening failed: %s", e)

    # Total return leaders (Enhancement 1)
    try:
        screens["total_return_leaders"] = screen_total_return_leaders(df_all)
        logger.info(
                "Total return leaders screen: %d stocks",
                len(screens["total_return_leaders"]),
                )
    except Exception as e:
        logger.warning("Total return leaders screening failed: %s", e)

    return screens


def filter_quality_stocks(
        summary: pd.DataFrame,
        source_df: pd.DataFrame,
        ) -> pd.DataFrame:
    """
    Apply quality screening to the expected returns summary.

    Enriches the summary with a ``quality_tier`` from composite scoring
    and flags financially healthy stocks.
    """
    if summary.empty or source_df.empty:
        return summary

    ranked = rank_stocks_by_composite_score(source_df)
    if "composite_score" in ranked.columns and "ticker" in ranked.columns:
        score_map = ranked.set_index("ticker")["composite_score"]
        summary["composite_score"] = summary["ticker"].map(score_map)

        summary["quality_tier"] = pd.cut(
                summary["composite_score"],
                bins=[0, 30, 50, 70, 100],
                labels=["Low", "Below Avg", "Above Avg", "High"],
                )
        logger.info(
                "Quality scoring: %d High, %d Above Avg",
                (summary["quality_tier"] == "High").sum(),
                (summary["quality_tier"] == "Above Avg").sum(),
                )

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Tri-Model & Quad-Model Alignment
# ═══════════════════════════════════════════════════════════════════════════════

_SIGNAL_LABELS = {
    0: "Strong Bearish (0/3)",
    1: "Bearish (1/3)",
    2: "Bullish (2/3)",
    3: "Strong Bullish (3/3)",
    }

_SIGNAL_LABELS_4 = {
    0: "Strong Bearish (0/4)",
    1: "Bearish (1/4)",
    2: "Neutral (2/4)",
    3: "Bullish (3/4)",
    4: "Strong Bullish (4/4)",
    }


def build_tri_model_alignment(
        mc: pd.DataFrame,
        kal: pd.DataFrame,
        pt: pd.DataFrame,
        ) -> pd.DataFrame:
    """
    Merge Monte Carlo, Kalman, and Price Target Achievement into a
    tri-model alignment DataFrame with direction agreement scores.
    """
    if mc.empty or kal.empty or pt.empty:
        logger.warning("Tri-model alignment skipped — one or more inputs empty")
        return pd.DataFrame()

    id_cols_set = get_identifier_cols_set()
    mc_id_cols = [c for c in mc.columns if c in id_cols_set]
    mc_select = list(
            set(
                    mc_id_cols
                    + [
                        "ticker",
                        "expected_upside_pct",
                        "price_target_mc",
                        "prob_positive_upside",
                        "var_5_pct",
                        "risk_reward_ratio",
                        ],
                    ),
            )

    tri = (
        mc[mc_select]
        .copy()
        .merge(
                kal[["ticker", "filtered_upside", "kalman_estimate", "kalman_variance"]],
                on="ticker",
                how="inner",
                )
        .merge(
                pt[
                    [
                        "ticker",
                        "expected_return_prob_weighted",
                        "achievement_probability",
                        "price_target_prob_weighted",
                        "confidence_level",
                        "analyst_conviction",
                        "eps_revision_momentum",
                        "analyst_rating_normalized",
                        ]
                ],
                on="ticker",
                how="inner",
                )
    )

    tri["mc_bullish"] = tri["expected_upside_pct"] > 0
    tri["kal_bullish"] = tri["filtered_upside"] > 0
    tri["pt_bullish"] = tri["expected_return_prob_weighted"] > 0
    tri["agreement_score"] = (
            tri["mc_bullish"].astype(int)
            + tri["kal_bullish"].astype(int)
            + tri["pt_bullish"].astype(int)
    )
    tri["signal"] = tri["agreement_score"].map(_SIGNAL_LABELS)

    logger.info(
            "Tri-model alignment: %d stocks, %d strong bullish",
            len(tri),
            (tri["agreement_score"] == 3).sum(),
            )
    return tri


def build_quad_model_alignment(
        tri: pd.DataFrame,
        beat: pd.DataFrame,
        beat_threshold: float = 0.6,
        ) -> pd.DataFrame:
    """Extend tri-model alignment with earnings beat probability for 4-model scoring."""
    if tri.empty or beat.empty:
        logger.warning("Quad-model alignment skipped — insufficient data")
        return pd.DataFrame()

    if "prob_beat_given_momentum" not in beat.columns:
        logger.warning("Quad-model skipped — beat results missing prob_beat_given_momentum")
        return pd.DataFrame()

    beat_slim = beat[["ticker", "prob_beat_given_momentum"]].rename(
            columns={"prob_beat_given_momentum": "beat_prob"},
            )
    quad = tri.merge(beat_slim, on="ticker", how="inner")
    if quad.empty:
        return quad

    quad["beat_bullish"] = (quad["beat_prob"] >= beat_threshold).astype(int)
    quad["quad_agreement"] = (
            quad["mc_bullish"].astype(int)
            + quad["kal_bullish"].astype(int)
            + quad["pt_bullish"].astype(int)
            + quad["beat_bullish"]
    )

    logger.info(
            "Quad-model alignment: %d stocks, full consensus (4/4): %d",
            len(quad),
            (quad["quad_agreement"] == 4).sum(),
            )
    return quad


def build_expected_returns_summary(
        mc: pd.DataFrame,
        kal: pd.DataFrame,
        pt: pd.DataFrame,
        earn: pd.DataFrame,
        anomaly_results: pd.DataFrame,
        source_df: pd.DataFrame | None = None,
        credit: pd.DataFrame | None = None,
        div_safety: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
    """
    Merge four expected-return model results into a unified summary DataFrame.

    v3.0: ``source_df`` is loaded from ``mv_all_stock_features`` (full
    superset) so that all identifier and market-data columns are available
    for enrichment without needing a backfill step.

    v3.3: Added optional ``credit`` and ``div_safety`` DataFrames to enrich
    the summary with credit-risk and dividend-safety columns.
    """
    if mc.empty or kal.empty or pt.empty or earn.empty:
        logger.warning(
                "Expected returns summary skipped — one or more inputs empty "
                "(mc=%d, kal=%d, pt=%d, earn=%d)",
                len(mc),
                len(kal),
                len(pt),
                len(earn),
                )
        return pd.DataFrame()

    id_cols_set = get_identifier_cols_set()
    mc_id_cols = [c for c in mc.columns if c in id_cols_set]

    market_data_cols = [
        "market_cap",
        "enterprise_value",
        "last_price",
        "price_target",
        "price_target_high",
        "price_target_low",
        "price_target_median",
        "volume_shrs",
        "shares_outstanding",
        ]
    available_market = [c for c in market_data_cols if c in mc.columns]

    mc_select = list(
            set(
                    mc_id_cols
                    + [
                        "ticker",
                        "expected_upside_pct",
                        "price_target_mc",
                        "prob_positive_upside",
                        "var_5_pct",
                        "risk_reward_ratio",
                        ]
                    + available_market,
                    ),
            )

    summary = (
        mc[mc_select]
        .copy()
        .merge(
                kal[["ticker", "filtered_upside", "kalman_estimate"]],
                on="ticker",
                how="inner",
                )
        .merge(
                pt[
                    [
                        "ticker",
                        "expected_return_prob_weighted",
                        "price_target_prob_weighted",
                        "achievement_probability",
                        "mh_achievement_probability",
                        "confidence_level",
                        "analyst_conviction",
                        "bullish_pct",
                        "eps_revision_momentum",
                        "analyst_rating_normalized",
                        ]
                ],
                on="ticker",
                how="inner",
                )
        .merge(
                earn[
                    [
                        "ticker",
                        "posterior_beat_prob",
                        "posterior_std",
                        "confidence_score",
                        "beat_classification",
                        "base_posterior_mean",
                        "resampled_posterior_mean",
                        "technical_adjustment",
                        "momentum_signal",
                        "volatility_regime_score",
                        "credible_interval_90",
                        "credible_interval_95",
                        "prob_beat_given_momentum",
                        "streak_type",
                        "continuation_probability",
                        "mean_reversion_probability",
                        "expected_next_outcome",
                        "prediction_confidence",
                        "model_confidence",
                        "map_estimate",
                        ]
                ],
                on="ticker",
                how="inner",
                )
    )

    # Merge anomaly results (accounting anomaly detection columns)
    _ANOMALY_COLS = [
        " gross_profit_margin_pct_fy",
        "gross_profit_margin_pct_ltm",
        "buyback_yield_ltm",
        "div_yield_1fyind",
        "div_yield_ttm",
        "div_yield_ntm",
        "div_yield_5yavgltm",
        "revenues_est_yoy_pct_fy1e",
        "price_chg_pct_1m",
        "price_chg_pct_3m",
        "one_day_pct",
        "eps_est_avg_rev_pct_fy1e_1w",
        "eps_est_avg_rev_pct_fy1e_1m",
        "eps_est_avg_rev_pct_fy1e_3m",
        "eps_est_avg_rev_pct_fy1e_6m",
        "eps_est_avg_rev_pct_fy1e_1y",
        "div_yield_2fyind",
        "div_yield_3fyind",
        "div_yield_4fyind",
        "div_yield_5fyind",
        "eps_gaap_est_avg_rev_pct_fy1e_1m",
        "eps_gaap_est_avg_rev_pct_fy1e_3m",
        "eps_gaap_est_avg_rev_pct_fy1e_6m",
        "eps_gaap_est_avg_rev_pct_fy1e_1y",
        "dividend_streak",
        "price_target_count",
        "analyst_rating",
        "num_strong_sell_ratings",
        "num_strong_buys_ratings",
        "num_hold_ratings",
        "num_buys_ratings",
        "num_sell_ratings",
        "num_no_opinion_ratings",
        "accounting_anomaly_score",
        "sector_relative_anomaly",
        "anomaly_feature_count",
        "accounting_anomaly_tier",
        "anomaly_severity_score",
        "anomaly_risk_rank",
        "sector_anomaly_percentile",
        "sector_posterior_mean",
        "multi_flag_alert",
        "anomaly_conditional_probability",
        "mh_anomaly_probability",
        ]
    if (
            anomaly_results is not None
            and not anomaly_results.empty
            and "ticker" in anomaly_results.columns
    ):
        available_anomaly = [
            c for c in _ANOMALY_COLS if c in anomaly_results.columns
            ]
        if available_anomaly:
            anomaly_subset = anomaly_results[
                ["ticker"] + available_anomaly
                ].drop_duplicates(subset="ticker")
            summary = summary.merge(anomaly_subset, on="ticker", how="left")
            logger.info(
                    "Enriched expected_returns_summary with %d anomaly columns",
                    len(available_anomaly),
                    )

    # Merge credit risk columns
    _CREDIT_COLS = [
        "beta_stability_score",
        "distress_probability",
        "liquidity_stress_score",
        "cash_runway_months",
        "altman_z_score",
        "altman_z_trend",
        "risk_level",
        "data_quality_score",
        "wealth_buffer",
        "ruin_probability",
        "survival_probability",
        # NEW: v3.4 enrichment columns from CreditRiskProbabilityModel
        "debt_3y_cagr",
        "debt_maturity_risk",
        "balance_sheet_strength",
        "wc_efficiency_score",
        "distress_risk_score",
        ]
    if (
            credit is not None
            and not credit.empty
            and "ticker" in credit.columns
    ):
        available_credit = [
            c for c in _CREDIT_COLS if c in credit.columns
            ]
        if available_credit:
            credit_subset = credit[
                ["ticker"] + available_credit
                ].drop_duplicates(subset="ticker")
            summary = summary.merge(credit_subset, on="ticker", how="left")
            logger.info(
                    "Enriched expected_returns_summary with %d credit risk columns",
                    len(available_credit),
                    )

    # Merge dividend safety columns
    _DIV_SAFETY_COLS = [
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
    if (
            div_safety is not None
            and not div_safety.empty
            and "ticker" in div_safety.columns
    ):
        available_div = [
            c for c in _DIV_SAFETY_COLS if c in div_safety.columns
            ]
        if available_div:
            div_subset = div_safety[
                ["ticker"] + available_div
                ].drop_duplicates(subset="ticker")
            summary = summary.merge(div_subset, on="ticker", how="left")
            logger.info(
                    "Enriched expected_returns_summary with %d dividend safety columns",
                    len(available_div),
                    )

    if summary.empty:
        logger.warning(
                "Expected returns summary: no overlapping tickers across all 4 models",
                )
        return summary

    # Enrich with market-data columns from mc (if present there)
    for col in available_market:
        if col not in summary.columns and col in mc.columns:
            price_map = (
                mc[["ticker", col]]
                .drop_duplicates(subset="ticker")
                .set_index("ticker")[col]
            )
            summary[col] = summary["ticker"].map(price_map)
            logger.debug("Merged market-data column '%s' from mc", col)

    # Enrich from source_df (mv_all_stock_features)
    if source_df is not None and "ticker" in source_df.columns:
        id_cols_ordered = load_identifier_columns()
        desired_cols = id_cols_ordered + market_data_cols
        missing_cols = [
            c
            for c in desired_cols
            if c in source_df.columns and c not in summary.columns
            ]
        if missing_cols:
            source_subset = source_df[["ticker"] + missing_cols].drop_duplicates(
                    subset="ticker",
                    )
            summary = summary.merge(source_subset, on="ticker", how="left")
            logger.info(
                    "Enriched expected_returns_summary with %d columns from mv_all_stock_features",
                    len(missing_cols),
                    )

    # Direction flags
    summary["mc_bullish"] = summary["expected_upside_pct"] > 0
    summary["kal_bullish"] = summary["filtered_upside"] > 0
    summary["pt_bullish"] = summary["expected_return_prob_weighted"] > 0
    summary["earn_bullish"] = summary["prob_beat_given_momentum"] >= 0.6

    # Agreement score: 0–4
    summary["agreement_score"] = (
            summary["mc_bullish"].astype(int)
            + summary["kal_bullish"].astype(int)
            + summary["pt_bullish"].astype(int)
            + summary["earn_bullish"].astype(int)
    )
    summary["signal"] = summary["agreement_score"].map(_SIGNAL_LABELS_4)

    # Confidence-weighted agreement (continuous 0–4 scale)
    mc_weight = summary["prob_positive_upside"].clip(0, 100) / 100.0
    kal_weight = 0.5
    pt_weight = (
        summary["confidence_level"]
        .map({"High": 0.9, "Medium": 0.6, "Low": 0.3})
        .fillna(0.5)
    )
    earn_weight = summary["confidence_score"].clip(0, 1)

    summary["weighted_agreement"] = (
            summary["mc_bullish"].astype(float) * mc_weight
            + summary["kal_bullish"].astype(float) * kal_weight
            + summary["pt_bullish"].astype(float) * pt_weight
            + summary["earn_bullish"].astype(float) * earn_weight
    )

    # v3.5: Remove duplicate columns before return to prevent export failures
    # (e.g. 'model_confidence' appearing from both earnings merge and source_df).
    summary = summary.loc[:, ~summary.columns.duplicated()]

    logger.info(
            "Expected returns summary: %d stocks, %d strong bullish (4/4)",
            len(summary),
            (summary["agreement_score"] == 4).sum(),
            )
    return summary


def extract_strong_consensus(
        tri: pd.DataFrame,
        min_prob_positive: float = 70.0,
        min_achievement: float = 0.7,
        top_n: int = 50,
        ) -> pd.DataFrame:
    """Filter strong consensus picks — all 3 models bullish with high confidence."""
    if tri.empty:
        return pd.DataFrame()

    strong = tri[
        (tri["agreement_score"] == 3)
        & (tri["prob_positive_upside"] >= min_prob_positive)
        & (tri["achievement_probability"] >= min_achievement)
        ].nlargest(top_n, "expected_upside_pct")

    logger.info("Strong consensus picks: %d stocks", len(strong))
    return strong


# ═══════════════════════════════════════════════════════════════════════════════
# Price Target Computation (unified)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_derived_price_target(
        df: pd.DataFrame,
        source_df: pd.DataFrame,
        price_col: str = "last_price",
        return_col: str = "expected_upside_pct",
        output_col: str = "price_target_derived",
        ) -> pd.DataFrame:
    """
    Calculate a derived price target from a return-percentage column.

    ``output_col = price_col * (1 + return_col / 100)``

    Generalised helper that replaces the formerly duplicated
    ``compute_price_target_mc`` and ``compute_price_target_prob_weighted``.

    Parameters
    ----------
    df : pd.DataFrame
        Model output DataFrame.
    source_df : pd.DataFrame
        Source DataFrame with ``price_col`` for price lookups.
    price_col : str, default "last_price"
        Column containing the current price.
    return_col : str
        Column containing the expected return percentage.
    output_col : str
        Name of the derived price target column to create.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``output_col`` appended.
    """
    if df.empty:
        logger.warning("compute_derived_price_target: empty input — skipping")
        return df

    result = df.copy()

    if price_col not in result.columns:
        if "ticker" not in source_df.columns or price_col not in source_df.columns:
            logger.warning(
                    "Cannot compute %s — '%s' or 'ticker' missing from source_df",
                    output_col,
                    price_col,
                    )
            result[output_col] = np.nan
            return result

        price_map = (
            source_df[["ticker", price_col]]
            .drop_duplicates(subset="ticker")
            .set_index("ticker")[price_col]
        )
        result[price_col] = result["ticker"].map(price_map)

    with np.errstate(invalid="ignore"):
        result[output_col] = result[price_col] * (1 + result[return_col] / 100.0)

    result[output_col] = result[output_col].replace([np.inf, -np.inf], np.nan)

    valid_count = result[output_col].notna().sum()
    logger.info(
            "Computed %s for %d / %d stocks (mean=%.2f)",
            output_col,
            valid_count,
            len(result),
            result[output_col].mean() if valid_count > 0 else 0.0,
            )
    return result


def compute_price_target_prob_weighted(
        pt: pd.DataFrame,
        source_df: pd.DataFrame,
        price_col: str = "last_price",
        return_col: str = "expected_return_prob_weighted",
        output_col: str = "price_target_prob_weighted",
        ) -> pd.DataFrame:
    """Calculate price target from probability-weighted return. Delegates to ``compute_derived_price_target``."""
    return compute_derived_price_target(
            pt, source_df, price_col, return_col, output_col,
            )


def compute_price_target_mc(
        pt: pd.DataFrame,
        source_df: pd.DataFrame,
        price_col: str = "last_price",
        return_col: str = "expected_upside_pct",
        output_col: str = "price_target_mc",
        ) -> pd.DataFrame:
    """Calculate price target from Monte Carlo expected upside. Delegates to ``compute_derived_price_target``."""
    return compute_derived_price_target(
            pt, source_df, price_col, return_col, output_col,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Analytical Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compute_sector_expected_returns(tri: pd.DataFrame) -> pd.DataFrame:
    """Aggregate expected return metrics by industry sector across all models."""
    if tri.empty or "industry" not in tri.columns:
        return pd.DataFrame()

    return (
        tri.groupby("industry")
        .agg(
                mc_mean=("expected_upside_pct", "mean"),
                mc_median=("expected_upside_pct", "median"),
                kalman_mean=("filtered_upside", "mean"),
                kalman_median=("filtered_upside", "median"),
                pt_mean=("expected_return_prob_weighted", "mean"),
                pt_median=("expected_return_prob_weighted", "median"),
                pct_bullish=("agreement_score", lambda x: (x == 3).mean() * 100),
                count=("ticker", "count"),
                )
        .reset_index()
        .sort_values("pt_mean", ascending=False)
    )


def compute_sector_return_analytics(
        summary: pd.DataFrame,
        group_col: str = "industry",
        ) -> pd.DataFrame:
    """
    Extended sector-level analytics with confidence intervals,
    distribution shape, and hit-rate diagnostics.
    """
    if summary.empty or group_col not in summary.columns:
        return pd.DataFrame()

    results = []
    for sector, group in summary.groupby(group_col):
        n = len(group)
        row = {group_col: sector, "count": n}

        for col, prefix in [
            ("expected_upside_pct", "mc"),
            ("filtered_upside", "kalman"),
            ("expected_return_prob_weighted", "pt"),
            ]:
            if col in group.columns:
                s = group[col].dropna()
                row[f"{prefix}_mean"] = float(s.mean()) if len(s) > 0 else None
                row[f"{prefix}_median"] = float(s.median()) if len(s) > 0 else None
                row[f"{prefix}_std"] = float(s.std()) if len(s) > 1 else None
                if len(s) > 1:
                    se = s.std() / np.sqrt(len(s))
                    row[f"{prefix}_ci_low"] = float(s.mean() - 1.96 * se)
                    row[f"{prefix}_ci_high"] = float(s.mean() + 1.96 * se)
                if len(s) > 3:
                    row[f"{prefix}_skew"] = float(s.skew())
                    row[f"{prefix}_kurtosis"] = float(s.kurtosis())

        if "agreement_score" in group.columns:
            row["pct_bullish_3plus"] = float(
                    (group["agreement_score"] >= 3).mean() * 100,
                    )
            row["pct_full_consensus"] = float(
                    (group["agreement_score"] == 3).mean() * 100,
                    )
        if "weighted_agreement" in group.columns:
            row["mean_weighted_agreement"] = float(group["weighted_agreement"].mean())

        if "expected_upside_pct" in group.columns:
            mc_mean = group["expected_upside_pct"].mean()
            mc_std = group["expected_upside_pct"].std()
            row["risk_adjusted_return"] = (
                float(mc_mean / mc_std) if mc_std > 0 else None
            )

        if "prob_beat_given_momentum" in group.columns:
            row["mean_beat_prob"] = float(group["prob_beat_given_momentum"].mean())

        results.append(row)

    return (
        pd.DataFrame(results)
        .sort_values("mc_mean", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def compute_return_zscore_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Add industry-relative z-scores and percentile ranks for key return metrics."""
    if summary.empty:
        return summary

    return_cols = [
        c
        for c in [
            "expected_upside_pct",
            "filtered_upside",
            "expected_return_prob_weighted",
            ]
        if c in summary.columns
        ]
    group_col = "industry" if "industry" in summary.columns else None

    if return_cols:
        summary = vectorized_zscore(summary, return_cols, group_col=group_col)
        summary = vectorized_percentile_rank(summary, return_cols, group_col=group_col)
        logger.info(
                "Added z-scores and percentile ranks for %d return metrics",
                len(return_cols),
                )

    return summary


def compute_cross_model_correlation(mc: pd.DataFrame, kal: pd.DataFrame) -> dict:
    """Compute correlation and copula dependency between MC and Kalman returns."""
    if mc.empty or kal.empty:
        return {"correlation": None, "n_stocks": 0}

    mc_cols = {"ticker", "expected_upside_pct"}
    kal_cols = {"ticker", "filtered_upside"}
    if not mc_cols.issubset(mc.columns) or not kal_cols.issubset(kal.columns):
        return {"correlation": None, "n_stocks": 0}

    merged = mc[["ticker", "expected_upside_pct"]].merge(
            kal[["ticker", "filtered_upside"]],
            on="ticker",
            how="inner",
            )
    if len(merged) < 10:
        return {"correlation": None, "n_stocks": len(merged)}

    corr = merged[["expected_upside_pct", "filtered_upside"]].corr().iloc[0, 1]
    result: dict = {"correlation": float(corr), "n_stocks": len(merged)}

    if len(merged) > 50:
        try:
            copula = fit_gaussian_copula(
                    merged,
                    features=["expected_upside_pct", "filtered_upside"],
                    )
            if copula:
                result["tail_dependence"] = copula.get("tail_dependence")
        except Exception as e:
            logger.debug("Copula fit skipped: %s", e)

    return result


def compute_cross_model_diagnostics(summary: pd.DataFrame) -> dict:
    """Comprehensive cross-model dispersion and convergence diagnostics."""
    if summary.empty:
        return {}

    return_cols = [
        "expected_upside_pct",
        "filtered_upside",
        "expected_return_prob_weighted",
        ]
    available_return_cols = [c for c in return_cols if c in summary.columns]
    if len(available_return_cols) < 2:
        return {}

    returns_df = summary[available_return_cols].dropna()

    pearson_corr = returns_df.corr(method="pearson").to_dict()
    spearman_corr = returns_df.corr(method="spearman").to_dict()

    row_means = returns_df.mean(axis=1)
    mad_per_stock = returns_df.sub(row_means, axis=0).abs().mean(axis=1)
    summary_copy = summary.loc[returns_df.index].copy()
    summary_copy["model_dispersion"] = mad_per_stock

    direction_agreement = (returns_df > 0).nunique(axis=1) == 1
    tail_agreement_pct = float(direction_agreement.mean() * 100)

    high_disp = pd.DataFrame()
    if "ticker" in summary.columns:
        summary_copy["ticker"] = summary.loc[returns_df.index, "ticker"].values
        high_disp = summary_copy.nlargest(20, "model_dispersion")[
            ["ticker", "model_dispersion"] + available_return_cols
            ]

    model_bias = {col: float(returns_df[col].mean()) for col in available_return_cols}

    try:
        from scipy.stats import kendalltau

        concordance_pairs = {}
        for i, c1 in enumerate(available_return_cols):
            for c2 in available_return_cols[i + 1:]:
                tau, p = kendalltau(returns_df[c1], returns_df[c2])
                concordance_pairs[f"{c1} ↔ {c2}"] = {
                    "kendall_tau": float(tau),
                    "p_value": float(p),
                    }
    except Exception:
        concordance_pairs = {}

    result = {
        "pairwise_pearson": pearson_corr,
        "pairwise_spearman": spearman_corr,
        "kendall_concordance": concordance_pairs,
        "mean_dispersion": float(mad_per_stock.mean()),
        "median_dispersion": float(mad_per_stock.median()),
        "tail_agreement_pct": tail_agreement_pct,
        "high_dispersion_tickers": high_disp,
        "model_bias": model_bias,
        "n_stocks": len(returns_df),
        }

    logger.info(
            "Cross-model diagnostics: tail agreement=%.1f%%, mean dispersion=%.2f",
            tail_agreement_pct,
            result["mean_dispersion"],
            )
    return result


def compute_return_distribution_analytics(
        mc: pd.DataFrame,
        summary: pd.DataFrame | None = None,
        ) -> dict:
    """Fit parametric distributions to MC simulation returns and compute risk metrics."""
    result = {}
    if mc.empty or "expected_upside_pct" not in mc.columns:
        return result

    upside = mc["expected_upside_pct"].dropna().values

    best_dist = None
    best_aic = np.inf
    candidates = [sp_stats.norm, sp_stats.t, sp_stats.skewnorm, sp_stats.laplace]

    for dist in candidates:
        try:
            params = dist.fit(upside)
            log_lik = dist.logpdf(upside, *params).sum()
            k = len(params)
            aic = 2 * k - 2 * log_lik
            if aic < best_aic:
                best_aic = aic
                best_dist = {
                    "name": dist.name,
                    "params": params,
                    "aic": float(aic),
                    "ks_statistic": float(
                            sp_stats.kstest(upside, dist.cdf, args=params).statistic,
                            ),
                    "ks_pvalue": float(
                            sp_stats.kstest(upside, dist.cdf, args=params).pvalue,
                            ),
                    }
        except Exception:
            continue

    result["mc_distribution"] = best_dist

    var_1 = float(np.percentile(upside, 1))
    var_5 = float(np.percentile(upside, 5))
    cvar_5 = float(upside[upside <= var_5].mean()) if (upside <= var_5).any() else var_5
    downside = upside[upside < 0]
    downside_deviation = (
        float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else 0.0
    )

    result["risk_metrics"] = {
        "var_1_pct": var_1,
        "var_5_pct": var_5,
        "cvar_5_pct": cvar_5,
        "downside_deviation": downside_deviation,
        "upside_capture": float((upside > 0).mean() * 100),
        "mean_positive_return": float(upside[upside > 0].mean())
        if (upside > 0).any()
        else 0.0,
        "mean_negative_return": float(downside.mean()) if len(downside) > 0 else 0.0,
        "gain_loss_ratio": (
            float(upside[upside > 0].mean() / abs(downside.mean()))
            if len(downside) > 0 and downside.mean() != 0
            else None
        ),
        }

    result["opportunity_tiers"] = {
        "high_conviction": (
            int(((upside > 20) & (mc["prob_positive_upside"].values > 70)).sum())
            if "prob_positive_upside" in mc.columns
            else 0
        ),
        "moderate": int(((upside > 0) & (upside <= 20)).sum()),
        "speculative": (
            int(
                    (
                            (upside > 0)
                            & (
                                    mc.get("prob_positive_upside", pd.Series(dtype=float)).values
                                    < 50
                            )
                    ).sum(),
                    )
            if "prob_positive_upside" in mc.columns
            else 0
        ),
        "avoid": int((upside <= 0).sum()),
        }

    # --- MCMC Student-t posterior for robust tail estimation ---
    try:
        mu_samples, df_samples = mcmc_student_t(upside, n_samples=8000, burn_in=2000)
        result["mcmc_student_t"] = {
            "posterior_mu_mean": float(np.mean(mu_samples)),
            "posterior_mu_std": float(np.std(mu_samples)),
            "posterior_df_mean": float(np.mean(df_samples)),
            "posterior_df_median": float(np.median(df_samples)),
            "heavy_tailed": bool(np.mean(df_samples) < 10),
            "ci_95": (
                float(np.percentile(mu_samples, 2.5)),
                float(np.percentile(mu_samples, 97.5)),
                ),
            }
        logger.info(
                "MCMC Student-t: \u03bc=%.2f (\u00b1%.2f), df=%.1f",
                result["mcmc_student_t"]["posterior_mu_mean"],
                result["mcmc_student_t"]["posterior_mu_std"],
                result["mcmc_student_t"]["posterior_df_mean"],
                )
    except Exception as e:
        logger.debug("MCMC Student-t skipped: %s", e)

    if summary is not None and not summary.empty:
        ensemble_cols = [
            "expected_upside_pct",
            "filtered_upside",
            "expected_return_prob_weighted",
            ]
        available = [c for c in ensemble_cols if c in summary.columns]
        if available:
            ensemble_return = summary[available].mean(axis=1).dropna()
            result["ensemble_distribution"] = compute_metric_statistics(ensemble_return)

    return result


def run_parallel_mcmc_return_analysis(
        mc: pd.DataFrame,
        n_chains: int = 8,
        n_samples: int = 50_000,
        ) -> dict:
    """
    Run parallel MCMC on MC expected upside to get converged posterior
    with Gelman-Rubin diagnostic.
    """
    if mc.empty or "expected_upside_pct" not in mc.columns:
        return {}

    data = mc["expected_upside_pct"].dropna().values
    if len(data) < 50:
        logger.warning("Parallel MCMC skipped — insufficient data (%d)", len(data))
        return {}

    result = parallel_mcmc_chains(
            data=data,
            n_chains=n_chains,
            n_samples=n_samples,
            )
    logger.info(
            "Parallel MCMC: R\u0302=%.4f, converged=%s, posterior mean=%.2f",
            result.get("r_hat", float("nan")),
            result.get("converged", False),
            result.get("posterior_mean", float("nan")),
            )
    return result


def run_resampled_posterior_analysis(
        df: pd.DataFrame,
        freq: str = "1ME",
        ) -> pd.DataFrame:
    """
    Compute Bayesian resampled return posteriors from historical price snapshots.

    Uses BayesianTechnicalResampler to derive per-stock posterior return
    distributions, providing a fifth model signal for cross-model alignment.
    """
    try:
        result_df, idata = resampled_posterior_returns(df, freq=freq, n_posterior_samples=4000, n_chains=4)
        if not result_df.empty:
            logger.info(
                    "Resampled posterior returns: %d stocks, mean posterior=%.2f%%",
                    len(result_df),
                    result_df["posterior_mean"].mean() * 100
                    if "posterior_mean" in result_df.columns
                    else 0,
                    )
        return result_df
    except Exception as e:
        logger.warning("Resampled posterior returns failed: %s", e)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualization Functions
#    Extracted to analytics.visualizations.expected_returns_viz
#    Imported at module level (create_mc_return_distribution, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Export Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def export_expected_returns_results(
        mc: pd.DataFrame,
        pt: pd.DataFrame,
        kal: pd.DataFrame,
        tri: pd.DataFrame,
        strong: pd.DataFrame,
        beat: pd.DataFrame,
        summary: pd.DataFrame | None = None,
        credit: pd.DataFrame | None = None,
        div_safety: pd.DataFrame | None = None,
        anomaly_results: pd.DataFrame | None = None,
        screens: dict[str, pd.DataFrame] | None = None,
        output_dir: str = "outputs",
        ) -> dict[str, str]:
    """
    Export all expected returns analytics to the ``analytics`` schema.

    v3.0: Added dividend safety and screening results exports.
    v3.2: Added accounting anomaly analysis export.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    exports: dict[str, str] = {}

    _EXPORT_PAIRS = [
        (mc, "monte_carlo_simulation"),
        (pt, "price_target_achievement"),
        (kal, "kalman_filtered_price_targets"),
        (tri, "expected_returns_tri_model"),
        (strong, "strong_consensus_picks"),
        (beat, "earnings_probability_analysis"),
        (summary, "expected_returns_summary"),
        (credit, "credit_risk_analysis"),
        (div_safety, "dividend_safety_analysis"),
        (anomaly_results, "accounting_anomaly_analysis"),
        ]

    for df, table in _EXPORT_PAIRS:
        if df is not None and not df.empty:
            try:
                reordered_df = reorder_with_identifiers(df)
                cfg = ExportConfig(table_name=table)
                export_to_db(reordered_df, cfg)
                exports[table] = f"analytics.{table}"
                logger.info("Exported %d rows → analytics.%s", len(df), table)
            except Exception as e:
                logger.warning("Export failed for %s: %s", table, e)

    # Export screening results
    if screens:
        _SCREEN_TABLE_MAP = {
            "quality": "quality_stocks",
            "earnings_quality": "earnings_quality_stocks",
            "value": "value_stocks",
            "growth": "growth_momentum_stocks",
            "garp": "garp_stocks",
            "dividend": "dividend_quality_stocks",
            "healthy": "healthy_stocks",
            "valuation_reversion": "valuation_reversion_stocks",
            "integrity_growth": "integrity_filtered_growth_stocks",
            "high_yield_safe": "high_yield_safe_dividend_stocks",
            "sector_relative": "sector_relative_ranking",
            }
        for screen_name, df in screens.items():
            if df is not None and not df.empty:
                table = _SCREEN_TABLE_MAP.get(screen_name, f"screen_{screen_name}")
                try:
                    reordered_df = reorder_with_identifiers(df)
                    cfg = ExportConfig(table_name=table)
                    export_to_db(reordered_df, cfg)
                    exports[table] = f"analytics.{table}"
                    logger.info(
                            "Exported screen %s: %d rows → analytics.%s",
                            screen_name,
                            len(df),
                            table,
                            )
                except Exception as e:
                    logger.warning("Export failed for screen %s: %s", screen_name, e)

    return exports


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def main(config: PipelineConfig | None = None):
    """
    Main expected returns analytics pipeline (v3.5).

    Steps:
        1.  Load feature data from materialized views (+ Kalman momentum filtering)
        1b. Pre-compute historical target drift enrichment
        2.  Run Monte Carlo simulation
        3.  Run Price Target Achievement model
        4.  Run Kalman filter
        5.  Run Earnings Beat analysis
        5b. Run Accounting Anomaly Detection
        5c. Run Credit Risk & Dividend Safety analysis
        5d. Run Stock Screening (+ productivity frontier & reporting lag enrichment)
        5e. Resampled Bayesian posterior returns (v3.1)
        6.  Build tri-model & quad-model alignment
        7.  Build expected_returns_summary (+ hierarchical sector MCMC)
        7a. Parallel MCMC return analysis with Gelman-Rubin convergence (v3.1)
        7b. Run per-category Bayesian probability analytics
        8.  Build InferenceData (ArviZ)
        9.  Generate visualizations (consuming InferenceData)
        10. Export results (deduplicated)
    """
    cfg = config or PipelineConfig.from_env()

    configure_logging(
            level=cfg.log_level,
            log_file=cfg.log_file,
            console=True,
            )

    _log_and_print("=" * 80)
    _log_and_print("Expected Returns Analytics Pipeline v3.5")
    _log_and_print("=" * 80)
    _log_and_print("")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = get_optimization_status()
    _log_and_print(
            f"🔧 Numba: {opt.get('numba_available', False)}, "
            f"Joblib: {opt.get('joblib_available', False)}",
            )
    _log_and_print("")

    # ── Initialize result containers (ensures downstream steps never see undefined vars) ──
    mc = pd.DataFrame()
    pt = pd.DataFrame()
    kal = pd.DataFrame()
    beat = pd.DataFrame()
    credit = pd.DataFrame()
    div_safety = pd.DataFrame()
    tri = pd.DataFrame()
    quad = pd.DataFrame()
    strong = pd.DataFrame()
    summary = pd.DataFrame()
    screens: dict[str, pd.DataFrame] = {}
    category_analytics: dict[str, dict] = {}
    corr_info: dict = {"correlation": None, "n_stocks": 0}
    df_features = pd.DataFrame()
    resampled_posterior = pd.DataFrame()
    mcmc_result: dict = {}
    anomaly_results = pd.DataFrame()

    # ========================================================================
    # 1. DATA LOADING (v3.2: Equities MV + Feature Views + All Stock Features MV)
    # ========================================================================
    _step_start = time.perf_counter()
    _log_and_print(
            "📦 Step 1: Loading feature data (v3.5: equities MV + feature views + all stock features MV)...",
            )
    _log_and_print("-" * 80)

    df, id_coords = load_expected_returns_data()
    if df.empty:
        _log_and_print("✗ No data loaded from mv_equities. Check DB_URL.")
        return

    _log_and_print(
            f"✓ Loaded mv_equities: {len(df):,} stocks × {len(df.columns)} features",
            )

    df_all, view_specs = load_all_stock_features()
    if not df_all.empty:
        _log_and_print(
                f"✓ Loaded feature views: {len(df_all):,} stocks × {len(df_all.columns)} features",
                )
        if view_specs:
            _log_and_print(f"  View specs loaded: {len(view_specs)} views")
    else:
        _log_and_print("⚠️ Feature views not loaded — screening will use mv_equities")
        df_all = df.copy()  # Defensive copy to prevent mutation of primary DataFrame
        view_specs = {}

    df_features = load_analytics_table()
    if not df_features.empty:
        _log_and_print(
                f"✓ Loaded mv_all_stock_features: {len(df_features):,} stocks × {len(df_features.columns)} features",
                )
    else:
        _log_and_print("⚠️ mv_all_stock_features not loaded — continuing without it")

    # ── Step 1a: Load schema metadata for InferenceData construction ──
    schema_metadata: EquitiesSchemaMetadata | None = None
    feature_registry: FeatureRegistryMetadata | None = None
    mv_equities_spec: EquitiesMaterializedViewSpec | None = None
    try:
        schema_metadata = load_equities_schema_metadata_from_db()
        feature_registry = load_feature_registry_metadata_from_db()
        if schema_metadata is not None:
            _log_and_print(f"  Schema metadata: {len(schema_metadata.column_names)} columns")
        if feature_registry is not None:
            _log_and_print(f"  Feature registry: {len(feature_registry.function_names)} functions")
    except Exception as e:
        _log_and_print(f"  Schema metadata unavailable: {e}")

    # Load mv_equities materialized view spec (column classification)
    try:
        mv_equities_spec = load_mv_equities_spec_from_db()
        if mv_equities_spec is not None:
            _log_and_print(
                    f"  mv_equities spec: {len(mv_equities_spec.price_columns)} price, "
                    f"{len(mv_equities_spec.financial_columns)} financial, "
                    f"{len(mv_equities_spec.historical_price_columns)} historical cols",
                    )
    except Exception as e:
        _log_and_print(f"  mv_equities spec unavailable: {e}")

    # Enrich identifier coordinates from dedicated view when available
    if id_coords is None:
        try:
            id_coords = load_identifier_coordinates_from_db()
            _log_and_print(
                    f"  IdentifierCoordinates (from DB): {len(id_coords.tickers)} tickers",
                    )
        except Exception as e:
            logger.debug("load_identifier_coordinates_from_db failed: %s", e)

    # ── Step 1b: Pre-compute historical target drift enrichment (used by Steps 2, 3, 4) ──
    _log_and_print("")
    _log_and_print("📦 Step 1b: Pre-computing historical target drift enrichment...")
    _log_and_print("-" * 80)
    hist_available = _resolve_available_historical_cols(df)
    _log_historical_coverage(hist_available)
    df_enriched = _enrich_with_historical_target_drift(df.copy(), hist_available)
    _log_and_print(
        f"✓ Historical drift enrichment complete ({len(df_enriched.columns) - len(df.columns)} derived columns)")

    _log_and_print(f"  ⏱ Step 1 completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 2. MONTE CARLO SIMULATION
    # ========================================================================
    _log_and_print(
            f"🎲 Step 2: Monte Carlo price target simulation ({cfg.mc_simulations:,} samples)...",
            )
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        mc = run_monte_carlo_analysis(
                df_enriched,
                n_simulations=cfg.mc_simulations,
                max_stocks=cfg.mc_max_stocks,
                use_historical_targets=False,
                )
        if not mc.empty and _has_required_columns(
                mc, ["expected_upside_pct", "prob_positive_upside"], "Monte Carlo",
                ):
            _log_and_print(f"✓ {len(mc):,} stocks simulated")
            _log_and_print(f"  Mean upside:  {mc['expected_upside_pct'].mean():.1f}%")
            _log_and_print(
                    f"  Median upside: {mc['expected_upside_pct'].median():.1f}%",
                    )
            _log_and_print(
                    f"  Positive prob (mean): {mc['prob_positive_upside'].mean():.1f}%",
                    )

            mc = compute_price_target_mc(mc, df)
            if "price_target_mc" in mc.columns:
                valid_mc = mc.dropna(subset=["price_target_mc", "last_price"])
                if not valid_mc.empty:
                    mean_price = valid_mc["last_price"].mean()
                    mean_target = valid_mc["price_target_mc"].mean()
                    implied_return = (
                                             safe_divide(mean_target, mean_price, default=1.0) - 1
                                     ) * 100
                    _log_and_print(
                            f"  Monte Carlo targets ({len(valid_mc):,} stocks): implied return={implied_return:.1f}%",
                            )

            mc_stats = compute_model_detailed_statistics(
                    mc,
                    "Monte Carlo",
                    [
                        "expected_upside_pct",
                        "prob_positive_upside",
                        "var_5_pct",
                        "risk_reward_ratio",
                        "price_target_mc",
                        ],
                    )
            print_model_statistics(
                    mc_stats, "Monte Carlo Simulation", show_sectors=True, top_n_sectors=20,
                    )

            dist_analytics = compute_return_distribution_analytics(mc)
            if dist_analytics.get("mc_distribution"):
                d = dist_analytics["mc_distribution"]
                _log_and_print(
                        f"\n  📐 Best-fit distribution: {d['name']} (AIC={d['aic']:.1f}, KS p={d['ks_pvalue']:.3f})",
                        )
            if dist_analytics.get("risk_metrics"):
                rm = dist_analytics["risk_metrics"]
                _log_and_print(
                        f"\n  📉 VaR 1%: {rm['var_1_pct']:.1f}%  |  CVaR 5%: {rm['cvar_5_pct']:.1f}%",
                        )
                _log_and_print(
                        f"     Downside deviation: {rm['downside_deviation']:.2f}",
                        )
                if rm.get("gain_loss_ratio"):
                    _log_and_print(f"     Gain/Loss ratio: {rm['gain_loss_ratio']:.2f}")
            if dist_analytics.get("opportunity_tiers"):
                t = dist_analytics["opportunity_tiers"]
                _log_and_print(
                        f"\n  🏷️  Tiers: High-conviction={t['high_conviction']}, "
                        f"Moderate={t['moderate']}, Speculative={t['speculative']}, Avoid={t['avoid']}",
                        )

    except Exception as e:
        logger.error("Step 2 (Monte Carlo) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 2 failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 2 completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 3. PRICE TARGET ACHIEVEMENT
    # ========================================================================
    _log_and_print("🎯 Step 3: Price target achievement model...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        pt = run_price_target_achievement(
                df_enriched, use_historical_targets=False, feature_df=df_all,
                )
        if not pt.empty and _has_required_columns(
                pt,
                ["achievement_probability", "expected_return_prob_weighted"],
                "Price Target",
                ):
            _log_and_print(f"✓ {len(pt):,} stocks analyzed")
            _log_and_print(
                    f"  Mean achievement prob: {pt['achievement_probability'].mean():.3f}",
                    )
            _log_and_print(
                    f"  Mean prob-weighted return: {pt['expected_return_prob_weighted'].mean():.1f}%",
                    )

            pt = compute_price_target_prob_weighted(pt, df)
            if "price_target_prob_weighted" in pt.columns:
                valid_pt = pt.dropna(
                        subset=["price_target_prob_weighted", "last_price"],
                        )
                if not valid_pt.empty:
                    mean_price = valid_pt["last_price"].mean()
                    mean_target = valid_pt["price_target_prob_weighted"].mean()
                    implied_return = (
                                             safe_divide(mean_target, mean_price, default=1.0) - 1
                                     ) * 100
                    _log_and_print(
                            f"  Prob-weighted targets ({len(valid_pt):,} stocks): implied return={implied_return:.1f}%",
                            )

            pt_stats = compute_model_detailed_statistics(
                    pt,
                    "Price Target Achievement",
                    [
                        "achievement_probability",
                        "expected_return_prob_weighted",
                        "analyst_conviction",
                        "eps_revision_momentum",
                        "price_target_prob_weighted",
                        ],
                    )
            print_model_statistics(
                    pt_stats, "Price Target Achievement", show_sectors=True,
                    )

    except Exception as e:
        logger.error("Step 3 (Price Target) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 3 failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 3 completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 4. KALMAN FILTER
    # ========================================================================
    _log_and_print("📐 Step 4: Kalman-filtered price targets...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        kal = run_kalman_filter(df_enriched, use_historical_targets=False)
        if not kal.empty and _has_required_columns(kal, ["filtered_upside"], "Kalman"):
            _log_and_print(f"✓ {len(kal):,} stocks filtered")
            _log_and_print(
                    f"  Mean filtered upside: {kal['filtered_upside'].mean():.1f}%",
                    )

            kal_stats = compute_model_detailed_statistics(
                    kal,
                    "Kalman Filter",
                    ["filtered_upside", "kalman_variance", "signal_strength"],
                    )
            print_model_statistics(kal_stats, "Kalman Filter", show_sectors=True)

    except Exception as e:
        logger.error("Step 4 (Kalman) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 4 failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 4 completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 5. EARNINGS BEAT ANALYSIS
    # ========================================================================
    _log_and_print("📊 Step 5: Bayesian earnings beat analysis...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        beat = run_earnings_beat_analysis(df_all if not df_all.empty else df, feature_df=df_all)
        if not beat.empty and _has_required_columns(
                beat, ["prob_beat_given_momentum"], "Earnings Beat",
                ):
            _log_and_print(f"✓ {len(beat):,} stocks analyzed")
            _log_and_print(f"  Mean P(beat): {beat['prob_beat_given_momentum'].mean():.3f}")
            if "beat_classification" in beat.columns:
                likely = (beat["beat_classification"] == "likely_beat").sum()
                _log_and_print(f"  Classified as 'likely_beat': {likely}")

    except Exception as e:
        logger.error("Step 5 (Earnings Beat) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5 failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 5 completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 5b. ACCOUNTING ANOMALY DETECTION (moved before credit risk)
    # ========================================================================
    _log_and_print("🔬 Step 5b: Accounting anomaly detection & analytics...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    anomaly_results = pd.DataFrame()
    try:
        anomaly_results = run_accounting_anomaly_analysis(
                df, feature_df=df_all,
                anomaly_z_threshold=cfg.anomaly_z_threshold,
                n_mcmc_samples=cfg.mcmc_burn_in * 5,
                burn_in=cfg.mcmc_burn_in,
                )

        if not anomaly_results.empty and "accounting_anomaly_score" in anomaly_results.columns:
            _log_anomaly_diagnostics(anomaly_results)

            # ── Export accounting anomaly analysis to DB ──
            try:
                anomaly_export_cols = ["ticker"] + [
                    c for c in anomaly_results.columns
                    if c.startswith("accounting_anomaly")
                       or c.endswith("_z_robust")
                       or c.endswith("_anomaly_flag")
                       or c.endswith("_dist_name")
                       or c.endswith("_dist_pvalue")
                       or c in (
                           "anomaly_feature_count",
                           "anomaly_severity_score",
                           "anomaly_risk_rank",
                           "sector_anomaly_percentile",
                           "multi_flag_alert",
                           "anomaly_conditional_probability",
                           "mahalanobis_distance",
                           "sector_relative_anomaly",
                           "benford_chi2_pvalue",
                           )
                    ]
                # Include identifier columns
                id_cols = load_identifier_columns()
                for id_col in id_cols:
                    if id_col in anomaly_results.columns and id_col not in anomaly_export_cols:
                        anomaly_export_cols.insert(1, id_col)

                anomaly_export = anomaly_results[
                    [c for c in anomaly_export_cols if c in anomaly_results.columns]
                ].copy()

                if not anomaly_export.empty:
                    anomaly_export = reorder_with_identifiers(anomaly_export)
                    anomaly_cfg = ExportConfig(table_name="accounting_anomaly_analysis")
                    export_to_db(anomaly_export, anomaly_cfg)
                    _log_and_print(
                            f"  ✓ Exported {len(anomaly_export):,} rows → analytics.accounting_anomaly_analysis",
                            )
            except Exception as e:
                logger.warning("Accounting anomaly DB export failed: %s", e)

            _log_and_print(f"✓ Accounting anomaly analysis complete: {len(anomaly_results):,} stocks")
        else:
            _log_and_print("  ⚠️ No accounting anomaly features available")

    except Exception as e:
        logger.error("Step 5b (Accounting Anomaly) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5b failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 5b completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 5c. CREDIT RISK & DIVIDEND SAFETY
    # ========================================================================
    _log_and_print("🛡️ Step 5c: Credit risk & dividend safety analysis...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        credit = run_credit_risk_analysis(
                df, feature_df=df_all,
                n_mcmc_samples=cfg.mcmc_burn_in * 5,
                burn_in=cfg.mcmc_burn_in,
                )
        if not credit.empty:
            high_risk = (
                credit["risk_level"].isin(["High", "Distressed"]).sum()
                if "risk_level" in credit.columns
                else 0
            )
            _log_and_print(
                    f"✓ Credit risk: {len(credit):,} stocks, {high_risk} high/distressed",
                    )
            if "ruin_probability" in credit.columns:
                _log_and_print(
                        f"  Mean ruin probability: {credit['ruin_probability'].mean():.3f}",
                        )

            # ── Merge anomaly columns into credit for downstream alignment ──
            if not anomaly_results.empty and "ticker" in anomaly_results.columns:
                anom_cols = [
                    c for c in anomaly_results.columns
                    if c != "ticker" and c not in credit.columns
                    ]
                if anom_cols:
                    credit = credit.merge(
                            anomaly_results[["ticker"] + anom_cols],
                            on="ticker",
                            how="left",
                            )
                    _log_and_print(
                            f"  Merged {len(anom_cols)} anomaly columns into credit risk DataFrame",
                            )

        div_safety = run_dividend_safety_analysis(
                df_all, feature_df=df_all,
                n_mcmc_samples=cfg.mcmc_burn_in * 5,
                burn_in=cfg.mcmc_burn_in,
                )
        if not div_safety.empty:
            at_risk = (
                (div_safety["risk_category"] == "At Risk").sum()
                if "risk_category" in div_safety.columns
                else 0
            )
            _log_and_print(
                    f"✓ Dividend safety: {len(div_safety):,} stocks, {at_risk} at risk",
                    )

    except Exception as e:
        logger.error("Step 5c (Credit/Dividend) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5c failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 5c completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 5d. STOCK SCREENING (v3.0 — NEW)
    # ========================================================================
    _log_and_print("🔍 Step 5d: Running stock screening strategies...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        # Enrich with productivity frontier scores before screening
        try:
            df_all = analyze_employee_productivity_frontier(df_all)
            if "productivity_frontier_score" in df_all.columns:
                _log_and_print("  \u2713 Productivity frontier scores added")
        except Exception as e:
            logger.debug("Productivity frontier enrichment skipped: %s", e)

        # Test "bad news travels slow" hypothesis
        try:
            lag_result = analyze_reporting_lag_sentiment(df_all)
            if lag_result.get("sample_size", 0) > 0:
                p_val = lag_result['p_value']
                if lag_result['hypothesis_confirmed']:
                    hyp_label = "confirmed"
                elif p_val < 0.10:
                    hyp_label = "marginally significant (p < 0.10)"
                else:
                    hyp_label = "not confirmed"
                _log_and_print(
                        f"   Reporting lag sentiment: corr={lag_result['correlation']:.3f}, "
                        f"p={p_val:.4f}, "
                        f"hypothesis={hyp_label}",
                        )
        except Exception as e:
            logger.debug("Reporting lag analysis skipped: %s", e)

        screens = run_stock_screening(df_all)
        for name, screen_df in screens.items():
            if not screen_df.empty:
                _log_and_print(f"  \u2713 {name}: {len(screen_df):,} stocks")

    except Exception as e:
        logger.error("Step 5d (Screening) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5d failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 5d completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 5e. RESAMPLED POSTERIOR RETURNS (v3.1 — NEW)
    # ========================================================================
    _log_and_print("\U0001f9ea Step 5e: Resampled Bayesian posterior returns...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        resampled_posterior = run_resampled_posterior_analysis(df)
        if not resampled_posterior.empty:
            _log_and_print(
                    f"  \u2713 Resampled posteriors: {len(resampled_posterior):,} stocks",
                    )
            if "posterior_mean" in resampled_posterior.columns:
                _log_and_print(
                        f"  Mean posterior return: {resampled_posterior['posterior_mean'].mean() * 100:.2f}%",
                        )
        else:
            _log_and_print("  \u26a0\ufe0f Resampled posterior returns: no results")

    except Exception as e:
        logger.error("Step 5e (Resampled Posterior) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5e failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 5e completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 6. CROSS-MODEL ALIGNMENT
    # ========================================================================
    _log_and_print("🔗 Step 6: Cross-model alignment...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        tri = build_tri_model_alignment(mc, kal, pt)
        strong = extract_strong_consensus(tri)
        quad = build_quad_model_alignment(tri, beat, beat_threshold=cfg.beat_threshold)

        if not tri.empty:
            _log_and_print(f"  Tri-model coverage: {len(tri):,} stocks")
            for label in _SIGNAL_LABELS.values():
                cnt = (tri["signal"] == label).sum()
                _log_and_print(f"    {label}: {cnt}")
            _log_and_print(f"  Strong consensus picks: {len(strong)}")

        if not quad.empty:
            full = (quad["quad_agreement"] == 4).sum()
            _log_and_print(f"  Quad-model (4/4): {full} stocks")

        corr_info = compute_cross_model_correlation(mc, kal)
        if corr_info.get("correlation") is not None:
            _log_and_print(f"  MC ↔ Kalman correlation: {corr_info['correlation']:.3f}")

    except Exception as e:
        logger.error("Step 6 (Alignment) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 6 failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 6 completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 7. EXPECTED RETURNS SUMMARY (4-MODEL MERGE)
    # ========================================================================
    _log_and_print("📋 Step 7: Building expected_returns_summary (4-model merge)...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    # Prefer df_features (mv_all_stock_features) as the enrichment source
    # because it is the full superset containing both identifier and market-data
    # columns.  df_all (feature views) may lack market-data columns after the
    # v3.2 migration to load_all_feature_views.  Fall back through df_all → df.
    _enrichment_source = (
        df_features if not df_features.empty else df_all if not df_all.empty else df
    )

    try:
        summary = build_expected_returns_summary(
                mc, kal, pt, beat, anomaly_results, source_df=_enrichment_source,
                credit=credit, div_safety=div_safety,
                )
        if not summary.empty:
            _log_and_print(f"  ✓ {len(summary):,} stocks in expected_returns_summary")
            full_consensus = (summary["agreement_score"] == 4).sum()
            _log_and_print(f"  Full consensus (4/4): {full_consensus} stocks")

            summary = filter_quality_stocks(summary, df_all)
            if "quality_tier" in summary.columns:
                high_quality_bullish = (
                        (summary["agreement_score"] == 4)
                        & (summary["quality_tier"].isin(["High", "Above Avg"]))
                ).sum()
                _log_and_print(
                        f"  High-quality full consensus: {high_quality_bullish} stocks",
                        )

            summary = compute_return_zscore_ranks(summary)

            _log_and_print("  Agreement distribution:")
            for label in _SIGNAL_LABELS_4.values():
                cnt = (summary["signal"] == label).sum()
                if cnt > 0:
                    _log_and_print(f"    {label}: {cnt}")

            cross_diag = compute_cross_model_diagnostics(summary)
            if cross_diag:
                _log_and_print("\n  🔬 Cross-Model Diagnostics:")
                _log_and_print(
                        f"     Direction agreement: {cross_diag['tail_agreement_pct']:.1f}%",
                        )
                _log_and_print(
                        f"     Mean inter-model dispersion: {cross_diag['mean_dispersion']:.2f}",
                        )
                for pair, info in cross_diag.get("kendall_concordance", {}).items():
                    _log_and_print(
                            f"     Kendall τ ({pair}): {info['kendall_tau']:.3f} (p={info['p_value']:.4f})",
                            )

            sector_analytics = compute_sector_return_analytics(summary)
            if not sector_analytics.empty:
                _log_and_print(
                        f"\n  \U0001f3e2 Sector Analytics: {len(sector_analytics)} sectors",
                        )

                # Hierarchical MCMC: shrinkage-based sector posteriors
                if "expected_upside_pct" in summary.columns:
                    try:
                        hier_results = hierarchical_mcmc_by_sector(
                                summary,
                                "expected_upside_pct",
                                sector_col="industry",
                                )
                        if isinstance(hier_results, dict):
                            sectors_data = hier_results.get("sectors", hier_results)
                            shrunk = {
                                s: v["posterior_mean"]
                                for s, v in sectors_data.items()
                                if isinstance(v, dict) and "posterior_mean" in v
                                }
                            if shrunk:
                                _log_and_print(
                                        f"   Hierarchical MCMC: {len(shrunk)} sector posteriors (Bayesian shrinkage)",
                                        )
                                for s in list(shrunk.keys())[:5]:
                                    raw = sectors_data[s].get("raw_mean", 0)
                                    post = shrunk[s]
                                    _log_and_print(
                                            f"     {s}: raw={raw:.1f}% \u2192 posterior={post:.1f}% "
                                            f"(shrinkage={sectors_data[s].get('shrinkage', 0):.2f})",
                                            )
                        # Multi-level hierarchical MCMC across all category columns
                        multi_hier = hierarchical_mcmc_multi_level(summary, "expected_upside_pct", group_cols=[
                            "region",
                            "country",
                            "trading_country",
                            "exchange",
                            "unit",
                            "sector",
                            "industry",
                            "style_class",
                            "size_class",
                            ], min_group_size=20, shrinkage_strength=10.0)
                        if multi_hier and "cross_level_summary" in multi_hier:
                            xls = multi_hier["cross_level_summary"]
                            if isinstance(xls, pd.DataFrame) and not xls.empty:
                                n_levels = xls["level"].nunique()
                                n_groups = len(xls)
                                _log_and_print(
                                        f"   Multi-level MCMC: {n_levels} category levels, "
                                        f"{n_groups} group posteriors",
                                        )
                                for level in xls["level"].unique():
                                    level_df = xls[xls["level"] == level]
                                    top = level_df.nlargest(n=50, columns="posterior_mean")
                                    for _, row in top.iterrows():
                                        _log_and_print(
                                                f"     [{level}] {row['group']}: "
                                                f"raw={row['raw_mean']:.1f}% → "
                                                f"posterior={row['posterior_mean']:.1f}% "
                                                f"(n={row['n_obs']}, shrink={row['shrinkage']:.2f})",
                                                )
                    except Exception as e:
                        logger.debug("Hierarchical MCMC skipped: %s", e)

                top_sectors = sector_analytics.head(50)
                group_col = (
                    "industry" if "industry" in sector_analytics.columns else "sector"
                )
                for _, row in top_sectors.iterrows():
                    consensus = row.get("pct_full_consensus", 0)
                    ra = row.get("risk_adjusted_return", 0)
                    _log_and_print(
                            f"     {row.get(group_col, 'Unknown')}: MC mean={row.get('mc_mean', 0):.1f}%, "
                            f"consensus={consensus:.0f}%, risk-adj={ra:.2f}",
                            )
        else:
            _log_and_print(
                    "  ⚠️ Expected returns summary: no overlapping tickers across 4 models",
                    )

    except Exception as e:
        logger.error("Step 7 (Summary) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 7 failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 7 completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 7a. PARALLEL MCMC RETURN ANALYSIS (v3.1 — NEW)
    # ========================================================================
    _log_and_print("\U0001f500 Step 7a: Parallel MCMC return analysis...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        mcmc_result = run_parallel_mcmc_return_analysis(mc, n_chains=cfg.mcmc_chains, n_samples=cfg.mcmc_samples)
        if mcmc_result:
            _log_and_print(
                    f"  R\u0302={mcmc_result.get('r_hat', float('nan')):.4f}, "
                    f"converged={mcmc_result.get('converged', False)}, "
                    f"posterior mean={mcmc_result.get('posterior_mean', float('nan')):.2f}",
                    )

            # Generate MCSE convergence visualization if InferenceData available
            if ARVIZ_AVAILABLE and mcmc_result.get("inference_data") is not None:
                _write_viz(
                        create_mcse_convergence_panel(mcmc_result["inference_data"],
                                                      var_name="expected_return_prob_weighted"),
                        output_dir, "er_mcse_convergence.html",
                        )
        else:
            _log_and_print("  \u26a0\ufe0f Parallel MCMC: skipped or insufficient data")

    except Exception as e:
        logger.error("Step 7a (MCMC) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 7a failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 7a completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    _log_and_print("\U0001f9ee Step 7b: Per-category Bayesian probability analytics...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        # Build categories from all 17 vw_features_* views for full feature coverage
        view_mapping = get_view_category_mapping()
        all_categories: dict[str, list[str]] = {}
        for view_name, info in view_mapping.items():
            cat_label = info.get("category", view_name)
            feat_cols = info.get("feature_cols", [])
            if feat_cols:
                # Merge features if multiple views map to the same category
                if cat_label in all_categories:
                    all_categories[cat_label].extend(
                            c for c in feat_cols if c not in all_categories[cat_label],
                            )
                else:
                    all_categories[cat_label] = list(feat_cols)
        _log_and_print(
                f"  Feature categories from {len(view_mapping)} views → "
                f"{len(all_categories)} categories, "
                f"{sum(len(v) for v in all_categories.values())} total features",
                )

        category_analytics = run_category_probability_analysis(
                df_all, categories=all_categories,
                use_mcmc=cfg.use_mcmc,
                n_mcmc_samples=cfg.mcmc_burn_in * 5,
                burn_in=cfg.mcmc_burn_in,
                )
        if category_analytics:
            _log_and_print(f"  ✓ Analyzed {len(category_analytics)} categories")
            for cat_name, cat_result in category_analytics.items():
                n_feat = cat_result.get("features_analyzed", 0)
                bayesian_keys = list(cat_result.get("bayesian_results", {}).keys())
                _log_and_print(
                        f"    {cat_name}: {n_feat} features — {len(bayesian_keys)} posteriors",
                        )
        else:
            _log_and_print("  ⚠️ No categories had sufficient features for analysis")

    except Exception as e:
        logger.error("Step 7b (Category Analytics) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 7b failed: {e}", logging.ERROR)

    _log_and_print(f"  ⏱ Step 7b completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 8. INFERENCE DATA (ArviZ) — built before visualizations
    # ========================================================================
    _step_start = time.perf_counter()
    idata_mc = None
    idata_beat = None
    idata_credit = None
    if ARVIZ_AVAILABLE:
        _log_and_print("🧪 Step 8: Building InferenceData (ArviZ)...")
        _log_and_print("-" * 80)
        try:
            if not mc.empty:
                idata_mc = build_monte_carlo_inference_data(
                        mc, df_all, n_simulations=25_000,
                        )
                idata_summary = summarize_inference_data(idata_mc)
                _log_and_print(
                        f"   ✓ MC InferenceData: {idata_summary.get('n_draws', 0)} draws, "
                        f"{idata_summary.get('n_equities', 0)} equities",
                        )
                if idata_summary.get("r_hat"):
                    for var, rhat in idata_summary["r_hat"].items():
                        _log_and_print(f"     R̂ ({var}): {rhat:.4f}")

            if not beat.empty and "posterior_alpha" in beat.columns:
                idata_beat = build_beat_probability_inference_data(
                        beat,
                        df_all,
                        n_posterior_samples=4000,
                        n_chains=4,
                        )
                beat_summary = summarize_inference_data(idata_beat)
                _log_and_print(
                        f"   ✓ Beat InferenceData: {beat_summary.get('n_chains', 0)} chains × "
                        f"{beat_summary.get('n_draws', 0)} draws",
                        )
            if not credit.empty:
                idata_credit = build_credit_risk_inference_data(
                        credit,
                        df_all,
                        )
                credit_summary = summarize_inference_data(idata_credit)
                _log_and_print(
                        f"   ✓ Credit Risk InferenceData: "
                        f"{credit_summary.get('n_equities', 0)} equities",
                        )

            # Log EquityCoordinates for traceability
            if EquityCoordinates is not None and not df_all.empty:
                try:
                    coords = EquityCoordinates.from_dataframe(df_all)
                    _log_and_print(
                            f"   ✓ EquityCoordinates: {len(coords.tickers)} tickers, "
                            f"{len(coords.sectors)} sectors",
                            )
                except Exception as e:
                    logger.debug("EquityCoordinates construction skipped: %s", e)

            # Build per-view InferenceData for ArviZ diagnostics
            if df_features is not None and not df_features.empty:
                _log_and_print("   Building per-view feature InferenceData...")
                for view_name in FEATURE_VIEW_REGISTRY:
                    try:
                        idata_view = build_feature_view_inference_data(
                                view_name, df_features,
                                )
                        view_summary = summarize_inference_data(idata_view)
                        _log_and_print(
                                f"     ✓ {view_name}: "
                                f"{view_summary.get('n_equities', 0)} equities",
                                )
                    except Exception as e:
                        logger.debug(
                                "InferenceData for %s failed: %s", view_name, e,
                                )

        except Exception as e:
            _log_and_print(f"   ⚠️ InferenceData error: {e}")
        _log_and_print("")
    else:
        _log_and_print("⏭️  Step 8: ArviZ not available — skipping InferenceData\n")

    _log_and_print(f"  ⏱ Step 8 completed in {time.perf_counter() - _step_start:.1f}s")

    # ========================================================================
    # 8b. ENRICH DataFrames WITH VIZ-CRITICAL COLUMNS
    # ========================================================================
    # Columns like altman_z_score, piotroski_f_score, beneish_m_score,
    # distress_risk_score, accounting_quality_score live in feature views
    # (df_all / df_features) but may be absent from df (mv_equities) and
    # beat.  Merge them so downstream viz functions find them.
    _VIZ_REQUIRED_COLUMNS: dict[str, list[str]] = {
        "create_quality_risk_quadrant": ["piotroski_f_score", "altman_z_score"],
        "create_distress_early_warning_dashboard": [
            "altman_z_score",
            "piotroski_f_score",
            "distress_risk_score",
            ],
        "create_beneish_mscore_analysis": [
            "beneish_m_score",
            ],
        "create_enhanced_beat_prob_dash": [
            "posterior_beat_prob",
            "prob_beat_given_momentum",
            "eps_revision_momentum",
            "gaap_adj_eps_gap_pct",
            "historical_beat_rate",
            "quarterly_beat_streak",
            "classification_confidence",
            ],
        "create_earnings_probability_dashboard": [
            "posterior_beat_prob",
            "prob_beat_given_momentum",
            "confidence_score",
            "historical_beat_rate",
            "gaap_revision_momentum",
            "gaap_vs_norm_revision_spread",
            "quarterly_beat_streak",
            ],
        "create_mcse_convergence_panel": [],  # uses InferenceData, not a DataFrame
        }

    _viz_needed_cols: set[str] = {
        col for cols in _VIZ_REQUIRED_COLUMNS.values() for col in cols
        }
    # Also include alias/fallback column names so they can be enriched
    for _col_name in list(_viz_needed_cols):
        for _alias in MV_COLUMN_ALIASES.get(_col_name, []):
            _viz_needed_cols.add(_alias)

    # Pick the richest available source for missing columns
    _viz_source = (
        df_features if not df_features.empty
        else df_all if not df_all.empty
        else pd.DataFrame()
    )

    # --- Enrich df (mv_equities) and beat with viz-critical columns ---
    df = _enrich_dataframe(df, _viz_source, _viz_needed_cols, "df (mv_equities)")
    beat = _enrich_dataframe(beat, _viz_source, _viz_needed_cols, "beat")

    # --- Validate coverage (alias-aware) ---
    _all_df_cols = set(df.columns) | set(beat.columns)
    _combined_df = pd.DataFrame(columns=list(_all_df_cols))  # stub for resolve_column
    _viz_gaps: dict[str, list[str]] = {}
    for func_name, required in _VIZ_REQUIRED_COLUMNS.items():
        missing = [
            c for c in required
            if c not in _all_df_cols and resolve_column(_combined_df, c) is None
            ]
        if missing:
            _viz_gaps[func_name] = missing
    if _viz_gaps:
        _log_and_print(f"  ⚠️ Viz column coverage gaps after enrichment: {_viz_gaps}")
    else:
        _log_and_print("  ✓ All viz column requirements satisfied")

    _log_and_print("")

    # ========================================================================
    # 9. VISUALIZATIONS (consuming InferenceData)
    # ========================================================================
    _log_and_print("📈 Step 9: Generating visualizations...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        if not mc.empty:
            _write_viz(create_mc_return_distribution(mc), output_dir, "er_mc_distribution.html")
            _write_viz(create_sector_risk_reward_scatter(mc, identifier_coords=id_coords), output_dir,
                       "er_sector_risk_reward.html")
            _write_viz(create_var_analysis(mc), output_dir, "er_var_analysis.html")
            _write_viz(create_posterior_return_forest(mc, top_n=25), output_dir, "er_posterior_return_forest.html")

        if not kal.empty:
            _write_viz(create_kalman_vs_raw_scatter(kal), output_dir, "er_kalman_vs_raw.html")

        if not tri.empty:
            _write_viz(create_tri_model_agreement_histogram(tri), output_dir, "er_tri_model_agreement.html")
            _write_viz(create_sector_heatmap(tri, schema_metadata=schema_metadata), output_dir,
                       "er_sector_heatmap.html")

        if not strong.empty:
            _write_viz(create_strong_consensus_bar(strong), output_dir, "er_strong_consensus.html")
            tri_cols = {
                "ticker",
                "expected_upside_pct",
                "filtered_upside",
                "expected_return_prob_weighted",
                }
            if tri_cols.issubset(strong.columns):
                _write_viz(create_tri_model_posterior_comparison(strong, top_n=24), output_dir,
                           "er_tri_model_posterior.html")

        if not beat.empty and not pt.empty:
            _write_viz(create_beat_vs_achievement_scatter(beat, pt), output_dir, "er_beat_vs_achievement.html")

        if not beat.empty and "prob_beat_given_momentum" in beat.columns:
            _write_viz(create_beat_probability_posterior(beat, top_n=24), output_dir,
                       "er_beat_probability_posterior.html")

        if not beat.empty:
            _write_viz(create_earnings_probability_dashboard(beat), output_dir,
                       "er_earnings_probability_dashboard.html")

        # v3.2: Earnings quality visualizations (revision momentum, GAAP divergence, enhanced beat)
        if not beat.empty and "eps_revision_momentum" in beat.columns:
            _write_viz(create_revision_momentum_chart(beat, top_n=30), output_dir, "er_revision_momentum.html")

        if not beat.empty and "gaap_adj_eps_gap_pct" in beat.columns:
            _write_viz(create_gaap_divergence_plot(beat), output_dir, "er_gaap_divergence.html")

        if not beat.empty and "prob_beat_given_momentum" in beat.columns:
            _write_viz(create_enhanced_beat_prob_dash(beat), output_dir, "er_enhanced_beat_probability.html")

        if not df.empty:
            _write_viz(create_quality_risk_quadrant(df), output_dir, "er_quality_risk_quadrant.html")
            _write_viz(create_distress_early_warning_dashboard(df), output_dir, "er_distress_early_warning.html")

        # v3.2: Quality & risk deep-dive visualizations
        if not df.empty and "piotroski_f_score" in df.columns:
            _write_viz(create_piotroski_fscore_breakdown(df), output_dir, "er_piotroski_fscore.html")

        if not df.empty and "altman_z_score" in df.columns:
            _write_viz(create_altman_zscore_distribution(df), output_dir, "er_altman_zscore.html")

        if not df.empty:
            _write_viz(create_beneish_mscore_analysis(df), output_dir, "er_beneish_mscore.html")

        if not df.empty and "combined_distress_risk_score" in df.columns:
            _write_viz(create_risk_tier_sunburst(df), output_dir, "er_risk_tier_sunburst.html")

        if not anomaly_results.empty and "accounting_anomaly_score" in anomaly_results.columns:
            _write_viz(create_accounting_anomaly_dashboard(anomaly_results), output_dir,
                       "er_accounting_anomaly_dashboard.html")

        if not anomaly_results.empty and "anomaly_severity_score" in anomaly_results.columns:
            _write_viz(create_anomaly_severity_dashboard(anomaly_results), output_dir,
                       "er_anomaly_severity_dashboard.html")

        if not anomaly_results.empty and "anomaly_conditional_probability" in anomaly_results.columns:
            _write_viz(create_anomaly_conditional_probability_chart(anomaly_results), output_dir,
                       "er_anomaly_conditional_probability.html")

        # v3.3: MCMC-enhanced probability model visualizations
        if not anomaly_results.empty and "anomaly_posterior_mean" in anomaly_results.columns:
            _write_viz(create_mcmc_anomaly_posterior_chart(anomaly_results), output_dir,
                       "er_mcmc_anomaly_posterior.html")

        if not credit.empty and "mcmc_distress_probability" in credit.columns:
            _write_viz(create_mcmc_credit_risk_chart(credit), output_dir, "er_mcmc_credit_risk_posterior.html")

        if not div_safety.empty and "mcmc_cut_probability" in div_safety.columns:
            _write_viz(create_mcmc_dividend_cut_chart(div_safety), output_dir, "er_mcmc_dividend_cut_posterior.html")

        if not pt.empty and "mcmc_achievement_probability" in pt.columns:
            _write_viz(create_mcmc_price_target_chart(pt), output_dir, "er_mcmc_price_target_posterior.html")

        if not credit.empty and "ruin_probability" in credit.columns:
            _write_viz(create_ruin_probability_diagnostic(credit, top_n=20, identifier_coords=id_coords), output_dir,
                       "er_ruin_probability_diagnostic.html")

        if not summary.empty:
            tri_cols = {
                "ticker",
                "expected_upside_pct",
                "filtered_upside",
                "expected_return_prob_weighted",
                }
            if tri_cols.issubset(summary.columns):
                _write_viz(create_tri_model_posterior_comparison(summary, top_n=24), output_dir,
                           "er_expected_returns_summary_posterior.html")
                _write_viz(create_model_dispersion_dashboard(summary), output_dir, "er_model_dispersion_dashboard.html")

            if not mc.empty:
                _write_viz(create_return_distribution_fit_chart(mc), output_dir, "er_return_distribution_fit.html")

            sector_analytics = compute_sector_return_analytics(summary)
            if not sector_analytics.empty:
                _write_viz(create_sector_return_analytics_heatmap(sector_analytics), output_dir,
                           "er_sector_return_analytics.html")

        # v3.0: Screening summary chart
        if screens:
            _write_viz(create_screening_summary_chart(screens), output_dir, "er_screening_summary.html")

        # v3.1: Price target drift dashboard
        if not df.empty:
            _write_viz(create_price_target_drift_dashboard(df, mv_spec=mv_equities_spec), output_dir,
                       "er_price_target_drift.html")

        # ── Valuation Analysis (v3.1) ──
        # Use df_all (feature views) which contains computed valuation ratios
        # (p_e_ratio, p_b_ratio, etc.); df (mv_equities) only has raw suffixed cols.
        _viz_df = df_all if not df_all.empty else df
        if not _viz_df.empty:
            _write_viz(create_valuation_multiples_comparison(_viz_df), output_dir, "er_valuation_multiples.html")
            _write_viz(create_valuation_distribution_dashboard(_viz_df), output_dir, "er_valuation_distribution.html")
            _write_viz(create_relative_valuation_matrix(_viz_df), output_dir, "er_relative_valuation_matrix.html")
            _write_viz(create_valuation_vs_growth_quadrant(_viz_df), output_dir, "er_valuation_vs_growth.html")
            _write_viz(create_historical_valuation_percentile(_viz_df), output_dir,
                       "er_historical_valuation_percentile.html")

        # ── Earnings Quality (v3.1) ──
        if not _viz_df.empty:
            _write_viz(create_earnings_surprise_dashboard(_viz_df), output_dir, "er_earnings_surprise.html")
            _write_viz(create_eps_trajectory_analysis(_viz_df), output_dir, "er_eps_trajectory.html")
            _write_viz(create_earnings_quality_decomposition(_viz_df), output_dir,
                       "er_earnings_quality_decomposition.html")
            _write_viz(create_beat_rate_heatmap(_viz_df), output_dir, "er_beat_rate_heatmap.html")
            _write_viz(create_earnings_consistency_matrix(_viz_df), output_dir, "er_earnings_consistency_matrix.html")

        # ── Growth Analysis (v3.1) ──
        if not _viz_df.empty:
            _write_viz(create_growth_waterfall_chart(_viz_df), output_dir, "er_growth_waterfall.html")
            _write_viz(create_growth_consistency_matrix(_viz_df), output_dir, "er_growth_consistency_matrix.html")
            _write_viz(create_growth_vs_profitability_quadrant(_viz_df), output_dir, "er_growth_vs_profitability.html")
            _write_viz(create_growth_acceleration_chart(_viz_df), output_dir, "er_growth_acceleration.html")
            _write_viz(create_sustainable_growth_analysis(_viz_df), output_dir, "er_sustainable_growth.html")

        # ── Feature View Posterior Panel (v3.1) ──
        # create_feature_view_posterior_panel expects InferenceData/xr.Dataset,
        # not a raw DataFrame.  Build per-view InferenceData first.
        if not df_all.empty and view_specs and build_feature_view_inference_data is not None:
            for _vs_name, _vs in view_specs.items():
                try:
                    _fv_idata = build_feature_view_inference_data(_vs_name, df_all)
                    _write_viz(create_feature_view_posterior_panel(_fv_idata, view_spec=_vs), output_dir,
                               f"er_feature_view_posterior_{_vs_name}.html")
                except Exception as _fv_err:
                    logger.debug("Feature view posterior %s skipped: %s", _vs_name, _fv_err)

        # Bayesian category ridge for analyst sentiment features
        sentiment_features = [
            f
            for f in [
                "analyst_bullish_pct",
                "upside_potential",
                "eps_revision_momentum",
                "analyst_conviction",
                "pt_consensus_convergence",
                ]
            if f in df_all.columns
            ]
        if sentiment_features:
            results = bayesian_category_analysis(
                    df_all, "Analyst Sentiment", sentiment_features,
                    )
            _write_viz(create_bayesian_category_ridge(results, category_name="Analyst Sentiment"), output_dir,
                       "er_bayesian_sentiment_ridge.html")
            _write_viz(create_mcmc_category_posterior_chart(results, category_name="Analyst Sentiment"), output_dir,
                       "er_mcmc_category_sentiment_posterior.html")

        # v3.0: Bayesian category ridge for profitability features
        profitability_features = [
            f
            for f in ["roe", "roa", "roic", "gross_margin_pct", "operating_margin_pct"]
            if f in df_all.columns
            ]
        if profitability_features:
            results = bayesian_category_analysis(
                    df_all, "Profitability", profitability_features,
                    )
            _write_viz(create_bayesian_category_ridge(results, category_name="Profitability"), output_dir,
                       "er_bayesian_profitability_ridge.html")

        # ── ArviZ Diagnostic Visualizations (v3.2) ──
        if _ARVIZ_DIAG_AVAILABLE:
            _log_and_print("   📊 Generating ArviZ diagnostic visualizations...")

            # Step 5d: Screening posterior ridge
            if screens:
                try:
                    fig = create_screening_posterior_ridge(screens)
                    if fig:
                        _write_viz(fig, output_dir, "er_screening_posterior_ridge.png", fmt="png")
                except Exception as e:
                    logger.debug("Screening posterior ridge skipped: %s", e)

            # Step 5d: Productivity frontier posterior
            if not df_all.empty and "productivity_frontier_score" in df_all.columns:
                try:
                    fig = create_productivity_frontier_posterior(df_all)
                    if fig:
                        _write_viz(fig, output_dir, "er_productivity_frontier_posterior.png", fmt="png")
                except Exception as e:
                    logger.debug("Productivity frontier posterior skipped: %s", e)

            # Step 5e: Resampled posterior diagnostics
            if not resampled_posterior.empty:
                try:
                    resamp_outputs = create_resampled_posterior_diagnostics(resampled_posterior, output_dir)
                    for ro in resamp_outputs:
                        _log_and_print(f"   ✓ {Path(ro).name}")
                except Exception as e:
                    logger.debug("Resampled posterior ArviZ skipped: %s", e)

                try:
                    fig = create_resampled_sector_forest(resampled_posterior, df)
                    if fig:
                        _write_viz(fig, output_dir, "er_resampled_sector_forest.png", fmt="png")
                except Exception as e:
                    logger.debug("Resampled sector forest skipped: %s", e)

            # Step 6: Model alignment ArviZ panel
            if not summary.empty:
                try:
                    align_outputs = create_model_alignment_arviz_panel(summary, output_dir)
                    for ao in align_outputs:
                        _log_and_print(f"   ✓ {Path(ao).name}")
                except Exception as e:
                    logger.debug("Model alignment ArviZ skipped: %s", e)

                try:
                    fig = create_agreement_posterior_by_sector(summary)
                    if fig:
                        _write_viz(fig, output_dir, "er_agreement_by_sector.png", fmt="png")
                except Exception as e:
                    logger.debug("Agreement by sector skipped: %s", e)

            # Step 7: Hierarchical shrinkage diagnostic
            if not summary.empty:
                try:
                    fig = create_hierarchical_shrinkage_diagnostic(summary)
                    if fig:
                        _write_viz(fig, output_dir, "er_hierarchical_shrinkage.png", fmt="png")
                except Exception as e:
                    logger.debug("Hierarchical shrinkage plot skipped: %s", e)

                try:
                    fig = create_multi_level_mcmc_comparison(summary)
                    if fig:
                        _write_viz(fig, output_dir, "er_multi_level_mcmc.png", fmt="png")
                except Exception as e:
                    logger.debug("Multi-level MCMC comparison skipped: %s", e)

            # Step 7a: MCMC convergence panel
            if mcmc_result:
                try:
                    mcmc_outputs = create_mcmc_convergence_panel_arviz(mcmc_result, output_dir)
                    for mo in mcmc_outputs:
                        _log_and_print(f"   ✓ {Path(mo).name}")
                except Exception as e:
                    logger.debug("MCMC convergence panel skipped: %s", e)

            # Step 7b: Category posterior diagnostics
            if category_analytics:
                try:
                    cat_outputs = create_category_posterior_diagnostics(category_analytics, df_all, output_dir)
                    for co in cat_outputs:
                        _log_and_print(f"   ✓ {Path(co).name}")

                    cross_path = create_cross_category_summary(category_analytics, output_dir)
                    if cross_path:
                        _log_and_print(f"   ✓ {Path(cross_path).name}")
                except Exception as e:
                    logger.debug("Category ArviZ diagnostics skipped: %s", e)
        else:
            _log_and_print("   ⏭️ ArviZ diagnostics not available — skipping PNG diagnostic plots")

    except Exception as e:
        _log_and_print(f"   ⚠️ Visualization error: {e}")
        import traceback

        traceback.print_exc()

    _log_and_print(f"  ⏱ Step 9 completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # 10. EXPORT RESULTS
    # ========================================================================
    _log_and_print("💾 Step 10: Exporting results...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    exports = export_expected_returns_results(
            mc=mc,
            pt=pt,
            kal=kal,
            tri=tri,
            strong=strong,
            beat=beat,
            summary=summary,
            credit=credit,
            div_safety=div_safety,
            anomaly_results=anomaly_results,
            screens=screens,
            output_dir=str(output_dir),
            )
    for name, dest in exports.items():
        _log_and_print(f"   ✓ {name} → {dest}")

    # Export probability analytics results (beat + streak + credit + dividend)
    try:
        streak_analyzer = EPSStreakAnalyzer()
        streak_df = streak_analyzer.analyze_dataframe(df_all if not df_all.empty else df)
        prob_exports = export_probability_analytics_results(
                probability_df=beat,
                streak_df=streak_df,
                output_dir=output_dir,
                credit_risk_df=None,  # Already exported in export_expected_returns_results
                dividend_safety_df=None,  # Already exported in export_expected_returns_results
                )
        for pname, pdest in prob_exports.items():
            _log_and_print(f"   ✓ {pname} → {pdest}")
    except Exception as e:
        logger.warning("Probability analytics export failed: %s", e)

    # Aggregate probability results for category analytics
    if category_analytics:
        try:
            for cat_name, cat_result in category_analytics.items():
                cat_prob = cat_result.get("conditional_probabilities")
                if isinstance(cat_prob, pd.DataFrame) and not cat_prob.empty:
                    aggregated = aggregate_probability_results(cat_prob)
                    if not aggregated.empty:
                        cfg = ExportConfig(
                                table_name=f"prob_{cat_name.lower().replace(' ', '_')}",
                                )
                        export_to_db(aggregated, cfg)
                        logger.info(
                                "Aggregated probability export: %s (%d rows)",
                                cat_name,
                                len(aggregated),
                                )
        except Exception as e:
            logger.warning("Aggregated probability export failed: %s", e)

    _log_and_print(f"  ⏱ Step 10 completed in {time.perf_counter() - _step_start:.1f}s")
    _log_and_print("")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    _log_and_print("=" * 80)
    _log_and_print("\u2705 EXPECTED RETURNS ANALYTICS v3.5 COMPLETE")
    _log_and_print("=" * 80)
    _log_and_print("")
    _log_and_print("  Data sources:")
    _log_and_print(
            f"    mv_expected_returns:       {len(df):,} stocks × {len(df.columns)} features",
            )
    _log_and_print(
            f"    mv_all_stock_features:     {len(df_all):,} stocks × {len(df_all.columns)} features",
            )
    _log_and_print("")
    _log_and_print("  Models:")
    _log_and_print(f"    Monte Carlo simulations:   {len(mc):,}")
    _log_and_print(f"    Price target achievements: {len(pt):,}")
    _log_and_print(f"    Kalman-filtered targets:   {len(kal):,}")
    _log_and_print(f"    Earnings beat analyses:    {len(beat):,}")
    _log_and_print(f"    Credit risk analyses:      {len(credit):,}")
    _log_and_print(f"    Dividend safety analyses:  {len(div_safety):,}")
    _log_and_print(f"    Accounting anomaly analyses: {len(anomaly_results):,}")
    _log_and_print("")

    _log_and_print("  Alignment:")
    _log_and_print(f"    Tri-model aligned:         {len(tri):,}")
    _log_and_print(f"    Strong consensus picks:    {len(strong):,}")
    if not quad.empty:
        _log_and_print(
                f"    Quad-model full consensus: {(quad['quad_agreement'] == 4).sum()}",
                )
    if not summary.empty:
        full_consensus = (summary["agreement_score"] == 4).sum()
        _log_and_print(
                f"    Expected returns summary:  {len(summary):,} stocks, {full_consensus} full consensus",
                )
    if corr_info.get("correlation") is not None:
        _log_and_print(f"    MC ↔ Kalman correlation:   {corr_info['correlation']:.3f}")
    _log_and_print("")
    _log_and_print("  Screening:")
    for name, screen_df in screens.items():
        if not screen_df.empty:
            _log_and_print(f"    {name}: {len(screen_df):,} stocks")
    _log_and_print("")
    _log_and_print("  Probability Analytics:")
    _log_and_print(f"    Categories analyzed:       {len(category_analytics)}")
    _log_and_print("")
    _log_and_print(f"  Outputs: {output_dir}/")
    _log_and_print("")


if __name__ == "__main__":
    main()
