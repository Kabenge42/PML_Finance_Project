"""
Expected Returns Analytics Module (v3.6)

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
- Added multi-level hierarchical MCMC across 9 category columns (region, country, exchange, sector,
  industry, style, size, unit, trading_country)
- Integrated MCMC posterior visualizations for anomaly, credit, dividend, price target models

**Migration from v3.0:**
- Replaced `load_expected_returns_data` → `load_equities_data_from_db` (`data_utils`)
- Replaced `load_all_stock_features` → `load_all_feature_views` (`data_utils`)
- Replaced `load_analytics_table` → `load_feature_data_from_db` (`data_utils`)
- Replaced hardcoded column lists → dynamic `get_equities_schema`
- Retained hardcoded fallbacks for offline/no-DB environments

**ArviZ 1.0 Integration (v3.6):**
- Migrated from legacy ``arviz`` (``az.*``) API to ArviZ 1.0 (``arviz_plots``/``arviz_stats``/``arviz_base``)
- `InferenceData` schema for Monte Carlo, Earnings Beat, Credit Risk, Feature Views
- Per-model `EquityCoordinates` with ticker/sector/industry/country dimensions
- MCMC convergence diagnostics (R-hat, ESS, MCSE) via ``arviz_stats``
- Posterior ridge plots, forest plots, trace plots via ``arviz_plots``
- New: ECDF + reference quantile lines, quantile dot plots, PPC rootogram
- New: Unified convergence dashboard across all pipeline MCMC outputs
- Backward-compatible fallback to legacy ``arviz`` when ``arviz_plots`` unavailable

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
- **Screening:** Posterior ridge, productivity frontier, model alignment panel, PPC rootogram
- **ArviZ 1.0 Diagnostics:** ECDF with quantile references, quantile dot plots, PPC rootogram,
  unified convergence dashboard, category posterior forest (ArviZ), hierarchical dot comparison

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

import hashlib
import json
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from numpy import dtype, integer, ndarray
from pandas import DataFrame

try:
    import arviz_base as azb
    import arviz_plots as azp
    import arviz_stats as azs
except ImportError:
    azp = None  # type: ignore[assignment]
    azs = None  # type: ignore[assignment]
    azb = None  # type: ignore[assignment]

try:
    import arviz as az
except ImportError:
    az = None  # type: ignore[assignment]

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from requests.exceptions import RequestsDependencyWarning
from scipy import stats as sp_stats

from finance_ml.ml_workflow.v3.cache import (
    CategoryAnalyticsCacheKey,
    build_cache_path,
    dataframe_stable_checksum,
    load_json,
    save_json,
)
from finance_ml.ml_workflow.v3.utils import (
    as_float_series,
    run_parallel_or_sequential,
)

# --- Data utilities ---
from probabilistic_ml_model.data_utils import (
    ExportConfig,
    aggregate_probability_results,
    backfill_feature_columns,
    compute_metric_statistics,
    export_to_db,
    get_equities_schema,
    get_identifier_cols_set,
    get_view_category_mapping,
    load_all_feature_views,
    load_equities_data_from_db,
    load_feature_categories_from_db,
    load_feature_data_from_db,
    load_identifier_columns,
    reorder_with_identifiers,
    validate_feature_alignment,
)

# --- Schema-driven feature catalog ---
from probabilistic_ml_model.data_utils.feature_catalog import (
    FeatureViewCatalog,
    auto_enrich_for_model,
    get_feature_catalog,
)

# --- Optimised operations ---
# --- Caching utilities ---
from probabilistic_ml_model.optimized_ops import (
    dataframe_hash,
    get_optimization_status,
    vectorized_percentile_rank,
    vectorized_zscore,
)
from probabilistic_ml_model.pipeline_runners import (
    PipelineConfig,
    PipelineResult,
    PipelineRunner,
)

# --- Probability models ---
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

# --- Screening (quality filtering of results) ---
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

# --- Statistical analysis ---
from probabilistic_ml_model.statistical_functions.statistical_models import (
    analyze_employee_productivity_frontier,
    analyze_reporting_lag_sentiment,
    bayesian_category_analysis,
    bayesian_earnings_beat_model,
    calculate_conditional_probabilities,
    calculate_ruin_probability,
    fit_distributions_by_category,
    fit_gaussian_copula,
    kalman_filter_price_target,
    kalman_momentum_filter,
    mcmc_student_t,
    monte_carlo_price_target_simulation,
    parallel_mcmc_chains,
    resampled_posterior_returns,
    run_category_probability_analytics,
)

# --- Hierarchical MCMC (via pipeline_runners wrappers) ---
from probabilistic_ml_model.pipeline_runners import (
    hierarchical_mcmc_by_sector,
    hierarchical_mcmc_multi_level,
)

# --- InferenceData schema (ArviZ / xarray bridge) ---
try:
    from probabilistic_ml_model.data_utils.inference_schema import (
        ARVIZ_AVAILABLE,
        FEATURE_VIEW_REGISTRY,
        EquitiesMaterializedViewSpec,
        EquitiesSchemaMetadata,
        EquityCoordinates,
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
        load_feature_registry_metadata_from_db,
        load_feature_view_spec_from_db,
        load_identifier_coordinates_from_db,
        load_mv_equities_spec_from_db,
        summarize_inference_data,
    )
except (ImportError, AttributeError) as _inference_err:
    logging.getLogger(__name__).warning(
        "inference_schema import failed: %s",
        _inference_err,
        exc_info=True,
    )
    ARVIZ_AVAILABLE = False  # ← was incorrectly True
    EquityCoordinates = None  # type: ignore[assignment,misc]
    IdentifierCoordinates = None  # type: ignore[assignment,misc]
    EquitiesSchemaMetadata = None  # type: ignore[assignment,misc]
    FeatureRegistryMetadata = None  # type: ignore[assignment,misc]
    FeatureViewSpec = None  # type: ignore[assignment,misc]
    EquitiesMaterializedViewSpec = None  # type: ignore[assignment,misc]
    FEATURE_VIEW_REGISTRY = {}  # type: ignore[misc]
    build_accounting_anomaly_inference_data = None  # type: ignore[assignment]
    build_beat_probability_inference_data = None  # type: ignore[assignment]
    build_category_analysis_inference_data = None  # type: ignore[assignment]
    build_credit_risk_inference_data = None  # type: ignore[assignment]
    build_monte_carlo_inference_data = None  # type: ignore[assignment]
    build_resampled_technical_inference_data = None  # type: ignore[assignment]
    build_feature_view_inference_data = None  # type: ignore[assignment]
    load_identifier_coordinates_from_db = None  # type: ignore[assignment]
    load_equities_schema_metadata_from_db = None  # type: ignore[assignment]
    load_feature_registry_metadata_from_db = None  # type: ignore[assignment]
    load_feature_view_spec_from_db = None  # type: ignore[assignment]
    load_mv_equities_spec_from_db = None  # type: ignore[assignment]
    summarize_inference_data = None  # type: ignore[assignment]

# --- Probabilistic visualizations (ArviZ-backed) ---
# --- Other visualizations ---
from probabilistic_ml_model.visualizations import (
    MV_COLUMN_ALIASES,
    PLOTLY_TEMPLATE,
    resolve_column,
)
from probabilistic_ml_model.visualizations.earnings_quality import (
    create_beat_rate_heatmap,
    create_earnings_consistency_matrix,
    create_earnings_quality_decomposition,
    create_earnings_surprise_dashboard,
    create_eps_trajectory_analysis,
    create_gaap_divergence_plot,
    create_revision_momentum_chart,
)

# --- Earnings Quality charts ---
from probabilistic_ml_model.visualizations.earnings_quality import (
    create_enhanced_beat_probability_dashboard as create_enhanced_beat_prob_dash,
)

# --- Expected Returns Pipeline charts ---
from probabilistic_ml_model.visualizations.expected_returns_viz import (
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

# --- Growth Analysis charts ---
from probabilistic_ml_model.visualizations.growth_analysis import (
    create_growth_acceleration_chart,
    create_growth_consistency_matrix,
    create_growth_vs_profitability_quadrant,
    create_growth_waterfall_chart,
    create_sustainable_growth_analysis,
)

# --- Probabilistic & Bayesian Analysis (ArviZ-backed) ---
from probabilistic_ml_model.visualizations.probability_viz import (
    create_anomaly_conditional_probability_chart,
    create_bayesian_category_ridge,
    create_beat_probability_posterior,
    create_feature_view_posterior_panel,
    # MCMC-enhanced probability model visualizations (v3.3)
    create_mcmc_anomaly_posterior_chart,
    # ArviZ 1.0 category posterior forest (v3.6)
    create_mcmc_category_posterior_arviz,
    create_mcmc_category_posterior_chart,
    create_mcmc_credit_risk_chart,
    create_mcmc_dividend_cut_chart,
    create_mcmc_price_target_chart,
    create_mcse_convergence_panel,
    create_posterior_return_forest,
    create_ruin_probability_diagnostic,
    create_tri_model_posterior_comparison,
    create_tri_model_posterior_price_target_comparison,
)

# --- Quality & Risk charts ---
from probabilistic_ml_model.visualizations.quality_risk import (
    create_accounting_anomaly_dashboard,
    create_altman_zscore_distribution,
    create_anomaly_severity_dashboard,
    create_beneish_mscore_analysis,
    create_distress_early_warning_dashboard,
    create_piotroski_fscore_breakdown,
    create_quality_risk_quadrant,
    create_risk_tier_sunburst,
)

# --- Valuation Analysis charts ---
from probabilistic_ml_model.visualizations.valuation import (
    create_historical_valuation_percentile,
    create_relative_valuation_matrix,
    create_valuation_distribution_dashboard,
    create_valuation_multiples_comparison,
    create_valuation_vs_growth_quadrant,
)

# --- ArviZ diagnostic visualizations ---
try:
    from probabilistic_ml_model.visualizations.arviz_diagnostics import (
        ARVIZ_AVAILABLE as _ARVIZ_DIAG_AVAILABLE,
    )
    from probabilistic_ml_model.visualizations.arviz_diagnostics import (
        create_agreement_posterior_by_sector,
        create_category_posterior_diagnostics,
        create_cross_category_summary,
        create_cross_model_ecdf_with_references,
        create_hierarchical_dot_comparison,
        create_hierarchical_shrinkage_diagnostic,
        create_mcmc_convergence_panel_arviz,
        create_model_alignment_arviz_panel,
        create_multi_level_mcmc_comparison,
        create_productivity_frontier_posterior,
        create_resampled_posterior_diagnostics,
        create_resampled_sector_forest,
        create_screening_posterior_ridge,
        # ArviZ 1.0 new visualization types (v3.6)
        create_screening_ppc_rootogram,
    )
except (ImportError, AttributeError) as _arviz_diag_err:
    logging.getLogger(__name__).warning(
        "arviz_diagnostics import failed: %s",
        _arviz_diag_err,
        exc_info=True,
    )
    _ARVIZ_DIAG_AVAILABLE = False

# --- Unified convergence diagnostics (ArviZ 1.0) ---
try:
    from probabilistic_ml_model.visualizations.convergence_diagnostics import (
        create_unified_convergence_dashboard,
    )
except ImportError, AttributeError:
    create_unified_convergence_dashboard = None  # type: ignore[assignment]

from probabilistic_ml_model.logging_config import configure_logging
from probabilistic_ml_model.utils import safe_divide

px.defaults.template = PLOTLY_TEMPLATE

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# — Defined in probabilistic_ml_model.pipeline_runners.PipelineConfig
# — Imported at module level (line 169)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Result Container (Task 1.2)
# — Defined in probabilistic_ml_model.pipeline_runners.PipelineResult
# — Imported at module level (line 169)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# MCMC Result Caching (Task 2.4)
# ═══════════════════════════════════════════════════════════════════════════════


def _get_cache_path(cache_dir: str, key: str, params: dict) -> Path:
    """Build a deterministic cache file path from data hash + parameters."""
    param_str = json.dumps(params, sort_keys=True, default=str)
    combined = f"{key}_{param_str}"
    cache_hash = hashlib.md5(combined.encode()).hexdigest()
    return Path(cache_dir) / f"{cache_hash}.pkl"


def _load_cached_result(cache_path: Path, ttl_hours: float = 24.0) -> Any | None:
    """Load a cached result if it exists and is recent (< ttl_hours)."""
    import pickle

    if not cache_path.exists():
        return None
    try:
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours > ttl_hours:
            logger.debug(
                "Cache expired (%.1fh > %.1fh): %s",
                age_hours,
                ttl_hours,
                cache_path.name,
            )
            return None

        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        logger.debug("Cache hit (%.1fh old): %s", age_hours, cache_path.name)
        return result
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as e:
        logger.debug("Cache load failed: %s", e)
        return None


def _save_cached_result(cache_path: Path, result: Any) -> None:
    """Persist a result to the cache directory."""
    import pickle

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    except (OSError, pickle.PicklingError, TypeError) as e:
        logger.debug("Cache write failed: %s", e)


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
    dest.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        fig.write_html(dest)
    else:
        fig.savefig(dest, dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as _plt

        _plt.close(fig)
    _log_and_print(f"   ✓ {filename}")


def _enrich_dataframe(
    target: pd.DataFrame,
    source: pd.DataFrame,
    needed_cols: set[str],
    label: str,
) -> pd.DataFrame:
    """
    Merge missing columns from *source* into *target* on ``isin``.

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
    target = _ensure_isin_column(target)
    source = _ensure_isin_column(source)

    if (
        target.empty
        or source.empty
        or "isin" not in target.columns
        or "isin" not in source.columns
    ):
        return target

    missing = [
        c for c in needed_cols if c not in target.columns and c in source.columns
    ]
    if not missing:
        return target

    src_subset = source[["isin"] + missing].drop_duplicates(subset="isin")
    target = target.merge(src_subset, on="isin", how="left")
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
            f"max: {score_stats['max']:.1f}"
        )

    # ── Anomaly feature count ──
    if "anomaly_feature_count" in anomaly_results.columns:
        flagged = (anomaly_results["anomaly_feature_count"] > 0).sum()
        multi_flagged = (anomaly_results["anomaly_feature_count"] >= 15).sum()
        _log_and_print(
            f"  Stocks with >0 flagged features: {flagged:,}, ≥15 flags: {multi_flagged:,}"
        )

    # ── Per-feature flag summary ──
    flag_cols = [c for c in anomaly_results.columns if c.endswith("_anomaly_flag")]
    if flag_cols:
        _log_and_print("  Per-feature anomaly flags:")
        for fc in sorted(flag_cols):
            feat_name = fc.replace("_anomaly_flag", "")
            n_flagged = (
                anomaly_results[fc].sum() if fc in anomaly_results.columns else 0
            )
            if n_flagged > 0:
                _log_and_print(f"    {feat_name}: {int(n_flagged):,} stocks flagged")

    # ── Mahalanobis distance ──
    if "mahalanobis_distance" in anomaly_results.columns:
        mahal = anomaly_results["mahalanobis_distance"].dropna()
        if len(mahal) > 0:
            _log_and_print(
                f"  Mahalanobis distance — computed: {len(mahal):,}, "
                f"mean: {mahal.mean():.2f}, p95: {mahal.quantile(0.95):.2f}"
            )

    # ── Sector-relative anomaly ──
    if "sector_relative_anomaly" in anomaly_results.columns:
        sra = anomaly_results["sector_relative_anomaly"].dropna()
        sector_outliers = (sra.abs() > 2.0).sum()
        _log_and_print(f"  Sector-relative outliers (|z| > 2): {sector_outliers:,}")

    # ── Severity score & conditional probability ──
    if "anomaly_severity_score" in anomaly_results.columns:
        sev = anomaly_results["anomaly_severity_score"].dropna()
        if len(sev) > 0:
            _log_and_print(
                f"  Severity score — mean: {sev.mean():.2f}, "
                f"median: {sev.median():.2f}, max: {sev.max():.2f}"
            )

    if "anomaly_conditional_probability" in anomaly_results.columns:
        cond_p = anomaly_results["anomaly_conditional_probability"].dropna()
        if len(cond_p) > 0:
            _log_and_print(
                f"  Conditional P(anomaly) — mean: {cond_p.mean():.3f}, "
                f"median: {cond_p.median():.3f}, max: {cond_p.max():.3f}"
            )

    if "multi_flag_alert" in anomaly_results.columns:
        n_alerts = anomaly_results["multi_flag_alert"].sum()
        _log_and_print(f"  Multi-flag alerts: {int(n_alerts):,}")

    # ── Quality frequency & repeat offender (v3.5) ──
    if "quality_frequency_score" in anomaly_results.columns:
        qfs = anomaly_results["quality_frequency_score"].dropna()
        if len(qfs) > 0:
            _log_and_print(
                f"  Quality frequency score — mean: {qfs.mean():.2f}, "
                f"max: {qfs.max():.0f}"
            )
    if "repeat_offender_flag" in anomaly_results.columns:
        n_repeat = int(anomaly_results["repeat_offender_flag"].sum())
        _log_and_print(f"  Repeat offenders (quality_frequency ≥ 10): {n_repeat:,}")

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
            _log_and_print(f"  Benford's Law chi² p-value: {p_val:.4f} ({verdict})")

    # ── Distribution fit summary ──
    dist_cols = [c for c in anomaly_results.columns if c.endswith("_dist_name")]
    if dist_cols:
        _log_and_print("  Best-fit distributions per feature:")
        for dc in sorted(dist_cols):
            feat_name = dc.replace("_dist_name", "")
            dist_name = (
                anomaly_results[dc].mode().iloc[0]
                if len(anomaly_results[dc].dropna()) > 0
                else "n/a"
            )
            pval_col = f"{feat_name}_dist_pvalue"
            pval = (
                anomaly_results[pval_col].mean()
                if pval_col in anomaly_results.columns
                else float("nan")
            )
            _log_and_print(f"    {feat_name}: {dist_name} (mean KS p={pval:.3f})")


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

    # Ensure cache is a dict to satisfy type checker and prevent runtime errors
    cache = _feature_categories_cache if _feature_categories_cache is not None else {}

    if _feature_categories_cache is not None:
        logger.info(
            "Loaded %d feature categories (%d total features)",
            len(cache),
            sum(len(v) for v in cache.values()),
        )
    return cache


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

    @property
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
    categories: dict[str, list[str]], df_columns: set[str]
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
            role = str(meta.get("role", "unknown"))
            role_cols.setdefault(role, []).append(alias)

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
            c for c, m in schema.items() if m["role"] == "historical_price"
        )
        historical_targets = sorted(
            c for c, m in schema.items() if m["role"] == "historical_price_target"
        )
        historical_targets_high = sorted(
            c for c, m in schema.items() if m["role"] == "historical_price_target_high"
        )
        historical_targets_low = sorted(
            c for c, m in schema.items() if m["role"] == "historical_price_target_low"
        )
        historical_targets_median = sorted(
            c
            for c, m in schema.items()
            if m["role"] == "historical_price_target_median"
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
            "compute_model_detailed_statistics: %s — empty DataFrame", model_name
        )
        return {}

    results = {}
    for col in key_columns:
        if col not in df.columns:
            continue

        base_stats = compute_metric_statistics(df[col])
        if base_stats is None:
            continue

        series = as_float_series(df[col])

        shape_stats = {}
        if len(series) > 3:
            # v3.5: Clip extreme values to prevent overflow in higher-moment calculations (Issue 12)
            # 1e12 is safe for 4th moments (kurtosis) in float64 while preserving large-cap values.
            safe_series = series.clip(-1e12, 1e12)
            shape_stats["skewness"] = float(safe_series.skew())
            shape_stats["kurtosis"] = float(safe_series.kurtosis())
            shape_stats["iqr"] = float(base_stats["q75"] - base_stats["q25"])
            shape_stats["coefficient_of_variation"] = (
                float(series.std() / series.mean()) if series.mean() != 0 else None
            )
            mean, std = series.mean(), series.std()
            if std > 0:
                shape_stats["pct_beyond_2std"] = float(
                    ((series < mean - 2 * std) | (series > mean + 2 * std)).sum()
                    / len(series)
                    * 100
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
    top_n_sectors: int = 50,
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
            f"Median: {g['median']:.2f}  |  Std: {g['std']:.2f}"
        )
        print(
            f"        Min: {g['min']:.2f}  |  Max: {g['max']:.2f}  |  "
            f"IQR: [{g['q25']:.2f}, {g['q75']:.2f}]"
        )
        print(
            f"        Positive: {g['positive_pct']:.1f}%  |  "
            f"Missing: {g['missing_pct']:.1f}%"
        )
        if s:
            print(
                f"        Skew: {s.get('skewness') or 0:.3f}  |  "
                f"Kurtosis: {s.get('kurtosis') or 0:.3f}  |  "
                f"CV: {s.get('coefficient_of_variation') or 0:.3f}"
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
                    f"median={sinfo['median']:.2f}, n={sinfo['count']}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading — Equities & Feature Views Backend (v3.2)
# ═══════════════════════════════════════════════════════════════════════════════


def load_expected_returns_data(
    db_url: Optional[str] = None,
    schema: str = "public",
) -> tuple[pd.DataFrame, "Optional[IdentifierCoordinates]"]:
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
    tuple[pd.DataFrame, Optional[IdentifierCoordinates]]
        Feature DataFrame and identifier coordinates (None on failure).
    """
    try:
        df = load_equities_data_from_db(db_url=db_url, schema=schema)
    except (ImportError, ValueError) as e:
        logger.warning("Failed to load equities data: %s", e)
        return pd.DataFrame(), None

    id_coords = None
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        df = _apply_backfill_and_kalman(df)
        df = df.fillna(0)

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
            feature_categories, set(df.columns)
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
) -> tuple[DataFrame, dict[Any, Any]] | tuple[DataFrame, dict[str, None]]:
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
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        df = _apply_backfill_and_kalman(df)
        df = df.fillna(0)

        # Build view specs from registry
        if FEATURE_VIEW_REGISTRY is not None:
            for view_name, category in FEATURE_VIEW_REGISTRY.items():
                try:
                    spec = load_feature_view_spec_from_db(
                        view_name, db_url=db_url, schema=schema
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

    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        df = _apply_backfill_and_kalman(df)
        df = df.fillna(0)
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
    ... (docstring) ...
    """
    # v3.9: Ensure unique isins to prevent duplicate simulations
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df
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

    # Winsorize implied_return_mc at 1st/99th percentile to match Kalman treatment
    # and prevent extreme outliers (max observed: 897%) from dominating ensemble means
    if not mc.empty and "implied_return_mc" in mc.columns:
        lower, upper = mc["implied_return_mc"].quantile([0.01, 0.99])
        mc["implied_return_mc"] = mc["implied_return_mc"].clip(lower, upper)

    # Diagnostic: log coverage gap between input and output
    input_count = len(sim_df)
    output_count = len(mc)
    if output_count < input_count * 0.90:
        logger.warning(
            "MC coverage gap: %d/%d stocks (%.1f%%) processed — "
            "%d stocks likely missing required price target columns",
            output_count,
            input_count,
            output_count / input_count * 100,
            input_count - output_count,
        )

    logger.info("Monte Carlo simulation: %d stocks processed", len(mc))
    return mc


def run_price_target_achievement(
    df: pd.DataFrame,
    use_historical_targets: bool = True,
    feature_df: pd.DataFrame | None = None,
    catalog: FeatureViewCatalog | None = None,
) -> pd.DataFrame:
    """
    Estimate probability of reaching consensus price targets.
    ... (docstring) ...
    """
    # v3.9: Strict input deduplication
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df
    cat = catalog or get_feature_catalog()
    pt_df = auto_enrich_for_model(
        df.copy(), feature_df, "price_target_achievement", cat
    )

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
    ... (docstring) ...
    """
    # v3.9: Strict input deduplication
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df
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

    # v3.5: Winsorize implied_return_kalman at 1st/99th percentile to prevent
    # extreme outliers inflating the mean (observed 116.6% vs MC 27.3%).
    if not kal.empty and "implied_return_kalman" in kal.columns:
        lower, upper = kal["implied_return_kalman"].quantile([0.01, 0.99])
        kal["implied_return_kalman"] = kal["implied_return_kalman"].clip(lower, upper)

    logger.info("Kalman filter: %d stocks processed", len(kal))
    return kal


# ═══════════════════════════════════════════════════════════════════════════════
# Historical Price Target Drift Enrichment
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_pct_change(
    current: pd.Series,
    previous: pd.Series,
) -> pd.Series:
    """Compute ((current - previous) / |previous|) * 100, replacing ±inf with NaN."""
    curr = pd.to_numeric(current, errors="coerce")
    prev = pd.to_numeric(previous, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = prev.abs()
        result = ((curr - prev) / denom) * 100.0
    return pd.Series(result, index=curr.index, dtype=float).replace(
        [np.inf, -np.inf], np.nan
    )


def _compute_spread(
    high: pd.Series, low: pd.Series
) -> ndarray[tuple[Any, ...], dtype[integer[Any]]]:
    """Compute numeric spread ``high − low``."""
    hi = pd.to_numeric(high, errors="coerce")
    lo = pd.to_numeric(low, errors="coerce")
    spreads = hi - lo
    return spreads


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
            "No historical price/target columns found — skipping drift enrichment"
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
        "Historical target drift enrichment: %d derived columns added", n_derived
    )
    # De-fragment after repeated column inserts
    return df.copy()


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
    catalog: FeatureViewCatalog | None = None,
) -> pd.DataFrame:
    """
    Run enhanced three-layer Bayesian earnings beat probability model.

    Uses ``EarningsBeatProbabilityModel.analyze_dataframe_enhanced()``
    which fuses historical EPS, revision momentum, and GAAP quality layers.
    Enriches results with EPS streak analysis via ``EPSStreakAnalyzer``,
    resampled technical priors via ``ResampledBeatProbabilityModel``,
    and classical Bayesian beat estimates via ``bayesian_earnings_beat_model``.

    Column requirements resolved from ``FeatureViewCatalog``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with earnings columns.
    feature_df : pd.DataFrame or None, optional
        Full feature DataFrame for merging missing quality columns.
    catalog : FeatureViewCatalog or None, optional
        Pre-loaded feature catalog. If None, uses the global singleton.
    """
    # v3.9: Ensure input uniqueness to prevent Cartesian explosion
    df = _ensure_isin_column(df)
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df
    cat = catalog or get_feature_catalog()
    beat_df = auto_enrich_for_model(df.copy(), feature_df, "earnings_beat", cat)

    # v3.5: momentum-adjusted priors and quality discounting
    model = EarningsBeatProbabilityModel(
        use_quality_adjustment=True,
        use_momentum_prior=True,
        momentum_prior_strength=0.1,
    )
    sector_col = "sector" if "sector" in beat_df.columns else "industry"
    beat = model.analyze_dataframe_enhanced(beat_df, sector_col=sector_col)
    beat = _ensure_isin_column(beat)
    logger.info("Earnings beat analysis: %d stocks processed", len(beat))

    # --- EPS streak analysis (Markov-chain continuation probabilities) ---
    try:
        streak_analyzer = EPSStreakAnalyzer()
        streak_df = streak_analyzer.analyze_dataframe(df)
        streak_df = _ensure_isin_column(streak_df)
        if not streak_df.empty and "isin" in streak_df.columns:
            # v3.9: Deduplicate side-car before merge
            streak_side = streak_df.drop_duplicates(subset="isin")
            streak_cols = [
                c for c in streak_side.columns if c != "isin" and c not in beat.columns
            ]
            if streak_cols:
                beat = beat.merge(
                    streak_side[["isin"] + streak_cols],
                    on="isin",
                    how="left",
                )
                logger.info("EPS streak enrichment: %d columns added", len(streak_cols))
    except Exception as e:
        logger.warning("EPS streak analysis failed: %s", e)

    # --- Resampled technical priors ---
    try:
        resampled_model = ResampledBeatProbabilityModel(base_model=model)
        resampled_df = resampled_model.analyze_dataframe(df)
        resampled_df = _ensure_isin_column(resampled_df)
        if not resampled_df.empty and "isin" in resampled_df.columns:
            # v3.9: Deduplicate side-car before merge
            resamp_side = resampled_df.drop_duplicates(subset="isin")
            resamp_cols = [
                c for c in resamp_side.columns if c != "isin" and c not in beat.columns
            ]
            if resamp_cols:
                beat = beat.merge(
                    resamp_side[["isin"] + resamp_cols],
                    on="isin",
                    how="left",
                )
                logger.info(
                    "Resampled beat enrichment: %d columns added", len(resamp_cols)
                )
    except Exception as e:
        logger.warning("Resampled beat probability failed: %s", e)

    # --- Classical Bayesian earnings beat model ---
    try:
        bayesian_beat = bayesian_earnings_beat_model(df)
        bayesian_beat = _ensure_isin_column(bayesian_beat)
        if not bayesian_beat.empty and "isin" in bayesian_beat.columns:
            # v3.9: Deduplicate side-car before merge
            bay_side = bayesian_beat.drop_duplicates(subset="isin")
            bay_cols = [
                c for c in bay_side.columns if c != "isin" and c not in beat.columns
            ]
            if bay_cols:
                beat = beat.merge(
                    bay_side[["isin"] + bay_cols],
                    on="isin",
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
    catalog: FeatureViewCatalog | None = None,
) -> pd.DataFrame:
    """
    Run credit risk and ruin probability analysis.

    Uses ``CreditRiskProbabilityModel`` for Bayesian distress estimation
    and ``calculate_ruin_probability`` for analytical ruin estimates
    (modified Gambler's Ruin framework).

    Column requirements resolved from ``FeatureViewCatalog`` instead of
    the former inline ``_CREDIT_RISK_COLS`` list.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with financial health columns.
    feature_df : pd.DataFrame or None, optional
        Full feature DataFrame for merging missing columns.
    n_mcmc_samples : int, default 5000
        Number of MCMC posterior samples for Bayesian distress estimation.
    burn_in : int, default 1000
        Number of initial MCMC samples to discard as burn-in.
    catalog : FeatureViewCatalog or None, optional
        Pre-loaded feature catalog. If None, uses the global singleton.
    """
    # v3.9: Strict input deduplication
    df = _ensure_isin_column(df)
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df
    cat = catalog or get_feature_catalog()
    credit_df = auto_enrich_for_model(df.copy(), feature_df, "credit_risk", cat)

    credit_model = CreditRiskProbabilityModel(n_mcmc_samples=n_mcmc_samples, burn_in=burn_in)
    credit = credit_model.analyze_dataframe(credit_df)
    credit = _ensure_isin_column(credit)

    # Recalibrate degenerate distress_risk_score: if >40% of stocks are at max,
    # apply percentile-based rescaling for better discrimination
    if not credit.empty and "distress_risk_score" in credit.columns:
        at_max = (credit["distress_risk_score"] >= credit["distress_risk_score"].max()).mean()
        if at_max > 0.40:
            logger.warning(
                "distress_risk_score degenerate: %.0f%% at maximum — applying percentile rescaling",
                at_max * 100,
            )
            credit["distress_risk_score"] = credit["distress_risk_score"].rank(pct=True) * 100

    # --- Hierarchical sector-level MCMC enrichment ---
    try:
        if "altman_z_score" in credit_df.columns:
            z_data = credit_df["altman_z_score"].dropna()
            if len(z_data) > 50:
                sector_mcmc = hierarchical_mcmc_by_sector(credit_df, "altman_z_score")
                # Unwrap ArviZ-wrapped result
                if "sectors" in sector_mcmc and isinstance(
                    sector_mcmc["sectors"], dict
                ):
                    sector_mcmc = sector_mcmc["sectors"]
                sector_mean_map = {
                    s: v.get("posterior_mean", np.nan)
                    for s, v in sector_mcmc.items()
                    if isinstance(v, dict)
                }
                sector_col = "industry" if "industry" in credit.columns else "sector"
                if sector_col in credit.columns:
                    credit["sector_z_posterior_mean"] = credit[sector_col].map(
                        sector_mean_map
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
        ruin = _ensure_isin_column(ruin)
        if not ruin.empty and not credit.empty and "isin" in ruin.columns:
            # v3.9: Deduplicate side-car
            ruin_side = ruin.drop_duplicates(subset="isin")
            ruin_cols = [
                c for c in ruin_side.columns if c != "isin" and c not in credit.columns
            ]
            if ruin_cols:
                credit = credit.merge(
                    ruin_side[["isin"] + ruin_cols],
                    on="isin",
                    how="left",
                )
                logger.info(
                    "Ruin probability enrichment: %d columns added", len(ruin_cols)
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
    catalog: FeatureViewCatalog | None = None,
) -> pd.DataFrame:
    """
    Run dividend cut probability analysis.

    Uses ``DividendCutProbabilityModel`` to estimate probability of
    dividend reduction based on FCF coverage, payout ratio, streak,
    and leverage/liquidity signals.

    Column requirements resolved from ``FeatureViewCatalog``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with dividend columns.
    feature_df : pd.DataFrame or None, optional
        Full feature DataFrame for merging missing leverage/liquidity columns.
    n_mcmc_samples : int, default 5000
        Number of MCMC posterior samples for dividend cut estimation.
    burn_in : int, default 1000
        Number of initial MCMC samples to discard as burn-in.
    catalog : FeatureViewCatalog or None, optional
        Pre-loaded feature catalog. If None, uses the global singleton.

    Returns
    -------
    pd.DataFrame
        Dividend safety results with ``dividend_cut_probability``,
        ``safety_score``, ``risk_category``.
    """
    # v3.9: Strict input deduplication
    df = _ensure_isin_column(df)
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df
    cat = catalog or get_feature_catalog()
    div_df = auto_enrich_for_model(df.copy(), feature_df, "dividend_safety", cat)

    model = DividendCutProbabilityModel(n_mcmc_samples=n_mcmc_samples, burn_in=burn_in)
    div_safety = model.analyze_dataframe(div_df)
    div_safety = _ensure_isin_column(div_safety)
    logger.info("Dividend safety analysis: %d stocks processed", len(div_safety))
    return div_safety


def run_accounting_anomaly_analysis(
    df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    *,
    severity_anomaly_weight: float = 0.75,
    severity_feature_weight: float = 0.25,
    multi_flag_threshold: int = 15,
    anomaly_z_threshold: float | None = None,
    tier_bins: list[float] | None = None,
    tier_labels: list[str] | None = None,
    n_mcmc_samples: int = 5000,
    burn_in: int = 1000,
    catalog: FeatureViewCatalog | None = None,
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

    Column requirements resolved from ``FeatureViewCatalog``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with quality/risk and earnings columns.
    feature_df : pd.DataFrame or None, optional
        Full feature DataFrame for merging missing accounting columns.
    severity_anomaly_weight : float, default 0.75
        Weight for anomaly_score in severity computation.
    severity_feature_weight : float, default 0.25
        Weight for feature_count in severity computation.
    multi_flag_threshold : int, default 10
        Minimum flagged features to trigger multi_flag_alert.
    anomaly_z_threshold : float or None
        Robust z-score threshold for flagging anomalies. None = auto-derived.
    tier_bins : list[float] or None
        Bin edges for anomaly tier classification. None = auto-derived.
    tier_labels : list[str] or None
        Labels for the tier bins. None = ['Clean', 'Watch', 'Flag', 'Alert'].
    n_mcmc_samples : int, default 5000
        Number of MCMC posterior samples for anomaly probability estimation.
    burn_in : int, default 1000
        Number of initial MCMC samples to discard as burn-in.
    catalog : FeatureViewCatalog or None, optional
        Pre-loaded feature catalog. If None, uses the global singleton.

    Returns
    -------
    pd.DataFrame
        DataFrame with anomaly scores, tiers, per-feature flags,
        Mahalanobis distance, Benford's Law test, sector-relative scoring,
        severity scores, risk ranks, and conditional anomaly probabilities.
    """
    # v3.9: Strict input deduplication
    df = _ensure_isin_column(df)
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df
    cat = catalog or get_feature_catalog()
    anomaly_df = auto_enrich_for_model(df.copy(), feature_df, "accounting_anomaly", cat)

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
    result = _ensure_isin_column(result)

    # --- Student-t MCMC for anomaly score posterior ---
    try:
        if "accounting_anomaly_score" in result.columns:
            s = result["accounting_anomaly_score"].dropna()
            anomaly_scores = np.asarray(s, dtype=float)
            if anomaly_scores.size > 30:
                mu_samples, df_samples = mcmc_student_t(anomaly_scores)
                result["anomaly_posterior_location"] = float(mu_samples.mean())
                logger.info(
                    "MCMC anomaly posterior: location=%.3f",
                    mu_samples.mean(),
                )
    except Exception as e:
        logger.warning("MCMC anomaly posterior failed: %s", e)

    logger.info("Accounting anomaly analysis: %d stocks processed", len(result))
    return result


def _analyze_single_category(
    df: pd.DataFrame,
    cat_name: str,
    available: list[str],
    use_mcmc: bool,
    n_mcmc_samples: int,
    burn_in: int,
) -> tuple[str, dict]:
    """
    Analyze a single feature category (designed for parallel execution).

    Task 2.1: Extracted from the loop body of ``run_category_probability_analysis``
    so it can be dispatched via ``joblib.Parallel``.
    """
    cat_results = run_category_probability_analytics(
        df,
        cat_name,
        available,
        n_simulations=5_000,
    )

    # --- CategoryProbabilityAnalyzer: Bayesian view-level analysis ---
    try:
        analyzer = CategoryProbabilityAnalyzer(
            category_name=cat_name,
            n_mcmc_samples=n_mcmc_samples,
            burn_in=burn_in,
            use_mcmc=use_mcmc,
        )
        view_result = analyzer.analyze_view(df, feature_cols=available)
        if view_result is not None:
            cat_results["category_probability_analysis"] = view_result
    except Exception as e:
        logger.debug("CategoryProbabilityAnalyzer skipped for %s: %s", cat_name, e)

    # --- Distribution fitting per category ---
    try:
        dist_results = fit_distributions_by_category(df, cat_name, available)
        if dist_results:
            cat_results["distribution_fits"] = dist_results
    except Exception as e:
        logger.debug("Distribution fitting skipped for %s: %s", cat_name, e)

    # --- Conditional probability analysis ---
    try:
        cond_probs = calculate_conditional_probabilities(df, {cat_name: available})
        if cond_probs is not None and not (
            isinstance(cond_probs, pd.DataFrame) and cond_probs.empty
        ):
            cat_results["conditional_probabilities"] = cond_probs
    except Exception as e:
        logger.debug("Conditional probabilities skipped for %s: %s", cat_name, e)

    return cat_name, cat_results


def run_category_probability_analysis(
    df: pd.DataFrame,
    categories: Optional[dict[str, list[str]]] = None,
    *,
    use_mcmc: bool = True,
    n_mcmc_samples: int = 5000,
    burn_in: int = 1000,
    n_jobs: int = -1,
    max_features_per_category: int = 25,
    cache_dir: str = ".cache",
    enable_caching: bool = True,
    cache_ttl_hours: float = 24.0,
) -> dict[str, dict]:
    """
    Run per-category Bayesian probability analytics.

    Computes Bayesian posterior estimation, distribution fitting,
    and conditional probability analysis for each feature category.

    v3.6 enhancements:
    - Task 2.1: Parallelized via ``joblib`` when ``n_jobs != 1``.
    - Task 2.2: Feature-level sampling budget via ``max_features_per_category``.
    - Task 2.4: Result caching keyed by data hash + parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    categories : dict, optional
        Feature categories to analyze. Defaults to FEATURE_CATEGORIES.
    n_jobs : int
        Number of parallel jobs for category-level MCMC (Task 2.1).
        1 = sequential, -1 = all cores.
    max_features_per_category : int
        Maximum features to analyze per category (Task 2.2).
        0 = no limit.
    cache_dir : str
        Directory for MCMC result caching (Task 2.4).
    enable_caching : bool
        Whether to use file-based caching for MCMC results.

    Returns
    -------
    dict[str, dict]
        Per-category analytics results.
        :param cache_ttl_hours:
        :param enable_caching:
        :param cache_dir:
        :param max_features_per_category:
        :param n_jobs:
        :param burn_in:
        :param n_mcmc_samples:
        :param categories:
        :param df:
        :param use_mcmc:
    """
    # v3.9: Ensure unique isins for category analytics
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df
    cats = categories or FEATURE_CATEGORIES
    results = {}

    # Guard against zero/negative MCMC parameters to prevent division-by-zero
    if n_mcmc_samples < 1:
        logger.warning(
            "n_mcmc_samples=%d is invalid, defaulting to 5000", n_mcmc_samples
        )
        n_mcmc_samples = 5000
    if burn_in < 0:
        logger.warning("burn_in=%d is invalid, defaulting to 1000", burn_in)
        burn_in = 1000
    if burn_in >= n_mcmc_samples:
        logger.warning(
            "burn_in (%d) >= n_mcmc_samples (%d), adjusting burn_in to %d",
            burn_in,
            n_mcmc_samples,
            n_mcmc_samples // 5,
        )
        burn_in = n_mcmc_samples // 5

    # --- Task 2.4: Check cache (stable key) ---
    cache_path = None
    if enable_caching and cache_dir:
        _candidate_id_cols = ["isin", "ticker", "name"]
        _id_cols = [c for c in _candidate_id_cols if c in df.columns]
        checksum = dataframe_stable_checksum(df, id_cols=_id_cols)  # stable
        key = CategoryAnalyticsCacheKey(
            data_checksum=checksum,
            n_categories=len(cats) if cats is not None else 0,
            use_mcmc=use_mcmc,
            n_mcmc_samples=n_mcmc_samples,
            burn_in=burn_in,
            max_features_per_category=max_features_per_category,
        )
        cache_path = build_cache_path(cache_dir, key.to_filename(), subdir=key.subdir)
        cached = load_json(cache_path, ttl_hours=cache_ttl_hours)
        if isinstance(cached, dict) and cached:
            logger.info("Category analytics loaded from cache (%s)", cache_path.name)
            return cached

    # Pre-filter categories to those with sufficient numeric features
    category_tasks: list[tuple[str, list[str]]] = []
    for cat_name, features in cats.items():
        available = [f for f in features if f in df.columns]
        available = [f for f in available if pd.api.types.is_numeric_dtype(df[f])]
        if len(available) < 2:
            continue
        # Task 2.2: Limit features per category to control sampling budget
        if 0 < max_features_per_category < len(available):
            # Keep the features with highest variance (most informative)
            variances = df[available].var().sort_values(ascending=False)
            available = variances.head(max_features_per_category).index.tolist()
            logger.info(
                "Sampling budget: %s trimmed to %d features (from %d)",
                cat_name,
                len(available),
                len(features),
            )
        category_tasks.append((cat_name, available))

    # --- Task 2.1: Parallel execution via joblib (simplified guard) ---
    def _safe_analyze_task(task: tuple[str, list[str]]):
        cat_name, available = task
        try:
            return _analyze_single_category(
                df,
                cat_name,
                available,
                use_mcmc,
                n_mcmc_samples,
                burn_in,
            )
        except Exception as e:
            logger.warning("Category analytics failed for %s: %s", cat_name, e)
            return cat_name, None

    task_results = run_parallel_or_sequential(
        category_tasks,
        n_jobs=n_jobs,
        worker=_safe_analyze_task,
    )

    for cat_name, cat_result in task_results:
        if not cat_result:
            continue
        results[cat_name] = cat_result
        logger.info(
            "Category %s: %d features analyzed",
            cat_name,
            cat_result.get("features_analyzed", 0),
        )

    # --- Task 2.4: Save to cache ---
    if cache_path is not None and results:
        try:
            save_json(cache_path, results)
            logger.info("Category analytics cached → %s", cache_path.name)
        except Exception as e:
            logger.debug("Failed to save category analytics cache: %s", e)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. Screening Runners (v3.0 — NEW)
# ═══════════════════════════════════════════════════════════════════════════════


def _adaptive_screen_fallback(
    df_all: pd.DataFrame,
    screen_result: pd.DataFrame,
    screen_name: str,
    min_pct: float = 1.0,
    fallback_percentile: float = 90.0,
) -> pd.DataFrame:
    """
    Apply percentile-based fallback when a screen yields < min_pct of universe.

    Task 3.1: When absolute thresholds produce too few results, relax to
    percentile-based thresholds and log the relaxed criteria used.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full universe DataFrame.
    screen_result : pd.DataFrame
        Result from the primary screen.
    screen_name : str
        Name of the screen (for logging).
    min_pct : float
        Minimum percentage of universe required before fallback triggers.
    fallback_percentile : float
        Percentile threshold for the fallback (e.g. 90 = top 10%).

    Returns
    -------
    pd.DataFrame
        Original result if sufficient, otherwise percentile-based fallback.
    """
    universe_size = len(df_all)
    if universe_size == 0:
        return screen_result

    result_pct = 100.0 * len(screen_result) / universe_size
    if result_pct >= min_pct:
        return screen_result

    logger.warning(
        "Adaptive screening: %s returned %d stocks (%.1f%% < %.1f%% threshold). "
        "Applying percentile-based fallback (top %.0f%%).",
        screen_name,
        len(screen_result),
        result_pct,
        min_pct,
        100.0 - fallback_percentile,
    )

    # Build a simple composite score from available numeric columns for fallback
    score_cols = []
    for col in ["composite_score", "piotroski_f_score", "roe", "altman_z_score"]:
        if col in df_all.columns and pd.api.types.is_numeric_dtype(df_all[col]):
            score_cols.append(col)

    if not score_cols:
        logger.warning(
            "Adaptive screening: no score columns available for %s fallback",
            screen_name,
        )
        return screen_result

    # Use the first available score column for percentile ranking
    score_col = score_cols[0]
    threshold = df_all[score_col].quantile(fallback_percentile / 100.0)
    fallback = df_all[df_all[score_col] >= threshold].copy()
    logger.info(
        "Adaptive screening: %s fallback using %s >= %.2f → %d stocks",
        screen_name,
        score_col,
        threshold,
        len(fallback),
    )
    return fallback


def run_stock_screening(
    df_all: pd.DataFrame,
    *,
    min_pct: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """
    Run all stock screening strategies on the full feature set.

    Uses ``mv_all_stock_features`` as input (broader feature coverage
    than ``mv_expected_returns``).

    v3.6: Added adaptive screening fallback (Task 3.1) — when a screen
    returns < ``min_pct`` of the universe, a percentile-based fallback
    is applied and logged.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full feature DataFrame from ``mv_all_stock_features``.
    min_pct : float
        Minimum percentage of universe for adaptive fallback (Task 3.1).

    Returns
    -------
    dict[str, pd.DataFrame]
        Screening results keyed by strategy name.
    """
    # v3.9: Ensure unique input universe for screening
    df_all = (
        df_all.drop_duplicates(subset="isin") if "isin" in df_all.columns else df_all
    )
    screens: dict[str, pd.DataFrame] = {}

    # Quality screening (dynamic thresholds from data distributions)
    try:
        screens["quality"] = create_enhanced_screener(df_all)
        logger.info("Quality screen: %d stocks", len(screens["quality"]))
        if len(screens["quality"]) == 0 or (
            100.0 * len(screens["quality"]) / max(len(df_all), 1) < min_pct
        ):
            logger.warning(
                "Quality screen returned %d stocks — applying adaptive fallback.",
                len(screens["quality"]),
            )
            screens["quality"] = _adaptive_screen_fallback(
                df_all,
                screens["quality"],
                "quality",
                min_pct=min_pct,
            )
    except Exception as e:
        logger.warning("Quality screening failed: %s", e)

    # Earnings quality
    try:
        screens["earnings_quality"] = screen_earnings_quality(df_all)
        logger.info(
            "Earnings quality screen: %d stocks", len(screens["earnings_quality"])
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
        if 100.0 * len(screens["dividend"]) / max(len(df_all), 1) < min_pct:
            logger.warning(
                "Dividend screen returned only %d stocks (%.1f%% of universe) — "
                "applying adaptive fallback.",
                len(screens["dividend"]),
                100.0 * len(screens["dividend"]) / max(len(df_all), 1),
            )
            screens["dividend"] = _adaptive_screen_fallback(
                df_all,
                screens["dividend"],
                "dividend",
                min_pct=min_pct,
                fallback_percentile=85.0,
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
            "Valuation reversion screen: %d stocks", len(screens["valuation_reversion"])
        )
    except Exception as e:
        logger.warning("Valuation reversion screening failed: %s", e)

    # Integrity-filtered growth
    try:
        screens["integrity_growth"] = screen_integrity_filtered_growth(df_all)
        logger.info(
            "Integrity growth screen: %d stocks", len(screens["integrity_growth"])
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
            else "expected_upside_pt",
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

    Issue 5: When probabilistic model columns are present in the summary
    they are merged into source_df before scoring so that
    ``rank_stocks_by_composite_score`` can use model-aware weights.
    Quality tier bins are tightened to [0, 30, 50, 70, 100] so that
    the "High" bucket is more selective.
    """
    if summary.empty or source_df.empty:
        return summary

    # Issue 5: enrich source_df with probabilistic columns from summary
    _prob_cols = [
        "analyst_conviction",
        "achievement_probability",
        "posterior_beat_prob",
        "resampled_posterior_mean",
        "prob_beat_given_momentum",
        "confidence_score",
    ]
    enriched_source = source_df.copy()
    prob_available = [
        c
        for c in _prob_cols
        if c in summary.columns and c not in enriched_source.columns
    ]
    if (
        prob_available
        and "isin" in summary.columns
        and "isin" in enriched_source.columns
    ):
        prob_subset = summary[["isin"] + prob_available].drop_duplicates(subset="isin")
        enriched_source = enriched_source.merge(prob_subset, on="isin", how="left")

    ranked = rank_stocks_by_composite_score(enriched_source)
    if "composite_score" in ranked.columns and "isin" in ranked.columns:
        score_map = ranked.set_index("isin")["composite_score"]
        summary["composite_score"] = summary["isin"].map(score_map)

        # Use data-adaptive quantile bins for balanced tier distribution
        valid_scores = summary["composite_score"].dropna()
        if len(valid_scores) > 100:
            q_bins = [
                valid_scores.min() - 0.01,
                valid_scores.quantile(0.10),
                valid_scores.quantile(0.30),
                valid_scores.quantile(0.50),
                valid_scores.quantile(0.70),
                valid_scores.quantile(0.90),
                valid_scores.max() + 0.01,
            ]
            q_bins = sorted(set(q_bins))
            if len(q_bins) >= 7:
                tier_labels = [
                    "Very Low", "Low", "Below Avg", "Above Avg", "High", "Premium",
                ]
            else:
                q_bins = [18, 25, 35, 45, 55, 60, 75]
                tier_labels = [
                    "Very Low", "Low", "Below Avg", "Above Avg", "High", "Premium",
                ]
        else:
            q_bins = [18, 25, 35, 45, 55, 60, 75]
            tier_labels = [
                "Very Low", "Low", "Below Avg", "Above Avg", "High", "Premium",
            ]

        summary["quality_tier"] = pd.cut(
            summary["composite_score"],
            bins=q_bins,
            labels=tier_labels[: len(q_bins) - 1],
        )

        # Detailed tier distribution logging
        tier_counts = summary["quality_tier"].value_counts().sort_index()
        logger.info(
            "Quality scoring: %d stocks scored (mean=%.1f, median=%.1f)",
            summary["composite_score"].notna().sum(),
            summary["composite_score"].mean(),
            summary["composite_score"].median(),
        )
        for tier_label in [
            "Premium",
            "High",
            "Above Avg",
            "Below Avg",
            "Low",
            "Very Low",
        ]:
            count = tier_counts.get(tier_label, 0)
            if count > 0:
                logger.info("  %s: %d stocks", tier_label, count)

        # Flag financially healthy stocks (composite >= 45)
        summary["financially_healthy"] = summary["composite_score"] >= 45
        logger.info(
            "  Financially healthy (score \u2265 45): %d stocks",
            summary["financially_healthy"].sum(),
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

_SIGNAL_LABELS_7 = {
    0: "Strong Bearish (0/7)",
    1: "Bearish (1/7)",
    2: "Bearish (2/7)",
    3: "Neutral (3/7)",
    4: "Neutral (4/7)",
    5: "Bullish (5/7)",
    6: "Bullish (6/7)",
    7: "Strong Bullish (7/7)",
}


def build_tri_model_alignment(
    mc: pd.DataFrame,
    kal: pd.DataFrame,
    pt: pd.DataFrame,
    *,
    bullish_return_threshold: float = 0.02,  # v3.9: was 0.0 — heavy-tail materiality floor
    bma_weights: tuple[float, float, float] = (0.45, 0.25, 0.30),  # (MC, Kalman, PT)
    use_log_score_reweighting: bool = True,
    cvar_alpha: float = 0.05,
    student_t_df: float | None = None,
) -> pd.DataFrame:
    """
    Merge Monte Carlo, Kalman, and Price Target Achievement into a
    tri-model alignment DataFrame with direction agreement scores.

    Parameters
    ----------
    mc : pd.DataFrame
        Monte Carlo simulation results with ``implied_return_mc``.
    kal : pd.DataFrame
        Kalman filter results with ``implied_return_kalman``.
    pt : pd.DataFrame
        Price target achievement results with ``implied_return_pt``.
    bullish_return_threshold : float
        Minimum implied return (%) to classify a model as bullish (Issue 8).
    """
    if mc.empty or kal.empty or pt.empty:
        logger.warning("Tri-model alignment skipped — one or more inputs empty")
        return pd.DataFrame()

    # Ensure 'isin' is a column for merging (Issue: KeyError 'isin' when it's the index)
    mc = _ensure_isin_column(mc)
    kal = _ensure_isin_column(kal)
    pt = _ensure_isin_column(pt)

    id_cols_set = get_identifier_cols_set()
    mc_id_cols = [c for c in mc.columns if c in id_cols_set]
    mc_select = list(
        set(
            mc_id_cols
            + [
                "isin",
                "expected_upside_mc",
                "implied_return_mc",
                "price_target_mc",
                "prob_positive_upside",
                "var_5_pct",
                "risk_reward_ratio",
            ]
        )
    )
    mc_select = [c for c in mc_select if c in mc.columns]

    tri = (
        mc[mc_select]
        .copy()
        .merge(
            kal[
                [
                    c
                    for c in [
                        "isin",
                        "implied_return_kalman",
                        "expected_upside_kalman",
                        "price_target_kalman",
                        "kalman_estimate",
                        "kalman_variance",
                        "kalman_gain",
                        "signal_strength",
                    ]
                    if c in kal.columns
                ]
            ],
            on="isin",
        )
        .merge(
            pt[
                [
                    c
                    for c in [
                        "isin",
                        "implied_return_pt",
                        "achievement_probability",
                        "price_target_prob_weighted",
                        "confidence_level",
                        "analyst_conviction",
                        "eps_revision_momentum",
                        "analyst_rating_normalized",
                    ]
                    if c in pt.columns
                ]
            ],
            on="isin",
        )
    )

    # Issue 8: scale-aware materiality threshold for bullish classification
    # Use model-specific thresholds based on each model's distribution
    # to prevent scale mismatch from biasing agreement scores
    mc_threshold = max(
        bullish_return_threshold, float(tri["implied_return_mc"].quantile(0.40))
    )
    kal_threshold = max(
        bullish_return_threshold, float(tri["implied_return_kalman"].quantile(0.40))
    )
    pt_threshold = max(bullish_return_threshold, float(tri["implied_return_pt"].quantile(0.40)))

    tri["mc_bullish"] = tri["implied_return_mc"] > mc_threshold
    tri["kal_bullish"] = tri["implied_return_kalman"] > kal_threshold
    tri["pt_bullish"] = tri["implied_return_pt"] > pt_threshold
    tri["agreement_score"] = (
        tri["mc_bullish"].astype(int)
        + tri["kal_bullish"].astype(int)
        + tri["pt_bullish"].astype(int)
    )

    # v3.9: Bayesian Model Averaging blended expected return (Finding #3)
    w_mc, w_kal, w_pt = bma_weights
    _w_total = w_mc + w_kal + w_pt
    if _w_total > 0:
        w_mc, w_kal, w_pt = w_mc / _w_total, w_kal / _w_total, w_pt / _w_total
    tri["blended_return_bma"] = (
        w_mc * tri["implied_return_mc"]
        + w_kal * tri["implied_return_kalman"]
        + w_pt * tri["implied_return_pt"]
    )

    # v3.9 Cross-cutting T-A: per-stock ``tail_df`` sourced from each
    # model's *Result dataclass (CreditRiskResult / DividendSafetyResult /
    # PriceTargetResult). Prefer a per-stock column when present; fall
    # back to the global ``student_t_df`` scalar for backwards
    # compatibility with the v3.8 code path.
    per_stock_tail_df: pd.Series | None = None
    for _src in (pt, mc, kal):
        if isinstance(_src, pd.DataFrame) and "tail_df" in _src.columns:
            _cand = (
                _src[["isin", "tail_df"]]
                .dropna(subset=["tail_df"])
                .drop_duplicates(subset=["isin"])
            )
            if not _cand.empty:
                per_stock_tail_df = _cand.set_index("isin")["tail_df"]
                break

    def _df_to_penalty(df_value: float) -> float:
        if not np.isfinite(df_value):
            return 1.0
        if df_value <= 3.0:
            return 0.5
        if df_value <= 5.0:
            return 0.75
        return 1.0

    if per_stock_tail_df is not None:
        tri["tail_df"] = tri["isin"].map(per_stock_tail_df).astype(float)
        if student_t_df is not None:
            tri["tail_df"] = tri["tail_df"].fillna(float(student_t_df))
        tri["tail_penalty"] = tri["tail_df"].map(_df_to_penalty).fillna(1.0)
        tail_penalty = float(tri["tail_penalty"].mean())
    else:
        if student_t_df is not None and student_t_df <= 3.0:
            tail_penalty = 0.5
        elif student_t_df is not None and student_t_df <= 5.0:
            tail_penalty = 0.75
        else:
            tail_penalty = 1.0
        tri["tail_df"] = float(student_t_df) if student_t_df is not None else float("nan")
        tri["tail_penalty"] = tail_penalty
    tri["blended_conviction"] = tri["agreement_score"] * tri["tail_penalty"]

    # v3.9: Expose CVaR column when available on MC output
    cvar_col = f"cvar_{int(cvar_alpha * 100)}"
    if cvar_col in mc.columns:
        mc_cvar = mc[["isin", cvar_col]].drop_duplicates(subset=["isin"])
        tri = tri.merge(mc_cvar, on="isin", how="left")
    elif "var_5_pct" in tri.columns and cvar_col not in tri.columns:
        tri[cvar_col] = tri["var_5_pct"]
    tri["signal"] = tri["agreement_score"].map(_SIGNAL_LABELS)

    logger.info(
        "Tri-model alignment: %d stocks, %d strong bullish",
        len(tri),
        (tri["agreement_score"] == 3).sum(),
    )
    return tri


def _ensure_isin_column(df: pd.DataFrame | None) -> pd.DataFrame:
    """Ensure 'isin' is a column in the DataFrame, resetting index if needed.

    Used to normalize model outputs (MC, Kalman, PT) which often return
    'isin' as the index.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_out = df.copy()
    if "isin" not in df_out.columns:
        if df_out.index.name == "isin":
            df_out = df_out.reset_index()
        elif "isin" in df_out.index.names:
            df_out = df_out.reset_index(level="isin")
        else:
            # Fallback: if 'isin' is nowhere, we might be in trouble,
            # but we try to see if any other column can be used or if index is strings
            logger.debug(
                "_ensure_isin_column: 'isin' not found in columns or index name"
            )

    return df_out


def _ensure_ticker_from_isin(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'ticker' column exists, deriving it from 'isin' if necessary.

    Many downstream components (e.g. EquitiesMaterializedViewSpec,
    InferenceData coordinate builders, cache checksum helpers) require a
    'ticker' column.  The project convention is to treat 'isin' as the
    canonical equity identifier, so this helper bridges the gap by:

    1. Returning *df* unchanged if 'ticker' already exists.
    2. Copying 'isin' → 'ticker' when only 'isin' is present.
    3. Falling back to the DataFrame index when named 'isin' or 'ticker'.

    The original DataFrame is never mutated; a copy is returned only when
    a column needs to be added.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    if "ticker" in df.columns:
        return df

    result = df.copy()

    if "isin" in result.columns:
        result["ticker"] = result["isin"]
        logger.debug(
            "_ensure_ticker_from_isin: derived 'ticker' from 'isin' (%d rows)",
            len(result),
        )
    elif result.index.name == "isin":
        result = result.reset_index()
        result["ticker"] = result["isin"]
    elif result.index.name == "ticker":
        result = result.reset_index()
    else:
        logger.debug(
            "_ensure_ticker_from_isin: neither 'ticker' nor 'isin' found — "
            "downstream spec validation may fail"
        )

    return result


def build_quad_model_alignment(
    tri: pd.DataFrame,
    beat: pd.DataFrame,
    beat_threshold: float = 0.55,  # v3.9: was 0.50
    credit: pd.DataFrame | None = None,
    div_safety: pd.DataFrame | None = None,
    anomaly: pd.DataFrame | None = None,
    *,
    bma_weights: dict[str, float] | None = None,  # v3.9: full six-model BMA weights
    credit_distress_threshold: float = 0.90,  # v3.9: softened from 0.99
    div_cut_threshold: float = 0.60,  # v3.9: softened from 0.67
    anomaly_severity_threshold: float | None = None,
    mcmc_result: dict | None = None,
    use_macro_tilt: bool = True,  # v3.9: regional tilt from macro covariates
) -> pd.DataFrame:
    """Extend tri-model alignment with up to 4 additional model signals.

    R4 refactorings (Issues 1–3):
    - Two-tier scoring: ``directional_agreement`` (0–4) + ``risk_quality_score`` (0–3).
    - ``full_consensus`` = directional 4/4 AND risk ≥ 2/3.
    - Median fillna + coverage flags for missing risk data.
    - Data-adaptive anomaly threshold when ``None``.

    Parameters
    ----------
    tri : pd.DataFrame
        Tri-model alignment DataFrame from ``build_tri_model_alignment``.
    beat : pd.DataFrame
        Earnings beat analysis results with ``prob_beat_given_momentum``.
    beat_threshold : float, default 0.50
        Minimum beat probability to classify as bullish.
    credit : pd.DataFrame or None, optional
        Credit risk analysis results with ``distress_probability``.
    div_safety : pd.DataFrame or None, optional
        Dividend safety results with ``dividend_cut_probability``.
    anomaly : pd.DataFrame or None, optional
        Accounting anomaly results with severity scores.
    credit_distress_threshold : float, default 0.50
        Maximum distress probability to classify as credit-safe.
    div_cut_threshold : float, default 0.50
        Maximum dividend cut probability to classify as dividend-safe.
    anomaly_severity_threshold : float or None
        When ``None`` the threshold is set to the median of the anomaly
        severity distribution (data-adaptive, Issue 3).
        :param mcmc_result:
    """
    if tri.empty or beat.empty:
        logger.warning("Quad-model alignment skipped — insufficient data")
        return pd.DataFrame()

    # v3.9: Full six-model BMA weights (Finding #3) — normalised default
    if bma_weights is None:
        bma_weights = {
            "mc": 0.30,
            "kalman": 0.20,
            "pt": 0.20,
            "beat": 0.15,
            "credit": 0.10,
            "div": 0.05,
        }
    _bma_total = sum(bma_weights.values()) or 1.0
    bma_weights = {k: v / _bma_total for k, v in bma_weights.items()}

    # Ensure 'isin' is a column for merging
    tri = _ensure_isin_column(tri)
    beat = _ensure_isin_column(beat)
    credit = _ensure_isin_column(credit)
    div_safety = _ensure_isin_column(div_safety)
    anomaly = _ensure_isin_column(anomaly)

    if "prob_beat_given_momentum" not in beat.columns:
        logger.warning(
            "Quad-model skipped — beat results missing prob_beat_given_momentum"
        )
        return pd.DataFrame()

    beat_slim = beat[["isin", "prob_beat_given_momentum"]].rename(
        columns={"prob_beat_given_momentum": "beat_prob"}
    )
    quad = tri.merge(beat_slim, on="isin")
    if quad.empty:
        return quad

    quad["beat_bullish"] = (quad["beat_prob"] >= beat_threshold).astype(int)

    # --- Credit risk signal (Issue 2: median fillna + coverage flag) ---
    if (
        credit is not None
        and not credit.empty
        and "isin" in credit.columns
        and "distress_probability" in credit.columns
    ):
        credit_slim = credit[["isin", "distress_probability"]].drop_duplicates(
            subset="isin"
        )
        median_distress = credit_slim["distress_probability"].median()
        quad = quad.merge(credit_slim, on="isin", how="left")
        quad["credit_coverage"] = quad["distress_probability"].notna().astype(int)
        quad["distress_probability"] = quad["distress_probability"].fillna(
            median_distress
        )
        quad["credit_safe"] = (
            quad["distress_probability"] < credit_distress_threshold
        ).astype(int)
        logger.info(
            "Credit risk signal merged: %d/%d stocks flagged credit-safe (median fill=%.3f)",
            quad["credit_safe"].sum(),
            len(quad),
            median_distress,
        )
    else:
        quad["credit_safe"] = 0
        quad["credit_coverage"] = 0
        logger.debug("Credit risk signal not available — defaulting to 0")

    # --- Dividend safety signal (Issue 2: median fillna + coverage flag) ---
    if (
        div_safety is not None
        and not div_safety.empty
        and "isin" in div_safety.columns
        and "dividend_cut_probability" in div_safety.columns
    ):
        div_slim = div_safety[["isin", "dividend_cut_probability"]].drop_duplicates(
            subset="isin"
        )
        median_div_cut = div_slim["dividend_cut_probability"].median()
        quad = quad.merge(div_slim, on="isin", how="left")
        quad["div_coverage"] = quad["dividend_cut_probability"].notna().astype(int)
        quad["dividend_cut_probability"] = quad["dividend_cut_probability"].fillna(
            median_div_cut
        )
        quad["div_safe"] = (
            quad["dividend_cut_probability"] < div_cut_threshold
        ).astype(int)
        logger.info(
            "Dividend safety signal merged: %d/%d stocks flagged div-safe (median fill=%.3f)",
            quad["div_safe"].sum(),
            len(quad),
            median_div_cut,
        )
    else:
        quad["div_safe"] = 0
        quad["div_coverage"] = 0
        logger.debug("Dividend safety signal not available — defaulting to 0")

    # --- Accounting anomaly signal (Issue 2 + Issue 3: adaptive threshold) ---
    if (
        anomaly is not None
        and not anomaly.empty
        and "isin" in anomaly.columns
        and "anomaly_severity_score" in anomaly.columns
    ):
        anomaly_slim = anomaly[["isin", "anomaly_severity_score"]].drop_duplicates(
            subset="isin"
        )
        # Issue 3: data-adaptive threshold when not explicitly set
        if anomaly_severity_threshold is None:
            anomaly_severity_threshold = float(
                anomaly_slim["anomaly_severity_score"].quantile(0.50)
            )
            logger.info(
                "Anomaly severity threshold set adaptively to median: %.1f",
                anomaly_severity_threshold,
            )
        median_severity = anomaly_slim["anomaly_severity_score"].median()
        quad = quad.merge(anomaly_slim, on="isin", how="left")
        quad["anomaly_coverage"] = quad["anomaly_severity_score"].notna().astype(int)
        quad["anomaly_severity_score"] = quad["anomaly_severity_score"].fillna(
            median_severity
        )
        quad["anomaly_clean"] = (
            quad["anomaly_severity_score"] < anomaly_severity_threshold
        ).astype(int)
        logger.info(
            "Anomaly signal merged: %d/%d stocks flagged anomaly-clean (threshold=%.1f)",
            quad["anomaly_clean"].sum(),
            len(quad),
            anomaly_severity_threshold,
        )
    else:
        quad["anomaly_clean"] = 0
        quad["anomaly_coverage"] = 0
        logger.debug("Accounting anomaly signal not available — defaulting to 0")

    # --- Issue 1: Two-tier scoring (directional + risk quality) ---
    quad["directional_agreement"] = (
        quad["mc_bullish"].astype(int)
        + quad["kal_bullish"].astype(int)
        + quad["pt_bullish"].astype(int)
        + quad["beat_bullish"]
    )
    quad["risk_quality_score"] = (
        quad["credit_safe"] + quad["div_safe"] + quad["anomaly_clean"]
    )
    quad["full_consensus"] = (quad["directional_agreement"] == 4) & (
        quad["risk_quality_score"] >= 2
    )

    # Legacy flat agreement kept for backward compatibility
    quad["quad_agreement"] = (
        quad["mc_bullish"].astype(int)
        + quad["kal_bullish"].astype(int)
        + quad["pt_bullish"].astype(int)
        + quad["beat_bullish"]
        + quad["credit_safe"]
        + quad["div_safe"]
        + quad["anomaly_clean"]
    )

    n_models = 4
    if credit is not None and not credit.empty:
        n_models += 1
    if div_safety is not None and not div_safety.empty:
        n_models += 1
    if anomaly is not None and not anomaly.empty:
        n_models += 1

    # Signal label based on directional agreement (0–4)
    quad["signal"] = quad["directional_agreement"].map(_SIGNAL_LABELS_4)

    logger.info(
        "Ensemble alignment (%d models): %d stocks, full consensus (4/4 dir + ≥2/3 risk): %d",
        n_models,
        len(quad),
        quad["full_consensus"].sum(),
    )

    # --- Task 2: Confidence-weighted ensemble return ---
    mc_w = quad["prob_positive_upside"].clip(0, 100) / 100.0

    if "kalman_variance" in quad.columns:
        max_var = quad["kalman_variance"].quantile(0.95)
        if max_var > 0:
            kal_w = (1 - quad["kalman_variance"].clip(0, max_var) / max_var).clip(0.2, 0.9)
        else:
            kal_w = 0.5
    else:
        kal_w = 0.5

    pt_w = (
        quad["achievement_probability"].clip(0, 1)
        if "achievement_probability" in quad.columns
        else 0.5
    )
    beat_w = quad["beat_prob"].clip(0, 1)

    total_w = mc_w + kal_w + pt_w + beat_w
    quad["ensemble_return"] = (
        quad["implied_return_mc"] * mc_w
        + quad["implied_return_kalman"] * kal_w
        + quad["implied_return_pt"] * pt_w
        + quad["implied_return_mc"] * beat_w  # beat has no own return; amplify MC
    ) / total_w

    # --- Task 3: Bayesian shrinkage toward MCMC posterior ---
    if mcmc_result and mcmc_result.get("posterior_mean") is not None:
        mcmc_mu = mcmc_result["posterior_mean"]
        mcmc_std = mcmc_result.get("posterior_std", 1.0)

        stock_std = quad[["implied_return_mc", "implied_return_kalman", "implied_return_pt"]].std(
            axis=1
        )
        shrinkage = (stock_std**2) / (stock_std**2 + mcmc_std**2)
        quad["mcmc_shrinkage"] = shrinkage
        quad["ensemble_return_shrunk"] = (
            shrinkage * quad["ensemble_return"] + (1 - shrinkage) * mcmc_mu
        )
    else:
        quad["ensemble_return_shrunk"] = quad["ensemble_return"]
        quad["mcmc_shrinkage"] = 1.0

    # --- Task 4: Risk penalty via risk_quality_score ---
    risk_discount = (
        quad["risk_quality_score"].map({0: 0.70, 1: 0.85, 2: 0.95, 3: 1.00}).fillna(0.85)
    )
    quad["risk_adj_return"] = quad["ensemble_return_shrunk"] * risk_discount

    # --- Task 5: Optional hierarchical sector adjustment ---
    if mcmc_result and "hierarchical" in mcmc_result and "industry" in quad.columns:
        hier = mcmc_result["hierarchical"]
        industry_posteriors = hier.get("levels", {}).get("industry", {})
        if industry_posteriors:
            sector_mu = quad["industry"].map(
                {k: v["posterior_mean"] for k, v in industry_posteriors.items()}
            )
            has_sector = sector_mu.notna()
            quad.loc[has_sector, "risk_adj_return"] = (
                quad.loc[has_sector, "mcmc_shrinkage"] * quad.loc[has_sector, "ensemble_return"]
                + (1 - quad.loc[has_sector, "mcmc_shrinkage"]) * sector_mu[has_sector]
            ) * risk_discount[has_sector]

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
    mcmc_result: dict | None = None,
    quad: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge four expected-return model results into a unified summary DataFrame.

    v3.0: ``source_df`` is loaded from ``mv_all_stock_features`` (full
    superset) so that all identifier and market-data columns are available
    for enrichment without needing a backfill step.

    v3.3: Added optional ``credit`` and ``div_safety`` DataFrames to enrich
    the summary with credit-risk and dividend-safety columns.

    v3.6: Task 6.4 — Accepts ``mcmc_result`` from ``run_parallel_mcmc_return_analysis``
    and merges Gelman-Rubin diagnostics and posterior means into the summary.

    v3.8: Issue 4 — When ``quad`` is provided its two-tier consensus scores are
    merged directly instead of recomputing a redundant 4-model agreement.
    Issue 7 — Kalman weight derived from ``kalman_variance``.
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

    # Ensure 'isin' is a column for merging
    mc = _ensure_isin_column(mc)
    kal = _ensure_isin_column(kal)
    pt = _ensure_isin_column(pt)
    earn = _ensure_isin_column(earn)
    anomaly_results = _ensure_isin_column(anomaly_results)
    credit = _ensure_isin_column(credit)
    div_safety = _ensure_isin_column(div_safety)

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
                "isin",
                "expected_upside_mc",
                "implied_return_mc",
                "price_target_mc",
                "pt_spread",
                "upside_std",
                "prob_positive_upside",
                "var_5_pct",
                "risk_reward_ratio",
            ]
            + available_market
        )
    )
    mc_select = [c for c in mc_select if c in mc.columns]

    # v3.9: Deduplicate all inputs to prevent Cartesian explosion during merges
    mc_dedup = mc[mc_select].drop_duplicates(subset="isin")
    kal_dedup = kal[
        [
            c
            for c in [
                "isin",
                "implied_return_kalman",
                "expected_upside_kalman",
                "price_target_kalman",
                "kalman_variance",
                "kalman_gain",
                "signal_strength",
            ]
            if c in kal.columns
        ]
    ].drop_duplicates(subset="isin")
    pt_dedup = pt[
        ["isin"]
        + [
            c
            for c in [
                "expected_upside_pt",
                "price_target_spread",
                "implied_return_pt",
                "price_target_prob_weighted",
                "achievement_probability",
                "mh_achievement_probability",
                "confidence_level",
                "analyst_conviction",
                "bullish_pct",
                "eps_revision_momentum",
                "analyst_rating_normalized",
            ]
            if c in pt.columns
        ]
    ].drop_duplicates(subset="isin")
    earn_dedup = earn[
        ["isin"]
        + [
            c
            for c in [
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
                # v3.10 §15.1 ResampledBeat posterior spread & chain diagnostics
                "hdi_low",
                "hdi_high",
                "chain_rhat",
                "chain_ess_bulk",
                "chain_ess_tail",
                "n_effective_samples",
                "volatility_regime",
                "streak_type",
                "continuation_probability",
                "mean_reversion_probability",
                "expected_next_outcome",
                "prediction_confidence",
                "model_confidence",
                "map_estimate",
            ]
            if c in earn.columns
        ]
    ].drop_duplicates(subset="isin")

    summary = (
        mc_dedup.copy()
        .merge(kal_dedup, on="isin")
        .merge(pt_dedup, on="isin")
        .merge(earn_dedup, on="isin")
    )

    # Merge anomaly results — column names aligned to mv_all_stock_features schema
    _ANOMALY_COLS = [
        # Profitability / margins (corrected from gross_profit_margin_pct_*)
        "gross_margin_pct",
        # Shareholder yield (corrected from buyback_yield_ltm)
        "buyback_yield",
        # Dividend yield columns (corrected from div_yield_ttm / div_yield_5yavgltm)
        "div_yield_1fy_ind",
        "div_yield_ltm",
        "div_yield_ntm",
        "div_yield_5y_avg",
        # Forward revenue growth (corrected from revenues_est_yoy_pct_fy1e)
        "forward_revenue_growth",
        # Price momentum (corrected from price_chg_pct_*)
        "price_momentum_1m",
        "price_momentum_3m",
        "one_day_pct",
        # EPS revision columns (corrected to mv_all_stock_features names)
        "eps_revision_momentum",
        "gaap_revision_1m",
        "gaap_revision_3m",
        "gaap_revision_6m",
        "gaap_revision_1y",
        # Dividend history forward curve
        "div_yield_2fyind",
        "div_yield_3fyind",
        "div_yield_4fyind",
        "div_yield_5fyind",
        # Analyst ratings
        "dividend_streak",
        "price_target_count",
        "analyst_rating",
        "num_strong_sell_ratings",
        "num_strong_buys_ratings",
        "num_hold_ratings",
        "num_buys_ratings",
        "num_sell_ratings",
        # Anomaly model outputs
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
        "mahalanobis_distance",
        # v3.5 enhanced anomaly metrics
        "quality_frequency_score",
        "repeat_offender_flag",
        "accumulated_deficit_flag",
        "negative_wc_flag",
        "wc_deteriorating_flag",
        "intangibles_growth_flag",
        "inventory_buildup_flag",
        "inventory_reduction_flag",
        "has_goodwill_impairment",
        "has_asset_writedown",
        "has_restructuring",
        "has_goodwill_impairment_ltm",
        "impairment_risk_score",
        "revenue_accelerating_flag",
        "overinvestment_flag",
        "recent_acquisition_flag",
        "high_rnd_intensity_flag",
        "has_unusual_items_flag",
        "low_tax_flag",
        "layoff_risk_flag",
        "analyst_bearish_pct",
        "debt_maturity_risk",
    ]
    if (
        anomaly_results is not None
        and not anomaly_results.empty
        and "isin" in anomaly_results.columns
    ):
        available_anomaly = [c for c in _ANOMALY_COLS if c in anomaly_results.columns]
        if available_anomaly:
            anomaly_subset = anomaly_results[
                ["isin"] + available_anomaly
            ].drop_duplicates(subset="isin")
            summary = summary.merge(anomaly_subset, on="isin", how="left")
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
    if credit is not None and not credit.empty and "isin" in credit.columns:
        available_credit = [c for c in _CREDIT_COLS if c in credit.columns]
        if available_credit:
            credit_subset = credit[["isin"] + available_credit].drop_duplicates(
                subset="isin"
            )
            summary = summary.merge(credit_subset, on="isin", how="left")
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
    if div_safety is not None and not div_safety.empty and "isin" in div_safety.columns:
        available_div = [c for c in _DIV_SAFETY_COLS if c in div_safety.columns]
        if available_div:
            div_subset = div_safety[["isin"] + available_div].drop_duplicates(
                subset="isin"
            )
            summary = summary.merge(div_subset, on="isin", how="left")
            logger.info(
                "Enriched expected_returns_summary with %d dividend safety columns",
                len(available_div),
            )

    if summary.empty:
        logger.warning(
            "Expected returns summary: no overlapping isins across all 4 models"
        )
        return summary

    # Enrich with market-data columns from mc (if present there)
    for col in available_market:
        if col not in summary.columns and col in mc.columns:
            price_map = (
                mc[["isin", col]].drop_duplicates(subset="isin").set_index("isin")[col]
            )
            summary[col] = summary["isin"].map(price_map)
            logger.debug("Merged market-data column '%s' from mc", col)

    # Enrich from source_df (mv_all_stock_features)
    if source_df is not None and "isin" in source_df.columns:
        id_cols_ordered = load_identifier_columns()
        desired_cols = id_cols_ordered + market_data_cols
        missing_cols = [
            c
            for c in desired_cols
            if c in source_df.columns and c not in summary.columns
        ]
        if missing_cols:
            source_subset = source_df[["isin"] + missing_cols].drop_duplicates(
                subset="isin"
            )
            summary = summary.merge(source_subset, on="isin", how="left")
            logger.info(
                "Enriched expected_returns_summary with %d columns from mv_all_stock_features",
                len(missing_cols),
            )

        # Merge forward consensus columns for cross-model diagnostics
        _FORWARD_CONSENSUS_COLS = [
            "pe_forward_discount",
            "forward_pe_vs_sector_proxy",
            "ebitda_forward_growth",
            "consensus_revenue_growth",
            "forward_adjustment_trend",
            "earnings_revision_divergence",
            "fcf_est_trend",
            "fcf_est_cagr_5y",
        ]
        fwd_missing = [
            c
            for c in _FORWARD_CONSENSUS_COLS
            if c in source_df.columns and c not in summary.columns
        ]
        if fwd_missing:
            fwd_subset = source_df[["isin"] + fwd_missing].drop_duplicates(
                subset="isin"
            )
            summary = summary.merge(fwd_subset, on="isin", how="left")
            logger.info(
                "Enriched expected_returns_summary with %d forward consensus columns",
                len(fwd_missing),
            )

    # Issue 4: Merge quad alignment scores when available (single source of truth)
    if quad is not None and not quad.empty and "directional_agreement" in quad.columns:
        _quad_score_cols = [
            "directional_agreement",
            "risk_quality_score",
            "full_consensus",
            "quad_agreement",
            "signal",
            "beat_bullish",
            "credit_safe",
            "div_safe",
            "anomaly_clean",
            "ensemble_return",
            "ensemble_return_shrunk",
            "mcmc_shrinkage",
            "risk_adj_return",
        ]
        available_quad_cols = [c for c in _quad_score_cols if c in quad.columns]
        quad_scores = quad[["isin"] + available_quad_cols].drop_duplicates(
            subset="isin"
        )
        overlap = [c for c in available_quad_cols if c in summary.columns]
        if overlap:
            summary = summary.drop(columns=overlap)
        summary = summary.merge(quad_scores, on="isin", how="left")
        if "mc_bullish" not in summary.columns:
            summary["mc_bullish"] = summary.get("implied_return_mc", 0) > 10.0
        if "kal_bullish" not in summary.columns:
            summary["kal_bullish"] = summary.get("implied_return_kalman", 0) > 10.0
        if "pt_bullish" not in summary.columns:
            summary["pt_bullish"] = summary.get("implied_return_pt", 0) > 10.0
        if "earn_bullish" not in summary.columns:
            if "prob_beat_given_momentum" in summary.columns:
                summary["earn_bullish"] = summary["prob_beat_given_momentum"] >= 0.5
            else:
                summary["earn_bullish"] = False
        summary["agreement_score"] = summary["directional_agreement"]
        logger.info(
            "Merged quad alignment scores into summary (single source of truth)"
        )
    else:
        # Fallback: compute direction flags locally (legacy path)
        summary["mc_bullish"] = summary["implied_return_mc"] > 0
        summary["kal_bullish"] = summary["implied_return_kalman"] > 0
        summary["pt_bullish"] = summary["implied_return_pt"] > 0
        if "prob_beat_given_momentum" in summary.columns:
            summary["earn_bullish"] = summary["prob_beat_given_momentum"] >= 0.6
        else:
            summary["earn_bullish"] = False
        summary["agreement_score"] = (
            summary["mc_bullish"].astype(int)
            + summary["kal_bullish"].astype(int)
            + summary["pt_bullish"].astype(int)
            + summary["earn_bullish"].astype(int)
        )
        summary["signal"] = summary["agreement_score"].map(_SIGNAL_LABELS_4)

    # Issue 7: Kalman variance-based confidence weighting
    mc_weight = summary["prob_positive_upside"].clip(0.0, 100.0) / 100.0
    if "kalman_variance" in summary.columns:
        max_var = summary["kalman_variance"].quantile(0.95)
        if max_var > 0:
            kal_weight = (
                1
                - summary["kalman_variance"].clip(0.0, float(max_var)) / float(max_var)
            ).clip(0.2, 0.9)
        else:
            kal_weight = 0.5
    else:
        kal_weight = 0.5
    pt_weight = (
        summary.get("confidence_level", pd.Series(index=summary.index, dtype=object))
        .map({"High": 0.9, "Medium": 0.6, "Low": 0.3})
        .fillna(0.5)
    )
    earn_weight = summary.get(
        "confidence_score", pd.Series(0.5, index=summary.index)
    ).clip(0.0, 1.0)

    summary["weighted_agreement"] = (
        summary["mc_bullish"].astype(float) * mc_weight
        + summary["kal_bullish"].astype(float) * kal_weight
        + summary["pt_bullish"].astype(float) * pt_weight
        + summary["earn_bullish"].astype(float) * earn_weight
    )

    # Incorporate FCF estimate curve into weighted agreement
    if "fcf_est_trend" in summary.columns:
        fcf_weight = summary["fcf_est_trend"].clip(-1.0, 1.0) * 0.3
        summary["weighted_agreement"] += fcf_weight

    # v3.6 Task 6.4: Merge parallel MCMC return analysis diagnostics
    if mcmc_result and isinstance(mcmc_result, dict):
        if mcmc_result.get("converged") is not None:
            summary["mcmc_converged"] = mcmc_result.get("converged", True)
        if mcmc_result.get("r_hat") is not None:
            summary["mcmc_r_hat"] = mcmc_result["r_hat"]
        if mcmc_result.get("posterior_mean") is not None:
            summary["mcmc_posterior_mean"] = mcmc_result["posterior_mean"]
        if mcmc_result.get("posterior_std") is not None:
            summary["mcmc_posterior_std"] = mcmc_result["posterior_std"]
        logger.info(
            "MCMC diagnostics merged into summary: R̂=%.4f, converged=%s",
            mcmc_result.get("r_hat", float("nan")),
            mcmc_result.get("converged", "N/A"),
        )

    # v3.9: Tail-aware risk-adjusted expected return + CVaR + position sizing (Finding #3)
    df_hat = (
        float(mcmc_result.get("student_t_df", 10.0))
        if mcmc_result and isinstance(mcmc_result, dict)
        else 10.0
    )
    if df_hat <= 3.0:
        haircut = 0.75
    elif df_hat <= 5.0:
        haircut = 0.90
    else:
        haircut = 1.0
    summary["tail_df"] = df_hat
    summary["tail_haircut"] = haircut

    _ret_src = None
    for _c in ("blended_return_bma", "ensemble_return", "implied_return_mc"):
        if _c in summary.columns:
            _ret_src = _c
            break
    if _ret_src is not None:
        summary["risk_adjusted_expected_return"] = summary[_ret_src] * haircut

    if "cvar_5" not in summary.columns:
        if "var_5_pct" in summary.columns:
            summary["cvar_5"] = summary["var_5_pct"]
        else:
            summary["cvar_5"] = np.nan

    _post_std = (
        summary["mcmc_posterior_std"]
        if "mcmc_posterior_std" in summary.columns
        else summary.get("posterior_std", pd.Series(1.0, index=summary.index))
    )
    if "ci_width" in summary.columns:
        _ci_width = summary["ci_width"]
    else:
        _ci_width = summary.get("ci_upper_95", 1.0) - summary.get("ci_lower_95", 0.0)
    try:
        _post_std_c = pd.to_numeric(_post_std, errors="coerce").clip(lower=1e-4)
        _ci_c = pd.to_numeric(_ci_width, errors="coerce").clip(lower=1e-4)
        summary["position_size_weight"] = 1.0 / (_post_std_c * _ci_c)
    except Exception:  # pragma: no cover — defensive fallback
        summary["position_size_weight"] = np.nan

    # Remove duplicate columns before return to prevent export failures
    summary = summary.loc[:, ~summary.columns.duplicated()]

    # Report using two-tier consensus when available
    if "full_consensus" in summary.columns:
        logger.info(
            "Expected returns summary: %d stocks, full consensus (4/4 dir + ≥2/3 risk): %d",
            len(summary),
            summary["full_consensus"].sum(),
        )
    else:
        logger.info(
            "Expected returns summary: %d stocks, %d strong bullish (4/4)",
            len(summary),
            (summary["agreement_score"] == 4).sum(),
        )
    return summary


def extract_strong_consensus(
    tri: pd.DataFrame,
    min_prob_positive: float = 50.0,  # v3.9: was 33.0 — tighter given df≈2 tail risk
    min_achievement: float = 0.60,  # v3.9: was 0.50
    top_n: int = 1500,  # v3.9: was 1000
) -> pd.DataFrame:
    """Filter strong consensus picks — all 3 models bullish with high confidence."""
    if tri.empty:
        return pd.DataFrame()

    strong = tri[
        (tri["agreement_score"] == 3)
        & (tri["prob_positive_upside"] >= min_prob_positive)
        & (tri["achievement_probability"] >= min_achievement)
    ].nlargest(top_n, "implied_return_pt")

    logger.info("Strong consensus picks: %d stocks", len(strong))
    return strong


# ═══════════════════════════════════════════════════════════════════════════════
# Price Target Computation (unified)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_derived_price_target(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    price_col: str = "last_price",
    return_col: str = "implied_return_pt",
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
        if "isin" not in source_df.columns or price_col not in source_df.columns:
            logger.warning(
                "Cannot compute %s — '%s' or 'isin' missing from source_df",
                output_col,
                price_col,
            )
            result[output_col] = np.nan
            return result

        price_map = (
            source_df[["isin", price_col]]
            .drop_duplicates(subset="isin")
            .set_index("isin")[price_col]
        )
        result[price_col] = result["isin"].map(price_map)

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
    return_col: str = "implied_return_pt",
    output_col: str = "price_target_prob_weighted",
) -> pd.DataFrame:
    """Calculate price target from probability-weighted return. Delegates to ``compute_derived_price_target``."""
    return compute_derived_price_target(pt, source_df, price_col, return_col, output_col)


def compute_price_target_mc(
    pt: pd.DataFrame,
    source_df: pd.DataFrame,
    price_col: str = "last_price",
    return_col: str = "implied_return_mc",
    output_col: str = "price_target_mc",
) -> pd.DataFrame:
    """Calculate price target from Monte Carlo expected upside. Delegates to ``compute_derived_price_target``."""
    return compute_derived_price_target(pt, source_df, price_col, return_col, output_col)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Analytical Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compute_sector_expected_returns(tri: pd.DataFrame) -> pd.DataFrame:
    """Aggregate expected return metrics by industry sector across all models."""
    if tri.empty or "industry" not in tri.columns:
        return pd.DataFrame()

    agg_dict = {
        "mc_mean": ("implied_return_mc", "mean"),
        "mc_median": ("implied_return_mc", "median"),
        "kalman_mean": ("implied_return_kalman", "mean"),
        "kalman_median": ("implied_return_kalman", "median"),
        "pt_mean": ("implied_return_pt", "mean"),
        "pt_median": ("implied_return_pt", "median"),
        "pct_bullish": ("agreement_score", lambda x: (x == 3).mean() * 100),
        "count": ("isin", "count"),
    }
    if "price_target_mc" in tri.columns:
        agg_dict["price_target_mc_mean"] = ("price_target_mc", "mean")
        agg_dict["price_target_mc_median"] = ("price_target_mc", "median")
    if "price_target_kalman" in tri.columns:
        agg_dict["price_target_kalman_mean"] = ("price_target_kalman", "mean")
        agg_dict["price_target_kalman_median"] = ("price_target_kalman", "median")

    return (
        tri.groupby("industry")
        .agg(**agg_dict)
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
            ("implied_return_mc", "mc"),
            ("price_target_mc", "mc_price"),
            ("implied_return_kalman", "kalman"),
            ("price_target_kalman", "kalman_price"),
            ("implied_return_pt", "pt"),
        ]:
            if col in group.columns:
                s = as_float_series(group[col]).dropna()
                row[f"{prefix}_mean"] = float(s.mean()) if len(s) > 0 else None
                row[f"{prefix}_median"] = float(s.median()) if len(s) > 0 else None
                row[f"{prefix}_std"] = float(s.std()) if len(s) > 1 else None
                if len(s) > 1:
                    se = s.std() / np.sqrt(len(s))
                    row[f"{prefix}_ci_low"] = float(s.mean() - 1.96 * se)
                    row[f"{prefix}_ci_high"] = float(s.mean() + 1.96 * se)
                if len(s) > 3:
                    # v3.5: Clip to prevent overflow in higher-moment calculations (Issue 12)
                    s_safe = s.clip(-1e9, 1e9)
                    row[f"{prefix}_skew"] = float(s_safe.skew())
                    row[f"{prefix}_kurtosis"] = float(s_safe.kurtosis())

        if "agreement_score" in group.columns:
            row["pct_bullish_3plus"] = float(
                (group["agreement_score"] >= 3).mean() * 100
            )
            row["pct_full_consensus"] = float(
                (group["agreement_score"] == 3).mean() * 100
            )
        if "weighted_agreement" in group.columns:
            row["mean_weighted_agreement"] = float(group["weighted_agreement"].mean())

        if "implied_return_mc" in group.columns:
            mc_mean = group["implied_return_mc"].mean()
            mc_std = group["implied_return_mc"].std()
            row["risk_adjusted_return"] = (
                float(mc_mean / mc_std) if mc_std > 0 else None
            )

        if "prob_beat_given_momentum" in group.columns:
            row["mean_beat_prob"] = float(group["prob_beat_given_momentum"].mean())

        results.append(row)

    return (
        pd.DataFrame(results)
        .sort_values("mc_mean", ascending=False)
        .reset_index(drop=True)
    )


def compute_return_zscore_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Add industry-relative z-scores and percentile ranks for key return metrics."""
    if summary.empty:
        return summary

    return_cols = [
        c
        for c in [
            "implied_return_mc",
            "price_target_mc",
            "implied_return_kalman",
            "price_target_kalman",
            "implied_return_pt",
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

    # Ensure 'isin' is a column for merging
    mc = _ensure_isin_column(mc)
    kal = _ensure_isin_column(kal)

    mc_cols = {"isin", "implied_return_mc"}
    kal_cols = {"isin", "implied_return_kalman"}
    if not mc_cols.issubset(mc.columns) or not kal_cols.issubset(kal.columns):
        return {"correlation": None, "n_stocks": 0}

    mc_merge_cols = ["isin", "implied_return_mc"]
    if "price_target_mc" in mc.columns:
        mc_merge_cols.append("price_target_mc")
    kal_merge_cols = ["isin", "implied_return_kalman"]
    if "price_target_kalman" in kal.columns:
        kal_merge_cols.append("price_target_kalman")

    merged = mc[mc_merge_cols].merge(
        kal[kal_merge_cols],
        on="isin",
    )
    if len(merged) < 10:
        return {"correlation": None, "n_stocks": len(merged)}

    corr = merged[["implied_return_mc", "implied_return_kalman"]].corr().iloc[0, 1]
    result: dict = {"correlation": float(corr), "n_stocks": len(merged)}

    # Add correlation between dollar-denominated price target columns
    if "price_target_mc" in merged.columns and "price_target_kalman" in merged.columns:
        price_corr = (
            merged[["price_target_mc", "price_target_kalman"]].corr().iloc[0, 1]
        )
        result["price_correlation"] = float(price_corr)

    if len(merged) > 50:
        try:
            copula = fit_gaussian_copula(
                merged,
                features=["implied_return_mc", "implied_return_kalman"],
            )
            if copula:
                result["tail_dependence"] = copula.get("tail_dependence")
        except Exception as e:
            logger.debug("Copula fit skipped: %s", e)

    return result


def compute_cross_model_diagnostics(summary: pd.DataFrame) -> dict:
    """Comprehensive cross-model dispersion and convergence diagnostics.

    Issue 6: Diagnostics are now split into **return-based** (implied_return_*)
    and **price-based** (price_target_*) groups so that percentage returns are
    never mixed with dollar price targets.  Dispersion and Kendall τ are
    computed within each group separately.
    """
    if summary.empty:
        return {}

    # Issue 6: separate return-based and price-based column groups
    _return_cols = ["implied_return_mc", "implied_return_kalman", "implied_return_pt"]
    _price_cols = [
        "price_target_mc",
        "price_target_kalman",
        "price_target_prob_weighted",
    ]

    available_return_cols = [c for c in _return_cols if c in summary.columns]
    available_price_cols = [c for c in _price_cols if c in summary.columns]

    if len(available_return_cols) < 2 and len(available_price_cols) < 2:
        return {}

    def _compute_group_diagnostics(
        df: pd.DataFrame, cols: list[str], group_name: str
    ) -> dict:
        """Compute dispersion, direction agreement, bias and Kendall τ for a column group."""
        sub = df[cols].dropna()
        if sub.empty or len(cols) < 2:
            return {}

        pearson_corr = sub.corr().to_dict()
        spearman_corr = sub.corr(method="spearman").to_dict()

        row_means = sub.mean(axis=1)
        mad_per_stock = sub.sub(row_means, axis=0).abs().mean(axis=1)

        direction_agreement = (sub > 0).nunique(axis=1) == 1
        tail_agreement_pct = float(direction_agreement.mean() * 100)

        model_bias = {col: float(sub[col].mean()) for col in cols}

        concordance_pairs = {}
        try:
            from scipy.stats import kendalltau

            for i, c1 in enumerate(cols):
                for c2 in cols[i + 1 :]:
                    tau, p = kendalltau(sub[c1], sub[c2])
                    concordance_pairs[f"{c1} ↔ {c2}"] = {
                        "kendall_tau": float(tau),
                        "p_value": float(p),
                    }
        except Exception:
            pass

        return {
            "pairwise_pearson": pearson_corr,
            "pairwise_spearman": spearman_corr,
            "kendall_concordance": concordance_pairs,
            "mean_dispersion": float(mad_per_stock.mean()),
            "median_dispersion": float(mad_per_stock.median()),
            "tail_agreement_pct": tail_agreement_pct,
            "model_bias": model_bias,
            "n_stocks": len(sub),
        }

    result: dict = {}

    # Return-based diagnostics (primary)
    if len(available_return_cols) >= 2:
        return_diag = _compute_group_diagnostics(
            summary, available_return_cols, "returns"
        )
        result.update(return_diag)
        result["return_diagnostics"] = return_diag

    # Price-based diagnostics (secondary)
    if len(available_price_cols) >= 2:
        price_diag = _compute_group_diagnostics(summary, available_price_cols, "prices")
        result["price_diagnostics"] = price_diag

    # High-dispersion tickers (from return group only)
    if len(available_return_cols) >= 2:
        returns_df = summary[available_return_cols].dropna()
        row_means = returns_df.mean(axis=1)
        mad_per_stock = returns_df.sub(row_means, axis=0).abs().mean(axis=1)
        summary_copy = summary.loc[returns_df.index].copy()
        summary_copy["model_dispersion"] = mad_per_stock
        if "isin" in summary.columns:
            summary_copy["isin"] = summary.loc[returns_df.index, "isin"].values
            result["high_dispersion_tickers"] = summary_copy.nlargest(
                20, "model_dispersion"
            )[["isin", "model_dispersion"] + available_return_cols]

    logger.info(
        "Cross-model diagnostics: tail agreement=%.1f%%, mean return dispersion=%.2f",
        result.get("tail_agreement_pct", 0.0),
        result.get("mean_dispersion", 0.0),
    )
    return result


def compute_return_distribution_analytics(
    mc: pd.DataFrame,
    summary: pd.DataFrame | None = None,
) -> dict:
    """Fit parametric distributions to MC simulation returns and compute risk metrics."""
    result = {}
    if mc.empty or "implied_return_mc" not in mc.columns:
        return result

    upside = mc["implied_return_mc"].dropna().values

    # v3.5: Clip extreme returns to prevent overflow in scipy.stats (Issue 12)
    # Return percentages above 10,000% or below -100% are typically data errors
    # or extreme penny stocks that break parametric fitting.
    upside = np.clip(upside, -100.0, 10000.0)

    best_dist = None
    best_aic = np.inf
    candidates = [sp_stats.norm, sp_stats.t, sp_stats.skewnorm, sp_stats.laplace]

    # Suppress RuntimeWarnings (overflow, divide by zero) during fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
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
                            sp_stats.kstest(upside, dist.cdf, args=params).statistic
                        ),
                        "ks_pvalue": float(
                            sp_stats.kstest(upside, dist.cdf, args=params).pvalue
                        ),
                    }
            except Exception:
                continue

    result["mc_distribution"] = best_dist

    var_1 = float(np.percentile(upside, 1))
    var_5 = float(np.percentile(upside, 5))
    cvar_5 = float(upside[upside <= var_5].mean()) if (upside <= var_5).any() else var_5
    downside = upside[upside < 0]
    downside_mean = float(np.mean(downside)) if len(downside) > 0 else 0.0
    downside_deviation = (
        float(np.sqrt((downside**2).mean())) if len(downside) > 0 else 0.0
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
        "mean_negative_return": downside_mean,
        "gain_loss_ratio": (
            float(upside[upside > 0].mean() / abs(downside_mean))
            if len(downside) > 0 and downside_mean != 0
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
                ).sum()
            )
            if "prob_positive_upside" in mc.columns
            else 0
        ),
        "avoid": int((upside <= 0).sum()),
    }

    # --- MCMC Student-t posterior for robust tail estimation ---
    try:
        mu_samples, df_samples = mcmc_student_t(upside, n_samples=5000)
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
            "implied_return_mc",
            "implied_return_kalman",
            "implied_return_pt",
        ]
        available = [c for c in ensemble_cols if c in summary.columns]
        if available:
            ensemble_return = summary[available].mean(axis=1).dropna()
            result["ensemble_distribution"] = compute_metric_statistics(ensemble_return)

        # Ensemble of dollar-denominated fair value estimates
        price_ensemble_cols = [
            "price_target_mc",
            "price_target_kalman",
            "price_target_prob_weighted",
        ]
        available_price = [c for c in price_ensemble_cols if c in summary.columns]
        if available_price:
            price_ensemble = summary[available_price].mean(axis=1).dropna()
            result["price_ensemble_distribution"] = compute_metric_statistics(
                price_ensemble
            )

    return result


def run_parallel_mcmc_return_analysis(
    pt: pd.DataFrame,
    n_chains: int = 8,
    n_samples: int = 10_000,
    *,
    cache_dir: str = ".cache",
    enable_caching: bool = True,
    cache_ttl_hours: float = 24.0,
) -> dict:
    """
    Run parallel MCMC on price-target expected returns (``implied_return_pt``)
    to get a converged posterior with Gelman-Rubin diagnostic.

    Also runs hierarchical multi-level MCMC by sector and fits a
    Student-t distribution via ``mcmc_student_t`` when the DataFrame
    contains an ``industry`` column.

    v3.7: Supports file-based result caching (same pattern as
    ``run_category_probability_analysis``).
    """
    from finance_ml.ml_workflow.v3.cache import (
        McmcReturnCacheKey,
        build_cache_path,
        dataframe_stable_checksum,
        load_json,
        save_json,
    )

    if pt.empty or "implied_return_pt" not in pt.columns:
        return {}

    data = np.asarray(pt["implied_return_pt"].dropna().values, dtype=float)
    if data.size < 50:
        logger.warning("Parallel MCMC skipped \u2014 insufficient data (%d)", data.size)
        return {}

    # --- Check cache ---
    cache_path = None
    if enable_caching and cache_dir:
        _candidate_id_cols = ["isin", "ticker", "name"]
        _id_cols = [c for c in _candidate_id_cols if c in pt.columns]
        checksum = dataframe_stable_checksum(pt, id_cols=_id_cols)
        key = McmcReturnCacheKey.for_return(
            data_checksum=checksum,
            n_chains=n_chains,
            n_samples=n_samples,
        )
        cache_path = build_cache_path(cache_dir, key.to_filename(), subdir=key.subdir)
        cached = load_json(cache_path, ttl_hours=cache_ttl_hours)
        if isinstance(cached, dict) and cached:
            logger.info("MCMC return analysis loaded from cache (%s)", cache_path.name)
            return cached

    result = parallel_mcmc_chains(data=data, n_chains=n_chains, n_samples=n_samples)
    logger.info(
        "Parallel MCMC: R\u0302=%.4f, converged=%s, posterior mean=%.2f",
        result.get("r_hat", float("nan")),
        result.get("converged", True),
        result.get("posterior_mean", float("nan")),
    )

    # Detailed chain diagnostics
    ci = result.get("ci_95")
    if ci:
        logger.info(
            "  95%% CI: [%.2f, %.2f], posterior std=%.2f",
            ci[0],
            ci[1],
            result.get("posterior_std", float("nan")),
        )
    if result.get("ess_bulk") is not None:
        logger.info(
            "  ESS bulk=%.0f, ESS tail=%.0f",
            result.get("ess_bulk", 0),
            result.get("ess_tail", 0),
        )
    chain_means = result.get("chain_means", [])
    chain_stds = result.get("chain_stds", [])
    if chain_means:
        logger.info(
            "  Chain means: %s",
            ", ".join(f"{m:.2f}" for m in chain_means),
        )
        logger.info(
            "  Chain stds:  %s",
            ", ".join(f"{s:.2f}" for s in chain_stds),
        )

    # Student-t MCMC fit
    try:
        mu_samples, df_samples = mcmc_student_t(data, n_samples=n_samples)
        result["student_t_mu"] = float(np.mean(mu_samples))
        result["student_t_df"] = float(np.mean(df_samples))
        logger.info(
            "  MCMC Student-t: \u03bc=%.2f (\u00b1%.2f), df=%.1f",
            result["student_t_mu"],
            float(np.std(mu_samples)),
            result["student_t_df"],
        )
    except Exception as e:
        logger.debug("MCMC Student-t fit skipped: %s", e)

    # Hierarchical multi-level MCMC by sector
    if "industry" in pt.columns:
        try:
            hier = hierarchical_mcmc_multi_level(
                pt, "implied_return_pt", n_samples=n_samples
            )
            if hier and "levels" in hier:
                result["hierarchical"] = hier
                g = hier.get("global", {})
                logger.info(
                    "  Hierarchical MCMC: global mean=%.2f, std=%.2f, n=%d",
                    g.get("mean", float("nan")),
                    g.get("std", float("nan")),
                    g.get("n_obs", 0),
                )
                for level_name, groups in hier["levels"].items():
                    logger.info(
                        "    Level '%s': %d groups",
                        level_name,
                        len(groups),
                    )
                    for grp, info in sorted(
                        groups.items(),
                        key=lambda x: x[1].get("n_obs", 0),
                        reverse=True,
                    )[:5]:
                        logger.info(
                            "      %s: posterior=%.2f (raw=%.2f), "
                            "shrinkage=%.3f, n=%d, P(>0)=%.1f%%",
                            grp,
                            info["posterior_mean"],
                            info["raw_mean"],
                            info["shrinkage"],
                            info["n_obs"],
                            info.get("prob_positive", 0) * 100,
                        )
        except Exception as e:
            logger.debug("Hierarchical MCMC skipped: %s", e)

    # Run MCMC on dollar-denominated price_target_mc when available
    if "price_target_mc" in pt.columns:
        mc_price_data = np.asarray(pt["price_target_mc"].dropna().values, dtype=float)
        if mc_price_data.size >= 50:
            try:
                mc_price_result = parallel_mcmc_chains(
                    data=mc_price_data, n_chains=n_chains, n_samples=n_samples
                )
                result["price_target_mc_posterior_mean"] = mc_price_result.get(
                    "posterior_mean"
                )
                result["price_target_mc_r_hat"] = mc_price_result.get("r_hat")
            except Exception as e:
                logger.debug("MCMC on price_target_mc skipped: %s", e)

    # --- Save to cache ---
    if cache_path is not None and result:
        try:
            save_json(cache_path, result)
            logger.info("MCMC return analysis cached → %s", cache_path.name)
        except Exception as e:
            logger.debug("Failed to save MCMC return cache: %s", e)

    return result


def run_resampled_posterior_analysis(
    df: pd.DataFrame,
    freq: str = "1QE",
) -> pd.DataFrame:
    """
    Compute Bayesian resampled return posteriors from historical price snapshots.

    Uses BayesianTechnicalResampler to derive per-stock posterior return
    distributions, providing a fifth model signal for cross-model alignment.

    Note: Upstream models now produce price_target_mc and price_target_kalman
    (dollar-denominated fair value estimates) for ensemble averaging.
    """
    # v3.9: Ensure unique input for posterior resampler
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df
    try:
        result_df, idata = resampled_posterior_returns(
            df, freq=freq, n_posterior_samples=5000, n_chains=8
        )
        if not result_df.empty:
            logger.info(
                "Resampled posterior returns: %d stocks, mean posterior=%.2f%%",
                len(result_df),
                (
                    result_df["posterior_mean"].mean() * 100
                    if "posterior_mean" in result_df.columns
                    else 0
                ),
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


def _trim_screen_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Trim a screening DataFrame to identifier + screening-relevant columns.

    Screening functions return the full ``df_all`` row (400+ columns from 17
    merged feature views).  PostgreSQL has an 8 KB row-size limit, so we keep
    only the identifier columns plus the feature columns actually referenced
    by the screening functions and a handful of key market-data columns.
    """
    if df.empty:
        return df

    id_cols = set(load_identifier_columns())

    # Columns referenced by analytics.screening functions + key market data
    _SCREEN_FEATURE_COLS = {
        # Quality / composite
        "piotroski_f_score",
        "combined_distress_risk_score",
        "eps_trajectory_score",
        "fcf_positive_years",
        "quality_momentum_score",
        "quality_momentum",
        "composite_score",
        "dilution_score",
        "debt_deleveraging",
        "secular_trend_flag",
        # Earnings quality
        "earnings_quality_composite",
        "eps_adjustment_pct",
        "gaap_positive_revision_flag",
        "net_income_positive_years",
        # Value
        "p_e_ratio",
        "p_e_vs_3y_avg",
        "ev_ebitda_vs_3y_avg",
        "peg_ratio",
        "price_to_tangible_book",
        "fcf_yield",
        "expected_upside_pt",
        # Growth / momentum
        "eps_yoy_growth",
        "fcf_est_cagr_5y",
        "price_momentum_1y",
        "total_return_ytd",
        "total_return_5y",
        "long_term_trend_score",
        # GARP
        "operating_leverage_score",
        "beta_stability_score",
        # Dividend
        "dividend_payout_ratio",
        "dividend_streak",
        "fcf_dividend_coverage",
        "dividend_growth_expectation",
        # Financial health
        "current_ratio",
        "debt_to_equity",
        "accounting_quality_score",
        "effective_tax_rate_ltm",
        "rnd_intensity_ltm",
        # Valuation reversion
        "volatility_1y",
        "volatility_compression",
        "beta_1y",
        # Integrity / other
        "merger_impact_ratio",
        "net_buyback_flag",
        # Sector-relative
        "analyst_rating",
        # Key market data
        "market_cap",
        "enterprise_value",
        "last_price",
        "price_target",
        "shares_outstanding",
        "volume_shrs",
    }

    keep = [c for c in df.columns if c in id_cols or c in _SCREEN_FEATURE_COLS]
    logger.debug(
        "Trimmed screening DataFrame from %d to %d columns for export",
        len(df.columns),
        len(keep),
    )
    return df[keep]


def _trim_credit_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Trim credit risk DataFrame to identifier + credit-relevant columns for export.

    The full credit DataFrame inherits 1000+ columns from the source materialized
    view, which exceeds PostgreSQL's 8 KB row size limit.  Keep only identifier
    columns and the columns actually produced or consumed by the credit risk model.
    """
    if df.empty:
        return df

    id_cols = set(load_identifier_columns())

    _CREDIT_EXPORT_COLS = {
        # CreditRiskProbabilityModel outputs
        "beta_stability_score",
        "combined_distress_score",
        "distress_probability",
        "liquidity_stress_score",
        "cash_runway_months",
        "altman_z_score",
        "altman_z_trend",
        "interest_coverage",
        "quick_ratio",
        "risk_level",
        "ci_lower",
        "ci_upper",
        "debt_3y_cagr",
        "debt_maturity_risk",
        "balance_sheet_strength",
        "wc_efficiency_score",
        "distress_risk_score",
        "data_quality_score",
        # MCMC enrichment
        "mcmc_distress_probability",
        "mcmc_ci_lower",
        "mcmc_ci_upper",
        "sector_z_posterior_mean",
        # Ruin probability outputs
        "expected_drift",
        "volatility",
        "wealth_buffer",
        "ruin_probability",
        "survival_probability",
        # v3.9 / v3.10 diagnostic + macro parity (CreditRiskResult §2.1, T-A, T-B)
        "tail_df",
        "cond_volatility",
        "cvar_5",
        "posterior_ess_bulk",
        "posterior_ess_tail",
        "r_hat",
        "schema_version",
        "macro_loading_yield_curve_10y2y",
        "macro_loading_vix",
        "macro_loading_dxy",
        "macro_loading_hy_oas",
        # Key market data for context
        "market_cap",
        "enterprise_value",
        "last_price",
        "price_target",
        "price_target_high",
        "price_target_low",
        "price_target_median",
        "volume_shrs",
        "shares_outstanding",
    }

    keep = [c for c in df.columns if c in id_cols or c in _CREDIT_EXPORT_COLS]
    logger.debug(
        "Trimmed credit risk DataFrame from %d to %d columns for export",
        len(df.columns),
        len(keep),
    )
    return df[keep]


def _trim_anomaly_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Trim accounting anomaly DataFrame to identifier + anomaly-relevant columns for export.

    The full anomaly DataFrame inherits 1000+ columns from the source materialized
    view, which exceeds PostgreSQL's 8 KB row size limit.  Keep only identifier
    columns and the columns actually produced by the anomaly detection model.
    """
    if df.empty:
        return df

    id_cols = set(load_identifier_columns())

    # Keep columns that are anomaly model outputs (by naming convention + explicit list)
    keep = []
    for c in df.columns:
        if c in id_cols:
            keep.append(c)
        elif (
            c.startswith("accounting_anomaly")
            or c.endswith("_z_robust")
            or c.endswith("_anomaly_flag")
            or c.endswith("_dist_name")
            or c.endswith("_dist_pvalue")
            or c
            in {
                "anomaly_feature_count",
                "anomaly_severity_score",
                "anomaly_risk_rank",
                "sector_anomaly_percentile",
                "multi_flag_alert",
                "anomaly_conditional_probability",
                "anomaly_posterior_mean",
                "anomaly_posterior_std",
                "anomaly_ci_lower",
                "anomaly_ci_upper",
                "sector_posterior_mean",
                "anomaly_posterior_location",
                "mahalanobis_distance",
                "sector_relative_anomaly",
                "benford_chi2_pvalue",
                "quality_frequency_score",
                "repeat_offender_flag",
                # v3.10 anomaly decomposition + diagnostics (§8.1 / §9.1 / §9.2)
                "flag_count_posterior_mean",
                "flag_count_ci_low",
                "flag_count_ci_high",
                "magnitude_posterior_mean",
                "combined_anomaly_score",
                "dominant_flag_category",
                "tail_df",
                "cond_volatility",
                "r_hat",
                "ess_bulk",
                "ess_tail",
                "schema_version",
                # v3.5 enhanced anomaly metrics
                "accumulated_deficit_flag",
                "negative_wc_flag",
                "wc_deteriorating_flag",
                "intangibles_growth_flag",
                "inventory_buildup_flag",
                "inventory_reduction_flag",
                "has_goodwill_impairment",
                "has_asset_writedown",
                "has_restructuring",
                "has_goodwill_impairment_ltm",
                "impairment_risk_score",
                "revenue_accelerating_flag",
                "overinvestment_flag",
                "recent_acquisition_flag",
                "high_rnd_intensity_flag",
                "has_unusual_items_flag",
                "low_tax_flag",
                "layoff_risk_flag",
                "analyst_bearish_pct",
                "debt_maturity_risk",
                # Key market/context data
                "market_cap",
                "enterprise_value",
                "last_price",
                "price_target",
                "shares_outstanding",
            }
        ):
            keep.append(c)

    logger.debug(
        "Trimmed anomaly DataFrame from %d to %d columns for export",
        len(df.columns),
        len(keep),
    )
    return df[keep]


def _export_single_table(
    df: pd.DataFrame,
    table: str,
    *,
    _previous_hashes: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """
    Export a single DataFrame to the analytics schema.

    Task 7.2: Skips export if the DataFrame hash matches the previous run.

    Returns
    -------
    tuple[str, str | None]
        (table_name, destination) or (table_name, None) if skipped/failed.
    """
    try:
        # Task 7.2: Selective export — skip unchanged tables
        if _previous_hashes is not None:
            current_hash = dataframe_hash(df)
            if _previous_hashes.get(table) == current_hash:
                logger.info("Export skipped (unchanged): analytics.%s", table)
                return table, f"analytics.{table} (cached)"
            _previous_hashes[table] = current_hash

        reordered_df = reorder_with_identifiers(df)
        export_config = ExportConfig(table_name=table)
        export_to_db(reordered_df, export_config)
        logger.info("Exported %d rows → analytics.%s", len(df), table)
        return table, f"analytics.{table}"
    except Exception as e:
        logger.warning("Export failed for %s: %s", table, e)
        return table, None


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
    max_workers: int = 4,
) -> dict[str, str]:
    # Note: mc now contains 'price_target_mc' (MC-simulated fair value price)
    # and kal now contains 'price_target_kalman' (Kalman-filtered fair value price)
    """
    Export all expected returns analytics to the ``analytics`` schema.

    v3.0: Added dividend safety and screening results exports.
    v3.2: Added accounting anomaly analysis export.
    v3.6: Task 7.1 — Parallelized exports via ThreadPoolExecutor.
          Task 7.2 — Selective export skips unchanged DataFrames.

    Parameters
    ----------
    mc : pd.DataFrame
        Monte Carlo simulation results.
    pt : pd.DataFrame
        Price target achievement results.
    kal : pd.DataFrame
        Kalman filter results.
    tri : pd.DataFrame
        Tri-model alignment results.
    strong : pd.DataFrame
        Strong consensus picks.
    beat : pd.DataFrame
        Earnings beat probability results.
    summary : pd.DataFrame or None, optional
        Expected returns summary.
    credit : pd.DataFrame or None, optional
        Credit risk analysis results.
    div_safety : pd.DataFrame or None, optional
        Dividend safety analysis results.
    anomaly_results : pd.DataFrame or None, optional
        Accounting anomaly analysis results.
    screens : dict[str, pd.DataFrame] or None, optional
        Stock screening results keyed by screen name.
    output_dir : str, default 'outputs'
        Directory for output artifacts.
    max_workers : int, default 4
        Maximum parallel export threads.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    exports: dict[str, str] = {}

    # Task 7.2: Track hashes for selective export
    _previous_hashes: dict[str, str] = {}
    _hash_file = Path(output_dir) / ".export_hashes.json"
    if _hash_file.exists():
        try:
            _previous_hashes = json.loads(_hash_file.read_text())
        except Exception:
            pass

    _EXPORT_PAIRS: list[tuple[pd.DataFrame | None, str]] = [
        (mc, "monte_carlo_simulation"),
        (pt, "price_target_achievement"),
        (kal, "kalman_filtered_price_targets"),
        (tri, "expected_returns_tri_model"),
        (strong, "strong_consensus_picks"),
        (beat, "earnings_probability_analysis"),
        (summary, "expected_returns_summary"),
        (
            _trim_credit_for_export(credit)
            if credit is not None and not credit.empty
            else credit,
            "credit_risk_analysis",
        ),
        (div_safety, "dividend_safety_analysis"),
        (
            _trim_anomaly_for_export(anomaly_results)
            if anomaly_results is not None and not anomaly_results.empty
            else anomaly_results,
            "accounting_anomaly_analysis",
        ),
    ]

    # Add screening results
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
    if screens:
        for screen_name, screen_df in screens.items():
            table = _SCREEN_TABLE_MAP.get(screen_name, f"screen_{screen_name}")
            trimmed = _trim_screen_for_export(screen_df)
            _EXPORT_PAIRS.append((trimmed, table))

    # Filter to non-empty DataFrames
    valid_pairs: list[tuple[pd.DataFrame, str]] = [
        (df, table) for df, table in _EXPORT_PAIRS if df is not None and not df.empty
    ]

    # Task 7.1: Parallel export via ThreadPoolExecutor
    if max_workers > 1 and len(valid_pairs) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _export_single_table,
                    df,
                    table,
                    _previous_hashes=_previous_hashes,
                ): table
                for df, table in valid_pairs
            }
            for future in as_completed(futures):
                table_name, dest = future.result()
                if dest:
                    exports[table_name] = dest
    else:
        for df, table in valid_pairs:
            table_name, dest = _export_single_table(
                df,
                table,
                _previous_hashes=_previous_hashes,
            )
            if dest:
                exports[table_name] = dest

    # Persist hashes for next run (Task 7.2)
    try:
        _hash_file.write_text(json.dumps(_previous_hashes, indent=2))
    except Exception:
        pass

    return exports


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Pipeline Step Functions (Task 1.1)
# ═══════════════════════════════════════════════════════════════════════════════


def _step_load_data(r: PipelineResult, cfg: PipelineConfig) -> PipelineResult:
    """Step 1: Load feature data from materialized views.

    v3.7: When ``cfg.prefer_materialized_view`` is True (default), the pipeline
    loads ``mv_all_stock_features`` first as the primary dataset and uses it for
    ``r.df``, ``r.df_all``, and ``r.df_features``.  A lightweight
    ``mv_equities`` load is still performed to supply historical snapshot
    columns (``price_*_ago``, ``price_target_*_ago``) that are not present in
    ``mv_all_stock_features``.  The redundant ``load_all_stock_features`` call
    (17-view merge) is skipped when the materialized view loads successfully.

    When the flag is False **or** when the MV load returns an empty DataFrame,
    the original three-source loading pattern is used as a fallback.
    """
    r = PipelineResult()

    # Initialize the schema-driven feature catalog (singleton)
    catalog = get_feature_catalog()
    if catalog._loaded:
        _log_and_print(
            f"✓ FeatureViewCatalog: {len(catalog.category_columns)} categories, "
            f"{len(catalog.view_columns)} views"
        )
    else:
        _log_and_print(
            "⚠️ FeatureViewCatalog: using fallback column lists (DB unavailable)"
        )

    if cfg.prefer_materialized_view:
        # --- Primary path: mv_all_stock_features as single source of truth ---
        r.df_features = load_analytics_table()

        if not r.df_features.empty:
            _log_and_print(
                f"✓ Loaded mv_all_stock_features: {len(r.df_features):,} stocks "
                f"× {len(r.df_features.columns)} features"
            )

            # v3.9: Strict deduplication at source
            r.df_features = r.df_features.drop_duplicates(subset="isin")
            _log_and_print(f"  Deduplicated to {len(r.df_features):,} unique isins")

            # Use mv_all_stock_features as the main working datasets
            r.df = r.df_features
            r.df_all = r.df_features
            r.catalog = catalog
            r.feature_df = r.df_features

            # Reconcile feature categories against the full column set
            global _feature_categories_cache
            feature_categories = get_feature_categories()
            feature_categories = reconcile_feature_categories(
                feature_categories, set(r.df_features.columns)
            )
            _feature_categories_cache = feature_categories

            # Lightweight mv_equities load for historical snapshot columns only
            try:
                df_equities, r.id_coords = load_expected_returns_data()
                if not df_equities.empty:
                    _log_and_print(
                        f"✓ Loaded mv_equities (historical cols): "
                        f"{len(df_equities):,} stocks × {len(df_equities.columns)} features"
                    )
                    # Merge historical columns not present in mv_all_stock_features
                    hist_cols = [
                        c
                        for c in df_equities.columns
                        if c not in r.df_features.columns and c != "isin"
                    ]
                    if (
                        hist_cols
                        and "isin" in df_equities.columns
                        and "isin" in r.df_features.columns
                    ):
                        # v3.9: Deduplicate historical side before merge
                        hist_side = df_equities[["isin"] + hist_cols].drop_duplicates(
                            subset="isin"
                        )
                        r.df_features = r.df_features.merge(
                            hist_side,
                            on="isin",
                            how="left",
                        )
                        r.df = r.df_features
                        r.df_all = r.df_features
                        r.catalog = catalog
                        r.feature_df = r.df_features
                        _log_and_print(
                            f"  Merged {len(hist_cols)} historical columns from mv_equities"
                        )
            except Exception as e:
                _log_and_print(f"  mv_equities load for historical cols skipped: {e}")
        else:
            # Fallback: MV empty — use original 3-source pattern
            _log_and_print(
                "⚠️ mv_all_stock_features not loaded — falling back to 3-source load"
            )
            r.df, r.id_coords = load_expected_returns_data()
            if r.df.empty:
                _log_and_print("✗ No data loaded from mv_equities. Check DB_URL.")
                return r
            r.df = r.df.drop_duplicates(subset="isin")
            _log_and_print(
                f"✓ Loaded mv_equities: {len(r.df):,} unique stocks × {len(r.df.columns)} features"
            )

            r.df_all, r.view_specs = load_all_stock_features()
            if not r.df_all.empty:
                r.df_all = r.df_all.drop_duplicates(subset="isin")
                _log_and_print(
                    f"✓ Loaded feature views: {len(r.df_all):,} unique stocks "
                    f"× {len(r.df_all.columns)} features"
                )
            else:
                _log_and_print(
                    "⚠️ Feature views not loaded — screening will use mv_equities"
                )
                r.df_all = r.df.copy()
    else:
        # --- Legacy path: 3-source loading (prefer_materialized_view=False) ---
        r.df, r.id_coords = load_expected_returns_data()
        if r.df.empty:
            _log_and_print("✗ No data loaded from mv_equities. Check DB_URL.")
            return r
        r.df = r.df.drop_duplicates(subset="isin")
        _log_and_print(
            f"✓ Loaded mv_equities: {len(r.df):,} unique stocks "
            f"× {len(r.df.columns)} features"
        )

        r.df_all, r.view_specs = load_all_stock_features()
        if not r.df_all.empty:
            r.df_all = r.df_all.drop_duplicates(subset="isin")
            _log_and_print(
                f"✓ Loaded feature views: {len(r.df_all):,} unique stocks "
                f"× {len(r.df_all.columns)} features"
            )
            if r.view_specs:
                _log_and_print(f"  View specs loaded: {len(r.view_specs)} views")
        else:
            _log_and_print(
                "⚠️ Feature views not loaded — screening will use mv_equities"
            )
            r.df_all = r.df.copy()

        r.df_features = load_analytics_table()
        if not r.df_features.empty:
            r.df_features = r.df_features.drop_duplicates(subset="isin")
            _log_and_print(
                f"✓ Loaded mv_all_stock_features: {len(r.df_features):,} unique stocks "
                f"× {len(r.df_features.columns)} features"
            )
        else:
            _log_and_print("⚠️ mv_all_stock_features not loaded — continuing without it")

    # Step 1a: Load schema metadata
    try:
        r.schema_metadata = load_equities_schema_metadata_from_db()
        r.feature_registry = load_feature_registry_metadata_from_db()
        if r.schema_metadata is not None:
            _log_and_print(
                f"  Schema metadata: {len(r.schema_metadata.column_names)} columns"
            )
        if r.feature_registry is not None:
            _log_and_print(
                f"  Feature registry: {len(r.feature_registry.function_names)} functions"
            )
    except Exception as e:
        _log_and_print(f"  Schema metadata unavailable: {e}")

    try:
        # Ensure 'ticker' column exists for EquitiesMaterializedViewSpec
        # validation — the project convention uses 'isin' as the primary
        # identifier, so we derive 'ticker' from 'isin' when absent.
        _spec_df = _ensure_ticker_from_isin(r.df)
        r.mv_equities_spec = load_mv_equities_spec_from_db(_spec_df)
        if r.mv_equities_spec is not None:
            _log_and_print(
                f"  mv_equities spec: {len(r.mv_equities_spec.price_columns)} price, "
                f"{len(r.mv_equities_spec.financial_columns)} financial, "
                f"{len(r.mv_equities_spec.historical_price_columns)} historical cols"
            )
    except Exception as e:
        _log_and_print(f"  mv_equities spec unavailable: {e}")

    if r.id_coords is None:
        try:
            r.id_coords = load_identifier_coordinates_from_db()
            _log_and_print(
                f"  IdentifierCoordinates (from DB): {len(r.id_coords.isins)} isins"
            )
        except Exception as e:
            logger.debug("load_identifier_coordinates_from_db failed: %s", e)

    # Ensure feature_df and catalog are consistently set for all paths
    r.feature_df = r.df_features if not r.df_features.empty else r.df_all
    r.catalog = catalog

    # Ensure 'ticker' column is present on all working DataFrames.
    # The project uses 'isin' as the canonical identifier, but several
    # downstream components (EquitiesMaterializedViewSpec, cache checksum
    # helpers, InferenceData coordinate builders) require 'ticker'.
    for _attr in ("df", "df_all", "df_features", "df_enriched", "feature_df"):
        _frame = getattr(r, _attr, None)
        if _frame is not None and isinstance(_frame, pd.DataFrame) and not _frame.empty:
            setattr(r, _attr, _ensure_ticker_from_isin(_frame))

    # Step 1b: Historical target drift enrichment
    _log_and_print("📦 Step 1b: Pre-computing historical target drift enrichment...")
    _log_and_print("-" * 80)
    hist_available = _resolve_available_historical_cols(r.df)
    _log_historical_coverage(hist_available)
    r.df_enriched = _enrich_with_historical_target_drift(r.df.copy(), hist_available)
    _log_and_print(
        f"✓ Historical drift enrichment complete ({len(r.df_enriched.columns) - len(r.df.columns)} derived columns)"
    )

    return r


def _step_monte_carlo(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 2: Monte Carlo simulation."""
    try:
        r.mc = run_monte_carlo_analysis(
            r.df_enriched,
            n_simulations=cfg.mc_simulations,
            max_stocks=cfg.mc_max_stocks,
        )
        if not r.mc.empty and _has_required_columns(
            r.mc, ["implied_return_mc", "prob_positive_upside"], "Monte Carlo"
        ):
            _log_and_print(f"✓ {len(r.mc):,} stocks simulated")
            _log_and_print(f"  Mean upside:  {r.mc['implied_return_mc'].mean():.1f}%")
            _log_and_print(
                f"  Median upside: {r.mc['implied_return_mc'].median():.1f}%"
            )
            _log_and_print(
                f"  Positive prob (mean): {r.mc['prob_positive_upside'].mean():.1f}%"
            )

            if "price_target_mc" in r.mc.columns:
                valid_mc = r.mc.dropna(subset=["price_target_mc", "last_price"])
                if not valid_mc.empty:
                    mean_price = valid_mc["last_price"].mean()
                    mean_target = valid_mc["price_target_mc"].mean()
                    implied_return = (
                        safe_divide(mean_target, mean_price, default=1.0) - 1
                    ) * 100
                    _log_and_print(
                        f"  Monte Carlo targets ({len(valid_mc):,} stocks): implied return={implied_return:.1f}%"
                    )

            mc_stats = compute_model_detailed_statistics(
                r.mc,
                "Monte Carlo",
                [
                    "implied_return_mc",
                    "prob_positive_upside",
                    "var_5_pct",
                    "risk_reward_ratio",
                    "price_target_mc",
                ],
            )
            print_model_statistics(mc_stats, "Monte Carlo Simulation", top_n_sectors=25)

            dist_analytics = compute_return_distribution_analytics(r.mc)
            if dist_analytics.get("mc_distribution"):
                d = dist_analytics["mc_distribution"]
                _log_and_print(
                    f"\n  📐 Best-fit distribution: {d['name']} (AIC={d['aic']:.1f}, KS p={d['ks_pvalue']:.3f})"
                )
            if dist_analytics.get("risk_metrics"):
                rm = dist_analytics["risk_metrics"]
                _log_and_print(
                    f"\n  📉 VaR 1%: {rm['var_1_pct']:.1f}%  |  CVaR 5%: {rm['cvar_5_pct']:.1f}%"
                )
                _log_and_print(
                    f"     Downside deviation: {rm['downside_deviation']:.2f}"
                )
                if rm.get("gain_loss_ratio"):
                    _log_and_print(f"     Gain/Loss ratio: {rm['gain_loss_ratio']:.2f}")
            if dist_analytics.get("opportunity_tiers"):
                t = dist_analytics["opportunity_tiers"]
                _log_and_print(
                    f"\n  🏷️  Tiers: High-conviction={t['high_conviction']}, "
                    f"Moderate={t['moderate']}, Speculative={t['speculative']}, Avoid={t['avoid']}"
                )

    except Exception as e:
        logger.error("Step 2 (Monte Carlo) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 2 failed: {e}", logging.ERROR)


def _step_price_target(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 3: Price target achievement model."""
    try:
        r.pt = run_price_target_achievement(r.df_enriched, feature_df=r.df_features)
        if not r.pt.empty and _has_required_columns(
            r.pt,
            ["achievement_probability", "implied_return_pt", "expected_upside_pt"],
            "Price Target",
        ):
            _log_and_print(f"✓ {len(r.pt):,} stocks analyzed")
            _log_and_print(
                f"  Mean achievement prob: {r.pt['achievement_probability'].mean():.3f}"
            )
            _log_and_print(
                f"  Mean prob-weighted return: {r.pt['implied_return_pt'].mean():.1f}%"
            )

            r.pt = compute_price_target_prob_weighted(r.pt, r.df)
            if "price_target_prob_weighted" in r.pt.columns:
                valid_pt = r.pt.dropna(
                    subset=["price_target_prob_weighted", "last_price"]
                )
                if not valid_pt.empty:
                    mean_price = valid_pt["last_price"].mean()
                    mean_target = valid_pt["price_target_prob_weighted"].mean()
                    implied_return = (
                        safe_divide(mean_target, mean_price, default=1.0) - 1
                    ) * 100
                    _log_and_print(
                        f"  Prob-weighted targets ({len(valid_pt):,} stocks): implied return={implied_return:.1f}%"
                    )

            pt_stats = compute_model_detailed_statistics(
                r.pt,
                "Price Target Achievement",
                [
                    "achievement_probability",
                    "implied_return_pt",
                    "analyst_conviction",
                    "eps_revision_momentum",
                    "price_target_prob_weighted",
                    "bullish_pct",
                    "expected_upside_pt",
                    "price_target_spread_pct",
                ],
            )
            print_model_statistics(pt_stats, "Price Target Achievement")

    except Exception as e:
        logger.error("Step 3 (Price Target) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 3 failed: {e}", logging.ERROR)


def _step_kalman(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 4: Kalman-filtered price targets."""
    try:
        r.kal = run_kalman_filter(r.df_enriched)
        if not r.kal.empty and _has_required_columns(
            r.kal, ["implied_return_kalman"], "Kalman"
        ):
            _log_and_print(f"✓ {len(r.kal):,} stocks filtered")
            _log_and_print(
                f"  Mean implied_return_kalman: {r.kal['implied_return_kalman'].mean():.1f}%"
            )

            if "price_target_kalman" in r.kal.columns:
                valid_kal = r.kal.dropna(
                    subset=["price_target_kalman", "original_price"]
                )
                if not valid_kal.empty:
                    mean_price = valid_kal["original_price"].mean()
                    mean_target = valid_kal["price_target_kalman"].mean()
                    implied_return = (
                        safe_divide(mean_target, mean_price, default=1.0) - 1
                    ) * 100
                    _log_and_print(
                        f"  Kalman targets ({len(valid_kal):,} stocks): implied return={implied_return:.1f}%"
                    )

            kal_stats = compute_model_detailed_statistics(
                r.kal,
                "Kalman Filter",
                [
                    "implied_return_kalman",
                    "price_target_kalman",
                    "kalman_variance",
                    "signal_strength",
                ],
            )
            print_model_statistics(kal_stats, "Kalman Filter")

    except Exception as e:
        logger.error("Step 4 (Kalman) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 4 failed: {e}", logging.ERROR)


def _step_earnings_beat(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 5: Bayesian earnings beat analysis."""
    try:
        r.beat = run_earnings_beat_analysis(
            r.df_features if not r.df_features.empty else r.df, feature_df=r.df_features
        )
        if not r.beat.empty and _has_required_columns(
            r.beat, ["posterior_beat_prob"], "Earnings Beat"
        ):
            _log_and_print(f"✓ {len(r.beat):,} stocks analyzed")
            beat_prob_col = (
                "prob_beat_given_momentum"
                if "prob_beat_given_momentum" in r.beat.columns
                else "posterior_beat_prob"
            )
            _log_and_print(f"  Mean P(beat): {r.beat[beat_prob_col].mean():.3f}")
            if "beat_classification" in r.beat.columns:
                likely = (r.beat["beat_classification"] == "likely_beat").sum()
                _log_and_print(f"  Classified as 'likely_beat': {likely}")

            beat_stats = compute_model_detailed_statistics(
                r.beat,
                "Earnings Beat",
                [
                    "posterior_beat_prob",
                    "confidence_score",
                    "analyst_conviction",
                    "eps_revision_momentum",
                ],
            )
            print_model_statistics(beat_stats, "Earnings Beat")

    except Exception as e:
        logger.error("Step 5 (Earnings Beat) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5 failed: {e}", logging.ERROR)


def _step_anomaly_detection(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 5b: Accounting anomaly detection & analytics."""
    from probabilistic_ml_model.pipeline_runners import cache_mcmc_result

    try:
        r.anomaly_results = run_accounting_anomaly_analysis(
            r.df,
            feature_df=r.df_features,
            anomaly_z_threshold=cfg.anomaly_z_threshold,
            n_mcmc_samples=cfg.mcmc_samples,
            burn_in=cfg.mcmc_burn_in,
        )

        if (
            not r.anomaly_results.empty
            and "accounting_anomaly_score" in r.anomaly_results.columns
        ):
            _log_anomaly_diagnostics(r.anomaly_results)

            # Export accounting anomaly analysis to DB
            try:
                anomaly_export_cols = ["isin"] + [
                    c
                    for c in r.anomaly_results.columns
                    if c.startswith("accounting_anomaly")
                    or c.endswith("_z_robust")
                    or c.endswith("_anomaly_flag")
                    or c.endswith("_dist_name")
                    or c.endswith("_dist_pvalue")
                    or c
                    in (
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
                id_cols = load_identifier_columns()
                for id_col in id_cols:
                    if (
                        id_col in r.anomaly_results.columns
                        and id_col not in anomaly_export_cols
                    ):
                        anomaly_export_cols.insert(1, id_col)

                anomaly_export = r.anomaly_results[
                    [c for c in anomaly_export_cols if c in r.anomaly_results.columns]
                ].copy()

                if not anomaly_export.empty:
                    anomaly_export = reorder_with_identifiers(anomaly_export)
                    anomaly_cfg = ExportConfig(table_name="accounting_anomaly_analysis")
                    export_to_db(anomaly_export, anomaly_cfg)
                    _log_and_print(
                        f"  ✓ Exported {len(anomaly_export):,} rows → analytics.accounting_anomaly_analysis"
                    )
            except Exception as e:
                logger.warning("Accounting anomaly DB export failed: %s", e)

            _log_and_print(
                f"✓ Accounting anomaly analysis complete: {len(r.anomaly_results):,} stocks"
            )

            # Cache MCMC anomaly results
            if cfg.enable_result_caching or cfg.enable_mcmc_caching:
                try:
                    cache_mcmc_result(
                        r.anomaly_results,
                        "accounting_anomaly",
                        cfg.mcmc_samples,
                        cache_dir=cfg.cache_dir,
                    )
                except Exception as e:
                    logger.debug("Failed to cache accounting anomaly results: %s", e)
        else:
            _log_and_print("  ⚠️ No accounting anomaly features available")

    except Exception as e:
        logger.error("Step 5b (Accounting Anomaly) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5b failed: {e}", logging.ERROR)


def _step_credit_dividend(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 5c: Credit risk & dividend safety analysis."""
    from probabilistic_ml_model.pipeline_runners import cache_mcmc_result

    try:
        r.credit = run_credit_risk_analysis(
            r.df,
            feature_df=r.df_features,
            n_mcmc_samples=cfg.mcmc_samples,
            burn_in=cfg.mcmc_burn_in,
        )
        if not r.credit.empty:
            high_risk = (
                r.credit["risk_level"].isin(["High", "Distressed"]).sum()
                if "risk_level" in r.credit.columns
                else 0
            )
            _log_and_print(
                f"✓ Credit risk: {len(r.credit):,} stocks, {high_risk} high/distressed"
            )
            if "ruin_probability" in r.credit.columns:
                _log_and_print(
                    f"  Mean ruin probability: {r.credit['ruin_probability'].mean():.3f}"
                )

            # Merge anomaly columns into credit
            if not r.anomaly_results.empty and "isin" in r.anomaly_results.columns:
                anom_cols = [
                    c
                    for c in r.anomaly_results.columns
                    if c != "isin" and c not in r.credit.columns
                ]
                if anom_cols:
                    r.credit = r.credit.merge(
                        r.anomaly_results[["isin"] + anom_cols],
                        on="isin",
                        how="left",
                    )
                    _log_and_print(
                        f"  Merged {len(anom_cols)} anomaly columns into credit risk DataFrame"
                    )

            # Cache MCMC credit risk results
            if cfg.enable_result_caching or cfg.enable_mcmc_caching:
                try:
                    cache_mcmc_result(
                        r.credit,
                        "credit_risk",
                        cfg.mcmc_samples,
                        cache_dir=cfg.cache_dir,
                    )
                except Exception as e:
                    logger.debug("Failed to cache credit risk results: %s", e)

        r.div_safety = run_dividend_safety_analysis(
            r.df_features if not r.df_features.empty else r.df_all,
            feature_df=r.df_features,
            n_mcmc_samples=cfg.mcmc_samples,
            burn_in=cfg.mcmc_burn_in,
        )
        if not r.div_safety.empty:
            at_risk = (
                (r.div_safety["risk_category"] == "At Risk").sum()
                if "risk_category" in r.div_safety.columns
                else 0
            )
            _log_and_print(
                f"✓ Dividend safety: {len(r.div_safety):,} stocks, {at_risk} at risk"
            )

            # Cache MCMC dividend safety results
            if cfg.enable_result_caching or cfg.enable_mcmc_caching:
                try:
                    cache_mcmc_result(
                        r.div_safety,
                        "dividend_safety",
                        cfg.mcmc_samples,
                        cache_dir=cfg.cache_dir,
                    )
                except Exception as e:
                    logger.debug("Failed to cache dividend safety results: %s", e)

    except Exception as e:
        logger.error("Step 5c (Credit/Dividend) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5c failed: {e}", logging.ERROR)


def _step_screening(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 5d: Stock screening strategies."""
    try:
        try:
            r.df_all = analyze_employee_productivity_frontier(r.df_all)
            if "productivity_frontier_score" in r.df_all.columns:
                _log_and_print("  ✓ Productivity frontier scores added")
        except Exception as e:
            logger.debug("Productivity frontier enrichment skipped: %s", e)

        try:
            lag_result = analyze_reporting_lag_sentiment(r.df_all)
            if lag_result.get("sample_size", 0) > 0:
                p_val = lag_result["p_value"]
                if lag_result["hypothesis_confirmed"]:
                    hyp_label = "confirmed"
                elif p_val < 0.10:
                    hyp_label = "marginally significant (p < 0.10)"
                else:
                    hyp_label = "not confirmed"
                _log_and_print(
                    f"   Reporting lag sentiment: corr={lag_result['correlation']:.3f}, "
                    f"p={p_val:.4f}, hypothesis={hyp_label}"
                )
        except Exception as e:
            logger.debug("Reporting lag analysis skipped: %s", e)

        r.screens = run_stock_screening(
            r.df_features if not r.df_features.empty else r.df_all,
            min_pct=cfg.screening_min_pct,
        )
        for name, screen_df in r.screens.items():
            if not screen_df.empty:
                _log_and_print(f"  ✓ {name}: {len(screen_df):,} stocks")

    except Exception as e:
        logger.error("Step 5d (Screening) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5d failed: {e}", logging.ERROR)


def _step_resampled_posterior(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 5e: Resampled Bayesian posterior returns."""
    try:
        r.resampled_posterior = run_resampled_posterior_analysis(r.df)
        if not r.resampled_posterior.empty:
            _log_and_print(
                f"  ✓ Resampled posteriors: {len(r.resampled_posterior):,} stocks"
            )
            if "posterior_mean" in r.resampled_posterior.columns:
                _log_and_print(
                    f"  Mean posterior return: {r.resampled_posterior['posterior_mean'].mean() * 100:.2f}%"
                )
        else:
            _log_and_print("  ⚠️ Resampled posterior returns: no results")

    except Exception as e:
        logger.error("Step 5e (Resampled Posterior) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 5e failed: {e}", logging.ERROR)


def _step_alignment(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 6: Cross-model alignment."""
    try:
        r.tri = build_tri_model_alignment(r.mc, r.kal, r.pt, bullish_return_threshold=cfg.bullish_return_threshold)
        r.strong = extract_strong_consensus(r.tri)
        r.quad = build_quad_model_alignment(r.tri, r.beat, beat_threshold=cfg.beat_threshold,
                                            credit=r.credit if not r.credit.empty else None,
                                            div_safety=r.div_safety if not r.div_safety.empty else None,
                                            anomaly=r.anomaly_results if not r.anomaly_results.empty else None,
                                            anomaly_severity_threshold=cfg.anomaly_severity_threshold)

        if not r.tri.empty:
            _log_and_print(f"  Tri-model coverage: {len(r.tri):,} stocks")
            for label in _SIGNAL_LABELS.values():
                cnt = (r.tri["signal"] == label).sum()
                _log_and_print(f"    {label}: {cnt}")
            _log_and_print(f"  Strong consensus picks: {len(r.strong)}")

        if not r.quad.empty:
            # Two-tier consensus reporting
            if "full_consensus" in r.quad.columns:
                full = r.quad["full_consensus"].sum()
                _n_models = 4
                if not r.credit.empty:
                    _n_models += 1
                if not r.div_safety.empty:
                    _n_models += 1
                if not r.anomaly_results.empty:
                    _n_models += 1
                _log_and_print(
                    f"  Ensemble alignment ({_n_models} models): {len(r.quad):,} stocks, "
                    f"full consensus (4/4 dir + ≥2/3 risk): {full}"
                )
            else:
                _n_models = 4
                if not r.credit.empty:
                    _n_models += 1
                if not r.div_safety.empty:
                    _n_models += 1
                if not r.anomaly_results.empty:
                    _n_models += 1
                full = (r.quad["quad_agreement"] == _n_models).sum()
                _log_and_print(
                    f"  Ensemble alignment ({_n_models} models): {len(r.quad):,} stocks, "
                    f"full consensus ({_n_models}/{_n_models}): {full}"
                )

        r.corr_info = compute_cross_model_correlation(r.mc, r.kal)
        if r.corr_info.get("correlation") is not None:
            _log_and_print(
                f"  MC ↔ Kalman correlation: {r.corr_info['correlation']:.3f}"
            )

    except Exception as e:
        logger.error("Step 6 (Alignment) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 6 failed: {e}", logging.ERROR)


def _step_mcmc_return_analysis(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 7a: Parallel MCMC return analysis."""
    output_dir = Path(cfg.output_dir)
    try:
        r.mcmc_result = run_parallel_mcmc_return_analysis(
            r.pt,
            n_chains=cfg.mcmc_chains,
            n_samples=cfg.mcmc_samples,
            cache_dir=cfg.cache_dir,
            enable_caching=cfg.enable_result_caching or cfg.enable_mcmc_caching,
            cache_ttl_hours=cfg.cache_ttl_hours,
        )
        if r.mcmc_result:
            _log_and_print(
                f"  R̂={r.mcmc_result.get('r_hat', float('nan')):.4f}, "
                f"converged={r.mcmc_result.get('converged', True)}, "
                f"posterior mean={r.mcmc_result.get('posterior_mean', float('nan')):.2f}"
            )

            # Detailed chain diagnostics
            ci = r.mcmc_result.get("ci_95")
            if ci:
                _log_and_print(
                    f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}], "
                    f"posterior std={r.mcmc_result.get('posterior_std', float('nan')):.2f}"
                )
            if r.mcmc_result.get("ess_bulk") is not None:
                _log_and_print(
                    f"  ESS bulk={r.mcmc_result['ess_bulk']:.0f}, "
                    f"ESS tail={r.mcmc_result.get('ess_tail', 0):.0f}"
                )

            # Student-t fit summary
            if r.mcmc_result.get("student_t_mu") is not None:
                _log_and_print(
                    f"  MCMC Student-t: μ={r.mcmc_result['student_t_mu']:.2f}, "
                    f"df={r.mcmc_result.get('student_t_df', float('nan')):.1f}"
                )

            # Hierarchical MCMC summary
            hier = r.mcmc_result.get("hierarchical")
            if hier and "levels" in hier:
                g = hier.get("global", {})
                _log_and_print(
                    f"  Hierarchical MCMC: global mean={g.get('mean', float('nan')):.2f}, "
                    f"std={g.get('std', float('nan')):.2f}, n={g.get('n_obs', 0)}"
                )
                for level_name, groups in hier["levels"].items():
                    _log_and_print(f"    Level '{level_name}': {len(groups)} groups")
                    for grp, info in sorted(
                        groups.items(),
                        key=lambda x: x[1].get("n_obs", 0),
                        reverse=True,
                    )[:5]:
                        _log_and_print(
                            f"      {grp}: posterior={info['posterior_mean']:.2f} "
                            f"(raw={info['raw_mean']:.2f}), "
                            f"shrinkage={info['shrinkage']:.3f}, "
                            f"n={info['n_obs']}, "
                            f"P(>0)={info.get('prob_positive', 0) * 100:.1f}%"
                        )

            # Task 14: Pass observed returns through for PPC plots
            r.mcmc_result["observed_returns"] = (
                r.pt["implied_return_pt"].dropna().values
            )

            # Task 6: Re-enrich quad with risk-adjusted return after MCMC
            if r.mcmc_result and not r.quad.empty:
                r.quad = build_quad_model_alignment(
                    r.tri,
                    r.beat,
                    beat_threshold=cfg.beat_threshold,
                    credit=r.credit if not r.credit.empty else None,
                    div_safety=r.div_safety if not r.div_safety.empty else None,
                    anomaly=r.anomaly_results if not r.anomaly_results.empty else None,
                    anomaly_severity_threshold=cfg.anomaly_severity_threshold,
                    mcmc_result=r.mcmc_result,
                )
                _log_and_print(f"  ✓ Risk-adjusted returns computed for {len(r.quad):,} stocks")

            if ARVIZ_AVAILABLE and r.mcmc_result.get("inference_data") is not None:
                _write_viz(
                    create_mcse_convergence_panel(
                        r.mcmc_result["inference_data"], var_name="implied_return_pt"
                    ),
                    output_dir,
                    "er_mcse_convergence.html",
                )
        else:
            _log_and_print("  ⚠️ Parallel MCMC: skipped or insufficient data")

    except Exception as e:
        logger.error("Step 7a (MCMC) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 7a failed: {e}", logging.ERROR)


def _step_category_analytics(r: PipelineResult, cfg: PipelineConfig) -> None:
    """Step 7b: Per-category Bayesian probability analytics."""
    try:
        view_mapping = get_view_category_mapping()
        all_categories: dict[str, list[str]] = {}
        for view_name, info in view_mapping.items():
            raw_cat_label = info.get("category", view_name)
            # Ensure cat_label is hashable (str)
            if isinstance(raw_cat_label, list):
                cat_label = tuple(raw_cat_label)
            else:
                cat_label = str(raw_cat_label)

            feat_cols = info.get("feature_cols", [])
            if feat_cols:
                if cat_label in all_categories:
                    all_categories[str(cat_label)].extend(
                        c for c in feat_cols if c not in all_categories[str(cat_label)]
                    )
                else:
                    all_categories[str(cat_label)] = list(feat_cols)
        _log_and_print(
            f"  Feature categories from {len(view_mapping)} views → "
            f"{len(all_categories)} categories, "
            f"{sum(len(v) for v in all_categories.values())} total features"
        )

        r.category_analytics = run_category_probability_analysis(
            r.df_all,
            categories=all_categories,
            use_mcmc=cfg.use_mcmc,
            n_mcmc_samples=cfg.mcmc_samples,
            burn_in=cfg.mcmc_burn_in,
            n_jobs=cfg.n_jobs,
            max_features_per_category=cfg.max_features_per_category,
            cache_dir=cfg.cache_dir,
            enable_caching=cfg.enable_result_caching or cfg.enable_mcmc_caching,
            cache_ttl_hours=cfg.cache_ttl_hours,
        )
        if r.category_analytics:
            _log_and_print(f"  ✓ Analyzed {len(r.category_analytics)} categories")
            for cat_name, cat_result in r.category_analytics.items():
                n_feat = cat_result.get("features_analyzed", 0)
                bayesian_keys = list(cat_result.get("bayesian_results", {}).keys())
                _log_and_print(
                    f"    {cat_name}: {n_feat} features — {len(bayesian_keys)} posteriors"
                )
        else:
            _log_and_print("  ⚠️ No categories had sufficient features for analysis")

    except Exception as e:
        logger.error("Step 7b (Category Analytics) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 7b failed: {e}", logging.ERROR)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


def main(config: PipelineConfig | None = None) -> PipelineResult:
    """
    Main expected returns analytics pipeline (v3.7).

    v3.7 additions:
    - MCMC return analysis file-based caching (mirrors category analytics).
    - MCMC hierarchical posteriors exported to ``analytics.mcmc_return_analysis``.
    - Cache subdirectories per cache-key type (``category_analytics/``, ``mcmc_return/``).

    v3.6 refactoring:
    - Task 1.1: Each step extracted into its own function.
    - Task 1.2: Results collected in ``BaselinePipelineResult`` dataclass.
    - Task 2.1: Category analytics parallelized via joblib.
    - Task 2.2: Feature-level sampling budget control.
    - Task 2.3: Numba JIT enabled for MCMC inner loops.
    - Task 2.4: MCMC result caching between runs.
    - Task 3.1: Adaptive screening thresholds.
    - Task 3.2: Screening thresholds configurable via PipelineConfig.
    - Task 4.1–4.2: ArviZ graceful degradation.
    - Task 6.4: MCMC diagnostics merged into summary.
    - Task 7.1: Parallelized database exports.
    - Task 7.2: Selective export based on changed data.

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

    Returns
    -------
    PipelineResult
        Structured container with all pipeline outputs.
    """
    cfg = config or PipelineConfig.from_env()

    configure_logging(level=cfg.log_level, log_file=cfg.log_file)

    _log_and_print("=" * 80)
    _log_and_print("Expected Returns Analytics Pipeline v3.7")
    _log_and_print("=" * 80)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = get_optimization_status()
    _log_and_print(
        f"🔧 Numba: {opt.get('numba_available', False)}, "
        f"Joblib: {opt.get('joblib_available', False)}, "
        f"ArviZ: {opt.get('arviz_available', False)}"
    )
    _log_and_print(
        f"   n_jobs={cfg.n_jobs}, max_features/cat={cfg.max_features_per_category}, "
        f"caching={'on' if cfg.caching_enabled else 'off'}"
        f"{f' (TTL={cfg.cache_ttl_hours}h)' if cfg.caching_enabled else ''}"
    )

    # Cleanup expired cache files on startup
    if cfg.caching_enabled:
        expired_removed = cfg.clear_cache(expired_only=False)
        if expired_removed > 0:
            _log_and_print(f"   🧹 Cleaned {expired_removed} expired cache files")

    # ========================================================================
    # Task 1.1: Orchestrate pipeline via extracted step functions
    # ========================================================================

    runner = PipelineRunner(cfg)
    r = runner.r

    # ── Step 1: Data Loading ──
    _step_start = time.perf_counter()
    _log_and_print(
        "📦 Step 1: Loading feature data (v3.6: equities MV + feature views "
        "+ all stock features MV)..."
    )
    _log_and_print("-" * 80)

    r = _step_load_data(r, cfg)
    runner.r = r
    if r.df_all.empty:
        return r

    _log_and_print(f"  ⏱ Step 1 completed in {time.perf_counter() - _step_start:.1f}s")

    # ── Step 2: Monte Carlo Simulation ──
    _log_and_print(
        f"🎲 Step 2: Monte Carlo price target simulation ({cfg.mc_simulations:,} samples)..."
    )
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    runner.run_monte_carlo(r.df_all)
    _log_and_print(f"  ⏱ Step 2 completed in {time.perf_counter() - _step_start:.1f}s")

    # ── Step 3: Price Target Achievement ──
    _log_and_print("🎯 Step 3: Price target achievement model...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    runner.run_price_target(r.df_all, r.feature_df, r.catalog)
    _log_and_print(f"  ⏱ Step 3 completed in {time.perf_counter() - _step_start:.1f}s")

    # ── Step 4: Kalman Filter ──
    _log_and_print("📐 Step 4: Kalman-filtered price targets...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    runner.run_kalman_filter(r.df_all)
    _log_and_print(f"  ⏱ Step 4 completed in {time.perf_counter() - _step_start:.1f}s")

    # ── Step 5: Earnings Beat Analysis ──
    _log_and_print("📊 Step 5: Bayesian earnings beat analysis...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    runner.run_earnings_beat(r.df_all, r.feature_df, r.catalog)
    _log_and_print(f"  ⏱ Step 5 completed in {time.perf_counter() - _step_start:.1f}s")

    # ── Step 5b: Accounting Anomaly Detection ──
    _log_and_print("🔬 Step 5b: Accounting anomaly detection & analytics...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    runner.run_accounting_anomaly(r.df_all, r.feature_df, r.catalog)
    _log_and_print(f"  ⏱ Step 5b completed in {time.perf_counter() - _step_start:.1f}s")

    # ── Step 5c: Credit Risk & Dividend Safety ──
    _log_and_print("🛡️ Step 5c: Credit risk & dividend safety analysis...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    runner.run_credit_risk(r.df_all, r.feature_df, r.catalog)
    runner.run_dividend_safety(r.df_all, r.feature_df, r.catalog)
    _log_and_print(f"  ⏱ Step 5c completed in {time.perf_counter() - _step_start:.1f}s")

    # ── Step 5d: Stock Screening ──
    _log_and_print("🔍 Step 5d: Running stock screening strategies...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    _step_screening(r, cfg)
    _log_and_print(f"  ⏱ Step 5d completed in {time.perf_counter() - _step_start:.1f}s")

    # ── Step 5e: Resampled Posterior Returns ──
    _log_and_print("\U0001f9ea Step 5e: Resampled Bayesian posterior returns...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    r.resampled_posterior = run_resampled_posterior_analysis(r.df_all)
    _log_and_print(f"  ⏱ Step 5e completed in {time.perf_counter() - _step_start:.1f}s")

    # ── Step 6: Cross-Model Alignment ──
    _log_and_print("🔗 Step 6: Cross-model alignment...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    r.tri = build_tri_model_alignment(r.mc, r.kal, r.pt)
    r.quad = build_quad_model_alignment(r.tri, r.beat, credit=r.credit, div_safety=r.div_safety,
                                        anomaly=r.anomaly_results)
    r.strong = extract_strong_consensus(r.tri)
    _log_and_print(f"  ⏱ Step 6 completed in {time.perf_counter() - _step_start:.1f}s")

    # ========================================================================
    # 7. Expected Returns Summary
    # ========================================================================
    _log_and_print(
        "📋 Step 7: Building expected_returns_summary (multi-model merge)..."
    )
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    _enrichment_source = r.feature_df if not r.feature_df.empty else r.df_all

    try:
        r.summary = build_expected_returns_summary(
            r.mc,
            r.kal,
            r.pt,
            r.beat,
            r.anomaly_results,
            source_df=_enrichment_source,
            credit=r.credit,
            div_safety=r.div_safety,
            mcmc_result=r.mcmc_result,
            quad=r.quad,
        )
        if not r.summary.empty:
            _log_and_print(f"  ✓ {len(r.summary):,} stocks in expected_returns_summary")
            r.summary = filter_quality_stocks(r.summary, r.df_all)
            r.summary = compute_return_zscore_ranks(r.summary)
            r.corr_info = compute_cross_model_correlation(r.mc, r.kal)
    except Exception as e:
        logger.error("Step 7 (Summary) failed: %s", e, exc_info=True)
        _log_and_print(f"⚠️ Step 7 failed: {e}", logging.ERROR)

    # ========================================================================
    # 7a. Parallel MCMC Return Analysis
    # ========================================================================
    _log_and_print("\U0001f500 Step 7a: Parallel MCMC return analysis...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    _step_mcmc_return_analysis(r, cfg)
    _log_and_print(f"  ⏱ Step 7a completed in {time.perf_counter() - _step_start:.1f}s")

    # ========================================================================
    # 7b. Per-Category Bayesian Probability Analytics
    # ========================================================================
    _log_and_print("\U0001f9ee Step 7b: Per-category Bayesian probability analytics...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()
    _step_category_analytics(r, cfg)
    _log_and_print(f"  ⏱ Step 7b completed in {time.perf_counter() - _step_start:.1f}s")

    # ========================================================================
    # 8. INFERENCE DATA (ArviZ) — built before visualizations
    # Task 4.1–4.2: Graceful degradation when ArviZ unavailable
    # ========================================================================
    _step_start = time.perf_counter()
    idata_mc = None
    idata_beat = None
    idata_credit = None
    if ARVIZ_AVAILABLE:
        _log_and_print("🧪 Step 8: Building InferenceData (ArviZ)...")
        _log_and_print("-" * 80)
        try:
            if not r.mc.empty:
                idata_mc = build_monte_carlo_inference_data(
                    r.mc, r.df_all, n_simulations=25_000
                )
                idata_summary = summarize_inference_data(idata_mc)
                _log_and_print(
                    f"   ✓ MC InferenceData: {idata_summary.get('n_draws', 0)} draws, "
                    f"{idata_summary.get('n_equities', 0)} equities"
                )
                if idata_summary.get("r_hat"):
                    for var, rhat in idata_summary["r_hat"].items():
                        _log_and_print(f"     R̂ ({var}): {rhat:.4f}")

            if not r.beat.empty and "posterior_alpha" in r.beat.columns:
                idata_beat = build_beat_probability_inference_data(
                    r.beat, r.df_all, n_posterior_samples=4000, n_chains=4
                )
                beat_summary = summarize_inference_data(idata_beat)
                _log_and_print(
                    f"   ✓ Beat InferenceData: {beat_summary.get('n_chains', 0)} chains × "
                    f"{beat_summary.get('n_draws', 0)} draws"
                )
            if not r.credit.empty:
                idata_credit = build_credit_risk_inference_data(r.credit, r.df_all)
                credit_summary = summarize_inference_data(idata_credit)
                _log_and_print(
                    f"   ✓ Credit Risk InferenceData: "
                    f"{credit_summary.get('n_equities', 0)} equities"
                )

            # Accounting Anomaly InferenceData
            if (
                not r.anomaly_results.empty
                and "accounting_anomaly_score" in r.anomaly_results.columns
            ):
                idata_anomaly = build_accounting_anomaly_inference_data(
                    r.anomaly_results, n_posterior_samples=4000, n_chains=4
                )
                anomaly_idata_summary = summarize_inference_data(idata_anomaly)
                _log_and_print(
                    f"   ✓ Anomaly InferenceData: "
                    f"{anomaly_idata_summary.get('n_equities', 0)} equities"
                )

            # Category Analysis InferenceData (one per category)
            idata_category: dict = {}
            if r.category_analytics and not r.df_all.empty:
                for cat_name, cat_results in r.category_analytics.items():
                    try:
                        features = [f for f in cat_results if f != "_meta"]
                        if not features:
                            continue
                        idata_cat = build_category_analysis_inference_data(
                            cat_results,
                            r.df_all,
                            category_name=cat_name,
                            features=features,
                        )
                        idata_category[cat_name] = idata_cat
                        cat_summary = summarize_inference_data(idata_cat)
                        _log_and_print(
                            f"   ✓ Category '{cat_name}' InferenceData: "
                            f"{cat_summary.get('n_draws', 0)} draws"
                        )
                    except Exception as e:
                        logger.debug(
                            "Category InferenceData for %s failed: %s", cat_name, e
                        )

            # Resampled Technical InferenceData
            idata_resampled = None
            if not r.df_all.empty:
                try:
                    idata_resampled = build_resampled_technical_inference_data(r.df_all, freq="1QE")
                    if idata_resampled is not None:
                        resampled_summary = summarize_inference_data(idata_resampled)
                        _log_and_print(
                            f"   ✓ Resampled Technical InferenceData: "
                            f"{resampled_summary.get('n_draws', 0)} draws"
                        )
                except Exception as e:
                    logger.debug("Resampled Technical InferenceData failed: %s", e)

            # Log EquityCoordinates for traceability
            if EquityCoordinates is not None and not r.df_all.empty:
                try:
                    coords = EquityCoordinates.from_dataframe(r.df_all)
                    _log_and_print(
                        f"   ✓ EquityCoordinates: {len(coords.isins)} isins, "
                        f"{len(coords.sectors)} sectors"
                    )
                except Exception as e:
                    logger.debug("EquityCoordinates construction skipped: %s", e)

            # Build per-view InferenceData for ArviZ diagnostics
            if not r.df_features.empty:
                _log_and_print("   Building per-view feature InferenceData...")
                for view_name in FEATURE_VIEW_REGISTRY:
                    try:
                        idata_view = build_feature_view_inference_data(
                            view_name, r.df_features
                        )
                        view_summary = summarize_inference_data(idata_view)
                        _log_and_print(
                            f"     ✓ {view_name}: "
                            f"{view_summary.get('n_equities', 0)} equities"
                        )
                    except Exception as e:
                        logger.debug("InferenceData for %s failed: %s", view_name, e)

        except Exception as e:
            _log_and_print(f"   ⚠️ InferenceData error: {e}")
        _log_and_print()
    else:
        # Task 4.1: Graceful degradation — log what's missing and continue
        _log_and_print("⏭️  Step 8: ArviZ not available — skipping InferenceData")
        _log_and_print(
            "   💡 Install arviz + xarray for full Bayesian diagnostics: pip install arviz xarray"
        )
        _log_and_print("   Pipeline continues with all non-ArviZ analytics.\n")

    _log_and_print(f"  ⏱ Step 8 completed in {time.perf_counter() - _step_start:.1f}s")

    # Sync local aliases back to BaselinePipelineResult for return
    r.idata_mc, r.idata_beat, r.idata_credit = idata_mc, idata_beat, idata_credit
    r.idata_anomaly = locals().get("idata_anomaly")
    r.idata_category = locals().get("idata_category", {})
    r.idata_resampled = locals().get("idata_resampled")

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
            "combined_distress_score,"
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
            "eps_positive_streak",
            "eps_cont_positive_streak",
            "classification_confidence",
        ],
        "create_earnings_probability_dashboard": [
            "posterior_beat_prob",
            "prob_beat_given_momentum",
            "confidence_score",
            "historical_beat_rate",
            "gaap_revision_momentum",
            "gaap_vs_norm_revision_spread",
            "eps_positive_streak",
            "eps_cont_positive_streak",
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

    # --- Pick the richest available source for missing columns ---
    _viz_source = (
        r.df_features
        if not r.df_features.empty
        else r.df_all
        if not r.df_all.empty
        else pd.DataFrame()
    )

    # --- Enrich r.df (mv_equities) and r.beat with viz-critical columns ---
    r.df = _enrich_dataframe(r.df, _viz_source, _viz_needed_cols, "df (mv_equities)")
    r.beat = _enrich_dataframe(r.beat, _viz_source, _viz_needed_cols, "beat")

    # --- Validate coverage (alias-aware) ---
    _all_df_cols = set(r.df.columns) | set(r.beat.columns)
    _combined_df = pd.DataFrame(columns=list(_all_df_cols))  # stub for resolve_column
    _viz_gaps: dict[str, list[str]] = {}
    for func_name, required in _VIZ_REQUIRED_COLUMNS.items():
        missing = [
            c
            for c in required
            if c not in _all_df_cols and resolve_column(_combined_df, c) is None
        ]
        if missing:
            _viz_gaps[func_name] = missing
    if _viz_gaps:
        _log_and_print(f"  ⚠️ Viz column coverage gaps after enrichment: {_viz_gaps}")
    else:
        _log_and_print("  ✓ All viz column requirements satisfied")

    # ========================================================================
    # 9. VISUALIZATIONS (consuming InferenceData)
    # ========================================================================
    _log_and_print("📈 Step 9: Generating visualizations...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    try:
        if not r.mc.empty:
            _write_viz(
                create_mc_return_distribution(r.mc),
                output_dir,
                "er_mc_distribution.html",
            )
            _write_viz(
                create_sector_risk_reward_scatter(r.mc, identifier_coords=r.id_coords),
                output_dir,
                "er_sector_risk_reward.html",
            )
            _write_viz(create_var_analysis(r.mc), output_dir, "er_var_analysis.html")
            _write_viz(
                create_posterior_return_forest(r.mc, top_n=25),
                output_dir,
                "er_posterior_return_forest.html",
            )

        if not r.kal.empty:
            _write_viz(
                create_kalman_vs_raw_scatter(r.kal), output_dir, "er_kalman_vs_raw.html"
            )

        if not r.tri.empty:
            _write_viz(
                create_tri_model_agreement_histogram(r.tri),
                output_dir,
                "er_tri_model_agreement.html",
            )
            _write_viz(
                create_sector_heatmap(
                    r.tri,
                    compute_sector_expected_returns,
                    schema_metadata=r.schema_metadata,
                ),
                output_dir,
                "er_sector_heatmap.html",
            )

        if not r.strong.empty:
            _write_viz(
                create_strong_consensus_bar(r.strong),
                output_dir,
                "er_strong_consensus.html",
            )
            tri_cols = {
                "isin",
                "implied_return_mc",
                "implied_return_kalman",
                "implied_return_pt",
            }
            if tri_cols.issubset(r.strong.columns):
                _write_viz(
                    create_tri_model_posterior_comparison(r.strong, top_n=24),
                    output_dir,
                    "er_tri_model_posterior.html",
                )
            pt_cols = {
                "ticker",
                "price_target_mc",
                "price_target_kalman",
                "price_target_pt",
            }
            if pt_cols.issubset(r.strong.columns):
                _write_viz(
                    create_tri_model_posterior_price_target_comparison(
                        r.strong, top_n=24
                    ),
                    output_dir,
                    "er_tri_model_price_target_posterior.html",
                )

        if not r.beat.empty and not r.pt.empty:
            _write_viz(
                create_beat_vs_achievement_scatter(r.beat, r.pt),
                output_dir,
                "er_beat_vs_achievement.html",
            )

        if not r.beat.empty and "prob_beat_given_momentum" in r.beat.columns:
            _write_viz(
                create_beat_probability_posterior(r.beat, top_n=24),
                output_dir,
                "er_beat_probability_posterior.html",
            )

        if not r.beat.empty:
            _write_viz(
                create_earnings_probability_dashboard(r.beat),
                output_dir,
                "er_earnings_probability_dashboard.html",
            )

        # v3.2: Earnings quality visualizations (revision momentum, GAAP divergence, enhanced beat)
        if not r.beat.empty and "eps_revision_momentum" in r.beat.columns:
            _write_viz(
                create_revision_momentum_chart(r.beat, top_n=30),
                output_dir,
                "er_revision_momentum.html",
            )

        if not r.beat.empty and "gaap_adj_eps_gap_pct" in r.beat.columns:
            _write_viz(
                create_gaap_divergence_plot(r.beat),
                output_dir,
                "er_gaap_divergence.html",
            )

        if not r.beat.empty and "prob_beat_given_momentum" in r.beat.columns:
            _write_viz(
                create_enhanced_beat_prob_dash(r.beat),
                output_dir,
                "er_enhanced_beat_probability.html",
            )

        if not r.df.empty:
            _write_viz(
                create_quality_risk_quadrant(r.df),
                output_dir,
                "er_quality_risk_quadrant.html",
            )
            _write_viz(
                create_distress_early_warning_dashboard(r.df),
                output_dir,
                "er_distress_early_warning.html",
            )

        # v3.2: Quality & risk deep-dive visualizations
        if not r.df.empty and "piotroski_f_score" in r.df.columns:
            _write_viz(
                create_piotroski_fscore_breakdown(r.df),
                output_dir,
                "er_piotroski_fscore.html",
            )

        if not r.df.empty and "altman_z_score" in r.df.columns:
            _write_viz(
                create_altman_zscore_distribution(r.df),
                output_dir,
                "er_altman_zscore.html",
            )

        if not r.df.empty:
            _write_viz(
                create_beneish_mscore_analysis(r.df),
                output_dir,
                "er_beneish_mscore.html",
            )

        if not r.df.empty and "combined_distress_risk_score" in r.df.columns:
            _write_viz(
                create_risk_tier_sunburst(r.df),
                output_dir,
                "er_risk_tier_sunburst.html",
            )

        if (
            not r.anomaly_results.empty
            and "accounting_anomaly_score" in r.anomaly_results.columns
        ):
            _write_viz(
                create_accounting_anomaly_dashboard(r.anomaly_results),
                output_dir,
                "er_accounting_anomaly_dashboard.html",
            )

        if (
            not r.anomaly_results.empty
            and "anomaly_severity_score" in r.anomaly_results.columns
        ):
            _write_viz(
                create_anomaly_severity_dashboard(r.anomaly_results),
                output_dir,
                "er_anomaly_severity_dashboard.html",
            )

        if (
            not r.anomaly_results.empty
            and "anomaly_conditional_probability" in r.anomaly_results.columns
        ):
            _write_viz(
                create_anomaly_conditional_probability_chart(r.anomaly_results),
                output_dir,
                "er_anomaly_conditional_probability.html",
            )

        # v3.3: MCMC-enhanced probability model visualizations
        if (
            not r.anomaly_results.empty
            and "anomaly_posterior_mean" in r.anomaly_results.columns
        ):
            _write_viz(
                create_mcmc_anomaly_posterior_chart(r.anomaly_results),
                output_dir,
                "er_mcmc_anomaly_posterior.html",
            )

        if not r.credit.empty and "mcmc_distress_probability" in r.credit.columns:
            _write_viz(
                create_mcmc_credit_risk_chart(r.credit),
                output_dir,
                "er_mcmc_credit_risk_posterior.html",
            )

        if not r.div_safety.empty and "mcmc_cut_probability" in r.div_safety.columns:
            _write_viz(
                create_mcmc_dividend_cut_chart(r.div_safety),
                output_dir,
                "er_mcmc_dividend_cut_posterior.html",
            )

        if not r.pt.empty and "mcmc_achievement_probability" in r.pt.columns:
            _write_viz(
                create_mcmc_price_target_chart(r.pt),
                output_dir,
                "er_mcmc_price_target_posterior.html",
            )

        if not r.credit.empty and "ruin_probability" in r.credit.columns:
            _write_viz(
                create_ruin_probability_diagnostic(
                    r.credit, top_n=20, identifier_coords=r.id_coords
                ),
                output_dir,
                "er_ruin_probability_diagnostic.html",
            )

        if not r.summary.empty:
            tri_cols = {
                "isin",
                "implied_return_mc",
                "implied_return_kalman",
                "implied_return_pt",
            }
            if tri_cols.issubset(r.summary.columns):
                _write_viz(
                    create_tri_model_posterior_comparison(r.summary, top_n=24),
                    output_dir,
                    "er_expected_returns_summary_posterior.html",
                )
            pt_cols = {
                "ticker",
                "price_target_mc",
                "price_target_kalman",
                "price_target_pt",
            }
            if pt_cols.issubset(r.summary.columns):
                _write_viz(
                    create_tri_model_posterior_price_target_comparison(
                        r.summary, top_n=24
                    ),
                    output_dir,
                    "er_expected_returns_summary_price_target_posterior.html",
                )
                _write_viz(
                    create_model_dispersion_dashboard(r.summary),
                    output_dir,
                    "er_model_dispersion_dashboard.html",
                )

            if not r.mc.empty:
                _write_viz(
                    create_return_distribution_fit_chart(r.mc),
                    output_dir,
                    "er_return_distribution_fit.html",
                )

            sector_analytics = compute_sector_return_analytics(r.summary)
            if not sector_analytics.empty:
                _write_viz(
                    create_sector_return_analytics_heatmap(sector_analytics),
                    output_dir,
                    "er_sector_return_analytics.html",
                )

        # v3.0: Screening summary chart
        if r.screens:
            _write_viz(
                create_screening_summary_chart(r.screens),
                output_dir,
                "er_screening_summary.html",
            )

        # v3.1: Price target drift dashboard
        if not r.df.empty:
            _write_viz(
                create_price_target_drift_dashboard(r.df, mv_spec=r.mv_equities_spec),
                output_dir,
                "er_price_target_drift.html",
            )

        # ── Valuation Analysis (v3.1) ──
        # Use r.df_all (feature views) which contains computed valuation ratios
        # (p_e_ratio, p_b_ratio, etc.); r.df (mv_equities) only has raw suffixed cols.
        _viz_df = r.df_all if not r.df_all.empty else r.df
        if not _viz_df.empty:
            _write_viz(
                create_valuation_multiples_comparison(_viz_df),
                output_dir,
                "er_valuation_multiples.html",
            )
            _write_viz(
                create_valuation_distribution_dashboard(_viz_df),
                output_dir,
                "er_valuation_distribution.html",
            )
            _write_viz(
                create_relative_valuation_matrix(_viz_df),
                output_dir,
                "er_relative_valuation_matrix.html",
            )
            _write_viz(
                create_valuation_vs_growth_quadrant(_viz_df),
                output_dir,
                "er_valuation_vs_growth.html",
            )
            _write_viz(
                create_historical_valuation_percentile(_viz_df),
                output_dir,
                "er_historical_valuation_percentile.html",
            )

        # ── Earnings Quality (v3.1) ──
        if not _viz_df.empty:
            _write_viz(
                create_earnings_surprise_dashboard(_viz_df),
                output_dir,
                "er_earnings_surprise.html",
            )
            _write_viz(
                create_eps_trajectory_analysis(_viz_df),
                output_dir,
                "er_eps_trajectory.html",
            )
            _write_viz(
                create_earnings_quality_decomposition(_viz_df),
                output_dir,
                "er_earnings_quality_decomposition.html",
            )
            _write_viz(
                create_beat_rate_heatmap(_viz_df),
                output_dir,
                "er_beat_rate_heatmap.html",
            )
            _write_viz(
                create_earnings_consistency_matrix(_viz_df),
                output_dir,
                "er_earnings_consistency_matrix.html",
            )

        # ── Growth Analysis (v3.1) ──
        if not _viz_df.empty:
            _write_viz(
                create_growth_waterfall_chart(_viz_df),
                output_dir,
                "er_growth_waterfall.html",
            )
            _write_viz(
                create_growth_consistency_matrix(_viz_df),
                output_dir,
                "er_growth_consistency_matrix.html",
            )
            _write_viz(
                create_growth_vs_profitability_quadrant(_viz_df),
                output_dir,
                "er_growth_vs_profitability.html",
            )
            _write_viz(
                create_growth_acceleration_chart(_viz_df),
                output_dir,
                "er_growth_acceleration.html",
            )
            _write_viz(
                create_sustainable_growth_analysis(_viz_df),
                output_dir,
                "er_sustainable_growth.html",
            )

        # ── Feature View Posterior Panel (v3.1) ──
        # create_feature_view_posterior_panel expects InferenceData/xr.Dataset,
        # not a raw DataFrame.  Build per-view InferenceData first.
        if (
            not r.df_all.empty
            and r.view_specs
            and build_feature_view_inference_data is not None
        ):
            for _vs_name, _vs in r.view_specs.items():
                try:
                    _fv_idata = build_feature_view_inference_data(_vs_name, r.df_all)
                    _write_viz(
                        create_feature_view_posterior_panel(_fv_idata, view_spec=_vs),
                        output_dir,
                        f"er_feature_view_posterior_{_vs_name}.html",
                    )
                except Exception as _fv_err:
                    logger.debug(
                        "Feature view posterior %s skipped: %s", _vs_name, _fv_err
                    )

        # Bayesian category ridge for analyst sentiment features
        sentiment_features = [
            f
            for f in [
                "analyst_bullish_pct",
                "expected_upside_pt",
                "eps_revision_momentum",
                "analyst_conviction",
                "pt_consensus_convergence",
            ]
            if f in r.df_all.columns
        ]
        if sentiment_features:
            results = bayesian_category_analysis(
                r.df_all, "Analyst Sentiment", sentiment_features
            )
            _write_viz(
                create_bayesian_category_ridge(
                    results, category_name="Analyst Sentiment"
                ),
                output_dir,
                "er_bayesian_sentiment_ridge.html",
            )
            _write_viz(
                create_mcmc_category_posterior_chart(
                    results, category_name="Analyst Sentiment"
                ),
                output_dir,
                "er_mcmc_category_sentiment_posterior.html",
            )

        # v3.0: Bayesian category ridge for profitability features
        profitability_features = [
            f
            for f in ["roe", "roa", "roic", "gross_margin_pct", "operating_margin_pct"]
            if f in r.df_all.columns
        ]
        if profitability_features:
            results = bayesian_category_analysis(
                r.df_all, "Profitability", profitability_features
            )
            _write_viz(
                create_bayesian_category_ridge(results, category_name="Profitability"),
                output_dir,
                "er_bayesian_profitability_ridge.html",
            )

    except Exception as e:
        _log_and_print(f"   ⚠️ HTML visualization error: {e}")
        import traceback

        traceback.print_exc()

    # ── ArviZ Diagnostic Visualizations (v3.2) ──
    # Separated into its own try/except block so that any failure in
    # the HTML visualization section above does not prevent ArviZ
    # diagnostic PNGs from being generated.
    try:
        if _ARVIZ_DIAG_AVAILABLE:
            _log_and_print("   📊 Generating ArviZ diagnostic visualizations...")

            # Step 5d: Screening posterior ridge
            if r.screens:
                try:
                    fig = create_screening_posterior_ridge(r.screens)
                    if fig:
                        _write_viz(
                            fig,
                            output_dir,
                            "er_screening_posterior_ridge.png",
                            fmt="png",
                        )
                except Exception as e:
                    logger.debug("Screening posterior ridge skipped: %s", e)

            # Step 5d: Productivity frontier posterior
            if not r.df_all.empty and "productivity_frontier_score" in r.df_all.columns:
                try:
                    fig = create_productivity_frontier_posterior(r.df_all)
                    if fig:
                        _write_viz(
                            fig,
                            output_dir,
                            "er_productivity_frontier_posterior.png",
                            fmt="png",
                        )
                except Exception as e:
                    logger.debug("Productivity frontier posterior skipped: %s", e)

            # Step 5e: Resampled posterior diagnostics
            if not r.resampled_posterior.empty:
                try:
                    resamp_outputs = create_resampled_posterior_diagnostics(
                        r.resampled_posterior, output_dir
                    )
                    for ro in resamp_outputs:
                        _log_and_print(f"   ✓ {Path(ro).name}")
                except Exception as e:
                    logger.debug("Resampled posterior ArviZ skipped: %s", e)

                try:
                    fig = create_resampled_sector_forest(r.resampled_posterior, r.df)
                    if fig:
                        _write_viz(
                            fig, output_dir, "er_resampled_sector_forest.png", fmt="png"
                        )
                except Exception as e:
                    logger.debug("Resampled sector forest skipped: %s", e)

            # Step 6: Model alignment ArviZ panel
            if not r.summary.empty:
                try:
                    align_outputs = create_model_alignment_arviz_panel(
                        r.summary, output_dir
                    )
                    for ao in align_outputs:
                        _log_and_print(f"   ✓ {Path(ao).name}")
                except Exception as e:
                    logger.debug("Model alignment ArviZ skipped: %s", e)

                try:
                    fig = create_agreement_posterior_by_sector(r.summary)
                    if fig:
                        _write_viz(
                            fig, output_dir, "er_agreement_by_sector.png", fmt="png"
                        )
                except Exception as e:
                    logger.debug("Agreement by sector skipped: %s", e)

            # Step 7: Hierarchical shrinkage diagnostic
            if not r.summary.empty:
                try:
                    fig = create_hierarchical_shrinkage_diagnostic(r.summary)
                    if fig:
                        _write_viz(
                            fig, output_dir, "er_hierarchical_shrinkage.png", fmt="png"
                        )
                except Exception as e:
                    logger.debug("Hierarchical shrinkage plot skipped: %s", e)

                try:
                    fig = create_multi_level_mcmc_comparison(r.summary)
                    if fig:
                        _write_viz(
                            fig, output_dir, "er_multi_level_mcmc.png", fmt="png"
                        )
                except Exception as e:
                    logger.debug("Multi-level MCMC comparison skipped: %s", e)

            # Step 7a: MCMC convergence panel
            if r.mcmc_result:
                try:
                    mcmc_outputs = create_mcmc_convergence_panel_arviz(
                        r.mcmc_result, output_dir
                    )
                    for mo in mcmc_outputs:
                        _log_and_print(f"   ✓ {Path(mo).name}")
                except Exception as e:
                    logger.debug("MCMC convergence panel skipped: %s", e)

            # Step 7b: Category posterior diagnostics
            if r.category_analytics:
                try:
                    cat_outputs = create_category_posterior_diagnostics(
                        r.category_analytics, r.df_all, output_dir
                    )
                    for co in cat_outputs:
                        _log_and_print(f"   ✓ {Path(co).name}")

                    cross_path = create_cross_category_summary(
                        r.category_analytics, output_dir
                    )
                    if cross_path:
                        _log_and_print(f"   ✓ {Path(cross_path).name}")
                except Exception as e:
                    logger.debug("Category ArviZ diagnostics skipped: %s", e)

            # ── ArviZ 1.0 New Visualization Types (v3.6) ──

            # Hierarchical dot plot — sector posterior comparison
            if not r.summary.empty:
                try:
                    fig = create_hierarchical_dot_comparison(r.summary)
                    if fig:
                        _write_viz(
                            fig, output_dir, "er_hierarchical_dotplot.png", fmt="png"
                        )
                except Exception as e:
                    logger.debug("Hierarchical dot comparison skipped: %s", e)

            # Cross-model ECDF with reference quantile lines
            if not r.summary.empty:
                try:
                    ecdf_path = create_cross_model_ecdf_with_references(
                        r.summary, output_dir
                    )
                    if ecdf_path:
                        _log_and_print(f"   ✓ {Path(ecdf_path).name}")
                except Exception as e:
                    logger.debug("Cross-model ECDF skipped: %s", e)

            # Screening PPC rootogram — predicted vs observed returns
            if r.screens:
                try:
                    fig = create_screening_ppc_rootogram(r.screens)
                    if fig:
                        _write_viz(
                            fig, output_dir, "er_screening_ppc_rootogram.png", fmt="png"
                        )
                except Exception as e:
                    logger.debug("Screening PPC rootogram skipped: %s", e)

            # ArviZ 1.0 category posterior forest
            if r.category_analytics:
                for cat_name, cat_data in r.category_analytics.items():
                    try:
                        fig = create_mcmc_category_posterior_arviz(
                            cat_data, category_name=cat_name
                        )
                        if fig:
                            safe_name = cat_name.lower().replace(" ", "_")[:30]
                            _write_viz(
                                fig,
                                output_dir,
                                f"er_category_forest_arviz_{safe_name}.png",
                                fmt="png",
                            )
                    except Exception as e:
                        logger.debug(
                            "Category ArviZ forest (%s) skipped: %s", cat_name, e
                        )

            # Unified convergence dashboard — cross-model trace + ESS
            if (
                create_unified_convergence_dashboard is not None
                and r.mcmc_result
                and not r.anomaly_results.empty
                and not r.summary.empty
            ):
                try:
                    unified_outputs = create_unified_convergence_dashboard(
                        r.mcmc_result,
                        r.anomaly_results,
                        r.summary,
                        output_dir,
                    )
                    for uo in unified_outputs:
                        _log_and_print(f"   ✓ {Path(uo).name}")
                except Exception as e:
                    logger.debug("Unified convergence dashboard skipped: %s", e)

        else:
            _log_and_print(
                "   ⏭️ ArviZ diagnostics not available — skipping PNG diagnostic plots"
            )

    except Exception as e:
        _log_and_print(f"   ⚠️ ArviZ diagnostic visualization error: {e}")
        import traceback

        traceback.print_exc()

    _log_and_print(f"  ⏱ Step 9 completed in {time.perf_counter() - _step_start:.1f}s")

    # ========================================================================
    # 10. EXPORT RESULTS
    # ========================================================================
    _log_and_print("💾 Step 10: Exporting results...")
    _log_and_print("-" * 80)
    _step_start = time.perf_counter()

    exports = export_expected_returns_results(mc=r.mc, pt=r.pt, kal=r.kal, tri=r.tri, strong=r.strong, beat=r.beat,
                                              summary=r.summary, credit=r.credit, div_safety=r.div_safety,
                                              anomaly_results=r.anomaly_results, screens=r.screens,
                                              output_dir=str(output_dir), max_workers=cfg.export_max_workers)
    for name, dest in exports.items():
        _log_and_print(f"   ✓ {name} → {dest}")

    # Export probability analytics results (beat + streak + credit + dividend)
    try:
        streak_analyzer = EPSStreakAnalyzer()
        streak_df = streak_analyzer.analyze_dataframe(
            r.df_all if not r.df_all.empty else r.df
        )
        prob_exports = export_probability_analytics_results(
            probability_df=r.beat, streak_df=streak_df, output_dir=output_dir
        )
        for pname, pdest in prob_exports.items():
            _log_and_print(f"   ✓ {pname} → {pdest}")
    except Exception as e:
        logger.warning("Probability analytics export failed: %s", e)

    # Aggregate probability results for category analytics
    if r.category_analytics:
        try:
            for cat_name, cat_result in r.category_analytics.items():
                cat_prob = cat_result.get("conditional_probabilities")
                if isinstance(cat_prob, pd.DataFrame) and not cat_prob.empty:
                    aggregated = aggregate_probability_results(cat_prob)
                    if not aggregated.empty:
                        export_config = ExportConfig(
                            table_name=f"prob_{cat_name.lower().replace(' ', '_')}"
                        )
                        export_to_db(aggregated, export_config)
                        logger.info(
                            "Aggregated probability export: %s (%d rows)",
                            cat_name,
                            len(aggregated),
                        )
        except Exception as e:
            logger.warning("Aggregated probability export failed: %s", e)

    # Export MCMC hierarchical posteriors
    if r.mcmc_result:
        try:
            mcmc_rows = []
            # Top-level convergence diagnostics as a single "global" row
            mcmc_rows.append({
                "level": "global",
                "group": "all",
                "posterior_mean": r.mcmc_result.get("posterior_mean"),
                "posterior_std": r.mcmc_result.get("posterior_std"),
                "r_hat": r.mcmc_result.get("r_hat"),
                "converged": r.mcmc_result.get("converged"),
                "ess_bulk": r.mcmc_result.get("ess_bulk"),
                "ess_tail": r.mcmc_result.get("ess_tail"),
                "ci_95_lower": r.mcmc_result.get("ci_95", [None, None])[0],
                "ci_95_upper": r.mcmc_result.get("ci_95", [None, None])[1],
                "student_t_mu": r.mcmc_result.get("student_t_mu"),
                "student_t_df": r.mcmc_result.get("student_t_df"),
                "n_obs": None,
                "raw_mean": None,
                "shrinkage": None,
                "prob_positive": None,
            })

            # Hierarchical level rows
            hier = r.mcmc_result.get("hierarchical", {})
            if hier and "levels" in hier:
                for level_name, groups in hier["levels"].items():
                    for grp_name, info in groups.items():
                        mcmc_rows.append({
                            "level": level_name,
                            "group": grp_name,
                            "posterior_mean": info.get("posterior_mean"),
                            "posterior_std": info.get("posterior_std"),
                            "r_hat": None,
                            "converged": None,
                            "ess_bulk": None,
                            "ess_tail": None,
                            "ci_95_lower": None,
                            "ci_95_upper": None,
                            "student_t_mu": None,
                            "student_t_df": None,
                            "n_obs": info.get("n_obs"),
                            "raw_mean": info.get("raw_mean"),
                            "shrinkage": info.get("shrinkage"),
                            "prob_positive": info.get("prob_positive"),
                        })

            mcmc_export_df = pd.DataFrame(mcmc_rows)
            if not mcmc_export_df.empty:
                export_config = ExportConfig(table_name="mcmc_return_analysis")
                export_to_db(mcmc_export_df, export_config)
                _log_and_print(
                    f"   ✓ mcmc_return_analysis → analytics.mcmc_return_analysis "
                    f"({len(mcmc_export_df)} rows)"
                )
        except Exception as e:
            logger.warning("MCMC return analysis export failed: %s", e)

    _log_and_print(f"  ⏱ Step 10 completed in {time.perf_counter() - _step_start:.1f}s")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    _log_and_print("=" * 80)
    _log_and_print("\u2705 EXPECTED RETURNS ANALYTICS v3.7 COMPLETE")
    _log_and_print("=" * 80)
    _log_and_print("  Data sources:")
    _log_and_print(
        f"    mv_expected_returns:       {len(r.df):,} stocks × {len(r.df.columns)} features"
    )
    _log_and_print(
        f"    mv_all_stock_features:     {len(r.df_all):,} stocks × {len(r.df_all.columns)} features"
    )
    _log_and_print("  Models:")
    _log_and_print(f"    Monte Carlo simulations:   {len(r.mc):,}")
    _log_and_print(f"    Price target achievements: {len(r.pt):,}")
    _log_and_print(f"    Kalman-filtered targets:   {len(r.kal):,}")
    _log_and_print(f"    Earnings beat analyses:    {len(r.beat):,}")
    _log_and_print(f"    Credit risk analyses:      {len(r.credit):,}")
    _log_and_print(f"    Dividend safety analyses:  {len(r.div_safety):,}")
    _log_and_print(f"    Accounting anomaly analyses: {len(r.anomaly_results):,}")

    _log_and_print("  Alignment:")
    _log_and_print(f"    Tri-model aligned:         {len(r.tri):,}")
    _log_and_print(f"    Strong consensus picks:    {len(r.strong):,}")
    if not r.quad.empty:
        max_agreement = float(r.quad["quad_agreement"].max())
        _log_and_print(
            f"    Quad-model full consensus: {(r.quad['quad_agreement'] == max_agreement).sum()}"
        )
    if not r.summary.empty:
        max_score = float(r.summary["agreement_score"].max())
        full_consensus_count = (r.summary["agreement_score"] == max_score).sum()
        _log_and_print(
            f"    Expected returns summary:  {len(r.summary):,} stocks, {full_consensus_count} full consensus"
        )
    if r.corr_info.get("correlation") is not None:
        _log_and_print(
            f"    MC ↔ Kalman correlation:   {r.corr_info['correlation']:.3f}"
        )
    _log_and_print("  Screening:")
    for name, screen_df in r.screens.items():
        if not screen_df.empty:
            _log_and_print(f"    {name}: {len(screen_df):,} stocks")
    _log_and_print("  Probability Analytics:")
    _log_and_print(f"    Categories analyzed:       {len(r.category_analytics)}")
    _log_and_print(f"  Outputs: {output_dir}/")

    return r


if __name__ == "__main__":
    main()
