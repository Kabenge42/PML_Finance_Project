"""
Pipeline runners for PML models (R1).

Migrated from ``expected_returns_v3.py`` to decouple v4 from v3.
Each runner calls PML model classes directly.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from probabilistic_ml_model.data_utils.feature_catalog import (
    FeatureViewCatalog,
    auto_enrich_for_model,
    get_feature_catalog,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------


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

    mc_simulations: int = 10_000
    mc_max_stocks: int = 10_000
    mcmc_chains: int = 8
    mcmc_samples: int = 5_000
    beat_threshold: float = 0.50
    output_dir: str = "outputs"
    log_file: str | None = "logs/expected_returns_pipeline.log"
    log_level: int = logging.INFO
    # v3.5: MCMC-specific settings surfaced for per-model configuration
    mcmc_burn_in: int = 1000
    use_mcmc: bool = True
    # ---------- Heavy-tail / distributional controls (Finding #1) ----------
    use_student_t: bool = True  # was False; df≈2 demands t-likelihood
    student_t_df_floor: float = 3.0  # clamp df to avoid infinite variance
    use_mixture_likelihood: bool = True  # 2-component Normal mixture fallback
    mixture_components: int = 2
    use_stable_distribution: bool = False  # enable alpha-stable when df<=2.5
    tail_risk_metric: str = "cvar"  # "var" | "cvar"
    cvar_alpha: float = 0.05
    # ---------- Time-varying volatility (Finding #2) ----------
    use_garch_volatility: bool = True
    garch_p: int = 1
    garch_q: int = 1
    use_stochastic_vol: bool = False  # SV alternative to GARCH
    vol_regime_window: int = 60  # trading days for rolling σ feature
    # ---------- Cross-model Bayesian averaging (Finding #3) ----------
    use_bayesian_model_averaging: bool = True
    bma_prior_weights: tuple = (0.30, 0.20, 0.20, 0.15, 0.10, 0.05)
    # order: (MC, Kalman, PriceTarget, EarningsBeat, Credit, DivSafety)
    bma_log_score_window: int = 252  # days of realized returns for weighting
    ensemble_shrinkage_kappa: float = 20.0  # James–Stein style shrinkage strength
    # ---------- Macro hierarchical predictors (Finding #4) ----------
    use_macro_covariates: bool = True
    macro_covariates: tuple = ("yield_curve_slope", "vix", "pmi", "dxy")
    macro_hierarchy_level: str = "region"  # pool macro effects by region
    # ---------- Rolling-window backtest calibration (Finding #5) ----------
    enable_rolling_backtest: bool = True
    backtest_window_months: int = 36
    backtest_step_months: int = 3
    ci_coverage_target: float = 0.95
    anomaly_z_threshold: float | None = None
    # v3.6: Screening threshold configuration (Task 3.2)
    screening_min_pct: float = 0.5  # Minimum % of universe for adaptive fallback
    screening_quality_roe_min: float = 0.25
    screening_quality_piotroski_min: float = 6.0
    screening_dividend_yield_min: float = 0.05
    screening_dividend_coverage_min: float = 2.5
    # v3.6: Performance tuning (Task 2.1–2.4)
    n_jobs: int = -1  # Joblib parallelism (-1 = all cores)
    max_features_per_category: int = 25  # Sampling budget control (Task 2.2)
    enable_result_caching: bool = True  # General pipeline result caching (Task 2.4)
    enable_mcmc_caching: bool = True  # MCMC-specific result caching
    cache_dir: str = ".cache"
    cache_ttl_hours: float = 24.0  # Cache time-to-live in hours
    # v3.6: Export parallelism (Task 7.1)
    export_max_workers: int = 6
    # v3.7: Data loading strategy (prefer mv_all_stock_features as primary)
    prefer_materialized_view: bool = True
    # v3.8: Ensemble alignment refactoring (Issues 1–8)
    bullish_return_threshold: float = (
        0.02  # Minimum % return for bullish classification (heavy-tail adjusted)
    )
    anomaly_severity_threshold: float | None = None  # None = data-adaptive (median)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build config from environment variables with sensible defaults."""
        return cls(
            mc_simulations=int(os.environ.get("ER_MC_SIMULATIONS", 50_000)),
            mc_max_stocks=int(os.environ.get("ER_MC_MAX_STOCKS", 25_000)),
            mcmc_chains=int(os.environ.get("ER_MCMC_CHAINS", 8)),
            mcmc_samples=int(os.environ.get("ER_MCMC_SAMPLES", 10_000)),
            output_dir=os.environ.get("ER_OUTPUT_DIR", "outputs"),
            log_file=os.environ.get("ER_LOG_FILE", "logs/expected_returns_pipeline.log"),
            mcmc_burn_in=int(os.environ.get("ER_MCMC_BURN_IN", 1000)),
            use_mcmc=os.environ.get("ER_USE_MCMC", "true").lower() == "true",
            use_student_t=os.environ.get("ER_USE_STUDENT_T", "true").lower() == "true",
            use_garch_volatility=os.environ.get("ER_USE_GARCH", "true").lower() == "true",
            use_bayesian_model_averaging=os.environ.get("ER_USE_BMA", "true").lower() == "true",
            use_macro_covariates=os.environ.get("ER_USE_MACRO", "true").lower() == "true",
            tail_risk_metric=os.environ.get("ER_TAIL_RISK", "cvar"),
            cvar_alpha=float(os.environ.get("ER_CVAR_ALPHA", 0.05)),
            student_t_df_floor=float(os.environ.get("ER_STUDENT_T_DF_FLOOR", 3.0)),
            # v3.6: Screening thresholds from env
            screening_min_pct=float(os.environ.get("ER_SCREENING_MIN_PCT", 0.5)),
            screening_quality_roe_min=float(os.environ.get("ER_SCREENING_QUALITY_ROE_MIN", 0.25)),
            screening_quality_piotroski_min=float(
                os.environ.get("ER_SCREENING_QUALITY_PIOTROSKI_MIN", 6.0)
            ),
            screening_dividend_yield_min=float(
                os.environ.get("ER_SCREENING_DIVIDEND_YIELD_MIN", 0.05)
            ),
            screening_dividend_coverage_min=float(
                os.environ.get("ER_SCREENING_DIVIDEND_COVERAGE_MIN", 2.5)
            ),
            # v3.6: Performance tuning from env
            n_jobs=int(os.environ.get("ER_N_JOBS", os.environ.get("N_JOBS", -1))),
            max_features_per_category=int(os.environ.get("ER_MAX_FEATURES_PER_CATEGORY", 25)),
            # FIX: Compare against "true" (not "false") so the flag is not inverted
            enable_result_caching=os.environ.get("ER_ENABLE_CACHING", "true").lower() == "true",
            enable_mcmc_caching=os.environ.get("ER_ENABLE_MCMC_CACHING", "true").lower() == "true",
            cache_dir=os.environ.get("ER_CACHE_DIR", os.environ.get("CACHE_DIR", ".cache")),
            cache_ttl_hours=float(os.environ.get("ER_CACHE_TTL_HOURS", 24.0)),
            export_max_workers=int(os.environ.get("ER_EXPORT_MAX_WORKERS", 6)),
            prefer_materialized_view=os.environ.get("ER_PREFER_MATERIALIZED_VIEW", "true").lower()
            == "true",
            bullish_return_threshold=float(os.environ.get("ER_BULLISH_RETURN_THRESHOLD", 0.02)),
            anomaly_severity_threshold=(
                float(os.environ["ER_ANOMALY_SEVERITY_THRESHOLD"])
                if "ER_ANOMALY_SEVERITY_THRESHOLD" in os.environ
                else None
            ),
        )

    def clear_cache(self, *, expired_only: bool = True) -> int:
        """
        Remove cached result files from the cache directory.

        Parameters
        ----------
        expired_only : bool, default True
            If True, only remove files older than ``cache_ttl_hours``.
            If False, remove all ``.json`` files in the cache directory.

        Returns
        -------
        int
            Number of files removed.
        """
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return 0

        removed = 0
        for cache_file in cache_path.glob("*.json"):
            try:
                if expired_only:
                    age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                    if age_hours <= self.cache_ttl_hours:
                        continue
                cache_file.unlink()
                removed += 1
            except OSError as e:
                logger.debug("Could not remove cache file %s: %s", cache_file, e)

        logger.info(
            "Cache cleanup (%s): removed %d files from %s",
            "expired only" if expired_only else "full purge",
            removed,
            cache_path,
        )
        return removed

    @property
    def caching_enabled(self) -> bool:
        """True if any form of result caching is active."""
        return self.enable_result_caching or self.enable_mcmc_caching


# ---------------------------------------------------------------------------
# Pipeline Result & Runner
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Container for intermediate and final results of the 8-phase workflow."""

    # Phase 1: Data
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_all: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_enriched: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    catalog: Optional[FeatureViewCatalog] = field(default_factory=dict)

    # Phase 2: Core Models
    mc: pd.DataFrame = field(default_factory=pd.DataFrame)
    pt: pd.DataFrame = field(default_factory=pd.DataFrame)
    kal: pd.DataFrame = field(default_factory=pd.DataFrame)
    beat: pd.DataFrame = field(default_factory=pd.DataFrame)
    anomaly_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    credit: pd.DataFrame = field(default_factory=pd.DataFrame)
    div_safety: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Phase 3: Market Models
    tri: pd.DataFrame = field(default_factory=pd.DataFrame)
    quad: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Phase 4: Statistics & Screening
    screens: Dict[str, pd.DataFrame] = field(default_factory=dict)
    sector_analytics: pd.DataFrame = field(default_factory=pd.DataFrame)
    mcmc_result: Dict[str, Any] = field(default_factory=dict)
    category_analytics: Dict[str, Any] = field(default_factory=dict)
    multi_hier: Dict[str, Any] = field(default_factory=dict)
    resampled_posterior: pd.DataFrame = field(default_factory=pd.DataFrame)
    corr_info: Dict[str, Any] = field(
        default_factory=lambda: {"correlation": None, "n_stocks": 0}
    )

    # Phase 5: Ensemble Alignment
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    strong: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Metadata
    id_coords: Any = None
    schema_metadata: Any = None
    feature_registry: Any = None
    mv_equities_spec: Any = None
    view_specs: Dict[str, Any] = field(default_factory=dict)

    # InferenceData (ArviZ)
    idata_mc: Any = None
    idata_beat: Any = None
    idata_credit: Any = None
    idata_anomaly: Any = None
    idata_category: Dict[str, Any] = field(default_factory=dict)
    idata_resampled: Any = None


class PipelineRunner:
    """
    Unified 8-phase workflow runner for PML expected returns analysis.

    Encapsulates the shared logic used by both expected_returns_v3 and v4.
    """

    def __init__(self, config: Any):
        self.cfg = config
        self.r = PipelineResult()

    def run_monte_carlo(self, df: pd.DataFrame):
        """

        :param df:
        """
        self.r.mc = run_monte_carlo_analysis(
            df,
            n_simulations=self.cfg.mc_simulations,
            max_stocks=self.cfg.mc_max_stocks,
        )

    def run_price_target(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, catalog: FeatureViewCatalog
    ):
        """

        :param df:
        :param feature_df:
        :param catalog:
        """
        self.r.pt = run_price_target_achievement(
            df, feature_df=feature_df, catalog=catalog
        )

    def run_kalman_filter(self, df: pd.DataFrame):
        """

        :param df:
        """
        self.r.kal = run_kalman_filter(df)

    def run_earnings_beat(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, catalog: FeatureViewCatalog
    ):
        """

        :param df:
        :param feature_df:
        :param catalog:
        """
        self.r.beat = run_earnings_beat_analysis(
            df, feature_df=feature_df, catalog=catalog
        )

    def run_accounting_anomaly(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, catalog: FeatureViewCatalog
    ):
        """

        :param df:
        :param feature_df:
        :param catalog:
        """
        self.r.anomaly_results = run_accounting_anomaly_analysis(
            df,
            feature_df=feature_df,
            n_mcmc_samples=self.cfg.mcmc_samples,
            burn_in=self.cfg.mcmc_samples // 5,
            catalog=catalog,
        )
        self._cache_mcmc_dataframe(
            self.r.anomaly_results, "accounting_anomaly", self.cfg.mcmc_samples
        )

    def run_credit_risk(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, catalog: FeatureViewCatalog
    ):
        """

        :param df:
        :param feature_df:
        :param catalog:
        """
        self.r.credit = run_credit_risk_analysis(
            df,
            feature_df=feature_df,
            catalog=catalog,
            n_mcmc_samples=self.cfg.mcmc_samples,
            burn_in=self.cfg.mcmc_samples // 5,
        )
        self._cache_mcmc_dataframe(self.r.credit, "credit_risk", self.cfg.mcmc_samples)

    def run_dividend_safety(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, catalog: FeatureViewCatalog
    ):
        """

        :param df:
        :param feature_df:
        :param catalog:
        """
        self.r.div_safety = run_dividend_safety_analysis(
            df,
            feature_df=feature_df,
            catalog=catalog,
            n_mcmc_samples=self.cfg.mcmc_samples,
            burn_in=self.cfg.mcmc_samples // 5,
        )
        self._cache_mcmc_dataframe(self.r.div_safety, "dividend_safety", self.cfg.mcmc_samples)

    def _cache_mcmc_dataframe(
        self, result_df: pd.DataFrame, analysis_type: str, n_samples: int
    ) -> None:
        """Cache an MCMC analysis DataFrame to the mcmc_results subdirectory."""
        if not (self.cfg.enable_result_caching or self.cfg.enable_mcmc_caching):
            return
        try:
            cache_mcmc_result(result_df, analysis_type, n_samples, cache_dir=self.cfg.cache_dir)
        except Exception as e:
            logger.debug("Failed to cache %s results: %s", analysis_type, e)

    def run_parallel_mcmc(self, pt: pd.DataFrame):
        """Step 7a: Parallel MCMC return analysis with hierarchical MCMC."""
        self.r.mcmc_result = run_parallel_mcmc_return_analysis(
            pt,
            n_chains=self.cfg.mcmc_chains,
            n_samples=self.cfg.mcmc_samples,
            cache_dir=self.cfg.cache_dir,
            enable_caching=self.cfg.enable_result_caching
            or self.cfg.enable_mcmc_caching,
            cache_ttl_hours=self.cfg.cache_ttl_hours,
        )

    def enrich_quad_with_mcmc(self) -> None:
        """Re-enrich quad with risk-adjusted returns after MCMC completes.

        Mirrors the post-MCMC enrichment in ``expected_returns_v3._step_mcmc_return_analysis``.
        When ``mcmc_result`` is populated and ``quad`` is non-empty, re-runs
        ``build_quad_model_alignment`` with the MCMC result so that
        ``ensemble_return``, ``ensemble_return_shrunk``, ``mcmc_shrinkage``,
        and ``risk_adj_return`` columns are computed.
        """
        from probabilistic_ml_model.statistical_functions.ensemble_models import (
            build_quad_model_alignment,
        )

        r = self.r
        if r.mcmc_result and not r.quad.empty:
            r.quad = build_quad_model_alignment(
                r.tri,
                r.beat,
                beat_threshold=getattr(self.cfg, "beat_threshold", 0.50),
                credit=r.credit if not r.credit.empty else None,
                div_safety=r.div_safety if not r.div_safety.empty else None,
                anomaly=r.anomaly_results if not r.anomaly_results.empty else None,
                anomaly_severity_threshold=getattr(self.cfg, "anomaly_severity_threshold", None),
                mcmc_result=r.mcmc_result,
            )
            logger.info("Risk-adjusted returns computed for %d stocks", len(r.quad))

    def run_resampled_posterior(self, df: pd.DataFrame):
        """Step 7c: Bayesian resampled return posteriors."""
        self.r.resampled = run_resampled_posterior_analysis(df)


# ---------------------------------------------------------------------------
# Standalone MCMC result caching helper
# ---------------------------------------------------------------------------


def cache_mcmc_result(
    result_df: pd.DataFrame,
    analysis_type: str,
    n_samples: int,
    cache_dir: str = ".cache",
) -> None:
    """Cache an MCMC analysis DataFrame to the appropriate mcmc_results subdirectory.

    Parameters
    ----------
    result_df : pd.DataFrame
        The analysis result to cache.
    analysis_type : str
        One of ``"accounting_anomaly"``, ``"credit_risk"``, ``"dividend_safety"``.
    n_samples : int
        Number of MCMC samples used (for the cache key).
    cache_dir : str
        Root cache directory (default ``.cache``).
    """
    if result_df.empty:
        return
    from finance_ml.ml_workflow.v3.cache import (
        McmcReturnCacheKey,
        build_cache_path,
        dataframe_stable_checksum,
        save_json,
    )

    # Build id_cols list from columns actually present — 'isin' is canonical,
    # 'ticker' may be absent when the project uses 'isin' exclusively.
    _candidate_id_cols = ["isin", "ticker", "name"]
    _id_cols = [c for c in _candidate_id_cols if c in result_df.columns]
    checksum = dataframe_stable_checksum(result_df, id_cols=_id_cols)
    factory = {
        "accounting_anomaly": McmcReturnCacheKey.for_accounting_anomaly,
        "credit_risk": McmcReturnCacheKey.for_credit_risk,
        "dividend_safety": McmcReturnCacheKey.for_dividend_safety,
    }[analysis_type]
    key = factory(data_checksum=checksum, n_chains=8, n_samples=n_samples)
    path = build_cache_path(cache_dir, key.to_filename(), subdir=key.subdir)
    save_json(path, result_df.to_dict(orient="list"))
    logger.info("%s results cached → %s", analysis_type, path.name)


# ---------------------------------------------------------------------------
# Lazy imports to avoid circular / heavy-load at module level
# ---------------------------------------------------------------------------


def _get_schema_columns() -> dict[str, list[str]]:
    """Derive column lists from equities schema with hardcoded fallbacks."""
    from probabilistic_ml_model.data_utils import get_equities_schema

    schema = get_equities_schema()

    _MC_FALLBACK = ["price_target", "price_target_high", "price_target_low", "last_price"]
    _KAL_FALLBACK = ["last_price", "price_target"]
    _HP_FALLBACK = [
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
    _HPT_FALLBACK = [
        "price_target_1w_ago",
        "price_target_1m_ago",
        "price_target_3m_ago",
        "price_target_6m_ago",
        "price_target_mtd_ago",
        "price_target_qtd_ago",
        "price_target_1y_ago",
    ]
    _HPT_HIGH_FALLBACK = [
        "price_target_high_1w_ago",
        "price_target_high_1m_ago",
        "price_target_high_6m_ago",
        "price_target_high_mtd_ago",
        "price_target_high_3m_ago",
        "price_target_high_qtd_ago",
        "price_target_high_1y_ago",
        "price_target_high_ytd_ago",
    ]
    _HPT_LOW_FALLBACK = [
        "price_target_low_1w_ago",
        "price_target_low_1m_ago",
        "price_target_low_3m_ago",
        "price_target_low_6m_ago",
        "price_target_low_mtd_ago",
        "price_target_low_qtd_ago",
        "price_target_low_ytd_ago",
        "price_target_low_1y_ago",
    ]
    _HPT_MED_FALLBACK = [
        "price_target_median_1w_ago",
        "price_target_median_1m_ago",
        "price_target_median_3m_ago",
        "price_target_median_6m_ago",
        "price_target_median_mtd_ago",
        "price_target_median_qtd_ago",
        "price_target_median_ytd_ago",
        "price_target_median_1y_ago",
    ]

    if schema:
        role_cols: dict[str, list[str]] = {}
        for alias, meta in schema.items():
            role_cols.setdefault(meta["role"], []).append(alias)

        mc_required = [
            c
            for c in ["price_target", "price_target_high", "price_target_low", "last_price"]
            if c in schema
        ]
        kalman_required = [c for c in ["last_price", "price_target"] if c in schema]
        return {
            "mc_required": mc_required or _MC_FALLBACK,
            "kalman_required": kalman_required or _KAL_FALLBACK,
            "historical_prices": sorted(
                c for c, m in schema.items() if m["role"] == "historical_price"
            )
            or _HP_FALLBACK,
            "historical_targets": sorted(
                c for c, m in schema.items() if m["role"] == "historical_price_target"
            )
            or _HPT_FALLBACK,
            "historical_targets_high": sorted(
                c for c, m in schema.items() if m["role"] == "historical_price_target_high"
            )
            or _HPT_HIGH_FALLBACK,
            "historical_targets_low": sorted(
                c for c, m in schema.items() if m["role"] == "historical_price_target_low"
            )
            or _HPT_LOW_FALLBACK,
            "historical_targets_median": sorted(
                c for c, m in schema.items() if m["role"] == "historical_price_target_median"
            )
            or _HPT_MED_FALLBACK,
        }

    return {
        "mc_required": _MC_FALLBACK,
        "kalman_required": _KAL_FALLBACK,
        "historical_prices": _HP_FALLBACK,
        "historical_targets": _HPT_FALLBACK,
        "historical_targets_high": _HPT_HIGH_FALLBACK,
        "historical_targets_low": _HPT_LOW_FALLBACK,
        "historical_targets_median": _HPT_MED_FALLBACK,
    }


def _resolve_available_historical_cols(df: pd.DataFrame) -> dict[str, list[str]]:
    """Identify which historical price/target columns are present in *df*."""
    schema_cols = _get_schema_columns()
    return {
        "historical_prices": [c for c in schema_cols["historical_prices"] if c in df.columns],
        "historical_targets": [c for c in schema_cols["historical_targets"] if c in df.columns],
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


# ---------------------------------------------------------------------------
# R2: Historical drift enrichment helpers
# ---------------------------------------------------------------------------


def _safe_pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Compute ``((current - previous) / |previous|) * 100``, replacing ±inf with NaN."""
    prev = pd.to_numeric(previous, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = ((current - prev) / prev.abs()) * 100.0
    return result.replace([np.inf, -np.inf], np.nan)


def _compute_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Compute numeric spread ``high − low``."""
    return pd.to_numeric(high, errors="coerce") - pd.to_numeric(low, errors="coerce")


def _add_drift_columns(
    df: pd.DataFrame,
    current_col: str,
    horizons: list[tuple[str, str]],
    output_prefix: str,
) -> None:
    """Add percentage-drift columns to *df* for each horizon."""
    current = df.get(current_col)
    if current is None:
        return
    for horizon, hist_col in horizons:
        if hist_col in df.columns:
            df[f"{output_prefix}_{horizon}"] = _safe_pct_change(current, df[hist_col])


def _enrich_with_historical_target_drift(
    df: pd.DataFrame,
    hist_available: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Compute derived drift and spread-evolution columns from historical
    price and price target data.
    """
    hist_keys = [
        "historical_targets",
        "historical_targets_high",
        "historical_targets_low",
        "historical_targets_median",
        "historical_prices",
    ]
    if not any(hist_available[k] for k in hist_keys):
        logger.debug("No historical price/target columns found — skipping drift enrichment")
        return df

    # Consensus target drift
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

    # Spread evolution
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

    # Median target drift
    _add_drift_columns(
        df,
        current_col="price_target_median",
        horizons=[
            ("1m", "price_target_median_1m_ago"),
            ("3m", "price_target_median_3m_ago"),
        ],
        output_prefix="pt_median_drift",
    )

    # Historical price anchor
    anchor_chain = ["price_5d_ago", "price_1w_ago", "price_1m_ago"]
    anchor = pd.Series(np.nan, index=df.index, dtype=float)
    for col in anchor_chain:
        if col in df.columns and anchor.isna().any():
            anchor = anchor.fillna(pd.to_numeric(df[col], errors="coerce"))
    if anchor.notna().any():
        df["historical_price_anchor"] = anchor

    # Price momentum vs historical levels
    _add_drift_columns(
        df,
        current_col="last_price",
        horizons=[
            ("1m", "price_1m_ago"),
            ("3m", "price_3m_ago"),
        ],
        output_prefix="price_vs_historical",
    )

    # Target-vs-price convergence signal
    if "pt_drift_1m" in df.columns and "price_vs_historical_1m" in df.columns:
        df["target_vs_price_convergence_1m"] = df["pt_drift_1m"] - df["price_vs_historical_1m"]

    _DERIVED_PREFIXES = (
        "pt_drift_",
        "pt_spread_change_",
        "pt_median_drift_",
        "historical_price_anchor",
        "price_vs_historical_",
        "target_vs_price_convergence_",
    )
    n_derived = sum(1 for c in df.columns if c.startswith(_DERIVED_PREFIXES))
    logger.info("Historical target drift enrichment: %d derived columns added", n_derived)
    return df


def _log_historical_coverage(available: dict[str, list[str]]) -> None:
    """Log how many historical price/target columns were found."""
    total_found = sum(len(v) for v in available.values())
    logger.info(
        "Historical price/target coverage: %d columns available (%s)",
        total_found,
        ", ".join(f"{k}={len(v)}" for k, v in available.items()),
    )


# ---------------------------------------------------------------------------
# R1: Model runners
# ---------------------------------------------------------------------------


def run_monte_carlo_analysis(
    df: pd.DataFrame,
    n_simulations: int = 25_000,
    max_stocks: int = 10_000,
    use_historical_targets: bool = True,
) -> pd.DataFrame:
    """Run Monte Carlo price target simulation."""
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        monte_carlo_price_target_simulation,
    )

    mc_cols = _get_schema_columns()["mc_required"]
    missing = [c for c in mc_cols if c not in df.columns]
    if missing:
        logger.warning("MC simulation skipped — missing columns: %s", missing)
        return pd.DataFrame()

    sim_df = df.copy()
    hist_available = _resolve_available_historical_cols(sim_df)
    if use_historical_targets:
        _log_historical_coverage(hist_available)
        sim_df = _enrich_with_historical_target_drift(sim_df, hist_available)

    mc = monte_carlo_price_target_simulation(
        sim_df, n_simulations=n_simulations, max_stocks=max_stocks
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

    if not mc.empty:
        mc = compute_price_target_mc(mc, df)

    logger.info("Monte Carlo simulation: %d stocks processed", len(mc))
    return mc


def run_price_target_achievement(
    df: pd.DataFrame,
    use_historical_targets: bool = True,
    feature_df: pd.DataFrame | None = None,
    catalog: FeatureViewCatalog | None = None,
) -> pd.DataFrame:
    """Estimate probability of reaching consensus price targets.

    Column requirements resolved from ``FeatureViewCatalog``.
    """
    from probabilistic_ml_model.statistical_functions.probability_models import (
        PriceTargetAchievementModel,
    )

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

    if not pt.empty:
        pt = compute_price_target_prob_weighted(pt, df)

    logger.info("Price target achievement: %d stocks processed", len(pt))
    return pt


def run_kalman_filter(
    df: pd.DataFrame,
    use_historical_targets: bool = True,
) -> pd.DataFrame:
    """Apply Kalman filter to smooth noisy analyst price targets."""
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        kalman_filter_price_target,
    )

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

    if not kal.empty and "implied_return_kalman" in kal.columns:
        lower, upper = kal["implied_return_kalman"].quantile([0.01, 0.99])
        kal["implied_return_kalman"] = kal["implied_return_kalman"].clip(lower, upper)

    logger.info("Kalman filter: %d stocks processed", len(kal))
    return kal


def run_earnings_beat_analysis(
    df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    catalog: FeatureViewCatalog | None = None,
    *,
    strict_streak_merge: bool = False,
) -> pd.DataFrame:
    """Run enhanced three-layer Bayesian earnings beat probability model.

    Column requirements resolved from ``FeatureViewCatalog``.

    v0.9.8.2 — The EPS streak analysis (which produces ``map_estimate`` and
    ``model_confidence``) now runs *before* ``analyze_dataframe_enhanced`` and
    ``ResampledBeatProbabilityModel.analyze_dataframe`` so the streak-merge
    columns are present on both the enriched beat frame and the source
    ``df`` consumed by the resampled wrapper (see CHANGELOG §12.5).
    """
    from probabilistic_ml_model.statistical_functions.probability_models import (
        EarningsBeatProbabilityModel,
        EPSStreakAnalyzer,
        ResampledBeatProbabilityModel,
    )
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        bayesian_earnings_beat_model,
    )

    cat = catalog or get_feature_catalog()
    beat_df = auto_enrich_for_model(df.copy(), feature_df, "earnings_beat", cat)

    # --- §12.5: run streak analyzer FIRST so map_estimate / model_confidence
    # are present on the DFs consumed by analyze_dataframe_enhanced and the
    # downstream ResampledBeatProbabilityModel (which re-invokes the base
    # enhanced analyzer internally). Previously the streak merge happened
    # after these calls, leading to silent column drops and degraded
    # momentum prior tilt for ~15 % of the universe.
    streak_df = pd.DataFrame()
    try:
        streak_analyzer = EPSStreakAnalyzer()
        streak_df = streak_analyzer.analyze_dataframe(df)
        if not streak_df.empty and "isin" in streak_df.columns:
            prior_cols = ["map_estimate", "model_confidence"]
            merge_cols = [c for c in prior_cols if c in streak_df.columns]
            if merge_cols:
                beat_df = beat_df.merge(streak_df[["isin"] + merge_cols], on="isin", how="left")
                # Also stamp onto the original `df` passed to the resampled model
                df = df.merge(streak_df[["isin"] + merge_cols], on="isin", how="left")
    except Exception as e:
        logger.warning("EPS streak pre-merge failed: %s", e)

    model = EarningsBeatProbabilityModel()
    sector_col = "sector" if "sector" in beat_df.columns else "industry"
    beat = model.analyze_dataframe_enhanced(
        beat_df, sector_col=sector_col, strict_streak_merge=strict_streak_merge
    )
    logger.info("Earnings beat analysis: %d stocks processed", len(beat))

    # Merge remaining streak diagnostic columns (those not already on beat)
    if not streak_df.empty and "isin" in streak_df.columns:
        streak_cols = [c for c in streak_df.columns if c != "isin" and c not in beat.columns]
        if streak_cols:
            beat = beat.merge(streak_df[["isin"] + streak_cols], on="isin", how="left")
            logger.info("EPS streak enrichment: %d columns added", len(streak_cols))

    # Resampled technical priors — now sees the streak columns on `df`
    try:
        resampled_model = ResampledBeatProbabilityModel(base_model=model)
        # Resampled model uses ticker_col="isin" to match the main beat DataFrame
        # and we pass it as ticker_col so it's included in the output DataFrame
        resampled_df = resampled_model.analyze_dataframe(df, ticker_col="isin")

        # Ensure 'isin' is in resampled_df even if it was called 'ticker' by the model internal vars()
        if (
            not resampled_df.empty
            and "ticker" in resampled_df.columns
            and "isin" not in resampled_df.columns
        ):
            resampled_df = resampled_df.rename(columns={"ticker": "isin"})

        if not resampled_df.empty and "isin" in resampled_df.columns:
            resamp_cols = [
                c for c in resampled_df.columns if c != "isin" and c not in beat.columns
            ]
            if resamp_cols:
                beat = beat.merge(
                    resampled_df[["isin"] + resamp_cols], on="isin", how="left"
                )
                logger.info("Resampled beat enrichment: %d columns added", len(resamp_cols))
    except Exception as e:
        logger.warning("Resampled beat probability failed: %s", e)

    # Classical Bayesian earnings beat model
    try:
        bayesian_beat = bayesian_earnings_beat_model(df)
        if not bayesian_beat.empty and "isin" in bayesian_beat.columns:
            bay_cols = [
                c
                for c in bayesian_beat.columns
                if c != "isin" and c not in beat.columns
            ]
            if bay_cols:
                beat = beat.merge(
                    bayesian_beat[["isin"] + bay_cols], on="isin", how="left"
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
    """Run credit risk and ruin probability analysis.

    Column requirements resolved from ``FeatureViewCatalog``.
    """
    from probabilistic_ml_model.statistical_functions.probability_models import (
        CreditRiskProbabilityModel,
    )
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        calculate_ruin_probability,
    )

    cat = catalog or get_feature_catalog()
    credit_df = auto_enrich_for_model(df.copy(), feature_df, "credit_risk", cat)

    credit_model = CreditRiskProbabilityModel(n_mcmc_samples=n_mcmc_samples, burn_in=burn_in)
    credit = credit_model.analyze_dataframe(credit_df)

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

    # Hierarchical sector-level MCMC enrichment
    try:
        if "altman_z_score" in credit_df.columns:
            z_data = credit_df["altman_z_score"].dropna()
            if len(z_data) > 50:
                sector_mcmc = hierarchical_mcmc_by_sector(credit_df, "altman_z_score")
                # Unwrap ArviZ-wrapped result
                if "industry" in sector_mcmc and isinstance(sector_mcmc["industry"], dict):
                    sector_mcmc = sector_mcmc["industry"]
                sector_mean_map = {
                    s: v.get("posterior_mean", np.nan)
                    for s, v in sector_mcmc.items()
                    if isinstance(v, dict)
                }
                sector_col = "industry" if "industry" in credit.columns else "sector"
                if sector_col in credit.columns:
                    credit["sector_z_posterior_mean"] = credit[sector_col].map(sector_mean_map)
                    logger.info(
                        "Hierarchical MCMC credit risk: %d sectors enriched", len(sector_mean_map)
                    )
    except Exception as e:
        logger.warning("Hierarchical MCMC for credit risk failed: %s", e)

    # Ruin probability
    try:
        ruin = calculate_ruin_probability(credit_df)
        if not ruin.empty and not credit.empty and "isin" in ruin.columns:
            ruin_cols = [
                c for c in ruin.columns if c != "isin" and c not in credit.columns
            ]
            if ruin_cols:
                credit = credit.merge(ruin[["isin"] + ruin_cols], on="isin", how="left")
                logger.info("Ruin probability enrichment: %d columns added", len(ruin_cols))
    except Exception as e:
        logger.warning("Ruin probability calculation failed: %s", e)

    logger.info("Credit risk analysis: %d stocks processed", len(credit))
    return credit


def run_dividend_safety_analysis(
    df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    *,
    n_mcmc_samples: int = 8000,
    burn_in: int = 2000,
    high_payout_threshold: float = 0.80,
    min_coverage: float = 1.2,
    risk_category_thresholds: tuple[float, float, float] = (0.20, 0.40, 0.65),
    posterior_pseudo_count: float = 40.0,
    exclude_non_payers: bool = True,
    catalog: FeatureViewCatalog | None = None,
) -> pd.DataFrame:
    """Run dividend cut probability analysis (v3.10 per-isin MCMC).

    Column requirements resolved from ``FeatureViewCatalog``. Input is
    de-duplicated on ``isin`` so that posterior rows are unique per
    instrument.
    """
    from probabilistic_ml_model.statistical_functions.probability_models import (
        DividendCutProbabilityModel,
    )

    # v3.10: guarantee per-isin inference rows
    if "isin" in df.columns:
        df = df.drop_duplicates(subset="isin").reset_index(drop=True)

    cat = catalog or get_feature_catalog()
    div_df = auto_enrich_for_model(df.copy(), feature_df, "dividend_safety", cat)

    model = DividendCutProbabilityModel(
        high_payout_threshold=high_payout_threshold,
        min_coverage=min_coverage,
        n_mcmc_samples=n_mcmc_samples,
        burn_in=burn_in,
        risk_category_thresholds=risk_category_thresholds,
        posterior_pseudo_count=posterior_pseudo_count,
        exclude_non_payers=exclude_non_payers,
    )
    div_safety = model.analyze_dataframe(div_df)
    logger.info("Dividend safety analysis (v3.10 per-isin): %d stocks processed", len(div_safety))
    return div_safety


def run_accounting_anomaly_analysis(
    df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    *,
    severity_anomaly_weight: float = 0.67,
    severity_feature_weight: float = 0.33,
    multi_flag_threshold: int = 15,
    anomaly_z_threshold: float | None = None,
    tier_bins: list[float] | None = None,
    tier_labels: list[str] | None = None,
    n_mcmc_samples: int = 5000,
    burn_in: int = 1000,
    catalog: FeatureViewCatalog | None = None,
) -> pd.DataFrame:
    """Run standalone accounting anomaly detection and analytics.

    Column requirements resolved from ``FeatureViewCatalog``.
    """
    from probabilistic_ml_model.statistical_functions.probability_models import (
        AccountingAnomalyProbabilityModel,
    )
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        mcmc_student_t,
    )

    cat = catalog or get_feature_catalog()
    anomaly_df = auto_enrich_for_model(df.copy(), feature_df, "accounting_anomaly", cat)

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

    # Student-t MCMC for anomaly score posterior
    try:
        if "accounting_anomaly_score" in result.columns:
            anomaly_series = result["accounting_anomaly_score"].dropna()
            if len(anomaly_series) > 30:
                anomaly_scores = np.asarray(anomaly_series, dtype=float)
                mu_samples, df_samples = mcmc_student_t(anomaly_scores)
                result["anomaly_posterior_location"] = float(np.mean(mu_samples))
                logger.info(
                    "MCMC anomaly posterior: location=%.3f", float(np.mean(mu_samples))
                )
    except Exception as e:
        logger.warning("MCMC anomaly posterior failed: %s", e)

    logger.info("Accounting anomaly analysis: %d stocks processed", len(result))
    return result


# ---------------------------------------------------------------------------
# Phase 4 runners (screening, resampled posterior, category analytics, MCMC)
# ---------------------------------------------------------------------------


def run_stock_screening(
    df_all: pd.DataFrame,
    *,
    min_pct: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Run all stock screening strategies on the full feature set."""
    from probabilistic_ml_model.statistical_functions.screening import (
        create_enhanced_screener,
        create_sector_relative_ranking,
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

    screens: dict[str, pd.DataFrame] = {}

    def _adaptive_screen_fallback(
        df_all_: pd.DataFrame,
        screen_result: pd.DataFrame,
        screen_name: str,
        min_pct_: float = 1.0,
        fallback_percentile: float = 90.0,
    ) -> pd.DataFrame:
        if (
            not screen_result.empty
            and 100.0 * len(screen_result) / max(len(df_all_), 1) >= min_pct_
        ):
            return screen_result
        score_cols = [
            c
            for c in ["composite_score", "expected_upside_pt", "implied_return_mc"]
            if c in df_all_.columns
        ]
        if not score_cols:
            return screen_result
        score_col = score_cols[0]
        threshold = df_all_[score_col].quantile(fallback_percentile / 100.0)
        fallback = df_all_[df_all_[score_col] >= threshold].copy()
        logger.info(
            "Adaptive screening: %s fallback using %s >= %.2f → %d stocks",
            screen_name,
            score_col,
            threshold,
            len(fallback),
        )
        return fallback

    _SCREEN_RUNNERS: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        ("quality", lambda: create_enhanced_screener(df_all)),
        ("earnings_quality", lambda: screen_earnings_quality(df_all)),
        ("value", lambda: screen_value_opportunities(df_all)),
        ("growth", lambda: screen_growth_momentum(df_all)),
        ("garp", lambda: screen_garp_opportunities(df_all)),
        ("dividend", lambda: screen_dividend_quality(df_all)),
        ("healthy", lambda: screen_financial_health(df_all)),
        ("valuation_reversion", lambda: screen_valuation_reversion_candidates(df_all)),
        ("integrity_growth", lambda: screen_integrity_filtered_growth(df_all)),
        ("high_yield_safe", lambda: screen_high_yield_safe_dividends(df_all)),
        ("low_vol_quality", lambda: screen_low_volatility_quality(df_all)),
        ("fcf_compounders", lambda: screen_fcf_growth_compounders(df_all)),
        ("total_return_leaders", lambda: screen_total_return_leaders(df_all)),
    ]

    for name, runner in _SCREEN_RUNNERS:
        try:
            screens[name] = runner()
            logger.info("%s screen: %d stocks", name, len(screens[name]))
            if name in ("quality", "dividend"):
                pct = 100.0 * len(screens[name]) / max(len(df_all), 1)
                if pct < min_pct:
                    logger.warning(
                        "%s screen returned %d stocks — applying adaptive fallback.",
                        name,
                        len(screens[name]),
                    )
                    screens[name] = _adaptive_screen_fallback(
                        df_all, screens[name], name, min_pct_=min_pct
                    )
        except Exception as e:
            logger.warning("%s screening failed: %s", name, e)

    # Sector-relative ranking
    try:
        screens["sector_relative"] = create_sector_relative_ranking(
            df_all,
            metric="composite_score"
            if "composite_score" in df_all.columns
            else "expected_upside_pt",
        )
        logger.info("Sector-relative ranking: %d stocks", len(screens["sector_relative"]))
    except Exception as e:
        logger.warning("Sector-relative ranking failed: %s", e)

    return screens


def hierarchical_mcmc_by_sector(
    df: pd.DataFrame,
    feature: str,
    sector_col: str = "industry",
    n_samples: int = 5000,
) -> dict:
    """Hierarchical MCMC: estimate sector-level means with pooling toward global mean.

    Thin wrapper that delegates to
    ``statistical_models.hierarchical_mcmc_by_sector``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature : str
        Feature name to analyze.
    sector_col : str, default 'industry'
        Column name for sector grouping.
    n_samples : int, default 5000
        Number of MCMC samples.

    Returns
    -------
    dict
        Dictionary mapping sectors to posterior statistics.
    """
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        hierarchical_mcmc_by_sector as _hierarchical_mcmc_by_sector,
    )

    return _hierarchical_mcmc_by_sector(
        df, feature, sector_col=sector_col, n_samples=n_samples
    )


def hierarchical_mcmc_multi_level(
    df: pd.DataFrame,
    feature: str,
    group_cols: list[str] | None = None,
    n_samples: int = 5000,
    min_group_size: int = 50,
    shrinkage_strength: float = 40.0,
) -> dict:
    """Multi-level hierarchical MCMC with nested category pooling.

    Thin wrapper that delegates to
    ``statistical_models.hierarchical_mcmc_multi_level``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature and categorical columns.
    feature : str
        Numeric feature to estimate.
    group_cols : list[str] or None, optional
        Categorical columns to group by.  Defaults to all available
        columns from ``_HIERARCHICAL_CATEGORY_COLS``.
    n_samples : int, default 5000
        Number of posterior MCMC draws per group.
    min_group_size : int, default 50
        Minimum observations per group.  Smaller groups get stronger
        shrinkage toward the parent mean.
    shrinkage_strength : float, default 40.0
        Controls the pooling intensity (higher = more shrinkage).
        Effective shrinkage = n / (n + shrinkage_strength).

    Returns
    -------
    dict
        Nested dictionary with global, levels, cross_level_summary, and
        optionally inference_data.
    """
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        hierarchical_mcmc_multi_level as _hierarchical_mcmc_multi_level,
    )

    return _hierarchical_mcmc_multi_level(
        df,
        feature,
        group_cols=group_cols,
        n_samples=n_samples,
        min_group_size=min_group_size,
        shrinkage_strength=shrinkage_strength,
    )


def run_resampled_posterior_analysis(
    df: pd.DataFrame,
    freq: str = "1QE",
) -> pd.DataFrame:
    """Compute Bayesian resampled return posteriors from historical price snapshots."""
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        resampled_posterior_returns,
    )

    try:
        result_df, idata = resampled_posterior_returns(df, freq=freq)
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


def _analyze_single_category(
    df: pd.DataFrame,
    cat_name: str,
    available: list[str],
    use_mcmc: bool,
    n_mcmc_samples: int,
    burn_in: int,
) -> tuple[str, dict]:
    """Analyze a single feature category (designed for parallel execution)."""
    from probabilistic_ml_model.statistical_functions.probability_models import (
        CategoryProbabilityAnalyzer,
    )
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        calculate_conditional_probabilities,
        fit_distributions_by_category,
        run_category_probability_analytics,
    )

    cat_results = run_category_probability_analytics(
        df, cat_name, available, n_simulations=10_000
    )

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

    try:
        dist_results = fit_distributions_by_category(df, cat_name, available)
        if dist_results:
            cat_results["distribution_fits"] = dist_results
    except Exception as e:
        logger.debug("Distribution fitting skipped for %s: %s", cat_name, e)

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
    """Run per-category Bayesian probability analytics.

    Supports parallelization via joblib, feature budget control, and
    file-based result caching.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    categories : dict, optional
        Feature categories to analyze.
    use_mcmc : bool
        Whether to run MCMC sampling.
    n_mcmc_samples : int
        Number of MCMC posterior samples.
    burn_in : int
        MCMC burn-in samples.
    n_jobs : int
        Parallel jobs (-1 = all cores, 1 = sequential).
    max_features_per_category : int
        Max features per category (0 = no limit).
    cache_dir : str
        Directory for result caching.
    enable_caching : bool
        Whether to use file-based caching.
    cache_ttl_hours : float
        Cache time-to-live in hours.
    """
    from finance_ml.ml_workflow.v3.cache import (
        CategoryAnalyticsCacheKey,
        dataframe_stable_checksum,
        build_cache_path,
        load_json,
        save_json,
    )
    from finance_ml.ml_workflow.v3.utils import run_parallel_or_sequential
    from probabilistic_ml_model.data_utils import load_feature_categories_from_db

    # Deduplicate input
    df = df.drop_duplicates(subset="isin") if "isin" in df.columns else df

    if categories is None:
        categories = load_feature_categories_from_db() or {}

    if not categories:
        logger.warning("No feature categories available — skipping category analytics")
        return {}

    # Guard against invalid MCMC parameters
    if n_mcmc_samples < 1:
        n_mcmc_samples = 5000
    if burn_in < 0:
        burn_in = 1000
    if burn_in >= n_mcmc_samples:
        burn_in = n_mcmc_samples // 5

    # --- Check cache ---
    cache_path = None
    if enable_caching and cache_dir:
        _candidate_id_cols = ["isin", "ticker", "name"]
        _id_cols = [c for c in _candidate_id_cols if c in df.columns]
        checksum = dataframe_stable_checksum(df, id_cols=_id_cols)
        key = CategoryAnalyticsCacheKey(
            data_checksum=checksum,
            n_categories=len(categories),
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
    for cat_name, features in categories.items():
        available = [f for f in features if f in df.columns]
        available = [f for f in available if pd.api.types.is_numeric_dtype(df[f])]
        if len(available) < 2:
            continue
        # Feature budget control: keep highest-variance features
        if 0 < max_features_per_category < len(available):
            variances = df[available].var().sort_values(ascending=False)
            available = variances.head(max_features_per_category).index.tolist()
            logger.info(
                "Sampling budget: %s trimmed to %d features (from %d)",
                cat_name,
                len(available),
                len(features),
            )
        category_tasks.append((cat_name, available))

    # Parallel or sequential execution
    def _safe_analyze_task(
        task: tuple[str, list[str]],
    ) -> tuple[str, dict | None]:
        cat_name, available = task
        try:
            return _analyze_single_category(
                df, cat_name, available, use_mcmc, n_mcmc_samples, burn_in
            )
        except Exception as e:
            logger.warning("Category analytics failed for %s: %s", cat_name, e)
            return cat_name, None

    task_results = run_parallel_or_sequential(
        category_tasks, n_jobs=n_jobs, worker=_safe_analyze_task
    )

    results: dict[str, dict] = {}
    for cat_name, cat_result in task_results:
        if not cat_result:
            continue
        results[cat_name] = cat_result
        logger.info(
            "Category %s: %d features analyzed",
            cat_name,
            cat_result.get("features_analyzed", 0),
        )

    # Save to cache
    if cache_path is not None and results:
        try:
            save_json(cache_path, results)
            logger.info("Category analytics cached → %s", cache_path.name)
        except Exception as e:
            logger.debug("Failed to save category analytics cache: %s", e)

    return results


def run_parallel_mcmc_return_analysis(
    pt: pd.DataFrame,
    n_chains: int = 8,
    n_samples: int = 5_000,
    *,
    cache_dir: str = ".cache",
    enable_caching: bool = True,
    cache_ttl_hours: float = 24.0,
) -> dict:
    """Run parallel MCMC on price-target expected returns (``implied_return_pt``)
    with Gelman-Rubin diagnostic.

    Also runs hierarchical multi-level MCMC by sector and fits a
    Student-t distribution via ``mcmc_student_t`` when the summary
    DataFrame contains an ``industry`` column.

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
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        mcmc_student_t,
        parallel_mcmc_chains,
    )

    if pt.empty or "implied_return_pt" not in pt.columns:
        return {}

    data = np.asarray(pt["implied_return_pt"].dropna(), dtype=float)
    if len(data) < -95:
        logger.warning("Parallel MCMC skipped — insufficient data (%d)", len(data))
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
            # Strip non-serializable keys (e.g. inference_data) before caching
            serializable = {
                k: v for k, v in result.items() if k != "inference_data"
            }
            save_json(cache_path, serializable)
            logger.info("MCMC return analysis cached → %s", cache_path.name)
        except Exception as e:
            logger.debug("Failed to save MCMC return cache: %s", e)

    return result


# ---------------------------------------------------------------------------
# Analytical helpers (used by Phase 5)
# ---------------------------------------------------------------------------


def compute_derived_price_target(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    price_col: str = "last_price",
    return_col: str = "implied_return_mc",
    output_col: str = "price_target_derived",
) -> pd.DataFrame:
    """Calculate a derived price target from a return-percentage column."""
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
    pt: pd.DataFrame, source_df: pd.DataFrame, **kw
) -> pd.DataFrame:
    """Calculate price target from probability-weighted return."""
    return compute_derived_price_target(
        pt,
        source_df,
        return_col="implied_return_pt",
        output_col="price_target_prob_weighted",
        **kw,
    )


def compute_price_target_mc(pt: pd.DataFrame, source_df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Calculate price target from Monte Carlo expected upside."""
    return compute_derived_price_target(
        pt, source_df, output_col="price_target_mc", **kw
    )


def filter_quality_stocks(summary: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality screening to the expected returns summary.

    Enriches the summary with a ``quality_tier`` from composite scoring
    and flags financially healthy stocks.

    When probabilistic model columns are present in the summary they are
    merged into *source_df* before scoring so that
    ``rank_stocks_by_composite_score`` can use model-aware weights.
    Quality tier bins use six levels for finer granularity:
    [0, 20, 40, 55, 70, 85, 100].
    """
    from probabilistic_ml_model.statistical_functions.screening import (
        rank_stocks_by_composite_score,
    )

    if summary.empty or source_df.empty:
        return summary

    # Enrich source_df with probabilistic columns from summary
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
                valid_scores.min(),
                valid_scores.quantile(0.10),
                valid_scores.quantile(0.30),
                valid_scores.quantile(0.50),
                valid_scores.quantile(0.70),
                valid_scores.quantile(0.90),
                valid_scores.max(),
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
            "  Financially healthy (score ≥ 45): %d stocks",
            summary["financially_healthy"].sum(),
        )

    return summary


def compute_return_zscore_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Add industry-relative z-scores and percentile ranks for key return metrics."""
    from probabilistic_ml_model.optimized_ops import vectorized_percentile_rank, vectorized_zscore

    if summary.empty:
        return summary

    return_cols = [
        c
        for c in ["implied_return_mc", "implied_return_kalman", "implied_return_pt"]
        if c in summary.columns
    ]
    group_col = "industry" if "industry" in summary.columns else None

    if return_cols:
        summary = vectorized_zscore(summary, return_cols, group_col=group_col)
        summary = vectorized_percentile_rank(summary, return_cols, group_col=group_col)
        logger.info("Added z-scores and percentile ranks for %d return metrics", len(return_cols))

    return summary


def _write_viz(
    fig,
    output_dir,
    filename: str,
    *,
    fmt: str = "html",
    dpi: int = 150,
) -> None:
    """Write a visualization figure to disk and log success."""
    from pathlib import Path

    dest = Path(output_dir) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        fig.write_html(dest)
    else:
        fig.savefig(dest, dpi=dpi, bbox_inches="tight")
    logger.info("   ✓ %s", filename)


# ---------------------------------------------------------------------------
# R4: Export logic
# ---------------------------------------------------------------------------


def _export_single_table(
    df: pd.DataFrame,
    table: str,
    *,
    _previous_hashes: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """Export a single DataFrame to the analytics schema."""
    from probabilistic_ml_model.data_utils import (
        ExportConfig,
        export_to_db,
        reorder_with_identifiers,
    )
    from probabilistic_ml_model.optimized_ops import dataframe_hash

    try:
        if _previous_hashes is not None:
            current_hash = dataframe_hash(df)
            if _previous_hashes.get(table) == current_hash:
                logger.info("Export skipped (unchanged): analytics.%s", table)
                return table, f"analytics.{table} (cached)"
            _previous_hashes[table] = current_hash

        reordered_df = reorder_with_identifiers(df)
        cfg = ExportConfig(table_name=table)
        export_to_db(reordered_df, cfg)
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
    """Export all expected returns analytics to the ``analytics`` schema."""
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    exports: dict[str, str] = {}

    _previous_hashes: dict[str, str] = {}
    _hash_file = Path(output_dir) / ".export_hashes.json"
    if _hash_file.exists():
        try:
            _previous_hashes = json.loads(_hash_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    _EXPORT_PAIRS: list[tuple[pd.DataFrame | None, str]] = [
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
            _EXPORT_PAIRS.append((screen_df, table))

    valid_pairs = [
        (df_, table) for df_, table in _EXPORT_PAIRS if df_ is not None and not df_.empty
    ]

    if max_workers > 1 and len(valid_pairs) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _export_single_table,
                    df_,
                    table,
                    _previous_hashes=_previous_hashes,
                ): table
                for df_, table in valid_pairs
            }
            for future in as_completed(futures):
                table_name, dest = future.result()
                if dest:
                    exports[table_name] = dest
    else:
        for df_, table in valid_pairs:
            table_name, dest = _export_single_table(
                df_,
                table,
                _previous_hashes=_previous_hashes,
            )
            if dest:
                exports[table_name] = dest

    try:
        _hash_file.write_text(json.dumps(_previous_hashes, indent=2))
    except OSError:
        pass

    return exports


# ---------------------------------------------------------------------------
# Cross-model analytics
# ---------------------------------------------------------------------------


def compute_cross_model_correlation(mc: pd.DataFrame, kal: pd.DataFrame) -> dict:
    """Compute correlation and copula dependency between MC and Kalman returns."""
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        fit_gaussian_copula,
    )

    if mc.empty or kal.empty:
        return {"correlation": None, "n_stocks": 0}

    mc_cols = {"isin", "implied_return_mc"}
    kal_cols = {"isin", "implied_return_kalman"}
    if not mc_cols.issubset(mc.columns) or not kal_cols.issubset(kal.columns):
        return {"correlation": None, "n_stocks": 0}

    merged = mc[["isin", "implied_return_mc"]].merge(
        kal[["isin", "implied_return_kalman"]],
        on="isin",
    )
    if len(merged) < 10:
        return {"correlation": None, "n_stocks": len(merged)}

    corr = merged[["implied_return_mc", "implied_return_kalman"]].corr().iloc[0, 1]
    result: dict = {"correlation": float(corr), "n_stocks": len(merged)}

    if len(merged) > 50:
        try:
            copula = fit_gaussian_copula(
                merged, features=["implied_return_mc", "implied_return_kalman"]
            )
            if copula:
                result["tail_dependence"] = copula.get("tail_dependence")
        except Exception as e:
            logger.debug("Copula fit skipped: %s", e)

    return result
