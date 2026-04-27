"""
Expected Returns Analytics Module (v4.0)

Automated pipeline for expected returns analysis using the v4.0+ analytics platform:

**Core PML Models:**
- **Monte Carlo Simulation** — Probabilistic upside/downside distributions with historical target drift
- **Price Target Achievement** — Probability-weighted expected returns with analyst sentiment & risk adjustment
- **Kalman Filtered Targets** — Noise-reduced price target signals with momentum-informed priors
- **Earnings Beat Analysis** — Three-layer Bayesian earnings beat probability with quality filters
- **Credit Risk Analysis** — Bayesian distress estimation with debt trajectory & balance sheet strength
- **Dividend Safety Analysis** — Dividend cut probability with FCF coverage & leverage signals
- **Accounting Anomaly Detection** — Multi-layered statistical anomaly detection with Mahalanobis distance

**Probabilistic Linear Market Model (MM) Regression**
- **Monte Carlo Simulation** — Probabilistic upside/downside distributions with historical target drift
- **Price Target Achievement** — Probability-weighted expected returns with analyst sentiment & risk adjustment
- **Kalman Filtered Targets** — Noise-reduced price target signals with momentum-informed priors
- **DCF Price Target Model** — discounted cash flow regression

**Probabilistic ML Ensembles**,
- **Prior Probability Distributions P(a, b, e)**
- **Likelihood Function P(Y| a, b, e, X)**
- **Marginal Likelihood Function P(Y|X)**
- **Posterior Probability Distributions P(a, b, e| X, Y)**
- **Multi-Level Hierarchical MCMC** — Cross-category shrinkage (region, country, sector, industry, style, size)
- **Feature View Posterior Panels** — Per-view InferenceData with ArviZ diagnostics


**Statistical Functions:**
- **Bayesian Category Analysis** — Per-feature-category posterior estimation
- **Gaussian Copula Dependency** — Tail dependence & joint distribution modeling
- **Parallel MCMC Chains** — Gelman-Rubin convergence diagnostics
- **Resampled Posterior Returns** — Bayesian technical resampling from historical snapshots
- **Student-t MCMC** — Heavy-tail robust posterior inference
- **Distribution Fitting** — AIC-based best-fit selection (Normal, Student-t, Skew-normal, Laplace)
- **Category-Level Distributions** — Per-category credible intervals & posterior means
- **Conditional Probability Analysis** — Feature-level P(anomaly | conditions)
- **Risk Metrics** — VaR, CVaR, downside deviation, gain/loss ratio


**Stock Screening:**
- **Undervalued Stocks Screening** — Investment opportunities with low P/E and high ROE
- **Earnings Quality Screening** — EPS consistency, GAAP divergence, revision momentum
- **Accounting Anomaly Screening** — Financial statement quality & consistency
- **Value Opportunities Screening** — Valuation reversion candidates
- **Growth Momentum Screening** — Revenue/EPS acceleration with profitability filters
- **GARP** — Growth at a reasonable price
- **Dividend Quality Screening** — Yield safety with coverage & streak metrics
- **Financial Health Screening** — Altman Z-score, Piotroski F-score, distress risk
- **Integrity-Filtered Growth Screening** — Accounting quality & growth alignment
- **High-Yield Safe Dividends** — Sustainable yield with leverage constraints
- **Low-Volatility Quality** — Beta stability with profitability
- **FCF Compounders** — Free cash flow growth consistency
- **Total Return Leaders** — Price appreciation + dividend yield

**Data Sources (v4.0 — Equities MV + Feature Views):**
- `public.mv_equities` — Core equities data via `load_equities_data_from_db`
- `public.mv_all_stock_features` — Full feature superset via `load_feature_data_from_db`
- `public.equities_schema_metadata` — Dynamic column discovery via `get_equities_schema`
- `public.calculated_features_registry` — Feature categories via `load_feature_categories_from_db`

Usage:
    python expected_returns_v4.py
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from requests.exceptions import RequestsDependencyWarning

from probabilistic_ml_model.pipeline_runners import PipelineRunner

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

logger = logging.getLogger(__name__)

# ── Patch arviz for PyMC 5.x compatibility with arviz >= 1.0 ─────────────────
try:
    from probabilistic_ml_model._pymc_arviz_compat import patch as _patch_arviz

    _patch_arviz()
except Exception:
    pass

# ── Configure pytensor to use Python-only mode ───────────────────────────────
# On Windows with official CPython 3.14 (MSVC-built), MinGW g++ cannot reliably
# link against python314.dll due to ABI incompatibility.  Additionally, libgcc ≥ 15
# triggers DLL load failures (https://github.com/pymc-devs/pytensor/issues/1398).
#
# Setting cxx="" disables C compilation entirely and uses PyTensor's pure-Python VM.
# This is functionally identical and officially supported — ~2-3× slower on large
# MCMC workloads, but avoids all compilation issues.
os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64,cxx="

# Ensure the stale compile cache doesn't interfere
_pytensor_cache = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "PyTensor")
if os.path.isdir(_pytensor_cache):
    import shutil

    try:
        shutil.rmtree(_pytensor_cache)
        logger.info("Cleared stale PyTensor cache: %s", _pytensor_cache)
    except OSError:
        pass

# Create/update .pytensorrc to match
_pytensorrc = os.path.join(os.path.expanduser("~"), ".pytensorrc")
try:
    with open(_pytensorrc, "w", encoding="utf-8", newline="\n") as _f:
        _f.write("[global]\n")
        _f.write("device = cpu\n")
        _f.write("floatX = float64\n")
        _f.write("cxx = \n\n")
        _f.write("[blas]\n")
        _f.write("ldflags = \n")
except OSError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """
    Centralized configuration for the expected returns analytics pipeline.

    All hardcoded magic numbers are surfaced here so they can be overridden
    from CLI arguments, environment variables, or test fixtures.
    """

    mc_simulations: int = 50_000
    mc_max_stocks: int = 10_000
    mcmc_chains: int = 6
    mcmc_samples: int = 50_000
    mcmc_burn_in: int = 2_000
    mcmc_tune: int = 2_000
    beat_threshold: float = 0.5
    use_mcmc: bool = True
    use_student_t: bool = False
    anomaly_z_threshold: float | None = None

    # Accounting anomaly model parameters
    anomaly_severity_anomaly_weight: float = 0.33
    anomaly_severity_feature_weight: float = 0.67
    anomaly_multi_flag_threshold: int = 20
    anomaly_tier_bins: list[float] | None = None
    anomaly_tier_labels: list[str] | None = None
    anomaly_n_mcmc_samples: int = 5_000
    anomaly_burn_in: int = 1_000

    # Credit risk / dividend safety MCMC parameters
    credit_n_mcmc_samples: int = 5_000
    credit_burn_in: int = 1_000
    dividend_n_mcmc_samples: int = 5_000
    dividend_burn_in: int = 1_000

    # Earnings beat model parameters
    beat_use_quality_adjustment: bool = True
    beat_use_momentum_prior: bool = True
    beat_momentum_prior_strength: float = 0.3

    # Historical target drift
    use_historical_targets: bool = True

    # Hierarchical MCMC parameters
    hier_mcmc_min_group_size: int = 50
    hier_mcmc_shrinkage_strength: float = 10.0
    hier_mcmc_group_cols: list[str] | None = None

    # Screening thresholds
    screening_min_pct: float = 0.01

    # Performance tuning
    n_jobs: int = -1
    max_features_per_category: int = 100
    enable_result_caching: bool = True
    cache_dir: str = ".cache/pipeline"
    export_max_workers: int = 4

    # Paths
    output_dir: str = "outputs"
    log_file: str | None = "logs/expected_returns_pipeline_v4.log"
    log_level: int = logging.INFO

    # v4.0: Probabilistic linear regression / DCF model toggles
    enable_plr_model: bool = False
    enable_dcf_model: bool = False
    plr_samples: int = 500
    plr_tune: int = 500
    plr_chains: int = 2
    plr_cores: int = 1
    plr_max_obs: int = 500
    plr_max_treedepth: int = 8
    dcf_samples: int = 2_000
    dcf_tune: int = 1_000
    dcf_chains: int = 2
    dcf_cores: int = 1

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build config from environment variables with sensible defaults."""
        return cls(
            mc_simulations=int(os.environ.get("ER_MC_SIMULATIONS", 50_000)),
            mc_max_stocks=int(os.environ.get("ER_MC_MAX_STOCKS", 10_000)),
            mcmc_chains=int(os.environ.get("ER_MCMC_CHAINS", 6)),
            mcmc_samples=int(os.environ.get("ER_MCMC_SAMPLES", 50_000)),
            mcmc_burn_in=int(os.environ.get("ER_MCMC_BURN_IN", 2_000)),
            mcmc_tune=int(os.environ.get("ER_MCMC_TUNE", 2_000)),
            use_mcmc=os.environ.get("ER_USE_MCMC", "true").lower() == "true",
            use_student_t=os.environ.get("ER_USE_STUDENT_T", "true").lower() == "true",
            output_dir=os.environ.get("ER_OUTPUT_DIR", "outputs"),
            log_file=os.environ.get("ER_LOG_FILE", "logs/expected_returns_pipeline.log"),
            n_jobs=int(os.environ.get("ER_N_JOBS", os.environ.get("N_JOBS", -1))),
            max_features_per_category=int(os.environ.get("ER_MAX_FEATURES_PER_CATEGORY", 100)),
            enable_result_caching=os.environ.get("ER_ENABLE_CACHING", "true").lower() == "true",
            cache_dir=os.environ.get(
                "ER_CACHE_DIR", os.environ.get("CACHE_DIR", ".cache/pipeline")
            ),
            export_max_workers=int(os.environ.get("ER_EXPORT_MAX_WORKERS", 4)),
            enable_plr_model=os.environ.get("ER_ENABLE_PLR", "true").lower() == "true",
            enable_dcf_model=os.environ.get("ER_ENABLE_DCF", "false").lower() == "false",
            plr_samples=int(os.environ.get("ER_PLR_SAMPLES", 500)),
            plr_tune=int(os.environ.get("ER_PLR_TUNE", 500)),
            plr_chains=int(os.environ.get("ER_PLR_CHAINS", 2)),
            plr_cores=int(os.environ.get("ER_PLR_CORES", 1)),
            plr_max_obs=int(os.environ.get("ER_PLR_MAX_OBS", 500)),
            plr_max_treedepth=int(os.environ.get("ER_PLR_MAX_TREEDEPTH", 8)),
            dcf_samples=int(os.environ.get("ER_DCF_SAMPLES", 2_000)),
            dcf_tune=int(os.environ.get("ER_DCF_TUNE", 1_000)),
            dcf_chains=int(os.environ.get("ER_DCF_CHAINS", 2)),
            dcf_cores=int(os.environ.get("ER_DCF_CORES", 1)),
            anomaly_severity_anomaly_weight=float(
                os.environ.get("ER_ANOMALY_SEVERITY_ANOMALY_WEIGHT", 0.75)
            ),
            anomaly_severity_feature_weight=float(
                os.environ.get("ER_ANOMALY_SEVERITY_FEATURE_WEIGHT", 0.25)
            ),
            anomaly_multi_flag_threshold=int(os.environ.get("ER_ANOMALY_MULTI_FLAG_THRESHOLD", 10)),
            anomaly_n_mcmc_samples=int(os.environ.get("ER_ANOMALY_MCMC_SAMPLES", 5_000)),
            anomaly_burn_in=int(os.environ.get("ER_ANOMALY_BURN_IN", 1_000)),
            credit_n_mcmc_samples=int(os.environ.get("ER_CREDIT_MCMC_SAMPLES", 5_000)),
            credit_burn_in=int(os.environ.get("ER_CREDIT_BURN_IN", 1_000)),
            dividend_n_mcmc_samples=int(os.environ.get("ER_DIVIDEND_MCMC_SAMPLES", 5_000)),
            dividend_burn_in=int(os.environ.get("ER_DIVIDEND_BURN_IN", 1_000)),
            beat_use_quality_adjustment=os.environ.get(
                "ER_BEAT_USE_QUALITY_ADJUSTMENT", "true"
            ).lower()
            == "true",
            beat_use_momentum_prior=os.environ.get("ER_BEAT_USE_MOMENTUM_PRIOR", "true").lower()
            == "true",
            beat_momentum_prior_strength=float(
                os.environ.get("ER_BEAT_MOMENTUM_PRIOR_STRENGTH", 0.3)
            ),
            use_historical_targets=os.environ.get("ER_USE_HISTORICAL_TARGETS", "true").lower()
            == "true",
            hier_mcmc_min_group_size=int(os.environ.get("ER_HIER_MCMC_MIN_GROUP_SIZE", 50)),
            hier_mcmc_shrinkage_strength=float(
                os.environ.get("ER_HIER_MCMC_SHRINKAGE_STRENGTH", 10.0)
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Result Container
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BaselinePipelineResult:
    """
    Structured container for all pipeline step outputs.

    Replaces loose variables, providing a single typed object flowing through
    all pipeline phases.
    """

    # Catalog (was previously set in a custom __init__)
    catalog: Any = None

    # Phase 1: Data sources
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_all: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_enriched: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Phase 2: Core PML model outputs
    mc: pd.DataFrame = field(default_factory=pd.DataFrame)
    pt: pd.DataFrame = field(default_factory=pd.DataFrame)
    kal: pd.DataFrame = field(default_factory=pd.DataFrame)
    beat: pd.DataFrame = field(default_factory=pd.DataFrame)
    anomaly_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    credit: pd.DataFrame = field(default_factory=pd.DataFrame)
    div_safety: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Phase 3: Probabilistic linear market model outputs
    plr_result: Any = None  # InferenceData from ProbabilisticLinearRegressionModel
    dcf_result: Any = None  # InferenceData from DCF_PriceTargetModel

    # Phase 4: Statistical functions & screening
    screens: dict[str, pd.DataFrame] = field(default_factory=dict)
    resampled_posterior: pd.DataFrame = field(default_factory=pd.DataFrame)
    category_analytics: dict[str, dict] = field(default_factory=dict)
    mcmc_result: dict = field(default_factory=dict)

    # Phase 5: Ensemble alignment
    tri: pd.DataFrame = field(default_factory=pd.DataFrame)
    quad: pd.DataFrame = field(default_factory=pd.DataFrame)
    strong: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    corr_info: dict = field(default_factory=lambda: {"correlation": None, "n_stocks": 0})

    # Phase 6: InferenceData (ArviZ)
    idata_mc: Any = None
    idata_beat: Any = None
    idata_credit: Any = None
    idata_plr: Any = None
    idata_dcf: Any = None

    # Metadata
    id_coords: Any = None
    schema_metadata: Any = None
    feature_registry: Any = None
    mv_equities_spec: Any = None
    view_specs: dict = field(default_factory=dict)

    # Timing
    phase_timings: dict[str, float] = field(default_factory=dict)

    # Per-model detailed statistics (populated during Phase 2/5)
    model_statistics: dict[str, dict] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _log_and_print(msg: str = "", level: int = logging.INFO) -> None:
    """Log a message and print it to stdout for pipeline visibility."""
    logger.log(level, msg)
    print(msg)


def _timed_phase(phase_name: str):
    """Decorator/context-manager helper for phase timing."""

    class _Timer:
        def __init__(self):
            self.elapsed = 0.0

        def __enter__(self):
            self._start = time.perf_counter()
            _log_and_print(f"\n{'─' * 80}")
            _log_and_print(f"▶ {phase_name}")
            _log_and_print(f"{'─' * 80}")
            return self

        def __exit__(self, *exc):
            self.elapsed = time.perf_counter() - self._start
            _log_and_print(f"  ⏱ {phase_name} completed in {self.elapsed:.1f}s\n")
            return False

    return _Timer()


def _run_model_step(step_num: int, label: str, runner: callable) -> None:
    """Wrapper for running a model step with logging and error handling."""
    try:
        _log_and_print(f"  [{step_num}] {label}...")
        runner()
    except Exception as e:
        logger.error("Step %d (%s) failed: %s", step_num, label, e, exc_info=True)
        _log_and_print(f"  ⚠️ Step {step_num} ({label}) failed: {e}", logging.ERROR)


def _as_float_series(series: pd.Series) -> pd.Series:
    """Coerce a Series to float64, dropping non-numeric values."""
    return pd.to_numeric(series, errors="coerce").dropna().astype(float)


def compute_metric_statistics(series: pd.Series) -> dict | None:
    """
    Compute summary statistics for a numeric Series.

    Returns a dict with count, mean, median, std, min, max, q25, q75,
    positive_pct, and missing_pct.  Returns ``None`` if the series has
    no valid numeric values.
    """
    numeric = _as_float_series(series)
    if numeric.empty:
        return None
    total = len(series)
    return {
        "count": len(numeric),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "q25": float(numeric.quantile(0.25)),
        "q75": float(numeric.quantile(0.75)),
        "positive_pct": float((numeric > 0).sum() / len(numeric) * 100),
        "missing_pct": float((total - len(numeric)) / total * 100) if total else 0.0,
    }


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

        series = _as_float_series(df[col])

        shape_stats = {}
        if len(series) > 3:
            # Clip extreme values to prevent overflow in higher-moment calculations
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
# Per-Model Key Column Definitions (for statistics)
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_KEY_COLUMNS: dict[str, list[str]] = {
    "Monte Carlo": ["expected_upside_mc", "prob_positive_upside", "implied_return_mc"],
    "Price Target": ["achievement_probability", "expected_upside_pt", "implied_return_pt"],
    "Kalman Filter": ["expected_upside_kalman", "implied_return_kalman"],
    "Earnings Beat": ["prob_beat_given_momentum", "posterior_alpha", "posterior_beta"],
    "Accounting Anomaly": ["accounting_anomaly_score", "anomaly_severity_score"],
    "Credit Risk": ["distress_probability", "altman_z_score"],
    "Dividend Safety": ["dividend_cut_probability", "fcf_coverage_ratio"],
}


class BaselinePipeline:
    """
    Orchestrates the full v4.0 expected returns pipeline.

    Delegates to PML sub-models from ``probabilistic_ml_model``
    and statistical functions from the analytics layer.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.cfg = config or PipelineConfig.from_env()
        self.runner = PipelineRunner(self.cfg)
        # Use our own BaselinePipelineResult which has all fields including phase_timings
        self.result = BaselinePipelineResult()
        # Synchronize back to the runner so they share the same dataframes/results
        self.runner.r = self.result

    # ── Phase 1: Data Ingestion & Enrichment ──────────────────────────────────

    def phase_1_load_data(self) -> None:
        """Load equities data, schema metadata, and apply enrichment."""
        from probabilistic_ml_model.data_utils.data_utils import (
            backfill_feature_columns,
            load_equities_data_from_db,
            load_feature_data_from_db,
        )
        from probabilistic_ml_model.data_utils.feature_catalog import (
            get_feature_catalog,
        )
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            kalman_momentum_filter,
        )

        r = self.result

        # Initialize the schema-driven feature catalog (singleton)
        r.catalog = get_feature_catalog()
        if r.catalog._loaded:
            _log_and_print(
                f"✓ FeatureViewCatalog: {len(r.catalog.category_columns)} categories, "
                f"{len(r.catalog.view_columns)} views"
            )
        else:
            _log_and_print("⚠️ FeatureViewCatalog: using fallback column lists (DB unavailable)")

        # Step 1: mv_equities
        _df = load_equities_data_from_db()
        r.df = _df if _df is not None else pd.DataFrame()
        if r.df.empty:
            _log_and_print("✗ No data loaded from mv_equities. Check DB_URL.")
            return
        _log_and_print(f"✓ mv_equities: {len(r.df):,} stocks × {len(r.df.columns)} features")

        # Step 2: Feature views (17 vw_features_*)
        _df_all = load_feature_data_from_db()
        r.df_all = _df_all if _df_all is not None else pd.DataFrame()
        if not r.df_all.empty:
            _log_and_print(f"✓ mv_all_stock_features: {len(r.df_all):,} stocks")

        else:
            _log_and_print("⚠️ Feature views not loaded — using mv_equities as fallback")
            r.df_all = r.df.copy()

        # Step 3: Schema metadata & feature registry
        try:
            from probabilistic_ml_model.data_utils.inference_schema import (
                load_equities_schema_metadata_from_db,
                load_feature_registry_metadata_from_db,
                load_mv_equities_spec_from_db,
            )

            r.schema_metadata = load_equities_schema_metadata_from_db()
            r.feature_registry = load_feature_registry_metadata_from_db()
            r.mv_equities_spec = load_mv_equities_spec_from_db()
        except Exception as e:
            logger.debug("Schema metadata unavailable: %s", e)

        # Step 5: Backfill + Kalman momentum smoothing
        _KALMAN_MOMENTUM_COLS = [
            "price_momentum_1m",
            "price_momentum_3m",
            "price_momentum_6m",
            "price_momentum_1y",
            "price_momentum_5d",
        ]
        for attr in ("df", "df_all"):
            df_ref = getattr(r, attr)
            if not df_ref.empty:
                df_ref = backfill_feature_columns(df_ref)
                kcols = [c for c in _KALMAN_MOMENTUM_COLS if c in df_ref.columns]
                if kcols:
                    df_ref = kalman_momentum_filter(df_ref, momentum_cols=kcols)
                setattr(r, attr, df_ref)

        # Step 6: Historical target drift enrichment
        from probabilistic_ml_model.pipeline_runners import (
            _enrich_with_historical_target_drift,
            _resolve_available_historical_cols,
        )

        hist_available = _resolve_available_historical_cols(r.df)
        r.df_enriched = _enrich_with_historical_target_drift(r.df.copy(), hist_available)
        n_derived = len(r.df_enriched.columns) - len(r.df.columns)
        _log_and_print(f"✓ Historical drift enrichment: {n_derived} derived columns added")

    # ── Phase 2: Core PML Model Execution ─────────────────────────────────────

    def phase_2_core_models(self) -> None:
        """Execute all core PML models from probabilistic_ml_model.pymc_models."""

        from probabilistic_ml_model.pipeline_runners import (
            compute_price_target_mc,
            compute_price_target_prob_weighted,
        )

        r = self.result
        if r.df_enriched.empty:
            _log_and_print("⚠️ Phase 2 skipped — no enriched data available")
            return

        # Model step definitions: (step_num, label, runner_method)
        model_steps = [
            (7, "Monte Carlo Simulation", self._run_monte_carlo),
            (8, "Price Target Achievement", self._run_price_target),
            (9, "Kalman Filter", self._run_kalman_filter),
            (10, "Earnings Beat Analysis", self._run_earnings_beat),
            (11, "Accounting Anomaly Detection", self._run_accounting_anomaly),
            (12, "Credit Risk Analysis", self._run_credit_risk),
            (13, "Dividend Safety Analysis", self._run_dividend_safety),
        ]

        for step_num, label, runner_fn in model_steps:
            _run_model_step(step_num, label, runner_fn)

        # Post-processing: derived price targets
        if not r.mc.empty:
            r.mc = compute_price_target_mc(r.mc, r.df)
        if not r.pt.empty:
            r.pt = compute_price_target_prob_weighted(r.pt, r.df)

        # Collect per-model detailed statistics
        model_result_map = {
            "Monte Carlo": r.mc,
            "Price Target": r.pt,
            "Kalman Filter": r.kal,
            "Earnings Beat": r.beat,
            "Accounting Anomaly": r.anomaly_results,
            "Credit Risk": r.credit,
            "Dividend Safety": r.div_safety,
        }
        for model_name, model_df in model_result_map.items():
            if not model_df.empty:
                key_cols = _MODEL_KEY_COLUMNS.get(model_name, [])
                stats = compute_model_detailed_statistics(model_df, model_name, key_cols)
                if stats:
                    r.model_statistics[model_name] = stats
                    print_model_statistics(stats, model_name, show_sectors=False)

    def _run_monte_carlo(self) -> None:
        r = self.result
        self.runner.run_monte_carlo(r.df_enriched)
        if not r.mc.empty:
            _log_and_print(
                f"    ✓ {len(r.mc):,} stocks — mean upside: "
                f"{r.mc.get('expected_upside_mc', pd.Series([0])).mean():.1f}%"
            )

    def _run_price_target(self) -> None:
        r = self.result
        self.runner.run_price_target(r.df_enriched, r.df_all, r.catalog)
        if not r.pt.empty:
            _log_and_print(
                f"    ✓ {len(r.pt):,} stocks — mean P(achieve): "
                f"{r.pt.get('achievement_probability', pd.Series([0])).mean():.3f}"
            )

    def _run_kalman_filter(self) -> None:
        r = self.result
        self.runner.run_kalman_filter(r.df_enriched)
        if not r.kal.empty:
            _log_and_print(
                f"    ✓ {len(r.kal):,} stocks — mean filtered upside: "
                f"{r.kal.get('expected_upside_kalman', pd.Series([0])).mean():.1f}%"
            )

    def _run_earnings_beat(self) -> None:
        r = self.result
        self.runner.run_earnings_beat(r.df_enriched, r.df_all, r.catalog)
        if not r.beat.empty:
            _log_and_print(
                f"    ✓ {len(r.beat):,} stocks — mean P(beat): "
                f"{r.beat.get('prob_beat_given_momentum', pd.Series([0])).mean():.3f}"
            )

    def _run_accounting_anomaly(self) -> None:
        r = self.result
        self.runner.run_accounting_anomaly(r.df_enriched, r.df_all, r.catalog)
        if not r.anomaly_results.empty:
            _log_and_print(f"    ✓ {len(r.anomaly_results):,} stocks analyzed")

    def _run_credit_risk(self) -> None:
        r = self.result
        self.runner.run_credit_risk(r.df_enriched, r.df_all, r.catalog)
        if not r.credit.empty:
            _log_and_print(f"    ✓ {len(r.credit):,} stocks — credit risk assessed")

    def _run_dividend_safety(self) -> None:
        r = self.result
        self.runner.run_dividend_safety(r.df_enriched, r.df_all, r.catalog)
        if not r.div_safety.empty:
            _log_and_print(f"    ✓ {len(r.div_safety):,} stocks — dividend safety assessed")

    # ── Phase 3: Probabilistic Linear Market Model ────────────────────────────

    def phase_3_market_models(self) -> None:
        """Run ProbabilisticLinearRegressionModel and DCF_PriceTargetModel."""
        cfg = self.cfg

        if cfg.enable_plr_model:
            _run_model_step(14, "Probabilistic Linear Regression", self._run_plr)

        if cfg.enable_dcf_model:
            _run_model_step(15, "DCF Price Target Model", self._run_dcf)

    def _run_plr(self) -> None:
        from probabilistic_ml_model.pymc_models.ProbabilisticLinearRegressionModel import (
            ProbabilisticLinearRegression,
        )

        r, cfg = self.result, self.cfg
        plr = ProbabilisticLinearRegression()
        feature_cols = [
            c
            for c in [
                "expected_upside_mc",
                "expected_upside_kalman",
                "achievement_probability",
                "prob_positive_upside",
                "beta_1y",
                "altman_z_score",
            ]
            if c in r.summary.columns or c in r.df_all.columns
        ]
        source = r.summary if not r.summary.empty else r.df_all
        if len(feature_cols) < 2 or source.empty:
            _log_and_print("    ⚠️ PLR skipped — insufficient feature columns")
            return
        X = source[feature_cols].dropna()
        if len(X) <= 50:
            _log_and_print("    ⚠️ PLR skipped — insufficient data after dropna")
            return
        target_col = (
            "expected_upside_mc" if "expected_upside_mc" in X.columns else feature_cols[0]
        )
        y = X.pop(target_col).values
        r.plr_result = plr.fit(
            X=X.values,
            y=y,
            feature_names=list(X.columns),
            samples=cfg.plr_samples,
            tune=cfg.plr_tune,
            chains=cfg.plr_chains,
            cores=cfg.plr_cores,
            max_obs=cfg.plr_max_obs,
            max_treedepth=cfg.plr_max_treedepth,
        )
        _log_and_print(
            f"    ✓ PLR model fitted on {len(X)} observations, {len(X.columns)} features"
        )

    def _run_dcf(self) -> None:
        from probabilistic_ml_model.pymc_models.DCF_PriceTargetModel import (
            DCFPriceTarget,
        )

        r, cfg = self.result, self.cfg
        dcf = DCFPriceTarget()
        fcf_col = "free_cash_flow" if "free_cash_flow" in r.df_all.columns else None
        price_col = "current_price" if "current_price" in r.df_all.columns else None
        if not (fcf_col and price_col):
            _log_and_print("    ⚠️ DCF skipped — required columns not found")
            return
        dcf_data = r.df_all[[fcf_col, price_col]].dropna()
        if len(dcf_data) <= 20:
            _log_and_print("    ⚠️ DCF skipped — insufficient data after dropna")
            return
        r.dcf_result = dcf.fit(
            historical_fcf=dcf_data[fcf_col].values,
            market_prices=dcf_data[price_col].values,
            samples=cfg.dcf_samples,
            tune=cfg.dcf_tune,
            chains=cfg.dcf_chains,
            cores=cfg.dcf_cores,
        )
        _log_and_print(f"    ✓ DCF model fitted on {len(dcf_data)} observations")

    # ── Phase 4: Statistical Functions & Screening ────────────────────────────

    def phase_4_statistics_and_screening(self) -> None:
        """Run screening, resampled posteriors, category analytics, and parallel MCMC."""
        from probabilistic_ml_model.pipeline_runners import (
            run_category_probability_analysis,
            run_parallel_mcmc_return_analysis,
            run_resampled_posterior_analysis,
            run_stock_screening,
        )

        r, cfg = self.result, self.cfg

        def _step_screening() -> None:
            r.screens = run_stock_screening(r.df_all, min_pct=cfg.screening_min_pct)
            for name, sdf in r.screens.items():
                if not sdf.empty:
                    _log_and_print(f"    ✓ {name}: {len(sdf):,} stocks")

        def _step_resampled() -> None:
            r.resampled_posterior = run_resampled_posterior_analysis(r.df)
            if not r.resampled_posterior.empty:
                _log_and_print(f"    ✓ {len(r.resampled_posterior):,} stocks")

        def _step_category() -> None:
            r.category_analytics = run_category_probability_analysis(
                r.df_all,
                use_mcmc=cfg.use_mcmc,
                n_mcmc_samples=cfg.mcmc_burn_in * 5,
                burn_in=cfg.mcmc_burn_in,
                n_jobs=cfg.n_jobs,
                max_features_per_category=cfg.max_features_per_category,
                cache_dir=cfg.cache_dir,
                enable_caching=cfg.enable_result_caching,
                cache_ttl_hours=24,
            )
            _log_and_print(f"    ✓ {len(r.category_analytics)} categories analyzed")

        def _step_mcmc() -> None:
            r.mcmc_result = run_parallel_mcmc_return_analysis(r.mc, n_chains=cfg.mcmc_chains,
                                                              n_samples=cfg.mcmc_samples)
            if r.mcmc_result:
                _log_and_print(
                    f"    ✓ R̂={r.mcmc_result.get('r_hat', float('nan')):.4f}, "
                    f"converged={r.mcmc_result.get('converged', False)}"
                )

        _run_model_step(16, "Stock screening strategies", _step_screening)
        _run_model_step(17, "Resampled Bayesian posterior returns", _step_resampled)
        _run_model_step(18, "Per-category Bayesian probability analytics", _step_category)
        _run_model_step(19, "Parallel MCMC return analysis", _step_mcmc)

    # ── Phase 5: Ensemble Alignment & Summary ─────────────────────────────────

    def phase_5_ensemble_alignment(self) -> None:
        """Build cross-model alignment, summary, and hierarchical MCMC."""
        from probabilistic_ml_model.pipeline_runners import (
            compute_cross_model_correlation,
            compute_return_zscore_ranks,
            filter_quality_stocks,
        )
        from probabilistic_ml_model.statistical_functions.ensemble_models import (
            build_expected_returns_summary,
            build_quad_model_alignment,
            build_tri_model_alignment,
            extract_strong_consensus,
        )
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            hierarchical_mcmc_multi_level,
        )

        r, cfg = self.result, self.cfg

        # Step 20–21: Tri-model & quad-model alignment
        r.tri = build_tri_model_alignment(r.mc, r.kal, r.pt)
        r.strong = extract_strong_consensus(r.tri)
        r.quad = build_quad_model_alignment(
            r.tri,
            r.beat,
            beat_threshold=cfg.beat_threshold,
            credit=r.credit if not r.credit.empty else None,
            div_safety=r.div_safety if not r.div_safety.empty else None,
            anomaly=r.anomaly_results if not r.anomaly_results.empty else None,
            mcmc_result=r.mcmc_result if r.mcmc_result else None,
        )

        if not r.tri.empty:
            _log_and_print(f"  ✓ Tri-model: {len(r.tri):,} stocks")
        if not r.quad.empty:
            full = (r.quad["quad_agreement"] == 4).sum()
            _log_and_print(f"  ✓ Quad-model: {len(r.quad):,} stocks, full consensus (4/4): {full}")

        # Step 22: Expected returns summary
        enrichment_src = (
            r.df_features if not r.df_features.empty else r.df_all if not r.df_all.empty else r.df
        )
        r.summary = build_expected_returns_summary(
            r.mc,
            r.kal,
            r.pt,
            r.beat,
            r.anomaly_results,
            source_df=enrichment_src,
            credit=r.credit,
            div_safety=r.div_safety,
            mcmc_result=r.mcmc_result,
        )
        if not r.summary.empty:
            r.summary = filter_quality_stocks(r.summary, r.df_all)
            r.summary = compute_return_zscore_ranks(r.summary)
            _log_and_print(
                f"  ✓ Summary: {len(r.summary):,} stocks, "
                f"{(r.summary['agreement_score'] == 4).sum()} full consensus"
            )

        # Step 23: Multi-level hierarchical MCMC
        if not r.summary.empty and "expected_upside_mc" in r.summary.columns:
            try:
                group_cols = cfg.hier_mcmc_group_cols or [
                    "region",
                    "country",
                    "sector",
                    "industry",
                    "style_class",
                    "size_class",
                ]
                multi_hier = hierarchical_mcmc_multi_level(r.summary, "expected_upside_mc", group_cols=group_cols,
                                                           min_group_size=cfg.hier_mcmc_min_group_size,
                                                           shrinkage_strength=cfg.hier_mcmc_shrinkage_strength)
                if multi_hier and "cross_level_summary" in multi_hier:
                    xls = multi_hier["cross_level_summary"]
                    if isinstance(xls, pd.DataFrame) and not xls.empty:
                        _log_and_print(
                            f"  ✓ Multi-level MCMC: {xls['level'].nunique()} levels, "
                            f"{len(xls)} group posteriors"
                        )
            except Exception as e:
                logger.debug("Multi-level MCMC skipped: %s", e)

        # Step 24: Merge PLR & DCF posteriors into ensemble summary
        if not r.summary.empty:
            if r.plr_result is not None:
                try:
                    if hasattr(r.plr_result, "posterior"):
                        plr_mean = float(r.plr_result.posterior["intercept"].mean())
                        r.summary["plr_intercept_posterior"] = plr_mean
                        _log_and_print(f"  ✓ PLR posterior intercept merged: {plr_mean:.4f}")
                except Exception as e:
                    logger.debug("PLR posterior merge skipped: %s", e)

            if r.dcf_result is not None:
                try:
                    if hasattr(r.dcf_result, "posterior"):
                        dcf_iv_mean = float(r.dcf_result.posterior["intrinsic_value"].mean())
                        r.summary["dcf_intrinsic_value_posterior"] = dcf_iv_mean
                        _log_and_print(
                            f"  ✓ DCF posterior intrinsic value merged: {dcf_iv_mean:.2f}"
                        )
                except Exception as e:
                    logger.debug("DCF posterior merge skipped: %s", e)

        r.corr_info = compute_cross_model_correlation(r.mc, r.kal)

    # ── Phase 6: Posterior Inference & InferenceData ──────────────────────────

    def phase_6_inference_data(self) -> None:
        """Build ArviZ InferenceData for all models and feature views."""
        try:
            from probabilistic_ml_model.data_utils.inference_schema import (
                ARVIZ_AVAILABLE,
                FEATURE_VIEW_REGISTRY,
                build_beat_probability_inference_data,
                build_credit_risk_inference_data,
                build_feature_view_inference_data,
                build_monte_carlo_inference_data,
                summarize_inference_data,
            )
        except ImportError:
            _log_and_print("⏭️ ArviZ not available — skipping InferenceData phase")
            return

        if not ARVIZ_AVAILABLE:
            _log_and_print("⏭️ ArviZ not available — skipping InferenceData phase")
            _log_and_print("    pip install arviz xarray for full Bayesian diagnostics")
            return

        r = self.result

        # Step 25: Per-model InferenceData
        if not r.mc.empty:
            try:
                r.idata_mc = build_monte_carlo_inference_data(r.mc, r.df_all, n_simulations=25_000)
                summary = summarize_inference_data(r.idata_mc)
                _log_and_print(f"  ✓ MC InferenceData: {summary.get('n_draws', 0)} draws")
            except Exception as e:
                logger.debug("MC InferenceData failed: %s", e)

        if not r.beat.empty and "posterior_alpha" in r.beat.columns:
            try:
                r.idata_beat = build_beat_probability_inference_data(r.beat, r.df_all)
                _log_and_print("  ✓ Beat InferenceData built")
            except Exception as e:
                logger.debug("Beat InferenceData failed: %s", e)

        if not r.credit.empty:
            try:
                r.idata_credit = build_credit_risk_inference_data(r.credit, r.df_all)
                _log_and_print("  ✓ Credit Risk InferenceData built")
            except Exception as e:
                logger.debug("Credit InferenceData failed: %s", e)

        # Store PLR/DCF InferenceData from Phase 3 results
        r.idata_plr = r.plr_result
        r.idata_dcf = r.dcf_result

        # Step 26: Per-feature-view InferenceData
        if not r.df_features.empty and FEATURE_VIEW_REGISTRY:
            for view_name in FEATURE_VIEW_REGISTRY:
                try:
                    build_feature_view_inference_data(view_name, r.df_features)
                except Exception:
                    pass

        # Step 27–28: Convergence diagnostics are computed within ArviZ summarize calls above

    # ── Phase 7: Visualization ────────────────────────────────────────────────

    def phase_7_visualizations(self) -> None:
        """Generate all visualization artifacts."""
        from probabilistic_ml_model.pipeline_runners import _write_viz

        # Import visualization functions
        try:
            from probabilistic_ml_model.visualizations.expected_returns_viz import (
                create_kalman_vs_raw_scatter,
                create_mc_return_distribution,
                create_screening_summary_chart,
                create_sector_heatmap,
                create_sector_risk_reward_scatter,
                create_strong_consensus_bar,
                create_tri_model_agreement_histogram,
            )
        except ImportError as e:
            _log_and_print(f"  ⚠️ Visualization imports failed: {e}")
            return

        r = self.result
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        viz_tasks = [
            (
                lambda: create_mc_return_distribution(r.mc),
                "er_mc_distribution.html",
                not r.mc.empty,
            ),
            (
                lambda: create_sector_risk_reward_scatter(r.mc, identifier_coords=r.id_coords),
                "er_sector_risk_reward.html",
                not r.mc.empty,
            ),
            (
                lambda: create_kalman_vs_raw_scatter(r.kal),
                "er_kalman_vs_raw.html",
                not r.kal.empty,
            ),
            (
                lambda: create_tri_model_agreement_histogram(r.tri),
                "er_tri_model_agreement.html",
                not r.tri.empty,
            ),
            (
                lambda: create_sector_heatmap(r.tri, True, schema_metadata=r.schema_metadata),
                "er_sector_heatmap.html",
                not r.tri.empty,
            ),
            (
                lambda: create_strong_consensus_bar(r.strong),
                "er_strong_consensus.html",
                not r.strong.empty,
            ),
            (
                lambda: create_screening_summary_chart(r.screens),
                "er_screening_summary.html",
                bool(r.screens),
            ),
        ]

        generated = 0
        for viz_fn, filename, condition in viz_tasks:
            if condition:
                try:
                    fig = viz_fn()
                    _write_viz(fig, output_dir, filename)
                    generated += 1
                except Exception as e:
                    logger.debug("Viz %s skipped: %s", filename, e)

        _log_and_print(f"  ✓ Generated {generated} visualizations → {output_dir}/")

    # ── Phase 8: Export & Reporting ───────────────────────────────────────────

    def phase_8_export(self) -> None:
        """Export all results to analytics schema and generate summary."""
        from probabilistic_ml_model.pipeline_runners import (
            export_expected_returns_results,
        )

        r, cfg = self.result, self.cfg

        exports = export_expected_returns_results(
            mc=r.mc,
            pt=r.pt,
            kal=r.kal,
            tri=r.tri,
            strong=r.strong,
            beat=r.beat,
            summary=r.summary,
            credit=r.credit,
            div_safety=r.div_safety,
            anomaly_results=r.anomaly_results,
            screens=r.screens,
            output_dir=str(cfg.output_dir),
            max_workers=cfg.export_max_workers,
        )
        for name, dest in exports.items():
            _log_and_print(f"  ✓ {name} → {dest}")

        # Pipeline summary
        _log_and_print()
        _log_and_print("=" * 80)
        _log_and_print("✅ EXPECTED RETURNS ANALYTICS v4.0 COMPLETE")
        _log_and_print("=" * 80)
        _log_and_print(
            f"  Models: MC={len(r.mc):,} | PT={len(r.pt):,} | Kal={len(r.kal):,} | "
            f"Beat={len(r.beat):,} | Credit={len(r.credit):,} | "
            f"Div={len(r.div_safety):,} | Anomaly={len(r.anomaly_results):,}"
        )
        _log_and_print(f"  Summary: {len(r.summary):,} stocks")
        if not r.summary.empty:
            _log_and_print(f"  Full consensus (4/4): {(r.summary['agreement_score'] == 4).sum()}")
        _log_and_print(f"  Screens: {len(r.screens)} strategies")
        _log_and_print(f"  Categories: {len(r.category_analytics)} analyzed")
        for phase, elapsed in r.phase_timings.items():
            _log_and_print(f"  ⏱ {phase}: {elapsed:.1f}s")

        # Print detailed model statistics collected during Phase 2
        if r.model_statistics:
            _log_and_print()
            _log_and_print("─" * 80)
            _log_and_print("Per-Model Detailed Statistics")
            _log_and_print("─" * 80)
            for model_name, stats in r.model_statistics.items():
                print_model_statistics(stats, model_name)

        _log_and_print()

    # ── Full Pipeline Orchestrator ────────────────────────────────────────────

    def run(self) -> BaselinePipelineResult:
        """Execute the full 8-phase pipeline."""
        from probabilistic_ml_model.logging_config import configure_logging

        cfg = self.cfg
        configure_logging(level=cfg.log_level, log_file=cfg.log_file, console=True)

        _log_and_print("=" * 80)
        _log_and_print("Expected Returns Analytics Pipeline v4.0")
        _log_and_print("=" * 80)

        runner = PipelineRunner(self.cfg)
        runner.r = self.result

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        phases = [
            ("Phase 1: Data Ingestion & Enrichment", self.phase_1_load_data),
            ("Phase 2: Core PML Model Execution", self.phase_2_core_models),
            ("Phase 3: Probabilistic Linear Market Models", self.phase_3_market_models),
            ("Phase 4: Statistical Functions & Screening", self.phase_4_statistics_and_screening),
            ("Phase 5: Ensemble Alignment & Summary", self.phase_5_ensemble_alignment),
            ("Phase 6: Posterior Inference & InferenceData", self.phase_6_inference_data),
            ("Phase 7: Visualization", self.phase_7_visualizations),
            ("Phase 8: Export & Reporting", self.phase_8_export),
        ]

        for phase_name, phase_fn in phases:
            with _timed_phase(phase_name) as timer:
                try:
                    phase_fn()
                except Exception as e:
                    logger.error("%s failed: %s", phase_name, e, exc_info=True)
                    _log_and_print(f"⚠️ {phase_name} failed: {e}", logging.ERROR)
            self.result.phase_timings[phase_name] = timer.elapsed

        return self.result


def main(config: PipelineConfig | None = None) -> BaselinePipelineResult:
    """Entry point for the expected returns v4.0 pipeline."""
    pipeline = BaselinePipeline(config=config)
    return pipeline.run()


if __name__ == "__main__":
    main()
