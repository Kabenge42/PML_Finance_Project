"""
Probability Analytics Module for Market Analytics

This module provides comprehensive probability and model confidence analysis
for earnings beat predictions, EPS streak analysis, and posterior probability
estimation using Bayesian inference methods.

Features:
- Bayesian Earnings Beat Probability Model with posterior updates and three-layer evidence fusion
- EPS Streak Analysis with predictive analytics and mean reversion modeling
- Model Confidence Estimation with Brier score, calibration error, and reliability diagrams
- Credit Risk Probability Model with Altman Z-score and distress indicators
- Dividend Cut Probability Model with FCF coverage and payout sustainability analysis
- Price Target Achievement Model with analyst consensus and revision momentum
- Accounting Anomaly Detection with Bayesian-informed multi-layered statistical analysis
- Resampled Beat Probability Model with technical signal conditioning and ArviZ integration
- Interactive dashboards for probability visualization and category-level analytics
- Enhanced MCMC posterior estimation (Metropolis-Hastings, Student-t, hierarchical by sector)
- Multi-component confidence scoring (volume, concentration, decisiveness)
- Forward estimate signal integration (GAAP vs Normalized divergence, revision momentum)
- Reported EPS history analysis with dynamic sample size derivation
- Comprehensive data export pipeline with standardized identifier column propagation

References:
- Bayesian methods for financial forecasting
- Posterior probability estimation techniques
- Beta-Binomial conjugate prior framework
- ArviZ probabilistic programming diagnostics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from probabilistic_ml_model.data_utils import (
    load_identifier_columns,
    reorder_with_identifiers,
    ExportConfig,
    export_to_db,
    export_to_csv,
    export_to_json,
)

logger = logging.getLogger(__name__)

# Identifier columns for model output propagation
_IDENTIFIER_COLS_CACHE: list[str] | None = None


def _get_identifier_cols() -> list[str]:
    """Cached access to identifier columns for model output."""
    global _IDENTIFIER_COLS_CACHE
    if _IDENTIFIER_COLS_CACHE is None:
        _IDENTIFIER_COLS_CACHE = load_identifier_columns()
    return _IDENTIFIER_COLS_CACHE


def compute_beta_confidence_score(
    post_alpha: float | np.ndarray,
    post_beta: float | np.ndarray,
    prior_alpha: float = 2.0,
    prior_beta: float = 2.0,
    normalization_factor: float = 20.0,
) -> float | np.ndarray:
    """
    Compute a multi-component confidence score for Beta posterior parameters.

    Combines:
    - Volume: effective sample size relative to normalization factor
    - Concentration: inverse posterior variance (tighter = more confident)
    - Decisiveness: distance of posterior mean from 0.5 (more extreme = more decisive)

    Works with both scalar and vectorized (numpy array) inputs.

    Parameters
    ----------
    post_alpha : float or np.ndarray
        Posterior alpha parameter(s).
    post_beta : float or np.ndarray
        Posterior beta parameter(s).
    prior_alpha : float
        Prior alpha used (subtracted for effective sample size).
    prior_beta : float
        Prior beta used (subtracted for effective sample size).
    normalization_factor : float
        Scale factor for volume component (default 20).

    Returns
    -------
    float or np.ndarray
        Confidence score(s) in [0, 1].
    """
    prior_total = prior_alpha + prior_beta
    total = post_alpha + post_beta
    effective_sample = total - prior_total

    volume_score = np.clip(effective_sample / normalization_factor, 0, 1)

    posterior_mean = post_alpha / total
    decisiveness = np.abs(posterior_mean - 0.5) * 2

    variance = (post_alpha * post_beta) / (total**2 * (total + 1))
    concentration_score = np.clip(1.0 / (1.0 + 20.0 * variance), 0, 1)

    confidence = 0.4 * volume_score + 0.35 * concentration_score + 0.25 * decisiveness
    return np.clip(confidence, 0.0, 1.0)


def _extract_identifiers(row: pd.Series) -> dict:
    """Extract all available identifier columns from a DataFrame row."""
    id_cols = _get_identifier_cols()
    return {
        col: row.get(col, None) for col in id_cols if col in row.index and pd.notna(row.get(col))
    }


# Columns that must be cast to numeric before export (Issue 7)
_NUMERIC_CAST_COLS = [
    "gaap_revision_momentum",
    "gaap_revison_1m",
    "gaap_revison_3m",
    "gaap_revison_6m",
    "gaap_revison_1y",
    "gaap_vs_norm_revision_spread",
    "forward_eps_gaap_adj_spread",
    "gaap_revision_acceleration",
    "eps_surprise_pct",
    "eps_norm_est_fy1e",
    "eps_basic_fy",
    "eps_adjustment_ratio",
    "gaap_adj_eps_gap_pct",
    "eps_quarterly_trend",
    "eps_yoy_growth",
    "eps_qoq_growth",
    # FIX: These were exported as text due to mixed None/float
    "gaap_norm_spread",
    "revision_trend_short",
    "revision_trend_medium",
    "eps_norm_est_ntm",
    "eps_gaap_est_ntm",
    "eps_gaap_est_fy1e",
    # Passthrough columns from mv_all_stock_features
    "accounting_quality_score",
    "distress_risk_score",
    "eps_revision_momentum",
    "altman_z_score",
    "model_confidence",
    "map_estimate",
]
_INTEGER_CAST_COLS = [
    "analyst_count",
    "quarterly_beat_streak",
    "gaap_positive_revision_flag",
    "piotroski_f_score",
    "eps_positive_streak",
]

# Lazy ArviZ import (consistent with inference_schema.py)
try:
    import arviz as az
    import xarray as xr

    ARVIZ_AVAILABLE = hasattr(az, "InferenceData")
except (ImportError, OSError, PermissionError, Exception):
    az = None  # type: ignore[assignment]
    xr = None  # type: ignore[assignment]
    ARVIZ_AVAILABLE = False


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class BeatProbabilityEstimate(TypedDict):
    """Type definition for beat probability estimation results."""

    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_std: float
    credible_interval_90: tuple[float, float]
    credible_interval_95: tuple[float, float]
    prob_exceeds_threshold: float
    confidence_score: float
    # NEW: Interpretability enhancements
    prior_influence_pct: float  # How much prior vs data drove the result
    effective_sample_size: float  # Statistical power indicator
    classification_confidence: str  # 'High', 'Medium', 'Low'


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================


@dataclass
class CreditRiskResult:
    """Result container for credit risk probability analysis."""

    ticker: str
    name: str
    sector: str
    distress_probability: float
    liquidity_stress_score: float
    cash_runway_months: float
    altman_z_score: float
    risk_level: str  # 'Low', 'Medium', 'High', 'Distressed'
    confidence_interval: tuple[float, float]
@dataclass
class AccountingAnomalyResult:
    """Result container for per-stock accounting anomaly analysis."""
    ticker: str
    name: str
    sector: str
    industry: str
    accounting_anomaly_score: float
    accounting_anomaly_tier: str
    anomaly_feature_count: int
    mahalanobis_distance: float
    sector_relative_anomaly: float
    benford_chi2_pvalue: float
    anomaly_severity_score: float
    multi_flag_alert: bool
    anomaly_conditional_probability: float


@dataclass
class AccountingAnomalyProbabilityModel:
    """
    Bayesian-informed accounting anomaly detection and analytics model.

    Wraps ``detect_accounting_anomalies`` (statistical_analysis.py) and
    ``analyze_accounting_anomalies`` into a single, composable probability
    model following the same interface as CreditRiskProbabilityModel,
    DividendCutProbabilityModel, and PriceTargetAchievementModel.

    Parameters
    ----------
    anomaly_z_threshold : float or None
        Robust z-score threshold for flagging anomalies. None = auto-derived.
    tier_bins : list[float] or None
        Bin edges for anomaly tier classification. None = auto-derived.
    tier_labels : list[str] or None
        Labels for the tier bins. None = ['Clean', 'Watch', 'Flag', 'Alert'].
    severity_anomaly_weight : float
        Weight for anomaly_score in severity computation (default 0.7).
    severity_feature_weight : float
        Weight for feature_count in severity computation (default 0.3).
    multi_flag_threshold : int
        Minimum flagged features to trigger multi_flag_alert (default 10).
    """

    anomaly_z_threshold: float | None = None
    tier_bins: list[float] | None = None
    tier_labels: list[str] | None = None
    severity_anomaly_weight: float = 0.7
    severity_feature_weight: float = 0.3
    multi_flag_threshold: int = 10
    n_mcmc_samples: int = 5000
    burn_in: int = 1000
    use_mcmc: bool = True
    # NEW: Comprehensive quality signals (v3.4)
    use_quality_frequency: bool = True
    use_balance_sheet_quality: bool = True

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full accounting anomaly detection + extended analytics.

        Delegates to ``detect_accounting_anomalies`` for multi-layered
        statistical detection (robust z-scores, distribution fitting,
        Mahalanobis distance, Benford's Law), then computes:
        - anomaly_severity_score (weighted combination)
        - anomaly_risk_rank (universe percentile)
        - sector_anomaly_percentile (within-sector rank)
        - multi_flag_alert (boolean threshold)

        Parameters
        ----------
        df : pd.DataFrame
            Financial data with accounting quality columns.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with all anomaly detection and analytics columns.
        """
        from probabilistic_ml_model.statistical_functions.statistical_analysis import (
            detect_accounting_anomalies,
        )

        # Phase 1: Multi-layered anomaly detection
        result = detect_accounting_anomalies(
            df,
            anomaly_z_threshold=self.anomaly_z_threshold,
            tier_bins=self.tier_bins,
            tier_labels=self.tier_labels,
        )

        if "accounting_anomaly_score" not in result.columns:
            return result

        # Phase 2: Extended analytics (severity, ranking, sector percentile)
        feature_count = result.get("anomaly_feature_count", pd.Series(0, index=result.index))

        result["anomaly_severity_score"] = (
            result["accounting_anomaly_score"] * self.severity_anomaly_weight
            + feature_count.clip(0, 9) / 9 * 100 * self.severity_feature_weight
        )

        # Universe-level percentile rank
        severity = result["anomaly_severity_score"].dropna()
        if len(severity) > 1:
            result["anomaly_risk_rank"] = (
                result["anomaly_severity_score"].rank(pct=True, ascending=True) * 100
            )
        else:
            result["anomaly_risk_rank"] = 50.0

        # Sector-level percentile
        sector_col = "industry" if "industry" in result.columns else "sector"
        if sector_col in result.columns:
            result["sector_anomaly_percentile"] = (
                result.groupby(sector_col)["accounting_anomaly_score"].rank(
                    pct=True, ascending=True
                )
                * 100
            )
        else:
            result["sector_anomaly_percentile"] = result["anomaly_risk_rank"]

        # Multi-flag alert
        result["multi_flag_alert"] = feature_count >= self.multi_flag_threshold

        # Phase 2b: Enrich severity with comprehensive quality frequency signals
        if self.use_quality_frequency:
            freq_cols = [
                "goodwill_impairment_frequency",
                "asset_writedown_frequency",
                "restructuring_frequency",
            ]
            available_freq = [c for c in freq_cols if c in result.columns]
            if available_freq:
                freq_sum = result[available_freq].fillna(0).sum(axis=1)
                result["anomaly_severity_score"] += freq_sum * 3.0

            if "quality_issues_count_5y" in result.columns:
                result["anomaly_severity_score"] += (
                    result["quality_issues_count_5y"].fillna(0) * 2.0
                )

        # Phase 2c: Balance sheet quality cross-check
        if self.use_balance_sheet_quality:
            if "retained_earnings_vs_5y" in result.columns:
                re_declining = result["retained_earnings_vs_5y"].fillna(1.0) < 0.7
                result.loc[re_declining, "anomaly_severity_score"] += 5.0

            if "intangibles_growth_flag" in result.columns:
                intang_growing = result["intangibles_growth_flag"].fillna(0) == 1
                result.loc[intang_growing, "anomaly_severity_score"] += 4.0

            if "asset_quality_score" in result.columns:
                low_quality = result["asset_quality_score"].fillna(50) < 25
                result.loc[low_quality, "anomaly_severity_score"] += 5.0

        # Phase 3: Conditional probability of anomaly per row
        # Compute per-feature conditional probabilities
        cond_probs = self.calculate_conditional_probabilities(result)

        if not cond_probs.empty:
            # For each row, compute a weighted average P(Anomaly) across
            # features based on whether the row's feature value is above or
            # below the median, weighted by each feature's separation score.
            prob_col = pd.Series(0.0, index=result.index)
            total_sep = 0.0

            for _, cp_row in cond_probs.iterrows():
                feat = cp_row["feature"]
                if feat not in result.columns:
                    continue
                sep = cp_row["separation"]
                if sep <= 0:
                    continue

                feat_data = result[feat]
                median_val = feat_data.median()
                # Assign P(Anomaly|High) or P(Anomaly|Low) per row
                row_prob = pd.Series(
                    np.where(
                        feat_data > median_val,
                        cp_row["p_anomaly_high"],
                        cp_row["p_anomaly_low"],
                    ),
                    index=result.index,
                )
                # NaN features get the base rate
                row_prob = row_prob.where(feat_data.notna(), cp_row["base_anomaly_rate"])
                prob_col += row_prob * sep
                total_sep += sep

            if total_sep > 0:
                result["anomaly_conditional_probability"] = prob_col / total_sep
            else:
                result["anomaly_conditional_probability"] = cond_probs.iloc[0]["base_anomaly_rate"]
        else:
            # Fallback: use severity score normalized to [0, 1]
            max_sev = result["anomaly_severity_score"].max()
            if max_sev > 0:
                result["anomaly_conditional_probability"] = (
                    result["anomaly_severity_score"] / max_sev
                )
            else:
                result["anomaly_conditional_probability"] = 0.0

        # Phase 4: MCMC posterior estimation (optional)
        if self.use_mcmc:
            result = self._apply_mcmc_posteriors(result)

        logger.info(
            "AccountingAnomalyProbabilityModel: severity computed for %d stocks, "
            "%d multi-flag alerts, mean conditional P(anomaly)=%.3f",
            len(result),
            result["multi_flag_alert"].sum(),
            result["anomaly_conditional_probability"].mean(),
        )

        return result

    def _apply_mcmc_posteriors(self, result: pd.DataFrame) -> pd.DataFrame:
        """Apply MCMC posterior estimation to anomaly scores."""
        from probabilistic_ml_model.statistical_functions.statistical_analysis import (
            mcmc_student_t,
            hierarchical_mcmc_by_sector,
            metropolis_hastings_sampler,
        )

        if "accounting_anomaly_score" not in result.columns:
            return result

        anomaly_scores = result["accounting_anomaly_score"].dropna().values
        if len(anomaly_scores) < 10:
            return result

        # Task 1.1: Student-t posterior for anomaly scores
        try:
            mu_samples, df_samples = mcmc_student_t(
                anomaly_scores,
                n_samples=self.n_mcmc_samples,
                burn_in=self.burn_in,
            )
            result["anomaly_posterior_mean"] = mu_samples.mean()
            result["anomaly_posterior_std"] = mu_samples.std()
            result["anomaly_ci_lower"] = np.percentile(mu_samples, 2.5)
            result["anomaly_ci_upper"] = np.percentile(mu_samples, 97.5)
        except Exception as e:
            logger.warning("MCMC Student-t for anomaly scores failed: %s", e)

        # Task 1.2: Hierarchical MCMC by sector
        sector_col = "industry" if "industry" in result.columns else "sector"
        if sector_col in result.columns:
            try:
                sector_posteriors = hierarchical_mcmc_by_sector(
                    result,
                    feature="accounting_anomaly_score",
                    sector_col=sector_col,
                    n_samples=self.n_mcmc_samples,
                )
                sector_mean_map = {
                    s: v.get("posterior_mean", np.nan)
                    for s, v in sector_posteriors.items()
                    if isinstance(v, dict)
                }
                result["sector_posterior_mean"] = result[sector_col].map(sector_mean_map)
            except Exception as e:
                logger.warning("Hierarchical MCMC for anomaly sectors failed: %s", e)

        return result

    def calculate_conditional_probabilities(
        self,
        df: pd.DataFrame,
        anomaly_threshold: float = 50.0,
        min_sample_size: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate conditional probability of anomaly given each accounting feature.

        Follows the same Bayesian-informed pattern as
        :func:`~analytics.statistical_analysis.calculate_conditional_probabilities`,
        adapted for accounting anomaly detection:

        - P(Anomaly | High Feature) vs P(Anomaly | Low Feature)
        - Lift ratios relative to the base anomaly rate
        - Separation score for feature importance ranking

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that has already been processed by :meth:`analyze_dataframe`
            (must contain ``anomaly_severity_score``).
        anomaly_threshold : float, default 50.0
            Severity score above which a stock is classified as anomalous.
        min_sample_size : int, default 10
            Minimum observations required per feature to compute probabilities.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``feature``, ``p_anomaly_high``,
            ``p_anomaly_low``, ``lift_high``, ``lift_low``, ``separation``,
            ``base_anomaly_rate``, sorted by ``separation`` descending.
        """
        _empty_cols = [
            "feature",
            "p_anomaly_high",
            "p_anomaly_low",
            "lift_high",
            "lift_low",
            "separation",
            "base_anomaly_rate",
        ]

        if "anomaly_severity_score" not in df.columns:
            return pd.DataFrame(columns=_empty_cols)

        is_anomaly = df["anomaly_severity_score"] >= anomaly_threshold
        base_anomaly_rate = float(is_anomaly.mean())

        # Identify accounting features: raw columns that have a corresponding
        # _z_robust column (produced by detect_accounting_anomalies), plus any
        # _anomaly_flag columns.
        raw_features = [
            col.replace("_z_robust", "")
            for col in df.columns
            if col.endswith("_z_robust") and col.replace("_z_robust", "") in df.columns
        ]

        results = []
        for feature in raw_features:
            data = df[[feature, "anomaly_severity_score"]].dropna()
            if len(data) < min_sample_size:
                continue
            if not pd.api.types.is_numeric_dtype(data[feature]):
                continue

            median_val = data[feature].median()
            feat_anomaly = data["anomaly_severity_score"] >= anomaly_threshold

            high_mask = data[feature] > median_val
            if high_mask.sum() == 0 or (~high_mask).sum() == 0:
                continue

            p_anomaly_high = float(feat_anomaly[high_mask].mean())
            p_anomaly_low = float(feat_anomaly[~high_mask].mean())

            lift_high = p_anomaly_high / base_anomaly_rate if base_anomaly_rate > 0 else 1.0
            lift_low = p_anomaly_low / base_anomaly_rate if base_anomaly_rate > 0 else 1.0

            results.append(
                {
                    "feature": feature,
                    "p_anomaly_high": p_anomaly_high,
                    "p_anomaly_low": p_anomaly_low,
                    "lift_high": lift_high,
                    "lift_low": lift_low,
                    "separation": abs(p_anomaly_high - p_anomaly_low),
                    "base_anomaly_rate": base_anomaly_rate,
                }
            )

        if not results:
            return pd.DataFrame(columns=_empty_cols)

        return pd.DataFrame(results).sort_values("separation", ascending=False)


@dataclass
class DividendSafetyResult:
    """Result container for dividend safety analysis."""

    ticker: str
    name: str
    dividend_cut_probability: float
    fcf_dividend_coverage: float
    payout_ratio: float
    dividend_streak: int
    safety_score: float
    risk_category: str  # 'Safe', 'Borderline', 'At Risk'

@dataclass
class PriceTargetResult:
    """Result container for price target achievement analysis."""

    ticker: str
    name: str
    achievement_probability: float
    upside_potential: float
    price_target_spread_pct: float
    analyst_rating_normalized: float
    expected_return_prob_weighted: float


@dataclass
class BeatProbabilityResult:
    """Result container for earnings beat probability analysis."""

    ticker: str
    name: str
    sector: str
    industry: str
    country: str
    exchange: str
    # Historical counts
    historical_beats: int
    total_reports: int
    dynamic_total_reports: int
    historical_beat_rate: float
    # Prior parameters
    prior_alpha: float
    prior_beta: float
    # Posterior parameters
    posterior_alpha: float
    posterior_beta: float
    posterior_beat_prob: float
    posterior_std: float
    # Credible intervals
    ci_90_lower: float
    ci_90_upper: float
    ci_95_lower: float
    ci_95_upper: float
    # Confidence & classification
    confidence_score: float
    prior_influence_pct: float
    effective_sample_size: float
    classification_confidence: str  # 'High', 'Medium', 'Low'
    beat_classification: str  # 'likely_beat', 'uncertain'
    # Forward estimate fields (from mv_all_stock_features via ForwardEstimateSignals)
    gaap_revision_momentum: Optional[float] = None
    gaap_norm_spread: Optional[float] = None
    revision_trend_short: Optional[float] = None
    revision_trend_medium: Optional[float] = None
    eps_norm_est_fy1e: Optional[float] = None
    eps_norm_est_ntm: Optional[float] = None
    eps_gaap_est_ntm: Optional[float] = None
    eps_gaap_est_fy1e: Optional[float] = None
    # Analyst coverage
    analyst_count: Optional[int] = None
    # Quarterly beat streak (from ReportedEPSHistory)
    quarterly_beat_streak: Optional[int] = None
    # Data source tag
    data_source: str = "trajectory_proxy"
    # Next earnings context
    next_earnings_status: Optional[str] = None
    # Resampled model fields (merged from ResampledBeatProbabilityModel)
    base_posterior_mean: Optional[float] = None
    resampled_posterior_mean: Optional[float] = None
    technical_adjustment: Optional[float] = None
    momentum_signal: Optional[float] = None
    volatility_regime_score: Optional[float] = None
    credible_interval_90: Optional[tuple[float, float]] = None
    credible_interval_95: Optional[tuple[float, float]] = None
    prob_beat_given_momentum: Optional[float] = None
    earnings_season_flag: Optional[int] = None
    pre_earnings_window: Optional[int] = None
    ess_bulk: Optional[float] = None
    r_hat: Optional[float] = None
    # EPS streak fields (merged from EPSStreakAnalyzer)
    eps_positive_streak: Optional[int] = None
    # Composite / quality fields (from mv_all_stock_features passthrough)
    model_confidence: Optional[float] = None
    map_estimate: Optional[float] = None
    accounting_quality_score: Optional[float] = None
    distress_risk_score: Optional[float] = None
    gaap_adj_eps_gap_pct: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    eps_revision_momentum: Optional[float] = None
    altman_z_score: Optional[float] = None


@dataclass
class EPSStreakResult:
    """Result container for EPS streak analysis."""

    ticker: str
    name: str
    sector: str
    industry: str
    country: str
    exchange: str
    current_streak: int
    streak_type: str  # 'beat', 'miss', 'meet'
    max_streak_beat: int
    max_streak_miss: int
    streak_continuation_prob: float
    mean_reversion_prob: float
    expected_next_outcome: str
    confidence_level: float
    # Dynamic data derived from reported history
    dynamic_total_reports: int = 3
    historical_beat_rate: float = 0.33
    # Forward signal passthrough
    gaap_revision_momentum: Optional[float] = None
    next_earnings_status: Optional[str] = None
    # EPS streak from mv_all_stock_features
    eps_positive_streak: Optional[int] = None
    # Composite / quality fields (from mv_all_stock_features passthrough)
    model_confidence: Optional[float] = None
    map_estimate: Optional[float] = None
    accounting_quality_score: Optional[float] = None
    distress_risk_score: Optional[float] = None
    gaap_adj_eps_gap_pct: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    eps_revision_momentum: Optional[float] = None
    altman_z_score: Optional[float] = None
    historical_pattern: list[str] = field(default_factory=list)


@dataclass
class ModelConfidenceResult:
    """Result container for model confidence estimation."""

    model_name: str
    brier_score: float
    log_loss: float
    calibration_error: float
    discrimination_auc: float
    reliability_diagram_data: dict
    confidence_intervals: dict
    overall_confidence: float


@dataclass(frozen=True)
class PriorParameters:
    """Immutable container for Beta distribution prior parameters."""

    alpha: float
    beta: float

    @property
    def expected_beat_rate(self) -> float:
        """Calculate the expected beat rate from prior parameters."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def mode(self) -> float | None:
        """Calculate the mode (most likely value) of the distribution."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None  # Mode undefined for alpha <= 1 or beta <= 1

    @property
    def variance(self) -> float:
        """Calculate the variance of the distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total**2 * (total + 1))

    @property
    def concentration(self) -> float:
        """Return concentration parameter (higher = more confident prior)."""
        return self.alpha + self.beta

    def as_tuple(self) -> tuple[float, float]:
        """Return parameters as (alpha, beta) tuple."""
        return (self.alpha, self.beta)

    def strength_description(self) -> str:
        """Human-readable description of prior strength."""
        concentration = self.concentration
        if concentration < 4:
            return "Weak (data-driven)"
        elif concentration < 10:
            return "Moderate"
        else:
            return "Strong (informative)"


# =============================================================================
# REPORTED EPS HISTORY & FORWARD ESTIMATE SIGNALS
# =============================================================================


@dataclass
class ReportedEPSHistory:
    """Actual reported EPS data for quarterly and annual periods.

    Fields follow the naming convention from the equities schema:
    - eps_basic_fq: most recent fiscal quarter
    - eps_basic_1fqfq: one quarter ago, etc.
    - eps_basic_fy: most recent fiscal year
    - eps_basic_1fy: one year ago, etc.
    """

    # Quarterly Net EPS - Basic (newest first)
    eps_basic_fq: Optional[float] = None
    eps_basic_1fqfq: Optional[float] = None
    eps_basic_2fqfq: Optional[float] = None
    eps_basic_3fqfq: Optional[float] = None
    eps_basic_4fqfq: Optional[float] = None

    # Annual Net EPS - Basic (newest first)
    eps_basic_fy: Optional[float] = None
    eps_basic_1fy: Optional[float] = None
    eps_basic_2fy: Optional[float] = None
    eps_basic_3fy: Optional[float] = None
    eps_basic_4fy: Optional[float] = None
    eps_basic_5fy: Optional[float] = None

    # Adjusted EPS
    eps_adj_fy: Optional[float] = None
    eps_adj_1fy: Optional[float] = None
    eps_adj_ltm: Optional[float] = None
    eps_adj_fq: Optional[float] = None
    eps_adj_1fqfq: Optional[float] = None
    eps_adj_2fqfq: Optional[float] = None
    eps_adj_3fqfq: Optional[float] = None
    eps_adj_4fqfq: Optional[float] = None

    # Continuing EPS
    eps_cont_fq: Optional[float] = None
    eps_cont_1fqfq: Optional[float] = None
    eps_cont_2fqfq: Optional[float] = None
    eps_cont_3fqfq: Optional[float] = None
    eps_cont_4fqfq: Optional[float] = None

    @property
    def quarterly_series(self) -> list[float]:
        """Return non-None quarterly EPS values, newest first."""
        fields = [
            self.eps_basic_fq,
            self.eps_basic_1fqfq,
            self.eps_basic_2fqfq,
            self.eps_basic_3fqfq,
            self.eps_basic_4fqfq,
        ]
        return [v for v in fields if v is not None]

    @property
    def annual_series(self) -> list[float]:
        """Return non-None annual EPS values, newest first."""
        fields = [
            self.eps_basic_fy,
            self.eps_basic_1fy,
            self.eps_basic_2fy,
            self.eps_basic_3fy,
            self.eps_basic_4fy,
            self.eps_basic_5fy,
        ]
        return [v for v in fields if v is not None]

    def count_yoy_improvements(self) -> tuple[int, int]:
        """Count year-over-year improvements in annual EPS.

        Compares each consecutive pair (newer vs older).
        Returns (n_beats, n_total) where n_total is the number of
        consecutive pairs available.
        """
        series = self.annual_series
        if len(series) < 2:
            return (0, 0)
        n_beats = 0
        n_total = len(series) - 1
        for i in range(n_total):
            if series[i] > series[i + 1]:
                n_beats += 1
        return (n_beats, n_total)

    def quarterly_beat_streak(self) -> int:
        """Count consecutive positive EPS quarters from most recent."""
        streak = 0
        for v in self.quarterly_series:
            if v > 0:
                streak += 1
            else:
                break
        return streak

    def count_quarterly_beats_vs_estimate(self, estimate: Optional[float]) -> tuple[int, int]:
        """Count how many quarterly actuals exceeded a forward estimate.

        Args:
            estimate: Forward EPS estimate to compare against.

        Returns:
            (n_beats, n_total) tuple.
        """
        if estimate is None:
            return (0, 0)
        series = self.quarterly_series
        if not series:
            return (0, 0)
        n_beats = sum(1 for v in series if v > estimate)
        return (n_beats, len(series))

    @property
    def total_reports_count(self) -> int:
        """Total number of non-null reported EPS observations across all series.

        Counts unique non-null entries across quarterly basic, annual basic,
        adjusted, and continuing EPS fields to dynamically derive the total
        number of available data points for historical beat rate calculations.
        """
        all_fields = [
            # Quarterly basic
            self.eps_basic_fq,
            self.eps_basic_1fqfq,
            self.eps_basic_2fqfq,
            self.eps_basic_3fqfq,
            self.eps_basic_4fqfq,
            # Annual basic
            self.eps_basic_fy,
            self.eps_basic_1fy,
            self.eps_basic_2fy,
            self.eps_basic_3fy,
            self.eps_basic_4fy,
            self.eps_basic_5fy,
            # Adjusted
            self.eps_adj_fy,
            self.eps_adj_1fy,
            self.eps_adj_ltm,
            self.eps_adj_fq,
            self.eps_adj_1fqfq,
            self.eps_adj_2fqfq,
            self.eps_adj_3fqfq,
            self.eps_adj_4fqfq,
            # Continuing
            self.eps_cont_fq,
            self.eps_cont_1fqfq,
            self.eps_cont_2fqfq,
            self.eps_cont_3fqfq,
            self.eps_cont_4fqfq,
        ]
        return sum(1 for v in all_fields if v is not None)

    @property
    def annual_reports_count(self) -> int:
        """Count of non-null annual EPS observations (basic series only)."""
        return len(self.annual_series)

    @property
    def quarterly_reports_count(self) -> int:
        """Count of non-null quarterly EPS observations (basic series only)."""
        return len(self.quarterly_series)


@dataclass
class ForwardEstimateSignals:
    """Forward-looking analyst estimate and revision signals.

    Captures consensus EPS estimates (Normalized and GAAP) plus
    revision percentages across multiple time horizons.
    """

    # Consensus estimates
    eps_norm_ntm: Optional[float] = None
    eps_norm_fy1e: Optional[float] = None
    eps_gaap_ntm: Optional[float] = None
    eps_gaap_fy1e: Optional[float] = None

    # Normalized revision percentages
    revision_1w: Optional[float] = None
    revision_1m: Optional[float] = None
    revision_3m: Optional[float] = None
    revision_6m: Optional[float] = None
    revision_1y: Optional[float] = None

    # GAAP revision percentages
    gaap_revision_1m: Optional[float] = None
    gaap_revision_3m: Optional[float] = None
    gaap_revision_6m: Optional[float] = None
    gaap_revision_1y: Optional[float] = None

    # Coverage
    analyst_count: Optional[int] = None

    # Recency weights for revision momentum (1W most important)
    _REVISION_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "revision_1w": 0.35,
            "revision_1m": 0.30,
            "revision_3m": 0.20,
            "revision_6m": 0.10,
            "revision_1y": 0.05,
        },
        init=False,
        repr=False,
    )

    @property
    def gaap_revision_momentum(self) -> float:
        """Compute a 0-100 momentum score from revision data.

        Uses recency-weighted scoring: each available revision contributes
        its weight toward 100 (positive) or 0 (negative).
        Returns 50.0 when no revision data is available.
        """
        available = []
        for field_name, weight in self._REVISION_WEIGHTS.items():
            val = getattr(self, field_name)
            if val is not None:
                available.append((val, weight))
        if not available:
            return 50.0
        # Renormalize weights to sum to 1
        total_weight = sum(w for _, w in available)
        score = 0.0
        for val, weight in available:
            normalized_w = weight / total_weight
            # Map: positive revision → 100, negative → 0
            score += normalized_w * (100.0 if val > 0 else (50.0 if val == 0 else 0.0))
        return score

    @property
    def revision_trend_short(self) -> Optional[float]:
        """Short-term revision acceleration: 1W - 1M."""
        if self.revision_1w is not None and self.revision_1m is not None:
            return self.revision_1w - self.revision_1m
        return None

    @property
    def revision_trend_medium(self) -> Optional[float]:
        """Medium-term revision acceleration: 1M - 3M."""
        if self.revision_1m is not None and self.revision_3m is not None:
            return self.revision_1m - self.revision_3m
        return None

    @property
    def gaap_norm_spread(self) -> Optional[float]:
        """GAAP-vs-Norm divergence as percentage of Norm estimate.

        Returns (GAAP - Norm) / Norm * 100. Negative means GAAP < Norm
        (potential accounting quality concern).
        """
        if self.eps_gaap_fy1e is not None and self.eps_norm_fy1e is not None:
            if self.eps_norm_fy1e != 0:
                return (self.eps_gaap_fy1e - self.eps_norm_fy1e) / self.eps_norm_fy1e * 100.0
        return None

    @property
    def has_sufficient_data(self) -> bool:
        """Check if enough forward data is available for enhanced analysis.

        Requires at least a FY1E estimate and one revision data point.
        """
        has_estimate = self.eps_norm_fy1e is not None
        revision_fields = [
            self.revision_1w,
            self.revision_1m,
            self.revision_3m,
            self.revision_6m,
            self.revision_1y,
        ]
        has_revision = any(v is not None for v in revision_fields)
        return has_estimate and has_revision


# =============================================================================
# BAYESIAN EARNINGS BEAT PROBABILITY MODEL
# =============================================================================


class EarningsBeatProbabilityModel:
    """
    Bayesian model for estimating earnings beat probabilities.

    Uses Beta-Binomial conjugate prior framework to compute posterior
    probabilities of earnings beats given historical data. The model
    supports incremental updates as new earnings data becomes available.

    The posterior probability is computed using Bayes' theorem:
    P(beat | data) ∝ P(data | beat) × P(beat)

    With Beta prior: Beta(α, β)
    And Binomial likelihood for beats/misses
    Posterior: Beta(α + beats, β + misses)
    """

    # Quantile values for credible intervals
    CI_90_LOWER_QUANTILE = 0.05
    CI_90_UPPER_QUANTILE = 0.95
    CI_95_LOWER_QUANTILE = 0.025
    CI_95_UPPER_QUANTILE = 0.975

    # Confidence score normalization factor (based on effective sample size)
    CONFIDENCE_NORMALIZATION_FACTOR = 20

    def __init__(
        self,
        prior_alpha: float = 1.5,
        prior_beta: float = 2.0,
        sector_priors: Optional[dict[str, PriorParameters]] = None,
        # NEW: Quality-adjusted beat probability (v3.4)
        use_quality_adjustment: bool = True,
    ):
        """
        Initialize the earnings beat probability model.

        Args:
            prior_alpha: Alpha parameter for Beta prior (default: 2.0 for mild optimism)
            prior_beta: Beta parameter for Beta prior (default: 2.0 for symmetry)
            sector_priors: Optional dict mapping sectors to PriorParameters
                          for sector-specific priors based on historical patterns
            use_quality_adjustment: Whether to adjust confidence scores based on
                accounting quality signals (accounting_quality_score,
                quality_issues_count_5y, balance_sheet_strength).
        """
        self.default_prior = PriorParameters(prior_alpha, prior_beta)
        self.sector_priors = sector_priors or self._create_default_sector_priors()
        self.use_quality_adjustment = use_quality_adjustment

    @property
    def prior_alpha(self) -> float:
        return self.default_prior.alpha

    @property
    def prior_beta(self) -> float:
        return self.default_prior.beta

    def _create_default_sector_priors(self) -> dict[str, PriorParameters]:
        """
        Create default sector-specific priors based on typical beat rates.

        Technology tends to have higher beat rates (~70%), while
        cyclical sectors like Energy have more variable outcomes.
        """
        return {
            "Information Technology": PriorParameters(3.5, 1.5),  # ~70% prior beat rate
            "Health Care": PriorParameters(3.0, 1.5),  # ~67% prior beat rate
            "Consumer Discretionary": PriorParameters(2.5, 1.5),  # ~63% prior beat rate
            "Industrials": PriorParameters(2.5, 1.5),  # ~63% prior beat rate
            "Financials": PriorParameters(2.5, 2.0),  # ~56% prior beat rate
            "Consumer Staples": PriorParameters(2.5, 2.0),  # ~56% prior beat rate
            "Materials": PriorParameters(2.0, 2.0),  # ~50% prior beat rate
            "Energy": PriorParameters(2.0, 2.5),  # ~44% prior beat rate
            "Utilities": PriorParameters(2.0, 2.0),  # ~50% prior beat rate
            "Communication Services": PriorParameters(3.0, 1.5),  # ~67% prior beat rate
            "Real Estate": PriorParameters(2.0, 2.0),  # ~50% prior beat rate
        }

    def _get_prior_parameters(
        self,
        sector: Optional[str],
        use_sector_prior: bool,
    ) -> PriorParameters:
        """
        Get the appropriate prior parameters based on sector.

        Args:
            sector: Optional sector name
            use_sector_prior: Whether to use sector-specific priors

        Returns:
            PriorParameters for the given sector or default
        """
        if not use_sector_prior or not sector:
            return self.default_prior

        return self.sector_priors.get(sector, self.default_prior)

    def _compute_posterior_statistics(
        self,
        alpha: float,
        beta: float,
    ) -> tuple[float, float]:
        """
        Compute mean and standard deviation of Beta posterior.

        Args:
            alpha: Posterior alpha parameter
            beta: Posterior beta parameter

        Returns:
            Tuple of (posterior_mean, posterior_std)
        """
        total = alpha + beta
        posterior_mean = alpha / total
        posterior_std = np.sqrt((alpha * beta) / (total**2 * (total + 1)))
        return posterior_mean, posterior_std

    def _compute_single_credible_interval(
        self,
        distribution: stats.rv_continuous,
        lower_quantile: float,
        upper_quantile: float,
    ) -> tuple[float, float]:
        """
        Compute a single credible interval from posterior distribution.

        Args:
            distribution: Scipy Beta distribution object
            lower_quantile: Lower quantile (e.g., 0.05 for 90% CI)
            upper_quantile: Upper quantile (e.g., 0.95 for 90% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        return (distribution.ppf(lower_quantile), distribution.ppf(upper_quantile))

    def _compute_credible_intervals(
        self,
        distribution: stats.rv_continuous,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Compute 90% and 95% credible intervals from posterior distribution.

        Args:
            distribution: Scipy Beta distribution object

        Returns:
            Tuple of (ci_90, ci_95) where each is a (lower, upper) tuple
        """
        ci_90 = self._compute_single_credible_interval(
            distribution,
            self.CI_90_LOWER_QUANTILE,
            self.CI_90_UPPER_QUANTILE,
        )
        ci_95 = self._compute_single_credible_interval(
            distribution,
            self.CI_95_LOWER_QUANTILE,
            self.CI_95_UPPER_QUANTILE,
        )
        return ci_90, ci_95

    def _compute_confidence_score(self, alpha: float, beta: float) -> float:
        """
        Compute confidence score based on effective sample size AND posterior certainty.

        The confidence score reflects both how much data supports the posterior
        estimate and how decisive the posterior is (distance from 0.5).

        Args:
            alpha: Posterior alpha parameter
            beta: Posterior beta parameter

        Returns:
            Confidence score between 0 and 1
        """
        return float(
            compute_beta_confidence_score(
                alpha,
                beta,
                prior_alpha=self.default_prior.alpha,
                prior_beta=self.default_prior.beta,
                normalization_factor=self.CONFIDENCE_NORMALIZATION_FACTOR,
            )
        )

    def compute_posterior(
        self,
        n_beats: int,
        n_total: int,
        sector: Optional[str] = None,
        use_sector_prior: bool = True,
    ) -> tuple[float, float]:
        """
        Compute posterior Beta parameters given observed beats.

        Args:
            n_beats: Number of earnings beats observed
            n_total: Total number of earnings observations
            sector: Optional sector for sector-specific prior
            use_sector_prior: Whether to use sector-specific priors

        Returns:
            Tuple of (posterior_alpha, posterior_beta)
        """
        n_misses = n_total - n_beats
        prior = self._get_prior_parameters(sector, use_sector_prior)
        alpha, beta = prior.alpha, prior.beta

        # Conjugate update: posterior = Beta(α + beats, β + misses)
        posterior_alpha = alpha + n_beats
        posterior_beta = beta + n_misses

        return posterior_alpha, posterior_beta

    def compute_beat_probability(
        self,
        n_beats: int,
        n_total: int,
        sector: Optional[str] = None,
        threshold: float = 0.67,
    ) -> BeatProbabilityEstimate:
        """
        Compute the probability of future earnings beat.

        Args:
            n_beats: Number of historical beats
            n_total: Total observations
            sector: Optional sector name
            threshold: Probability threshold for "likely beat" classification

        Returns:
            Dictionary with probability estimates and confidence metrics
        """
        post_alpha, post_beta = self.compute_posterior(n_beats, n_total, sector, threshold)

        posterior_mean, posterior_std = self._compute_posterior_statistics(post_alpha, post_beta)

        dist = stats.beta(post_alpha, post_beta)
        ci_90, ci_95 = self._compute_credible_intervals(dist)

        prob_exceeds_threshold = 1 - dist.cdf(threshold)
        confidence_score = self._compute_confidence_score(post_alpha, post_beta)

        return {
            "posterior_alpha": post_alpha,
            "posterior_beta": post_beta,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "credible_interval_90": ci_90,
            "credible_interval_95": ci_95,
            "prob_exceeds_threshold": prob_exceeds_threshold,
            "confidence_score": confidence_score,
        }

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        beats_col: str = "eps_positive_years",
        total_col: str = "eps_positive_streak",
        sector_col: str = "sector",
        ticker_col: str = "ticker",
        name_col: str = "name",
    ) -> pd.DataFrame:
        """
        Analyze earnings beat probabilities for entire DataFrame.

        Args:
            df: DataFrame with earnings data
            beats_col: Column name for beat counts
            total_col: Column name for total report counts
            sector_col: Column name for sector
            ticker_col: Column name for ticker
            name_col: Column name for company name

        Returns:
            DataFrame with probability analysis results
        """
        # --- Vectorized Beta-Binomial update ---
        has_beats = beats_col in df.columns
        has_total = total_col in df.columns

        n_beats_series = (
            df[beats_col].fillna(0).astype(int) if has_beats else pd.Series(0, index=df.index)
        )
        n_total_series = (
            df[total_col].fillna(0).astype(int) if has_total else pd.Series(0, index=df.index)
        )

        # Proxy fallback: use eps_trajectory_score when direct columns are missing/zero
        proxy_mask = n_total_series == 0
        if proxy_mask.any() and "eps_trajectory_score" in df.columns:
            trajectory = df.loc[proxy_mask, "eps_trajectory_score"].fillna(50)

            # Dynamic n_total: use available data columns or graduate by trajectory
            if "eps_positive_years" in df.columns:
                n_total_proxy = df.loc[proxy_mask, "eps_positive_years"].fillna(0).clip(lower=0)
                n_total_proxy = n_total_proxy.clip(lower=3, upper=15).astype(int)
            elif "eps_improvement_count" in df.columns:
                n_total_proxy = (
                    df.loc[proxy_mask, "eps_improvement_count"]
                    .fillna(3)
                    .clip(lower=3, upper=15)
                    .astype(int)
                )
            else:
                # Graduated proxy: higher trajectory scores imply more consistent data
                n_total_proxy = pd.Series(
                    np.where(
                        trajectory >= 80,
                        8,
                        np.where(
                            trajectory >= 60,
                            6,
                            np.where(trajectory >= 40, 5, np.where(trajectory >= 20, 4, 3)),
                        ),
                    ),
                    index=df.loc[proxy_mask].index,
                )
            n_total_series.loc[proxy_mask] = n_total_proxy
            n_beats_series.loc[proxy_mask] = (
                (trajectory / 100 * n_total_proxy).astype(int).clip(lower=0, upper=n_total_proxy)
            )

        # Drop rows still without data
        valid = n_total_series > 0
        if not valid.any():
            return pd.DataFrame()

        df_valid = df.loc[valid].copy()
        n_beats_valid = n_beats_series.loc[valid]
        n_total_valid = n_total_series.loc[valid]

        # Vectorized posterior computation
        post_alpha = self.prior_alpha + n_beats_valid
        post_beta = self.prior_beta + (n_total_valid - n_beats_valid)
        posterior_mean = post_alpha / (post_alpha + post_beta)
        posterior_std = np.sqrt(
            (post_alpha * post_beta)
            / ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
        )

        # Credible intervals (vectorized via scipy)
        ci_90_lower = stats.beta.ppf(0.05, post_alpha, post_beta)
        ci_90_upper = stats.beta.ppf(0.95, post_alpha, post_beta)
        ci_95_lower = stats.beta.ppf(0.025, post_alpha, post_beta)
        ci_95_upper = stats.beta.ppf(0.975, post_alpha, post_beta)

        # Multi-component confidence score (replaces constant concentration/20)
        confidence_score = compute_beta_confidence_score(
            post_alpha.values,
            post_beta.values,
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
            normalization_factor=self.CONFIDENCE_NORMALIZATION_FACTOR,
        )

        beat_classification = np.where(posterior_mean > 0.5, "likely_beat", "uncertain")

        result_df = pd.DataFrame(
            {
                "historical_beats": n_beats_valid.values,
                "total_reports": n_total_valid.values,
                "historical_beat_rate": (n_beats_valid / n_total_valid).values,
                "prior_alpha": self.prior_alpha,
                "prior_beta": self.prior_beta,
                "posterior_alpha": post_alpha.values,
                "posterior_beta": post_beta.values,
                "posterior_beat_prob": posterior_mean.values,
                "posterior_std": posterior_std.values,
                "ci_90_lower": ci_90_lower,
                "ci_90_upper": ci_90_upper,
                "ci_95_lower": ci_95_lower,
                "ci_95_upper": ci_95_upper,
                "confidence_score": np.asarray(confidence_score),
                "beat_classification": beat_classification,
            },
            index=df_valid.index,
        )

        # NEW: Quality-adjusted beat probability (v3.4)
        if self.use_quality_adjustment:
            if "accounting_quality_score" in df_valid.columns:
                aq_score = df_valid["accounting_quality_score"].fillna(50).values
                quality_penalty = np.where(aq_score < 30, 0.85, np.where(aq_score < 50, 0.95, 1.0))
                result_df["confidence_score"] = result_df["confidence_score"] * quality_penalty

            if "quality_issues_count_5y" in df_valid.columns:
                qi_count = df_valid["quality_issues_count_5y"].fillna(0).values
                qi_penalty = np.where(qi_count >= 3, 0.80, np.where(qi_count >= 1, 0.90, 1.0))
                result_df["confidence_score"] = result_df["confidence_score"] * qi_penalty

            if "balance_sheet_strength" in df_valid.columns:
                bs = df_valid["balance_sheet_strength"].fillna(50).values
                bs_penalty = np.where(bs < 25, 0.90, 1.0)
                result_df["confidence_score"] = result_df["confidence_score"] * bs_penalty

        # Attach identifier columns
        for id_col in _get_identifier_cols():
            if id_col in df_valid.columns:
                result_df[id_col] = df_valid[id_col].values

        return result_df.reset_index(drop=True)

    # -----------------------------------------------------------------
    # Enhanced: Three-layer evidence fusion
    # -----------------------------------------------------------------

    # GAAP divergence threshold (%) below which no penalty is applied
    GAAP_DIVERGENCE_THRESHOLD = 20.0
    # Maximum pseudo-observations added by revision momentum
    MAX_REVISION_PSEUDO_OBS = 3.0

    def compute_forward_adjusted_beat_probability(
        self,
        reported_history: ReportedEPSHistory,
        forward_signals: ForwardEstimateSignals,
        sector: Optional[str] = None,
    ) -> BeatProbabilityEstimate:
        """Compute beat probability fusing historical, revision, and GAAP quality layers.

        Layer 1 – Historical beats from actual reported EPS (YoY improvements).
        Layer 2 – Revision momentum converted to pseudo-observations.
        Layer 3 – GAAP-vs-Norm divergence penalty (shrinks toward prior).

        Args:
            reported_history: Actual reported EPS data.
            forward_signals: Forward-looking analyst signals.
            sector: Optional sector for sector-specific prior.

        Returns:
            BeatProbabilityEstimate dict with full posterior statistics.
        """
        prior = self._get_prior_parameters(sector, use_sector_prior=True)

        # --- Layer 1: Historical beat counting ---
        n_beats, n_total = reported_history.count_yoy_improvements()
        if n_total == 0:
            # Fallback to quarterly streak as pseudo-observations
            streak = reported_history.quarterly_beat_streak()
            # Dynamically derive total from non-null quarterly data
            n_total = reported_history.quarterly_reports_count
            n_beats = min(streak, n_total)

        # If still zero, use the full non-null count as a last resort
        if n_total == 0:
            n_total = reported_history.total_reports_count

        # --- Layer 2: Revision momentum pseudo-observations ---
        momentum = forward_signals.gaap_revision_momentum  # 0-100
        # Convert to pseudo beat fraction and scale
        pseudo_beat_frac = momentum / 100.0
        pseudo_n = self.MAX_REVISION_PSEUDO_OBS if forward_signals.has_sufficient_data else 0.0
        pseudo_beats = pseudo_beat_frac * pseudo_n
        pseudo_misses = pseudo_n - pseudo_beats

        # --- Posterior before GAAP penalty ---
        post_alpha = prior.alpha + n_beats + pseudo_beats
        post_beta = prior.beta + (n_total - n_beats) + pseudo_misses

        # --- Layer 3: GAAP quality guard ---
        spread = forward_signals.gaap_norm_spread
        if spread is not None and abs(spread) > self.GAAP_DIVERGENCE_THRESHOLD:
            # Proportional shrinkage toward prior mean
            excess = abs(spread) - self.GAAP_DIVERGENCE_THRESHOLD
            # penalty_factor in (0, 1]; larger excess → stronger shrinkage
            penalty_factor = min(1.0, excess / 100.0)
            # Also penalise divergent GAAP revisions vs norm revisions
            if (
                forward_signals.gaap_revision_1m is not None
                and forward_signals.revision_1m is not None
            ):
                rev_sign_mismatch = (
                    forward_signals.revision_1m > 0 and forward_signals.gaap_revision_1m < 0
                )
                if rev_sign_mismatch:
                    penalty_factor = min(1.0, penalty_factor + 0.15)

            # Shrink posterior toward prior by blending
            prior_total = prior.alpha + prior.beta
            post_total = post_alpha + post_beta
            data_alpha = post_alpha - prior.alpha
            data_beta = post_beta - prior.beta
            post_alpha = prior.alpha + data_alpha * (1 - penalty_factor)
            post_beta = prior.beta + data_beta * (1 - penalty_factor)

        # --- Compute statistics ---
        posterior_mean, posterior_std = self._compute_posterior_statistics(post_alpha, post_beta)
        dist = stats.beta(post_alpha, post_beta)
        ci_90, ci_95 = self._compute_credible_intervals(dist)
        prob_exceeds = 1 - dist.cdf(0.5)
        confidence_score = self._compute_confidence_score(post_alpha, post_beta)

        # Prior influence
        prior_total = prior.alpha + prior.beta
        post_total = post_alpha + post_beta
        prior_influence = prior_total / post_total * 100.0

        effective_sample = post_total - prior_total

        # Classification confidence
        if confidence_score >= 0.6:
            classification_confidence = "High"
        elif confidence_score >= 0.3:
            classification_confidence = "Medium"
        else:
            classification_confidence = "Low"

        return {
            "posterior_alpha": post_alpha,
            "posterior_beta": post_beta,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "credible_interval_90": ci_90,
            "credible_interval_95": ci_95,
            "prob_exceeds_threshold": prob_exceeds,
            "confidence_score": confidence_score,
            "prior_influence_pct": prior_influence,
            "effective_sample_size": effective_sample,
            "classification_confidence": classification_confidence,
        }

    # -----------------------------------------------------------------
    # Enhanced DataFrame analysis
    # -----------------------------------------------------------------

    # Column mappings from equities table to dataclass fields
    _FORWARD_COL_MAP: dict[str, str] = {
        "eps_norm_est_avg_ntm": "eps_norm_ntm",
        "eps_norm_est_avg_fy1e": "eps_norm_fy1e",
        "eps_gaap_est_avg_ntm": "eps_gaap_ntm",
        "eps_gaap_est_avg_fy1e": "eps_gaap_fy1e",
        "eps_est_avg_rev_pct_fy1e_1w": "revision_1w",
        "eps_est_avg_rev_pct_fy1e_1m": "revision_1m",
        "eps_est_avg_rev_pct_fy1e_3m": "revision_3m",
        "eps_est_avg_rev_pct_fy1e_6m": "revision_6m",
        "eps_est_avg_rev_pct_fy1e_1y": "revision_1y",
        "eps_gaap_est_avg_rev_pct_fy1e_1m": "gaap_revision_1m",
        "eps_gaap_est_avg_rev_pct_fy1e_3m": "gaap_revision_3m",
        "eps_gaap_est_avg_rev_pct_fy1e_6m": "gaap_revision_6m",
        "eps_gaap_est_avg_rev_pct_fy1e_1y": "gaap_revision_1y",
        "eps_norm_est_num_fy1e": "analyst_count",
    }

    _HISTORY_COL_MAP: dict[str, str] = {
        "net_eps_basic_fq": "eps_basic_fq",
        "net_eps_basic_1fqfq": "eps_basic_1fqfq",
        "net_eps_basic_2fqfq": "eps_basic_2fqfq",
        "net_eps_basic_3fqfq": "eps_basic_3fqfq",
        "net_eps_basic_4fqfq": "eps_basic_4fqfq",
        "net_eps_basic_fy": "eps_basic_fy",
        "net_eps_basic_1fy": "eps_basic_1fy",
        "net_eps_basic_2fy": "eps_basic_2fy",
        "net_eps_basic_3fy": "eps_basic_3fy",
        "net_eps_basic_4fy": "eps_basic_4fy",
        "net_eps_basic_5fy": "eps_basic_5fy",
        "eps_adj_ltm": "eps_adj_ltm",
        "eps_adj_fy": "eps_adj_fy",
        "eps_adj_1fy": "eps_adj_1fy",
        "eps_adj_fq": "eps_adj_fq",
        "eps_adj_1fqfq": "eps_adj_1fqfq",
        "eps_adj_2fqfq": "eps_adj_2fqfq",
        "eps_adj_3fqfq": "eps_adj_3fqfq",
        "eps_adj_4fqfq": "eps_adj_4fqfq",
        "eps_cont_fq": "eps_cont_fq",
        "eps_cont_1fqfq": "eps_cont_1fqfq",
        "eps_cont_2fqfq": "eps_cont_2fqfq",
        "eps_cont_3fqfq": "eps_cont_3fqfq",
        "eps_cont_4fqfq": "eps_cont_4fqfq",
    }

    def _row_to_forward_signals(self, row: pd.Series) -> Optional[ForwardEstimateSignals]:
        """Extract ForwardEstimateSignals from a DataFrame row."""
        kwargs: dict = {}
        any_present = False
        for df_col, field_name in self._FORWARD_COL_MAP.items():
            val = row.get(df_col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                kwargs[field_name] = int(val) if field_name == "analyst_count" else float(val)
                any_present = True
        if not any_present:
            return None
        return ForwardEstimateSignals(**kwargs)

    def _row_to_history(self, row: pd.Series) -> ReportedEPSHistory:
        """Extract ReportedEPSHistory from a DataFrame row."""
        kwargs: dict = {}
        for df_col, field_name in self._HISTORY_COL_MAP.items():
            val = row.get(df_col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                kwargs[field_name] = float(val)
        return ReportedEPSHistory(**kwargs)

    def analyze_dataframe_enhanced(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        ticker_col: str = "ticker",
        name_col: str = "name",
    ) -> pd.DataFrame:
        """Analyze earnings beat probabilities using enhanced three-layer fusion.

        Falls back to trajectory-proxy method when forward data is unavailable.
        Uses vectorized computation for the proxy path to improve performance.

        Args:
            df: DataFrame with equities data.
            sector_col: Column name for sector.
            ticker_col: Column name for ticker.
            name_col: Column name for company name.

        Returns:
            DataFrame with enriched probability analysis results.
        """
        # Composite/quality columns to pass through from mv_all_stock_features
        _PASSTHROUGH_COLS = {
            "accounting_quality_score": "accounting_quality_score",
            "combined_distress_risk_score": "distress_risk_score",
            "gaap_adj_eps_gap_pct": "gaap_adj_eps_gap_pct",
            "piotroski_f_score": "piotroski_f_score",
            "eps_revision_momentum": "eps_revision_momentum",
            "altman_z_score": "altman_z_score",
        }

        # Check which rows have forward data available
        forward_col_keys = list(self._FORWARD_COL_MAP.keys())
        has_any_forward = (
            df[[c for c in forward_col_keys if c in df.columns]].notna().any(axis=1)
            if any(c in df.columns for c in forward_col_keys)
            else pd.Series(False, index=df.index)
        )

        # Split into enhanced (row-by-row, needs dataclass construction) and proxy (vectorizable)
        enhanced_mask = has_any_forward
        proxy_mask = ~enhanced_mask & (
            df["eps_trajectory_score"].notna()
            if "eps_trajectory_score" in df.columns
            else pd.Series(False, index=df.index)
        )

        results = []

        # --- Enhanced path (row-by-row for forward signal construction) ---
        if enhanced_mask.any():
            for idx in df.index[enhanced_mask]:
                row = df.loc[idx]
                ticker = row.get(ticker_col, "UNKNOWN")
                name = row.get(name_col, "UNKNOWN")
                sector = row.get(sector_col, None)

                forward_signals = self._row_to_forward_signals(row)
                history = self._row_to_history(row)

                if forward_signals is None:
                    continue

                prob_result = self.compute_forward_adjusted_beat_probability(
                    reported_history=history,
                    forward_signals=forward_signals,
                    sector=sector,
                )
                n_beats, n_total = history.count_yoy_improvements()
                dynamic_total = history.total_reports_count
                effective_total = max(n_total, dynamic_total) if dynamic_total > 0 else n_total
                historical_beat_rate = n_beats / effective_total if effective_total > 0 else 0.0

                beat_classification = (
                    "likely_beat" if prob_result["posterior_mean"] > 0.5 else "uncertain"
                )
                # Compute eps_positive_streak from reported history
                eps_positive_streak = history.quarterly_beat_streak()

                record = _extract_identifiers(row)
                record.update(
                    {
                        "ticker": ticker,
                        "name": name,
                        "sector": sector,
                        "historical_beats": n_beats,
                        "total_reports": effective_total,
                        "dynamic_total_reports": dynamic_total,
                        "historical_beat_rate": historical_beat_rate,
                        "posterior_beat_prob": prob_result["posterior_mean"],
                        "posterior_std": prob_result["posterior_std"],
                        "ci_90_lower": prob_result["credible_interval_90"][0],
                        "ci_90_upper": prob_result["credible_interval_90"][1],
                        "ci_95_lower": prob_result["credible_interval_95"][0],
                        "ci_95_upper": prob_result["credible_interval_95"][1],
                        "confidence_score": prob_result["confidence_score"],
                        "prior_influence_pct": prob_result["prior_influence_pct"],
                        "effective_sample_size": prob_result["effective_sample_size"],
                        "classification_confidence": prob_result["classification_confidence"],
                        "beat_classification": beat_classification,
                        "gaap_revision_momentum": forward_signals.gaap_revision_momentum,
                        "gaap_norm_spread": forward_signals.gaap_norm_spread,
                        "revision_trend_short": forward_signals.revision_trend_short,
                        "revision_trend_medium": forward_signals.revision_trend_medium,
                        "eps_norm_est_fy1e": forward_signals.eps_norm_fy1e,
                        "eps_norm_est_ntm": forward_signals.eps_norm_ntm,
                        "eps_gaap_est_ntm": forward_signals.eps_gaap_ntm,
                        "eps_gaap_est_fy1e": forward_signals.eps_gaap_fy1e,
                        "analyst_count": forward_signals.analyst_count,
                        "next_earnings_status": row.get("next_earnings_status", None),
                        "quarterly_beat_streak": eps_positive_streak,
                        "data_source": "forward_enhanced",
                        "eps_positive_streak": eps_positive_streak,
                    }
                )

                # Pass through composite/quality columns from mv_all_stock_features
                for src_col, out_key in _PASSTHROUGH_COLS.items():
                    val = row.get(src_col, None)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        record[out_key] = val
                    else:
                        record[out_key] = None

                results.append(record)

        # --- Proxy path (vectorized Beta-Binomial for trajectory scores) ---
        if proxy_mask.any():
            proxy_df = df.loc[proxy_mask].copy()
            trajectory = proxy_df["eps_trajectory_score"].fillna(50)

            # Dynamic n_total: use eps_positive_years or eps_improvement_count if available,
            # else scale by trajectory score confidence band
            if "eps_positive_years" in proxy_df.columns:
                n_total_proxy = proxy_df["eps_positive_years"].fillna(0).clip(lower=0)
                n_total_proxy = n_total_proxy.clip(lower=3, upper=15).astype(int)
            elif "eps_improvement_count" in proxy_df.columns:
                n_total_proxy = (
                    proxy_df["eps_improvement_count"].fillna(3).clip(lower=3, upper=15).astype(int)
                )
            else:
                # Graduated proxy: higher trajectory scores imply more consistent data
                n_total_proxy = pd.Series(
                    np.where(
                        trajectory >= 80,
                        8,
                        np.where(
                            trajectory >= 60,
                            6,
                            np.where(trajectory >= 40, 5, np.where(trajectory >= 20, 4, 3)),
                        ),
                    ),
                    index=proxy_df.index,
                )
            n_beats_proxy = (
                (trajectory / 100 * n_total_proxy).astype(int).clip(lower=0, upper=n_total_proxy)
            )

            # Vectorized posterior
            prior = self.default_prior
            post_alpha = prior.alpha + n_beats_proxy
            post_beta = prior.beta + (n_total_proxy - n_beats_proxy)
            posterior_mean = post_alpha / (post_alpha + post_beta)
            posterior_std = np.sqrt(
                (post_alpha * post_beta)
                / ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
            )
            ci_90_lower = pd.Series(
                stats.beta.ppf(0.05, post_alpha, post_beta), index=proxy_df.index
            )
            ci_90_upper = pd.Series(
                stats.beta.ppf(0.95, post_alpha, post_beta), index=proxy_df.index
            )
            ci_95_lower = pd.Series(
                stats.beta.ppf(0.025, post_alpha, post_beta), index=proxy_df.index
            )
            ci_95_upper = pd.Series(
                stats.beta.ppf(0.975, post_alpha, post_beta), index=proxy_df.index
            )
            # Multi-component confidence score (replaces constant concentration/20)
            prior_total = prior.alpha + prior.beta
            confidence_score = pd.Series(
                compute_beta_confidence_score(
                    post_alpha.values,
                    post_beta.values,
                    prior_alpha=prior.alpha,
                    prior_beta=prior.beta,
                    normalization_factor=self.CONFIDENCE_NORMALIZATION_FACTOR,
                ),
                index=proxy_df.index,
            )
            concentration = post_alpha + post_beta
            prior_influence = prior_total / concentration * 100.0
            effective_sample = concentration - prior_total
            beat_class = np.where(posterior_mean > 0.5, "likely_beat", "uncertain")
            class_conf = np.where(
                confidence_score >= 0.6, "High", np.where(confidence_score >= 0.3, "Medium", "Low")
            )

            for i, idx in enumerate(proxy_df.index):
                row = proxy_df.loc[idx]
                record = _extract_identifiers(row)
                record.update(
                    {
                        "historical_beats": int(n_beats_proxy.iloc[i]),
                        "total_reports": int(n_total_proxy.iloc[i]),
                        "dynamic_total_reports": 0,
                        "historical_beat_rate": float(
                            n_beats_proxy.iloc[i] / n_total_proxy.iloc[i]
                        ),
                        "prior_alpha": prior.alpha,
                        "prior_beta": prior.beta,
                        "posterior_alpha": float(post_alpha.iloc[i]),
                        "posterior_beta": float(post_beta.iloc[i]),
                        "posterior_beat_prob": float(posterior_mean.iloc[i]),
                        "posterior_std": float(posterior_std.iloc[i]),
                        "ci_90_lower": float(ci_90_lower.iloc[i]),
                        "ci_90_upper": float(ci_90_upper.iloc[i]),
                        "ci_95_lower": float(ci_95_lower.iloc[i]),
                        "ci_95_upper": float(ci_95_upper.iloc[i]),
                        "confidence_score": float(confidence_score.iloc[i]),
                        "prior_influence_pct": float(prior_influence.iloc[i]),
                        "effective_sample_size": float(effective_sample.iloc[i]),
                        "classification_confidence": str(class_conf[i]),
                        "beat_classification": str(beat_class[i]),
                        "gaap_revision_momentum": None,
                        "gaap_norm_spread": None,
                        "revision_trend_short": None,
                        "revision_trend_medium": None,
                        "eps_norm_est_fy1e": None,
                        "eps_norm_est_ntm": None,
                        "eps_gaap_est_ntm": None,
                        "eps_gaap_est_fy1e": None,
                        "analyst_count": None,
                        "next_earnings_status": row.get("next_earnings_status", None),
                        "quarterly_beat_streak": None,
                        "data_source": "trajectory_proxy",
                        "eps_positive_streak": None,
                    }
                )

                # Pass through composite/quality columns from mv_all_stock_features
                for src_col, out_key in _PASSTHROUGH_COLS.items():
                    val = row.get(src_col, None)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        record[out_key] = val
                    else:
                        record[out_key] = None

                results.append(record)

        return pd.DataFrame(results)


# =============================================================================
# EPS STREAK ANALYZER
# =============================================================================


class EPSStreakAnalyzer:
    """
    Analyzer for EPS beat/miss streaks with predictive capabilities.

    Uses Markov chain analysis and historical patterns to predict
    streak continuation vs. mean reversion probabilities.
    """

    def __init__(self, mean_reversion_weight: float = 0.3):
        """
        Initialize streak analyzer.

        Args:
            mean_reversion_weight: Weight for mean reversion in predictions (0-1)
                                  Higher values increase mean reversion tendency
        """
        self.mean_reversion_weight = mean_reversion_weight

    def compute_streak_from_trajectory(
        self,
        eps_trajectory_score: float,
        eps_positive_streak: Optional[int] = None,
        eps_improvement_count: Optional[int] = None,
        ticker: str = "",
        name: str = "",
        sector: str = "",
        industry: str = "",
        country: str = "",
        exchange: str = "",
        reported_history: Optional[ReportedEPSHistory] = None,
        forward_signals: Optional[ForwardEstimateSignals] = None,
    ) -> EPSStreakResult:
        """
        Compute streak analysis from trajectory score and related metrics.

        Args:
            eps_trajectory_score: EPS trajectory score (0-100)
            eps_positive_streak: Number of positive EPS quarters
            eps_improvement_count: Number of YoY improvements
            ticker: Ticker symbol
            name: Company name
            sector: Sector
            industry: Industry
            country: Country
            exchange: Exchange
            reported_history: Optional actual reported EPS data for dynamic counts
            forward_signals: Optional forward estimate signals for probability refinement

        Returns:
            EPSStreakResult with analysis
        """
        # --- Dynamically derive streak from reported history if available ---
        if reported_history is not None and reported_history.quarterly_reports_count > 0:
            dynamic_streak = reported_history.quarterly_beat_streak()
            dynamic_total = reported_history.quarterly_reports_count
        else:
            dynamic_streak = None
            dynamic_total = 0

        # Estimate current streak from available metrics
        if eps_positive_streak is not None and not pd.isna(eps_positive_streak):
            current_streak = int(eps_positive_streak)
            streak_type = "beat" if current_streak > 0 else "miss"
        elif dynamic_streak is not None and dynamic_streak > 0:
            current_streak = dynamic_streak
            streak_type = "beat"
        elif eps_trajectory_score is not None and not pd.isna(eps_trajectory_score):
            # Infer from trajectory score
            if eps_trajectory_score >= 80:
                current_streak = 4
                streak_type = "beat"
            elif eps_trajectory_score >= 60:
                current_streak = 3
                streak_type = "beat"
            elif eps_trajectory_score >= 40:
                current_streak = 1
                streak_type = "meet"
            elif eps_trajectory_score >= 20:
                current_streak = 2
                streak_type = "miss"
            else:
                current_streak = 3
                streak_type = "miss"
        else:
            current_streak = 0
            streak_type = "meet"

        # Compute continuation probability using geometric decay model
        # P(continue) = base_rate * decay^streak_length
        base_continuation = 0.65  # Base continuation probability
        decay_factor = 0.85  # Decay per streak length

        continuation_prob = base_continuation * (decay_factor ** abs(current_streak))

        # --- Forward estimate adjustment ---
        # If revision momentum is positive and streak is a beat, boost continuation
        if forward_signals is not None and forward_signals.has_sufficient_data:
            momentum = forward_signals.gaap_revision_momentum  # 0-100
            # Positive momentum reinforces beat streaks, undermines miss streaks
            momentum_adjustment = (momentum - 50.0) / 200.0  # Range: -0.25 to +0.25
            if streak_type == "beat":
                continuation_prob += momentum_adjustment
            elif streak_type == "miss":
                continuation_prob -= momentum_adjustment

            # GAAP-Norm spread: large divergence reduces confidence in continuation
            spread = forward_signals.gaap_norm_spread
            if spread is not None and abs(spread) > 20.0:
                continuation_prob -= min(0.10, abs(spread) / 500.0)

            continuation_prob = max(0.05, min(0.95, continuation_prob))

        # Apply mean reversion adjustment
        mean_reversion_prob = 1 - continuation_prob
        mean_reversion_prob = mean_reversion_prob * (
            1 - self.mean_reversion_weight
        ) + self.mean_reversion_weight * (1 - continuation_prob)

        # Confidence based on streak length and data availability
        confidence = max(0.3, 1 - abs(current_streak) * 0.1)
        # Boost confidence if we have more dynamic data points
        if dynamic_total > 3:
            confidence = min(1.0, confidence + 0.1)
        if forward_signals is not None and forward_signals.has_sufficient_data:
            confidence = min(1.0, confidence + 0.05)

        # Expected next outcome
        if continuation_prob > 0.5:
            expected_next = streak_type
        else:
            expected_next = "beat" if streak_type == "miss" else "miss"

        return EPSStreakResult(
            ticker=ticker,
            name=name,
            sector=sector,
            industry=industry,
            country=country,
            exchange=exchange,
            current_streak=current_streak,
            streak_type=streak_type,
            max_streak_beat=max(current_streak if streak_type == "beat" else 0, 0),
            max_streak_miss=max(current_streak if streak_type == "miss" else 0, 0),
            streak_continuation_prob=continuation_prob,
            mean_reversion_prob=mean_reversion_prob,
            expected_next_outcome=expected_next,
            confidence_level=confidence,
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        ticker_col: str = "ticker",
        trajectory_col: str = "eps_trajectory_score",
        streak_col: str = "eps_positive_streak",
        improvement_col: str = "eps_improvement_count",
        name_col: str = "name",
        sector_col: str = "sector",
        industry_col: str = "industry",
        country_col: str = "country",
        exchange_col: str = "exchange",
    ) -> pd.DataFrame:
        """
        Analyze EPS streaks for entire DataFrame.

        Now dynamically derives total_reports from non-null reported EPS data
        and incorporates forward estimate signals when available.

        Args:
            df: DataFrame with EPS data
            trajectory_col: Column for trajectory score
            streak_col: Column for positive streak count
            improvement_col: Column for improvement count
            ticker_col: Column for ticker
            name_col: Column for company name
            sector_col: Column for sector
            industry_col: Column for industry
            country_col: Column for country
            exchange_col: Column for exchange

        Returns:
            DataFrame with streak analysis
        """
        # Define the expected output columns so an empty result retains the schema
        _OUTPUT_COLS = [
            "current_streak",
            "streak_type",
            "continuation_probability",
            "mean_reversion_probability",
            "expected_next_outcome",
            "prediction_confidence",
            "dynamic_total_reports",
            "historical_beat_rate",
            "gaap_revision_momentum",
            "next_earnings_status",
            "eps_positive_streak",
            "model_confidence",
            "map_estimate",
            "accounting_quality_score",
            "distress_risk_score",
            "gaap_adj_eps_gap_pct",
            "piotroski_f_score",
            "eps_revision_momentum",
            "altman_z_score",
        ]

        # Composite/quality columns to pass through from mv_all_stock_features
        _PASSTHROUGH_COLS = [
            "accounting_quality_score",
            "combined_distress_risk_score",
            "gaap_adj_eps_gap_pct",
            "piotroski_f_score",
            "eps_revision_momentum",
            "altman_z_score",
        ]

        # Check if forward/history columns are available
        has_history_cols = any(
            col in df.columns for col in EarningsBeatProbabilityModel._HISTORY_COL_MAP
        )
        has_forward_cols = any(
            col in df.columns for col in EarningsBeatProbabilityModel._FORWARD_COL_MAP
        )

        # Create a temporary model instance for column mapping helpers
        _model = EarningsBeatProbabilityModel()

        results = []

        for _, row in df.iterrows():
            trajectory = row.get(trajectory_col, None)
            streak = row.get(streak_col, None)
            improvement = row.get(improvement_col, None)
            ticker = row.get(ticker_col, "UNKNOWN")
            name = row.get(name_col, "")
            sector = row.get(sector_col, "")
            industry = row.get(industry_col, "")
            country = row.get(country_col, "")
            exchange = row.get(exchange_col, "")

            if trajectory is None or pd.isna(trajectory):
                continue

            # Build reported history and forward signals when columns exist
            reported_history = None
            forward_signals = None
            dynamic_total = 0

            if has_history_cols:
                reported_history = _model._row_to_history(row)
                dynamic_total = reported_history.total_reports_count

            if has_forward_cols:
                forward_signals = _model._row_to_forward_signals(row)

            result = self.compute_streak_from_trajectory(
                eps_trajectory_score=trajectory,
                eps_positive_streak=streak,
                eps_improvement_count=improvement,
                ticker=ticker,
                name=name,
                sector=sector,
                industry=industry,
                country=country,
                exchange=exchange,
                reported_history=reported_history,
                forward_signals=forward_signals,
            )

            # Historical beat rate from dynamically derived total
            n_beats_yoy, n_total_yoy = (
                reported_history.count_yoy_improvements()
                if reported_history is not None
                else (0, 0)
            )
            effective_total = max(n_total_yoy, dynamic_total) if dynamic_total > 0 else n_total_yoy
            historical_beat_rate = n_beats_yoy / effective_total if effective_total > 0 else 0.0

            # --- Compute model_confidence and map_estimate ---
            # model_confidence: how decisive the streak prediction is (0-1)
            model_confidence = (
                abs(result.streak_continuation_prob - 0.5) * 2.0 * result.confidence_level
            )

            # map_estimate: MAP (maximum a posteriori) beat probability
            # Use Beta posterior mode when we have enough data
            if effective_total > 0:
                prior = _model.default_prior
                n_beats = n_beats_yoy
                post_a = prior.alpha + n_beats
                post_b = prior.beta + (effective_total - n_beats)
                if post_a > 1 and post_b > 1:
                    map_estimate = (post_a - 1) / (post_a + post_b - 2)
                else:
                    map_estimate = post_a / (post_a + post_b)
            else:
                map_estimate = trajectory / 100.0 if trajectory is not None else 0.5

            record = _extract_identifiers(row)
            record.update(
                {
                    "current_streak": result.current_streak,
                    "streak_type": result.streak_type,
                    "continuation_probability": result.streak_continuation_prob,
                    "mean_reversion_probability": result.mean_reversion_prob,
                    "expected_next_outcome": result.expected_next_outcome,
                    "prediction_confidence": result.confidence_level,
                    "dynamic_total_reports": dynamic_total,
                    "historical_beat_rate": historical_beat_rate,
                    "gaap_revision_momentum": (
                        forward_signals.gaap_revision_momentum
                        if forward_signals is not None
                        else None
                    ),
                    "next_earnings_status": row.get("next_earnings_status", None),
                    # EPS positive streak passthrough
                    "eps_positive_streak": (
                        int(streak) if streak is not None and not pd.isna(streak) else None
                    ),
                    # Model-derived fields
                    "model_confidence": round(model_confidence, 10),
                    "map_estimate": round(map_estimate, 10),
                }
            )

            # Pass through composite/quality columns from mv_all_stock_features
            for col in _PASSTHROUGH_COLS:
                val = row.get(col, None)
                # Rename combined_distress_risk_score → distress_risk_score for export
                out_key = "distress_risk_score" if col == "combined_distress_risk_score" else col
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    record[out_key] = val
                else:
                    record[out_key] = None

            results.append(record)

        if not results:
            # Return empty DataFrame with expected columns to prevent downstream schema errors
            return pd.DataFrame(columns=_OUTPUT_COLS)

        return pd.DataFrame(results)


# =============================================================================
# MODEL CONFIDENCE ESTIMATOR
# =============================================================================


class ModelConfidenceEstimator:
    """
    Estimator for model confidence and calibration metrics.

    Provides comprehensive confidence assessment including:
    - Brier score for probability calibration
    - Reliability diagrams
    - Confidence interval coverage
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize confidence estimator.

        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins

    def compute_brier_score(
        self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray
    ) -> float:
        """
        Compute Brier score for probability predictions.

        Brier score = (1/N) * Σ(predicted - actual)²
        Lower is better, 0 = perfect, 0.25 = random for binary.
        """
        return np.mean((predicted_probs - actual_outcomes) ** 2)

    def compute_calibration_error(
        self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray
    ) -> tuple[float, dict]:
        """
        Compute Expected Calibration Error (ECE) and reliability diagram data.

        ECE measures how well predicted probabilities match observed frequencies.
        """
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        reliability_data = {
            "bin_centers": [],
            "observed_freq": [],
            "predicted_mean": [],
            "count": [],
        }

        total_samples = len(predicted_probs)
        ece = 0.0

        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_pred = predicted_probs[mask].mean()
                bin_actual = actual_outcomes[mask].mean()
                bin_count = mask.sum()

                reliability_data["bin_centers"].append((bins[i] + bins[i + 1]) / 2)
                reliability_data["observed_freq"].append(bin_actual)
                reliability_data["predicted_mean"].append(bin_pred)
                reliability_data["count"].append(bin_count)

                ece += (bin_count / total_samples) * abs(bin_actual - bin_pred)

        return ece, reliability_data

    def compute_confidence_metrics(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        model_name: str = "Earnings Beat Model",
    ) -> ModelConfidenceResult:
        """
        Compute comprehensive confidence metrics for predictions.

        Args:
            predicted_probs: Array of predicted probabilities
            actual_outcomes: Array of actual binary outcomes (0 or 1)
            model_name: Name for the model

        Returns:
            ModelConfidenceResult with all metrics
        """
        # Brier score
        brier = self.compute_brier_score(predicted_probs, actual_outcomes)

        # Log loss (cross-entropy)
        eps = 1e-15
        clipped_probs = np.clip(predicted_probs, eps, 1 - eps)
        log_loss = -np.mean(
            actual_outcomes * np.log(clipped_probs)
            + (1 - actual_outcomes) * np.log(1 - clipped_probs)
        )

        # Calibration error and reliability data
        ece, reliability_data = self.compute_calibration_error(predicted_probs, actual_outcomes)

        # AUC-ROC for discrimination
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(actual_outcomes, predicted_probs)
        except Exception:
            # Fallback: simple rank-based AUC approximation
            n_pos = actual_outcomes.sum()
            n_neg = len(actual_outcomes) - n_pos
            if n_pos > 0 and n_neg > 0:
                ranks = stats.rankdata(predicted_probs)
                auc = (ranks[actual_outcomes == 1].sum() - n_pos * (n_pos + 1) / 2) / (
                    n_pos * n_neg
                )
            else:
                auc = 0.5

        # Confidence intervals for predictions
        ci_coverage = self._compute_ci_coverage(predicted_probs, actual_outcomes, n_observations=5)

        # Overall confidence score (0-100)
        # Weighted combination of metrics with discrimination floor
        base_score = (
            (1 - brier) * 30  # Lower brier is better
            + (1 - ece) * 30  # Lower ECE is better
            + auc * 40  # Higher AUC is better
        )

        # Penalty: AUC below 0.5 means model is anti-discriminating
        if auc < 0.5:
            discrimination_penalty = (0.5 - auc) * 60  # Up to -30 points
            base_score -= discrimination_penalty

        overall = min(100, max(0, base_score))

        return ModelConfidenceResult(
            model_name=model_name,
            brier_score=brier,
            log_loss=log_loss,
            calibration_error=ece,
            discrimination_auc=auc,
            reliability_diagram_data=reliability_data,
            confidence_intervals=ci_coverage,
            overall_confidence=overall,
        )

    def _compute_ci_coverage(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        n_observations: int = 5,
    ) -> dict:
        """Compute confidence interval coverage rates."""
        coverage_90 = 0.0
        coverage_95 = 0.0
        n = len(predicted_probs)

        for i, (prob, actual) in enumerate(zip(predicted_probs, actual_outcomes)):
            # Use actual observation count instead of hardcoded 10
            std = np.sqrt(prob * (1 - prob) / max(n_observations, 1))
            ci_90 = (max(0, prob - 1.645 * std), min(1, prob + 1.645 * std))
            ci_95 = (max(0, prob - 1.96 * std), min(1, prob + 1.96 * std))

            if ci_90[0] <= actual <= ci_90[1]:
                coverage_90 += 1
            if ci_95[0] <= actual <= ci_95[1]:
                coverage_95 += 1

        return {
            "coverage_90": coverage_90 / n if n > 0 else 0,
            "coverage_95": coverage_95 / n if n > 0 else 0,
        }


class CreditRiskProbabilityModel:
    """
    Bayesian framework to estimate likelihood of financial distress.

    Enhanced features: altman_z_score, altman_z_trend, liquidity_stress_score,
    cash_runway_months, accumulated_deficit_flag, combined_distress_score,
    wc_deteriorating_flag, debt_deleveraging, interest_coverage, quick_ratio,
    beta_stability_score

    Leverage & Liquidity enrichment (v3.4):
    debt_3y_cagr, debt_4q_trend, debt_yoy_change, adequate_cash_buffer,
    cash_vs_5y_avg, balance_sheet_strength, debt_maturity_risk, equity_ratio,
    wc_volatility, wc_efficiency_score, retained_earnings_vs_5y

    Quality & Risk enrichment (v3.4):
    distress_risk_score, retained_earnings_growth, beta_trend
    """

    def __init__(
        self,
        distress_threshold: float = 1.81,
        prior_alpha: float = 2.0,
        prior_beta: float = 3.0,  # Slightly pessimistic prior
        n_mcmc_samples: int = 10000,
        burn_in: int = 2000,
        use_mcmc: bool = True,
        # NEW: Leverage & Liquidity enrichment
        use_debt_trajectory: bool = True,
        use_cash_buffer_signals: bool = True,
        use_balance_sheet_quality: bool = True,
        use_wc_deep_signals: bool = True,
        # NEW: Quality & Risk enrichment
        use_quality_risk_flags: bool = True,
    ):
        self.distress_threshold = distress_threshold
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_mcmc_samples = n_mcmc_samples
        self.burn_in = burn_in
        self.use_mcmc = use_mcmc
        self.use_debt_trajectory = use_debt_trajectory
        self.use_cash_buffer_signals = use_cash_buffer_signals
        self.use_balance_sheet_quality = use_balance_sheet_quality
        self.use_wc_deep_signals = use_wc_deep_signals
        self.use_quality_risk_flags = use_quality_risk_flags

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze dataframe for credit risk with enhanced features."""
        results = []

        for _, row in df.iterrows():
            # Core distress indicators
            z_score = row.get("altman_z_score", 3.0)
            z_trend = row.get("altman_z_trend", 0)  # NEW: Z-score trajectory
            liquidity_stress = row.get("liquidity_stress_score", 50)
            cash_runway = row.get("cash_runway_months", 24)
            accumulated_deficit = row.get("accumulated_deficit_flag", 0)

            # NEW: Additional risk factors from views
            combined_distress = row.get("combined_distress_risk_score", 50)
            wc_deteriorating = row.get("wc_deteriorating_flag", 0)
            debt_deleveraging = row.get("debt_deleveraging", 0)  # Negative = more debt
            interest_coverage = row.get("interest_coverage", 5.0)
            quick_ratio = row.get("quick_ratio", 1.5)
            beta_stability = row.get("beta_stability_score", 50)

            # NEW: Debt trajectory signals (calc_total_debt_temporal)
            debt_3y_cagr = row.get("debt_3y_cagr", 0)
            debt_4q_trend = row.get("debt_4q_trend", 0)
            debt_yoy_change = row.get("debt_yoy_change", 0)

            # NEW: Cash buffer signals (calc_financial_distress_features + calc_balance_sheet_dynamics)
            adequate_cash_buffer = row.get("adequate_cash_buffer", 1)
            cash_vs_5y_avg = row.get("cash_vs_5y_avg", 1.0)

            # NEW: Balance sheet quality (calc_balance_sheet_dynamics)
            balance_sheet_strength = row.get("balance_sheet_strength", 50)
            debt_maturity_risk = row.get("debt_maturity_risk", 0)
            equity_ratio = row.get("equity_ratio", 0.5)

            # NEW: Working capital deep (calc_working_capital_deep_features + temporal)
            wc_volatility = row.get("wc_volatility", 0)
            wc_efficiency_score = row.get("wc_efficiency_score", 50)
            retained_earnings_vs_5y = row.get("retained_earnings_vs_5y", 1.0)

            # NEW: Quality & Risk flags
            distress_risk_score = row.get("distress_risk_score", 50)
            retained_earnings_growth = row.get("retained_earnings_growth", 0)
            beta_trend = row.get("beta_trend", 0)

            # Bayesian-style probability estimation with enhanced inputs
            # Prior based on Z-Score zones
            if z_score < 1.81:
                base_prob = 0.75
            elif z_score < 2.67:  # Grey zone
                base_prob = 0.35
            elif z_score < 3.0:
                base_prob = 0.15
            else:
                base_prob = 0.05

            # Evidence-based adjustments
            adjustments = 0.0

            # Z-score trend (deteriorating = higher risk)
            if z_trend < -0.5:
                adjustments += 0.15
            elif z_trend > 0.5:
                adjustments -= 0.05

            # Liquidity stress
            if liquidity_stress > 70:
                adjustments += 0.12
            elif liquidity_stress < 30:
                adjustments -= 0.05

            # Cash runway
            if cash_runway < 6:
                adjustments += 0.18
            elif cash_runway < 12:
                adjustments += 0.08
            elif cash_runway > 24:
                adjustments -= 0.03

            # Working capital deterioration
            if wc_deteriorating == 1:
                adjustments += 0.10

            # Debt trends
            if debt_deleveraging is not None and debt_deleveraging < -0.1:
                adjustments += 0.08  # Increasing debt burden

            # Interest coverage
            if interest_coverage is not None and interest_coverage < 1.5:
                adjustments += 0.15
            elif interest_coverage is not None and interest_coverage < 3.0:
                adjustments += 0.05

            # Quick ratio
            if quick_ratio is not None and quick_ratio < 0.8:
                adjustments += 0.10

            if accumulated_deficit == 1:
                adjustments += 0.08

            # NEW: Debt trajectory — rising debt over 3 years
            if self.use_debt_trajectory:
                if debt_3y_cagr is not None and debt_3y_cagr > 10:
                    adjustments += 0.10
                if debt_4q_trend is not None and debt_4q_trend > 0.05:
                    adjustments += 0.06
                if debt_yoy_change is not None and debt_yoy_change > 20:
                    adjustments += 0.08

            # NEW: Cash buffer adequacy
            if self.use_cash_buffer_signals:
                if adequate_cash_buffer == 0:
                    adjustments += 0.12
                if cash_vs_5y_avg is not None and cash_vs_5y_avg < 0.5:
                    adjustments += 0.08

            # NEW: Balance sheet quality composite
            if self.use_balance_sheet_quality:
                if balance_sheet_strength is not None and balance_sheet_strength < 25:
                    adjustments += 0.12
                elif balance_sheet_strength is not None and balance_sheet_strength > 75:
                    adjustments -= 0.05
                if debt_maturity_risk is not None and debt_maturity_risk > 70:
                    adjustments += 0.10
                if equity_ratio is not None and equity_ratio < 0.15:
                    adjustments += 0.10

            # NEW: Working capital deep signals
            if self.use_wc_deep_signals:
                if wc_volatility is not None and wc_volatility > 0.5:
                    adjustments += 0.06
                if wc_efficiency_score is not None and wc_efficiency_score < 25:
                    adjustments += 0.08
                if retained_earnings_vs_5y is not None and retained_earnings_vs_5y < 0.5:
                    adjustments += 0.07

            # NEW: Quality & Risk flags
            if self.use_quality_risk_flags:
                if distress_risk_score is not None and distress_risk_score > 70:
                    adjustments += 0.10
                if retained_earnings_growth is not None and retained_earnings_growth < -20:
                    adjustments += 0.08
                if beta_trend is not None and beta_trend > 0.3:
                    adjustments += 0.05

            prob = min(0.99, max(0.01, base_prob + adjustments))

            # Compute confidence interval width based on data completeness
            data_points = sum(
                1
                for v in [
                    z_score,
                    liquidity_stress,
                    cash_runway,
                    interest_coverage,
                    quick_ratio,
                    debt_3y_cagr,
                    balance_sheet_strength,
                    debt_maturity_risk,
                    wc_efficiency_score,
                    equity_ratio,
                    distress_risk_score,
                ]
                if v is not None and not pd.isna(v)
            )
            ci_width = max(0.03, 0.18 - (data_points * 0.015))

            risk_level = "Low"
            if prob > 0.7:
                risk_level = "Distressed"
            elif prob > 0.5:
                risk_level = "High"
            elif prob > 0.3:
                risk_level = "Medium"

            record = _extract_identifiers(row)
            record.update(
                {
                    "beta_stability_score": beta_stability,
                    "combined_distress_risk_score": combined_distress,
                    "distress_probability": prob,
                    "liquidity_stress_score": liquidity_stress,
                    "cash_runway_months": cash_runway,
                    "altman_z_score": z_score,
                    "altman_z_trend": z_trend,
                    "interest_coverage": interest_coverage,
                    "quick_ratio": quick_ratio,
                    "risk_level": risk_level,
                    "ci_lower": max(0, prob - ci_width),
                    "ci_upper": min(1, prob + ci_width),
                    "debt_3y_cagr": debt_3y_cagr,
                    "debt_maturity_risk": debt_maturity_risk,
                    "balance_sheet_strength": balance_sheet_strength,
                    "wc_efficiency_score": wc_efficiency_score,
                    "distress_risk_score": distress_risk_score,
                    "data_quality_score": data_points / 11.0,
                }
            )
            results.append(record)

        result_df = pd.DataFrame(results)

        # MCMC enrichment path
        if self.use_mcmc and not result_df.empty:
            result_df = self._apply_mcmc_posteriors(result_df, df)

        return result_df

    def _apply_mcmc_posteriors(
        self, result_df: pd.DataFrame, source_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply MCMC posterior estimation for distress probability."""
        from probabilistic_ml_model.statistical_functions.statistical_analysis import (
            metropolis_hastings_sampler,
            mcmc_student_t,
            hierarchical_mcmc_by_sector,
        )

        z_scores = (
            source_df["altman_z_score"].dropna().values
            if "altman_z_score" in source_df.columns
            else np.array([])
        )
        if len(z_scores) < 10:
            return result_df

        try:
            # Task 2.1: MH sampler on z-scores
            samples, acc_rate = metropolis_hastings_sampler(
                z_scores,
                n_samples=self.n_mcmc_samples,
                burn_in=self.burn_in,
                prior_mean=self.distress_threshold,
                prior_std=1.0,
            )
            # Per-stock: P(distress) = P(posterior_mean < stock_z)
            stock_z = (
                result_df["altman_z_score"].values
                if "altman_z_score" in result_df.columns
                else np.full(len(result_df), 3.0)
            )
            distress_prob_per_stock = np.mean(samples[:, None] > stock_z[None, :], axis=0)
            result_df["mcmc_distress_probability"] = np.clip(distress_prob_per_stock, 0, 1)

            # Task 2.2: Student-t for robust estimation
            mu_samples, df_samples = mcmc_student_t(
                z_scores, n_samples=self.n_mcmc_samples, burn_in=self.burn_in
            )
            result_df["mcmc_ci_lower"] = np.percentile(mu_samples, 2.5)
            result_df["mcmc_ci_upper"] = np.percentile(mu_samples, 97.5)
        except Exception as e:
            logger.warning("MCMC credit risk posterior failed: %s", e)
            result_df["mcmc_distress_probability"] = np.nan
            result_df["mcmc_ci_lower"] = np.nan
            result_df["mcmc_ci_upper"] = np.nan

        # Task 2.3: Hierarchical MCMC by sector
        try:
            sector_col = "industry" if "industry" in source_df.columns else "sector"
            if sector_col in source_df.columns and "altman_z_score" in source_df.columns:
                sector_results = hierarchical_mcmc_by_sector(
                    source_df,
                    feature="altman_z_score",
                    sector_col=sector_col,
                    n_samples=self.n_mcmc_samples,
                )
                sector_mean_map = {
                    s: v.get("posterior_mean", np.nan)
                    for s, v in sector_results.items()
                    if isinstance(v, dict)
                }
                if sector_col in result_df.columns:
                    result_df["sector_z_posterior_mean"] = result_df[sector_col].map(
                        sector_mean_map
                    )
        except Exception as e:
            logger.warning("Hierarchical MCMC for credit risk failed: %s", e)

        return result_df


class DividendCutProbabilityModel:
    """
    Identify high-yield stocks where distribution is likely to be reduced.

    Enhanced features: fcf_dividend_coverage, dividend_payout_ratio, dividend_streak,
    dividend_growth_expectation, sustainable_dividend_flag, dividend_consistency,
    dividend_yield_vs_5y_avg, recent_dividend_change

    Leverage & Liquidity enrichment (v3.4):
    interest_coverage, debt_to_equity, cash_ratio, working_capital_ratio,
    balance_sheet_strength, cash_runway_months, retained_earnings_growth, debt_3y_cagr
    """

    def __init__(
        self,
        high_payout_threshold: float = 0.9,
        min_coverage: float = 1.2,
        n_mcmc_samples: int = 8000,
        burn_in: int = 2000,
        use_mcmc: bool = True,
        # NEW: Leverage & Liquidity signals for dividend sustainability
        use_leverage_signals: bool = True,
        use_balance_sheet: bool = True,
    ):
        self.high_payout_threshold = high_payout_threshold
        self.min_coverage = min_coverage
        self.n_mcmc_samples = n_mcmc_samples
        self.burn_in = burn_in
        self.use_mcmc = use_mcmc
        self.use_leverage_signals = use_leverage_signals
        self.use_balance_sheet = use_balance_sheet

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            # Core dividend metrics
            fcf_coverage = row.get("fcf_dividend_coverage", 2.0)
            payout_ratio = row.get("dividend_payout_ratio", 50)
            streak = row.get("dividend_streak", 10)
            growth_exp = row.get("dividend_growth_expectation", 0)

            # NEW: Enhanced metrics from vw_features_dividends
            sustainable_flag = row.get("sustainable_dividend_flag", 1)
            consistency = row.get("dividend_consistency", 0.8)
            yield_vs_5y = row.get("dividend_yield_vs_5y_avg", 1.0)
            recent_change = row.get("recent_dividend_change", 0)
            high_yield_flag = row.get("high_yield_flag", 0)

            # NEW: Leverage signals (calc_leverage_features)
            interest_coverage = row.get("interest_coverage", 5.0)
            debt_to_equity = row.get("debt_to_equity", 0.5)
            cash_ratio_val = row.get("cash_ratio", 0.5)
            working_capital_ratio = row.get("working_capital_ratio", 1.0)

            # NEW: Balance sheet health (calc_balance_sheet_dynamics + distress)
            balance_sheet_strength = row.get("balance_sheet_strength", 50)
            cash_runway = row.get("cash_runway_months", 24)
            retained_earnings_growth = row.get("retained_earnings_growth", 0)
            debt_3y_cagr = row.get("debt_3y_cagr", 0)

            # Base probability with more granular assessment
            prob = 0.05  # Low base rate for established dividend payers

            # FCF coverage is the strongest predictor
            if fcf_coverage is not None and not pd.isna(fcf_coverage):
                if fcf_coverage < 0.5:
                    prob += 0.45
                elif fcf_coverage < 1.0:
                    prob += 0.30
                elif fcf_coverage < 1.2:
                    prob += 0.15
                elif fcf_coverage > 2.0:
                    prob -= 0.03

            # Payout ratio stress
            if payout_ratio is not None and not pd.isna(payout_ratio):
                if payout_ratio > 100:
                    prob += 0.25  # Paying from reserves
                elif payout_ratio > 90:
                    prob += 0.15
                elif payout_ratio > 75:
                    prob += 0.05

            # Streak provides historical reliability signal
            if streak is not None:
                if streak < 2:
                    prob += 0.10
                elif streak >= 10:
                    prob -= 0.05  # Dividend aristocrat effect
                elif streak >= 5:
                    prob -= 0.02

            # NEW: Sustainability flag from comprehensive calc
            if sustainable_flag == 0:
                prob += 0.12

            # NEW: Consistency score
            if consistency is not None and consistency < 0.5:
                prob += 0.10

            # NEW: Yield vs historical average (abnormally high = warning)
            if yield_vs_5y is not None and yield_vs_5y > 1.5:
                prob += 0.08  # Price has dropped, yield spiked

            # NEW: Recent dividend changes
            if recent_change is not None and recent_change < -10:
                prob += 0.15  # Already cutting

            # Negative growth expectation
            if growth_exp is not None and growth_exp < -5:
                prob += 0.12

            # NEW: Leverage stress signals
            if self.use_leverage_signals:
                if interest_coverage is not None and interest_coverage < 2.0:
                    prob += 0.12
                if debt_to_equity is not None and debt_to_equity > 3.0:
                    prob += 0.10
                if cash_ratio_val is not None and cash_ratio_val < 0.1:
                    prob += 0.08
                if working_capital_ratio is not None and working_capital_ratio < 0.5:
                    prob += 0.06

            # NEW: Balance sheet deterioration
            if self.use_balance_sheet:
                if balance_sheet_strength is not None and balance_sheet_strength < 25:
                    prob += 0.10
                if cash_runway is not None and cash_runway < 12:
                    prob += 0.08
                if retained_earnings_growth is not None and retained_earnings_growth < -15:
                    prob += 0.07
                if debt_3y_cagr is not None and debt_3y_cagr > 15:
                    prob += 0.06

            prob = min(0.95, max(0.03, prob))

            risk_cat = "Safe"
            if prob > 0.6:
                risk_cat = "At Risk"
            elif prob > 0.35:
                risk_cat = "Borderline"
            elif prob > 0.15:
                risk_cat = "Monitor"

            record = _extract_identifiers(row)
            record.update(
                {
                    "high_yield_flag": high_yield_flag,
                    "dividend_cut_probability": prob,
                    "fcf_dividend_coverage": fcf_coverage,
                    "payout_ratio": payout_ratio,
                    "dividend_streak": streak,
                    "dividend_consistency": consistency,
                    "yield_vs_5y_avg": yield_vs_5y,
                    "sustainable_flag": sustainable_flag,
                    "safety_score": 100 * (1 - prob),
                    "risk_category": risk_cat,
                }
            )
            results.append(record)

        result_df = pd.DataFrame(results)

        # MCMC enrichment path
        if self.use_mcmc and not result_df.empty:
            result_df = self._apply_mcmc_posteriors(result_df, df)

        return result_df

    def _apply_mcmc_posteriors(
        self, result_df: pd.DataFrame, source_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply MCMC posterior estimation for dividend cut probability."""
        from probabilistic_ml_model.statistical_functions.statistical_analysis import (
            metropolis_hastings_sampler,
            mcmc_student_t,
        )

        probs = []

        # Task 3.1: FCF coverage posterior
        fcf_prob = 0.5
        try:
            fcf_data = (
                source_df["fcf_dividend_coverage"].dropna().values
                if "fcf_dividend_coverage" in source_df.columns
                else np.array([])
            )
            if len(fcf_data) >= 10:
                samples, _ = metropolis_hastings_sampler(
                    fcf_data,
                    n_samples=self.n_mcmc_samples,
                    burn_in=self.burn_in,
                    prior_mean=self.min_coverage,
                    prior_std=1.0,
                )
                fcf_prob = float(np.mean(samples < self.min_coverage))
                probs.append(fcf_prob)
        except Exception as e:
            logger.warning("MCMC FCF coverage posterior failed: %s", e)

        # Task 3.2: Payout ratio posterior (Student-t)
        payout_prob = 0.5
        try:
            payout_data = (
                source_df["dividend_payout_ratio"].dropna().values
                if "dividend_payout_ratio" in source_df.columns
                else np.array([])
            )
            if len(payout_data) >= 10:
                mu_samples, _ = mcmc_student_t(
                    payout_data, n_samples=self.n_mcmc_samples, burn_in=self.burn_in
                )
                payout_prob = float(np.mean(mu_samples > self.high_payout_threshold * 100))
                probs.append(payout_prob)
        except Exception as e:
            logger.warning("MCMC payout ratio posterior failed: %s", e)

        # Task 3.4: Composite posterior (multiply individual posteriors)
        if probs:
            composite = 1.0
            for p in probs:
                composite *= max(p, 0.01)
            # Normalize to reasonable range
            composite = min(0.95, max(0.03, composite))
            result_df["mcmc_cut_probability"] = composite
            # CI from FCF samples if available
            try:
                if len(fcf_data) >= 10:
                    result_df["mcmc_ci_lower"] = np.percentile(samples, 2.5)
                    result_df["mcmc_ci_upper"] = np.percentile(samples, 97.5)
                else:
                    result_df["mcmc_ci_lower"] = np.nan
                    result_df["mcmc_ci_upper"] = np.nan
            except Exception:
                result_df["mcmc_ci_lower"] = np.nan
                result_df["mcmc_ci_upper"] = np.nan
        else:
            result_df["mcmc_cut_probability"] = np.nan
            result_df["mcmc_ci_lower"] = np.nan
            result_df["mcmc_ci_upper"] = np.nan

        return result_df


class PriceTargetAchievementModel:
    """
    Estimates probability of reaching consensus price target.

    Enhanced features: upside_potential, price_target_spread_pct, pt_momentum_1m,
    analyst_rating_normalized, pt_consensus_convergence, analyst_conviction,
    pt_acceleration_short, eps_revision_momentum, analyst_coverage_trend

    Risk-adjusted enrichment (v3.4):
    beta_1y, beta_stability_score, distress_risk_score,
    balance_sheet_strength, debt_maturity_risk
    """

    def __init__(
        self,
        time_horizon_months: int = 12,
        n_mcmc_samples: int = 10000,
        burn_in: int = 2000,
        use_mcmc: bool = True,
        # NEW: Risk-adjusted achievement
        use_risk_adjustment: bool = True,
        use_financial_health: bool = True,
    ):
        self.time_horizon_months = time_horizon_months
        self.n_mcmc_samples = n_mcmc_samples
        self.burn_in = burn_in
        self.use_mcmc = use_mcmc
        self.use_risk_adjustment = use_risk_adjustment
        self.use_financial_health = use_financial_health

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            # Core metrics
            upside = row.get("upside_potential", 10)
            spread = row.get("price_target_spread_pct", 20)
            pt_momentum = row.get("pt_momentum_1m", 0)
            rating = row.get("analyst_rating_normalized", 50)

            # NEW: Enhanced analyst sentiment features
            conviction = row.get("analyst_conviction", 50)
            consensus_convergence = row.get("pt_consensus_convergence", 0)
            pt_accel = row.get("pt_acceleration_short", 0)
            eps_revision = row.get("eps_revision_momentum", 0)
            coverage_trend = row.get("analyst_coverage_trend", 0)
            bullish_pct = row.get("analyst_bullish_pct", 50)

            # NEW: Risk adjustment (calc_beta_risk_features)
            beta_1y = row.get("beta_1y", 1.0)
            beta_stability = row.get("beta_stability_score", 50)
            distress_risk = row.get("distress_risk_score", 50)

            # NEW: Financial health (calc_balance_sheet_dynamics)
            bs_strength = row.get("balance_sheet_strength", 50)
            debt_mat_risk = row.get("debt_maturity_risk", 0)

            # Base probability - inversely related to upside magnitude
            if upside is None or pd.isna(upside):
                base_prob = 0.5
            elif upside <= 0:
                base_prob = 0.85  # Already at/above target
            elif upside < 10:
                base_prob = 0.70
            elif upside < 20:
                base_prob = 0.55
            elif upside < 30:
                base_prob = 0.40
            elif upside < 50:
                base_prob = 0.25
            else:
                base_prob = 0.15

            adjustments = 0.0

            # PT momentum signals conviction strengthening
            if pt_momentum is not None and pt_momentum > 0.05:
                adjustments += 0.08
            elif pt_momentum is not None and pt_momentum < -0.05:
                adjustments -= 0.10

            # Strong analyst consensus (low spread)
            if spread is not None and spread < 15:
                adjustments += 0.08
            elif spread is not None and spread > 40:
                adjustments -= 0.08

            # NEW: Analyst conviction score
            if conviction is not None and conviction > 70:
                adjustments += 0.07
            elif conviction is not None and conviction < 30:
                adjustments -= 0.05

            # NEW: Consensus converging (analysts agreeing)
            if consensus_convergence is not None and consensus_convergence > 0:
                adjustments += 0.05

            # NEW: PT acceleration (momentum building)
            if pt_accel is not None and pt_accel > 0.02:
                adjustments += 0.06

            # NEW: EPS revisions supporting the price target
            if eps_revision is not None and eps_revision > 5:
                adjustments += 0.08
            elif eps_revision is not None and eps_revision < -5:
                adjustments -= 0.10

            # NEW: Growing analyst coverage = more attention
            if coverage_trend is not None and coverage_trend > 0:
                adjustments += 0.03

            # NEW: Risk-adjusted achievement probability
            if self.use_risk_adjustment:
                if beta_1y is not None and beta_1y > 1.5:
                    adjustments -= 0.08
                elif beta_1y is not None and beta_1y < 0.7:
                    adjustments += 0.04
                if beta_stability is not None and beta_stability < 25:
                    adjustments -= 0.05
                if distress_risk is not None and distress_risk > 70:
                    adjustments -= 0.12

            # NEW: Financial health supports target achievement
            if self.use_financial_health:
                if bs_strength is not None and bs_strength > 75:
                    adjustments += 0.05
                elif bs_strength is not None and bs_strength < 25:
                    adjustments -= 0.08
                if debt_mat_risk is not None and debt_mat_risk > 70:
                    adjustments -= 0.06

            # Rating strength
            if rating is not None and rating > 75:
                adjustments += 0.05

            prob = min(0.90, max(0.05, base_prob + adjustments))

            record = _extract_identifiers(row)
            record.update(
                {
                    "bullish_pct": bullish_pct,
                    "achievement_probability": prob,
                    "upside_potential": upside,
                    "price_target_spread_pct": spread,
                    "analyst_conviction": conviction,
                    "eps_revision_momentum": eps_revision,
                    "analyst_rating_normalized": rating,
                    "expected_return_prob_weighted": (upside or 0) * prob,
                    "confidence_level": (
                        "High"
                        if spread and spread < 20
                        else "Medium" if spread and spread < 35 else "Low"
                    ),
                }
            )
            results.append(record)

        result_df = pd.DataFrame(results)

        # MCMC enrichment path
        if self.use_mcmc and not result_df.empty:
            result_df = self._apply_mcmc_posteriors(result_df, df)

        return result_df

    def _apply_mcmc_posteriors(
        self, result_df: pd.DataFrame, source_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply MCMC posterior estimation for price target achievement."""
        from probabilistic_ml_model.statistical_functions.statistical_analysis import (
            metropolis_hastings_sampler,
            mcmc_student_t,
            parallel_mcmc_chains,
        )

        returns_data = (
            source_df["upside_potential"].dropna().values
            if "upside_potential" in source_df.columns
            else np.array([])
        )
        if len(returns_data) < 10:
            return result_df

        try:
            # Task 4.2: Student-t for heavy-tailed returns
            mu_samples, df_samples = mcmc_student_t(
                returns_data, n_samples=self.n_mcmc_samples, burn_in=self.burn_in
            )
            achievement_prob = float(np.mean(mu_samples > 0))
            result_df["mcmc_achievement_probability"] = achievement_prob
            result_df["mcmc_ci_lower"] = np.percentile(mu_samples, 2.5)
            result_df["mcmc_ci_upper"] = np.percentile(mu_samples, 97.5)

            # Task 4.4: Posterior mean weighted return
            posterior_mean_return = float(mu_samples.mean())
            result_df["mcmc_expected_return_prob_weighted"] = (
                posterior_mean_return * achievement_prob
            )
        except Exception as e:
            logger.warning("MCMC price target posterior failed: %s", e)
            result_df["mcmc_achievement_probability"] = np.nan
            result_df["mcmc_ci_lower"] = np.nan
            result_df["mcmc_ci_upper"] = np.nan
            result_df["mcmc_expected_return_prob_weighted"] = np.nan

        # Task 4.3: Parallel MCMC with Gelman-Rubin
        try:
            mcmc_result = parallel_mcmc_chains(
                returns_data, n_chains=4, n_samples=self.n_mcmc_samples
            )
            result_df["mcmc_gelman_rubin"] = mcmc_result.get("r_hat", np.nan)
        except Exception as e:
            logger.warning("Parallel MCMC for price target failed: %s", e)
            result_df["mcmc_gelman_rubin"] = np.nan

        return result_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def create_earnings_probability_dashboard(
    probability_df: pd.DataFrame,
    title: str = "Earnings Beat Probability Analysis",
) -> go.Figure:
    """
    Create comprehensive dashboard for earnings beat probabilities.

    Supports both legacy output (from analyze_dataframe) and enhanced output
    (from analyze_dataframe_enhanced) with revision momentum and GAAP divergence
    columns. When enhanced columns are present, additional panels are shown.

    Args:
        probability_df: DataFrame from EarningsBeatProbabilityModel.analyze_dataframe
            or analyze_dataframe_enhanced
        title: Dashboard title

    Returns:
        Plotly Figure with probability analysis dashboard
    """
    # Detect enhanced columns
    has_momentum = "gaap_revision_momentum" in probability_df.columns
    has_spread = "gaap_norm_spread" in probability_df.columns
    has_streak = "quarterly_beat_streak" in probability_df.columns
    is_enhanced = has_momentum or has_spread

    if is_enhanced:
        n_rows = 3
        subplot_titles = (
            "Posterior Beat Probability Distribution",
            "Confidence Score by Sector",
            "Historical vs Posterior Beat Rate",
            "Probability Classification",
            "Revision Momentum vs P(Beat)" if has_momentum else "Beat Streak Distribution",
            "GAAP-Norm Spread vs P(Beat)" if has_spread else "Beat Streak Distribution",
        )
        specs = [
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ]
    else:
        n_rows = 2
        subplot_titles = (
            "Posterior Beat Probability Distribution",
            "Confidence Score by Sector",
            "Historical vs Posterior Beat Rate",
            "Probability Classification",
        )
        specs = [
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "pie"}],
        ]

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Color scheme matching Global Equity Research Dashboard theme
    colors = {
        "primary": "#0A7EA4",
        "secondary": "#00A878",
        "accent": "#6C63FF",
        "warning": "#FFD93D",
        "danger": "#E63946",
    }

    # 1. Posterior probability histogram
    fig.add_trace(
        go.Histogram(
            x=probability_df["posterior_beat_prob"],
            nbinsx=20,
            name="Posterior P(Beat)",
            marker_color=colors["primary"],
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    # Add vertical line at 0.5 threshold
    fig.add_vline(x=0.5, line_dash="dash", line_color=colors["danger"], row=1, col=1)

    # 2. Confidence by sector
    if "sector" in probability_df.columns:
        sector_conf = (
            probability_df.groupby("sector")["confidence_score"].mean().sort_values(ascending=True)
        )
        fig.add_trace(
            go.Bar(
                y=sector_conf.index,
                x=sector_conf.values,
                orientation="h",
                name="Avg Confidence",
                marker_color=colors["secondary"],
            ),
            row=1,
            col=2,
        )

    # 3. Historical vs Posterior scatter
    fig.add_trace(
        go.Scatter(
            x=probability_df["historical_beat_rate"],
            y=probability_df["posterior_beat_prob"],
            mode="markers",
            name="Stocks",
            marker=dict(
                size=8,
                color=probability_df["confidence_score"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Confidence", x=0.45),
            ),
            text=probability_df["ticker"],
            hovertemplate="<b>%{text}</b><br>Historical: %{x:.1%}<br>Posterior: %{y:.1%}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. Classification pie chart
    if "beat_classification" in probability_df.columns:
        classification_counts = probability_df["beat_classification"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=classification_counts.index,
                values=classification_counts.values,
                marker_colors=[colors["secondary"], colors["warning"]],
                hole=0.4,
            ),
            row=2,
            col=2,
        )

    # 5. Enhanced panel: Revision momentum vs posterior (row 3, col 1)
    if is_enhanced and has_momentum:
        plot_df = probability_df[["gaap_revision_momentum", "posterior_beat_prob"]].dropna()
        if len(plot_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["gaap_revision_momentum"],
                    y=plot_df["posterior_beat_prob"],
                    mode="markers",
                    name="Momentum vs P(Beat)",
                    marker=dict(
                        size=8,
                        color=colors["secondary"],
                        opacity=0.6,
                    ),
                    text=(
                        probability_df.loc[plot_df.index, "ticker"]
                        if "ticker" in probability_df.columns
                        else None
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Momentum: %{x:.0f}<br>"
                        "P(Beat): %{y:.1%}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )
    elif is_enhanced and has_streak:
        streak_data = probability_df["quarterly_beat_streak"].dropna()
        if len(streak_data) > 0:
            streak_counts = streak_data.value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=streak_counts.index.astype(str),
                    y=streak_counts.values,
                    name="Beat Streak",
                    marker_color=colors["accent"],
                ),
                row=3,
                col=1,
            )

    # 6. Enhanced panel: GAAP spread vs posterior (row 3, col 2)
    if is_enhanced and has_spread:
        plot_df = probability_df[["gaap_norm_spread", "posterior_beat_prob"]].dropna()
        if len(plot_df) > 0:
            abs_spread = plot_df["gaap_norm_spread"].abs()
            fig.add_trace(
                go.Scatter(
                    x=plot_df["gaap_norm_spread"],
                    y=plot_df["posterior_beat_prob"],
                    mode="markers",
                    name="GAAP Spread vs P(Beat)",
                    marker=dict(
                        size=8,
                        color=abs_spread,
                        colorscale="YlOrRd",
                        showscale=False,
                        opacity=0.7,
                    ),
                    text=(
                        probability_df.loc[plot_df.index, "ticker"]
                        if "ticker" in probability_df.columns
                        else None
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Spread: %{x:.1f}%<br>"
                        "P(Beat): %{y:.1%}<extra></extra>"
                    ),
                ),
                row=3,
                col=2,
            )
    elif is_enhanced and has_streak:
        streak_data = probability_df["quarterly_beat_streak"].dropna()
        if len(streak_data) > 0:
            streak_counts = streak_data.value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=streak_counts.index.astype(str),
                    y=streak_counts.values,
                    name="Beat Streak",
                    marker_color=colors["accent"],
                ),
                row=3,
                col=2,
            )

    # Update layout
    height = 1000 if is_enhanced else 700
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color="#1A2332")),
        height=height,
        showlegend=False,
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
    )

    # Update axes labels
    fig.update_xaxes(title_text="P(Beat)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Score", row=1, col=2)
    fig.update_xaxes(title_text="Historical Beat Rate", row=2, col=1)
    fig.update_yaxes(title_text="Posterior Beat Probability", row=2, col=1)

    if is_enhanced:
        if has_momentum:
            fig.update_xaxes(title_text="Revision Momentum (0-100)", row=3, col=1)
            fig.update_yaxes(title_text="Posterior P(Beat)", row=3, col=1)
        if has_spread:
            fig.update_xaxes(title_text="GAAP-Norm Spread %", row=3, col=2)
            fig.update_yaxes(title_text="Posterior P(Beat)", row=3, col=2)

    return fig


def create_confidence_calibration_chart(
    confidence_result: ModelConfidenceResult,
) -> go.Figure:
    """
    Create reliability diagram and confidence metrics chart.

    Args:
        confidence_result: ModelConfidenceResult from confidence estimator

    Returns:
        Plotly Figure with calibration analysis
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Reliability Diagram", "Model Confidence Metrics"),
        specs=[[{"type": "scatter"}, {"type": "indicator"}]],
    )

    reliability = confidence_result.reliability_diagram_data

    # Reliability diagram
    if reliability["bin_centers"]:
        fig.add_trace(
            go.Scatter(
                x=reliability["bin_centers"],
                y=reliability["observed_freq"],
                mode="markers+lines",
                name="Observed",
                marker=dict(size=10, color="#0A7EA4"),
            ),
            row=1,
            col=1,
        )

        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfect Calibration",
            ),
            row=1,
            col=1,
        )

    # Confidence gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence_result.overall_confidence,
            title={"text": "Model Confidence"},
            delta={"reference": 70},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0A7EA4"},
                "steps": [
                    {"range": [0, 40], "color": "#E63946"},
                    {"range": [40, 70], "color": "#FFD93D"},
                    {"range": [70, 100], "color": "#00A878"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Model Calibration: {confidence_result.model_name}",
        height=400,
        template="plotly_dark",
    )

    fig.update_xaxes(title_text="Predicted Probability", row=1, col=1)
    fig.update_yaxes(title_text="Observed Frequency", row=1, col=1)

    return fig


def create_eps_streak_analysis_chart(
    streak_df: pd.DataFrame,
    title: str = "EPS Streak Analysis & Predictions",
) -> go.Figure:
    """
    Create visualization for EPS streak analysis.

    Args:
        streak_df: DataFrame from EPSStreakAnalyzer.analyze_dataframe
        title: Chart title

    Returns:
        Plotly Figure with streak analysis
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Current Streak Distribution",
            "Continuation vs Reversion Probability",
            "Prediction Confidence by Streak Length",
            "Expected Outcomes",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "pie"}],
        ],
    )

    colors = {"beat": "#00A878", "miss": "#E63946", "meet": "#FFD93D"}

    # 1. Streak distribution
    fig.add_trace(
        go.Histogram(
            x=streak_df["current_streak"],
            nbinsx=15,
            name="Streak Length",
            marker_color="#0A7EA4",
        ),
        row=1,
        col=1,
    )

    # 2. Continuation vs Reversion
    fig.add_trace(
        go.Scatter(
            x=streak_df["continuation_probability"],
            y=streak_df["mean_reversion_probability"],
            mode="markers",
            marker=dict(
                size=8,
                color=[colors.get(t, "#0A7EA4") for t in streak_df["streak_type"]],
            ),
            text=streak_df["ticker"],
            hovertemplate="<b>%{text}</b><br>Continue: %{x:.1%}<br>Revert: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 3. Confidence by streak length
    streak_conf = streak_df.groupby("current_streak")["prediction_confidence"].mean()
    fig.add_trace(
        go.Scatter(
            x=streak_conf.index,
            y=streak_conf.values,
            mode="markers+lines",
            marker=dict(size=10, color="#6C63FF"),
            name="Avg Confidence",
        ),
        row=2,
        col=1,
    )

    # 4. Expected outcomes pie
    outcome_counts = streak_df["expected_next_outcome"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=outcome_counts.index,
            values=outcome_counts.values,
            marker_colors=[colors.get(o, "#0A7EA4") for o in outcome_counts.index],
            hole=0.4,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        height=700,
        showlegend=False,
        template="plotly_dark",
    )

    return fig


class CategoryProbabilityAnalyzer:
    """
    Probability analyzer for a specific feature category/view.

    Provides Bayesian estimation, confidence intervals, and
    probability distributions for all features in a category.
    """

    def __init__(
        self,
        category_name: str,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        n_mcmc_samples: int = 5000,
        burn_in: int = 1000,
        use_mcmc: bool = False,
        use_student_t: bool = False,
    ):
        self.category_name = category_name
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_mcmc_samples = n_mcmc_samples
        self.burn_in = burn_in
        self.use_mcmc = use_mcmc
        self.use_student_t = use_student_t

    def analyze_view(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """
        Analyze all features in a view and return probability metrics.

        Returns DataFrame with probability estimates per stock per feature.
        """
        results = []
        identifier_cols = load_identifier_columns()
        id_data = df[[c for c in identifier_cols if c in df.columns]].copy()

        # MCMC posterior stats per feature (computed once)
        mcmc_stats = {}
        if self.use_mcmc:
            mcmc_stats = self._compute_mcmc_posteriors(df, feature_cols)

        for feat in feature_cols:
            if feat not in df.columns:
                continue

            data = pd.to_numeric(df[feat], errors="coerce")
            if data.dropna().empty:
                continue

            # Calculate percentile rank as probability proxy
            percentile = data.rank(pct=True)

            # Bayesian credible intervals
            mean_val = data.mean()
            std_val = data.std()

            feat_results = id_data.copy()
            feat_results["feature"] = feat
            feat_results["value"] = data
            feat_results["percentile"] = percentile
            feat_results["z_score"] = (data - mean_val) / std_val if std_val > 0 else 0
            feat_results["prob_above_median"] = (percentile > 0.5).astype(float)

            # Add MCMC posterior columns if available
            if feat in mcmc_stats:
                stats = mcmc_stats[feat]
                feat_results["posterior_mean"] = stats["posterior_mean"]
                feat_results["posterior_std"] = stats["posterior_std"]
                feat_results["ci_lower_95"] = stats["ci_lower_95"]
                feat_results["ci_upper_95"] = stats["ci_upper_95"]

            results.append(feat_results)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    def _compute_mcmc_posteriors(self, df: pd.DataFrame, feature_cols: list[str]) -> dict:
        """Compute MCMC posteriors per feature."""
        from probabilistic_ml_model.statistical_functions.statistical_analysis import (
            metropolis_hastings_sampler,
            mcmc_student_t,
        )

        stats = {}
        for feat in feature_cols:
            if feat not in df.columns:
                continue
            data = pd.to_numeric(df[feat], errors="coerce").dropna().values
            if len(data) < 10:
                continue

            try:
                if self.use_student_t:
                    mu_samples, _ = mcmc_student_t(
                        data, n_samples=self.n_mcmc_samples, burn_in=self.burn_in
                    )
                else:
                    mu_samples, _ = metropolis_hastings_sampler(
                        data,
                        n_samples=self.n_mcmc_samples,
                        burn_in=self.burn_in,
                        prior_mean=self.prior_alpha,
                        prior_std=self.prior_beta,
                    )
                stats[feat] = {
                    "posterior_mean": float(mu_samples.mean()),
                    "posterior_std": float(mu_samples.std()),
                    "ci_lower_95": float(np.percentile(mu_samples, 2.5)),
                    "ci_upper_95": float(np.percentile(mu_samples, 97.5)),
                }
            except Exception as e:
                logger.warning("MCMC posterior for feature %s failed: %s", feat, e)

        return stats


# =============================================================================
# RESAMPLED BEAT PROBABILITY MODEL (ArviZ-enhanced)
# =============================================================================


@dataclass
class ResampledBeatEstimate:
    """Result container for resampled earnings beat probability with technical conditioning."""

    ticker: str
    name: str
    sector: str
    base_posterior_mean: float
    resampled_posterior_mean: float
    technical_adjustment: float
    momentum_signal: float
    volatility_regime_score: float
    credible_interval_90: tuple[float, float]
    credible_interval_95: tuple[float, float]
    prob_beat_given_momentum: float
    earnings_season_flag: Optional[int] = None
    pre_earnings_window: Optional[int] = None


class ResampledBeatProbabilityModel:
    """
    Extends EarningsBeatProbabilityModel with technical resampling priors.

    Conditions the Beta posterior on technical signals from
    ``vw_features_technical_analysis`` and ``vw_features_momentum``,
    then uses multi-timeframe resampled returns as informative priors.

    Parameters
    ----------
    base_model : EarningsBeatProbabilityModel
        Pre-configured base model for standard Bayesian beat probabilities.
    momentum_weight : float
        Weight of momentum signal in prior adjustment (0–1).
    volatility_weight : float
        Weight of volatility regime in prior adjustment (0–1).
    n_posterior_samples : int
        Number of posterior draws for ArviZ output.
    n_chains : int
        Number of MCMC chains.
    """

    _MOMENTUM_COLS = [
        "price_momentum_1m",
        "price_momentum_3m",
        "price_momentum_6m",
        "range_52w_position",
        "ema_crossover_20_50",
    ]
    _TECHNICAL_COLS = [
        "ema_slope_20d",
        "ema_trend_consistency",
        "breakout_signal",
        "volatility_compression",
    ]
    _TEMPORAL_COLS = [
        "earnings_season_flag",
        "pre_earnings_window",
        "days_to_earnings",
        "reporting_freshness_score",
    ]
    _EARNINGS_COLS = [
        "eps_trajectory_score",
        "eps_positive_streak",
        "revision_quality_divergence",
    ]

    def __init__(
        self,
        base_model: Optional["EarningsBeatProbabilityModel"] = None,
        momentum_weight: float = 0.3,
        volatility_weight: float = 0.2,
        n_posterior_samples: int = 4000,
        n_chains: int = 4,
        random_seed: int = 42,
    ):
        self.base_model = base_model or EarningsBeatProbabilityModel()
        self.momentum_weight = np.clip(momentum_weight, 0, 1)
        self.volatility_weight = np.clip(volatility_weight, 0, 1)
        self.n_posterior_samples = n_posterior_samples
        self.n_chains = n_chains
        self.rng = np.random.default_rng(random_seed)

    def _compute_momentum_signal(self, row: pd.Series) -> float:
        """Composite momentum signal from available features (normalised to [-1, 1])."""
        signals = []
        for col in self._MOMENTUM_COLS:
            if col in row.index and pd.notna(row[col]):
                signals.append(float(row[col]))
        if not signals:
            return 0.0
        raw = np.mean(signals)
        return float(np.clip(raw / 100.0, -1.0, 1.0))

    def _compute_volatility_regime(self, row: pd.Series) -> float:
        """Volatility regime score (0=high vol, 1=low/compressed vol)."""
        score = 0.5
        if "volatility_compression" in row.index and pd.notna(row["volatility_compression"]):
            score = float(np.clip(row["volatility_compression"], 0, 1))
        elif "volatility_term_structure" in row.index and pd.notna(
            row["volatility_term_structure"]
        ):
            score = float(np.clip(1.0 - abs(row["volatility_term_structure"]) / 100, 0, 1))
        return score

    def _adjust_prior(
        self,
        base_alpha: float,
        base_beta: float,
        momentum_signal: float,
        vol_regime: float,
    ) -> tuple[float, float]:
        """
        Adjust Beta prior parameters based on technical signals.

        Positive momentum + low volatility → shift prior toward higher beat rate.
        """
        adjustment = (
            self.momentum_weight * momentum_signal + self.volatility_weight * (vol_regime - 0.5) * 2
        )
        concentration = base_alpha + base_beta
        shift = adjustment * 0.2 * concentration

        adjusted_alpha = max(0.5, base_alpha + shift)
        adjusted_beta = max(0.5, base_beta - shift)
        return adjusted_alpha, adjusted_beta

    def _run_analysis(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        ticker_col: str = "ticker",
    ) -> list[ResampledBeatEstimate]:
        """Core analysis loop returning a list of ResampledBeatEstimate."""
        base_results = self.base_model.analyze_dataframe_enhanced(
            df, sector_col=sector_col, ticker_col=ticker_col
        )
        if base_results.empty:
            return []

        results: list[ResampledBeatEstimate] = []
        for _, row in base_results.iterrows():
            ticker = row.get(ticker_col, row.get("ticker", ""))

            orig_mask = (
                df[ticker_col] == ticker
                if ticker_col in df.columns
                else pd.Series(False, index=df.index)
            )
            orig_row = df.loc[orig_mask].iloc[0] if orig_mask.any() else pd.Series(dtype=float)

            momentum = self._compute_momentum_signal(orig_row)
            vol_regime = self._compute_volatility_regime(orig_row)

            base_alpha = row.get("posterior_alpha", 2.0)
            base_beta = row.get("posterior_beta", 2.0)
            base_mean = base_alpha / (base_alpha + base_beta)

            adj_alpha, adj_beta = self._adjust_prior(base_alpha, base_beta, momentum, vol_regime)
            adj_mean = adj_alpha / (adj_alpha + adj_beta)

            ci_90 = (
                float(stats.beta.ppf(0.05, adj_alpha, adj_beta)),
                float(stats.beta.ppf(0.95, adj_alpha, adj_beta)),
            )
            ci_95 = (
                float(stats.beta.ppf(0.025, adj_alpha, adj_beta)),
                float(stats.beta.ppf(0.975, adj_alpha, adj_beta)),
            )

            results.append(
                ResampledBeatEstimate(
                    ticker=str(ticker),
                    name=str(row.get("name", "")),
                    sector=str(row.get(sector_col, row.get("sector", ""))),
                    base_posterior_mean=float(base_mean),
                    resampled_posterior_mean=float(adj_mean),
                    technical_adjustment=float(adj_mean - base_mean),
                    momentum_signal=momentum,
                    volatility_regime_score=vol_regime,
                    credible_interval_90=ci_90,
                    credible_interval_95=ci_95,
                    prob_beat_given_momentum=float(1.0 - stats.beta.cdf(0.5, adj_alpha, adj_beta)),
                    earnings_season_flag=(
                        int(orig_row["earnings_season_flag"])
                        if "earnings_season_flag" in orig_row.index
                        and pd.notna(orig_row.get("earnings_season_flag"))
                        else None
                    ),
                    pre_earnings_window=(
                        int(orig_row["pre_earnings_window"])
                        if "pre_earnings_window" in orig_row.index
                        and pd.notna(orig_row.get("pre_earnings_window"))
                        else None
                    ),
                )
            )

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Run resampled beat probability analysis on equities DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Merged feature data (ideally from multiple vw_features_* views).
        sector_col : str
            Sector grouping column.
        ticker_col : str
            Ticker identifier column.

        Returns
        -------
        pd.DataFrame
            Enhanced beat probability results with technical conditioning.
        """
        estimates = self._run_analysis(df, sector_col=sector_col, ticker_col=ticker_col)
        if not estimates:
            return pd.DataFrame()

        result_df = pd.DataFrame([vars(r) for r in estimates])

        # Attach ESS and R-hat from InferenceData for quality control
        if ARVIZ_AVAILABLE and az is not None and not result_df.empty:
            try:
                idata = self.build_inference_data(df, sector_col=sector_col, ticker_col=ticker_col)
                if idata is not None:
                    summary = az.summary(idata)
                    if "ess_bulk" in summary.columns and len(summary) == len(result_df):
                        result_df["ess_bulk"] = summary["ess_bulk"].values
                    if "r_hat" in summary.columns and len(summary) == len(result_df):
                        result_df["r_hat"] = summary["r_hat"].values
            except Exception:
                pass

        return result_df

    def build_inference_data(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        ticker_col: str = "ticker",
    ) -> "az.InferenceData | xr.Dataset | None":
        """
        Build ArviZ InferenceData from resampled beat probability posteriors.

        Returns
        -------
        arviz.InferenceData, xr.Dataset, or None
        """
        estimates = self._run_analysis(df, sector_col=sector_col, ticker_col=ticker_col)
        if not estimates:
            return None
        result_df = pd.DataFrame([vars(r) for r in estimates])
        if result_df.empty:
            return None

        tickers = result_df["ticker"].values
        n_equities = len(tickers)

        base_results = self.base_model.analyze_dataframe_enhanced(
            df, sector_col=sector_col, ticker_col=ticker_col
        )

        adj_alphas = np.full(n_equities, 2.0)
        adj_betas = np.full(n_equities, 2.0)

        for i, ticker in enumerate(tickers):
            base_row = base_results.loc[base_results["ticker"] == ticker]
            if base_row.empty:
                continue
            base_a = float(base_row["posterior_alpha"].iloc[0])
            base_b = float(base_row["posterior_beta"].iloc[0])
            mom = float(result_df.iloc[i]["momentum_signal"])
            vol = float(result_df.iloc[i]["volatility_regime_score"])
            adj_alphas[i], adj_betas[i] = self._adjust_prior(base_a, base_b, mom, vol)

        posterior_samples = np.stack(
            [
                self.rng.beta(adj_alphas, adj_betas, size=(self.n_posterior_samples, n_equities))
                for _ in range(self.n_chains)
            ]
        )

        pp_samples = (self.rng.random(posterior_samples.shape) < posterior_samples).astype(int)

        coords = {
            "chain": np.arange(self.n_chains),
            "draw": np.arange(self.n_posterior_samples),
            "equity": tickers,
        }

        if ARVIZ_AVAILABLE and az is not None:
            return az.from_dict(
                posterior={"beat_probability": posterior_samples},
                posterior_predictive={"beat_outcome": pp_samples},
                observed_data={
                    "base_posterior_mean": result_df["base_posterior_mean"].values,
                    "momentum_signal": result_df["momentum_signal"].values,
                },
                constant_data={
                    "momentum_weight": np.array([self.momentum_weight]),
                    "volatility_weight": np.array([self.volatility_weight]),
                },
                coords=coords,
                dims={
                    "beat_probability": ["chain", "draw", "equity"],
                    "beat_outcome": ["chain", "draw", "equity"],
                },
            )
        elif xr is not None:
            return xr.Dataset(
                {"beat_probability": (["chain", "draw", "equity"], posterior_samples)},
                coords=coords,
            )
        return None


def create_view_probability_dashboard(
    view_df: pd.DataFrame,
    view_name: str,
    category_name: str,
) -> "go.Figure":
    """
    Create interactive probability dashboard for a feature view.

    Parameters
    ----------
    view_df : pd.DataFrame
        DataFrame from a specific vw_features view
    view_name : str
        Name of the view (e.g., "vw_features_momentum")
    category_name : str
        Display name for the category

    Returns
    -------
    go.Figure
        Plotly figure with probability distributions
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from probabilistic_ml_model.data_utils import get_identifier_cols_set

    identifier_cols = get_identifier_cols_set()

    feature_cols = [c for c in view_df.columns if c not in identifier_cols][:6]  # Top 6

    n_features = len(feature_cols)
    if n_features == 0:
        fig = go.Figure()
        fig.add_annotation(text="No features available", x=0.5, y=0.5)
        return fig

    rows = (n_features + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=[f"{feat}" for feat in feature_cols])

    for idx, feat in enumerate(feature_cols):
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        data = pd.to_numeric(view_df[feat], errors="coerce").dropna()
        if len(data) > 10:
            fig.add_trace(
                go.Histogram(x=data, name=feat, showlegend=False, nbinsx=30),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=f"{category_name} - Probability Distributions",
        height=300 * rows,
        showlegend=False,
    )

    return fig


def export_probability_analytics_results(
    probability_df: pd.DataFrame,
    streak_df: pd.DataFrame,
    output_dir: Path,
    confidence_result: Optional[ModelConfidenceResult] = None,
    credit_risk_df: Optional[pd.DataFrame] = None,
    dividend_safety_df: Optional[pd.DataFrame] = None,
    price_target_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Export all probability analytics results to database and files.

    Uses standardized identifier columns from vw_identifier_columns to
    build each output table with consistent identifier ordering.

    Args:
        probability_df: DataFrame with probability analysis
        streak_df: DataFrame with streak analysis
        output_dir: Output directory path
        confidence_result: Optional confidence metrics
        credit_risk_df: Optional credit risk analysis results
        dividend_safety_df: Optional dividend safety analysis results
        price_target_df: Optional price target achievement results

    Returns:
        Dictionary with export information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exports = {}

    def _safe_export(df: pd.DataFrame, table_name: str, reorder: bool = True) -> None:
        """Export a DataFrame via ExportConfig pipeline with error handling."""
        try:
            ordered = reorder_with_identifiers(df) if reorder else df
            cfg = ExportConfig(
                table_name=table_name,
                output_dir=str(output_dir),
            )
            export_to_db(ordered, cfg)
            export_to_csv(ordered, cfg)
            export_to_json(ordered, cfg)
            exports[f"{table_name}_db"] = f"analytics.{table_name}"
            exports[f"{table_name}_csv"] = str(output_dir / f"{table_name}.csv")
        except Exception as e:
            logger.error("Failed to export %s: %s", table_name, e)

    # Issue 7: Cast mixed-type columns to proper numeric dtypes before export
    for col in _NUMERIC_CAST_COLS:
        if col in probability_df.columns:
            probability_df[col] = pd.to_numeric(probability_df[col], errors="coerce")
    for col in _INTEGER_CAST_COLS:
        if col in probability_df.columns:
            probability_df[col] = pd.to_numeric(probability_df[col], errors="coerce").astype(
                "Int64"
            )

    # Also cast streak_df columns to proper numeric dtypes
    for col in _NUMERIC_CAST_COLS:
        if col in streak_df.columns:
            streak_df[col] = pd.to_numeric(streak_df[col], errors="coerce")
    for col in _INTEGER_CAST_COLS:
        if col in streak_df.columns:
            streak_df[col] = pd.to_numeric(streak_df[col], errors="coerce").astype("Int64")

    # 1. Export probability analysis (Issue 3: table_name is canonical for both DB and CSV)
    _safe_export(probability_df, "earnings_probability_analysis")

    # 2. Export streak analysis
    _safe_export(streak_df, "eps_streak_analysis")

    # 3. Export confidence metrics
    if confidence_result:
        conf_df = pd.DataFrame(
            [
                {
                    "model_name": confidence_result.model_name,
                    "brier_score": confidence_result.brier_score,
                    "log_loss": confidence_result.log_loss,
                    "calibration_error": confidence_result.calibration_error,
                    "discrimination_auc": confidence_result.discrimination_auc,
                    "overall_confidence": confidence_result.overall_confidence,
                }
            ]
        )
        _safe_export(conf_df, "model_confidence_metrics", reorder=False)

    # 4. Create and export summary statistics (Issue 6: validate columns first)
    required_prob_cols = {"posterior_beat_prob", "beat_classification", "confidence_score"}
    required_streak_cols = {"current_streak", "streak_type"}
    missing_prob = required_prob_cols - set(probability_df.columns)
    missing_streak = required_streak_cols - set(streak_df.columns)

    if missing_prob or missing_streak:
        if streak_df.empty:
            logger.warning(
                "Summary skipped — streak_df is empty (no valid EPS trajectory data). "
                "Ensure the input data contains 'eps_trajectory_score' values."
            )
        else:
            logger.warning(
                "Summary skipped — missing columns: prob=%s, streak=%s",
                missing_prob or "none",
                missing_streak or "none",
            )
    else:
        try:
            summary_data = {
                "metric": [
                    "Total Stocks Analyzed",
                    "Mean Posterior Beat Probability",
                    "Median Posterior Beat Probability",
                    "Stocks Classified as Likely Beat",
                    "Mean Confidence Score",
                    "Mean Streak Length",
                    "Stocks with Beat Streak",
                    "Stocks with Miss Streak",
                ],
                "value": [
                    float(len(probability_df)),
                    float(probability_df["posterior_beat_prob"].mean()),
                    float(probability_df["posterior_beat_prob"].median()),
                    float((probability_df["beat_classification"] == "likely_beat").sum()),
                    float(probability_df["confidence_score"].mean()),
                    float(streak_df["current_streak"].abs().mean()),
                    float((streak_df["streak_type"] == "beat").sum()),
                    float((streak_df["streak_type"] == "miss").sum()),
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            _safe_export(summary_df, "probability_analytics_summary", reorder=False)
        except Exception as e:
            logger.error("Failed to compute/export summary statistics: %s", e)

    # 5. Export credit risk results
    if credit_risk_df is not None and len(credit_risk_df) > 0:
        _safe_export(credit_risk_df, "credit_risk_analysis")

    # 6. Export dividend safety results
    if dividend_safety_df is not None and len(dividend_safety_df) > 0:
        _safe_export(dividend_safety_df, "dividend_safety_analysis")

    # 7. Export price target achievement results
    if price_target_df is not None and len(price_target_df) > 0:
        _safe_export(price_target_df, "price_target_achievement")

    logger.info("Exported probability analytics results to database and %s", output_dir)
    return exports
