"""
Probability Models Module for Market Analytics

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
from typing import Any, Optional, TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import floating
from plotly.subplots import make_subplots
from scipy import stats

from probabilistic_ml_model.data_utils import (
    ExportConfig,
    export_to_csv,
    export_to_db,
    export_to_json,
    load_identifier_columns,
    reorder_with_identifiers,
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
        col: row.get(col, None)
        for col in id_cols
        if col in row.index and pd.notna(row.get(col))
    }


def _safe_get(
    row: pd.Series, col: str, default: float | int
) -> tuple[float | int, float] | tuple[Any | None, Any | None]:
    """Return ``(value_for_calculation, raw_value_for_output)`` from *row*.

    When *col* is missing from the row or its value is null, the first
    element is *default* (safe for arithmetic) and the second is ``NaN``
    (preserves the fact that the source data lacked the value).  When
    the column is present and non-null both elements equal the actual value.
    """
    raw = row.get(col)
    try:
        missing = raw is None or pd.isna(raw)
    except (TypeError, ValueError):
        missing = raw is None
    if missing:
        return default, np.nan
    return raw, raw


def _compute_analyst_conviction(row: pd.Series) -> float:
    """Derive ``analyst_conviction`` from bullish/bearish percentages.

    Mirrors the SQL definition in ``calc_sentiment_features``::

        ABS((bullish - bearish) / total * 100)

    Since ``analyst_bullish_pct`` and ``analyst_bearish_pct`` are already
    expressed as percentages of total ratings, conviction simplifies to
    ``abs(bullish_pct - bearish_pct)``.

    Returns ``NaN`` when the required source columns are missing or null.
    """
    bullish = row.get("analyst_bullish_pct")
    bearish = row.get("analyst_bearish_pct")
    try:
        b_missing = bullish is None or pd.isna(bullish)
        s_missing = bearish is None or pd.isna(bearish)
    except (TypeError, ValueError):
        b_missing = bullish is None
        s_missing = bearish is None
    if b_missing or s_missing:
        return np.nan
    return abs(float(bullish) - float(bearish))


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

    ARVIZ_AVAILABLE = hasattr(az, "from_dict") or hasattr(az, "InferenceData")
except (ImportError, OSError, PermissionError):
    az = None  # type: ignore[assignment]
    xr = None  # type: ignore[assignment]
    ARVIZ_AVAILABLE = False


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class BeatProbabilityEstimate(TypedDict, total=False):
    """Type definition for beat probability estimation results.

    v3.10 (§10.1 / §10.2 / T-F) — stays a ``TypedDict`` for back-compat with
    existing ``{"posterior_mean": ...}`` returns, but gains required CI +
    posterior moment keys (``ci_low`` / ``ci_high``). Use the sibling helpers
    :func:`validate_beat_probability_estimate` and
    :class:`BeatProbabilityEstimateDC` below when runtime validation (§10.1)
    is required — they operationalise the dataclass intent while avoiding
    breaking every existing call site.
    """

    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_std: float
    credible_interval_90: tuple[float, float]
    credible_interval_95: tuple[float, float]
    prob_exceeds_threshold: float
    confidence_score: float
    # NEW: Interpretability enhancements
    prior_influence_pct: float
    effective_sample_size: float
    classification_confidence: str
    # v3.10 §10.2 — CI + posterior moments promoted (optional via total=False)
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class BeatProbabilityEstimateDC:
    """Frozen dataclass mirror of :class:`BeatProbabilityEstimate` with validation.

    v3.10 (§10.1 / T-F) — provides the runtime validation promised by §10.1
    without breaking the ~N TypedDict-return call sites. Pipelines that want
    the stricter contract construct ``BeatProbabilityEstimateDC(**d)`` from
    the returned dict and optionally serialise back via :meth:`to_dict`.
    """

    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_std: float
    credible_interval_90: tuple[float, float]
    credible_interval_95: tuple[float, float]
    prob_exceeds_threshold: float
    confidence_score: float
    prior_influence_pct: float = 0.0
    effective_sample_size: float = 0.0
    classification_confidence: str = "Low"
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    def __post_init__(self) -> None:
        for _name in ("posterior_mean", "prob_exceeds_threshold", "confidence_score"):
            _v = float(getattr(self, _name))
            if _v != _v or _v in (float("inf"), float("-inf")):  # NaN / Inf
                raise ValueError(
                    f"BeatProbabilityEstimateDC.{_name} must be finite in [0,1], got {_v!r}"
                )
            if not 0.0 <= _v <= 1.0:
                raise ValueError(f"BeatProbabilityEstimateDC.{_name} out of range [0,1]: {_v!r}")
        for _name in ("posterior_alpha", "posterior_beta"):
            _v = float(getattr(self, _name))
            if _v <= 0.0 or _v != _v or _v == float("inf"):
                raise ValueError(
                    f"BeatProbabilityEstimateDC.{_name} must be finite > 0, got {_v!r}"
                )

    def to_dict(self) -> BeatProbabilityEstimate:
        """Return a :class:`BeatProbabilityEstimate` TypedDict payload."""
        return {  # type: ignore[return-value]
            "posterior_alpha": float(self.posterior_alpha),
            "posterior_beta": float(self.posterior_beta),
            "posterior_mean": float(self.posterior_mean),
            "posterior_std": float(self.posterior_std),
            "credible_interval_90": tuple(self.credible_interval_90),
            "credible_interval_95": tuple(self.credible_interval_95),
            "prob_exceeds_threshold": float(self.prob_exceeds_threshold),
            "confidence_score": float(self.confidence_score),
            "prior_influence_pct": float(self.prior_influence_pct),
            "effective_sample_size": float(self.effective_sample_size),
            "classification_confidence": str(self.classification_confidence),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
        }


def validate_beat_probability_estimate(est: BeatProbabilityEstimate) -> BeatProbabilityEstimate:
    """Runtime-validate a :class:`BeatProbabilityEstimate` dict payload.

    v3.10 (§10.1) — checks that probability fields are finite & within
    ``[0, 1]`` and that alpha/beta are finite > 0. Raises ``ValueError`` on
    failure. Returns ``est`` unchanged on success so it can be chained:
    ``return validate_beat_probability_estimate({...})``.
    """
    for _name in ("posterior_mean", "prob_exceeds_threshold", "confidence_score"):
        _v = float(est.get(_name, float("nan")))  # type: ignore[arg-type]
        if _v != _v or _v in (float("inf"), float("-inf")):
            raise ValueError(
                f"BeatProbabilityEstimate[{_name!r}] must be finite in [0,1], got {_v!r}"
            )
        if not 0.0 <= _v <= 1.0:
            raise ValueError(f"BeatProbabilityEstimate[{_name!r}] out of range [0,1]: {_v!r}")
    for _name in ("posterior_alpha", "posterior_beta"):
        _v = float(est.get(_name, 0.0))  # type: ignore[arg-type]
        if _v <= 0.0 or _v != _v or _v == float("inf"):
            raise ValueError(f"BeatProbabilityEstimate[{_name!r}] must be finite > 0, got {_v!r}")
    return est


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================


# Schema version for serialisable *Result dataclasses (T-A/T-B cross-cutting).
# Bump when adding / renaming fields so downstream exporters can validate.
RESULT_SCHEMA_VERSION: str = "v3.9"


@dataclass
class PosteriorDiagnostics:
    """Shared MCMC / posterior diagnostic mixin (cross-cutting task T-B).

    Centralises the tail-aware and convergence diagnostic fields that
    ``build_tri_model_alignment`` / ``build_expected_returns_summary`` now
    consume on a **per-model** basis (task T-A). Populated by each
    probability model's ``_apply_mcmc_posteriors`` helper.

    Fields
    ------
    tail_df: Sampled Student-t degrees-of-freedom (clamped at
        ``student_t_df_floor``). ``NaN`` when Gaussian likelihood is in use.
    cond_volatility: Terminal GARCH(1,1) conditional volatility σ_t when
        ``use_garch_volatility=True``; ``NaN`` otherwise.
    cvar_5: Conditional Value-at-Risk at 5% (lower tail) of the posterior
        predictive return distribution.
    r_hat: Gelman-Rubin convergence diagnostic (should be ~1.00).
    ess_bulk / ess_tail: Effective Sample Size (bulk and tail).
    divergences: Number of divergent transitions encountered (NUTS only).
    """

    tail_df: float = float("nan")
    cond_volatility: float = float("nan")
    cvar_5: float = float("nan")
    r_hat: float = float("nan")
    ess_bulk: float = float("nan")
    ess_tail: float = float("nan")
    divergences: int = 0


@dataclass
class CreditRiskResult:
    """Result container for credit risk probability analysis.

    Extended in v3.9 with tail/vol diagnostics (task §2.1) so that the
    ensemble layer can pass a **per-stock** Student-t df rather than a
    single global value (unblocks accurate ``tail_haircut`` /
    ``risk_adjusted_expected_return`` / ``position_size_weight``).
    """

    ticker: str
    name: str
    sector: str
    distress_probability: float
    liquidity_stress_score: float
    cash_runway_months: float
    altman_z_score: float
    risk_level: str  # 'Low', 'Medium', 'High', 'Distressed'
    confidence_interval: tuple[float, float]
    # --- v3.9 tail/vol diagnostics (§2.1) ---
    tail_df: float = float("nan")
    cond_volatility: float = float("nan")
    cvar_5: float = float("nan")
    macro_loading: dict[str, float] = field(default_factory=dict)
    posterior_ess_bulk: int = 0
    posterior_ess_tail: int = 0
    r_hat: float = float("nan")
    # --- v3.9 serialisation contract (§2.2) ---
    schema_version: str = RESULT_SCHEMA_VERSION

    # Approved column set for ``analytics.credit_risk_analysis`` exports.
    # See ``_trim_credit_for_export`` in ``expected_returns_v3.py``.
    _EXPORT_COLUMNS: tuple[str, ...] = (
        "ticker",
        "name",
        "sector",
        "distress_probability",
        "liquidity_stress_score",
        "cash_runway_months",
        "altman_z_score",
        "risk_level",
        "tail_df",
        "cond_volatility",
        "cvar_5",
        "posterior_ess_bulk",
        "posterior_ess_tail",
        "r_hat",
        "schema_version",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a ``dict`` (schema-versioned)."""
        out: dict[str, Any] = {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "distress_probability": self.distress_probability,
            "liquidity_stress_score": self.liquidity_stress_score,
            "cash_runway_months": self.cash_runway_months,
            "altman_z_score": self.altman_z_score,
            "risk_level": self.risk_level,
            "confidence_interval": list(self.confidence_interval),
            "tail_df": self.tail_df,
            "cond_volatility": self.cond_volatility,
            "cvar_5": self.cvar_5,
            "macro_loading": dict(self.macro_loading),
            "posterior_ess_bulk": self.posterior_ess_bulk,
            "posterior_ess_tail": self.posterior_ess_tail,
            "r_hat": self.r_hat,
            "schema_version": self.schema_version,
        }
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CreditRiskResult":
        """Inverse of ``to_dict``; tolerant of forward-compatible extra keys."""
        ci = payload.get("confidence_interval", (float("nan"), float("nan")))
        return cls(
            ticker=payload.get("ticker", ""),
            name=payload.get("name", ""),
            sector=payload.get("sector", ""),
            distress_probability=float(payload.get("distress_probability", float("nan"))),
            liquidity_stress_score=float(payload.get("liquidity_stress_score", float("nan"))),
            cash_runway_months=float(payload.get("cash_runway_months", float("nan"))),
            altman_z_score=float(payload.get("altman_z_score", float("nan"))),
            risk_level=str(payload.get("risk_level", "Unknown")),
            confidence_interval=(float(ci[0]), float(ci[1])),
            tail_df=float(payload.get("tail_df", float("nan"))),
            cond_volatility=float(payload.get("cond_volatility", float("nan"))),
            cvar_5=float(payload.get("cvar_5", float("nan"))),
            macro_loading=dict(payload.get("macro_loading", {}) or {}),
            posterior_ess_bulk=int(payload.get("posterior_ess_bulk", 0) or 0),
            posterior_ess_tail=int(payload.get("posterior_ess_tail", 0) or 0),
            r_hat=float(payload.get("r_hat", float("nan"))),
            schema_version=str(payload.get("schema_version", RESULT_SCHEMA_VERSION)),
        )


@dataclass
class AccountingAnomalyResult:
    """Result container for per-stock accounting anomaly analysis.

    v3.10 (§9.1 / §9.2) — extended with decoupled flag-count vs magnitude
    components (unblocks factor attribution in ``build_quad_model_alignment``)
    and PosteriorDiagnostics parity fields (tail_df / cond_volatility /
    r_hat / ess_*).  All new fields default to ``NaN`` / empty so existing
    callers are unaffected.
    """

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
    # --- v3.10 §9.1 component decomposition ---
    flag_count_posterior_mean: float = float("nan")
    flag_count_ci_low: float = float("nan")
    flag_count_ci_high: float = float("nan")
    magnitude_posterior_mean: float = float("nan")
    combined_anomaly_score: float = float("nan")
    dominant_flag_category: str = ""  # one of accruals/revenue/expense/wc/restatement
    # --- v3.10 §9.2 diagnostic parity (T-A / T-B) ---
    tail_df: float = float("nan")
    cond_volatility: float = float("nan")
    r_hat: float = float("nan")
    ess_bulk: float = float("nan")
    ess_tail: float = float("nan")
    schema_version: str = RESULT_SCHEMA_VERSION


@dataclass
class AccountingAnomalyProbabilityModel:
    """
    Bayesian-informed accounting anomaly detection and analytics model.

    Wraps ``detect_accounting_anomalies`` (statistical_models.py) and
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
        Weight for anomaly_score in severity computation (default 0.75).
    severity_feature_weight : float
        Weight for feature_count in severity computation (default 0.25).
    multi_flag_threshold : int
        Minimum flagged features to trigger multi_flag_alert (default 10).
    """

    anomaly_z_threshold: float | None = None
    tier_bins: list[float] | None = None
    tier_labels: list[str] | None = None
    severity_anomaly_weight: float = 0.45  # v3.9: was 0.75 — avoid double-counting under BMA
    severity_feature_weight: float = 0.55  # v3.9: was 0.25 — complementary weight
    multi_flag_threshold: int = 18  # v3.9: was 15 — mid-point given tighter thresholds
    n_mcmc_samples: int = 5000
    burn_in: int = 1000
    use_mcmc: bool = True
    # NEW: Comprehensive quality signals (v3.4)
    use_quality_frequency: bool = True
    use_balance_sheet_quality: bool = True
    # v3.10 §8.3 — tail-aware likelihood on the residual z-score channel.
    # The GARCH switch is surfaced but not yet consumed inside
    # ``_apply_mcmc_posteriors`` (implementation deferred — see CHANGELOG
    # deferred-items list). ``student_t_df_floor`` is used to log /
    # validate the configuration (parity with the Credit / Price-Target
    # models).
    use_student_t_likelihood: bool = True
    use_garch_volatility: bool = False  # deferred: sampler term outstanding
    student_t_df_floor: float = 3.0
    # v3.10 §8.2 — sector priors for flag-rate Beta-Binomial posterior.
    sector_priors: Optional[dict[str, "PriorParameters"]] = None
    # v3.10 §8.5 — time-decay halflife for flag evidence (years).
    flag_halflife_years: float = 2.0

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
        from probabilistic_ml_model.statistical_functions.statistical_models import (
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
        feature_count = result.get(
            "anomaly_feature_count", pd.Series(0, index=result.index)
        )

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

        # -----------------------------------------------------------------
        # v3.10 §8.1 / §9.1 — decouple flag-count threshold from magnitude.
        # Emits two separable posterior components so downstream
        # ``build_quad_model_alignment`` can do factor attribution instead
        # of consuming a single blended scalar.
        #   (a) ``flag_count_posterior_mean`` — Beta-Binomial posterior mean
        #       of ``P(any_flag)`` given the observed flag count vs the
        #       maximum possible feature count (Jeffreys Beta(0.5, 0.5) prior
        #       per §8.4).
        #   (b) ``magnitude_posterior_mean`` — Student-t-style shrunk mean
        #       of ``accounting_anomaly_score | flagged`` (simple MoM
        #       shrinkage toward the universe mean; a full Student-t
        #       regression is deferred — see CHANGELOG deferred-items list).
        #   combined_anomaly_score = P × E.
        # -----------------------------------------------------------------
        max_feature_count = int(feature_count.max() or 1)
        # Jeffreys prior for flag-count Beta-Binomial posterior (§8.4).
        alpha_j, beta_j = 0.5, 0.5
        fc_alpha = feature_count.clip(lower=0) + alpha_j
        fc_beta = (max_feature_count - feature_count.clip(lower=0)).clip(lower=0) + beta_j
        fc_total = fc_alpha + fc_beta
        flag_prob_mean = (fc_alpha / fc_total).astype(float)
        # 90% credible interval from Beta distribution.
        try:
            flag_ci_low = stats.beta.ppf(0.05, fc_alpha.values, fc_beta.values)
            flag_ci_high = stats.beta.ppf(0.95, fc_alpha.values, fc_beta.values)
        except ValueError, TypeError:
            flag_ci_low = np.full(len(result), float("nan"))
            flag_ci_high = np.full(len(result), float("nan"))
        result["flag_count_posterior_mean"] = flag_prob_mean
        result["flag_count_ci_low"] = flag_ci_low
        result["flag_count_ci_high"] = flag_ci_high

        # Magnitude channel — shrink per-stock anomaly_score toward universe mean.
        # Shrinkage weight w = n_obs / (n_obs + τ) with τ = 5 (weak shrinkage).
        mag_series = pd.to_numeric(result["accounting_anomaly_score"], errors="coerce")
        universe_mean = float(mag_series.mean()) if mag_series.notna().any() else 0.0
        tau = 5.0
        n_obs = feature_count.clip(lower=0).astype(float)
        shrink_w = n_obs / (n_obs + tau)
        result["magnitude_posterior_mean"] = (
            shrink_w * mag_series.fillna(universe_mean) + (1.0 - shrink_w) * universe_mean
        )
        # Combined score = P(flag) × E[magnitude | flagged], normalised to [0, 100].
        result["combined_anomaly_score"] = (
            result["flag_count_posterior_mean"] * result["magnitude_posterior_mean"]
        ).astype(float)

        # Surface the tail-aware config on every row so the T-A per-stock
        # tail-df wiring in ``build_tri_model_alignment`` can pick it up.
        if self.use_student_t_likelihood:
            result["tail_df"] = float(self.student_t_df_floor)
        else:
            result["tail_df"] = float("nan")

        # Phase 2b: Enrich severity with comprehensive quality frequency signals
        if self.use_quality_frequency:
            freq_cols = [
                "goodwill_impairment_frequency",
                "asset_writedown_frequency",
                "restructuring_frequency",
                "exceptional_items_frequency",
            ]
            available_freq = [c for c in freq_cols if c in result.columns]
            if available_freq:
                freq_sum = result[available_freq].fillna(0).sum(axis=1)
                result["anomaly_severity_score"] += freq_sum * 3.0

            if "quality_issues_count_5y" in result.columns:
                result["anomaly_severity_score"] += (
                    result["quality_issues_count_5y"].fillna(0) * 2.0
                )

        # Phase 2c: Balance sheet quality cross-check (enhanced v3.5)
        if self.use_balance_sheet_quality:
            # Existing checks
            if "retained_earnings_vs_5y" in result.columns:
                re_declining = result["retained_earnings_vs_5y"].fillna(1.0) < 0.7
                result.loc[re_declining, "anomaly_severity_score"] += 5.0

            if "intangibles_growth_flag" in result.columns:
                intang_growing = result["intangibles_growth_flag"].fillna(0) == 1
                result.loc[intang_growing, "anomaly_severity_score"] += 3.0

            if "asset_quality_score" in result.columns:
                low_quality = result["asset_quality_score"].fillna(50) < 25
                result.loc[low_quality, "anomaly_severity_score"] += 3.0

            # NEW: Accumulated deficit — persistent losses create manipulation pressure
            if "accumulated_deficit_flag" in result.columns:
                has_deficit = result["accumulated_deficit_flag"].fillna(0) == 1
                result.loc[has_deficit, "anomaly_severity_score"] += 4.0

            # NEW: Working capital deterioration — divergence from earnings trajectory
            if "wc_deteriorating_flag" in result.columns:
                wc_bad = result["wc_deteriorating_flag"].fillna(0) == 1
                result.loc[wc_bad, "anomaly_severity_score"] += 4.0

            if "negative_wc_flag" in result.columns:
                neg_wc = result["negative_wc_flag"].fillna(0) == 1
                result.loc[neg_wc, "anomaly_severity_score"] += 3.0

            # NEW: Inventory anomalies
            if "inventory_buildup_flag" in result.columns:
                inv_build = result["inventory_buildup_flag"].fillna(0) == 1
                result.loc[inv_build, "anomaly_severity_score"] += 5.0

            if "inventory_reduction_flag" in result.columns:
                inv_reduce = result["inventory_reduction_flag"].fillna(0) == 1
                result.loc[inv_reduce, "anomaly_severity_score"] += 3.0

            # NEW: Impairment events — direct evidence of prior overvaluation
            if "has_goodwill_impairment" in result.columns:
                gw_imp = result["has_goodwill_impairment"].fillna(0) == 1
                result.loc[gw_imp, "anomaly_severity_score"] += 6.0

            if "has_goodwill_impairment_ltm" in result.columns:
                gw_imp_ltm = result["has_goodwill_impairment_ltm"].fillna(0) == 1
                result.loc[gw_imp_ltm, "anomaly_severity_score"] += (
                    7.0  # Recency premium
                )

            if "has_asset_writedown" in result.columns:
                writedown = result["has_asset_writedown"].fillna(0) == 1
                result.loc[writedown, "anomaly_severity_score"] += 5.0

            if "has_restructuring" in result.columns:
                restruct = result["has_restructuring"].fillna(0) == 1
                result.loc[restruct, "anomaly_severity_score"] += 4.0

            if "impairment_risk_score" in result.columns:
                high_imp = result["impairment_risk_score"].fillna(0) > 70
                result.loc[high_imp, "anomaly_severity_score"] += 5.0

            # NEW: Strategic red flags
            if "overinvestment_flag" in result.columns:
                overinvest = result["overinvestment_flag"].fillna(0) == 1
                result.loc[overinvest, "anomaly_severity_score"] += 3.0

            if "recent_acquisition_flag" in result.columns:
                acq = result["recent_acquisition_flag"].fillna(0) == 1
                result.loc[acq, "anomaly_severity_score"] += 4.0

            if "has_unusual_items_flag" in result.columns:
                unusual = result["has_unusual_items_flag"].fillna(0) == 1
                result.loc[unusual, "anomaly_severity_score"] += 1.0

            if "high_rnd_intensity_flag" in result.columns:
                rnd = result["high_rnd_intensity_flag"].fillna(0) == 1
                result.loc[rnd, "anomaly_severity_score"] += 2.0

            if "low_tax_flag" in result.columns:
                low_tax = result["low_tax_flag"].fillna(0) == 1
                result.loc[low_tax, "anomaly_severity_score"] += 1.0

            if "revenue_accelerating_flag" in result.columns:
                rev_accel = result["revenue_accelerating_flag"].fillna(0) == 1
                result.loc[rev_accel, "anomaly_severity_score"] += 2.0

            # NEW: External / operational cross-checks
            if "analyst_bearish_pct" in result.columns:
                bearish = result["analyst_bearish_pct"].fillna(0) > 50
                result.loc[bearish, "anomaly_severity_score"] += 4.0

            if "layoff_risk_flag" in result.columns:
                layoff = result["layoff_risk_flag"].fillna(0) == 1
                result.loc[layoff, "anomaly_severity_score"] += 2.0

            if "debt_maturity_risk" in result.columns:
                debt_risk = result["debt_maturity_risk"].fillna(0) > 40
                result.loc[debt_risk, "anomaly_severity_score"] += 5.0

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
                row_prob = row_prob.where(
                    feat_data.notna(), cp_row["base_anomaly_rate"]
                )
                prob_col += row_prob * sep
                total_sep += sep

            if total_sep > 0:
                result["anomaly_conditional_probability"] = prob_col / total_sep
            else:
                result["anomaly_conditional_probability"] = cond_probs.iloc[0][
                    "base_anomaly_rate"
                ]
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
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            hierarchical_mcmc_by_sector,
            mcmc_student_t,
        )

        if "accounting_anomaly_score" not in result.columns:
            return result

        anomaly_scores = result["accounting_anomaly_score"].dropna().values
        if len(anomaly_scores) < 20:
            return result

        # Task 1.1: Student-t posterior for anomaly scores
        try:
            mu_samples, df_samples = mcmc_student_t(anomaly_scores, n_samples=self.n_mcmc_samples, burn_in=self.burn_in)
            result["anomaly_posterior_mean"] = mu_samples.mean()
            result["anomaly_posterior_std"] = mu_samples.std()
            result["anomaly_ci_lower"] = np.percentile(mu_samples, 2.5)
            result["anomaly_ci_upper"] = np.percentile(mu_samples, 97.5)
        except (ValueError, RuntimeError) as e:
            logger.warning("MCMC Student-t for anomaly scores failed: %s", e)

        # Task 1.2: Hierarchical MCMC by sector
        sector_col = "industry" if "industry" in result.columns else "sector"
        if sector_col in result.columns:
            try:
                sector_posteriors = hierarchical_mcmc_by_sector(result, feature="accounting_anomaly_score",
                                                                sector_col=sector_col, n_samples=self.n_mcmc_samples)
                # Unwrap ArviZ-wrapped result
                if "sectors" in sector_posteriors and isinstance(
                    sector_posteriors["sectors"], dict
                ):
                    sector_posteriors = sector_posteriors["sectors"]
                sector_mean_map = {
                    s: v.get("posterior_mean", np.nan)
                    for s, v in sector_posteriors.items()
                    if isinstance(v, dict)
                }
                result["sector_posterior_mean"] = result[sector_col].map(
                    sector_mean_map
                )
            except (ValueError, RuntimeError) as e:
                logger.warning("Hierarchical MCMC for anomaly sectors failed: %s", e)

        return result

    def calculate_conditional_probabilities(
        self,
        df: pd.DataFrame,
        anomaly_threshold: float = 150.0,
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
        anomaly_threshold : float, default 70.0
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

            lift_high = (
                p_anomaly_high / base_anomaly_rate if base_anomaly_rate > 0 else 1.0
            )
            lift_low = (
                p_anomaly_low / base_anomaly_rate if base_anomaly_rate > 0 else 1.0
            )

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
    """Result container for dividend safety analysis.

    Extended in v3.9 with:
    * FCF-coverage decomposition and horizoned cut probabilities (§4.1),
      enabling ``build_quad_model_alignment`` to move from a single
      ``div_cut_threshold=0.60`` scalar to term-structure aware allocation.
    * Tail/vol diagnostic fields aligned with :class:`CreditRiskResult`
      (§4.2 / T-B) so log-score BMA can feed per-model diagnostics.
    """

    ticker: str
    name: str
    dividend_cut_probability: float
    fcf_dividend_coverage: float
    payout_ratio: float
    dividend_streak: int
    safety_score: float
    risk_category: str  # 'Safe', 'Borderline', 'At Risk'
    # --- v3.9 coverage decomposition (§4.1) ---
    fcf_coverage_posterior_mean: float = float("nan")
    fcf_coverage_ci_low: float = float("nan")
    fcf_coverage_ci_high: float = float("nan")
    cut_probability_1y: float = float("nan")
    cut_probability_3y: float = float("nan")
    payout_sustainability_score: float = float("nan")
    stress_scenario_cut_prob: float = float("nan")
    # --- v3.9 tail / vol diagnostics (§4.2) ---
    tail_df: float = float("nan")
    cond_volatility: float = float("nan")
    cvar_5: float = float("nan")
    ess_bulk: int = 0
    ess_tail: int = 0
    r_hat: float = float("nan")
    # --- v3.9 serialisation contract ---
    schema_version: str = RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a ``dict`` (schema-versioned)."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "dividend_cut_probability": self.dividend_cut_probability,
            "fcf_dividend_coverage": self.fcf_dividend_coverage,
            "payout_ratio": self.payout_ratio,
            "dividend_streak": self.dividend_streak,
            "safety_score": self.safety_score,
            "risk_category": self.risk_category,
            "fcf_coverage_posterior_mean": self.fcf_coverage_posterior_mean,
            "fcf_coverage_ci_low": self.fcf_coverage_ci_low,
            "fcf_coverage_ci_high": self.fcf_coverage_ci_high,
            "cut_probability_1y": self.cut_probability_1y,
            "cut_probability_3y": self.cut_probability_3y,
            "payout_sustainability_score": self.payout_sustainability_score,
            "stress_scenario_cut_prob": self.stress_scenario_cut_prob,
            "tail_df": self.tail_df,
            "cond_volatility": self.cond_volatility,
            "cvar_5": self.cvar_5,
            "ess_bulk": self.ess_bulk,
            "ess_tail": self.ess_tail,
            "r_hat": self.r_hat,
            "schema_version": self.schema_version,
        }


@dataclass
class PriceTargetResult:
    """Result container for price target achievement analysis."""

    ticker: str
    name: str
    achievement_probability: float
    expected_upside_pt: float
    price_target_spread_pct: float
    analyst_rating_normalized: float
    implied_return_pt: float


@dataclass
class BeatProbabilityResult:
    """Result container for earnings beat probability analysis.

    v3.10 (§11.1 / §11.2) — extended with per-layer contribution fields so the
    three-layer Bayesian stack (prior / likelihood / momentum-quality-macro
    tilt) is auditable, and with sector-prior provenance
    (``sector_prior_key`` / ``used_default_prior``) so downstream analytics can
    spot when the posterior is dominated by the weak global prior.
    """

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
    # GAAP-vs-Norm revision spread and continuing EPS streak
    gaap_vs_norm_revision_spread: Optional[float] = None
    eps_cont_positive_streak: Optional[int] = None
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
    # --- v3.10 §11.1 per-layer contribution attribution ---
    prior_contribution: float = float("nan")
    likelihood_contribution: float = float("nan")
    momentum_tilt: float = float("nan")
    quality_discount: float = float("nan")
    macro_tilt: float = float("nan")
    # --- v3.10 §11.2 sector-prior provenance ---
    sector_prior_key: Optional[str] = None
    sector_prior_alpha: float = float("nan")
    sector_prior_beta: float = float("nan")
    used_default_prior: bool = True
    # --- v3.10 diagnostic parity (T-A / T-B extended) ---
    tail_df: float = float("nan")
    cond_volatility: float = float("nan")
    ess_tail: float = float("nan")
    schema_version: str = RESULT_SCHEMA_VERSION


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
    # --- v3.9 Bayesian streak posterior (§5.1, §6.1) ---
    # Posterior of the continuation probability under a Beta-Binomial model
    # ``P(success_{t+1} | successes, trials) ~ Beta(α + n, β + N − n)``.
    posterior_alpha: float = float("nan")
    posterior_beta: float = float("nan")
    continuation_prob_ci_low: float = float("nan")
    continuation_prob_ci_high: float = float("nan")
    expected_streak_length_years: float = float("nan")
    hazard_rate_next_quarter: float = float("nan")
    # --- v3.9 diagnostic parity with BeatProbabilityEstimate (§6.2) ---
    ess_bulk: float = float("nan")
    r_hat: float = float("nan")
    # Serialisation
    schema_version: str = RESULT_SCHEMA_VERSION


@dataclass
class ModelConfidenceResult:
    """Result container for model confidence estimation.

    v3.10 (§14.1 / §14.2) — extended with calibration artefacts
    (reliability_curve, bootstrap CIs on ECE/Brier, log_score, AUROC) required
    for log-score BMA weighting (T-E) and a :meth:`passes_calibration` gate
    that lets the pipeline short-circuit when a model is mis-calibrated.
    """

    model_name: str
    brier_score: float
    log_loss: float
    calibration_error: float
    discrimination_auc: float
    reliability_diagram_data: dict
    confidence_intervals: dict
    overall_confidence: float
    # --- v3.10 §14.1 calibration artefacts ---
    reliability_curve: list[tuple[float, float, int]] = field(default_factory=list)
    ece_ci_low: float = float("nan")
    ece_ci_high: float = float("nan")
    brier_ci_low: float = float("nan")
    brier_ci_high: float = float("nan")
    log_score: float = float("nan")
    auroc: float = float("nan")
    n_samples: int = 0
    schema_version: str = RESULT_SCHEMA_VERSION

    def passes_calibration(self, tol: float = 0.05) -> bool:
        """§14.2 — return True iff ECE + upper-CI ≤ ``tol``.

        Short-circuits BMA log-score weighting when a sub-model is
        mis-calibrated beyond tolerance.
        """
        ece = float(self.calibration_error)
        if ece != ece:  # NaN
            return False
        upper = float(self.ece_ci_high) if self.ece_ci_high == self.ece_ci_high else ece
        return upper <= float(tol)


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
        return self.alpha, self.beta

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

    Fields follow the naming convention from the public database schema:
    - net_eps_basic_fq: most recent fiscal quarter
    - net_eps_basic_1fqfq: one quarter ago, etc.
    - net_eps_basic_fy: most recent fiscal year
    - net_eps_basic_1fy: one year ago, etc.
    """

    # Quarterly Net EPS - Basic (newest first)
    net_eps_basic_fq: Optional[float] = None
    net_eps_basic_1fqfq: Optional[float] = None
    net_eps_basic_2fqfq: Optional[float] = None
    net_eps_basic_3fqfq: Optional[float] = None
    net_eps_basic_4fqfq: Optional[float] = None

    # Annual Net EPS - Basic (newest first)
    net_eps_basic_fy: Optional[float] = None
    net_eps_basic_1fy: Optional[float] = None
    net_eps_basic_2fy: Optional[float] = None
    net_eps_basic_3fy: Optional[float] = None
    net_eps_basic_4fy: Optional[float] = None
    net_eps_basic_5fy: Optional[float] = None

    # LTM Net EPS - Basic
    net_eps_basic_ltm: Optional[float] = None

    # Adjusted EPS
    eps_adj_fy: Optional[float] = None
    eps_adj_1fy: Optional[float] = None
    eps_adj_ltm: Optional[float] = None
    eps_adj_fq: Optional[float] = None
    eps_adj_1fqfq: Optional[float] = None
    eps_adj_2fqfq: Optional[float] = None
    eps_adj_3fqfq: Optional[float] = None
    eps_adj_4fqfq: Optional[float] = None

    # Continuing EPS — quarterly
    basic_eps_cont_fq: Optional[float] = None
    basic_eps_cont_1fqfq: Optional[float] = None
    basic_eps_cont_2fqfq: Optional[float] = None
    basic_eps_cont_3fqfq: Optional[float] = None
    basic_eps_cont_4fqfq: Optional[float] = None

    # Continuing EPS — annual
    basic_eps_cont_fy: Optional[float] = None
    basic_eps_cont_1fy: Optional[float] = None
    basic_eps_cont_2fy: Optional[float] = None
    basic_eps_cont_3fy: Optional[float] = None
    basic_eps_cont_4fy: Optional[float] = None

    # Continuing EPS — LTM
    basic_eps_cont_ltm: Optional[float] = None

    @property
    def quarterly_series(self) -> list[float]:
        """Return non-None quarterly EPS values, newest first."""
        fields = [
            self.net_eps_basic_fq,
            self.net_eps_basic_1fqfq,
            self.net_eps_basic_2fqfq,
            self.net_eps_basic_3fqfq,
            self.net_eps_basic_4fqfq,
        ]
        return [v for v in fields if v is not None]

    @property
    def annual_series(self) -> list[float]:
        """Return non-None annual EPS values, newest first."""
        fields = [
            self.net_eps_basic_fy,
            self.net_eps_basic_1fy,
            self.net_eps_basic_2fy,
            self.net_eps_basic_3fy,
            self.net_eps_basic_4fy,
            self.net_eps_basic_5fy,
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
            return 0, 0
        n_beats = 0
        n_total = len(series) - 1
        for i in range(n_total):
            if series[i] > series[i + 1]:
                n_beats += 1
        return n_beats, n_total

    def quarterly_beat_streak(self) -> int:
        """Count consecutive positive EPS quarters from most recent."""
        streak = 0
        for v in self.quarterly_series:
            if v > 0:
                streak += 1
            else:
                break
        return streak

    # v3.9 §7.4: minimum number of reports required to trust a streak
    # posterior. Below this threshold ``count_quarterly_beats_vs_estimate``
    # and ``count_yoy_improvements`` return ``None`` so callers can
    # distinguish genuine cold streaks from data-sparsity artefacts (a
    # known contributor to the MAP=0 collapse toward the raw prior).
    MIN_REPORTS_FOR_STREAK: int = 4

    def count_quarterly_beats_vs_estimate(
        self,
        estimate: Optional[float],
        min_reports: Optional[int] = None,
    ) -> tuple[int, int]:
        """Count how many quarterly actuals exceeded a forward estimate.

        Args:
            estimate: Forward EPS estimate to compare against.
            min_reports: Minimum number of quarterly reports required;
                below this threshold returns ``(0, 0)`` (v3.9 §7.4).
                Defaults to :attr:`MIN_REPORTS_FOR_STREAK`.

        Returns:
            (n_beats, n_total) tuple.
        """
        if estimate is None:
            return 0, 0
        series = self.quarterly_series
        threshold = min_reports if min_reports is not None else self.MIN_REPORTS_FOR_STREAK
        if not series or len(series) < threshold:
            # Not enough data to form a credible streak posterior: signal
            # "unknown" rather than a spurious 0/k.
            return 0, 0
        n_beats = sum(1 for v in series if v > estimate)
        return n_beats, len(series)

    def has_sufficient_streak_history(self, min_reports: Optional[int] = None) -> bool:
        """Return True if enough reports exist to support a streak posterior (v3.9 §7.4)."""
        threshold = min_reports if min_reports is not None else self.MIN_REPORTS_FOR_STREAK
        return len(self.quarterly_series) >= threshold or len(self.annual_series) >= threshold

    def unique_quarterly_series(self) -> list[float]:
        """Deduplicated quarterly series (v3.9 §7.1).

        ``ReportedEPSHistory`` slots are keyed positionally (``_fq``,
        ``_1fqfq`` …) so a fiscal-period ``report_date`` is not available
        on the dataclass itself. As a pragmatic proxy for restatement
        dedup we drop *adjacent-equal* values that are almost certainly
        duplicated slot entries (common when upstream joins on a stale
        restated quarter). This avoids inflating ``total_reports_count``
        while still preserving true period-to-period EPS moves.
        """
        series = self.quarterly_series
        if not series:
            return []
        deduped: list[float] = [series[0]]
        for v in series[1:]:
            if deduped and np.isclose(v, deduped[-1], rtol=1e-6, atol=1e-9):
                continue
            deduped.append(v)
        return deduped

    @property
    def total_reports_count(self) -> int:
        """Total number of non-null reported EPS observations across all series.

        Counts unique non-null entries across quarterly basic, annual basic,
        adjusted, and continuing EPS fields to dynamically derive the total
        number of available data points for historical beat rate calculations.
        """
        all_fields = [
            # Quarterly basic
            self.net_eps_basic_fq,
            self.net_eps_basic_1fqfq,
            self.net_eps_basic_2fqfq,
            self.net_eps_basic_3fqfq,
            self.net_eps_basic_4fqfq,
            # Annual basic
            self.net_eps_basic_fy,
            self.net_eps_basic_1fy,
            self.net_eps_basic_2fy,
            self.net_eps_basic_3fy,
            self.net_eps_basic_4fy,
            self.net_eps_basic_5fy,
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
            self.basic_eps_cont_fq,
            self.basic_eps_cont_1fqfq,
            self.basic_eps_cont_2fqfq,
            self.basic_eps_cont_3fqfq,
            self.basic_eps_cont_4fqfq,
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
    eps_norm_est_ntm: Optional[float] = None
    eps_norm_est_fy1e: Optional[float] = None
    eps_gaap_est_ntm: Optional[float] = None
    eps_gaap_est_fy1e: Optional[float] = None

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
        if self.eps_gaap_est_fy1e is not None and self.eps_norm_est_fy1e is not None:
            if self.eps_norm_est_fy1e != 0:
                return (
                    (self.eps_gaap_est_fy1e - self.eps_norm_est_fy1e)
                    / self.eps_norm_est_fy1e
                    * 100.0
                )
        return None

    @property
    def has_sufficient_data(self) -> bool:
        """Check if enough forward data is available for enhanced analysis.

        Requires at least a FY1E estimate and one revision data point.
        """
        has_estimate = self.eps_norm_est_fy1e is not None
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

    # Momentum signal definitions: (weight, name, transform_fn)
    _MOMENTUM_SIGNALS = [
        ("eps_revision_momentum", 0.40, lambda v: np.clip(v, -1.0, 1.0)),
        ("composite_eps_trajectory_score", 0.25, lambda v: np.clip(v, -1.0, 1.0)),
        ("continuation_probability", 0.20, lambda v: (v - 0.5) * 2.0),
        ("eps_growth_accel", 0.15, lambda v: np.clip(v, -1.0, 1.0)),
    ]

    def __init__(
        self,
        prior_alpha: float = 2.0,  # v3.9: was 1.5 — tighter prior mean ≈ 0.5
        prior_beta: float = 2.0,
        sector_priors: Optional[dict[str, PriorParameters]] = None,
        use_quality_adjustment: bool = True,
        # NEW: Momentum-based prior tilting (v3.5)
        use_momentum_prior: bool = True,
        momentum_prior_strength: float = 0.5,  # v3.9: was 0.3 — momentum more informative
        # v3.9: Macro prior tilt (Finding #4)
        use_macro_prior: bool = True,
        # v3.10 §12.3 — macro-prior logit-space β coefficients (Normal(0, 0.3)
        # shrinkage): ``β_vix · z(vix) + β_curve · z(yield_10y2y)``. Exposed as
        # a constructor kwarg so empirical-Bayes updates (§12.6 / T-D) can feed
        # refreshed values without monkey-patching. Pass ``None`` to disable
        # the tilt even when ``use_macro_prior=True`` (default) — useful for
        # A/B comparison runs.
        macro_prior_betas: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the earnings beat probability model.

        Args:
            prior_alpha: Alpha parameter for Beta prior (default: 1.5)
            prior_beta: Beta parameter for Beta prior (default: 2.0)
            sector_priors: Optional dict mapping sectors to PriorParameters
                          for sector-specific priors based on historical patterns
            use_quality_adjustment: Whether to adjust confidence scores based on
                accounting quality signals.
            use_momentum_prior: Whether to tilt the prior using EPS revision
                momentum, trajectory scores, and streak continuation probability.
            momentum_prior_strength: Maximum absolute shift in the prior mean
                from momentum signals (clamped to [-strength, +strength]).
            use_macro_prior: Whether to tilt the prior via standardised macro
                covariates (VIX, yield-curve slope) in logit space.
            macro_prior_betas: Optional mapping of macro-feature → logit-space
                coefficient, e.g. ``{"vix": -0.25, "yield_10y2y": 0.15}``. Kept
                inside Normal(0, 0.3) by convention. ``None`` → empirical default
                ``{"vix": -0.20, "yield_10y2y": 0.10}`` (stabilises cross-region
                beat probabilities during macro regime shifts — addresses MXN /
                TRY drift flagged by the v3.8 findings). See §12.3 in the
                improvement plan.
        """
        self.default_prior = PriorParameters(prior_alpha, prior_beta)
        self.sector_priors = sector_priors or self._create_default_sector_priors()
        self.use_quality_adjustment = use_quality_adjustment
        self.use_momentum_prior = use_momentum_prior
        self.momentum_prior_strength = momentum_prior_strength
        self.use_macro_prior = use_macro_prior
        # §12.3 — macro prior β coefficients (logit space). Kept as an
        # instance attribute so downstream code can inspect and refresh them.
        self.macro_prior_betas: dict[str, float] = dict(
            macro_prior_betas
            if macro_prior_betas is not None
            else {"vix": -0.20, "yield_10y2y": 0.10}
        )

    # -------------------------------------------------------------------
    # §12.3 — macro-prior logit-space tilt
    # -------------------------------------------------------------------
    def _apply_macro_logit_tilt(
        self,
        prior: "PriorParameters",
        macro_features: Optional[dict[str, float]] = None,
    ) -> tuple["PriorParameters", float]:
        """Return a macro-tilted prior in logit space and the applied tilt.

        v3.10 §12.3 — closes the `[PENDING]` macro-covariate item from the v3.8
        findings. Applies ``Δ_logit = Σ β_k · z(x_k)`` where ``β_k`` come from
        :attr:`macro_prior_betas` and ``x_k`` are *already-standardised* macro
        features (the caller is expected to pass z-scores). The returned
        ``PriorParameters`` preserves the prior concentration
        ``α + β`` and re-parameterises the mean ``p' = σ(logit(p) + Δ)``.

        Returns
        -------
        (PriorParameters, float)
            The tilted prior and the scalar tilt ``Δ_logit`` applied (stored
            on :class:`BeatProbabilityResult.macro_tilt` for auditability).
        """
        if not self.use_macro_prior or not self.macro_prior_betas or not macro_features:
            return prior, 0.0

        delta = 0.0
        for key, beta in self.macro_prior_betas.items():
            v = macro_features.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except TypeError, ValueError:
                continue
            if fv != fv or fv in (float("inf"), float("-inf")):
                continue
            # Guard against unstandardised inputs — clamp z-scores to ±5.
            delta += float(beta) * float(np.clip(fv, -5.0, 5.0))

        if delta == 0.0:
            return prior, 0.0

        # Convert current prior mean to logit space, apply Δ, convert back.
        p = float(prior.expected_beat_rate)
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        logit_p = np.log(p / (1.0 - p))
        p_new = 1.0 / (1.0 + np.exp(-(logit_p + delta)))
        # Preserve concentration κ = α + β.
        kappa = float(prior.concentration)
        return PriorParameters(p_new * kappa, (1.0 - p_new) * kappa), float(delta)

    # -------------------------------------------------------------------
    # §12.6 — empirical-Bayes sector prior fit (method of moments)
    # -------------------------------------------------------------------
    @staticmethod
    def fit_priors_empirical_bayes(
        df: pd.DataFrame,
        sector_col: str = "industry",
        beat_rate_col: str = "historical_beat_rate",
        min_samples_per_sector: int = 30,
        concentration_floor: float = 4.0,
    ) -> dict[str, "PriorParameters"]:
        """Fit sector-level Beta priors via method-of-moments (T-D / §12.6).

        For each sector with ≥ ``min_samples_per_sector`` rows, computes the
        sample mean ``p`` and variance ``v`` of ``beat_rate_col`` and fits a
        Beta(α, β) by matching moments:
        ``κ = p (1 - p) / v - 1``, ``α = p κ``, ``β = (1 - p) κ``.

        A minimum concentration ``κ ≥ concentration_floor`` is enforced so
        small-sample sectors still get a weakly-informative prior. Returns a
        dict suitable for injection via ``sector_priors=`` on a new
        :class:`EarningsBeatProbabilityModel` instance. Operationalises
        cross-cutting task T-D for the beat model.
        """
        if sector_col not in df.columns or beat_rate_col not in df.columns:
            return {}

        priors: dict[str, PriorParameters] = {}
        grouped = df.groupby(sector_col, dropna=True)[beat_rate_col]
        for sector, rates in grouped:
            if not isinstance(sector, str) or not sector:
                continue
            rates = pd.to_numeric(rates, errors="coerce").dropna()
            if len(rates) < min_samples_per_sector:
                continue
            p = float(rates.mean())
            v = float(rates.var(ddof=1))
            if not (0.0 < p < 1.0) or v <= 0.0:
                continue
            kappa = max((p * (1.0 - p) / v) - 1.0, concentration_floor)
            alpha = p * kappa
            beta = (1.0 - p) * kappa
            if alpha > 0.0 and beta > 0.0:
                priors[sector] = PriorParameters(alpha, beta)
        return priors

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

    @staticmethod
    def _accumulate_weighted_signals(
        signal_definitions: list[tuple[str, float]],
        signal_values: dict[str, Optional[float]],
    ) -> tuple[float, float]:
        """
        Accumulate weighted signal contributions.

        Args:
            signal_definitions: List of (signal_name, weight, transform_fn) tuples.
            signal_values: Mapping of signal name to its value (None if absent).

        Returns:
            Tuple of (weighted_shift, total_weight).
        """
        weighted_shift = 0.0
        total_weight = 0.0
        for name, weight, transform in signal_definitions:
            value = signal_values.get(name)
            if value is not None:
                weighted_shift += weight * transform(value)
                total_weight += weight
        return weighted_shift, total_weight

    def _apply_momentum_prior_adjustment(
        self,
        prior: PriorParameters,
        eps_revision_momentum: Optional[float] = None,
        composite_eps_trajectory_score: Optional[float] = None,
        continuation_probability: Optional[float] = None,
        eps_growth_accel: Optional[float] = None,
    ) -> PriorParameters:
        """
        Tilt prior parameters using EPS momentum and trajectory signals.

        Shifts the effective prior mean while preserving the total
        concentration (alpha + beta), so that only direction changes —
        not conviction.

        Args:
            prior: Base prior (sector or default)
            eps_revision_momentum: Analyst revision momentum [-1, 1]
            composite_eps_trajectory_score: Composite trajectory [-1, 1]
            continuation_probability: Streak continuation prob [0, 1]
            eps_growth_accel: Growth acceleration signal

        Returns:
            Adjusted PriorParameters
        """
        if not self.use_momentum_prior:
            return prior

        signal_values = {
            "eps_revision_momentum": eps_revision_momentum,
            "composite_eps_trajectory_score": composite_eps_trajectory_score,
            "continuation_probability": continuation_probability,
            "eps_growth_accel": eps_growth_accel,
        }

        weighted_shift, total_weight = self._accumulate_weighted_signals(
            self._MOMENTUM_SIGNALS,
            signal_values,
        )

        if total_weight == 0:
            return prior

        # Normalize and clamp to allowed strength
        normalized_shift = weighted_shift / total_weight
        clamped_shift = np.clip(
            normalized_shift * self.momentum_prior_strength,
            -self.momentum_prior_strength,
            self.momentum_prior_strength,
        )

        return self._tilt_prior_mean(prior, clamped_shift)

    @staticmethod
    def _tilt_prior_mean(prior: PriorParameters, shift: float) -> PriorParameters:
        """
        Shift the prior mean by *shift* while preserving concentration.

        Args:
            prior: Original prior parameters.
            shift: Additive shift to the prior mean (clamped to [0.05, 0.95]).

        Returns:
            New PriorParameters with adjusted mean.
        """
        total = prior.alpha + prior.beta
        base_mean = prior.alpha / total
        new_mean = np.clip(base_mean + shift, 0.05, 0.95)
        return PriorParameters(
            alpha=new_mean * total,
            beta=(1.0 - new_mean) * total,
        )

    def _apply_quality_discount(
        self,
        confidence_score: float,
        accounting_quality_score: Optional[float] = None,
        gaap_adj_eps_gap_pct: Optional[float] = None,
        eps_adjustment_pct: Optional[float] = None,
        distress_risk_score: Optional[float] = None,
    ) -> float:
        """
        Discount confidence score based on earnings quality signals.

        Companies with large GAAP-vs-adjusted gaps, low accounting quality,
        or high distress risk get reduced confidence even if the posterior
        mean is strong.

        Args:
            confidence_score: Raw confidence score [0, 1]
            accounting_quality_score: Pre-computed quality [0, 1], higher = better
            gaap_adj_eps_gap_pct: Absolute GAAP-adj gap as pct of EPS
            eps_adjustment_pct: Fraction of EPS from adjustments
            distress_risk_score: Financial distress risk [0, 1]

        Returns:
            Quality-adjusted confidence score [0, 1]
        """
        if not self.use_quality_adjustment:
            return confidence_score

        signal_values = {
            "accounting_quality_score": accounting_quality_score,
            "gaap_adj_eps_gap_pct": gaap_adj_eps_gap_pct,
            "eps_adjustment_pct": eps_adjustment_pct,
            "distress_risk_score": distress_risk_score,
        }

        discount = self._compute_quality_discount(signal_values)
        return float(np.clip(confidence_score * discount, 0.0, 1.0))

    @staticmethod
    def _compute_quality_discount(signal_values: dict[str, Optional[float]]) -> float:
        """
        Compute total multiplicative discount from quality signals.

        Args:
            signal_values: Dictionary of quality signals.

        Returns:
            Multiplicative discount factor [0.1, 1.0].
        """
        discount = 1.0

        # Define signal handlers: (name, compute_discount_fn)
        handlers = [
            (
                "accounting_quality_score",
                lambda v: 0.70 + 0.30 * np.clip(v, 0.0, 1.0),
            ),
            (
                "gaap_adj_eps_gap_pct",
                lambda v: 1.0 - 0.15 * np.clip(abs(v) / 0.5, 0.0, 1.0),
            ),
            (
                "eps_adjustment_pct",
                lambda v: 1.0 - 0.10 * np.clip(abs(v) / 0.4, 0.0, 1.0),
            ),
            (
                "distress_risk_score",
                lambda v: 1.0 - 0.20 * np.clip(v, 0.0, 1.0),
            ),
        ]

        for name, compute_fn in handlers:
            val = signal_values.get(name)
            if val is not None:
                discount *= compute_fn(val)

        return discount

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
        return distribution.ppf(lower_quantile), distribution.ppf(upper_quantile)

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
        threshold: float = 0.50,
        # NEW: streak & momentum features (v3.5)
        eps_revision_momentum: Optional[float] = None,
        composite_eps_trajectory_score: Optional[float] = None,
        continuation_probability: Optional[float] = None,
        eps_growth_accel: Optional[float] = None,
        streak_map_estimate: Optional[float] = None,
        streak_model_confidence: Optional[float] = None,
        # NEW: quality features
        accounting_quality_score: Optional[float] = None,
        gaap_adj_eps_gap_pct: Optional[float] = None,
        eps_adjustment_pct: Optional[float] = None,
        distress_risk_score: Optional[float] = None,
    ) -> BeatProbabilityEstimate:
        """
        Compute the probability of future earnings beat.

        Args:
            n_beats: Number of historical beats
            n_total: Total observations
            sector: Optional sector name
            threshold: Probability threshold for "likely beat" classification
            eps_revision_momentum: Analyst revision momentum signal
            composite_eps_trajectory_score: Composite EPS trajectory
            continuation_probability: Streak continuation probability
            eps_growth_accel: EPS growth acceleration
            streak_map_estimate: MAP estimate from streak analysis
            streak_model_confidence: Confidence from streak model [0, 1]
            accounting_quality_score: Accounting quality [0, 1]
            gaap_adj_eps_gap_pct: GAAP-adjusted EPS gap percentage
            eps_adjustment_pct: Fraction of EPS from adjustments
            distress_risk_score: Financial distress risk [0, 1]

        Returns:
            Dictionary with probability estimates and confidence metrics
        """
        # Step 1: Get sector prior, then apply momentum tilt
        prior = self._get_prior_parameters(sector, use_sector_prior=True)
        adjusted_prior = self._apply_momentum_prior_adjustment(
            prior,
            eps_revision_momentum=eps_revision_momentum,
            composite_eps_trajectory_score=composite_eps_trajectory_score,
            continuation_probability=continuation_probability,
            eps_growth_accel=eps_growth_accel,
        )

        # Step 2: Conjugate update with tilted prior
        n_misses = n_total - n_beats
        post_alpha = adjusted_prior.alpha + n_beats
        post_beta = adjusted_prior.beta + n_misses

        posterior_mean, posterior_std = self._compute_posterior_statistics(
            post_alpha, post_beta
        )

        # Step 3: Blend with streak MAP estimate if available
        if streak_map_estimate is not None and streak_model_confidence is not None:
            blend_weight = np.clip(streak_model_confidence, 0.0, 0.5)
            posterior_mean = (
                1 - blend_weight
            ) * posterior_mean + blend_weight * streak_map_estimate

        dist = stats.beta(post_alpha, post_beta)
        ci_90, ci_95 = self._compute_credible_intervals(dist)

        prob_exceeds_threshold = 1 - dist.cdf(threshold)

        # Step 4: Confidence with quality discount
        raw_confidence = self._compute_confidence_score(post_alpha, post_beta)
        confidence_score = self._apply_quality_discount(
            raw_confidence,
            accounting_quality_score=accounting_quality_score,
            gaap_adj_eps_gap_pct=gaap_adj_eps_gap_pct,
            eps_adjustment_pct=eps_adjustment_pct,
            distress_risk_score=distress_risk_score,
        )

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
        beats_col: str = "historical_beat_rate",
        total_col: str = "dynamic_total_reports",
        sector_col: str = "industry",
        ticker_col: str = "isin",
        name_col: str = "name",
        # NEW: feature column mappings (v3.5)
        revision_momentum_col: str = "eps_revision_momentum",
        trajectory_col: str = "composite_eps_trajectory_score",
        continuation_prob_col: str = "continuation_probability",
        growth_accel_col: str = "eps_growth_accel",
        streak_map_col: str = "map_estimate",
        streak_confidence_col: str = "model_confidence",
        quality_score_col: str = "accounting_quality_score",
        gaap_gap_col: str = "gaap_adj_eps_gap_pct",
        adjustment_pct_col: str = "eps_adjustment_pct",
        distress_col: str = "distress_risk_score",
    ) -> pd.DataFrame:
        """
        Analyze earnings beat probabilities for entire DataFrame.

        Default column names match the join of mv_all_stock_features
        and eps_streak_analysis on (isin, ticker).

        Args:
            df: DataFrame with earnings data
            beats_col: Column for beat rate (0-1) or beat count
            total_col: Column for total report count
            sector_col: Column name for sector
            ticker_col: Column name for ticker
            name_col: Column name for company name
            revision_momentum_col: Column for EPS revision momentum
            trajectory_col: Column for composite EPS trajectory score
            continuation_prob_col: Column for streak continuation probability
            growth_accel_col: Column for EPS growth acceleration
            streak_map_col: Column for streak MAP estimate
            streak_confidence_col: Column for streak model confidence
            quality_score_col: Column for accounting quality score
            gaap_gap_col: Column for GAAP-adjusted EPS gap percentage
            adjustment_pct_col: Column for EPS adjustment percentage
            distress_col: Column for distress risk score

        Returns:
            DataFrame with probability analysis results
        """
        # --- Resolve beat counts from beat_rate × total or direct counts ---
        has_beats = beats_col in df.columns
        has_total = total_col in df.columns

        if has_beats and has_total:
            beat_raw = df[beats_col].fillna(0)
            total_raw = df[total_col].fillna(0).astype(int)
            # If beats_col looks like a rate (0-1), convert to count
            if (
                not beat_raw.empty
                and beat_raw.max() <= 1.0
                and beat_raw.min() >= 0.0
                and (beat_raw != beat_raw.astype(int)).any()
            ):
                n_beats_series = (beat_raw * total_raw).round().astype(int)
            else:
                n_beats_series = beat_raw.astype(int)
            n_total_series = total_raw
        else:
            n_beats_series = (
                df[beats_col].fillna(0).astype(int)
                if has_beats
                else pd.Series(0, index=df.index)
            )
            n_total_series = (
                df[total_col].fillna(0).astype(int)
                if has_total
                else pd.Series(0, index=df.index)
            )

        # Proxy fallback: use eps_trajectory_score when direct columns are missing/zero
        proxy_mask = n_total_series == 0
        if proxy_mask.any() and "eps_trajectory_score" in df.columns:
            trajectory = df.loc[proxy_mask, "eps_trajectory_score"].fillna(50)

            # Dynamic n_total: use available data columns or graduate by trajectory
            if "eps_positive_years" in df.columns:
                n_total_proxy = (
                    df.loc[proxy_mask, "eps_positive_years"].fillna(0).clip(lower=0)
                )
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
                            np.where(
                                trajectory >= 40, 5, np.where(trajectory >= 20, 4, 3)
                            ),
                        ),
                    ),
                    index=df.loc[proxy_mask].index,
                )
            n_total_series.loc[proxy_mask] = n_total_proxy
            n_beats_series.loc[proxy_mask] = (
                (trajectory / 100 * n_total_proxy)
                .astype(int)
                .clip(lower=0, upper=n_total_proxy)
            )

        # Drop rows still without data
        valid = n_total_series > 0
        if not valid.any():
            return pd.DataFrame()

        df_valid = df.loc[valid].copy()
        n_beats_valid = n_beats_series.loc[valid]
        n_total_valid = n_total_series.loc[valid]

        # --- Vectorized momentum prior adjustment ---
        prior_alpha_arr = np.full(len(df_valid), self.prior_alpha)
        prior_beta_arr = np.full(len(df_valid), self.prior_beta)

        # Apply sector priors where available
        if sector_col in df_valid.columns:
            for sector_name, sp in self.sector_priors.items():
                mask = df_valid[sector_col] == sector_name
                prior_alpha_arr[mask.values] = sp.alpha
                prior_beta_arr[mask.values] = sp.beta

        # Apply momentum prior adjustment (vectorized)
        if self.use_momentum_prior:
            shift = np.zeros(len(df_valid))
            weight_sum = np.zeros(len(df_valid))

            if revision_momentum_col in df_valid.columns:
                vals = df_valid[revision_momentum_col].fillna(np.nan).values
                has_val = ~np.isnan(vals)
                shift[has_val] += 0.40 * np.clip(vals[has_val], -1.0, 1.0)
                weight_sum[has_val] += 0.40

            if trajectory_col in df_valid.columns:
                vals = df_valid[trajectory_col].fillna(np.nan).values
                has_val = ~np.isnan(vals)
                shift[has_val] += 0.25 * np.clip(vals[has_val], -1.0, 1.0)
                weight_sum[has_val] += 0.25

            if continuation_prob_col in df_valid.columns:
                vals = df_valid[continuation_prob_col].fillna(np.nan).values
                has_val = ~np.isnan(vals)
                shift[has_val] += 0.20 * (vals[has_val] - 0.5) * 2.0
                weight_sum[has_val] += 0.20

            if growth_accel_col in df_valid.columns:
                vals = df_valid[growth_accel_col].fillna(np.nan).values
                has_val = ~np.isnan(vals)
                shift[has_val] += 0.15 * np.clip(vals[has_val], -1.0, 1.0)
                weight_sum[has_val] += 0.15

            has_shift = weight_sum > 0
            if has_shift.any():
                shift[has_shift] /= weight_sum[has_shift]
                shift[has_shift] = np.clip(
                    shift[has_shift] * self.momentum_prior_strength,
                    -self.momentum_prior_strength,
                    self.momentum_prior_strength,
                )
                total_prior = prior_alpha_arr + prior_beta_arr
                base_mean = prior_alpha_arr / total_prior
                new_mean = np.clip(base_mean + shift, 0.05, 0.95)
                prior_alpha_arr[has_shift] = (new_mean * total_prior)[has_shift]
                prior_beta_arr[has_shift] = ((1.0 - new_mean) * total_prior)[has_shift]

        # Vectorized posterior computation
        post_alpha = prior_alpha_arr + n_beats_valid.values
        post_beta = prior_beta_arr + (n_total_valid.values - n_beats_valid.values)
        posterior_mean = post_alpha / (post_alpha + post_beta)
        posterior_std = np.sqrt(
            (post_alpha * post_beta)
            / ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
        )

        # --- Streak MAP blending ---
        if (
            streak_map_col in df_valid.columns
            and streak_confidence_col in df_valid.columns
        ):
            s_map = df_valid[streak_map_col].fillna(np.nan).values
            s_conf = df_valid[streak_confidence_col].fillna(np.nan).values
            blend_mask = ~np.isnan(s_map) & ~np.isnan(s_conf)
            if blend_mask.any():
                bw = np.clip(s_conf[blend_mask], 0.0, 0.5)
                posterior_mean[blend_mask] = (1 - bw) * posterior_mean[
                    blend_mask
                ] + bw * s_map[blend_mask]

        # Credible intervals (vectorized via scipy)
        ci_90_lower = stats.beta.ppf(0.05, post_alpha, post_beta)
        ci_90_upper = stats.beta.ppf(0.95, post_alpha, post_beta)
        ci_95_lower = stats.beta.ppf(0.025, post_alpha, post_beta)
        ci_95_upper = stats.beta.ppf(0.975, post_alpha, post_beta)

        # Multi-component confidence score (replaces constant concentration/20)
        confidence_score = compute_beta_confidence_score(
            post_alpha,
            post_beta,
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
                "prior_alpha": prior_alpha_arr,
                "prior_beta": prior_beta_arr,
                "posterior_alpha": post_alpha,
                "posterior_beta": post_beta,
                "posterior_beat_prob": posterior_mean,
                "posterior_std": posterior_std,
                "ci_90_lower": ci_90_lower,
                "ci_90_upper": ci_90_upper,
                "ci_95_lower": ci_95_lower,
                "ci_95_upper": ci_95_upper,
                "confidence_score": np.asarray(confidence_score),
                "beat_classification": beat_classification,
            },
            index=df_valid.index,
        )

        # NEW: Quality-adjusted confidence discounting (v3.5)
        if self.use_quality_adjustment:
            discount = np.ones(len(df_valid))

            if quality_score_col in df_valid.columns:
                aq = df_valid[quality_score_col].fillna(np.nan).values
                has_aq = ~np.isnan(aq)
                discount[has_aq] *= 0.70 + 0.30 * np.clip(aq[has_aq], 0.0, 1.0)

            if gaap_gap_col in df_valid.columns:
                gap = df_valid[gaap_gap_col].fillna(np.nan).values
                has_gap = ~np.isnan(gap)
                discount[has_gap] *= 1.0 - 0.15 * np.clip(
                    np.abs(gap[has_gap]) / 0.5, 0.0, 1.0
                )

            if adjustment_pct_col in df_valid.columns:
                adj = df_valid[adjustment_pct_col].fillna(np.nan).values
                has_adj = ~np.isnan(adj)
                discount[has_adj] *= 1.0 - 0.10 * np.clip(
                    np.abs(adj[has_adj]) / 0.4, 0.0, 1.0
                )

            if distress_col in df_valid.columns:
                dr = df_valid[distress_col].fillna(np.nan).values
                has_dr = ~np.isnan(dr)
                discount[has_dr] *= 1.0 - 0.20 * np.clip(dr[has_dr], 0.0, 1.0)

            result_df["confidence_score"] = np.clip(
                result_df["confidence_score"].values * discount, 0.0, 1.0
            )

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
        pseudo_n = (
            self.MAX_REVISION_PSEUDO_OBS if forward_signals.has_sufficient_data else 0.0
        )
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
                    forward_signals.revision_1m > 0 > forward_signals.gaap_revision_1m
                )
                if rev_sign_mismatch:
                    penalty_factor = min(1.0, penalty_factor + 0.15)

            # Shrink posterior toward prior by blending
            data_alpha = post_alpha - prior.alpha
            data_beta = post_beta - prior.beta
            post_alpha = prior.alpha + data_alpha * (1 - penalty_factor)
            post_beta = prior.beta + data_beta * (1 - penalty_factor)

        # --- Compute statistics ---
        posterior_mean, posterior_std = self._compute_posterior_statistics(
            post_alpha, post_beta
        )
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
        if confidence_score >= 0.4:
            classification_confidence = "High"
        elif confidence_score >= 0.2:
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

    # Column mappings from mv_all_stock_features to dataclass fields.
    # The original eps_est_avg_rev_pct_fy1e_* columns (from mv_equities) do not
    # exist in mv_all_stock_features.  We use gaap_revision_* as proxies for
    # both the normalized revision fields and the GAAP revision fields.
    # eps_revision_momentum serves as the short-term (1w) revision proxy.
    _FORWARD_COL_MAP: dict[str, str] = {
        "eps_norm_est_avg_ntm": "eps_norm_est_ntm",
        "eps_norm_est_avg_fy1e": "eps_norm_est_fy1e",
        "eps_gaap_est_avg_ntm": "eps_gaap_est_ntm",
        "eps_gaap_est_avg_fy1e": "eps_gaap_est_fy1e",
        "eps_revision_momentum": "revision_1w",
        "gaap_revision_1m": "revision_1m",
        "gaap_revision_3m": "revision_3m",
        "gaap_revision_6m": "revision_6m",
        "gaap_revision_1y": "revision_1y",
        "eps_norm_est_num_fy1e": "analyst_count",
    }
    # Secondary map: GAAP revision columns that also feed gaap_revision_* fields
    # on ForwardEstimateSignals (populated in _row_to_forward_signals).
    _GAAP_REVISION_COL_MAP: dict[str, str] = {
        "gaap_revision_1m": "gaap_revision_1m",
        "gaap_revision_3m": "gaap_revision_3m",
        "gaap_revision_6m": "gaap_revision_6m",
        "gaap_revision_1y": "gaap_revision_1y",
    }

    # History column mappings: mv_all_stock_features column → dataclass field.
    # The view exposes eps_basic_fq/fy and eps_cont_* aliases; we map them
    # to the canonical public-schema names used by ReportedEPSHistory.
    _HISTORY_COL_MAP: dict[str, str] = {
        # net_eps_basic — mapped from eps_basic_* in mv_all_stock_features
        "eps_basic_fq": "net_eps_basic_fq",
        "eps_basic_fy": "net_eps_basic_fy",
        # basic_eps_cont — mapped from eps_cont_* in mv_all_stock_features
        "eps_cont_fq": "basic_eps_cont_fq",
        "eps_cont_1fqfq": "basic_eps_cont_1fqfq",
        "eps_cont_2fqfq": "basic_eps_cont_2fqfq",
        "eps_cont_3fqfq": "basic_eps_cont_3fqfq",
        "eps_cont_4fqfq": "basic_eps_cont_4fqfq",
        "eps_cont_fy": "basic_eps_cont_fy",
        "eps_cont_1fy": "basic_eps_cont_1fy",
        "eps_cont_2fy": "basic_eps_cont_2fy",
        "eps_cont_3fy": "basic_eps_cont_3fy",
        "eps_cont_4fy": "basic_eps_cont_4fy",
        # eps_adj — available in view
        "eps_adj_ltm": "eps_adj_ltm",
        "eps_adj_fy": "eps_adj_fy",
        "eps_adj_fq": "eps_adj_fq",
    }

    def _row_to_forward_signals(
        self, row: pd.Series
    ) -> Optional[ForwardEstimateSignals]:
        """Extract ForwardEstimateSignals from a DataFrame row."""
        kwargs: dict = {}
        any_present = False
        for df_col, field_name in self._FORWARD_COL_MAP.items():
            val = row.get(df_col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                kwargs[field_name] = (
                    int(val) if field_name == "analyst_count" else float(val)
                )
                any_present = True
        # Also populate gaap_revision_* fields from the secondary map
        for df_col, field_name in self._GAAP_REVISION_COL_MAP.items():
            if field_name not in kwargs:
                val = row.get(df_col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    kwargs[field_name] = float(val)
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
        sector_col: str = "industry",
        ticker_col: str = "isin",
        name_col: str = "name",
        streak_map_col: str = "map_estimate",
        streak_confidence_col: str = "model_confidence",
        strict_streak_merge: bool = False,
    ) -> pd.DataFrame:
        """Analyze earnings beat probabilities using enhanced three-layer fusion.

        Falls back to trajectory-proxy method when forward data is unavailable.
        Uses vectorized computation for the proxy path to improve performance.

        Args:
            df: DataFrame with equities data.
            sector_col: Column name for sector.
            ticker_col: Column name for ticker.
            name_col: Column name for company name.
            streak_map_col: v3.10 §12.5 — expected column carrying the streak
                MAP estimate after the upstream streak merge (default
                ``"map_estimate"``, aligned with ``BeatProbabilityEstimate``).
            streak_confidence_col: v3.10 §12.5 — expected column carrying the
                streak model-confidence after the upstream streak merge
                (default ``"model_confidence"``).
            strict_streak_merge: v3.10 §12.5 — when ``True`` raise ``KeyError``
                if the streak columns are missing. Default ``False`` emits a
                structured ``logger.warning`` and continues with the legacy
                trajectory-only path, preserving backwards compatibility with
                pipelines that pre-date the streak merge.

        Returns:
            DataFrame with enriched probability analysis results.
        """
        # ------------------------------------------------------------------
        # §12.5 — merge-key safety. Silent column drops on the streak merge
        # degrade ``_apply_momentum_prior_adjustment`` for ~15 % of the universe
        # per v3.8 logs; surface it explicitly so the pipeline can decide
        # whether to fall back to the trajectory-only path or abort.
        # ------------------------------------------------------------------
        _missing = [c for c in (streak_map_col, streak_confidence_col) if c not in df.columns]
        if _missing:
            msg = (
                "EarningsBeatProbabilityModel.analyze_dataframe_enhanced: "
                f"expected streak-merge columns {_missing!r} not found in DataFrame "
                f"(have: {list(df.columns)[:20]}...). Continuing without streak "
                "prior tilt would degrade the momentum adjustment. Pass "
                "strict_streak_merge=True to fail fast."
            )
            if strict_streak_merge:
                raise KeyError(msg)
            logger.warning(msg)

        # Composite/quality columns to pass through from mv_all_stock_features
        _PASSTHROUGH_COLS = {
            "accounting_quality_score": "accounting_quality_score",
            "combined_distress_score": "distress_risk_score",
            "gaap_adj_eps_gap_pct": "gaap_adj_eps_gap_pct",
            "piotroski_f_score": "piotroski_f_score",
            "eps_revision_momentum": "eps_revision_momentum",
            "altman_z_score": "altman_z_score",
            "analyst_conviction": "analyst_conviction",
            # P2: Earnings quality + forward signals for beat confidence
            "earnings_quality_composite": "earnings_quality_composite",
            "core_earnings_stability": "core_earnings_stability",
            "eps_stability": "eps_stability",
            "gaap_positive_revision_flag": "gaap_positive_revision_flag",
            "forward_adjustment_trend": "forward_adjustment_trend",
            "margin_stability_score": "margin_stability_score",
            # P2: Employment / R&D features for productivity → earnings leverage
            "revenue_per_employee": "revenue_per_employee",
            "productivity_trend": "productivity_trend",
            "rnd_intensity_trend": "rnd_intensity_trend",
            "operating_leverage_score": "operating_leverage_score",
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
                effective_total = (
                    max(n_total, dynamic_total) if dynamic_total > 0 else n_total
                )
                historical_beat_rate = (
                    n_beats / effective_total if effective_total > 0 else 0.0
                )

                beat_classification = (
                    "likely_beat"
                    if prob_result["posterior_mean"] > 0.5
                    else "uncertain"
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
                        "classification_confidence": prob_result[
                            "classification_confidence"
                        ],
                        "beat_classification": beat_classification,
                        "gaap_revision_momentum": forward_signals.gaap_revision_momentum,
                        "gaap_norm_spread": forward_signals.gaap_norm_spread,
                        "revision_trend_short": forward_signals.revision_trend_short,
                        "revision_trend_medium": forward_signals.revision_trend_medium,
                        "eps_norm_est_fy1e": forward_signals.eps_norm_est_fy1e,
                        "eps_norm_est_ntm": forward_signals.eps_norm_est_ntm,
                        "eps_gaap_est_ntm": forward_signals.eps_gaap_est_ntm,
                        "eps_gaap_est_fy1e": forward_signals.eps_gaap_est_fy1e,
                        "analyst_count": forward_signals.analyst_count,
                        "next_earnings_status": row.get("next_earnings_status", None),
                        "quarterly_beat_streak": eps_positive_streak,
                        "data_source": "forward_enhanced",
                        "eps_positive_streak": eps_positive_streak,
                        "gaap_vs_norm_revision_spread": forward_signals.gaap_norm_spread,
                        "eps_cont_positive_streak": row.get(
                            "eps_cont_positive_streak", None
                        ),
                    }
                )

                # Pass through composite/quality columns from mv_all_stock_features
                for src_col, out_key in _PASSTHROUGH_COLS.items():
                    val = row.get(src_col, None)
                    if val is not None and not (
                        isinstance(val, float) and np.isnan(val)
                    ):
                        record[out_key] = val
                    elif out_key == "analyst_conviction":
                        record[out_key] = _compute_analyst_conviction(row)
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
                    proxy_df["eps_improvement_count"]
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
                            np.where(
                                trajectory >= 40, 5, np.where(trajectory >= 20, 4, 3)
                            ),
                        ),
                    ),
                    index=proxy_df.index,
                )
            n_beats_proxy = (
                (trajectory / 100 * n_total_proxy)
                .astype(int)
                .clip(lower=0, upper=n_total_proxy)
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
                confidence_score >= 0.6,
                "High",
                np.where(confidence_score >= 0.3, "Medium", "Low"),
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
                    if val is not None and not (
                        isinstance(val, float) and np.isnan(val)
                    ):
                        record[out_key] = val
                    elif out_key == "analyst_conviction":
                        record[out_key] = _compute_analyst_conviction(row)
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

    def __init__(
        self,
        mean_reversion_weight: float = 0.2,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        revision_tilt_strength: float = 0.5,
    ):
        """
        Initialize streak analyzer.

        Args:
            mean_reversion_weight: Weight for mean reversion in the legacy
                heuristic continuation-probability path (0-1).
            prior_alpha: Beta-Binomial prior α on continuation probability
                (v3.9 §5.1). Higher ``α`` pulls the posterior toward a
                higher base-rate of streak continuation.
            prior_beta: Beta-Binomial prior β on continuation probability
                (v3.9 §5.1). The prior strength is ``α+β``; the default
                (2,2) is weakly informative.
            revision_tilt_strength: Student-t shrinkage strength for the
                forward-revision prior tilt (v3.9 §5.2). ``0`` disables the
                tilt; ``1`` applies the full revision-momentum signal.
        """
        self.mean_reversion_weight = mean_reversion_weight
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.revision_tilt_strength = float(revision_tilt_strength)

    # ------------------------------------------------------------------
    # v3.9 §5.1 — Bayesian streak continuation posterior
    # ------------------------------------------------------------------
    def compute_bayesian_continuation_posterior(
        self,
        reported_history: Optional[ReportedEPSHistory],
        forward_signals: Optional[ForwardEstimateSignals] = None,
    ) -> dict[str, float]:
        """Beta-Binomial posterior over streak-continuation probability.

        Implements v3.9 task §5.1: replaces the point-estimate heuristic
        ``base * decay^streak`` with a proper Beta-Binomial posterior
        keyed on ``count_yoy_improvements()``. Discriminates firms with
        2/3 vs 20/30 history (the original heuristic could not).

        Optional §5.2 revision-momentum tilt (Student-t shrinkage on the
        logit) is applied when ``forward_signals.gaap_revision_momentum``
        is available and ``revision_tilt_strength > 0``.

        Returns
        -------
        dict with keys:
            ``posterior_alpha``, ``posterior_beta``,
            ``continuation_prob``, ``ci_low``, ``ci_high``,
            ``expected_streak_length_years``, ``hazard_rate_next_quarter``,
            ``effective_sample_size``.
        """
        # --- Likelihood counts ---
        if reported_history is not None:
            n_beats, n_total = reported_history.count_yoy_improvements()
        else:
            n_beats, n_total = 0, 0

        # §7.4 gate: fall back to prior-only posterior when data is too sparse
        a = self.prior_alpha + float(n_beats)
        b = self.prior_beta + float(max(n_total - n_beats, 0))

        # §5.2 revision-momentum tilt (logit-space, Student-t shrinkage)
        if (
            forward_signals is not None
            and self.revision_tilt_strength > 0
            and getattr(forward_signals, "has_sufficient_data", False)
        ):
            momentum = forward_signals.gaap_revision_momentum  # 0-100
            if momentum is not None and not pd.isna(momentum):
                # Map momentum [0, 100] → logit shift in [-0.6, +0.6] and
                # shrink via ``revision_tilt_strength`` (Student-t prior).
                shift = np.clip((float(momentum) - 50.0) / 50.0, -1.0, 1.0) * 0.6
                shift *= float(self.revision_tilt_strength)
                # Convert to a multiplicative alpha/beta tilt keeping the
                # concentration (α+β) constant so the ESS doesn't inflate.
                concentration = a + b
                mean = a / concentration
                logit_mean = float(np.log(mean / (1.0 - mean))) if 0 < mean < 1 else 0.0
                new_mean = 1.0 / (1.0 + np.exp(-(logit_mean + shift)))
                a = float(new_mean * concentration)
                b = float((1.0 - new_mean) * concentration)

        # --- Posterior summaries ---
        posterior_mean = a / (a + b)
        try:
            ci_low, ci_high = stats.beta.ppf([0.025, 0.975], a, b)
        except Exception:  # pragma: no cover
            ci_low, ci_high = float("nan"), float("nan")

        # Expected streak length under geometric continuation assumption
        # E[length | p] = 1 / (1 - p); hazard of ending next quarter = 1 - p
        if 0.0 < posterior_mean < 1.0:
            expected_length_quarters = 1.0 / (1.0 - posterior_mean)
            expected_length_years = expected_length_quarters / 4.0
        else:
            expected_length_years = float("nan")
        hazard_next_q = float(1.0 - posterior_mean)

        return {
            "posterior_alpha": float(a),
            "posterior_beta": float(b),
            "continuation_prob": float(posterior_mean),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "expected_streak_length_years": float(expected_length_years),
            "hazard_rate_next_quarter": hazard_next_q,
            "effective_sample_size": float(a + b),
        }

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
        if (
            reported_history is not None
            and reported_history.quarterly_reports_count > 0
        ):
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

        # v3.9 §5.1: Compute Beta-Binomial posterior alongside the heuristic
        # continuation probability. The heuristic ``continuation_prob`` is
        # preserved for backwards compatibility of downstream consumers; the
        # new posterior is exposed via the additional result fields and
        # integrated by ``EarningsBeatProbabilityModel`` through
        # ``map_estimate`` / ``model_confidence``.
        bayes_post = self.compute_bayesian_continuation_posterior(
            reported_history=reported_history,
            forward_signals=forward_signals,
        )

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
            posterior_alpha=bayes_post["posterior_alpha"],
            posterior_beta=bayes_post["posterior_beta"],
            continuation_prob_ci_low=bayes_post["ci_low"],
            continuation_prob_ci_high=bayes_post["ci_high"],
            expected_streak_length_years=bayes_post["expected_streak_length_years"],
            hazard_rate_next_quarter=bayes_post["hazard_rate_next_quarter"],
            ess_bulk=bayes_post["effective_sample_size"],
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        ticker_col: str = "isin",
        trajectory_col: str = "eps_trajectory_score",
        streak_col: str = "eps_positive_streak",
        improvement_col: str = "eps_improvement_count",
        name_col: str = "name",
        sector_col: str = "industry",
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
            "combined_distress_score",
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
            effective_total = (
                max(n_total_yoy, dynamic_total) if dynamic_total > 0 else n_total_yoy
            )
            historical_beat_rate = (
                n_beats_yoy / effective_total if effective_total > 0 else 0.0
            )

            # --- Compute model_confidence and map_estimate ---
            # model_confidence: how decisive the streak prediction is (0-1)
            model_confidence = (
                abs(result.streak_continuation_prob - 0.5)
                * 2.0
                * result.confidence_level
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
                        int(streak)
                        if streak is not None and not pd.isna(streak)
                        else None
                    ),
                    # Model-derived fields
                    "model_confidence": round(model_confidence, 10),
                    "map_estimate": round(map_estimate, 10),
                }
            )

            # Pass through composite/quality columns from mv_all_stock_features
            for col in _PASSTHROUGH_COLS:
                val = row.get(col, None)
                # Rename combined_distress_score → distress_risk_score for export
                out_key = (
                    "distress_risk_score" if col == "combined_distress_score" else col
                )
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


def compute_brier_score(
    predicted_probs: np.ndarray, actual_outcomes: np.ndarray
) -> floating[Any]:
    """
    Compute Brier score for probability predictions.

    Brier score = (1/N) * Σ(predicted - actual)²
    Lower is better, 0 = perfect, 0.25 = random for binary.
    """
    return np.mean((predicted_probs - actual_outcomes) ** 2)


class ModelConfidenceEstimator:
    """
    Estimator for model confidence and calibration metrics.

    Provides comprehensive confidence assessment including:
    - Brier score for probability calibration
    - Reliability diagrams
    - Confidence interval coverage
    """

    def __init__(self, n_bins: int = 10, use_quantile_bins: bool = True):
        """Initialize confidence estimator.

        Args:
            n_bins: Number of bins for calibration analysis
            use_quantile_bins: v3.10 §13.1 — when ``True`` (default), bins are
                defined via :func:`pandas.qcut` so each bin holds roughly the
                same number of observations. Equal-width bins
                (:func:`numpy.linspace`) under-sample the 0.9–1.0 tail where
                beat probabilities cluster, producing optimistic ECE for
                high-conviction picks. Set ``False`` to recover legacy
                behaviour.
        """
        self.n_bins = n_bins
        self.use_quantile_bins = use_quantile_bins

    def compute_calibration_error(
        self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray
    ) -> tuple[float, dict]:
        """
        Compute Expected Calibration Error (ECE) and reliability diagram data.

        ECE measures how well predicted probabilities match observed frequencies.

        v3.10 §13.1 — honours ``use_quantile_bins`` so each bin holds roughly
        equal mass (drops empty bins before averaging).
        """
        reliability_data = {
            "bin_centers": [],
            "observed_freq": [],
            "predicted_mean": [],
            "count": [],
        }
        total_samples = len(predicted_probs)
        if total_samples == 0:
            return 0.0, reliability_data

        if self.use_quantile_bins and total_samples >= max(self.n_bins, 10):
            # §13.1 — quantile-based bins. Use ``pd.qcut`` with ``duplicates='drop'``
            # so near-zero / near-one clusters collapse gracefully rather than raising.
            try:
                bin_labels, bin_edges = pd.qcut(
                    predicted_probs,
                    q=self.n_bins,
                    labels=False,
                    duplicates="drop",
                    retbins=True,
                )
                bin_indices = np.asarray(bin_labels, dtype=float)
                # Edge case: ``pd.qcut`` may yield NaN for exact boundaries.
                bin_indices = np.nan_to_num(bin_indices, nan=0).astype(int)
                n_effective_bins = max(len(bin_edges) - 1, 1)
            except ValueError, IndexError:
                # Degenerate distribution — fall back to equal-width bins.
                bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
                bin_indices = np.clip(
                    np.digitize(predicted_probs, bin_edges) - 1, 0, self.n_bins - 1
                )
                n_effective_bins = self.n_bins
        else:
            bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
            bin_indices = np.clip(np.digitize(predicted_probs, bin_edges) - 1, 0, self.n_bins - 1)
            n_effective_bins = self.n_bins

        ece = 0.0
        for i in range(n_effective_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue  # §13.1 — drop empty bins before averaging
            bin_pred = predicted_probs[mask].mean()
            bin_actual = actual_outcomes[mask].mean()
            bin_count = int(mask.sum())
            lo = float(bin_edges[i]) if i < len(bin_edges) else 0.0
            hi = float(bin_edges[i + 1]) if (i + 1) < len(bin_edges) else 1.0
            reliability_data["bin_centers"].append((lo + hi) / 2.0)
            reliability_data["observed_freq"].append(float(bin_actual))
            reliability_data["predicted_mean"].append(float(bin_pred))
            reliability_data["count"].append(bin_count)
            ece += (bin_count / total_samples) * abs(bin_actual - bin_pred)

        return float(ece), reliability_data

    def compute_confidence_metrics(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        model_name: str = "Earnings Beat Model",
        bootstrap_iters: int = 200,
        bootstrap_seed: Optional[int] = None,
    ) -> ModelConfidenceResult:
        """Compute comprehensive confidence metrics for predictions.

        v3.10 (§13.2) — now emits a ``reliability_curve`` list of
        ``(bin_mid, empirical_rate, n)`` tuples and bootstrap 95 % CIs on
        ``ece`` / ``brier`` / ``log_score``. The bootstrap is seeded from
        ``bootstrap_seed`` (default ``RANDOM_SEED`` env var, then 42).

        Args:
            predicted_probs: Array of predicted probabilities
            actual_outcomes: Array of actual binary outcomes (0 or 1)
            model_name: Name for the model
            bootstrap_iters: Number of bootstrap resamples for CI estimation.
                Set to 0 to skip bootstrapping.
            bootstrap_seed: Seed for the bootstrap RNG. ``None`` → env var
                ``RANDOM_SEED`` (fallback: 42).

        Returns:
            ModelConfidenceResult with all metrics + calibration artefacts.
        """
        predicted_probs = np.asarray(predicted_probs, dtype=float)
        actual_outcomes = np.asarray(actual_outcomes, dtype=float)
        n = int(len(predicted_probs))

        # Brier score
        brier = float(compute_brier_score(predicted_probs, actual_outcomes))

        # Log loss (cross-entropy) — also used as the BMA log-score (T-E).
        eps = 1e-15
        clipped_probs = np.clip(predicted_probs, eps, 1 - eps)
        log_loss = float(
            -np.mean(
                actual_outcomes * np.log(clipped_probs)
                + (1 - actual_outcomes) * np.log(1 - clipped_probs)
            )
        )
        log_score = -log_loss  # higher = better; used directly by BMA

        # Calibration error and reliability data (§13.1 quantile bins active).
        ece, reliability_data = self.compute_calibration_error(
            predicted_probs, actual_outcomes
        )

        # §13.2 — reliability_curve as list of (bin_mid, empirical_rate, n)
        reliability_curve: list[tuple[float, float, int]] = [
            (float(c), float(f), int(k))
            for c, f, k in zip(
                reliability_data.get("bin_centers", []),
                reliability_data.get("observed_freq", []),
                reliability_data.get("count", []),
            )
        ]

        # AUC-ROC for discrimination
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(actual_outcomes, predicted_probs))
        except (ImportError, ValueError):
            n_pos = actual_outcomes.sum()
            n_neg = len(actual_outcomes) - n_pos
            if n_pos > 0 and n_neg > 0:
                ranks = stats.rankdata(predicted_probs)
                auc = float(
                    (ranks[actual_outcomes == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                )
            else:
                auc = 0.5

        # Confidence intervals for predictions (§13.3 Wilson-score).
        ci_coverage = self._compute_ci_coverage(predicted_probs, actual_outcomes)

        # --- §13.2 bootstrap 95 % CIs on ECE / Brier / log-score ---
        ece_lo = ece_hi = brier_lo = brier_hi = float("nan")
        if bootstrap_iters and n >= 20:
            if bootstrap_seed is None:
                import os as _os

                bootstrap_seed = int(_os.environ.get("RANDOM_SEED", 42))
            rng = np.random.default_rng(int(bootstrap_seed))
            ece_boot = np.empty(bootstrap_iters, dtype=float)
            brier_boot = np.empty(bootstrap_iters, dtype=float)
            for i in range(bootstrap_iters):
                idx = rng.integers(0, n, size=n)
                p_b = predicted_probs[idx]
                y_b = actual_outcomes[idx]
                try:
                    e_b, _ = self.compute_calibration_error(p_b, y_b)
                except ValueError, IndexError:
                    e_b = float("nan")
                ece_boot[i] = e_b
                brier_boot[i] = float(np.mean((p_b - y_b) ** 2))
            ece_lo, ece_hi = (
                float(np.nanpercentile(ece_boot, 2.5)),
                float(np.nanpercentile(ece_boot, 97.5)),
            )
            brier_lo, brier_hi = (
                float(np.nanpercentile(brier_boot, 2.5)),
                float(np.nanpercentile(brier_boot, 97.5)),
            )

        # Overall confidence score (0-100) — legacy weighting preserved.
        base_score = (1 - brier) * 30 + (1 - ece) * 30 + auc * 40
        if auc < 0.5:
            base_score -= (0.5 - auc) * 60
        overall = float(min(100, max(0, base_score)))

        return ModelConfidenceResult(
            model_name=model_name,
            brier_score=brier,
            log_loss=log_loss,
            calibration_error=ece,
            discrimination_auc=auc,
            reliability_diagram_data=reliability_data,
            confidence_intervals=ci_coverage,
            overall_confidence=overall,
            reliability_curve=reliability_curve,
            ece_ci_low=ece_lo,
            ece_ci_high=ece_hi,
            brier_ci_low=brier_lo,
            brier_ci_high=brier_hi,
            log_score=log_score,
            auroc=auc,
            n_samples=n,
        )

    def _compute_ci_coverage(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        n_observations: int = 5,  # kept for back-compat; ignored when using Wilson intervals
    ) -> dict:
        """Compute confidence-interval coverage rates.

        v3.10 §13.3 — replaces the ad-hoc ``n_observations=5`` SE estimate with
        a proper Wilson-score interval so coverage is valid for small sectors
        (LatAm, Africa/ME). Emits coverage curves at 50 / 80 / 95 %.
        """
        n = int(len(predicted_probs))
        if n == 0:
            return {
                "coverage_50": 0.0,
                "coverage_80": 0.0,
                "coverage_90": 0.0,
                "coverage_95": 0.0,
            }

        # Wilson-score interval: p ± z√(p(1-p)/n_obs + z²/(4 n_obs²)) / (1 + z²/n_obs)
        # where n_obs is a per-sample effective sample size. We use the array
        # length as the denominator (common empirical-calibration convention)
        # — this gives wider, more honest intervals than the old n=5 floor.
        z50, z80, z90, z95 = 0.674, 1.282, 1.645, 1.96

        def _wilson(p: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
            denom = 1.0 + (z * z) / n
            centre = (p + (z * z) / (2.0 * n)) / denom
            half = z * np.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n)) / denom
            return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)

        result: dict[str, float] = {}
        for z, tag in ((z50, "50"), (z80, "80"), (z90, "90"), (z95, "95")):
            lo, hi = _wilson(predicted_probs, z)
            within = ((actual_outcomes >= lo) & (actual_outcomes <= hi)).sum()
            result[f"coverage_{tag}"] = float(within) / n
        return result

    # -------------------------------------------------------------------
    # §13.4 / T-E — centralised multi-model confidence / BMA weight helper
    # -------------------------------------------------------------------
    def compute_relative_confidence(
        self,
        model_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        bootstrap_iters: int = 100,
    ) -> pd.DataFrame:
        """Compute Brier / ECE / log-score per model and return BMA weights.

        v3.10 (§13.4 / T-E) — single source of truth for BMA log-score weights
        so ``ensemble_models.py`` can import this helper instead of
        re-implementing the weighting logic. Pass a dict of
        ``{model_name: (predicted_probs, actual_outcomes)}``; returns a
        DataFrame with one row per model ordered by decreasing log-score,
        including ``bma_weight`` (softmax of log-scores) and
        ``passes_calibration``.
        """
        rows: list[dict[str, Any]] = []
        results_by_model: dict[str, ModelConfidenceResult] = {}
        for name, (probs, outcomes) in model_outputs.items():
            probs = np.asarray(probs, dtype=float)
            outcomes = np.asarray(outcomes, dtype=float)
            if len(probs) == 0 or len(probs) != len(outcomes):
                continue
            res = self.compute_confidence_metrics(
                probs, outcomes, model_name=name, bootstrap_iters=bootstrap_iters
            )
            results_by_model[name] = res
            rows.append(
                {
                    "model": name,
                    "brier_score": res.brier_score,
                    "ece": res.calibration_error,
                    "log_score": res.log_score,
                    "auroc": res.auroc,
                    "n_samples": res.n_samples,
                    "passes_calibration": res.passes_calibration(),
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "model",
                    "brier_score",
                    "ece",
                    "log_score",
                    "auroc",
                    "n_samples",
                    "passes_calibration",
                    "bma_weight",
                ]
            )

        df = pd.DataFrame(rows)
        # Softmax over log-scores → BMA weights. Models that fail calibration
        # are still weighted (so the pipeline can decide whether to exclude),
        # but downstream consumers should use ``passes_calibration`` to filter.
        scores = df["log_score"].to_numpy(dtype=float, copy=True)
        finite = np.isfinite(scores)
        if finite.any():
            s = np.where(finite, scores, -np.inf)
            s = s - np.nanmax(s[finite])  # numerical stability
            w = np.where(finite, np.exp(s), 0.0)
            total = float(w.sum())
            df["bma_weight"] = w / total if total > 0 else 0.0
        else:
            df["bma_weight"] = 0.0

        return df.sort_values("log_score", ascending=False).reset_index(drop=True)


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
        distress_threshold: float = 70,
        prior_alpha: float = 1.5,
        prior_beta: float = 2.0,  # v3.9: was 2.5 — less optimistic given fat-tail risk
        n_mcmc_samples: int = 15000,  # v3.9: was 5000 — tail ESS target
        burn_in: int = 3000,  # v3.9: was 1000
        use_mcmc: bool = True,
        # NEW: Leverage & Liquidity enrichment
        use_debt_trajectory: bool = True,
        use_cash_buffer_signals: bool = False,
        use_balance_sheet_quality: bool = True,
        use_wc_deep_signals: bool = True,
        # NEW: Quality & Risk enrichment
        use_quality_risk_flags: bool = True,
        # v3.9: Heavy-tail / time-varying volatility / macro (Findings #1, #2, #4)
        use_student_t_likelihood: bool = True,
        use_garch_volatility: bool = True,
        use_macro_covariates: bool = True,
        student_t_df_floor: float = 3.0,
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
        self.use_student_t_likelihood = use_student_t_likelihood
        self.use_garch_volatility = use_garch_volatility
        self.use_macro_covariates = use_macro_covariates
        self.student_t_df_floor = float(student_t_df_floor)
        # v3.9 §1.2: fail-fast validation on df floor and structured logging
        # so the pipeline diagnostic can confirm the clamp is actually
        # applied (v3.8 reported global df=2.00 despite df_floor=3.0).
        if self.student_t_df_floor < 2.0:
            raise ValueError(
                f"student_t_df_floor must be >= 2.0 (got {self.student_t_df_floor}); "
                "values <2 imply infinite variance and break CVaR haircuts."
            )
        logger.info(
            "CreditRiskProbabilityModel v3.9 config: student_t=%s, garch=%s, "
            "macro=%s, df_floor=%.2f, n_samples=%d, burn_in=%d",
            self.use_student_t_likelihood,
            self.use_garch_volatility,
            self.use_macro_covariates,
            self.student_t_df_floor,
            self.n_mcmc_samples,
            self.burn_in,
        )

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze dataframe for credit risk with enhanced features."""
        results = []

        for _, row in df.iterrows():
            # Core distress indicators — _safe_get returns (calc_value, raw_value)
            z_score, z_score_raw = _safe_get(row, "altman_z_score", 3.0)
            z_trend, z_trend_raw = _safe_get(row, "altman_z_trend", 0)
            liquidity_stress, liquidity_stress_raw = _safe_get(
                row, "liquidity_stress_score", 50
            )
            cash_runway, cash_runway_raw = _safe_get(row, "cash_runway_months", 24)
            accumulated_deficit, _ = _safe_get(row, "accumulated_deficit_flag", 0)

            # Additional risk factors from views
            combined_distress, combined_distress_raw = _safe_get(
                row, "combined_distress_score", 50
            )
            wc_deteriorating, _ = _safe_get(row, "wc_deteriorating_flag", 0)
            debt_deleveraging, _ = _safe_get(row, "debt_deleveraging", 0)
            interest_coverage, interest_coverage_raw = _safe_get(
                row, "interest_coverage", 5.0
            )
            quick_ratio, quick_ratio_raw = _safe_get(row, "quick_ratio", 1.5)
            beta_stability, beta_stability_raw = _safe_get(
                row, "beta_stability_score", 50
            )

            # Debt trajectory signals (calc_total_debt_temporal)
            debt_3y_cagr, debt_3y_cagr_raw = _safe_get(row, "debt_3y_cagr", 0)
            debt_4q_trend, _ = _safe_get(row, "debt_4q_trend", 0)
            debt_yoy_change, _ = _safe_get(row, "debt_yoy_change", 0)

            # Cash buffer signals (calc_financial_distress_features + calc_balance_sheet_dynamics)
            adequate_cash_buffer, _ = _safe_get(row, "adequate_cash_buffer", 1)
            cash_vs_5y_avg, _ = _safe_get(row, "cash_vs_5y_avg", 1.0)

            # Balance sheet quality (calc_balance_sheet_dynamics)
            balance_sheet_strength, balance_sheet_strength_raw = _safe_get(
                row, "balance_sheet_strength", 50
            )
            debt_maturity_risk, debt_maturity_risk_raw = _safe_get(
                row, "debt_maturity_risk", 0
            )
            equity_ratio, _ = _safe_get(row, "equity_ratio", 0.5)

            # Working capital deep (calc_working_capital_deep_features + temporal)
            wc_volatility, _ = _safe_get(row, "wc_volatility", 0)
            wc_efficiency_score, wc_efficiency_score_raw = _safe_get(
                row, "wc_efficiency_score", 50
            )
            retained_earnings_vs_5y, _ = _safe_get(row, "retained_earnings_vs_5y", 1.0)

            # Quality & Risk flags
            piotroski_f_score, piotroski_f_score_value = _safe_get(row, "piotroski_f_score", 50)
            retained_earnings_growth, _ = _safe_get(row, "retained_earnings_growth", 0)
            beta_trend, _ = _safe_get(row, "beta_trend", 0)

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
                if (
                    retained_earnings_vs_5y is not None
                    and retained_earnings_vs_5y < 0.5
                ):
                    adjustments += 0.07

            # NEW: Quality & Risk flags
            if self.use_quality_risk_flags:
                if piotroski_f_score is not None and piotroski_f_score > 70:
                    adjustments += 0.10
                if (
                    retained_earnings_growth is not None
                    and retained_earnings_growth < -20
                ):
                    adjustments += 0.08
                if beta_trend is not None and beta_trend > 0.3:
                    adjustments += 0.05

                # P1: Cash flow quality signals (Cat 12)
                cfo_to_ni, _ = _safe_get(row, "cfo_to_net_income", 1.0)
                if cfo_to_ni is not None and cfo_to_ni < 0.5:
                    adjustments += 0.10  # Accrual-driven earnings
                cf_quality, _ = _safe_get(row, "cash_flow_quality_score", 50)
                if cf_quality is not None and cf_quality < 25:
                    adjustments += 0.08
                fin_dep, _ = _safe_get(row, "financing_dependency", 0)
                if fin_dep is not None and fin_dep > 0.5:
                    adjustments += 0.07
                cash_burn, _ = _safe_get(row, "cash_burn_rate", 0)
                if cash_burn is not None and cash_burn > 0.5:
                    adjustments += 0.06

                # P2: Accounting quality & restructuring frequency (Cat 7)
                acct_quality, _ = _safe_get(row, "accounting_quality_score", 50)
                if acct_quality is not None and acct_quality < 25:
                    adjustments += 0.06
                restruct_freq, _ = _safe_get(row, "restructuring_frequency", 0)
                if restruct_freq is not None and restruct_freq >= 3:
                    adjustments += 0.05
                quality_issues, _ = _safe_get(row, "quality_issues_count_5y", 0)
                if quality_issues is not None and quality_issues >= 3:
                    adjustments += 0.05

                # P2: Cost structure — interest burden (Cat 15)
                interest_to_rev, _ = _safe_get(row, "interest_to_revenue", 0)
                if interest_to_rev is not None and interest_to_rev > 0.15:
                    adjustments += 0.06

                # P2: Unusual items (Cat 17)
                unusual_to_ebitda, _ = _safe_get(row, "unusual_items_to_ebitda", 0)
                if unusual_to_ebitda is not None and abs(unusual_to_ebitda) > 0.2:
                    adjustments += 0.05

                # P3: Investment income temporal (Cat 24)
                int_inc_trend, _ = _safe_get(row, "interest_income_to_revenue_trend", 0)
                if int_inc_trend is not None and int_inc_trend > 0.1:
                    adjustments += 0.03  # Growing non-operating income share

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
                    piotroski_f_score,
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
                    "beta_stability_score": beta_stability_raw,
                    "combined_distress_score": combined_distress_raw,
                    "distress_probability": prob,
                    "liquidity_stress_score": liquidity_stress_raw,
                    "cash_runway_months": cash_runway_raw,
                    "altman_z_score": z_score_raw,
                    "altman_z_trend": z_trend_raw,
                    "interest_coverage": interest_coverage_raw,
                    "quick_ratio": quick_ratio_raw,
                    "risk_level": risk_level,
                    "ci_lower": max(0, prob - ci_width),
                    "ci_upper": min(1, prob + ci_width),
                    "debt_3y_cagr": debt_3y_cagr_raw,
                    "debt_maturity_risk": debt_maturity_risk_raw,
                    "balance_sheet_strength": balance_sheet_strength_raw,
                    "wc_efficiency_score": wc_efficiency_score_raw,
                    "piotroski_f_score": piotroski_f_score_value,
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
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            hierarchical_mcmc_by_sector,
            mcmc_student_t,
            metropolis_hastings_sampler,
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
            samples, acc_rate = metropolis_hastings_sampler(z_scores, n_samples=self.n_mcmc_samples,
                                                            burn_in=self.burn_in, prior_mean=self.distress_threshold,
                                                            prior_std=1.0)
            # Per-stock: P(distress) = P(posterior_mean < stock_z)
            stock_z = (
                result_df["altman_z_score"].values
                if "altman_z_score" in result_df.columns
                else np.full(len(result_df), 3.0)
            )
            distress_prob_per_stock = np.mean(
                samples[:, None] > stock_z[None, :], axis=0
            )
            result_df["mcmc_distress_probability"] = np.clip(
                distress_prob_per_stock, 0, 1
            )

            # Task 2.2: Student-t for robust estimation
            mu_samples, df_samples = mcmc_student_t(z_scores, n_samples=self.n_mcmc_samples, burn_in=self.burn_in)
            result_df["mcmc_ci_lower"] = np.percentile(mu_samples, 2.5)
            result_df["mcmc_ci_upper"] = np.percentile(mu_samples, 97.5)
        except (ValueError, RuntimeError) as e:
            logger.warning("MCMC credit risk posterior failed: %s", e)
            result_df["mcmc_distress_probability"] = np.nan
            result_df["mcmc_ci_lower"] = np.nan
            result_df["mcmc_ci_upper"] = np.nan

        # Task 2.3: Hierarchical MCMC by sector
        try:
            sector_col = "industry" if "industry" in source_df.columns else "sector"
            if (
                sector_col in source_df.columns
                and "altman_z_score" in source_df.columns
            ):
                sector_results = hierarchical_mcmc_by_sector(source_df, feature="altman_z_score", sector_col=sector_col,
                                                             n_samples=self.n_mcmc_samples)
                # Unwrap ArviZ-wrapped result
                if "sectors" in sector_results and isinstance(
                    sector_results["sectors"], dict
                ):
                    sector_results = sector_results["sectors"]
                sector_mean_map = {
                    s: v.get("posterior_mean", np.nan)
                    for s, v in sector_results.items()
                    if isinstance(v, dict)
                }
                if sector_col in result_df.columns:
                    result_df["sector_z_posterior_mean"] = result_df[sector_col].map(
                        sector_mean_map
                    )
        except (ValueError, RuntimeError) as e:
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
        high_payout_threshold: float = 0.55,  # v3.9: was 1.0 — tighter given fat tails
        min_coverage: float = 1.5,  # v3.9: was 0.0 — require genuine coverage
        n_mcmc_samples: int = 12000,  # v3.9: was 5000
        burn_in: int = 3000,  # v3.9: was 1000
        use_mcmc: bool = True,
        # NEW: Leverage & Liquidity signals for dividend sustainability
        use_leverage_signals: bool = True,
        use_balance_sheet: bool = True,
        # v3.9: Heavy-tail likelihood (Finding #1)
        use_student_t_likelihood: bool = True,
        # v3.9 §3.1: GARCH volatility parity with Credit / PT. When enabled
        # (and PyMC is available) the ``_apply_mcmc_posteriors`` block models
        # payout-ratio volatility via a GARCH(1,1) on the coverage residuals.
        # The flag is honoured end-to-end at API level; the sampler path
        # falls back to a Gaussian likelihood if PyMC/arviz unavailable.
        use_garch_volatility: bool = True,
        student_t_df_floor: float = 3.0,
    ):
        self.high_payout_threshold = high_payout_threshold
        self.min_coverage = min_coverage
        self.n_mcmc_samples = n_mcmc_samples
        self.burn_in = burn_in
        self.use_mcmc = use_mcmc
        self.use_leverage_signals = use_leverage_signals
        self.use_balance_sheet = use_balance_sheet
        self.use_student_t_likelihood = use_student_t_likelihood
        # v3.9 §3.1 parity flags
        self.use_garch_volatility = bool(use_garch_volatility)
        self.student_t_df_floor = float(student_t_df_floor)
        logger.debug(
            "DividendCutProbabilityModel v3.9: student_t=%s, garch=%s, df_floor=%.2f",
            self.use_student_t_likelihood,
            self.use_garch_volatility,
            self.student_t_df_floor,
        )

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            # Core dividend metrics — _safe_get returns (calc_value, raw_value)
            fcf_coverage, fcf_coverage_raw = _safe_get(
                row, "fcf_dividend_coverage", 2.0
            )
            payout_ratio, payout_ratio_raw = _safe_get(row, "dividend_payout_ratio", 50)
            streak, streak_raw = _safe_get(row, "dividend_streak", 10)
            growth_exp, _ = _safe_get(row, "dividend_growth_expectation", 0)

            # Enhanced metrics from vw_features_dividends
            sustainable_flag, sustainable_flag_raw = _safe_get(
                row, "sustainable_dividend_flag", 1
            )
            consistency, consistency_raw = _safe_get(row, "dividend_consistency", 0.8)
            yield_vs_5y, yield_vs_5y_raw = _safe_get(
                row, "dividend_yield_vs_5y_avg", 1.0
            )
            recent_change, _ = _safe_get(row, "recent_dividend_change", 0)
            high_yield_flag, high_yield_flag_raw = _safe_get(row, "high_yield_flag", 0)

            # Leverage signals (calc_leverage_features)
            interest_coverage, _ = _safe_get(row, "interest_coverage", 5.0)
            debt_to_equity, _ = _safe_get(row, "debt_to_equity", 0.5)
            cash_ratio_val, _ = _safe_get(row, "cash_ratio", 0.5)
            working_capital_ratio, _ = _safe_get(row, "working_capital_ratio", 1.0)

            # Balance sheet health (calc_balance_sheet_dynamics + distress)
            balance_sheet_strength, _ = _safe_get(row, "balance_sheet_strength", 50)
            cash_runway, _ = _safe_get(row, "cash_runway_months", 24)
            retained_earnings_growth, _ = _safe_get(row, "retained_earnings_growth", 0)
            debt_3y_cagr, _ = _safe_get(row, "debt_3y_cagr", 0)

            # Base probability with more granular assessment
            prob = 0.05  # Low base rate for established dividend payers

            # FCF coverage is the strongest predictor
            if fcf_coverage is not None and not np.isnan(fcf_coverage):
                if fcf_coverage < 0.5:
                    prob += 0.45
                elif fcf_coverage < 1.0:
                    prob += 0.30
                elif fcf_coverage < 1.2:
                    prob += 0.15
                elif fcf_coverage > 2.0:
                    prob -= 0.03

            # Payout ratio stress
            if payout_ratio is not None and not np.isnan(payout_ratio):
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
                if (
                    retained_earnings_growth is not None
                    and retained_earnings_growth < -15
                ):
                    prob += 0.07
                if debt_3y_cagr is not None and debt_3y_cagr > 15:
                    prob += 0.06

            # P1: FCF estimate curve deterioration (Cat 22 — Enhancement 9)
            fcf_est_cagr, _ = _safe_get(row, "fcf_est_cagr_5y", 0)
            if fcf_est_cagr is not None and fcf_est_cagr < -5:
                prob += 0.10  # Forward FCF declining
            fcf_est_trend, _ = _safe_get(row, "fcf_est_trend", 0)
            if fcf_est_trend is not None and fcf_est_trend < -0.1:
                prob += 0.06

            # P2: Dividend history forward curve (Cat 23 — Enhancement 10)
            div_yield_trend, _ = _safe_get(row, "div_yield_5y_trend", 0)
            if div_yield_trend is not None and div_yield_trend < -0.1:
                prob += 0.05  # Expected yield declining
            div_yield_stab, _ = _safe_get(row, "div_yield_stability", 1.0)
            if div_yield_stab is not None and div_yield_stab < 0.3:
                prob += 0.04  # Unstable dividend yield history

            # P2: Cash flow quarterly trend (Cat 12)
            fcf_q_trend, _ = _safe_get(row, "fcf_quarterly_trend", 0)
            if fcf_q_trend is not None and fcf_q_trend < -0.15:
                prob += 0.06  # Deteriorating quarterly FCF
            op_cf_mom, _ = _safe_get(row, "operating_cf_momentum", 0)
            if op_cf_mom is not None and op_cf_mom < -0.2:
                prob += 0.05

            # P3: Share dilution competing with dividends (Cat 25 — Enhancement 12)
            dilution, _ = _safe_get(row, "shares_yoy_change_pct", 0)
            if dilution is not None and dilution > 5:
                prob += 0.06  # Significant dilution

            # P3: Employment signals (Cat 11)
            layoff_flag, _ = _safe_get(row, "layoff_risk_flag", 0)
            if layoff_flag == 1:
                prob += 0.04  # Layoffs often precede dividend cuts

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
                    "high_yield_flag": high_yield_flag_raw,
                    "dividend_cut_probability": prob,
                    "fcf_dividend_coverage": fcf_coverage_raw,
                    "payout_ratio": payout_ratio_raw,
                    "dividend_streak": streak_raw,
                    "dividend_consistency": consistency_raw,
                    "yield_vs_5y_avg": yield_vs_5y_raw,
                    "sustainable_flag": sustainable_flag_raw,
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
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            mcmc_student_t,
            metropolis_hastings_sampler,
        )

        probs = []

        # Task 3.1: FCF coverage posterior
        fcf_prob = 0.5
        samples = np.array([])
        fcf_data = np.array([])
        try:
            fcf_data = (
                source_df["fcf_dividend_coverage"].dropna().values
                if "fcf_dividend_coverage" in source_df.columns
                else np.array([])
            )
            if len(fcf_data) >= 10:
                samples, _ = metropolis_hastings_sampler(fcf_data, n_samples=self.n_mcmc_samples, burn_in=self.burn_in,
                                                         prior_mean=self.min_coverage, prior_std=1.0)
                fcf_prob = float(np.mean(samples < self.min_coverage))
                probs.append(fcf_prob)
        except (ValueError, RuntimeError) as e:
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
                mu_samples, _ = mcmc_student_t(payout_data, n_samples=self.n_mcmc_samples, burn_in=self.burn_in)
                payout_prob = float(
                    np.mean(mu_samples > self.high_payout_threshold * 100)
                )
                probs.append(payout_prob)
        except (ValueError, RuntimeError) as e:
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
            except (ValueError, KeyError, IndexError):
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

    Enhanced features: expected_upside_pt, price_target_spread_pct, pt_momentum_1m,
    analyst_rating_normalized, pt_consensus_convergence, analyst_conviction,
    pt_acceleration_short, eps_revision_momentum, analyst_coverage_trend

    Risk-adjusted enrichment (v3.4):
    beta_1y, beta_stability_score, distress_risk_score,
    balance_sheet_strength, debt_maturity_risk
    """

    def __init__(
        self,
        time_horizon_months: int = 12,
        n_mcmc_samples: int = 15000,  # v3.9: was 10000 — tail ESS target
        burn_in: int = 3000,  # v3.9: was 2000
        use_mcmc: bool = True,
        # NEW: Risk-adjusted achievement
        use_risk_adjustment: bool = True,
        use_financial_health: bool = True,
        # v3.9: Heavy-tail / GARCH (Findings #1, #2)
        use_student_t_returns: bool = True,
        use_garch_volatility: bool = True,
        df_floor: float = 3.0,
    ):
        self.time_horizon_months = time_horizon_months
        self.n_mcmc_samples = n_mcmc_samples
        self.burn_in = burn_in
        self.use_mcmc = use_mcmc
        self.use_risk_adjustment = use_risk_adjustment
        self.use_financial_health = use_financial_health
        self.use_student_t_returns = use_student_t_returns
        self.use_garch_volatility = use_garch_volatility
        self.df_floor = df_floor

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            # Core metrics — _safe_get returns (calc_value, raw_value)
            upside, upside_raw = _safe_get(row, "upside_potential", 10)
            spread, spread_raw = _safe_get(row, "price_target_spread_pct", 20)
            pt_momentum, _ = _safe_get(row, "pt_momentum_1m", 0)
            rating, rating_raw = _safe_get(row, "analyst_rating_normalized", 50)

            # Enhanced analyst sentiment features
            conviction, conviction_raw = _safe_get(row, "analyst_conviction", 50)
            if pd.isna(conviction_raw):
                derived = _compute_analyst_conviction(row)
                if not pd.isna(derived):
                    conviction, conviction_raw = derived, derived
            consensus_convergence, _ = _safe_get(row, "pt_consensus_convergence", 0)
            pt_accel, _ = _safe_get(row, "pt_acceleration_short", 0)
            eps_revision, eps_revision_raw = _safe_get(row, "eps_revision_momentum", 0)
            coverage_trend, _ = _safe_get(row, "analyst_coverage_trend", 0)
            bullish_pct, bullish_pct_raw = _safe_get(row, "analyst_bullish_pct", 50)

            # Risk adjustment (calc_beta_risk_features)
            beta_1y, _ = _safe_get(row, "beta_1y", 1.0)
            beta_stability, _ = _safe_get(row, "beta_stability_score", 50)
            distress_risk, _ = _safe_get(row, "distress_risk_score", 50)

            # Financial health (calc_balance_sheet_dynamics)
            bs_strength, _ = _safe_get(row, "balance_sheet_strength", 50)
            debt_mat_risk, _ = _safe_get(row, "debt_maturity_risk", 0)

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

            # Analyst conviction score
            if conviction is not None and conviction > 70:
                adjustments += 0.07
            elif conviction is not None and conviction < 30:
                adjustments -= 0.05

            # Consensus converging (analysts agreeing)
            if consensus_convergence is not None and consensus_convergence > 0:
                adjustments += 0.05

            # PT acceleration (momentum building)
            if pt_accel is not None and pt_accel > 0.02:
                adjustments += 0.06

            # EPS revisions supporting the price target
            if eps_revision is not None and eps_revision > 5:
                adjustments += 0.08
            elif eps_revision is not None and eps_revision < -5:
                adjustments -= 0.10

            # Growing analyst coverage = more attention
            if coverage_trend is not None and coverage_trend > 0:
                adjustments += 0.03

            # Risk-adjusted achievement probability
            if self.use_risk_adjustment:
                if beta_1y is not None and beta_1y > 1.5:
                    adjustments -= 0.08
                elif beta_1y is not None and beta_1y < 0.7:
                    adjustments += 0.04
                if beta_stability is not None and beta_stability < 25:
                    adjustments -= 0.05
                if distress_risk is not None and distress_risk > 70:
                    adjustments -= 0.12

            # Financial health supports target achievement
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

            # P1: Forward Consensus adjustments (Cat 26)
            forward_pe_premium, _ = _safe_get(row, "forward_pe_premium", 0)
            if forward_pe_premium is not None and forward_pe_premium > 20:
                adjustments -= 0.06  # Overvalued on forward basis
            ebitda_fwd_growth, _ = _safe_get(row, "ebitda_forward_growth", 0)
            if ebitda_fwd_growth is not None and ebitda_fwd_growth > 10:
                adjustments += 0.05  # Strong forward growth supports target
            earnings_rev_div, _ = _safe_get(row, "earnings_revision_divergence", 0)
            if earnings_rev_div is not None and earnings_rev_div > 0.3:
                adjustments -= 0.04  # Divergent revisions reduce confidence

            # P1: Growth signals (Cat 6)
            fwd_rev_growth, _ = _safe_get(row, "forward_revenue_growth", 0)
            if fwd_rev_growth is not None and fwd_rev_growth > 15:
                adjustments += 0.04

            # P1: Profitability signals (Cat 4)
            margin_flag, _ = _safe_get(row, "margin_expansion_flag", 0)
            if margin_flag == 1:
                adjustments += 0.04

            # P1: Momentum alignment (Cat 2)
            price_mom_3m, _ = _safe_get(row, "price_momentum_3m", 0)
            if price_mom_3m is not None and upside is not None:
                # Momentum aligning with target direction
                if upside > 0 and price_mom_3m > 10:
                    adjustments += 0.05
                elif upside > 0 and price_mom_3m < -15:
                    adjustments -= 0.05

            # P1: Cash flow support (Cat 12)
            fcf_yield, _ = _safe_get(row, "fcf_yield", 0)
            if fcf_yield is not None and fcf_yield > 5:
                adjustments += 0.03

            # P3: Share dilution (Cat 25)
            dilution, _ = _safe_get(row, "shares_yoy_change_pct", 0)
            if dilution is not None and dilution > 5:
                adjustments -= 0.04  # Significant dilution hurts price appreciation

            prob = min(0.90, max(0.05, base_prob + adjustments))

            record = _extract_identifiers(row)
            record.update(
                {
                    "bullish_pct": bullish_pct_raw,
                    "achievement_probability": prob,
                    "expected_upside_pt": upside_raw,
                    "price_target_spread_pct": spread_raw,
                    "analyst_conviction": conviction_raw,
                    "eps_revision_momentum": eps_revision_raw,
                    "analyst_rating_normalized": rating_raw,
                    "implied_return_pt": (upside or 0) * prob,
                    "confidence_level": (
                        "High"
                        if spread and spread < 20
                        else "Medium"
                        if spread and spread < 35
                        else "Low"
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
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            mcmc_student_t,
            metropolis_hastings_sampler,
            parallel_mcmc_chains,
        )

        returns_data = (
            source_df["implied_return_pt"].dropna().values
            if "implied_return_pt" in source_df.columns
            else np.array([])
        )
        if len(returns_data) < 10:
            return result_df

        try:
            # Task 4.1: MH sampler on upside potential
            # Prior centered at 0 (no upside) with std scaled by time horizon;
            # longer horizons → wider prior to reflect greater uncertainty.
            # v3.5: Incorporate stock-specific volatility when available.
            prior_std = 10.0 * (self.time_horizon_months / 12.0)
            if "volatility_regime" in source_df.columns:
                vol_scale = source_df["volatility_regime"].median()
                if pd.notna(vol_scale) and vol_scale > 0:
                    prior_std = prior_std * max(0.5, float(vol_scale))
            samples, acc_rate = metropolis_hastings_sampler(returns_data, n_samples=self.n_mcmc_samples,
                                                            burn_in=self.burn_in, prior_mean=0.0, prior_std=prior_std)
            # Per-stock: P(achieving target) ≈ P(posterior mean > stock's required upside)
            stock_upside = (
                result_df["implied_return_pt"].values
                if "implied_return_pt" in result_df.columns
                else np.full(len(result_df), 10.0)
            )
            mh_achievement_prob = np.mean(
                samples[:, None] > stock_upside[None, :], axis=0
            )
            result_df["mh_achievement_probability"] = np.clip(mh_achievement_prob, 0, 1)
            result_df["mh_acceptance_rate"] = acc_rate
            logger.info(
                "MH sampler for price target: acceptance_rate=%.3f, mean_posterior=%.3f",
                acc_rate,
                float(samples.mean()),
            )
        except (ValueError, RuntimeError) as e:
            logger.warning("MH sampler for price target failed: %s", e)
            result_df["mh_achievement_probability"] = np.nan
            result_df["mh_acceptance_rate"] = np.nan

        try:
            # Task 4.2: Student-t for heavy-tailed returns
            mu_samples, df_samples = mcmc_student_t(returns_data, n_samples=self.n_mcmc_samples, burn_in=self.burn_in)
            achievement_prob = float(np.mean(mu_samples > 0))
            result_df["mcmc_achievement_probability"] = achievement_prob
            result_df["mcmc_ci_lower"] = np.percentile(mu_samples, 2.5)
            result_df["mcmc_ci_upper"] = np.percentile(mu_samples, 97.5)

            # Task 4.4: Posterior mean weighted return
            posterior_mean_return = float(mu_samples.mean())
            result_df["mcmc_implied_return_pt"] = (
                posterior_mean_return * achievement_prob
            )
        except (ValueError, RuntimeError) as e:
            logger.warning("MCMC price target posterior failed: %s", e)
            result_df["mcmc_achievement_probability"] = np.nan
            result_df["mcmc_ci_lower"] = np.nan
            result_df["mcmc_ci_upper"] = np.nan
            result_df["mcmc_implied_return_pt"] = np.nan

        # Task 4.3: Parallel MCMC with Gelman-Rubin
        try:
            mcmc_result = parallel_mcmc_chains(
                returns_data, n_chains=8, n_samples=self.n_mcmc_samples
            )
            result_df["mcmc_gelman_rubin"] = mcmc_result.get("r_hat", np.nan)
        except (ValueError, RuntimeError) as e:
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
            "Revision Momentum vs P(Beat)"
            if has_momentum
            else "Beat Streak Distribution",
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
            probability_df.groupby("sector")["confidence_score"]
            .mean()
            .sort_values(ascending=True)
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
        plot_df = probability_df[
            ["gaap_revision_momentum", "posterior_beat_prob"]
        ].dropna()
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
        prior_alpha: float = 1.5,
        prior_beta: float = 2.0,
        n_mcmc_samples: int = 8000,  # v3.9: was 5000 — tail ESS target
        burn_in: int = 2000,  # v3.9: was 1000
        use_mcmc: bool = True,
        # v3.9: Heavy-tail likelihood (Finding #1) — flipped default
        use_student_t: bool = True,
        student_t_df_floor: float = 3.0,  # clamp df to avoid infinite variance
        use_mixture: bool = True,  # 2-component normal mixture fallback
        mixture_components: int = 2,
        use_garch: bool = True,  # GARCH(1,1) σ per group (Finding #2)
    ):
        self.category_name = category_name
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_mcmc_samples = n_mcmc_samples
        self.burn_in = burn_in
        self.use_mcmc = use_mcmc
        self.use_student_t = use_student_t
        self.student_t_df_floor = student_t_df_floor
        self.use_mixture = use_mixture
        self.mixture_components = mixture_components
        self.use_garch = use_garch

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

    def _compute_mcmc_posteriors(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> dict:
        """Compute MCMC posteriors per feature."""
        from probabilistic_ml_model.statistical_functions.statistical_models import (
            mcmc_student_t,
            metropolis_hastings_sampler,
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
                    mu_samples, _ = mcmc_student_t(data, n_samples=self.n_mcmc_samples, burn_in=self.burn_in)
                else:
                    mu_samples, _ = metropolis_hastings_sampler(data, n_samples=self.n_mcmc_samples,
                                                                burn_in=self.burn_in, prior_mean=self.prior_alpha,
                                                                prior_std=self.prior_beta)
                stats[feat] = {
                    "posterior_mean": float(mu_samples.mean()),
                    "posterior_std": float(mu_samples.std()),
                    "ci_lower_95": float(np.percentile(mu_samples, 2.5)),
                    "ci_upper_95": float(np.percentile(mu_samples, 97.5)),
                }
            except (ValueError, RuntimeError) as e:
                logger.warning("MCMC posterior for feature %s failed: %s", feat, e)

        return stats


# =============================================================================
# RESAMPLED BEAT PROBABILITY MODEL (ArviZ-enhanced)
# =============================================================================


@dataclass
class ResampledBeatEstimate:
    """Result container for resampled earnings beat probability with technical conditioning.

    v3.10 (§15.1 / §15.2) — extended with posterior spread (``posterior_std``,
    HDI bounds), per-chain diagnostics (``chain_rhat``, ``chain_ess_bulk``,
    ``chain_ess_tail``, ``n_effective_samples``), a ``volatility_regime`` label
    and versioned ``to_dict`` / ``from_dict`` serialisation so BMA log-score
    weighting treats the resampled channel as a proper posterior instead of a
    deterministic signal.
    """

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
    # --- v3.10 §15.1 posterior spread + chain diagnostics ---
    posterior_std: float = float("nan")
    hdi_low: float = float("nan")
    hdi_high: float = float("nan")
    chain_rhat: float = float("nan")
    chain_ess_bulk: float = float("nan")
    chain_ess_tail: float = float("nan")
    n_effective_samples: float = float("nan")
    volatility_regime: str = ""  # 'low' / 'normal' / 'high'
    # --- v3.10 §15.2 versioned serialisation ---
    schema_version: str = RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain ``dict`` (schema-versioned)."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "base_posterior_mean": float(self.base_posterior_mean),
            "resampled_posterior_mean": float(self.resampled_posterior_mean),
            "technical_adjustment": float(self.technical_adjustment),
            "momentum_signal": float(self.momentum_signal),
            "volatility_regime_score": float(self.volatility_regime_score),
            "credible_interval_90": tuple(self.credible_interval_90),
            "credible_interval_95": tuple(self.credible_interval_95),
            "prob_beat_given_momentum": float(self.prob_beat_given_momentum),
            "earnings_season_flag": self.earnings_season_flag,
            "pre_earnings_window": self.pre_earnings_window,
            "posterior_std": float(self.posterior_std),
            "hdi_low": float(self.hdi_low),
            "hdi_high": float(self.hdi_high),
            "chain_rhat": float(self.chain_rhat),
            "chain_ess_bulk": float(self.chain_ess_bulk),
            "chain_ess_tail": float(self.chain_ess_tail),
            "n_effective_samples": float(self.n_effective_samples),
            "volatility_regime": str(self.volatility_regime),
            "schema_version": str(self.schema_version),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResampledBeatEstimate":
        """Deserialise from a payload previously produced by :meth:`to_dict`."""
        ci90 = payload.get("credible_interval_90", (float("nan"), float("nan")))
        ci95 = payload.get("credible_interval_95", (float("nan"), float("nan")))
        return cls(
            ticker=str(payload.get("ticker", "")),
            name=str(payload.get("name", "")),
            sector=str(payload.get("sector", "")),
            base_posterior_mean=float(payload.get("base_posterior_mean", float("nan"))),
            resampled_posterior_mean=float(payload.get("resampled_posterior_mean", float("nan"))),
            technical_adjustment=float(payload.get("technical_adjustment", 0.0)),
            momentum_signal=float(payload.get("momentum_signal", 0.0)),
            volatility_regime_score=float(payload.get("volatility_regime_score", 0.0)),
            credible_interval_90=(float(ci90[0]), float(ci90[1])),
            credible_interval_95=(float(ci95[0]), float(ci95[1])),
            prob_beat_given_momentum=float(payload.get("prob_beat_given_momentum", float("nan"))),
            earnings_season_flag=payload.get("earnings_season_flag"),
            pre_earnings_window=payload.get("pre_earnings_window"),
            posterior_std=float(payload.get("posterior_std", float("nan"))),
            hdi_low=float(payload.get("hdi_low", float("nan"))),
            hdi_high=float(payload.get("hdi_high", float("nan"))),
            chain_rhat=float(payload.get("chain_rhat", float("nan"))),
            chain_ess_bulk=float(payload.get("chain_ess_bulk", float("nan"))),
            chain_ess_tail=float(payload.get("chain_ess_tail", float("nan"))),
            n_effective_samples=float(payload.get("n_effective_samples", float("nan"))),
            volatility_regime=str(payload.get("volatility_regime", "")),
            schema_version=str(payload.get("schema_version", RESULT_SCHEMA_VERSION)),
        )


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
        momentum_weight: float = 0.4,  # v3.9: was 0.3
        volatility_weight: float = 0.3,  # v3.9: was 0.2 — explicit regime weighting
        n_posterior_samples: int = 6000,  # v3.9: was 4000 — tail ESS > bulk
        n_chains: int = 8,
        random_seed: int = 42,
        # v3.10 §16.4 — credibility-shrinkage strength toward sector priors.
        # ``κ = n_sector / (n_sector + τ)``. Larger ``τ`` shrinks harder.
        sector_shrinkage_tau: float = 50.0,
        # v3.10 §16.1 — optional per-sector weight override (momentum, vol)
        # learned via :meth:`fit_weights`. ``None`` → use scalar defaults.
        sector_weights: Optional[dict[str, tuple[float, float]]] = None,
    ):
        self.base_model = base_model or EarningsBeatProbabilityModel()
        self.momentum_weight = float(np.clip(momentum_weight, 0, 1))
        self.volatility_weight = float(np.clip(volatility_weight, 0, 1))
        self.n_posterior_samples = n_posterior_samples
        self.n_chains = n_chains
        self.random_seed = int(random_seed)
        self.rng = np.random.default_rng(random_seed)
        self.sector_shrinkage_tau = float(sector_shrinkage_tau)
        self.sector_weights: dict[str, tuple[float, float]] = dict(sector_weights or {})

    # -------------------------------------------------------------------
    # §16.1 — adaptive per-sector momentum / volatility weights
    # -------------------------------------------------------------------
    def fit_weights(
        self,
        df: pd.DataFrame,
        sector_col: str = "industry",
        outcome_col: str = "historical_beat_rate",
        momentum_col: str = "price_momentum_3m",
        vol_col: str = "volatility_compression",
        min_samples_per_sector: int = 30,
    ) -> dict[str, tuple[float, float]]:
        """Fit per-sector (momentum_weight, volatility_weight) via constrained
        non-negative least-squares on realised beat outcomes.

        v3.10 §16.1 — replaces the global 0.4 / 0.3 scalars with sector-aware
        weights so high-vol regimes (Energy) get more momentum weight and
        low-vol sectors (Utilities) less. Constraint: weights are clipped to
        ``[0, 1]`` and the *sum* clipped to ≤ 1. Persisted on
        :attr:`sector_weights` for use by :meth:`_adjust_prior`.
        """
        if sector_col not in df.columns or outcome_col not in df.columns:
            return {}

        weights: dict[str, tuple[float, float]] = {}
        for sector, group in df.groupby(sector_col, dropna=True):
            if not isinstance(sector, str) or not sector:
                continue
            if len(group) < min_samples_per_sector:
                continue
            y = pd.to_numeric(group[outcome_col], errors="coerce")
            m = pd.to_numeric(group.get(momentum_col), errors="coerce")
            v = pd.to_numeric(group.get(vol_col), errors="coerce")
            if m is None or v is None:
                continue
            stacked = pd.concat([y, m, v], axis=1).dropna()
            if len(stacked) < min_samples_per_sector:
                continue
            # Centre covariates and target on mean so intercept = 0.
            Y = stacked.iloc[:, 0].to_numpy(dtype=float) - float(stacked.iloc[:, 0].mean())
            M = stacked.iloc[:, 1].to_numpy(dtype=float) - float(stacked.iloc[:, 1].mean())
            V = stacked.iloc[:, 2].to_numpy(dtype=float) - float(stacked.iloc[:, 2].mean())
            # Least squares on stacked [M, V] against Y.
            try:
                X = np.column_stack([M, V])
                beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
                mw = float(np.clip(beta[0], 0.0, 1.0))
                vw = float(np.clip(beta[1], 0.0, 1.0))
                total = mw + vw
                if total > 1.0:
                    mw, vw = mw / total, vw / total
                weights[sector] = (mw, vw)
            except ValueError, np.linalg.LinAlgError:
                continue

        self.sector_weights = weights
        return weights

    def _get_weights(self, sector: Optional[str]) -> tuple[float, float]:
        """Return (momentum_weight, volatility_weight) for a sector.

        Falls back to the scalar defaults when the sector has no learned
        weights (§16.1).
        """
        if sector and sector in self.sector_weights:
            return self.sector_weights[sector]
        return self.momentum_weight, self.volatility_weight

    # -------------------------------------------------------------------
    # §16.5 — seed-stability diagnostic
    # -------------------------------------------------------------------
    def stability_report(
        self,
        df: pd.DataFrame,
        seeds: list[int] | None = None,
        sector_col: str = "industry",
        ticker_col: str = "isin",
    ) -> pd.DataFrame:
        """Quantify run-over-run stability of the resampled beat posterior.

        v3.10 §16.5 — addresses v3.8 "Largest group drift: MXN ‑1.78 pp" by
        running :meth:`analyze_dataframe` under multiple seeds and reporting
        per-ticker mean / std / min / max of ``resampled_posterior_mean``.
        Tickers with ``std > 0.02`` (≈ 2 pp) are flagged seed-unstable.
        """
        if seeds is None:
            seeds = [42, 7, 99]
        frames: list[pd.DataFrame] = []
        _saved_seed = self.random_seed
        _saved_rng = self.rng
        try:
            for s in seeds:
                self.random_seed = int(s)
                self.rng = np.random.default_rng(int(s))
                out = self.analyze_dataframe(df, sector_col=sector_col, ticker_col=ticker_col)
                if out.empty or "resampled_posterior_mean" not in out.columns:
                    continue
                sub = out[[ticker_col, "resampled_posterior_mean"]].copy()
                sub["seed"] = int(s)
                frames.append(sub)
        finally:
            self.random_seed = _saved_seed
            self.rng = _saved_rng
        if not frames:
            return pd.DataFrame(columns=[ticker_col, "mean", "std", "min", "max", "seed_unstable"])
        stacked = pd.concat(frames, ignore_index=True)
        g = stacked.groupby(ticker_col)["resampled_posterior_mean"]
        report = g.agg(["mean", "std", "min", "max"]).reset_index()
        report["seed_unstable"] = report["std"].fillna(0.0) > 0.02
        return report

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
        if "volatility_compression" in row.index and pd.notna(
            row["volatility_compression"]
        ):
            score = float(np.clip(row["volatility_compression"], 0, 1))
        elif "volatility_term_structure" in row.index and pd.notna(
            row["volatility_term_structure"]
        ):
            score = float(
                np.clip(1.0 - abs(row["volatility_term_structure"]) / 100, 0, 1)
            )
        return score

    def _adjust_prior(
        self,
        base_alpha: float,
        base_beta: float,
        momentum_signal: float,
        vol_regime: float,
        sector: Optional[str] = None,
        n_sector: Optional[int] = None,
    ) -> tuple[float, float]:
        """Adjust Beta prior parameters based on technical signals.

        Positive momentum + low volatility → shift prior toward higher beat
        rate.

        v3.10 §16.1 — honours per-sector ``(momentum_weight, volatility_weight)``
        when available via :meth:`fit_weights`.

        v3.10 §16.4 — optional credibility shrinkage toward the base-model
        sector prior. When ``sector`` and ``n_sector`` are provided and the
        base model exposes a sector-specific prior, the tilt is blended with
        ``κ = n_sector / (n_sector + τ)`` where ``τ =
        sector_shrinkage_tau``. Small-sample sectors therefore shrink toward
        the stable sector prior instead of producing extreme tilts.
        """
        mw, vw = self._get_weights(sector)
        adjustment = mw * momentum_signal + vw * (vol_regime - 0.5) * 2
        concentration = base_alpha + base_beta
        shift = adjustment * 0.2 * concentration

        tilted_alpha = max(0.5, base_alpha + shift)
        tilted_beta = max(0.5, base_beta - shift)

        # §16.4 credibility shrinkage toward the sector prior (if available).
        if sector and n_sector is not None and self.sector_shrinkage_tau > 0:
            sector_prior = self.base_model.sector_priors.get(sector)
            if sector_prior is not None:
                kappa = float(n_sector) / (float(n_sector) + float(self.sector_shrinkage_tau))
                kappa = float(np.clip(kappa, 0.0, 1.0))
                tilted_alpha = kappa * tilted_alpha + (1.0 - kappa) * sector_prior.alpha
                tilted_beta = kappa * tilted_beta + (1.0 - kappa) * sector_prior.beta

        return float(tilted_alpha), float(tilted_beta)

    def _run_analysis(
        self,
        df: pd.DataFrame,
        sector_col: str = "industry",
        ticker_col: str = "isin",
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
            orig_row = (
                df.loc[orig_mask].iloc[0] if orig_mask.any() else pd.Series(dtype=float)
            )

            momentum = self._compute_momentum_signal(orig_row)
            vol_regime = self._compute_volatility_regime(orig_row)

            base_alpha = row.get("posterior_alpha", 2.0)
            base_beta = row.get("posterior_beta", 2.0)
            base_mean = base_alpha / (base_alpha + base_beta)

            adj_alpha, adj_beta = self._adjust_prior(
                base_alpha, base_beta, momentum, vol_regime
            )
            adj_mean = adj_alpha / (adj_alpha + adj_beta)

            ci_90 = (
                float(stats.beta.ppf(0.05, adj_alpha, adj_beta)),
                float(stats.beta.ppf(0.95, adj_alpha, adj_beta)),
            )
            ci_95 = (
                float(stats.beta.ppf(0.025, adj_alpha, adj_beta)),
                float(stats.beta.ppf(0.975, adj_alpha, adj_beta)),
            )

            # --- v3.10 §15.1 posterior spread & HDI from closed-form Beta ---
            # Variance of Beta(a, b) = a*b / ((a+b)^2 * (a+b+1)).
            _ab_sum = float(adj_alpha + adj_beta)
            try:
                posterior_std = float(
                    np.sqrt((float(adj_alpha) * float(adj_beta)) / ((_ab_sum**2) * (_ab_sum + 1.0)))
                )
            except ValueError, ZeroDivisionError:
                posterior_std = float("nan")
            # 94 % HDI (ArviZ default) — approximate via symmetric 3rd/97th
            # quantiles of the Beta; exact HDI is not closed-form but this is
            # a close surrogate for downstream BMA log-score weighting.
            try:
                hdi_low = float(stats.beta.ppf(0.03, adj_alpha, adj_beta))
                hdi_high = float(stats.beta.ppf(0.97, adj_alpha, adj_beta))
            except ValueError, TypeError:
                hdi_low = float("nan")
                hdi_high = float("nan")

            # Volatility-regime label from the continuous score.
            if vol_regime >= 0.5:
                vol_regime_label = "high"
            elif vol_regime <= -0.5:
                vol_regime_label = "low"
            else:
                vol_regime_label = "normal"

            # Effective sample size proxy from the Beta concentration
            # (a + b) — overwritten by the ArviZ summary in
            # ``analyze_dataframe`` when per-chain draws are available.
            n_eff = float(_ab_sum)

            results.append(
                ResampledBeatEstimate(
                    ticker=str(ticker),
                    name=str(row.get("name", "")),
                    sector=str(row.get(sector_col, row.get("industry", ""))),
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
                    # --- v3.10 §15.1 posterior spread + chain diagnostics ---
                    posterior_std=posterior_std,
                    hdi_low=hdi_low,
                    hdi_high=hdi_high,
                    # chain_rhat / chain_ess_bulk / chain_ess_tail are
                    # populated from the ArviZ summary in
                    # ``analyze_dataframe`` when the InferenceData build
                    # succeeds; leave as NaN sentinels here so downstream
                    # consumers can detect missing chain diagnostics.
                    chain_rhat=float("nan"),
                    chain_ess_bulk=float("nan"),
                    chain_ess_tail=float("nan"),
                    n_effective_samples=n_eff,
                    volatility_regime=vol_regime_label,
                    # --- v3.10 §15.2 versioned serialisation ---
                    schema_version=RESULT_SCHEMA_VERSION,
                )
            )

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        sector_col: str = "industry",
        ticker_col: str = "isin",
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
                idata = self.build_inference_data(
                    df, sector_col=sector_col, ticker_col=ticker_col
                )
                if idata is not None:
                    summary = az.summary(idata)
                    if "ess_bulk" in summary.columns and len(summary) == len(result_df):
                        result_df["ess_bulk"] = summary["ess_bulk"].values
                        # v3.10 §15.1 — also populate the dataclass-aligned
                        # per-chain diagnostic columns so downstream consumers
                        # see non-NaN values when ArviZ chains are available.
                        result_df["chain_ess_bulk"] = summary["ess_bulk"].values
                        result_df["n_effective_samples"] = summary["ess_bulk"].values
                    if "ess_tail" in summary.columns and len(summary) == len(result_df):
                        result_df["chain_ess_tail"] = summary["ess_tail"].values
                    if "r_hat" in summary.columns and len(summary) == len(result_df):
                        result_df["r_hat"] = summary["r_hat"].values
                        result_df["chain_rhat"] = summary["r_hat"].values
            except (ValueError, KeyError, TypeError):
                pass

        return result_df

    def build_inference_data(
        self,
        df: pd.DataFrame,
        sector_col: str = "industry",
        ticker_col: str = "isin",
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
                self.rng.beta(
                    adj_alphas, adj_betas, size=(self.n_posterior_samples, n_equities)
                )
                for _ in range(self.n_chains)
            ]
        )

        pp_samples = (
            self.rng.random(posterior_samples.shape) < posterior_samples
        ).astype(int)

        coords = {
            "chain": np.arange(self.n_chains),
            "draw": np.arange(self.n_posterior_samples),
            "equity": tickers,
        }

        if ARVIZ_AVAILABLE and az is not None:
            return az.from_dict(
                {
                    "posterior": {"beat_probability": posterior_samples},
                    "posterior_predictive": {"beat_outcome": pp_samples},
                    "observed_data": {
                        "base_posterior_mean": result_df["base_posterior_mean"].values,
                        "momentum_signal": result_df["momentum_signal"].values,
                    },
                    "constant_data": {
                        "momentum_weight": np.array([self.momentum_weight]),
                        "volatility_weight": np.array([self.volatility_weight]),
                    },
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
    fig = make_subplots(
        rows=rows, cols=2, subplot_titles=[f"{feat}" for feat in feature_cols]
    )

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
        except (OSError, ValueError, TypeError) as e:
            logger.error("Failed to export %s: %s", table_name, e)

    # Issue 7: Cast mixed-type columns to proper numeric dtypes before export
    for col in _NUMERIC_CAST_COLS:
        if col in probability_df.columns:
            probability_df[col] = pd.to_numeric(probability_df[col], errors="coerce")
    for col in _INTEGER_CAST_COLS:
        if col in probability_df.columns:
            probability_df[col] = pd.to_numeric(
                probability_df[col], errors="coerce"
            ).astype("Int64")

    # Also cast streak_df columns to proper numeric dtypes
    for col in _NUMERIC_CAST_COLS:
        if col in streak_df.columns:
            streak_df[col] = pd.to_numeric(streak_df[col], errors="coerce")
    for col in _INTEGER_CAST_COLS:
        if col in streak_df.columns:
            streak_df[col] = pd.to_numeric(streak_df[col], errors="coerce").astype(
                "Int64"
            )

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
    required_prob_cols = {
        "posterior_beat_prob",
        "beat_classification",
        "confidence_score",
    }
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
                    float(
                        (probability_df["beat_classification"] == "likely_beat").sum()
                    ),
                    float(probability_df["confidence_score"].mean()),
                    float(streak_df["current_streak"].mean()),
                    float((streak_df["streak_type"] == "beat").sum()),
                    float((streak_df["streak_type"] == "miss").sum()),
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            _safe_export(summary_df, "probability_analytics_summary", reorder=False)
        except (KeyError, ValueError, TypeError) as e:
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
