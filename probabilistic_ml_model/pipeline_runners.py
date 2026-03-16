"""
Pipeline runners for PML models (R1).

Migrated from ``expected_returns_v3.py`` to decouple v4 from v3.
Each runner calls PML model classes directly.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    from probabilistic_ml_model.statistical_functions.statistical_analysis import (
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
    logger.info("Monte Carlo simulation: %d stocks processed", len(mc))
    return mc


def run_price_target_achievement(
    df: pd.DataFrame,
    use_historical_targets: bool = True,
    feature_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate probability of reaching consensus price targets."""
    from probabilistic_ml_model.statistical_functions.probability_analytics import (
        PriceTargetAchievementModel,
    )

    pt_df = df.copy()

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
            c for c in _SENTIMENT_COLS if c not in pt_df.columns and c in feature_df.columns
        ]
        if missing_sentiment:
            sentiment_subset = feature_df[["ticker"] + missing_sentiment].drop_duplicates(
                subset="ticker"
            )
            pt_df = pt_df.merge(sentiment_subset, on="ticker", how="left")
            logger.info(
                "Price target achievement: merged %d sentiment columns from feature views",
                len(missing_sentiment),
            )

    _PT_RISK_COLS = [
        "beta_1y",
        "beta_stability_score",
        "distress_risk_score",
        "balance_sheet_strength",
        "debt_maturity_risk",
    ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing_risk = [
            c for c in _PT_RISK_COLS if c not in pt_df.columns and c in feature_df.columns
        ]
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
    """Apply Kalman filter to smooth noisy analyst price targets."""
    from probabilistic_ml_model.statistical_functions.statistical_analysis import (
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

    if not kal.empty and "filtered_upside" in kal.columns:
        lower, upper = kal["filtered_upside"].quantile([0.01, 0.99])
        kal["filtered_upside"] = kal["filtered_upside"].clip(lower, upper)

    logger.info("Kalman filter: %d stocks processed", len(kal))
    return kal


def run_earnings_beat_analysis(
    df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run enhanced three-layer Bayesian earnings beat probability model."""
    from probabilistic_ml_model.statistical_functions.probability_analytics import (
        EarningsBeatProbabilityModel,
        EPSStreakAnalyzer,
        ResampledBeatProbabilityModel,
    )
    from probabilistic_ml_model.statistical_functions.statistical_analysis import (
        bayesian_earnings_beat_model,
    )

    beat_df = df.copy()

    _BEAT_QUALITY_COLS = [
        "accounting_quality_score",
        "quality_issues_count_5y",
        "balance_sheet_strength",
    ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing_q = [
            c for c in _BEAT_QUALITY_COLS if c not in beat_df.columns and c in feature_df.columns
        ]
        if missing_q:
            q_subset = feature_df[["ticker"] + missing_q].drop_duplicates(subset="ticker")
            beat_df = beat_df.merge(q_subset, on="ticker", how="left")
            logger.info("Earnings beat: merged %d quality columns", len(missing_q))

    model = EarningsBeatProbabilityModel()
    sector_col = "sector" if "sector" in beat_df.columns else "industry"
    beat = model.analyze_dataframe_enhanced(beat_df, sector_col=sector_col)
    logger.info("Earnings beat analysis: %d stocks processed", len(beat))

    # EPS streak analysis
    try:
        streak_analyzer = EPSStreakAnalyzer()
        streak_df = streak_analyzer.analyze_dataframe(df)
        if not streak_df.empty and "ticker" in streak_df.columns:
            streak_cols = [c for c in streak_df.columns if c != "ticker" and c not in beat.columns]
            if streak_cols:
                beat = beat.merge(streak_df[["ticker"] + streak_cols], on="ticker", how="left")
                logger.info("EPS streak enrichment: %d columns added", len(streak_cols))
    except Exception as e:
        logger.warning("EPS streak analysis failed: %s", e)

    # Resampled technical priors
    try:
        resampled_model = ResampledBeatProbabilityModel(base_model=model)
        resampled_df = resampled_model.analyze_dataframe(df)
        if not resampled_df.empty and "ticker" in resampled_df.columns:
            resamp_cols = [
                c for c in resampled_df.columns if c != "ticker" and c not in beat.columns
            ]
            if resamp_cols:
                beat = beat.merge(resampled_df[["ticker"] + resamp_cols], on="ticker", how="left")
                logger.info("Resampled beat enrichment: %d columns added", len(resamp_cols))
    except Exception as e:
        logger.warning("Resampled beat probability failed: %s", e)

    # Classical Bayesian earnings beat model
    try:
        bayesian_beat = bayesian_earnings_beat_model(df)
        if not bayesian_beat.empty and "ticker" in bayesian_beat.columns:
            bay_cols = [c for c in bayesian_beat.columns if c != "ticker" and c not in beat.columns]
            if bay_cols:
                beat = beat.merge(bayesian_beat[["ticker"] + bay_cols], on="ticker", how="left")
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
    """Run credit risk and ruin probability analysis."""
    from probabilistic_ml_model.statistical_functions.probability_analytics import (
        CreditRiskProbabilityModel,
    )
    from probabilistic_ml_model.statistical_functions.statistical_analysis import (
        calculate_ruin_probability,
        hierarchical_mcmc_by_sector,
    )

    credit_df = df.copy()

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
        "debt_3y_cagr",
        "debt_4q_trend",
        "debt_yoy_change",
        "adequate_cash_buffer",
        "cash_vs_5y_avg",
        "retained_earnings_vs_5y",
        "distress_risk_score",
        "retained_earnings_growth",
        "beta_trend",
    ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing_risk = [
            c for c in _CREDIT_RISK_COLS if c not in credit_df.columns and c in feature_df.columns
        ]
        if missing_risk:
            risk_subset = feature_df[["ticker"] + missing_risk].drop_duplicates(subset="ticker")
            credit_df = credit_df.merge(risk_subset, on="ticker", how="left")
            logger.info(
                "Credit risk analysis: merged %d risk columns from feature views", len(missing_risk)
            )

    credit_model = CreditRiskProbabilityModel(n_mcmc_samples=n_mcmc_samples, burn_in=burn_in)
    credit = credit_model.analyze_dataframe(credit_df)

    # Hierarchical sector-level MCMC enrichment
    try:
        if "altman_z_score" in credit_df.columns:
            z_data = credit_df["altman_z_score"].dropna()
            if len(z_data) > 50:
                sector_mcmc = hierarchical_mcmc_by_sector(credit_df, "altman_z_score")
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
        if not ruin.empty and not credit.empty and "ticker" in ruin.columns:
            ruin_cols = [c for c in ruin.columns if c != "ticker" and c not in credit.columns]
            if ruin_cols:
                credit = credit.merge(ruin[["ticker"] + ruin_cols], on="ticker", how="left")
                logger.info("Ruin probability enrichment: %d columns added", len(ruin_cols))
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
    """Run dividend cut probability analysis."""
    from probabilistic_ml_model.statistical_functions.probability_analytics import (
        DividendCutProbabilityModel,
    )

    div_df = df.copy()

    _DIV_LEVERAGE_COLS = [
        "interest_coverage",
        "debt_to_equity",
        "cash_ratio",
        "working_capital_ratio",
        "balance_sheet_strength",
        "cash_runway_months",
        "retained_earnings_growth",
        "debt_3y_cagr",
    ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing = [
            c for c in _DIV_LEVERAGE_COLS if c not in div_df.columns and c in feature_df.columns
        ]
        if missing:
            subset = feature_df[["ticker"] + missing].drop_duplicates(subset="ticker")
            div_df = div_df.merge(subset, on="ticker", how="left")
            logger.info("Dividend safety: merged %d leverage columns", len(missing))

    model = DividendCutProbabilityModel(n_mcmc_samples=n_mcmc_samples, burn_in=burn_in)
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
    """Run standalone accounting anomaly detection and analytics."""
    from probabilistic_ml_model.statistical_functions.probability_analytics import (
        AccountingAnomalyProbabilityModel,
    )
    from probabilistic_ml_model.statistical_functions.statistical_analysis import mcmc_student_t

    anomaly_df = df.copy()

    _ACCOUNTING_COLS = [
        "exceptional_items_frequency",
        "gaap_adj_eps_gap_pct",
        "asset_sale_boost",
        "ebitda_adjustment_ratio",
        "eps_adjustment_ratio",
        "exceptional_items_to_ebitda",
        "restructuring_intensity",
        "goodwill_change_rate",
        "eps_adj_ltm",
        "eps_adjustment_ratio_comp",
        "eps_adjustment_spread_ltm",
        "eps_adjustment_spread_fy",
        "eps_adjustment_pct",
        "net_income_adjustment_ratio_ltm",
        "net_income_adjustment_ratio_fy",
        "net_income_adjustment_pct",
        "ebitda_adjustment_pct_ltm",
        "ebitda_adjustment_pct_fy",
        "ebit_adjustment_pct_ltm",
        "ebit_adjustment_pct_fy",
        "forward_eps_gaap_adj_spread",
        "gaap_vs_norm_revision_spread",
        "gaap_revision_momentum",
        "gaap_revision_1m",
        "gaap_revision_3m",
        "gaap_revision_6m",
        "gaap_revision_1y",
        "discontinued_ops_impact",
        "earnings_quality_warning",
        "revision_quality_divergence",
        "eps_growth_accel",
        "eps_surprise_pct",
        "revenue_surprise_pct",
    ]
    if feature_df is not None and "ticker" in feature_df.columns:
        missing_acct = [
            c for c in _ACCOUNTING_COLS if c not in anomaly_df.columns and c in feature_df.columns
        ]
        if missing_acct:
            acct_subset = feature_df[["ticker"] + missing_acct].drop_duplicates(subset="ticker")
            anomaly_df = anomaly_df.merge(acct_subset, on="ticker", how="left")
            logger.info(
                "Accounting anomaly analysis: merged %d columns from feature views",
                len(missing_acct),
            )

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
            anomaly_scores = result["accounting_anomaly_score"].dropna().values
            if len(anomaly_scores) > 50:
                mu_samples, df_samples = mcmc_student_t(anomaly_scores)
                result["anomaly_posterior_location"] = mu_samples.mean()
                logger.info("MCMC anomaly posterior: location=%.3f", mu_samples.mean())
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
            for c in ["composite_score", "upside_potential", "expected_upside_pct"]
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

    _SCREEN_RUNNERS: list[tuple[str, callable]] = [
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
            metric="composite_score" if "composite_score" in df_all.columns else "upside_potential",
        )
        logger.info("Sector-relative ranking: %d stocks", len(screens["sector_relative"]))
    except Exception as e:
        logger.warning("Sector-relative ranking failed: %s", e)

    return screens


def run_resampled_posterior_analysis(
    df: pd.DataFrame,
    freq: str = "1ME",
) -> pd.DataFrame:
    """Compute Bayesian resampled return posteriors from historical price snapshots."""
    from probabilistic_ml_model.statistical_functions.statistical_analysis import (
        resampled_posterior_returns,
    )

    try:
        result_df, idata = resampled_posterior_returns(
            df, freq=freq, n_posterior_samples=4000, n_chains=4
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


def run_category_probability_analysis(
    df: pd.DataFrame,
    categories: Optional[dict[str, list[str]]] = None,
    *,
    use_mcmc: bool = True,
    n_mcmc_samples: int = 5000,
    burn_in: int = 1000,
    n_jobs: int = 1,
    max_features_per_category: int = 0,
    cache_dir: str = "",
    enable_caching: bool = False,
) -> dict[str, dict]:
    """Run per-category Bayesian probability analytics.

    Delegates to the v3 implementation which handles parallelization,
    caching, and feature budget control.
    """
    from probabilistic_ml_model.data_utils import load_feature_categories_from_db
    from probabilistic_ml_model.statistical_functions.probability_analytics import (
        CategoryProbabilityAnalyzer,
    )
    from probabilistic_ml_model.statistical_functions.statistical_analysis import (
        calculate_conditional_probabilities,
        fit_distributions_by_category,
        run_category_probability_analytics,
    )

    if categories is None:
        categories = load_feature_categories_from_db() or {}

    if not categories:
        logger.warning("No feature categories available — skipping category analytics")
        return {}

    all_results: dict[str, dict] = {}

    for cat_name, cat_cols in categories.items():
        available = [c for c in cat_cols if c in df.columns]
        if not available:
            continue
        if max_features_per_category > 0:
            available = available[:max_features_per_category]

        try:
            cat_results = run_category_probability_analytics(
                df, cat_name, available, n_simulations=10_000
            )

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

            all_results[cat_name] = cat_results
        except Exception as e:
            logger.warning("Category %s analysis failed: %s", cat_name, e)

    return all_results


def run_parallel_mcmc_return_analysis(
    mc: pd.DataFrame,
    n_chains: int = 4,
    n_samples: int = 10_000,
) -> dict:
    """Run parallel MCMC on MC expected upside with Gelman-Rubin diagnostic."""
    from probabilistic_ml_model.statistical_functions.statistical_analysis import (
        parallel_mcmc_chains,
    )

    if mc.empty or "expected_upside_pct" not in mc.columns:
        return {}

    data = mc["expected_upside_pct"].dropna().values
    if len(data) < 50:
        logger.warning("Parallel MCMC skipped — insufficient data (%d)", len(data))
        return {}

    result = parallel_mcmc_chains(data=data, n_chains=n_chains, n_samples=n_samples)
    logger.info(
        "Parallel MCMC: R\u0302=%.4f, converged=%s, posterior mean=%.2f",
        result.get("r_hat", float("nan")),
        result.get("converged", False),
        result.get("posterior_mean", float("nan")),
    )
    return result


# ---------------------------------------------------------------------------
# Analytical helpers (used by Phase 5)
# ---------------------------------------------------------------------------


def compute_derived_price_target(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    price_col: str = "last_price",
    return_col: str = "expected_upside_pct",
    output_col: str = "price_target_derived",
) -> pd.DataFrame:
    """Calculate a derived price target from a return-percentage column."""
    if df.empty:
        logger.warning("compute_derived_price_target: empty input — skipping")
        return df

    result = df.copy()

    if price_col not in result.columns:
        if "ticker" not in source_df.columns or price_col not in source_df.columns:
            logger.warning(
                "Cannot compute %s — '%s' or 'ticker' missing from source_df", output_col, price_col
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
    pt: pd.DataFrame, source_df: pd.DataFrame, **kw
) -> pd.DataFrame:
    """Calculate price target from probability-weighted return."""
    return compute_derived_price_target(
        pt,
        source_df,
        return_col="expected_return_prob_weighted",
        output_col="price_target_prob_weighted",
        **kw,
    )


def compute_price_target_mc(pt: pd.DataFrame, source_df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Calculate price target from Monte Carlo expected upside."""
    return compute_derived_price_target(
        pt, source_df, return_col="expected_upside_pct", output_col="price_target_mc", **kw
    )


def filter_quality_stocks(summary: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality screening to the expected returns summary."""
    from probabilistic_ml_model.statistical_functions.screening import (
        rank_stocks_by_composite_score,
    )

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


def compute_return_zscore_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Add industry-relative z-scores and percentile ranks for key return metrics."""
    from probabilistic_ml_model.optimized_ops import vectorized_percentile_rank, vectorized_zscore

    if summary.empty:
        return summary

    return_cols = [
        c
        for c in ["expected_upside_pct", "filtered_upside", "expected_return_prob_weighted"]
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
    output_dir: str = "outputs/analytics",
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
    except Exception:
        pass

    return exports


# ---------------------------------------------------------------------------
# Cross-model analytics
# ---------------------------------------------------------------------------


def compute_cross_model_correlation(mc: pd.DataFrame, kal: pd.DataFrame) -> dict:
    """Compute correlation and copula dependency between MC and Kalman returns."""
    from probabilistic_ml_model.statistical_functions.statistical_analysis import (
        fit_gaussian_copula,
    )

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
                merged, features=["expected_upside_pct", "filtered_upside"]
            )
            if copula:
                result["tail_dependence"] = copula.get("tail_dependence")
        except Exception as e:
            logger.debug("Copula fit skipped: %s", e)

    return result
