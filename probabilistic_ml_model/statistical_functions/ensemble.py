"""
Ensemble alignment builders (R3).

Migrated from ``expected_returns_v3.py`` to decouple v4 from v3.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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

    from probabilistic_ml_model.data_utils import get_identifier_cols_set

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
            ]
        )
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
        columns={"prob_beat_given_momentum": "beat_prob"}
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
    mcmc_result: dict | None = None,
) -> pd.DataFrame:
    """Merge four expected-return model results into a unified summary DataFrame."""
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

    from probabilistic_ml_model.data_utils import (
        get_identifier_cols_set,
        load_identifier_columns,
    )

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
            + available_market
        )
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

    # Merge anomaly results
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
        available_anomaly = [c for c in _ANOMALY_COLS if c in anomaly_results.columns]
        if available_anomaly:
            anomaly_subset = anomaly_results[["ticker"] + available_anomaly].drop_duplicates(
                subset="ticker"
            )
            summary = summary.merge(anomaly_subset, on="ticker", how="left")
            logger.info(
                "Enriched expected_returns_summary with %d anomaly columns", len(available_anomaly)
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
        "debt_3y_cagr",
        "debt_maturity_risk",
        "balance_sheet_strength",
        "wc_efficiency_score",
        "distress_risk_score",
    ]
    if credit is not None and not credit.empty and "ticker" in credit.columns:
        available_credit = [c for c in _CREDIT_COLS if c in credit.columns]
        if available_credit:
            credit_subset = credit[["ticker"] + available_credit].drop_duplicates(subset="ticker")
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
    if div_safety is not None and not div_safety.empty and "ticker" in div_safety.columns:
        available_div = [c for c in _DIV_SAFETY_COLS if c in div_safety.columns]
        if available_div:
            div_subset = div_safety[["ticker"] + available_div].drop_duplicates(subset="ticker")
            summary = summary.merge(div_subset, on="ticker", how="left")
            logger.info(
                "Enriched expected_returns_summary with %d dividend safety columns",
                len(available_div),
            )

    if summary.empty:
        logger.warning("Expected returns summary: no overlapping tickers across all 4 models")
        return summary

    # Enrich with market-data columns from mc
    for col in available_market:
        if col not in summary.columns and col in mc.columns:
            price_map = (
                mc[["ticker", col]].drop_duplicates(subset="ticker").set_index("ticker")[col]
            )
            summary[col] = summary["ticker"].map(price_map)
            logger.debug("Merged market-data column '%s' from mc", col)

    # Enrich from source_df
    if source_df is not None and "ticker" in source_df.columns:
        id_cols_ordered = load_identifier_columns()
        desired_cols = id_cols_ordered + market_data_cols
        missing_cols = [
            c for c in desired_cols if c in source_df.columns and c not in summary.columns
        ]
        if missing_cols:
            source_subset = source_df[["ticker"] + missing_cols].drop_duplicates(subset="ticker")
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

    # Confidence-weighted agreement
    mc_weight = summary["prob_positive_upside"].clip(0, 100) / 100.0
    kal_weight = 0.5
    pt_weight = (
        summary["confidence_level"].map({"High": 0.9, "Medium": 0.6, "Low": 0.3}).fillna(0.5)
    )
    earn_weight = summary["confidence_score"].clip(0, 1)

    summary["weighted_agreement"] = (
        summary["mc_bullish"].astype(float) * mc_weight
        + summary["kal_bullish"].astype(float) * kal_weight
        + summary["pt_bullish"].astype(float) * pt_weight
        + summary["earn_bullish"].astype(float) * earn_weight
    )

    # Merge parallel MCMC return analysis diagnostics
    if mcmc_result and isinstance(mcmc_result, dict):
        if mcmc_result.get("converged") is not None:
            summary["mcmc_converged"] = mcmc_result.get("converged", False)
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

    # Remove duplicate columns
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
