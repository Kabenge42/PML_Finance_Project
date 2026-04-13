"""
Ensemble alignment builders (R3).

Migrated from ``expected_returns_v3.py`` to decouple v4 from v3.

R4 refactorings (Issues 1–8):
- Two-tier ensemble scoring (directional + risk quality).
- Neutral fillna for missing risk data with coverage flags.
- Data-adaptive anomaly severity threshold.
- Single source of truth for consensus scoring (no redundant 4-model scoring).
- Kalman variance-based confidence weighting.
- Materiality threshold for bullish classification.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


def _ensure_isin_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'isin' is a column in the DataFrame, resetting index if needed."""
    if df is None or df.empty:
        return pd.DataFrame()

    df_out = df.copy()
    if "isin" not in df_out.columns:
        if df_out.index.name == "isin":
            df_out = df_out.reset_index()
        elif "isin" in df_out.index.names:
            df_out = df_out.reset_index(level="isin")
    return df_out


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
    bullish_return_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Merge Monte Carlo, Kalman, and Price Target Achievement into a
    tri-model alignment DataFrame with direction agreement scores.

    Parameters
    ----------
    bullish_return_threshold : float
        Minimum implied return (%) to classify a model as bullish.
        Default 2.0% prevents near-zero returns from inflating the
        "Strong Bullish" bucket (Issue 8).
    """
    if mc.empty or kal.empty or pt.empty:
        logger.warning("Tri-model alignment skipped — one or more inputs empty")
        return pd.DataFrame()

    mc = _ensure_isin_column(mc)
    kal = _ensure_isin_column(kal)
    pt = _ensure_isin_column(pt)

    from probabilistic_ml_model.data_utils import get_identifier_cols_set

    id_cols_set = get_identifier_cols_set()
    mc_id_cols = [c for c in mc.columns if c in id_cols_set]
    mc_select = list(
        set(
            mc_id_cols
            + [
                "isin",
                "ticker",
                "implied_return_mc",
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
            kal[["isin", "implied_return_kalman", "kalman_estimate", "kalman_variance"]],
            on="isin",
            how="inner",
        )
        .merge(
            pt[
                [
                    "isin",
                    "implied_return_pt",
                    "achievement_probability",
                    "price_target_prob_weighted",
                    "confidence_level",
                    "analyst_conviction",
                    "eps_revision_momentum",
                    "analyst_rating_normalized",
                ]
            ],
            on="isin",
            how="inner",
        )
    )

    # Issue 8: materiality threshold for bullish classification
    tri["mc_bullish"] = tri["implied_return_mc"] > bullish_return_threshold
    tri["kal_bullish"] = tri["implied_return_kalman"] > bullish_return_threshold
    tri["pt_bullish"] = tri["implied_return_pt"] > bullish_return_threshold
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
    beat_threshold: float = 0.50,
    credit: pd.DataFrame | None = None,
    div_safety: pd.DataFrame | None = None,
    anomaly: pd.DataFrame | None = None,
    *,
    credit_distress_threshold: float = 0.50,
    div_cut_threshold: float = 0.40,
    anomaly_severity_threshold: float | None = None,
) -> pd.DataFrame:
    """Extend tri-model alignment with up to 4 additional model signals.

    R4 refactorings applied:
    - **Issue 1**: Two-tier scoring — ``directional_agreement`` (0–4) for
      return models (MC/Kalman/PT/Earnings) and ``risk_quality_score`` (0–3)
      for risk filters (credit/dividend/anomaly).  ``full_consensus`` requires
      directional agreement = 4/4 AND risk quality ≥ 2/3.
    - **Issue 2**: Missing risk data filled with median values instead of
      worst-case; ``*_coverage`` flags distinguish "no data" from "failed".
    - **Issue 3**: When *anomaly_severity_threshold* is ``None`` (default),
      the threshold is computed adaptively as the median of the available
      anomaly severity distribution.

    Parameters
    ----------
    tri : pd.DataFrame
        Tri-model alignment output (must contain ``mc_bullish``, ``kal_bullish``,
        ``pt_bullish``).
    beat : pd.DataFrame
        Earnings beat results with ``prob_beat_given_momentum``.
    beat_threshold : float
        Minimum beat probability to flag bullish.
    credit : pd.DataFrame or None
        Credit risk results with ``distress_probability``.
    div_safety : pd.DataFrame or None
        Dividend safety results with ``dividend_cut_probability``.
    anomaly : pd.DataFrame or None
        Accounting anomaly results with ``anomaly_severity_score``.
    credit_distress_threshold : float
        Maximum distress probability to flag as credit-safe.
    div_cut_threshold : float
        Maximum dividend cut probability to flag as dividend-safe.
    anomaly_severity_threshold : float or None
        Maximum anomaly severity score to flag as anomaly-clean.
        When ``None`` the threshold is set to the **median** of the
        available anomaly severity distribution (data-adaptive).

    Returns
    -------
    pd.DataFrame
        Alignment DataFrame with two-tier scoring columns:
        ``directional_agreement`` (0–4), ``risk_quality_score`` (0–3),
        ``full_consensus`` (bool), and the legacy ``quad_agreement`` (0–7).
    """
    if tri.empty or beat.empty:
        logger.warning("Quad-model alignment skipped — insufficient data")
        return pd.DataFrame()

    tri = _ensure_isin_column(tri)
    beat = _ensure_isin_column(beat)
    credit = _ensure_isin_column(credit)
    div_safety = _ensure_isin_column(div_safety)
    anomaly = _ensure_isin_column(anomaly)

    if "prob_beat_given_momentum" not in beat.columns:
        logger.warning("Quad-model skipped — beat results missing prob_beat_given_momentum")
        return pd.DataFrame()

    beat_slim = beat[["isin", "prob_beat_given_momentum"]].rename(
        columns={"prob_beat_given_momentum": "beat_prob"}
    )
    quad = tri.merge(beat_slim, on="isin", how="inner")
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
        credit_slim = credit[["isin", "distress_probability"]].drop_duplicates(subset="isin")
        median_distress = credit_slim["distress_probability"].median()
        quad = quad.merge(credit_slim, on="isin", how="left")
        quad["credit_coverage"] = quad["distress_probability"].notna().astype(int)
        quad["distress_probability"] = quad["distress_probability"].fillna(median_distress)
        quad["credit_safe"] = (quad["distress_probability"] < credit_distress_threshold).astype(int)
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
        quad["dividend_cut_probability"] = quad["dividend_cut_probability"].fillna(median_div_cut)
        quad["div_safe"] = (quad["dividend_cut_probability"] < div_cut_threshold).astype(int)
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
        quad["anomaly_severity_score"] = quad["anomaly_severity_score"].fillna(median_severity)
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
    quad["full_consensus"] = (
        (quad["directional_agreement"] == 4) & (quad["risk_quality_score"] >= 2)
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

    # Determine the number of active models for labelling
    n_models = 4  # base: MC + Kalman + PT + Beat
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
    """Merge four expected-return model results into a unified summary DataFrame.

    R4 refactorings:
    - **Issue 4**: When *quad* is provided its two-tier consensus scores
      (``directional_agreement``, ``risk_quality_score``, ``full_consensus``,
      ``signal``) are merged directly — no redundant 4-model scoring.
    - **Issue 7**: Kalman weight derived from ``kalman_variance`` (inverse
      variance weighting) instead of a hardcoded 0.5.
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

    mc = _ensure_isin_column(mc)
    kal = _ensure_isin_column(kal)
    pt = _ensure_isin_column(pt)
    earn = _ensure_isin_column(earn)
    anomaly_results = _ensure_isin_column(anomaly_results)
    credit = _ensure_isin_column(credit)
    div_safety = _ensure_isin_column(div_safety)

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
                "isin",
                "ticker",
                "expected_upside_mc",
                "implied_return_mc",
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
            kal[["isin", "expected_upside_kalman", "implied_return_kalman", "kalman_estimate"]],
            on="isin",
            how="inner",
        )
        .merge(
            pt[
                [
                    "isin",
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
            ],
            on="isin",
            how="inner",
        )
        .merge(
            earn[
                [
                    "isin",
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
            on="isin",
            how="inner",
        )
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
        "mh_anomaly_probability",
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
            anomaly_subset = anomaly_results[["isin"] + available_anomaly].drop_duplicates(
                subset="isin"
            )
            summary = summary.merge(anomaly_subset, on="isin", how="left")
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
    if credit is not None and not credit.empty and "isin" in credit.columns:
        available_credit = [c for c in _CREDIT_COLS if c in credit.columns]
        if available_credit:
            credit_subset = credit[["isin"] + available_credit].drop_duplicates(subset="isin")
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
            div_subset = div_safety[["isin"] + available_div].drop_duplicates(subset="isin")
            summary = summary.merge(div_subset, on="isin", how="left")
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
                mc[["isin", col]].drop_duplicates(subset="isin").set_index("isin")[col]
            )
            summary[col] = summary["isin"].map(price_map)
            logger.debug("Merged market-data column '%s' from mc", col)

    # Enrich from source_df
    if source_df is not None and "isin" in source_df.columns:
        id_cols_ordered = load_identifier_columns()
        desired_cols = id_cols_ordered + market_data_cols
        missing_cols = [
            c for c in desired_cols if c in source_df.columns and c not in summary.columns
        ]
        if missing_cols:
            source_subset = source_df[["isin"] + missing_cols].drop_duplicates(subset="isin")
            summary = summary.merge(source_subset, on="isin", how="left")
            logger.info(
                "Enriched expected_returns_summary with %d columns from mv_all_stock_features",
                len(missing_cols),
            )

        # P2/Task 4: Merge forward consensus columns for cross-model diagnostics
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
            fwd_subset = source_df[["isin"] + fwd_missing].drop_duplicates(subset="isin")
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
        ]
        available_quad_cols = [c for c in _quad_score_cols if c in quad.columns]
        quad_scores = quad[["isin"] + available_quad_cols].drop_duplicates(subset="isin")
        # Drop any overlapping columns already in summary before merge
        overlap = [c for c in available_quad_cols if c in summary.columns]
        if overlap:
            summary = summary.drop(columns=overlap)
        summary = summary.merge(quad_scores, on="isin", how="left")
        # Derive bullish flags from quad for consistency
        if "mc_bullish" not in summary.columns:
            summary["mc_bullish"] = summary.get("implied_return_mc", 0) > 2.0
        if "kal_bullish" not in summary.columns:
            summary["kal_bullish"] = summary.get("implied_return_kalman", 0) > 2.0
        if "pt_bullish" not in summary.columns:
            summary["pt_bullish"] = summary.get("implied_return_pt", 0) > 2.0
        if "earn_bullish" not in summary.columns:
            summary["earn_bullish"] = summary["prob_beat_given_momentum"] >= 0.6
        # Use directional_agreement as the primary agreement_score
        summary["agreement_score"] = summary["directional_agreement"]
        logger.info("Merged quad alignment scores into summary (single source of truth)")
    else:
        # Fallback: compute direction flags locally (legacy path)
        summary["mc_bullish"] = summary["implied_return_mc"] > 0
        summary["kal_bullish"] = summary["implied_return_kalman"] > 0
        summary["pt_bullish"] = summary["implied_return_pt"] > 0
        summary["earn_bullish"] = summary["prob_beat_given_momentum"] >= 0.6
        summary["agreement_score"] = (
            summary["mc_bullish"].astype(int)
            + summary["kal_bullish"].astype(int)
            + summary["pt_bullish"].astype(int)
            + summary["earn_bullish"].astype(int)
        )
        summary["signal"] = summary["agreement_score"].map(_SIGNAL_LABELS_4)

    # Issue 7: Kalman variance-based confidence weighting
    mc_weight = summary["prob_positive_upside"].clip(0, 100) / 100.0
    if "kalman_variance" in summary.columns:
        max_var = summary["kalman_variance"].quantile(0.95)
        if max_var > 0:
            kal_weight = (
                1 - summary["kalman_variance"].clip(0, max_var) / max_var
            ).clip(0.2, 0.9)
        else:
            kal_weight = 0.5
    else:
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

    # Incorporate FCF estimate curve into weighted agreement
    if "fcf_est_trend" in summary.columns:
        fcf_weight = summary["fcf_est_trend"].clip(-1, 1) * 0.3
        summary["weighted_agreement"] = summary["weighted_agreement"] + fcf_weight

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
    min_prob_positive: float = 65.0,
    min_achievement: float = 0.50,
    top_n: int = 2000,
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
