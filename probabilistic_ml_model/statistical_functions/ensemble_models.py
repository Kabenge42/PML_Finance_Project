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
    bullish_return_threshold: float = 0.02,  # v3.9: was 0.0 — heavy tails penalise marginal bulls
    bma_weights: tuple[float, float, float] = (0.45, 0.25, 0.30),  # (MC, Kalman, PT)
    use_log_score_reweighting: bool = True,
    cvar_alpha: float = 0.05,
    student_t_df: float | None = None,  # v3.9: passed from mcmc_result for tail-aware conviction
) -> pd.DataFrame:
    """
    Merge Monte Carlo, Kalman, and Price Target Achievement into a
    tri-model alignment DataFrame with direction agreement scores.

    Parameters
    ----------
    use_log_score_reweighting : bool
    student_t_df : float | None
        passed from mcmc_result for tail-aware conviction
    cvar_alpha
    bma_weights : tuple[float, float, float]
    bullish_return_threshold : float
        Minimum implied return (%) to classify a model as bullish.
        Default 2.0% prevents near-zero returns from inflating the
        "Strong Bullish" bucket (Issue 8).
        :param bullish_return_threshold:
        :param pt:
        :param kal:
        :param mc:
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

    # Issue 8: scale-aware materiality threshold for bullish classification
    # Use model-specific thresholds based on each model's distribution
    # to prevent scale mismatch from biasing agreement scores
    mc_threshold = max(
        bullish_return_threshold, float(tri["implied_return_mc"].quantile(0.40))
    )
    kal_threshold = max(
        bullish_return_threshold, float(tri["implied_return_kalman"].quantile(0.40))
    )
    pt_threshold = max(
        bullish_return_threshold, float(tri["implied_return_pt"].quantile(0.40))
    )

    tri["mc_bullish"] = tri["implied_return_mc"] > mc_threshold
    tri["kal_bullish"] = tri["implied_return_kalman"] > kal_threshold
    tri["pt_bullish"] = tri["implied_return_pt"] > pt_threshold
    tri["agreement_score"] = (
        tri["mc_bullish"].astype(int)
        + tri["kal_bullish"].astype(int)
        + tri["pt_bullish"].astype(int)
    )
    tri["signal"] = tri["agreement_score"].map(_SIGNAL_LABELS)

    # v3.9: Bayesian Model Averaging blended expected return (Finding #3)
    w_mc, w_kal, w_pt = bma_weights
    weight_sum = w_mc + w_kal + w_pt
    if weight_sum > 0:
        w_mc, w_kal, w_pt = w_mc / weight_sum, w_kal / weight_sum, w_pt / weight_sum
    tri["blended_return_bma"] = (
        w_mc * tri["implied_return_mc"]
        + w_kal * tri["implied_return_kalman"]
        + w_pt * tri["implied_return_pt"]
    )

    # v3.9 Cross-cutting T-A: Per-stock tail_df sourced from each model's
    # *Result dataclass (CreditRiskResult.tail_df, DividendSafetyResult.tail_df,
    # PriceTargetResult.tail_df). When a per-stock ``tail_df`` column is
    # present on any of the input frames we use it to compute a **per-stock**
    # tail penalty rather than broadcasting the global ``student_t_df``.
    # Falls back to the global scalar when no per-stock column is available
    # (backwards compatible with prior v3.8 behaviour).
    per_stock_tail_df: pd.Series | None = None
    for src in (pt, mc, kal):
        if "tail_df" in src.columns:
            cand = (
                src[["isin", "tail_df"]].dropna(subset=["tail_df"]).drop_duplicates(subset=["isin"])
            )
            if not cand.empty:
                per_stock_tail_df = cand.set_index("isin")["tail_df"]
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
        # Backfill with the global scalar when individual rows are missing
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

    # v3.9: Expose CVaR column when available on MC output (Finding #3 risk management)
    cvar_col = f"cvar_{int(cvar_alpha * 100)}"
    if cvar_col in mc.columns:
        mc_cvar = mc[["isin", cvar_col]].drop_duplicates(subset=["isin"])
        tri = tri.merge(mc_cvar, on="isin", how="left")
    elif "var_5_pct" in tri.columns:
        # Fallback: approximate CVaR from VaR when CVaR isn't emitted yet
        tri[cvar_col] = tri["var_5_pct"]

    logger.info(
        "Tri-model alignment: %d stocks, %d strong bullish (tail_penalty=%.2f)",
        len(tri),
        (tri["agreement_score"] == 3).sum(),
        tail_penalty,
    )
    return tri


def build_quad_model_alignment(
    tri: pd.DataFrame,
    beat: pd.DataFrame,
    beat_threshold: float = 0.55,  # v3.9: was 0.50 — higher bar given tail risk
    credit: pd.DataFrame | None = None,
    div_safety: pd.DataFrame | None = None,
    anomaly: pd.DataFrame | None = None,
    *,
    bma_weights: dict[str, float] | None = None,  # v3.9: full six-model BMA weights
    credit_distress_threshold: float = 0.90,  # v3.9: softened from 0.99
    div_cut_threshold: float = 0.60,  # v3.9: softened from 0.75
    anomaly_severity_threshold: float | None = None,
    mcmc_result: dict | None = None,
    use_macro_tilt: bool = True,  # v3.9: regional tilt from macro covariates
    # v3.10 T-E — optional realised outcomes per model for centralised BMA
    # weighting via ``ModelConfidenceEstimator.compute_relative_confidence``.
    # Pass a mapping ``{model_name: (probs, outcomes)}`` and ``bma_weights``
    # will be overridden with the log-score-derived softmax weights. Falls
    # back to the static dict when ``None`` (default / backwards compatible).
    validation_outcomes: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
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
    bma_weights
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
    mcmc_result : pd.DataFrame or None
        MCMC results with ``anomaly_severity_score``.

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

    # v3.10 T-E — centralise BMA log-score weighting in
    # ``ModelConfidenceEstimator.compute_relative_confidence`` when the caller
    # provides realised outcomes per model. Backwards compatible: unchanged
    # behaviour when ``validation_outcomes`` is ``None``.
    if validation_outcomes:
        try:
            from probabilistic_ml_model.statistical_functions.probability_models import (
                ModelConfidenceEstimator,
            )

            _mce = ModelConfidenceEstimator(n_bins=10, use_quantile_bins=True)
            _rel = _mce.compute_relative_confidence(validation_outcomes, bootstrap_iters=0)
            if not _rel.empty and "bma_weight" in _rel.columns:
                learned = {str(r.model): float(r.bma_weight) for r in _rel.itertuples()}
                # Merge learned weights into the default dict, keeping entries
                # for models without validation data.
                bma_weights = {**bma_weights, **learned}
                logger.info(
                    "Quad-model BMA weights overridden via ModelConfidenceEstimator (T-E): %s",
                    {k: round(v, 4) for k, v in learned.items()},
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("T-E BMA weight override skipped: %s", exc)

    _bma_total = sum(bma_weights.values()) or 1.0
    bma_weights = {k: v / _bma_total for k, v in bma_weights.items()}

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
    quad["risk_quality_score"] = quad["credit_safe"] + quad["div_safe"] + quad["anomaly_clean"]
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
    if mcmc_result.get("posterior_mean") is not None:
        mcmc_mu = mcmc_result["posterior_mean"]
        mcmc_std = mcmc_result.get("posterior_std", 1.0)

        # Per-stock shrinkage: higher ensemble variance → more shrinkage toward prior
        stock_std = quad[["implied_return_mc", "implied_return_kalman", "implied_return_pt"]].std(
            axis=1
        )
        shrinkage = (stock_std**2) / (stock_std**2 + mcmc_std**2)
        # shrinkage ∈ [0, 1]: 0 = trust prior, 1 = trust stock estimate
        quad["mcmc_shrinkage"] = shrinkage
        quad["ensemble_return_shrunk"] = (
            shrinkage * quad["ensemble_return"] + (1 - shrinkage) * mcmc_mu
        )
    else:
        quad["ensemble_return_shrunk"] = quad["ensemble_return"]
        quad["mcmc_shrinkage"] = 1.0  # no shrinkage

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
            # Blend: where sector prior exists, use it instead of global
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

    # Robustify heavy-tailed metrics to prevent outlier domination
    _HEAVY_TAIL_COLS = ["pt_spread", "risk_reward_ratio", "upside_std"]
    for col in _HEAVY_TAIL_COLS:
        if col in summary.columns:
            lo, hi = summary[col].quantile(0.02), summary[col].quantile(0.98)
            summary[col] = summary[col].clip(lo, hi)

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
            "ensemble_return",
            "ensemble_return_shrunk",
            "mcmc_shrinkage",
            "risk_adj_return",
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
            kal_weight = (1 - summary["kalman_variance"].clip(0, max_var) / max_var).clip(0.2, 0.9)
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

    # Pick the best available expected-return column for risk adjustment
    _ret_src = None
    for _c in ("blended_return_bma", "ensemble_return", "implied_return_mc"):
        if _c in summary.columns:
            _ret_src = _c
            break
    if _ret_src is not None:
        summary["risk_adjusted_expected_return"] = summary[_ret_src] * haircut

    # CVaR column (surfaced from tri/mc); fallback to VaR if absent
    if "cvar_5" not in summary.columns:
        if "var_5_pct" in summary.columns:
            summary["cvar_5"] = summary["var_5_pct"]
        else:
            summary["cvar_5"] = np.nan

    # Position size weight: inversely proportional to posterior σ × CI width
    _post_std = (
        summary["mcmc_posterior_std"]
        if "mcmc_posterior_std" in summary.columns
        else summary.get("posterior_std", pd.Series(1.0, index=summary.index))
    )
    _ci_width = (
        summary.get("ci_width")
        if "ci_width" in summary.columns
        else (summary.get("ci_upper_95", 1.0) - summary.get("ci_lower_95", 0.0))
    )
    try:
        _post_std_c = pd.to_numeric(_post_std, errors="coerce").clip(lower=1e-4)
        _ci_c = pd.to_numeric(_ci_width, errors="coerce").clip(lower=1e-4)
        summary["position_size_weight"] = 1.0 / (_post_std_c * _ci_c)
    except Exception:  # pragma: no cover — defensive fallback
        summary["position_size_weight"] = np.nan

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
    min_prob_positive: float = 50.0,  # v3.9: was 33.0 — tighter given df≈2 tail risk
    min_achievement: float = 0.60,  # v3.9: was 0.50
    top_n: int = 1500,  # v3.9: was 2000
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
