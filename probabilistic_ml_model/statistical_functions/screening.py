"""
Stock screening and filtering utilities for feature analytics.

This module provides functions for:
- Multi-factor stock screening
- Quality scoring and ranking
- Feature-based filtering
- Investment strategy screening
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Dynamic threshold computation from statistical distributions
# =============================================================================


def _compute_dynamic_thresholds(
    df: pd.DataFrame,
    feature_threshold_specs: dict[str, dict],
) -> dict[str, float]:
    """
    Compute screening thresholds dynamically from data distributions.

    Uses ``run_category_probability_analytics`` to fit distributions and
    derive percentile-based or posterior-based cutoffs for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input data used to estimate distributions.
    feature_threshold_specs : dict[str, dict]
        Mapping of feature name -> spec dict with keys:
        - 'direction': 'min' (keep above threshold) or 'max' (keep below)
        - 'percentile': target percentile for the cutoff (e.g. 25 for Q1)
        - 'fallback': hardcoded default if the feature is missing or
          has insufficient data

    Returns
    -------
    dict[str, float]
        Mapping of feature name -> computed threshold value.
    """
    from probabilistic_ml_model.statistical_functions.statistical_models import (
        run_category_probability_analytics,
    )

    features = [f for f in feature_threshold_specs if f in df.columns]
    thresholds: dict[str, float] = {}

    if not features:
        return {f: spec["fallback"] for f, spec in feature_threshold_specs.items()}

    analytics = run_category_probability_analytics(
        df,
        category_name="screening_threshold_estimation",
        features=features,
    )

    bayesian = analytics.get("bayesian_results", {})
    dist_fits = analytics.get("distribution_fits", {})
    summary = analytics.get("summary_statistics", {})

    for feat, spec in feature_threshold_specs.items():
        direction = spec["direction"]
        target_pct = spec["percentile"]
        fallback = spec["fallback"]

        if feat not in df.columns or feat not in summary:
            thresholds[feat] = fallback
            continue

        data = df[feat].dropna()
        if len(data) < 30:
            thresholds[feat] = fallback
            continue

        # Strategy 1: Use fitted distribution quantile (most accurate)
        if feat in dist_fits:
            fit_info = dist_fits[feat]
            dist_name = fit_info.get("best_distribution")
            params = fit_info.get("params")
            if dist_name and params:
                from scipy import stats as sp_stats

                dist_map = {
                    "normal": sp_stats.norm,
                    "student_t": sp_stats.t,
                    "skew_normal": sp_stats.skewnorm,
                }
                dist_obj = dist_map.get(dist_name)
                if dist_obj is not None:
                    try:
                        thresholds[feat] = float(dist_obj.ppf(target_pct / 100.0, *params))
                        continue
                    except (ValueError, TypeError):
                        pass

        # Strategy 2: Use Bayesian posterior credible interval
        if feat in bayesian:
            post = bayesian[feat]
            post_mean = post.get("posterior_mean", fallback)
            post_std = post.get("posterior_std", 0)
            if post_std > 0:
                from scipy import stats as sp_stats

                thresholds[feat] = float(sp_stats.norm.ppf(target_pct / 100.0, post_mean, post_std))
                continue

        # Strategy 3: Empirical percentile fallback
        thresholds[feat] = float(data.quantile(target_pct / 100.0))

    return thresholds


# =============================================================================
# Column resolution helpers (align with equities_schema_metadata aliases)
# =============================================================================


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name present in df, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def create_enhanced_screener(
    df: pd.DataFrame,
    min_fscore: int | None = None,
    min_quality_momentum: float | None = None,
    max_distress_risk: float | None = None,
    min_eps_trajectory: float | None = None,
    min_fcf_positive_years: int | None = None,
    require_deleveraging: bool = True,
    require_secular_trend: bool = True,
    sector_filter: str = "All",
) -> pd.DataFrame:
    """
    Enhanced stock screener with multiple quality and momentum criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock features
    min_fscore : int or None
        Minimum Piotroski F-Score (0-9). None -> derived from distribution
        (25th percentile of fitted distribution).
    min_quality_momentum : float or None
        Minimum quality momentum score. None -> 25th percentile.
    max_distress_risk : float or None
        Maximum distress risk score (inverted: higher = safer).
        None -> 75th percentile.
    min_eps_trajectory : float or None
        Minimum EPS trajectory score. None -> 25th percentile.
    min_fcf_positive_years : int or None
        Minimum FCF positive years (0-5). None -> 25th percentile.
    require_deleveraging : bool, default False
        Only stocks actively reducing debt
    require_secular_trend : bool, default False
        Only stocks in secular uptrend
    sector_filter : str, default 'All'
        Filter by sector ('All' for no filter)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame sorted by quality

    Examples
    --------
    >>> screened = create_enhanced_screener(df)  # fully dynamic thresholds
    """
    # Ensure necessary columns exist
    required_cols = [
        "piotroski_f_score",
        "combined_distress_score",
        "eps_trajectory_score",
        "fcf_positive_years",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        return pd.DataFrame()

    # Build specs only for parameters that need dynamic computation
    specs: dict[str, dict] = {}
    if min_fscore is None:
        specs["piotroski_f_score"] = {"direction": "min", "percentile": 25, "fallback": 5}
    if max_distress_risk is None:
        specs["combined_distress_score"] = {
            "direction": "min",
            "percentile": 25,
            "fallback": 30,
        }
    if min_eps_trajectory is None:
        specs["eps_trajectory_score"] = {"direction": "min", "percentile": 25, "fallback": 40}
    if min_fcf_positive_years is None:
        specs["fcf_positive_years"] = {"direction": "min", "percentile": 25, "fallback": 3}

    qm_col = _resolve_col(df, "quality_momentum_score", "quality_momentum")
    if min_quality_momentum is None and qm_col is not None:
        specs[qm_col] = {"direction": "min", "percentile": 25, "fallback": 40}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    # Resolve final values: explicit parameter wins, else dynamic, else fallback
    eff_min_fscore = (
        min_fscore if min_fscore is not None else int(dynamic.get("piotroski_f_score", 5))
    )
    eff_max_distress = (
        max_distress_risk
        if max_distress_risk is not None
        else dynamic.get("combined_distress_score", 30)
    )
    eff_min_eps_traj = (
        min_eps_trajectory
        if min_eps_trajectory is not None
        else dynamic.get("eps_trajectory_score", 40)
    )
    eff_min_fcf_yrs = (
        min_fcf_positive_years
        if min_fcf_positive_years is not None
        else int(dynamic.get("fcf_positive_years", 3))
    )
    eff_min_qm = (
        min_quality_momentum
        if min_quality_momentum is not None
        else dynamic.get(qm_col, 40) if qm_col else 40
    )

    # Apply filters
    mask = (
        (df["piotroski_f_score"] >= eff_min_fscore)
        & (df["combined_distress_score"] >= (100 - eff_max_distress))
        & (df["eps_trajectory_score"] >= eff_min_eps_traj)
        & (df["fcf_positive_years"] >= eff_min_fcf_yrs)
    )

    if require_deleveraging and "debt_deleveraging" in df.columns:
        mask &= df["debt_deleveraging"] == 1

    if require_secular_trend and "secular_trend_flag" in df.columns:
        mask &= df["secular_trend_flag"] == 1

    if qm_col is not None:
        mask &= df[qm_col] >= eff_min_qm

    filtered = df[mask].copy()

    if sector_filter != "All":
        sector_col = "industry" if "industry" in filtered.columns else "sector"
        if sector_col in filtered.columns:
            filtered = filtered[filtered[sector_col] == sector_filter]

    # Sort by composite quality
    if "piotroski_f_score" in filtered.columns:
        filtered = filtered.sort_values("piotroski_f_score", ascending=False)

    return filtered


def screen_earnings_quality(
    df: pd.DataFrame,
    min_quality_score: float | None = None,
    max_adjustment_pct: float | None = None,
    require_positive_revisions: bool = False,
    min_positive_years: int | None = None,
    sector_filter: str = "All",
) -> pd.DataFrame:
    """
    Screen stocks based on earnings quality criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_quality_score : float or None
        Minimum earnings quality composite score (0-100). None -> 25th pctl.
    max_adjustment_pct : float or None
        Maximum absolute EPS adjustment percentage. None -> 75th pctl.
    require_positive_revisions : bool, default False
        Only include stocks with positive GAAP revision flag
    min_positive_years : int or None
        Minimum net income positive years (0-5). None -> 25th pctl.
    sector_filter : str, default 'All'
        Filter by sector

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame sorted by earnings quality

    Examples
    --------
    >>> high_quality = screen_earnings_quality(df)
    >>> print(f"Found {len(high_quality)} high-quality earnings stocks")
    """
    specs: dict[str, dict] = {}
    if min_quality_score is None:
        specs["earnings_quality_composite"] = {"direction": "min", "percentile": 25, "fallback": 60}
    if max_adjustment_pct is None:
        specs["eps_adjustment_pct"] = {"direction": "max", "percentile": 75, "fallback": 20}
    if min_positive_years is None:
        specs["net_income_positive_years"] = {"direction": "min", "percentile": 25, "fallback": 3}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_min_quality = (
        min_quality_score
        if min_quality_score is not None
        else dynamic.get("earnings_quality_composite", 60)
    )
    eff_max_adj = (
        max_adjustment_pct
        if max_adjustment_pct is not None
        else dynamic.get("eps_adjustment_pct", 20)
    )
    eff_min_pos_yrs = (
        min_positive_years
        if min_positive_years is not None
        else int(dynamic.get("net_income_positive_years", 3))
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if "earnings_quality_composite" in df.columns:
        mask &= df["earnings_quality_composite"] >= eff_min_quality

    if "eps_adjustment_pct" in df.columns:
        mask &= df["eps_adjustment_pct"].abs() <= eff_max_adj

    if require_positive_revisions and "gaap_positive_revision_flag" in df.columns:
        mask &= df["gaap_positive_revision_flag"] == 1

    if "net_income_positive_years" in df.columns:
        mask &= df["net_income_positive_years"] >= eff_min_pos_yrs

    if sector_filter != "All":
        sector_col = "industry" if "industry" in df.columns else "sector"
        if sector_col in df.columns:
            mask &= df[sector_col] == sector_filter

    result = df[mask].copy()

    if "earnings_quality_composite" in result.columns:
        result = result.sort_values("earnings_quality_composite", ascending=False)

    return result


def screen_value_opportunities(
    df: pd.DataFrame,
    max_pe_ratio: float | None = None,
    min_upside_potential: float | None = None,
    max_price_to_tangible_book: float | None = None,
    min_quality_score: float | None = None,
    require_positive_fcf: bool = True,
) -> pd.DataFrame:
    """
    Screen for value investment opportunities.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    max_pe_ratio : float or None
        Maximum P/E ratio. None -> 75th percentile of fitted distribution.
    min_upside_potential : float or None
        Minimum analyst upside potential (%). None -> 25th pctl.
    max_price_to_tangible_book : float or None
        Maximum price to tangible book ratio. None -> 75th pctl.
    min_quality_score : float or None
        Minimum quality score. None -> 25th pctl.
    require_positive_fcf : bool, default True
        Require positive free cash flow

    Returns
    -------
    pd.DataFrame
        Value opportunities sorted by upside potential

    Examples
    --------
    >>> value_stocks = screen_value_opportunities(df)
    """
    specs: dict[str, dict] = {}
    if max_pe_ratio is None:
        specs["p_e_ratio"] = {"direction": "max", "percentile": 75, "fallback": 30}
    if min_upside_potential is None:
        specs["expected_upside_pt"] = {
            "direction": "min",
            "percentile": 25,
            "fallback": 15,
        }
    if max_price_to_tangible_book is None:
        specs["price_to_tangible_book"] = {"direction": "max", "percentile": 75, "fallback": 2.0}
    if min_quality_score is None:
        specs["piotroski_f_score"] = {"direction": "min", "percentile": 25, "fallback": 5}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_max_pe = max_pe_ratio if max_pe_ratio is not None else dynamic.get("p_e_ratio", 30)
    eff_min_upside = (
        min_upside_potential
        if min_upside_potential is not None
        else dynamic.get("expected_upside_pt", 15)
    )
    eff_max_ptb = (
        max_price_to_tangible_book
        if max_price_to_tangible_book is not None
        else dynamic.get("price_to_tangible_book", 2.0)
    )
    eff_min_quality = (
        min_quality_score
        if min_quality_score is not None
        else dynamic.get("piotroski_f_score", 5) * 10
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if "p_e_ratio" in df.columns:
        mask &= (df["p_e_ratio"] > 0) & (df["p_e_ratio"] <= eff_max_pe)

    if "expected_upside_pt" in df.columns:
        mask &= df["expected_upside_pt"] >= eff_min_upside

    if "price_to_tangible_book" in df.columns:
        mask &= (df["price_to_tangible_book"] > 0) & (df["price_to_tangible_book"] <= eff_max_ptb)

    if "piotroski_f_score" in df.columns:
        mask &= df["piotroski_f_score"] >= eff_min_quality / 10

    if require_positive_fcf and "fcf_yield" in df.columns:
        mask &= df["fcf_yield"] > 0

    result = df[mask].copy()

    if "expected_upside_pt" in result.columns:
        result = result.sort_values("expected_upside_pt", ascending=False)

    return result


def screen_growth_momentum(
    df: pd.DataFrame,
    min_revenue_growth: float | None = None,
    min_eps_growth: float | None = None,
    min_price_momentum_1y: float | None = None,
    min_secular_trend_score: float | None = None,
    require_rnd_investment: bool = False,
) -> pd.DataFrame:
    """
    Screen for growth and momentum stocks.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_revenue_growth : float or None
        Minimum revenue YoY growth (%). None -> 25th pctl.
    min_eps_growth : float or None
        Minimum EPS YoY growth (%). None -> 25th pctl.
    min_price_momentum_1y : float or None
        Minimum 1-year price momentum (%). None -> 25th pctl.
    min_secular_trend_score : float or None
        Minimum long-term trend score. None -> 25th pctl.
    require_rnd_investment : bool, default False
        Require R&D investment

    Returns
    -------
    pd.DataFrame
        Growth stocks sorted by momentum

    Examples
    --------
    >>> growth_stocks = screen_growth_momentum(df)
    """
    rev_col = _resolve_col(df, "revenue_yoy_growth", "revenue_growth_yoy")

    specs: dict[str, dict] = {}
    if min_revenue_growth is None and rev_col is not None:
        specs[rev_col] = {"direction": "min", "percentile": 25, "fallback": 5}
    if min_eps_growth is None:
        specs["eps_yoy_growth"] = {"direction": "min", "percentile": 25, "fallback": 10}
    if min_price_momentum_1y is None:
        specs["price_momentum_1y"] = {"direction": "min", "percentile": 25, "fallback": 0}
    if min_secular_trend_score is None:
        specs["long_term_trend_score"] = {"direction": "min", "percentile": 25, "fallback": 1}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_min_rev = (
        min_revenue_growth
        if min_revenue_growth is not None
        else dynamic.get(rev_col, 5) if rev_col else 5
    )
    eff_min_eps = (
        min_eps_growth if min_eps_growth is not None else dynamic.get("eps_yoy_growth", 10)
    )
    eff_min_mom = (
        min_price_momentum_1y
        if min_price_momentum_1y is not None
        else dynamic.get("price_momentum_1y", 0)
    )
    eff_min_trend = (
        min_secular_trend_score
        if min_secular_trend_score is not None
        else dynamic.get("long_term_trend_score", 1)
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if rev_col is not None:
        mask &= df[rev_col] >= eff_min_rev

    if "eps_yoy_growth" in df.columns:
        mask &= df["eps_yoy_growth"] >= eff_min_eps

    if "price_momentum_1y" in df.columns:
        mask &= df["price_momentum_1y"] >= eff_min_mom

    if "long_term_trend_score" in df.columns:
        mask &= df["long_term_trend_score"] >= eff_min_trend

    if require_rnd_investment and "rnd_intensity_ltm" in df.columns:
        mask &= df["rnd_intensity_ltm"] > 0

    result = df[mask].copy()

    if "long_term_trend_score" in result.columns:
        result = result.sort_values("long_term_trend_score", ascending=False)

    return result


def screen_dividend_quality(
    df: pd.DataFrame,
    min_dividend_yield: float | None = None,
    min_dividend_streak: int | None = None,
    max_payout_ratio: float | None = None,
    min_fcf_coverage: float | None = None,
    require_dividend_growth: bool = True,
) -> pd.DataFrame:
    """
    Screen for quality dividend stocks.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_dividend_yield : float or None
        Minimum dividend yield (%). None -> 25th pctl.
    min_dividend_streak : int or None
        Minimum consecutive years of dividends. None -> 25th pctl.
    max_payout_ratio : float or None
        Maximum dividend payout ratio (%). None -> 75th pctl.
    min_fcf_coverage : float or None
        Minimum FCF dividend coverage ratio. None -> 25th pctl.
    require_dividend_growth : bool, default True
        Require expected dividend growth

    Returns
    -------
    pd.DataFrame
        Dividend stocks sorted by yield

    Examples
    --------
    >>> dividend_stocks = screen_dividend_quality(df)
    """
    yield_col = "dividend_yield_ltm" if "dividend_yield_ltm" in df.columns else "dividend_yield"

    specs: dict[str, dict] = {}
    if min_dividend_yield is None:
        specs[yield_col] = {"direction": "min", "percentile": 25, "fallback": 0}
    if min_dividend_streak is None:
        specs["dividend_streak"] = {"direction": "min", "percentile": 25, "fallback": 3}
    if max_payout_ratio is None:
        specs["dividend_payout_ratio"] = {"direction": "max", "percentile": 75, "fallback": 0}
    if min_fcf_coverage is None:
        specs["fcf_dividend_coverage"] = {"direction": "min", "percentile": 25, "fallback": 1.2}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_min_yield = (
        min_dividend_yield if min_dividend_yield is not None else dynamic.get(yield_col, 0)
    )
    eff_min_streak = (
        min_dividend_streak
        if min_dividend_streak is not None
        else int(dynamic.get("dividend_streak", 3))
    )
    eff_max_payout = (
        max_payout_ratio
        if max_payout_ratio is not None
        else dynamic.get("dividend_payout_ratio", 0)
    )
    eff_min_fcf_cov = (
        min_fcf_coverage
        if min_fcf_coverage is not None
        else dynamic.get("fcf_dividend_coverage", 1.2)
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if yield_col in df.columns:
        mask &= df[yield_col] >= eff_min_yield

    if "dividend_streak" in df.columns:
        mask &= df["dividend_streak"] >= eff_min_streak

    if "dividend_payout_ratio" in df.columns:
        mask &= df["dividend_payout_ratio"] <= eff_max_payout

    if "fcf_dividend_coverage" in df.columns:
        mask &= df["fcf_dividend_coverage"] >= eff_min_fcf_cov

    if require_dividend_growth and "dividend_growth_expectation" in df.columns:
        mask &= df["dividend_growth_expectation"] > 0

    result = df[mask].copy()

    if yield_col in result.columns:
        result = result.sort_values(yield_col, ascending=False)

    return result


def screen_valuation_reversion_candidates(
    df: pd.DataFrame,
    min_discount_pct: float | None = None,
    min_quality_score: float | None = None,
    max_distress_risk: float | None = None,
) -> pd.DataFrame:
    """
    Find stocks trading at a deep discount to their 3-year historical mean
    while maintaining stable fundamental scores.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_discount_pct : float or None
        Minimum discount % vs 3Y avg. None -> derived from p_e_vs_3y_avg
        distribution (25th percentile inverted).
    min_quality_score : float or None
        Minimum quality score. None -> 25th pctl.
    max_distress_risk : float or None
        Maximum distress risk. None -> 75th pctl.

    Features: p_e_vs_3y_avg, ev_ebitda_vs_3y_avg, p_b_momentum_yoy
    """
    specs: dict[str, dict] = {}
    if min_discount_pct is None:
        specs["p_e_vs_3y_avg"] = {"direction": "max", "percentile": 25, "fallback": 80}
    if min_quality_score is None:
        specs["piotroski_f_score"] = {"direction": "min", "percentile": 25, "fallback": 5}
    if max_distress_risk is None:
        specs["combined_distress_score"] = {
            "direction": "min",
            "percentile": 25,
            "fallback": 60,
        }

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    # For discount: threshold on the ratio column directly (lower = more discounted)
    eff_discount_threshold = (
        (100 - min_discount_pct)
        if min_discount_pct is not None
        else dynamic.get("p_e_vs_3y_avg", 80)
    )
    eff_min_quality = (
        min_quality_score
        if min_quality_score is not None
        else dynamic.get("piotroski_f_score", 5) * 10
    )
    eff_max_distress = (
        max_distress_risk
        if max_distress_risk is not None
        else 100 - dynamic.get("combined_distress_score", 60)
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if "p_e_vs_3y_avg" in df.columns:
        mask &= df["p_e_vs_3y_avg"] <= eff_discount_threshold

    if "ev_ebitda_vs_3y_avg" in df.columns:
        mask &= df["ev_ebitda_vs_3y_avg"] <= eff_discount_threshold

    if "piotroski_f_score" in df.columns:
        mask &= df["piotroski_f_score"] >= (eff_min_quality / 10)

    if "combined_distress_score" in df.columns:
        mask &= df["combined_distress_score"] >= (100 - eff_max_distress)

    result = df[mask].copy()

    sort_col = "p_e_vs_3y_avg" if "p_e_vs_3y_avg" in result.columns else "ev_ebitda_vs_3y_avg"
    if sort_col in result.columns:
        result = result.sort_values(sort_col)

    return result


def screen_integrity_filtered_growth(
    df: pd.DataFrame,
    min_revenue_growth: float | None = None,
    min_accounting_quality: float | None = None,
    max_dilution_score: float | None = None,
) -> pd.DataFrame:
    """
    Growth portfolio filter that excludes companies with low accounting quality
    or high dilution.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_revenue_growth : float or None
        Minimum revenue growth %. None -> 75th pctl (growth filter is selective).
    min_accounting_quality : float or None
        Minimum accounting quality score. None -> 25th pctl.
    max_dilution_score : float or None
        Maximum dilution score. None -> 75th pctl.

    Features: accounting_quality_score, dilution_score, merger_impact_ratio
    """
    rev_col = _resolve_col(df, "revenue_yoy_growth", "revenue_growth_yoy")

    specs: dict[str, dict] = {}
    if min_revenue_growth is None and rev_col is not None:
        specs[rev_col] = {"direction": "min", "percentile": 75, "fallback": 15}
    if min_accounting_quality is None:
        specs["accounting_quality_score"] = {"direction": "min", "percentile": 25, "fallback": 60}
    if max_dilution_score is None:
        specs["dilution_score"] = {"direction": "max", "percentile": 75, "fallback": 40}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_min_rev = (
        min_revenue_growth
        if min_revenue_growth is not None
        else dynamic.get(rev_col, 15) if rev_col else 15
    )
    eff_min_acct = (
        min_accounting_quality
        if min_accounting_quality is not None
        else dynamic.get("accounting_quality_score", 60)
    )
    eff_max_dilution = (
        max_dilution_score if max_dilution_score is not None else dynamic.get("dilution_score", 40)
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if rev_col is not None:
        mask &= df[rev_col] >= eff_min_rev

    if "accounting_quality_score" in df.columns:
        mask &= df["accounting_quality_score"] >= eff_min_acct

    if "dilution_score" in df.columns:
        mask &= df["dilution_score"] <= eff_max_dilution

    if "merger_impact_ratio" in df.columns:
        mask &= df["merger_impact_ratio"] <= 30.0

    result = df[mask].copy()

    if "accounting_quality_score" in result.columns:
        result = result.sort_values("accounting_quality_score", ascending=False)

    return result


def screen_financial_health(
    df: pd.DataFrame,
    min_distress_score: float | None = None,
    max_debt_to_equity: float | None = None,
    min_current_ratio: float | None = None,
    min_interest_coverage: float | None = None,
    require_positive_wc: bool = True,
) -> pd.DataFrame:
    """
    Screen for financially healthy companies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_distress_score : float or None
        Minimum distress risk score (higher = safer). None -> 25th pctl.
    max_debt_to_equity : float or None
        Maximum debt-to-equity ratio. None -> 75th pctl.
    min_current_ratio : float or None
        Minimum current ratio. None -> 25th pctl.
    min_interest_coverage : float or None
        Minimum interest coverage ratio. None -> 25th pctl.
    require_positive_wc : bool, default True
        Require positive working capital

    Returns
    -------
    pd.DataFrame
        Financially healthy stocks

    Examples
    --------
    >>> healthy_stocks = screen_financial_health(df)
    """
    interest_col = _resolve_col(df, "interest_coverage", "interest_coverage_ratio")

    specs: dict[str, dict] = {}
    if min_distress_score is None:
        specs["combined_distress_score"] = {
            "direction": "min",
            "percentile": 25,
            "fallback": 70,
        }
    if max_debt_to_equity is None:
        specs["debt_to_equity"] = {"direction": "max", "percentile": 75, "fallback": 1.0}
    if min_current_ratio is None:
        specs["current_ratio"] = {"direction": "min", "percentile": 25, "fallback": 1.5}
    if min_interest_coverage is None and interest_col is not None:
        specs[interest_col] = {"direction": "min", "percentile": 25, "fallback": 3.0}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_min_distress = (
        min_distress_score
        if min_distress_score is not None
        else dynamic.get("combined_distress_score", 70)
    )
    eff_max_dte = (
        max_debt_to_equity if max_debt_to_equity is not None else dynamic.get("debt_to_equity", 1.0)
    )
    eff_min_cr = (
        min_current_ratio if min_current_ratio is not None else dynamic.get("current_ratio", 1.5)
    )
    eff_min_ic = (
        min_interest_coverage
        if min_interest_coverage is not None
        else dynamic.get(interest_col, 3.0) if interest_col else 3.0
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if "combined_distress_score" in df.columns:
        mask &= df["combined_distress_score"] >= eff_min_distress

    if "debt_to_equity" in df.columns:
        mask &= df["debt_to_equity"] <= eff_max_dte

    if "current_ratio" in df.columns:
        mask &= df["current_ratio"] >= eff_min_cr

    if interest_col is not None:
        mask &= df[interest_col] >= eff_min_ic

    wc_col = _resolve_col(df, "wc_ltm", "working_capital_ltm", "wc_fq")
    if require_positive_wc and wc_col is not None:
        mask &= df[wc_col] > 0

    result = df[mask].copy()

    if "combined_distress_score" in result.columns:
        result = result.sort_values("combined_distress_score", ascending=False)

    return result


# Issue 5: Default weights for fundamental-only scoring (legacy)
_FUNDAMENTAL_WEIGHTS: dict[str, float] = {
    "piotroski_f_score": 0.25,
    "combined_distress_score": 0.25,
    "earnings_quality_composite": 0.25,
    "cash_flow_quality_score": 0.25,
}

# Issue 5: Model-aware weights blending fundamentals + probabilistic outputs
_MODEL_AWARE_WEIGHTS: dict[str, float] = {
    "piotroski_f_score": 0.20,
    "combined_distress_score": 0.10,
    "earnings_quality_composite": 0.10,
    "cash_flow_quality_score": 0.10,
    "prob_positive_upside": 0.10,
    "achievement_probability": 0.10,
    "prob_beat_given_momentum": 0.15,
    "confidence_score": 0.15,
}

# Normalization specs: column -> (min_val, max_val) for 0-100 scaling
_NORMALIZATION_SPECS: dict[str, tuple[float, float]] = {
    "piotroski_f_score": (0.0, 9.0),
    "prob_positive_upside": (0.0, 100.0),
    "achievement_probability": (0.0, 1.0),
    "prob_beat_given_momentum": (0.0, 1.0),
    "confidence_score": (0.0, 1.0),
}


def rank_stocks_by_composite_score(
    df: pd.DataFrame,
    weights: Optional[dict] = None,
    export: bool = False,
    use_model_aware: bool = True,
) -> pd.DataFrame:
    """
    Rank stocks by composite quality score with customizable weights.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    weights : dict, optional
        Dictionary of score weights. When ``None`` the function auto-selects
        model-aware weights (if probabilistic columns are present) or falls
        back to fundamental-only weights.
    export : bool
        Export results to analytics DB.
    use_model_aware : bool
        When ``True`` (default) and *weights* is ``None``, prefer
        ``_MODEL_AWARE_WEIGHTS`` when at least one probabilistic column
        is available in *df*.

    Returns
    -------
    pd.DataFrame
        DataFrame with composite_score column, sorted by score

    Examples
    --------
    >>> ranked = rank_stocks_by_composite_score(df)
    >>> top_10 = ranked.head(10)
    """
    if weights is None:
        _prob_cols = {
            "prob_positive_upside",
            "achievement_probability",
            "prob_beat_given_momentum",
            "confidence_score",
        }
        if use_model_aware and _prob_cols & set(df.columns):
            weights = dict(_MODEL_AWARE_WEIGHTS)
        else:
            weights = dict(_FUNDAMENTAL_WEIGHTS)

    result = df.copy()
    result["composite_score"] = 0.0

    for score_col, weight in weights.items():
        if score_col in result.columns:
            col_vals = result[score_col]
            # Normalize to 0-100 scale using known specs or percentile-based
            if score_col in _NORMALIZATION_SPECS:
                lo, hi = _NORMALIZATION_SPECS[score_col]
                normalized = (col_vals - lo) / (hi - lo) * 100 if hi > lo else col_vals
            else:
                normalized = col_vals
            result["composite_score"] += normalized.fillna(50) * weight

    result = result.sort_values("composite_score", ascending=False)

    if export:
        try:
            from probabilistic_ml_model.data_utils import export_to_analytics_db

            export_cols = ["ticker", "name", "sector", "industry", "composite_score"]
            available = [c for c in export_cols if c in result.columns]
            export_to_analytics_db(result[available], "composite_scores_statistics")
        except (ImportError, OSError) as e:
            logger.debug("Export to analytics DB failed: %s", e)

    return result


def create_sector_relative_ranking(
    df: pd.DataFrame, metric: str, sector_col: str = "industry"
) -> pd.DataFrame:
    """
    Create sector-relative rankings for a metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    metric : str
        Metric column name to rank
    sector_col : str, default 'industry'
        Sector grouping column

    Returns
    -------
    pd.DataFrame
        DataFrame with sector_rank and sector_percentile columns

    Examples
    --------
    >>> ranked = create_sector_relative_ranking(df, 'roe')
    >>> top_in_sector = ranked[ranked['sector_percentile'] > 75]
    """
    if metric not in df.columns or sector_col not in df.columns:
        return df

    result = df.copy()

    # Rank within sector
    result["sector_rank"] = result.groupby(sector_col)[metric].rank(ascending=False, method="min")

    # Calculate percentile within sector
    result["sector_percentile"] = result.groupby(sector_col)[metric].rank(pct=True) * 100

    return result


def screen_garp_opportunities(
    df: pd.DataFrame,
    max_peg_ratio: float | None = None,
    min_eps_growth: float | None = None,
    max_pe_ratio: float | None = None,
    min_quality_score: float | None = None,
) -> pd.DataFrame:
    """
    Screen for Growth at a Reasonable Price (GARP) opportunities.

    Combines growth criteria with valuation (PEG ratio) and quality.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    max_peg_ratio : float or None
        Maximum PEG ratio. None -> 75th pctl of fitted distribution.
    min_eps_growth : float or None
        Minimum EPS growth (%). None -> 25th pctl.
    max_pe_ratio : float or None
        Maximum P/E ratio. None -> 75th pctl.
    min_quality_score : float or None
        Minimum quality score (0-100). None -> 25th pctl.

    Returns
    -------
    pd.DataFrame
        GARP opportunities sorted by PEG ratio
    """
    growth_col = _resolve_col(df, "eps_yoy_growth", "revenue_yoy_growth", "revenue_growth_yoy")

    specs: dict[str, dict] = {}
    if max_peg_ratio is None:
        specs["peg_ratio"] = {"direction": "max", "percentile": 75, "fallback": 1.2}
    if min_eps_growth is None and growth_col is not None:
        specs[growth_col] = {"direction": "min", "percentile": 25, "fallback": 10}
    if max_pe_ratio is None:
        specs["p_e_ratio"] = {"direction": "max", "percentile": 75, "fallback": 35}
    if min_quality_score is None:
        specs["piotroski_f_score"] = {"direction": "min", "percentile": 25, "fallback": 5}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_max_peg = max_peg_ratio if max_peg_ratio is not None else dynamic.get("peg_ratio", 1.2)
    eff_min_growth = (
        min_eps_growth
        if min_eps_growth is not None
        else dynamic.get(growth_col, 10) if growth_col else 10
    )
    eff_max_pe = max_pe_ratio if max_pe_ratio is not None else dynamic.get("p_e_ratio", 35)
    eff_min_quality = (
        min_quality_score
        if min_quality_score is not None
        else dynamic.get("piotroski_f_score", 5) * 10
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if growth_col is not None:
        mask &= df[growth_col] >= eff_min_growth

    if "peg_ratio" in df.columns:
        mask &= (df["peg_ratio"] > 0) & (df["peg_ratio"] <= eff_max_peg)

    if "p_e_ratio" in df.columns:
        mask &= (df["p_e_ratio"] > 0) & (df["p_e_ratio"] <= eff_max_pe)

    if "piotroski_f_score" in df.columns:
        mask &= df["piotroski_f_score"] >= (eff_min_quality / 10)

    result = df[mask].copy()

    if "peg_ratio" in result.columns:
        result = result.sort_values("peg_ratio")

    return result


def screen_high_yield_safe_dividends(
    df: pd.DataFrame,
    min_yield: float | None = None,
    max_payout: float | None = None,
    min_distress_score: float | None = None,
    min_fcf_coverage: float | None = None,
) -> pd.DataFrame:
    """
    Screen for high-yielding dividends that are well-covered and safe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_yield : float or None
        Minimum dividend yield (%). None -> 25th pctl.
    max_payout : float or None
        Maximum dividend payout ratio (%). None -> 75th pctl.
    min_distress_score : float or None
        Minimum financial health score. None -> 25th pctl.
    min_fcf_coverage : float or None
        Minimum FCF dividend coverage. None -> 25th pctl.

    Returns
    -------
    pd.DataFrame
        Safe high-yield stocks sorted by yield
    """
    yield_col = "dividend_yield_ltm" if "dividend_yield_ltm" in df.columns else "dividend_yield"

    specs: dict[str, dict] = {}
    if min_yield is None:
        specs[yield_col] = {"direction": "min", "percentile": 25, "fallback": 0.1}
    if max_payout is None:
        specs["dividend_payout_ratio"] = {"direction": "max", "percentile": 75, "fallback": 100}
    if min_distress_score is None:
        specs["combined_distress_score"] = {
            "direction": "min",
            "percentile": 25,
            "fallback": 60,
        }
    if min_fcf_coverage is None:
        specs["fcf_dividend_coverage"] = {"direction": "min", "percentile": 25, "fallback": 1.1}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_min_yield = min_yield if min_yield is not None else dynamic.get(yield_col, 0.1)
    eff_max_payout = (
        max_payout if max_payout is not None else dynamic.get("dividend_payout_ratio", 100)
    )
    eff_min_distress = (
        min_distress_score
        if min_distress_score is not None
        else dynamic.get("combined_distress_score", 60)
    )
    eff_min_fcf_cov = (
        min_fcf_coverage
        if min_fcf_coverage is not None
        else dynamic.get("fcf_dividend_coverage", 1.1)
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if yield_col in df.columns:
        mask &= df[yield_col] >= eff_min_yield

    if "dividend_payout_ratio" in df.columns:
        mask &= df["dividend_payout_ratio"] <= eff_max_payout

    if "combined_distress_score" in df.columns:
        mask &= df["combined_distress_score"] >= eff_min_distress

    if "fcf_dividend_coverage" in df.columns:
        mask &= df["fcf_dividend_coverage"] >= eff_min_fcf_cov

    result = df[mask].copy()

    if yield_col in result.columns:
        result = result.sort_values(yield_col, ascending=False)

    return result


def screen_low_volatility_quality(
    df: pd.DataFrame,
    max_volatility_1y: float | None = None,
    min_quality_score: float | None = None,
    min_beta_stability: float | None = None,
    max_beta_1y: float | None = None,
    require_low_vol_regime: bool = False,
) -> pd.DataFrame:
    """
    Screen for low-volatility, high-quality stocks.

    Leverages volatility surface features (Enhancement 2) and beta risk
    features (Enhancement 3) for risk-adjusted stock selection.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with volatility and quality features
    max_volatility_1y : float or None
        Maximum 1-year volatility. None -> 75th pctl.
    min_quality_score : float or None
        Minimum Piotroski F-Score. None -> 25th pctl.
    min_beta_stability : float or None
        Minimum beta stability score. None -> 25th pctl.
    max_beta_1y : float or None
        Maximum 1-year beta. None -> 75th pctl.
    require_low_vol_regime : bool, default False
        Only include stocks in low-volatility regime

    Returns
    -------
    pd.DataFrame
        Low-volatility quality stocks sorted by volatility
    """
    vol_col = _resolve_col(df, "volatility_1y", "volatility_regime")

    specs: dict[str, dict] = {}
    if max_volatility_1y is None and vol_col is not None:
        specs[vol_col] = {"direction": "max", "percentile": 75, "fallback": 30}
    if min_quality_score is None:
        specs["piotroski_f_score"] = {"direction": "min", "percentile": 25, "fallback": 5}
    if min_beta_stability is None and "beta_stability_score" in df.columns:
        specs["beta_stability_score"] = {"direction": "min", "percentile": 25, "fallback": 50}
    if max_beta_1y is None and "beta_1y" in df.columns:
        specs["beta_1y"] = {"direction": "max", "percentile": 75, "fallback": 1.2}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_max_vol = (
        max_volatility_1y
        if max_volatility_1y is not None
        else dynamic.get(vol_col, 30) if vol_col else 30
    )
    eff_min_quality = (
        min_quality_score if min_quality_score is not None else dynamic.get("piotroski_f_score", 5)
    )
    eff_min_beta_stab = (
        min_beta_stability
        if min_beta_stability is not None
        else dynamic.get("beta_stability_score", 50)
    )
    eff_max_beta = max_beta_1y if max_beta_1y is not None else dynamic.get("beta_1y", 1.2)

    mask = pd.Series([True] * len(df), index=df.index)

    if vol_col is not None and vol_col in df.columns:
        mask &= df[vol_col] <= eff_max_vol

    if "piotroski_f_score" in df.columns:
        mask &= df["piotroski_f_score"] >= eff_min_quality

    if "beta_stability_score" in df.columns:
        mask &= df["beta_stability_score"] >= eff_min_beta_stab

    if "beta_1y" in df.columns:
        mask &= df["beta_1y"] <= eff_max_beta

    if require_low_vol_regime and "volatility_compression" in df.columns:
        mask &= df["volatility_compression"] > 0

    result = df[mask].copy()

    if vol_col is not None and vol_col in result.columns:
        result = result.sort_values(vol_col)

    return result


def screen_fcf_growth_compounders(
    df: pd.DataFrame,
    min_fcf_est_cagr: float | None = None,
    min_fcf_positive_years: int | None = None,
    max_effective_tax_rate: float | None = None,
    require_net_buyback: bool = False,
    min_operating_leverage: float | None = None,
) -> pd.DataFrame:
    """
    Screen for FCF growth compounders with favorable tax and capital allocation.

    Leverages FCF estimate curve (Enhancement 9), tax rate features
    (Enhancement 4), share dilution tracking (Enhancement 12), and
    OpEx temporal features (Enhancement 5).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with FCF, tax, and dilution features
    min_fcf_est_cagr : float or None
        Minimum estimated FCF CAGR (%). None -> 25th pctl.
    min_fcf_positive_years : int or None
        Minimum FCF positive years. None -> 25th pctl.
    max_effective_tax_rate : float or None
        Maximum effective tax rate. None -> 75th pctl.
    require_net_buyback : bool, default False
        Only include stocks with net share buybacks
    min_operating_leverage : float or None
        Minimum operating leverage score. None -> 25th pctl.

    Returns
    -------
    pd.DataFrame
        FCF compounders sorted by estimated FCF CAGR
    """
    specs: dict[str, dict] = {}
    if min_fcf_est_cagr is None and "fcf_est_cagr_5y" in df.columns:
        specs["fcf_est_cagr_5y"] = {"direction": "min", "percentile": 25, "fallback": 5}
    if min_fcf_positive_years is None and "fcf_positive_years" in df.columns:
        specs["fcf_positive_years"] = {"direction": "min", "percentile": 25, "fallback": 3}
    if max_effective_tax_rate is None and "effective_tax_rate_ltm" in df.columns:
        specs["effective_tax_rate_ltm"] = {"direction": "max", "percentile": 75, "fallback": 0.35}
    if min_operating_leverage is None and "operating_leverage_score" in df.columns:
        specs["operating_leverage_score"] = {"direction": "min", "percentile": 25, "fallback": 0}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_min_cagr = (
        min_fcf_est_cagr if min_fcf_est_cagr is not None else dynamic.get("fcf_est_cagr_5y", 5)
    )
    eff_min_fcf_yrs = (
        min_fcf_positive_years
        if min_fcf_positive_years is not None
        else int(dynamic.get("fcf_positive_years", 3))
    )
    eff_max_tax = (
        max_effective_tax_rate
        if max_effective_tax_rate is not None
        else dynamic.get("effective_tax_rate_ltm", 0.35)
    )
    eff_min_op_lev = (
        min_operating_leverage
        if min_operating_leverage is not None
        else dynamic.get("operating_leverage_score", 0)
    )

    mask = pd.Series([True] * len(df), index=df.index)

    if "fcf_est_cagr_5y" in df.columns:
        mask &= df["fcf_est_cagr_5y"] >= eff_min_cagr

    if "fcf_positive_years" in df.columns:
        mask &= df["fcf_positive_years"] >= eff_min_fcf_yrs

    if "effective_tax_rate_ltm" in df.columns:
        mask &= df["effective_tax_rate_ltm"] <= eff_max_tax

    if require_net_buyback and "net_buyback_flag" in df.columns:
        mask &= df["net_buyback_flag"] == 1

    if "operating_leverage_score" in df.columns:
        mask &= df["operating_leverage_score"] >= eff_min_op_lev

    result = df[mask].copy()

    if "fcf_est_cagr_5y" in result.columns:
        result = result.sort_values("fcf_est_cagr_5y", ascending=False)

    return result


def screen_total_return_leaders(
    df: pd.DataFrame,
    min_total_return_ytd: float | None = None,
    min_total_return_5y: float | None = None,
    min_analyst_rating: float | None = None,
    max_volatility: float | None = None,
) -> pd.DataFrame:
    """
    Screen for total return leaders using direct reference columns.

    Leverages Enhancement 1 direct reference columns (total returns,
    analyst rating) combined with volatility surface features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with total return and analyst features
    min_total_return_ytd : float or None
        Minimum YTD total return (%). None -> 25th pctl.
    min_total_return_5y : float or None
        Minimum 5Y total return (%). None -> 25th pctl.
    min_analyst_rating : float or None
        Minimum analyst rating (1-5 scale). None -> no filter.
    max_volatility : float or None
        Maximum 1Y volatility. None -> 75th pctl.

    Returns
    -------
    pd.DataFrame
        Total return leaders sorted by YTD return
    """
    specs: dict[str, dict] = {}
    if min_total_return_ytd is None and "total_return_ytd" in df.columns:
        specs["total_return_ytd"] = {"direction": "min", "percentile": 25, "fallback": 0}
    if min_total_return_5y is None and "total_return_5y" in df.columns:
        specs["total_return_5y"] = {"direction": "min", "percentile": 25, "fallback": 0}
    if max_volatility is None and "volatility_1y" in df.columns:
        specs["volatility_1y"] = {"direction": "max", "percentile": 75, "fallback": 40}

    dynamic = _compute_dynamic_thresholds(df, specs) if specs else {}

    eff_min_ytd = (
        min_total_return_ytd
        if min_total_return_ytd is not None
        else dynamic.get("total_return_ytd", 0)
    )
    eff_min_5y = (
        min_total_return_5y
        if min_total_return_5y is not None
        else dynamic.get("total_return_5y", 0)
    )
    eff_max_vol = max_volatility if max_volatility is not None else dynamic.get("volatility_1y", 40)

    mask = pd.Series([True] * len(df), index=df.index)

    if "total_return_ytd" in df.columns:
        mask &= df["total_return_ytd"] >= eff_min_ytd

    if "total_return_5y" in df.columns:
        mask &= df["total_return_5y"] >= eff_min_5y

    if min_analyst_rating is not None and "analyst_rating" in df.columns:
        mask &= df["analyst_rating"] <= min_analyst_rating  # lower = more bullish

    if "volatility_1y" in df.columns:
        mask &= df["volatility_1y"] <= eff_max_vol

    result = df[mask].copy()

    if "total_return_ytd" in result.columns:
        result = result.sort_values("total_return_ytd", ascending=False)

    return result
