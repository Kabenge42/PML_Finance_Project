"""
Performance-optimized operations for market analytics.

This module provides:
- Numba-accelerated Monte Carlo simulations
- Vectorized ruin probability calculations
- Caching utilities for expensive computations
- DataFrame hashing for cache keys

Dependencies:
- numba: JIT compilation for numerical operations
- joblib: Caching and parallel processing
"""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Try to import numba, fall back to pure Python if not available
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = False
except ImportError:
    NUMBA_AVAILABLE = False

    # Create dummy decorators
    def jit(*args, **kwargs):
        """
        Dummy JIT decorator fallback when numba is not available.

        This function mimics the numba.jit decorator interface but returns
        the original function unchanged. It allows code using @jit decorators
        to run without numba installed, albeit without JIT compilation benefits.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the decorator (ignored).
        **kwargs : dict
            Keyword arguments passed to the decorator (ignored).

        Returns
        -------
        Callable
            A decorator that returns the original function unchanged.

        Examples
        --------
        >>> @jit(nopython=True)
        ... def my_func(x):
        ...     return x * 2
        >>> my_func(5)  # Runs without JIT compilation
        10
        """

        def decorator(func):
            return func

        return decorator

    prange = range


# =============================================================================
# Caching Utilities
# =============================================================================


def dataframe_hash(df: pd.DataFrame) -> str:
    """
    Generate a hash for DataFrame caching.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to hash

    Returns
    -------
    str
        MD5 hash of the DataFrame

    Examples
    --------
    >>> hash_key = dataframe_hash(df)
    >>> print(f"Cache key: {hash_key[:16]}...")
    """
    try:
        # Use pandas hash function for efficiency
        hash_values = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.md5(hash_values.tobytes()).hexdigest()
    except Exception:
        # Fallback to string representation
        return hashlib.md5(str(df.values.tobytes()).encode()).hexdigest()


# Global cache for database queries
_db_cache: Dict[tuple, pd.DataFrame] = {}


def load_feature_data_from_db_cached(
    earnings_date_filter: str = "2026-01-01",
    limit: Optional[int] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Cached version of database loader.

    Cache key: (earnings_date_filter, limit)

    Parameters
    ----------
    earnings_date_filter : str, default "2026-01-01"
        Filter for earnings date
    limit : int, optional
        Maximum rows to return
    use_cache : bool, default True
        Whether to use cached results

    Returns
    -------
    pd.DataFrame
        Feature data from database

    Examples
    --------
    >>> df = load_feature_data_from_db_cached(limit=1000)
    >>> df2 = load_feature_data_from_db_cached(limit=1000)  # Returns cached
    """
    from analytics.data_utils import load_feature_data_from_db

    cache_key = (earnings_date_filter, limit)

    if use_cache and cache_key in _db_cache:
        return _db_cache[cache_key].copy()

    df = load_feature_data_from_db(earnings_date_filter=earnings_date_filter, limit=limit)

    _db_cache[cache_key] = df.copy()
    return df


def clear_db_cache():
    """Clear the database query cache."""
    global _db_cache
    _db_cache = {}


# Cache for expensive statistical computations
_stats_cache: Dict[str, Any] = {}


def get_cached_stats(cache_key: str) -> Optional[Any]:
    """Get cached statistical result."""
    return _stats_cache.get(cache_key)


def set_cached_stats(cache_key: str, value: Any):
    """Set cached statistical result."""
    _stats_cache[cache_key] = value


def clear_stats_cache():
    """Clear the statistics cache."""
    global _stats_cache
    _stats_cache = {}


# =============================================================================
# Numba-Accelerated Monte Carlo Simulation
# =============================================================================


if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True)
    def _fast_monte_carlo_core(
        pt_low: np.ndarray,
        pt_median: np.ndarray,
        pt_high: np.ndarray,
        last_price: np.ndarray,
        n_simulations: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Numba-accelerated core Monte Carlo simulation.

        Uses triangular distribution for price target sampling.
        """
        n_stocks = len(pt_low)
        expected_upside = np.zeros(n_stocks)
        upside_std = np.zeros(n_stocks)
        var_5 = np.zeros(n_stocks)
        prob_positive = np.zeros(n_stocks)

        for i in prange(n_stocks):
            if pt_high[i] <= pt_low[i] or last_price[i] <= 0:
                expected_upside[i] = 0.0
                upside_std[i] = 0.0
                var_5[i] = 0.0
                prob_positive[i] = 0.0
                continue

            # Triangular distribution sampling
            simulated = np.zeros(n_simulations)
            fc = (pt_median[i] - pt_low[i]) / (pt_high[i] - pt_low[i])

            for j in range(n_simulations):
                u = np.random.random()
                if u < fc:
                    simulated[j] = pt_low[i] + np.sqrt(
                        u * (pt_high[i] - pt_low[i]) * (pt_median[i] - pt_low[i]),
                    )
                else:
                    simulated[j] = pt_high[i] - np.sqrt(
                        (1 - u) * (pt_high[i] - pt_low[i]) * (pt_high[i] - pt_median[i]),
                    )

            # Calculate upside percentages
            upside = (simulated - last_price[i]) / last_price[i] * 100

            expected_upside[i] = np.mean(upside)
            upside_std[i] = np.std(upside)

            # VaR at 5%
            sorted_upside = np.sort(upside)
            var_5[i] = sorted_upside[int(n_simulations * 0.05)]

            # Probability of positive return
            positive_count = 0
            for j in range(n_simulations):
                if upside[j] > 0:
                    positive_count += 1
            prob_positive[i] = positive_count / n_simulations * 100

        return expected_upside, upside_std, var_5, prob_positive

else:

    def _fast_monte_carlo_core(
        pt_low: np.ndarray,
        pt_median: np.ndarray,
        pt_high: np.ndarray,
        last_price: np.ndarray,
        n_simulations: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pure Python fallback for Monte Carlo simulation."""
        n_stocks = len(pt_low)
        expected_upside = np.zeros(n_stocks)
        upside_std = np.zeros(n_stocks)
        var_5 = np.zeros(n_stocks)
        prob_positive = np.zeros(n_stocks)

        for i in range(n_stocks):
            if pt_high[i] <= pt_low[i] or last_price[i] <= 0:
                continue

            # Use scipy triangular distribution
            from scipy import stats

            scale = pt_high[i] - pt_low[i]
            c = (pt_median[i] - pt_low[i]) / scale if scale > 0 else 0.5

            simulated = stats.triang.rvs(c, loc=pt_low[i], scale=scale, size=n_simulations)
            upside = (simulated - last_price[i]) / last_price[i] * 100

            expected_upside[i] = np.mean(upside)
            upside_std[i] = np.std(upside)
            var_5[i] = np.percentile(upside, 5)
            prob_positive[i] = (upside > 0).sum() / n_simulations * 100

        return expected_upside, upside_std, var_5, prob_positive


def fast_monte_carlo_simulation(
    df: pd.DataFrame,
    n_simulations: int = 10000,
    pt_low_col: str = "price_target_low",
    pt_median_col: str = "price_target_median",
    pt_high_col: str = "price_target_high",
    price_col: str = "last_price",
) -> pd.DataFrame:
    """
    Fast Monte Carlo price target simulation using Numba acceleration.

    Replaces the loop in monte_carlo_price_target_simulation()
    with vectorized parallel operations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with price target columns
    n_simulations : int, default 10000
        Number of Monte Carlo simulations
    pt_low_col : str, default 'price_target_low'
        Column for low price target
    pt_median_col : str, default 'price_target_median'
        Column for median price target
    pt_high_col : str, default 'price_target_high'
        Column for high price target
    price_col : str, default 'last_price'
        Column for current price

    Returns
    -------
    pd.DataFrame
        DataFrame with simulation results:
        - ticker: Stock identifier
        - expected_upside: Mean simulated upside (%)
        - upside_std: Standard deviation of upside
        - var_5_pct: 5% Value at Risk
        - prob_positive: Probability of positive return (%)
        - risk_reward_ratio: Expected upside / VaR

    Examples
    --------
    >>> mc_results = fast_monte_carlo_simulation(df, n_simulations=10000)
    >>> top_opportunities = mc_results.nlargest(20, 'risk_reward_ratio')
    """
    required_cols = [pt_low_col, pt_median_col, pt_high_col, price_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return pd.DataFrame(
            columns=[
                "ticker",
                "expected_upside",
                "upside_std",
                "var_5_pct",
                "prob_positive",
                "risk_reward_ratio",
            ],
        )

    # Prepare arrays
    pt_low = df[pt_low_col].fillna(0).values.astype(np.float64)
    pt_median = df[pt_median_col].fillna(0).values.astype(np.float64)
    pt_high = df[pt_high_col].fillna(0).values.astype(np.float64)
    last_price = df[price_col].fillna(0).values.astype(np.float64)

    # Ensure pt_high > pt_low
    pt_high = np.maximum(pt_high, pt_low + 0.01)
    pt_median = np.clip(pt_median, pt_low, pt_high)

    # Run simulation
    expected_upside, upside_std, var_5, prob_positive = _fast_monte_carlo_core(
        pt_low,
        pt_median,
        pt_high,
        last_price,
        n_simulations,
    )

    # Calculate risk-reward ratio
    risk_reward = np.where(
        var_5 < 0,
        expected_upside / np.abs(var_5 + 1e-6),
        expected_upside / (upside_std + 1e-6),
    )

    # Build result DataFrame with identifier columns
    result_data: dict = {
        "ticker": df["ticker"].values if "ticker" in df.columns else range(len(df)),
    }
    for id_col in ("name", "sector", "industry", "country", "exchange"):
        if id_col in df.columns:
            result_data[id_col] = df[id_col].values

    result_data.update(
        {
            "expected_upside": expected_upside,
            "upside_std": upside_std,
            "var_5_pct": var_5,
            "prob_positive": prob_positive,
            "risk_reward_ratio": risk_reward,
        },
    )
    result = pd.DataFrame(result_data)

    # Add original price data
    result["last_price"] = last_price
    result["pt_median"] = pt_median

    return result.sort_values("risk_reward_ratio", ascending=False)


# =============================================================================
# Numba-Accelerated Ruin Probability
# =============================================================================


if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True)
    def _fast_ruin_probability_core(
        capital: np.ndarray,
        cash_burn: np.ndarray,
        volatility: np.ndarray,
        n_simulations: int,
        n_days: int,
    ) -> np.ndarray:
        """
        Numba-accelerated ruin probability calculation.

        Simulates random walks to estimate probability of ruin.
        """
        n_stocks = len(capital)
        ruin_probs = np.zeros(n_stocks)

        for i in prange(n_stocks):
            if capital[i] <= 0 or cash_burn[i] <= 0:
                ruin_probs[i] = 1.0
                continue

            daily_burn = cash_burn[i] / 252
            daily_vol = volatility[i] / np.sqrt(252)

            # Simulate random walks
            ruin_count = 0
            for _ in range(n_simulations):
                balance = capital[i]
                for _ in range(n_days):
                    # Daily cash burn
                    balance -= daily_burn
                    # Random return
                    balance *= 1 + np.random.normal(0, daily_vol)
                    if balance <= 0:
                        ruin_count += 1
                        break

            ruin_probs[i] = ruin_count / n_simulations

        return ruin_probs

else:

    def _fast_ruin_probability_core(
        capital: np.ndarray,
        cash_burn: np.ndarray,
        volatility: np.ndarray,
        n_simulations: int,
        n_days: int,
    ) -> np.ndarray:
        """Pure Python fallback for ruin probability."""
        n_stocks = len(capital)
        ruin_probs = np.zeros(n_stocks)

        for i in range(n_stocks):
            if capital[i] <= 0 or cash_burn[i] <= 0:
                ruin_probs[i] = 1.0
                continue

            daily_burn = cash_burn[i] / 252
            daily_vol = volatility[i] / np.sqrt(252)

            ruin_count = 0
            for _ in range(n_simulations):
                balance = capital[i]
                for _ in range(n_days):
                    balance -= daily_burn
                    balance *= 1 + np.random.normal(0, daily_vol)
                    if balance <= 0:
                        ruin_count += 1
                        break

            ruin_probs[i] = ruin_count / n_simulations

        return ruin_probs


def fast_ruin_probability(
    df: pd.DataFrame,
    capital_col: str = "market_cap",
    cash_burn_col: str = "cash_burn_rate",
    volatility_col: str = "volatility",
    n_simulations: int = 1000,
    n_days: int = 252,
) -> pd.DataFrame:
    """
    Fast ruin probability calculation using Numba acceleration.

    Vectorizes the Gambler's Ruin computation with Monte Carlo simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with financial metrics
    capital_col : str, default 'market_cap'
        Column for initial capital
    cash_burn_col : str, default 'cash_burn_rate'
        Column for annual cash burn rate
    volatility_col : str, default 'volatility'
        Column for annualized volatility
    n_simulations : int, default 1000
        Number of Monte Carlo paths per stock
    n_days : int, default 252
        Number of trading days to simulate (1 year)

    Returns
    -------
    pd.DataFrame
        DataFrame with ruin probabilities and risk tiers

    Examples
    --------
    >>> ruin_df = fast_ruin_probability(df, n_simulations=5000)
    >>> high_risk = ruin_df[ruin_df['ruin_probability'] > 0.5]
    """
    # Prepare arrays with fallback values
    if capital_col in df.columns:
        capital = df[capital_col].fillna(1e9).values.astype(np.float64)
    else:
        capital = np.full(len(df), 1e9, dtype=np.float64)

    if cash_burn_col in df.columns:
        cash_burn = df[cash_burn_col].fillna(0).abs().values.astype(np.float64)
    else:
        # Estimate from FCF if available
        if "fcf_margin" in df.columns and "market_cap" in df.columns:
            fcf = df["fcf_margin"].fillna(0) / 100 * df["market_cap"].fillna(1e9)
            cash_burn = np.where(fcf < 0, np.abs(fcf), 0).astype(np.float64)
        else:
            cash_burn = np.zeros(len(df), dtype=np.float64)

    if volatility_col in df.columns:
        volatility = df[volatility_col].fillna(0.25).abs().values.astype(np.float64)
    elif "beta_momentum" in df.columns:
        volatility = (
            (df["beta_momentum"].fillna(1).abs() * 0.2).clip(0.1, 0.8).values.astype(np.float64)
        )
    else:
        volatility = np.full(len(df), 0.25, dtype=np.float64)

    # Ensure reasonable bounds
    capital = np.clip(capital, 1e6, 1e15)
    cash_burn = np.clip(cash_burn, 0, capital)
    volatility = np.clip(volatility, 0.05, 1.0)

    # Run simulation
    ruin_probs = _fast_ruin_probability_core(capital, cash_burn, volatility, n_simulations, n_days)

    # Build result DataFrame
    result = pd.DataFrame(
        {
            "ticker": df["ticker"].values if "ticker" in df.columns else range(len(df)),
            "ruin_probability": ruin_probs,
            "survival_probability": 1 - ruin_probs,
            "capital": capital,
            "cash_burn": cash_burn,
            "volatility": volatility,
        },
    )

    # Add risk tier
    result["risk_level"] = pd.cut(
        result["ruin_probability"],
        bins=[0, 0.1, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
    )

    return result.sort_values("ruin_probability", ascending=False)


# =============================================================================
# Vectorized Statistical Operations
# =============================================================================


def vectorized_zscore(
    df: pd.DataFrame,
    columns: list,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute z-scores for multiple columns, optionally within groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        Columns to compute z-scores for
    group_col : str, optional
        Column to group by for within-group z-scores

    Returns
    -------
    pd.DataFrame
        DataFrame with z-score columns (suffixed with '_zscore')

    Examples
    --------
    >>> df_z = vectorized_zscore(df, ['roe', 'p_e_ratio'], group_col='industry')
    """
    result = df.copy()
    available_cols = [col for col in columns if col in df.columns]

    for col in available_cols:
        if group_col and group_col in df.columns:
            # Within-group z-score
            grouped = df.groupby(group_col)[col]
            result[f"{col}_zscore"] = grouped.transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10),
            )
        else:
            # Global z-score
            mean = df[col].mean()
            std = df[col].std()
            result[f"{col}_zscore"] = (df[col] - mean) / (std + 1e-10)

    return result


def vectorized_percentile_rank(
    df: pd.DataFrame,
    columns: list,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute percentile ranks for multiple columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        Columns to compute percentile ranks for
    group_col : str, optional
        Column to group by for within-group ranks

    Returns
    -------
    pd.DataFrame
        DataFrame with percentile rank columns (suffixed with '_pctile')

    Examples
    --------
    >>> df_pct = vectorized_percentile_rank(df, ['roe', 'roa'], group_col='sector')
    """
    result = df.copy()
    available_cols = [col for col in columns if col in df.columns]

    for col in available_cols:
        if group_col and group_col in df.columns:
            # Within-group percentile
            result[f"{col}_pctile"] = df.groupby(group_col)[col].rank(pct=True) * 100
        else:
            # Global percentile
            result[f"{col}_pctile"] = df[col].rank(pct=True) * 100

    return result


# =============================================================================
# Module Info
# =============================================================================


def get_optimization_status() -> dict:
    """
    Get status of optimization features.

    Returns
    -------
    dict
        Dictionary with optimization feature availability
    """
    try:
        from analytics.inference_schema import ARVIZ_AVAILABLE as _arviz
    except ImportError:
        _arviz = False

    return {
        "numba_available": NUMBA_AVAILABLE,
        "arviz_available": _arviz,
        "db_cache_size": len(_db_cache),
        "stats_cache_size": len(_stats_cache),
        "parallel_available": True,  # joblib is always available
    }
