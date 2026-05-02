"""
Statistical analysis functions for feature analytics.

This module provides advanced statistical analysis including:
- Bayesian parameter estimation
- MCMC sampling (Metropolis-Hastings)
- Monte Carlo simulations
- Distribution fitting
- Conditional probability analysis
- Investor's ruin probability models
- Bayesian resampled technical return analysis (ArviZ-enhanced)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ---------------------------------------------------------------------------
# Lazy ArviZ import (matches inference_schema.py pattern)
# ---------------------------------------------------------------------------
try:
    import arviz as az
    import xarray as xr

    ARVIZ_AVAILABLE = hasattr(az, "InferenceData")
except (ImportError, OSError, PermissionError):
    az = None  # type: ignore[assignment]
    xr = None  # type: ignore[assignment]
    ARVIZ_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED UTILITIES
# =============================================================================


def _normal_normal_conjugate_posterior(
    sample_mean: float,
    sample_var: float,
    n: int,
    prior_mean: float,
    prior_var: float,
) -> tuple[float, float]:
    """
    Compute Normal-Normal conjugate posterior parameters.

    Prior:      μ ~ N(prior_mean, prior_var)
    Likelihood: X | μ ~ N(μ, sample_var / n)

    Returns
    -------
    tuple[float, float]
        (posterior_mean, posterior_var)
    """
    if sample_var == 0 and prior_var == 0:
        return prior_mean, 0.0
    if prior_var == 0:
        return prior_mean, 0.0
    if sample_var == 0:
        return sample_mean, 0.0
    posterior_var = 1.0 / (1.0 / prior_var + n / sample_var)
    posterior_mean = posterior_var * (prior_mean / prior_var + n * sample_mean / sample_var)
    return posterior_mean, posterior_var


# =============================================================================
# Dynamic threshold computation from statistical distributions
# =============================================================================


def _compute_dynamic_thresholds(
    df: pd.DataFrame,
    feature_threshold_specs: dict[str, dict],
) -> dict[str, float]:
    """
    Compute thresholds dynamically from data distributions.

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
    features = [f for f in feature_threshold_specs if f in df.columns]
    thresholds: dict[str, float] = {}

    if not features:
        return {f: spec["fallback"] for f, spec in feature_threshold_specs.items()}

    analytics = run_category_probability_analytics(
        df,
        category_name="dynamic_threshold_estimation",
        features=features,
    )

    bayesian = analytics.get("bayesian_results", {})
    dist_fits = analytics.get("distribution_fits", {})
    summary = analytics.get("summary_statistics", {})

    for feat, spec in feature_threshold_specs.items():
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
                dist_map = {
                    "normal": stats.norm,
                    "student_t": stats.t,
                    "skew_normal": stats.skewnorm,
                }
                dist_obj = dist_map.get(dist_name)
                if dist_obj is not None:
                    try:
                        thresholds[feat] = float(dist_obj.ppf(target_pct / 100.0, *params))
                        continue
                    except Exception:
                        pass

        # Strategy 2: Use Bayesian posterior credible interval
        if feat in bayesian:
            post = bayesian[feat]
            post_mean = post.get("posterior_mean", fallback)
            post_std = post.get("posterior_std", 0)
            if post_std > 0:
                thresholds[feat] = float(stats.norm.ppf(target_pct / 100.0, post_mean, post_std))
                continue

        # Strategy 3: Empirical percentile fallback
        thresholds[feat] = float(data.quantile(target_pct / 100.0))

    return thresholds


# =============================================================================
# RESAMPLED BAYESIAN TECHNICAL ANALYSIS
# =============================================================================


@dataclass
class ResampledReturnDistribution:
    """Result container for resampled return posterior analysis."""

    isin: str
    frequency: str  # e.g. '1W', '1ME', '1QE'
    n_periods: int
    sample_mean: float
    sample_std: float
    posterior_mean: float
    posterior_std: float
    credible_interval_90: tuple[float, float]
    credible_interval_95: tuple[float, float]
    prob_positive_return: float
    skewness: float
    kurtosis: float
    var_5: float  # Value-at-Risk 5th percentile
    cvar_5: float  # Conditional VaR (Expected Shortfall)


class BayesianTechnicalResampler:
    """
    Bayesian resampling engine for historical stock price data.

    Constructs multi-timeframe return distributions from equities price
    snapshots (Last Price, Price 1M Ago, 3M, 6M, 1Y, 3Y, 5Y) and performs
    posterior updating using Normal-Normal conjugate priors.

    Produces ArviZ InferenceData objects when arviz is available, enabling
    standardised diagnostics (R-hat, ESS, posterior predictive checks).

    Parameters
    ----------
    prior_return_mean : float
        Prior expected annual return (e.g. 0.08 for 8%).
    prior_return_std : float
        Prior uncertainty on the expected return.
    n_posterior_samples : int
        Number of posterior draws per chain.
    n_chains : int
        Number of MCMC chains for ArviZ InferenceData.
    """

    _PRICE_SNAPSHOT_MAP: dict[str, str] = {
        "5D": "price_5d_ago",
        "1W": "price_1w_ago",
        "1M": "price_1m_ago",
        "3M": "price_3m_ago",
        "6M": "price_6m_ago",
        "1Y": "price_1y_ago",
        "3Y": "price_3y_ago",
        "5Y": "price_5y_ago",
    }

    # SQL-style column name fallbacks (as returned by PostgreSQL views)
    _PRICE_SNAPSHOT_SQL_MAP: dict[str, str] = {
        "5D": "Price (5D Ago)",
        "1W": "Price (1W Ago)",
        "1M": "Price (1M Ago)",
        "3M": "Price (3M Ago)",
        "6M": "Price (6M Ago)",
        "1Y": "Price (1Y Ago)",
        "3Y": "Price (3Y Ago)",
        "5Y": "Price (5Y Ago)",
    }

    _MOMENTUM_FEATURES: list[str] = [
        "price_momentum_1m",
        "price_momentum_3m",
        "price_momentum_6m",
        "price_momentum_1y",
        "price_momentum_5d",
    ]

    _TECHNICAL_FEATURES: list[str] = [
        "ema_slope_20d",
        "ema_trend_consistency",
        "volume_momentum_score",
        "breakout_signal",
        "volatility_compression",
        "volatility_term_structure",
    ]

    def __init__(
        self,
        prior_return_mean: float = 0.05,
        prior_return_std: float = 0.20,
        n_posterior_samples: int = 4000,
        n_chains: int = 8,
        random_seed: int = 42,
    ):
        self.prior_return_mean = prior_return_mean
        self.prior_return_std = prior_return_std
        self.n_posterior_samples = n_posterior_samples
        self.n_chains = n_chains
        self.rng = np.random.default_rng(random_seed)

    def _compute_historical_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive annualised return series from equities price snapshot columns.

        Returns DataFrame with columns: ticker, period, return_pct, annualised_return.
        """
        last_price_col = "last_price"
        if last_price_col not in df.columns:
            last_price_col = "Last Price"
        if last_price_col not in df.columns:
            logger.warning("No last_price column found; returning empty DataFrame")
            return pd.DataFrame()

        records = []
        period_days = {
            "5D": 5,
            "1W": 7,
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365,
            "3Y": 1095,
            "5Y": 1825,
        }

        for period, col in self._PRICE_SNAPSHOT_MAP.items():
            if col not in df.columns:
                # Fallback to SQL-style column name
                sql_col = self._PRICE_SNAPSHOT_SQL_MAP.get(period)
                if sql_col and sql_col in df.columns:
                    col = sql_col
                else:
                    continue
            mask = df[last_price_col].notna() & df[col].notna() & (df[col] > 0)
            subset = df.loc[mask]
            if subset.empty:
                continue

            hpr = (subset[last_price_col] - subset[col]) / subset[col]
            days = period_days[period]
            ann_factor = 365.0 / days
            ann_return = (1 + hpr) ** ann_factor - 1
            # v3.5: Hard clip extreme annualised returns to prevent downstream overflow (Issue 12)
            ann_return = np.clip(ann_return, -0.9999, 1000.0)  # max 100,000% annualised

            ticker_col = "isin" if "isin" in df.columns else "ISIN"
            for idx, row_idx in enumerate(subset.index):
                records.append(
                    {
                        "isin": (
                            subset.loc[row_idx, ticker_col]
                            if ticker_col in subset.columns
                            else str(row_idx)
                        ),
                        "period": period,
                        "days": days,
                        "return_pct": float(hpr.iloc[idx]) * 100,
                        "annualised_return": float(ann_return.iloc[idx]),
                    }
                )

        return pd.DataFrame(records)

    def resample_returns(
        self,
        df: pd.DataFrame,
        freq: str = "1QE",
        group_col: str = "industry",
    ) -> pd.DataFrame:
        """
        Compute resampled Bayesian posterior return distributions.

        Parameters
        ----------
        df : pd.DataFrame
            Equities data with price snapshot columns and feature columns.
        freq : str
            Resampling frequency (e.g. '1W', '1ME', '1QE').
        group_col : str
            Column for group-level hierarchical priors (e.g. 'sector').

        Returns
        -------
        pd.DataFrame
            One row per equity with posterior return statistics.
        """
        returns_df = self._compute_historical_returns(df)
        if returns_df.empty:
            return pd.DataFrame()

        prior_var = self.prior_return_std**2
        results = []

        for ticker, group in returns_df.groupby("isin"):
            data = group["annualised_return"].dropna().values
            if len(data) < 2:
                continue

            n = len(data)
            sample_mean = data.mean()
            sample_var = data.var(ddof=1) if n > 1 else prior_var

            # Normal-Normal conjugate posterior
            posterior_mean, posterior_var = _normal_normal_conjugate_posterior(
                sample_mean,
                sample_var,
                n,
                self.prior_return_mean,
                prior_var,
            )
            posterior_std = np.sqrt(posterior_var)

            ci_90 = (
                posterior_mean - 1.645 * posterior_std,
                posterior_mean + 1.645 * posterior_std,
            )
            ci_95 = (
                posterior_mean - 1.96 * posterior_std,
                posterior_mean + 1.96 * posterior_std,
            )
            # Guard: if posterior_std is zero, prob is deterministic
            if posterior_std < 1e-15:
                prob_positive = 1.0 if posterior_mean > 0 else 0.0
            else:
                prob_positive = float(1 - stats.norm.cdf(0, posterior_mean, posterior_std))

            var_5 = float(np.percentile(data, 5))
            cvar_5 = (
                float(data[data <= np.percentile(data, 5)].mean())
                if (data <= np.percentile(data, 5)).any()
                else var_5
            )

            results.append(
                ResampledReturnDistribution(
                    isin=str(ticker),
                    frequency=freq,
                    n_periods=n,
                    sample_mean=float(sample_mean),
                    sample_std=float(np.sqrt(sample_var)),
                    posterior_mean=float(posterior_mean),
                    posterior_std=float(posterior_std),
                    credible_interval_90=ci_90,
                    credible_interval_95=ci_95,
                    prob_positive_return=prob_positive,
                    skewness=(
                        float(stats.skew(np.clip(data, -10.0, 10.0)))
                        if np.std(np.clip(data, -10.0, 10.0)) > 1e-12
                        else 0.0
                    ),
                    kurtosis=(
                        float(stats.kurtosis(np.clip(data, -10.0, 10.0)))
                        if np.std(np.clip(data, -10.0, 10.0)) > 1e-12
                        else 0.0
                    ),
                    var_5=var_5,
                    cvar_5=cvar_5,
                )
            )

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame([vars(r) for r in results])

        ticker_col = "isin" if "isin" in df.columns else "ISIN"
        available_tech = [c for c in self._TECHNICAL_FEATURES if c in df.columns]
        available_mom = [c for c in self._MOMENTUM_FEATURES if c in df.columns]

        if available_tech or available_mom:
            enrich_cols = [ticker_col] + available_tech + available_mom
            if group_col in df.columns:
                enrich_cols.append(group_col)
            enrichment = df[list(set(enrich_cols))].copy()
            if ticker_col != "isin":
                enrichment = enrichment.rename(columns={ticker_col: "isin"})
            result_df = result_df.merge(enrichment, on="isin", how="left")

        return result_df

    def build_inference_data(
            self,
            df: pd.DataFrame,
            freq: str = "1QE",
            result_df: pd.DataFrame | None = None,
    ) -> "az.InferenceData | xr.Dataset | None":
        """
        Build ArviZ InferenceData from resampled posterior return distributions.

        Parameters
        ----------
        df : pd.DataFrame
            Equities data (used only if result_df is not provided).
        freq : str
            Resampling frequency.
        result_df : pd.DataFrame, optional
            Pre-computed output from ``resample_returns()``. When provided,
            avoids recomputing the resampling step.

        Returns
        -------
        arviz.InferenceData, xr.Dataset, or None
        """
        if result_df is None or result_df.empty:
            result_df = self.resample_returns(df, freq=freq)
        if result_df.empty:
            logger.warning("No resampled returns to build InferenceData")
            return None

        tickers = result_df["isin"].values
        n_equities = len(tickers)
        post_means = result_df["posterior_mean"].values
        post_stds = np.maximum(result_df["posterior_std"].values, 1e-12)

        # Shape contract: (n_chains, n_draws, n_equities) — matches dims=("chain","draw","equity")
        posterior_samples = np.stack(
            [
                self.rng.normal(
                    post_means,
                    post_stds,
                    size=(self.n_posterior_samples, n_equities),
                )
                for _ in range(self.n_chains)
            ],
            axis=0,
        )
        assert posterior_samples.shape == (
            self.n_chains,
            self.n_posterior_samples,
            n_equities,
        ), (
            f"posterior_samples shape mismatch: got {posterior_samples.shape}, "
            f"expected {(self.n_chains, self.n_posterior_samples, n_equities)}"
        )

        obs_stds = np.maximum(result_df["sample_std"].values, 1e-12)
        pp_samples = posterior_samples + self.rng.normal(
            0, obs_stds, size=posterior_samples.shape
        )

        observed_means = result_df["sample_mean"].values
        log_lik = stats.norm.logpdf(
            observed_means[np.newaxis, np.newaxis, :],
            loc=posterior_samples,
            scale=obs_stds[np.newaxis, np.newaxis, :] + 1e-12,
        )

        # Coords scoped per group so that observed_data / constant_data don't
        # inherit the (chain, draw) axes from posterior — this was the source
        # of the "conflicting sizes for dimension 'chain': length 1 ... length 8"
        # error when az.from_dict tried to broadcast the (1,)-shaped scalars
        # against the length-8 chain coordinate.
        chain_draw_equity_coords = {
            "chain": np.arange(self.n_chains),
            "draw": np.arange(self.n_posterior_samples),
            "equity": tickers,
        }
        equity_only_coords = {"equity": tickers}

        if ARVIZ_AVAILABLE and az is not None:
            # ArviZ ≥ 1.0: each InferenceData group must be passed as its own
            # keyword argument. Bundling them under a single positional dict
            # routes every variable into the `posterior` group (which is what
            # produced the "log_likelihood variable found in posterior group"
            # warning AND the chain-size conflict above).
            return az.from_dict(
                posterior={"implied_return_pt": posterior_samples},
                posterior_predictive={"future_return": pp_samples},
                log_likelihood={"return_obs": log_lik},
                observed_data={"observed_return": observed_means},
                constant_data={
                    "prior_mean": np.array([self.prior_return_mean]),
                    "prior_std": np.array([self.prior_return_std]),
                    "frequency": np.array([freq]),
                },
                coords=chain_draw_equity_coords,
                dims={
                    "implied_return_pt": ["chain", "draw", "equity"],
                    "future_return": ["chain", "draw", "equity"],
                    "return_obs": ["chain", "draw", "equity"],
                    "observed_return": ["equity"],
                    # Scalar (length-1) constants — no chain/draw axes
                    "prior_mean": ["scalar_dim"],
                    "prior_std": ["scalar_dim"],
                    "frequency": ["scalar_dim"],
                },
                # Add the scalar coord only here; ArviZ will only attach
                # 'scalar_dim' to variables that declare it in dims.
                save_warmup=False,
            )
        elif xr is not None:
            return xr.Dataset(
                {"implied_return_pt": (["chain", "draw", "equity"], posterior_samples)},
                coords=chain_draw_equity_coords,
            )
        return None


def resampled_posterior_returns(
    df: pd.DataFrame,
    freq: str = "1QE",
    prior_return_mean: float = 0.0,
    prior_return_std: float = 0.20,
    n_posterior_samples: int = 4000,
    n_chains: int = 8,
) -> tuple[pd.DataFrame, "az.InferenceData | xr.Dataset | None"]:
    """
    Convenience function: compute resampled posterior returns + InferenceData.

    Parameters
    ----------
    df : pd.DataFrame
        Equities data with price snapshot and feature columns.
    freq : str
        Pandas resampling frequency (e.g. '1W', '1ME', '1QE').
    prior_return_mean : float
        Prior expected annual return.
    prior_return_std : float
        Prior uncertainty.
    n_posterior_samples : int
        Posterior draws per chain.
    n_chains : int
        Number of chains.

    Returns
    -------
    tuple[pd.DataFrame, InferenceData | xr.Dataset | None]
        (result_df, idata)
    """
    resampler = BayesianTechnicalResampler(prior_return_mean=prior_return_mean, prior_return_std=prior_return_std,
                                           n_posterior_samples=n_posterior_samples, n_chains=n_chains)
    result_df = resampler.resample_returns(df, freq=freq)
    idata = resampler.build_inference_data(df, freq=freq, result_df=result_df)
    return result_df, idata


def bayesian_category_analysis(
    df: pd.DataFrame,
    category_name: str,
    features: list,
    prior_mean: float = 0,
    prior_std: float = 10,
) -> dict:
    """
    Bayesian analysis of feature distributions within a category.

    Uses Normal-Normal conjugate prior for continuous features.
    Prior: μ ~ N(prior_mean, prior_std²)
    Likelihood: X | μ ~ N(μ, σ²)
    Posterior: μ | X ~ N(posterior_mean, posterior_var)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature data
    category_name : str
        Name of the feature category
    features : list
        List of feature names to analyze
    prior_mean : float, default 0
        Prior mean for the parameter
    prior_std : float, default 10
        Prior standard deviation

    Returns
    -------
    dict
        Dictionary mapping feature names to posterior statistics

    Examples
    --------
    >>> results = bayesian_category_analysis(df, 'Profitability', ['roe', 'roa'])
    >>> print(results['roe']['posterior_mean'])
    """
    results = {}

    # Iterates features; computes and stores posterior statistics
    for feature in features:
        if feature not in df.columns:
            continue

        data = df[feature].dropna()
        if len(data) < 50:
            continue

        n = len(data)
        sample_mean = data.mean()
        sample_var = data.var()

        # Skip features with near-zero variance (constant / degenerate data)
        # to avoid division-by-zero in scipy when posterior_std ≈ 0
        if not np.isfinite(sample_var).all() or (sample_var < 1e-12).any():
            continue

        # Posterior parameters (Normal-Normal conjugate)
        prior_var = prior_std**2
        posterior_mean, posterior_var = _normal_normal_conjugate_posterior(
            sample_mean,
            sample_var,
            n,
            prior_mean,
            prior_var,
        )
        posterior_std = np.sqrt(posterior_var)

        # 95% Credible Interval
        ci_low = posterior_mean - 1.96 * posterior_std
        ci_high = posterior_mean + 1.96 * posterior_std

        # Probability that true mean > 0
        # Guard: if posterior_std is zero, prob is deterministic
        if posterior_std < 1e-15:
            prob_positive = 1.0 if posterior_mean > 0 else 0.0
            samples = np.full(4000, posterior_mean)
        else:
            prob_positive = 1 - stats.norm.cdf(0, posterior_mean, posterior_std)
            samples = np.random.normal(posterior_mean, posterior_std, 4000)

        feature_result = {
            "n_obs": n,
            "sample_mean": sample_mean,
            "sample_std": np.sqrt(sample_var),
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "prob_positive": prob_positive,
        }

        if ARVIZ_AVAILABLE and az is not None:
            feature_result["inference_data"] = az.from_dict(
                posterior={"mu": samples.reshape(1, -1)},
            )

        results[feature] = feature_result

    return results


def metropolis_hastings_sampler(
    data: np.ndarray,
    n_samples: int = 5000,
    burn_in: int = 2000,
    proposal_std: float = 0.5,
    prior_mean: float = 0,
    prior_std: float = 10,
    random_seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Metropolis-Hastings MCMC sampler for estimating posterior of mean parameter.

    Assumes: X ~ N(μ, σ²) with σ known from data
             Prior: μ ~ N(prior_mean, prior_std²)

    Parameters
    ----------
    data : np.ndarray
        Observed data
    n_samples : int, default 5000
        Number of MCMC samples to generate
    burn_in : int, default 2000
        Number of initial samples to discard
    proposal_std : float, default 0.5
        Standard deviation of proposal distribution
    prior_mean : float, default 0
        Prior mean
    prior_std : float, default 10
        Prior standard deviation

    Returns
    -------
    tuple
        (samples, acceptance_rate) - MCMC samples and acceptance rate

    Examples
    --------
    >>> samples, acc_rate = metropolis_hastings_sampler(data,n_samples=5000)
    >>> print(f"Acceptance rate: {acc_rate:.2%}")
    """
    rng = np.random.default_rng(random_seed)

    data_mean = np.mean(data)
    data_std = np.std(data)
    n = len(data)

    # Guard against zero variance (constant data) to prevent divide-by-zero
    if data_std == 0:
        logger.debug(
            "metropolis_hastings_sampler: data has zero variance (all values identical). "
            "Returning constant samples at data mean=%.4f.",
            data_mean,
        )
        return np.full(n_samples, data_mean), 0.0

    # Initialize
    current = data_mean
    samples = np.zeros(n_samples)
    accepted = 0

    def log_posterior(mu):
        # Log-likelihood
        ll = -n / 2 * np.log(2 * np.pi * data_std**2) - np.sum((data - mu) ** 2) / (2 * data_std**2)
        # Log-prior
        lp = -0.5 * ((mu - prior_mean) / prior_std) ** 2
        return ll + lp

    current_log_post = log_posterior(current)

    for i in range(n_samples + burn_in):
        # Adaptive proposal tuning during burn-in (~25% acceptance target)
        if i < burn_in and i % 100 == 0 and i > 0:
            accept_rate = accepted / i
            if accept_rate < 0.2:
                proposal_std *= 0.9
            elif accept_rate > 0.3:
                proposal_std *= 1.1

        # Propose new value
        proposal = current + rng.standard_normal() * proposal_std
        proposal_log_post = log_posterior(proposal)

        # Acceptance ratio (log scale)
        log_alpha = proposal_log_post - current_log_post

        # Accept or reject
        if np.log(rng.uniform()) < log_alpha:
            current = proposal
            current_log_post = proposal_log_post
            accepted += 1

        if i >= burn_in:
            samples[i - burn_in] = current

    acceptance_rate = accepted / (n_samples + burn_in)

    return samples, acceptance_rate


def mcmc_student_t(
    data: np.ndarray, n_samples: int = 5000, burn_in: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MCMC for Student's t location parameter with heavier tails.

    Better for financial data with outliers.

    Parameters
    ----------
    data : np.ndarray
        Observed data
    n_samples : int, default 5000
        Number of MCMC samples
    burn_in : int, default 2000
        Burn-in period

    Returns
    -------
    tuple
        (samples_mu, samples_df) - Location and degrees of freedom samples

    Examples
    --------
    >>> mu_samples, df_samples = mcmc_student_t(data)
    >>> print(f"Posterior mean: {mu_samples.mean():.2f}")
    """
    from scipy.stats import t as student_t

    # Initial estimates
    current_mu = np.median(data)
    current_df = 5  # degrees of freedom
    data_scale = stats.median_abs_deviation(data)

    # Guard against zero scale (constant or near-constant data)
    if data_scale == 0:
        data_scale = np.std(data)
    if data_scale == 0:
        logger.debug(
            "mcmc_student_t: data has zero dispersion. "
            "Returning constant samples at median=%.4f.",
            current_mu,
        )
        return np.full(n_samples, current_mu), np.full(n_samples, float(current_df))

    samples_mu = np.zeros(n_samples)
    samples_df = np.zeros(n_samples)

    def log_likelihood(mu, df):
        return np.sum(student_t.logpdf(data, df, loc=mu, scale=data_scale))

    current_ll = log_likelihood(current_mu, current_df)

    for i in range(n_samples + burn_in):
        # Propose new mu and df
        prop_mu = current_mu + np.random.normal(0, 0.1)
        prop_df = max(2, current_df + np.random.normal(0, 0.5))

        prop_ll = log_likelihood(prop_mu, prop_df)

        if np.log(np.random.random()) < (prop_ll - current_ll):
            current_mu = prop_mu
            current_df = prop_df
            current_ll = prop_ll

        if i >= burn_in:
            samples_mu[i - burn_in] = current_mu
            samples_df[i - burn_in] = current_df

    return samples_mu, samples_df


def hierarchical_mcmc_by_sector(
    df: pd.DataFrame, feature: str, sector_col: str = "industry", n_samples: int = 5000
) -> dict:
    """
    Hierarchical MCMC: estimate sector-level means with pooling toward global mean.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature : str
        Feature name to analyze
    sector_col : str, default 'industry'
        Column name for sector grouping
    n_samples : int, default 8000
        Number of MCMC samples

    Returns
    -------
    dict
        Dictionary mapping sectors to posterior statistics

    Examples
    --------
    >>> results = hierarchical_mcmc_by_sector(df,'roe')
    >>> print(results['Technology']['posterior_mean'])
    """
    results = {}
    sectors = df[sector_col].dropna().unique()

    # Global parameters
    global_data = df[feature].dropna()
    global_mean = global_data.mean()

    # Computes sector‑level shrinkage toward global mean
    for sector in sectors:
        sector_data = df[df[sector_col] == sector][feature].dropna().values
        if len(sector_data) < 30:
            continue

        # Shrinkage toward global mean based on sample size
        n = len(sector_data)
        shrinkage = n / (n + 10)  # Simple shrinkage factor

        sector_mean = sector_data.mean()
        sector_std = sector_data.std()

        # Posterior with shrinkage
        posterior_mean = shrinkage * sector_mean + (1 - shrinkage) * global_mean
        posterior_std = sector_std / np.sqrt(n)

        # MCMC samples from posterior
        samples = np.random.normal(posterior_mean, posterior_std, n_samples)

        results[sector] = {
            "raw_mean": sector_mean,
            "posterior_mean": posterior_mean,
            "shrinkage": shrinkage,
            "samples": samples,
            "n_obs": n,
        }

    # Build multi-group InferenceData with sector-level coordinates
    if ARVIZ_AVAILABLE and az is not None and results:
        sector_names = list(results.keys())
        sector_samples = [results[s]["samples"] for s in sector_names]
        try:
            # Shape: (1 chain, n_samples draws, n_sectors) so ArviZ
            # correctly maps the "sector" dimension.
            stacked = np.stack(sector_samples, axis=-1)[np.newaxis, :]
            idata = az.from_dict(
                {"posterior": {"sector_mu": stacked}},
                coords={"industry": sector_names},
                dims={"sector_mu": ["industry"]},
            )
            result = {"sectors": results, "inference_data": idata}
            return result
        except (ValueError, TypeError) as e:
            logger.debug("InferenceData construction failed: %s", e)

    return results


# ── Category columns available for hierarchical grouping ──
# The canonical tuple now lives in ``probabilistic_ml_model.pymc_models._hierarchy``
# so PyMC models, the multi-level shrinkage helper below, and downstream code
# share a single source of truth (recommendation §12.4 #1).
try:
    from probabilistic_ml_model.pymc_models._hierarchy import (
        HIERARCHICAL_CATEGORY_COLS as _CANONICAL_HIERARCHICAL_CATEGORY_COLS,
        PARENT_MAP as _CANONICAL_PARENT_MAP,
    )

    _HIERARCHICAL_CATEGORY_COLS: list[str] = list(_CANONICAL_HIERARCHICAL_CATEGORY_COLS)
except ImportError:  # pragma: no cover - defensive fallback
    _HIERARCHICAL_CATEGORY_COLS: list[str] = [
        "region",
        "country",
        "trading_country",
        "exchange",
        "unit",
        "sector",
        "industry",
        "style_class",
        "size_class",
    ]
    _CANONICAL_PARENT_MAP = None


def hierarchical_mcmc_multi_level(
    df: pd.DataFrame,
    feature: str,
    group_cols: list[str] | None = None,
    n_samples: int = 5000,
    min_group_size: int = 50,
    shrinkage_strength: float = 10.0,
) -> dict:
    """
    Multi-level hierarchical MCMC with nested category pooling.

    Estimates group-level means for every available categorical column,
    each shrunk toward its parent-level mean in a hierarchy:

        global → region → country → exchange → sector → industry

    Groups with fewer than ``min_group_size`` observations are pooled
    more aggressively toward the parent mean (stronger shrinkage).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature and categorical columns.
    feature : str
        Numeric feature to estimate (e.g. 'implied_return_pt', 'roe').
    group_cols : list[str], optional
        Categorical columns to group by.  Defaults to all available
        columns from ``_HIERARCHICAL_CATEGORY_COLS``.
    n_samples : int, default 8000
        Number of posterior MCMC draws per group.
    min_group_size : int, default 50
        Minimum observations per group. Smaller groups get stronger
        shrinkage toward the parent mean.
    shrinkage_strength : float, default 10.0
        Controls the pooling intensity (higher = more shrinkage).
        Effective shrinkage = n / (n + shrinkage_strength).

    Returns
    -------
    dict
        Nested dictionary with structure::

            {
                "global": { "mean": ..., "std": ..., "n_obs": ... },
                "levels": {
                    "region":   { "United States and Canada": { ... }, ... },
                    "industry": { "Software": { ... }, ... },
                    ...
                },
                "cross_level_summary": pd.DataFrame,
                "inference_data": az.InferenceData  (if ArviZ available)
            }

    Examples
    --------
    >>> result = hierarchical_mcmc_multi_level(df,'implied_return_pt')
    >>> result['levels']['region']['United States and Canada']['posterior_mean']
    >>> result['levels']['sector']['Technology']['shrinkage']
    """
    if feature not in df.columns:
        logger.warning("Feature '%s' not in DataFrame columns", feature)
        return {}

    global_data = df[feature].dropna()
    if len(global_data) < 50:
        logger.warning("Insufficient data for hierarchical MCMC (%d obs)", len(global_data))
        return {}

    global_mean = float(global_data.mean())
    global_std = float(global_data.std())

    # Resolve which categorical columns are actually present
    if group_cols is None:
        group_cols = [c for c in _HIERARCHICAL_CATEGORY_COLS if c in df.columns]
    else:
        group_cols = [c for c in group_cols if c in df.columns]

    if not group_cols:
        logger.warning("No categorical columns found for hierarchical grouping")
        return {"global": {"mean": global_mean, "std": global_std, "n_obs": len(global_data)}}

    # Define parent-child nesting for shrinkage inheritance
    _PARENT_MAP: dict[str, str | None] = {
        "region": None,
        "country": "region",
        "unit": None,
        "trading_country": None,
        "exchange": "country",
        "sector": "exchange",
        "industry": "sector",
        "style_class": None,
        "size_class": "style_class",
    }

    levels: dict[str, dict] = {}
    all_samples_for_idata: dict[str, np.ndarray] = {}
    all_coords_for_idata: dict[str, list[str]] = {}
    cross_level_rows: list[dict] = []

    for col in group_cols:
        level_results: dict[str, dict] = {}
        groups = df[col].dropna().unique()

        # Determine parent-level means for nested shrinkage
        parent_col = _PARENT_MAP.get(col)
        parent_means: dict[str, float] = {}
        if parent_col and parent_col in levels:
            for group_val in groups:
                group_mask = df[col] == group_val
                if parent_col in df.columns:
                    parent_vals = df.loc[group_mask, parent_col].dropna().mode()
                    if len(parent_vals) > 0:
                        parent_key = str(parent_vals.iloc[0])
                        parent_info = levels[parent_col].get(parent_key)
                        if parent_info:
                            parent_means[str(group_val)] = parent_info["posterior_mean"]

        for group_val in groups:
            group_data = df[df[col] == group_val][feature].dropna().values
            n = len(group_data)
            if n < min_group_size:
                continue

            group_mean = float(group_data.mean())
            group_std = float(group_data.std()) if n > 1 else global_std

            # Adaptive shrinkage: small groups shrink more
            effective_strength = shrinkage_strength * max(1.0, 30.0 / max(n, 1))
            shrinkage = n / (n + effective_strength)

            # Determine shrinkage target (parent mean or global mean)
            target_mean = parent_means.get(str(group_val), global_mean)

            posterior_mean = shrinkage * group_mean + (1 - shrinkage) * target_mean
            posterior_std = group_std / np.sqrt(n)

            # MCMC posterior samples
            samples = np.random.normal(posterior_mean, posterior_std, n_samples)

            ci_95 = (
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            )
            prob_positive = float((samples > 0).mean())

            level_results[str(group_val)] = {
                "raw_mean": group_mean,
                "raw_std": group_std,
                "posterior_mean": posterior_mean,
                "posterior_std": posterior_std,
                "shrinkage": shrinkage,
                "shrinkage_target": target_mean,
                "ci_95": ci_95,
                "prob_positive": prob_positive,
                "samples": samples,
                "n_obs": n,
            }

            cross_level_rows.append(
                {
                    "level": col,
                    "group": str(group_val),
                    "n_obs": n,
                    "raw_mean": group_mean,
                    "posterior_mean": posterior_mean,
                    "shrinkage": shrinkage,
                    "ci_95_low": ci_95[0],
                    "ci_95_high": ci_95[1],
                    "prob_positive": prob_positive,
                }
            )

        levels[col] = level_results

        # Collect for InferenceData
        if level_results:
            names = list(level_results.keys())
            # stacked shape: (n_groups, n_samples) → (1, n_samples, n_groups)
            # so arviz sees (chain=1, draw=n_samples, *group_dim)
            stacked = np.stack([level_results[s]["samples"] for s in names])
            all_samples_for_idata[f"{col}_mu"] = stacked.T[np.newaxis, :, :]
            all_coords_for_idata[col] = names

    result: dict = {
        "global": {"mean": global_mean, "std": global_std, "n_obs": len(global_data)},
        "levels": levels,
        "cross_level_summary": pd.DataFrame(cross_level_rows),
    }

    # Build unified InferenceData across all levels
    if ARVIZ_AVAILABLE and az is not None and all_samples_for_idata:
        try:
            idata = az.from_dict(
                {"posterior": all_samples_for_idata},
                coords=all_coords_for_idata,
                dims={k: [k.replace("_mu", "")] for k in all_samples_for_idata},
            )
            result["inference_data"] = idata
        except (ValueError, TypeError) as e:
            logger.debug("InferenceData construction failed for multi-level MCMC: %s", e)

    logger.info(
        "Multi-level hierarchical MCMC: %d levels, %d total groups for '%s'",
        len(levels),
        sum(len(v) for v in levels.values()),
        feature,
    )
    return result


def fit_distributions_by_category(
    df: pd.DataFrame, category: str, features: list, n_simulations: int = 10000
) -> dict:
    """
    Fit multiple distributions and select best fit using AIC.

    Simulate future scenarios using best-fit distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    category : str
        Category name
    features : list
        List of features to fit
    n_simulations : int, default 10000
        Number of simulations

    Returns
    -------
    dict
        Dictionary with fitted distributions and simulations

    Examples
    --------
    >>> fits = fit_distributions_by_category(df, 'Profitability', ['roe', 'roa'])
    >>> print(fits['roe']['best_distribution'])
    """
    from scipy.stats import norm, skewnorm, t

    results = {}

    # Fits distributions; simulates scenarios; calculates risk metrics
    for feature in features:
        if feature not in df.columns:
            continue

        # Skip non-numeric columns (e.g. string identifiers)
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue

        try:
            data = df[feature].dropna()
            if len(data) < 100:
                continue

            # Skip features with near-zero variance (constant / degenerate data)
            data_var = float(data.var())
            if not np.isfinite(data_var) or data_var < 1e-12:
                logger.debug(
                    "Skipping distribution fit for %s: near-zero variance (%.2e)",
                    feature,
                    data_var,
                )
                continue

            # Remove extreme outliers for fitting
            q01, q99 = data.quantile([0.01, 0.99])
            data_clean = data[(data >= q01) & (data <= q99)]

            # v3.5: Hard clip to prevent overflow in scipy.stats fitters (Issue 12)
            # 1e6 is safe for 4th moments in float64 (1e24 < 1e308)
            data_clean = np.clip(data_clean, -1e6, 1e6)

            if len(data_clean) < 30:
                continue

            # Re-check variance after trimming — quantile clipping can
            # collapse near-constant features to truly constant.
            clean_var = float(data_clean.var())
            if not np.isfinite(clean_var) or clean_var < 1e-12:
                logger.debug(
                    "Skipping distribution fit for %s (post-trim): "
                    "near-zero variance (%.2e)",
                    feature,
                    clean_var,
                )
                continue

            # Fit distributions — suppress scipy RuntimeWarnings for
            # near-identical data (catastrophic cancellation, divide by zero)
            fits = {}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                # Normal
                try:
                    params_norm = norm.fit(data_clean)
                    # Reject degenerate fit (scale ≈ 0)
                    if params_norm[1] > 1e-12:
                        ll_norm = norm.logpdf(data_clean, *params_norm).sum()
                        if np.isfinite(ll_norm):
                            fits["normal"] = {
                                "params": params_norm,
                                "aic": 2 * 2 - 2 * ll_norm,
                            }
                except (ValueError, RuntimeError):
                    pass

                # Student's t
                try:
                    params_t = t.fit(data_clean)
                    # params_t = (df, loc, scale); reject if scale ≈ 0
                    if params_t[2] > 1e-12:
                        ll_t = t.logpdf(data_clean, *params_t).sum()
                        if np.isfinite(ll_t):
                            fits["student_t"] = {
                                "params": params_t,
                                "aic": 2 * 3 - 2 * ll_t,
                            }
                except (ValueError, RuntimeError):
                    pass

                # Skew Normal
                try:
                    params_skew = skewnorm.fit(data_clean)
                    # params_skew = (a, loc, scale); reject if scale ≈ 0
                    if params_skew[2] > 1e-12:
                        ll_skew = skewnorm.logpdf(data_clean, *params_skew).sum()
                        if np.isfinite(ll_skew):
                            fits["skew_normal"] = {
                                "params": params_skew,
                                "aic": 2 * 3 - 2 * ll_skew,
                            }
                except (ValueError, RuntimeError):
                    pass

            if not fits:
                continue

            # Select best fit by AIC
            best_dist = min(fits.keys(), key=lambda k: fits[k]["aic"])
            best_params = fits[best_dist]["params"]

            # Simulate from best distribution — suppress warnings from
            # edge-case parameter combinations that are numerically valid
            # but trigger divide-by-zero in scipy internals.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                if best_dist == "normal":
                    simulations = norm.rvs(*best_params, size=n_simulations)
                elif best_dist == "student_t":
                    simulations = t.rvs(*best_params, size=n_simulations)
                else:
                    simulations = skewnorm.rvs(*best_params, size=n_simulations)

                # Guard against NaN/Inf in simulated values
                simulations = simulations[np.isfinite(simulations)]
                if len(simulations) < 100:
                    continue

                # Calculate VaR and CVaR
                var_5 = np.percentile(simulations, 5)
                tail = simulations[simulations <= var_5]
                cvar_5 = tail.mean() if len(tail) > 0 else var_5

            results[feature] = {
                "best_distribution": best_dist,
                "params": best_params,
                "aic": fits[best_dist]["aic"],
                "simulated_mean": float(np.nanmean(simulations)),
                "simulated_std": float(np.nanstd(simulations)),
                "var_5_pct": var_5,
                "cvar_5_pct": cvar_5,
                "simulations": simulations,
            }

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.warning(
                "Distribution fitting failed for feature %s in %s: %s",
                feature,
                category,
                e,
            )
            continue

    return results


def calculate_ruin_probability(
    df: pd.DataFrame,
    *,
    volatility_col: str = "volatility_regime",
    # NEW: Additional leverage/liquidity inputs (v3.4)
    debt_maturity_risk_col: str = "debt_maturity_risk",
    balance_sheet_strength_col: str = "balance_sheet_strength",
    cash_ratio_col: str = "cash_ratio",
    risk_tier_bins: list[float] | None = None,
    risk_tier_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate investor's ruin probability using modified Gambler's Ruin framework.

    P(ruin) ≈ exp(-2 * μ * W / σ²) for μ > 0 (drift)
    where W = wealth buffer (years of runway), μ = expected return, σ = volatility

    Recalibrated to avoid systematic overestimation:
    - Wealth buffer uses a log-transform with a floor to prevent near-zero values
    - Drift blends FCF yield, EPS trajectory, AND distress score for robustness
    - Volatility floor raised to prevent extreme σ² domination
    - Negative-drift branch uses a bounded logistic instead of linear ramp

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with financial metrics
    volatility_col : str, default 'volatility_regime'
        Column for volatility measure
    risk_tier_bins : list[float] or None
        Bin edges for risk tier classification. None -> derived from
        the ruin probability distribution quartiles; falls back to
        ``[0, 0.15, 0.35, 0.60, 1.0]``.
    risk_tier_labels : list[str] or None
        Labels for the risk tiers. None -> ``['Low Risk', 'Moderate Risk',
        'High Risk', 'Critical Risk']``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ruin probabilities and risk tiers
    """
    base_cols = ["isin","ticker", "name", "industry", "market_cap"]
    optional_cols = ["combined_distress_score"]
    use_cols = base_cols + [c for c in optional_cols if c in df.columns]
    result = df[[c for c in use_cols if c in df.columns]].copy()
    if "combined_distress_score" not in result.columns:
        result["combined_distress_score"] = 50.0  # neutral default

    # ── 1. Expected drift: blend FCF yield, EPS trajectory, and distress score ──
    if "fcf_yield" in df.columns and "eps_trajectory_score" in df.columns:
        fcf_norm = df["fcf_yield"].clip(-20, 50) / 100
        eps_norm = df["eps_trajectory_score"] / 100

        # Add distress-score contribution: higher combined_distress_score → healthier
        distress_drift = (
            df.get("combined_distress_score", pd.Series(50, index=df.index)) / 200
        )  # 0–0.5
        result["expected_drift"] = (
            fcf_norm * 0.40 + eps_norm * 0.30 + distress_drift * 0.30
        ).fillna(0.05)
    else:
        result["expected_drift"] = 0.08  # More realistic default drift

    # ── 2. Volatility proxy: tighter floor to prevent σ² domination ──
    if volatility_col in df.columns:
        result["volatility"] = df[volatility_col].abs().clip(10, 60) / 100  # [0.10, 0.60]
    elif "beta_momentum" in df.columns:
        result["volatility"] = (df["beta_momentum"].abs() * 0.15 + 0.10).clip(0.10, 0.60)
    else:
        result["volatility"] = 0.20

    # ── 3. Wealth buffer: log-scaled with a meaningful floor ──
    if "cash_runway_months" in df.columns:
        # After SQL fix, profitable companies have runway = 120 months (10 years)
        raw_years = df["cash_runway_months"].clip(1, 120) / 12.0
        # Log-transform: compresses extreme values, raises the floor
        # log1p(1) = 0.69, log1p(10) = 2.40 → more separation in the useful range
        result["wealth_buffer"] = np.log1p(raw_years)
    else:
        result["wealth_buffer"] = np.log1p(5.0)  # Default ~5 years → 1.79

    # ── 4. Ruin probability with recalibrated formula ──
    mu = result["expected_drift"]
    sigma = result["volatility"]
    W = result["wealth_buffer"]

    sigma_sq = sigma**2 + 1e-6

    # Positive drift: classical continuous-time ruin formula
    # Negative drift: bounded logistic (avoids the linear ramp to 1.0)
    positive_drift_ruin = np.exp(-2 * mu * W / sigma_sq).clip(0, 1)
    negative_drift_ruin = (1.0 / (1.0 + np.exp(-10 * np.abs(mu)))).clip(0.5, 0.95)

    result["ruin_probability"] = np.where(
        mu > 0,
        positive_drift_ruin,
        negative_drift_ruin,
    )

    result["survival_probability"] = 1 - result["ruin_probability"]

    # ── 4b. Leverage/liquidity adjustment (v3.4) ──
    if debt_maturity_risk_col in df.columns:
        debt_risk = df[debt_maturity_risk_col].fillna(0) / 100
        result["ruin_probability"] = np.clip(result["ruin_probability"] + debt_risk * 0.15, 0, 1)
    if balance_sheet_strength_col in df.columns:
        bs = df[balance_sheet_strength_col].fillna(50) / 100
        result["ruin_probability"] = np.clip(result["ruin_probability"] * (1.3 - bs * 0.6), 0, 1)
    if cash_ratio_col in df.columns:
        cr = df[cash_ratio_col].fillna(0.5)
        # Low cash ratio increases ruin probability
        cash_penalty = np.where(cr < 0.1, 0.10, np.where(cr < 0.3, 0.05, 0.0))
        result["ruin_probability"] = np.clip(result["ruin_probability"] + cash_penalty, 0, 1)
    result["survival_probability"] = 1 - result["ruin_probability"]

    # ── 5. Risk tier classification (wider low-risk band) ──
    if risk_tier_labels is None:
        risk_tier_labels = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]

    if risk_tier_bins is None:
        # Derive bins from ruin probability distribution if sufficient data
        ruin_data = result["ruin_probability"].dropna()
        if len(ruin_data) >= 30:
            specs: dict[str, dict] = {
                "ruin_probability": {"direction": "max", "percentile": 75, "fallback": 0.60},
            }
            dynamic = _compute_dynamic_thresholds(result, specs)
            q75 = dynamic.get("ruin_probability", 0.60)
            # Build bins: [0, q25, q50, q75, 1.0] from distribution
            q25 = float(ruin_data.quantile(0.25))
            q50 = float(ruin_data.quantile(0.50))
            # Ensure monotonically increasing and within [0, 1]
            risk_tier_bins = [
                0,
                max(0.01, min(q25, 0.30)),
                max(q25 + 0.01, min(q50, 0.55)),
                max(q50 + 0.01, min(q75, 0.90)),
                1.0,
            ]
            # Deduplicate: when quantiles collapse, edges can repeat
            risk_tier_bins = sorted(set(risk_tier_bins))
            # Adjust labels to match the (possibly reduced) number of bins
            if len(risk_tier_bins) - 1 < len(risk_tier_labels):
                risk_tier_labels = risk_tier_labels[: len(risk_tier_bins) - 1]
        else:
            risk_tier_bins = [0, 0.15, 0.35, 0.60, 1.0]

    result["risk_level"] = pd.cut(
        result["ruin_probability"],
        bins=risk_tier_bins,
        labels=risk_tier_labels,
    )

    return result


def calculate_conditional_probabilities(
    df: pd.DataFrame, feature_categories: dict, distress_threshold: float = 65
) -> pd.DataFrame:
    """
    Calculate conditional probability of financial distress given feature conditions.

    P(Distress | High Feature) vs P(Distress | Low Feature)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_categories : dict
        Dictionary of feature categories
    distress_threshold : float, default 65
        Threshold for distress classification

    Returns
    -------
    pd.DataFrame
        DataFrame with conditional probabilities

    Examples
    --------
    >>> cond_probs = calculate_conditional_probabilities(df, feature_categories)
    >>> top_features = cond_probs.nlargest(50, 'separation')
    """
    results = []

    df = df.copy()
    df["is_distressed"] = df["combined_distress_score"] < distress_threshold
    base_distress_rate = df["is_distressed"].mean()

    for category, features in feature_categories.items():
        for feature in features[:50]:  # Top 50 features per category
            if feature not in df.columns:
                continue

            data = df[[feature, "is_distressed"]].dropna()
            if len(data) < 100:
                continue

            # Skip non-numeric features – median/comparison not meaningful
            if not pd.api.types.is_numeric_dtype(data[feature]):
                continue

            median_val = data[feature].median()

            # P(Distress | Feature > Median)
            high_mask = data[feature] > median_val
            p_distress_high = data.loc[high_mask, "is_distressed"].mean()

            # P(Distress | Feature <= Median)
            p_distress_low = data.loc[~high_mask, "is_distressed"].mean()

            # Lift ratio
            lift_high = p_distress_high / base_distress_rate if base_distress_rate > 0 else 1
            lift_low = p_distress_low / base_distress_rate if base_distress_rate > 0 else 1

            results.append(
                {
                    "category": category,
                    "feature": feature,
                    "p_distress_high": p_distress_high,
                    "p_distress_low": p_distress_low,
                    "lift_high": lift_high,
                    "lift_low": lift_low,
                    "separation": abs(p_distress_high - p_distress_low),
                }
            )

    if not results:
        return pd.DataFrame(
            columns=[
                "category",
                "feature",
                "p_distress_high",
                "p_distress_low",
                "lift_high",
                "lift_low",
                "separation",
            ]
        )

    return pd.DataFrame(results).sort_values("separation", ascending=False)


# =============================================================================
# Enhanced Statistical Methods
# =============================================================================


def monte_carlo_price_target_simulation(
    df: pd.DataFrame,
    n_simulations: int = 25000,
    max_stocks: int = 7000,
) -> pd.DataFrame:
    """
    Monte Carlo simulation of price targets based on analyst spread.

    Uses the analyst price target range (high/low/median) to model
    uncertainty and generate probabilistic fair value estimates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - ticker, name, industry, last_price
        - price_target, price_target_high, price_target_low, price_target_median
    n_simulations : int, default 10000
        Number of Monte Carlo simulations per stock
    max_stocks : int, default 7000
        Maximum number of stocks to simulate (for performance)
    confidence_level : float, default 0.95
        Confidence level for VaR calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with simulation results including:
        - ticker, name, industry, last_price
        - implied_return_mc: MC-simulated implied return (percentage)
        - upside_std, var_5_pct
        - prob_positive_upside, risk_reward_ratio
    """
    rng = np.random.default_rng(42)

    required_cols = ["price_target", "price_target_high", "price_target_low", "last_price"]

    # v3.10: Separate valid/invalid rows so all ISINs appear in output
    valid_mask = (
        df[required_cols].notna().all(axis=1)
        & (df["price_target_high"].fillna(0) > df["price_target_low"].fillna(0))
        & (df["last_price"].fillna(0) > 0)
    )
    valid_df = df[valid_mask].head(max_stocks).copy()
    invalid_df = df[~valid_mask].copy()

    if valid_df.empty:
        # Return stub rows for all stocks with NaN simulation columns
        id_cols = ["isin", "ticker", "name", "sector", "industry", "region", "country", "exchange"]
        stub = df[[c for c in id_cols if c in df.columns]].copy()
        stub["last_price"] = df.get("last_price", np.nan)
        return stub

    # Resolve median column
    if "price_target_median" in valid_df.columns:
        pt_median = valid_df["price_target_median"].fillna(valid_df["price_target"]).values.copy()
    else:
        pt_median = valid_df["price_target"].values.copy()

    pt_low = valid_df["price_target_low"].values.copy()
    pt_high = valid_df["price_target_high"].values.copy()
    last_price = valid_df["last_price"].values
    n_stocks = len(valid_df)

    # P1: Volatility-scaled spread adjustment (Cat 18 — Volatility Surface)
    if "volatility_1y" in valid_df.columns:
        vol_scale = valid_df["volatility_1y"].clip(10, 60).values / 30.0  # normalize around 1.0
        pt_low *= 1 - 0.1 * (vol_scale - 1)  # widen spread for high-vol
        pt_high *= 1 + 0.1 * (vol_scale - 1)
        # Ensure low < median < high after adjustment
        pt_low = np.minimum(pt_low, pt_median - 0.01)
        pt_high = np.maximum(pt_high, pt_median + 0.01)

    # P1: Momentum-adjusted drift (Cat 2 — Momentum)
    if "price_momentum_3m" in valid_df.columns:
        mom_3m = valid_df["price_momentum_3m"].fillna(0).clip(-50, 50).values / 100.0
        pt_median *= 1 + mom_3m * 0.05  # subtle drift toward momentum direction
        # Re-clamp median within bounds
        pt_median = np.clip(pt_median, pt_low + 0.01, pt_high - 0.01)

    # Vectorized triangular simulation: (n_stocks, n_simulations)
    simulated_pts = rng.triangular(
        pt_low[:, np.newaxis],
        pt_median[:, np.newaxis],
        pt_high[:, np.newaxis],
        size=(n_stocks, n_simulations),
    )

    # Vectorized upside calculation
    simulated_upside = (simulated_pts - last_price[:, np.newaxis]) / last_price[:, np.newaxis] * 100
    # v3.5: Clip extreme returns to prevent overflow in higher-moment calculations (Issue 12)
    simulated_upside = np.clip(simulated_upside, -100.0, 10000.0)

    # Vectorized statistics across simulation axis
    expected_upside_mc = simulated_upside.mean(axis=1)
    upside_std = simulated_upside.std(axis=1)
    var_5 = np.percentile(simulated_upside, 5, axis=1)
    prob_positive = (simulated_upside > 0).mean(axis=1) * 100
    risk_reward = np.divide(
        expected_upside_mc, upside_std, out=np.zeros_like(expected_upside_mc), where=upside_std > 0
    )

    # Implied return as percentage-based (mean_target / last_price - 1) * 100
    mc_fair_value = simulated_pts.mean(axis=1)
    implied_return_mc = np.where(last_price > 0, (mc_fair_value / last_price - 1) * 100, 0.0)
    implied_return_mc = np.clip(implied_return_mc, -100.0, 10000.0)

    # Build result DataFrame
    result_df = pd.DataFrame(
        {
            "isin": valid_df.get("isin", pd.Series("", index=valid_df.index)).values,
            "ticker": valid_df.get("ticker", pd.Series("", index=valid_df.index)).values,
            "name": valid_df.get("name", pd.Series("", index=valid_df.index)).values,
            "sector": valid_df.get("sector", pd.Series("", index=valid_df.index)).values,
            "industry": valid_df.get("industry", pd.Series("", index=valid_df.index)).values,
            "region": valid_df.get("region", pd.Series("", index=valid_df.index)).values,
            "country": valid_df.get("country", pd.Series("", index=valid_df.index)).values,
            "exchange": valid_df.get("exchange", pd.Series("", index=valid_df.index)).values,
            "last_price": last_price,
            "expected_upside_mc": expected_upside_mc,
            "implied_return_mc": implied_return_mc,
            "price_target_mc": mc_fair_value,
            "pt_median": pt_median,
            "pt_spread": pt_high - pt_low,
            "upside_std": upside_std,
            "var_5_pct": var_5,
            "prob_positive_upside": prob_positive,
            "risk_reward_ratio": risk_reward,
        }
    )

    # v3.10: Append stub rows for stocks that failed data-quality checks
    if not invalid_df.empty:
        id_cols = ["isin", "ticker", "name", "sector", "industry", "region", "country", "exchange"]
        stub = invalid_df[[c for c in id_cols if c in invalid_df.columns]].copy()
        stub["last_price"] = invalid_df.get("last_price", np.nan)
        # All simulation columns will be NaN for these rows
        result_df = pd.concat([result_df, stub], ignore_index=True)

    return result_df


def kalman_filter_price_target(
    df: pd.DataFrame,
    observation_col: str = "last_price",
    target_col: str = "price_target",
    process_variance: float | None = None,
    measurement_variance: float | None = None,
) -> pd.DataFrame:
    """
    Kalman filter for smoothing price targets and estimating true value.

    State-space model:
    - State: True underlying value
    - Observation: Noisy analyst price targets

    Parameters
    ----------
    df : pd.DataFrame
        Must contain observation_col and target_col
    observation_col : str, default 'last_price'
        Column with current price observations
    target_col : str, default 'price_target'
        Column with analyst price targets
    process_variance : float or None
        Q - process noise covariance (how much true value changes).
        None -> derived from the observation column distribution
        (scaled variance); falls back to 1e-5.
    measurement_variance : float or None
        R - measurement noise covariance (analyst estimate error).
        None -> derived from the target-observation residual
        distribution; falls back to 0.1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ticker: Stock identifier
        - implied_return_kalman: Kalman-filtered implied return (percentage)
        - kalman_estimate: Filtered price estimate
        - kalman_variance: Estimation uncertainty
        - kalman_gain: Filter gain at each step
        - signal_strength: Confidence in the estimate (1/variance)
    """
    # ── Dynamic variance computation ──
    if process_variance is None or measurement_variance is None:
        specs: dict[str, dict] = {}
        if process_variance is None and observation_col in df.columns:
            specs[observation_col] = {
                "direction": "max",
                "percentile": 50,
                "fallback": 1e-5,
            }
        if (
            measurement_variance is None
            and target_col in df.columns
            and observation_col in df.columns
        ):
            # Use the spread between target and observation as proxy
            residual_col = "_kalman_residual_proxy"
            df = df.copy()
            obs_valid = df[observation_col].notna() & (df[observation_col] > 0)
            tgt_valid = df[target_col].notna() & (df[target_col] > 0)
            df[residual_col] = np.where(
                obs_valid & tgt_valid,
                ((df[target_col] - df[observation_col]) / df[observation_col]).abs(),
                np.nan,
            )
            specs[residual_col] = {
                "direction": "max",
                "percentile": 50,
                "fallback": 0.1,
            }

        if specs:
            dynamic = _compute_dynamic_thresholds(df, specs)
            if process_variance is None:
                # Scale observation median to a small process variance
                obs_median = dynamic.get(observation_col, 1e-5)
                process_variance = max(1e-8, (obs_median / 1e6) ** 2) if obs_median > 0 else 1e-5
            if measurement_variance is None:
                measurement_variance = max(0.01, dynamic.get("_kalman_residual_proxy", 0.1))
                # Clean up temporary column
                if "_kalman_residual_proxy" in df.columns:
                    df = df.drop(columns=["_kalman_residual_proxy"])
        else:
            if process_variance is None:
                process_variance = 1e-5
            if measurement_variance is None:
                measurement_variance = 0.1
    if observation_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(
            columns=[
                "isin",
                "implied_return_kalman",
                "kalman_estimate",
                "kalman_variance",
                "kalman_gain",
                "signal_strength",
            ]
        )

    # Filter valid rows
    mask = (
        df[observation_col].notna()
        & df[target_col].notna()
        & (df[observation_col] > 0)
        & (df[target_col] > 0)
    )
    valid_df = df.loc[mask].copy()

    if valid_df.empty:
        return pd.DataFrame()

    obs = valid_df[observation_col].values.astype(float)
    z = valid_df[target_col].values.astype(float)

    # P1: Per-stock adaptive variance from Volatility Surface & Analyst Sentiment
    pv = np.full(len(valid_df), process_variance)
    mv = np.full(len(valid_df), measurement_variance)

    if "volatility_1y" in valid_df.columns:
        vol_1y = valid_df["volatility_1y"].clip(10, 60).values
        mv = vol_1y / 100.0 * 0.3  # higher vol → higher measurement noise

    if "price_target_count" in valid_df.columns:
        pt_count = valid_df["price_target_count"].clip(1, 30).values
        pv = 1e-5 / pt_count  # more analysts → lower process noise

    if "beta_stability_score" in valid_df.columns:
        beta_stab = valid_df["beta_stability_score"].clip(0, 100).values
        # Unstable beta (low score) → increase process variance
        pv *= 1 + (100 - beta_stab) / 200.0

    # Vectorized single-step Kalman update (cross-sectional, not time-series)
    x_pred = obs  # Initialize state with observation
    p_pred = 1.0 + pv  # Initial covariance + process noise

    kalman_gain = p_pred / (p_pred + mv)
    x_est = x_pred + kalman_gain * (z - x_pred)
    p_est = (1 - kalman_gain) * p_pred
    signal_strength = 1.0 / (p_est + 1e-10)
    expected_upside_kalman = np.where(obs > 0, (x_est - obs) / obs * 100, 0.0)

    # Implied return as percentage-based (target / observation - 1) * 100
    implied_return_kalman = np.where(obs > 0, (z / obs - 1) * 100, 0.0)

    result_df = pd.DataFrame(
        {
            "isin": valid_df.get(
                "isin", pd.Series(valid_df.index.astype(str), index=valid_df.index)
            ).values,
            "ticker": valid_df.get(
                "ticker",
                valid_df.get(
                    "isin", pd.Series(valid_df.index.astype(str), index=valid_df.index)
                ),
            ).values,
            "name": valid_df.get("name", pd.Series("", index=valid_df.index)).values,
            "sector": valid_df.get("sector", pd.Series("", index=valid_df.index)).values,
            "industry": valid_df.get("industry", pd.Series("", index=valid_df.index)).values,
            "country": valid_df.get("country", pd.Series("", index=valid_df.index)).values,
            "exchange": valid_df.get("exchange", pd.Series("", index=valid_df.index)).values,
            "implied_return_kalman": implied_return_kalman,
            "expected_upside_kalman": expected_upside_kalman,
            "price_target_kalman": x_est,
            "kalman_estimate": x_est,
            "kalman_variance": p_est,
            "kalman_gain": kalman_gain,
            "signal_strength": signal_strength,
            "original_price": obs,
            "original_target": z,
        }
    )

    return result_df


def kalman_momentum_filter(
    df: pd.DataFrame,
    momentum_cols: list = None,
    process_variance: float = 0.05,
    measurement_variance: float = 0.25,
) -> pd.DataFrame:
    """
    Apply Kalman filter to smooth noisy momentum indicators.

    Useful for reducing whipsaw signals in trend following.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with momentum columns
    momentum_cols : list, optional
        List of momentum column names. Default: ['price_momentum_1m',
        'price_momentum_3m', 'price_momentum_6m']
    process_variance : float, default 0.05
        Process noise variance
    measurement_variance : float, default 0.25
        Measurement noise variance

    Returns
    -------
    pd.DataFrame
        DataFrame with filtered momentum columns (suffixed with '_filtered')

    Examples
    --------
    >>> filtered_df = kalman_momentum_filter(df)
    >>> print(filtered_df['price_momentum_1m_filtered'].head())
    """
    if momentum_cols is None:
        momentum_cols = [
            "price_momentum_1m",
            "price_momentum_3m",
            "price_momentum_6m",
            "price_momentum_1y",
            "price_momentum_5d",
            "price_momentum_3y",
            "price_momentum_5y",
        ]

    available_cols = [col for col in momentum_cols if col in df.columns]

    if not available_cols:
        return df.copy()

    result = df.copy()

    for col in available_cols:
        data = df[col].values
        n = len(data)

        # Initialize
        x_est = np.zeros(n)
        p_est = np.ones(n)

        # First value initialization
        valid_idx = np.where(~np.isnan(data))[0]
        if len(valid_idx) == 0:
            result[f"{col}_filtered"] = np.nan
            continue

        first_valid = valid_idx[0]
        x_est[first_valid] = data[first_valid]
        p_est[first_valid] = 1.0

        # Forward pass
        for i in range(first_valid + 1, n):
            if np.isnan(data[i]):
                x_est[i] = x_est[i - 1]
                p_est[i] = p_est[i - 1] + process_variance
            else:
                # Predict
                x_pred = x_est[i - 1]
                p_pred = p_est[i - 1] + process_variance

                # Update
                k = p_pred / (p_pred + measurement_variance)
                x_est[i] = x_pred + k * (data[i] - x_pred)
                p_est[i] = (1 - k) * p_pred

        result[f"{col}_filtered"] = x_est
        result[f"{col}_variance"] = p_est

    return result


def fit_gaussian_copula(df: pd.DataFrame, features: list, n_simulations: int = 25000) -> dict:
    """
    Fit Gaussian copula to capture dependency structure between features.

    Useful for:
    - Understanding tail dependencies between risk factors
    - Generating correlated Monte Carlo samples
    - Stress testing with realistic correlation structures

    Parameters
    ----------
    df : pd.DataFrame
        Data with features to model
    features : list
        Column names to include in copula
    n_simulations : int, default 10000
        Number of samples to generate

    Returns
    -------
    dict
        Dictionary with:
        - correlation_matrix: Estimated correlation structure
        - features: List of features used
        - simulated_samples: Correlated uniform samples
        - tail_dependence: Lower/upper tail dependence coefficients
        - marginal_params: Parameters of marginal distributions

    Examples
    --------
    >>> copula = fit_gaussian_copula(df,['roe', 'debt_to_equity', 'p_e_ratio'])
    >>> print(copula['correlation_matrix'])
    """
    # Filter to available features
    available_features = [f for f in features if f in df.columns]

    if len(available_features) < 2:
        return {
            "correlation_matrix": np.array([[1.0]]),
            "features": available_features,
            "simulated_samples": np.array([]),
            "tail_dependence": {"lower": np.array([]), "upper": np.array([])},
            "marginal_params": {},
        }

    # Extract data and handle missing values
    data = df[available_features].dropna()

    if len(data) < 50:
        return {
            "correlation_matrix": np.eye(len(available_features)),
            "features": available_features,
            "simulated_samples": np.array([]),
            "tail_dependence": {"lower": np.array([]), "upper": np.array([])},
            "marginal_params": {},
        }

    n_features = len(available_features)

    # Transform to uniform marginals using empirical CDF
    uniform_data = np.zeros((len(data), n_features))
    marginal_params = {}

    for i, feat in enumerate(available_features):
        col_data = data[feat].values
        ranks = stats.rankdata(col_data)
        uniform_data[:, i] = ranks / (len(col_data) + 1)

        # Store marginal statistics
        # v3.5: Clip to prevent overflow in skew/kurtosis (Issue 12)
        col_data_safe = np.clip(col_data, -1e12, 1e12)
        marginal_params[feat] = {
            "mean": float(np.mean(col_data)),
            "std": float(np.std(col_data)),
            "median": float(np.median(col_data)),
            "skew": float(stats.skew(col_data_safe)),
            "kurtosis": float(stats.kurtosis(col_data_safe)),
        }

    # Transform to normal and estimate correlation
    normal_data = stats.norm.ppf(uniform_data)
    normal_data = np.nan_to_num(normal_data, nan=0, posinf=0, neginf=0)

    # Estimate correlation matrix
    correlation_matrix = np.corrcoef(normal_data.T)

    # Ensure positive definiteness
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Generate correlated samples
    try:
        cholesky = np.linalg.cholesky(correlation_matrix)
        z = np.random.standard_normal((n_simulations, n_features))
        correlated_normal = z @ cholesky.T
        simulated_uniform = stats.norm.cdf(correlated_normal)
    except np.linalg.LinAlgError:
        # Fallback to independent samples if Cholesky fails
        simulated_uniform = np.random.uniform(0, 1, (n_simulations, n_features))

    # Calculate tail dependence
    tail_dep = _calculate_tail_dependence(uniform_data)

    return {
        "correlation_matrix": correlation_matrix,
        "features": available_features,
        "simulated_samples": simulated_uniform,
        "tail_dependence": tail_dep,
        "marginal_params": marginal_params,
        "n_observations": len(data),
    }


def _calculate_tail_dependence(uniform_data: np.ndarray, threshold: float = 0.05) -> dict:
    """
    Calculate lower and upper tail dependence coefficients.

    Parameters
    ----------
    uniform_data : np.ndarray
        Data transformed to uniform marginals
    threshold : float, default 0.05
        Threshold for tail definition

    Returns
    -------
    dict
        Dictionary with 'lower' and 'upper' tail dependence matrices
    """
    n_vars = uniform_data.shape[1]
    lower_dep = np.zeros((n_vars, n_vars))
    upper_dep = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # Lower tail: P(V < q | U < q)
            mask_lower = uniform_data[:, i] < threshold
            if mask_lower.sum() > 0:
                lower_dep[i, j] = (uniform_data[mask_lower, j] < threshold).mean()
                lower_dep[j, i] = lower_dep[i, j]

            # Upper tail: P(V > 1-q | U > 1-q)
            mask_upper = uniform_data[:, i] > (1 - threshold)
            if mask_upper.sum() > 0:
                upper_dep[i, j] = (uniform_data[mask_upper, j] > (1 - threshold)).mean()
                upper_dep[j, i] = upper_dep[i, j]

    # Set diagonal to 1
    np.fill_diagonal(lower_dep, 1.0)
    np.fill_diagonal(upper_dep, 1.0)

    return {"lower": lower_dep, "upper": upper_dep}


def parallel_mcmc_chains(
    data: np.ndarray, n_chains: int = 8, n_samples: int = 10_000, n_jobs: int = -1
) -> dict:
    """
    Run multiple MCMC chains in parallel for better convergence diagnostics.

    Uses joblib (already in requirements.txt) for parallel execution.

    Parameters
    ----------
    data : np.ndarray
        Input data for sampling
    n_chains : int, default 4
        Number of parallel chains
    n_samples : int, default 10000
        Samples per chain
    n_jobs : int, default -1
        Number of parallel jobs (-1 = all cores)

    Returns
    -------
    dict
        Dictionary with:
        - chains: List of sample arrays
        - r_hat: Gelman-Rubin convergence diagnostic
        - combined_samples: Merged samples from all chains
        - converged: Boolean indicating if R-hat < 1.1
        - chain_means: Mean of each chain
        - chain_stds: Std of each chain

    Examples
    --------
    >>> result = parallel_mcmc_chains(data,n_chains=8)
    >>> if result['converged']:
    ...     print(f"Posterior mean: {result['combined_samples'].mean():.2f}")
    """
    try:
        from joblib import Parallel, delayed

        use_parallel = True
    except ImportError:
        Parallel = None  # type: ignore[assignment,misc]
        delayed = None  # type: ignore[assignment]
        use_parallel = False

    def run_single_chain(seed: int) -> np.ndarray:
        """Run a single MCMC chain with given seed."""
        samples, _ = metropolis_hastings_sampler(data, n_samples=n_samples, burn_in=n_samples // 5, random_seed=seed)
        return samples

    # Run chains
    if use_parallel and n_jobs != 1:
        chains = Parallel(n_jobs=n_jobs)(
            delayed(run_single_chain)(seed) for seed in range(n_chains)
        )
    else:
        # Sequential fallback
        chains = [run_single_chain(seed) for seed in range(n_chains)]

    # Combine samples
    combined_samples = np.concatenate(chains)

    # Chain statistics
    chain_means = [np.mean(c) for c in chains]
    chain_stds = [np.std(c) for c in chains]

    result = {
        "chains": chains,
        "combined_samples": combined_samples,
        "chain_means": chain_means,
        "chain_stds": chain_stds,
        "posterior_mean": np.mean(combined_samples),
        "posterior_std": np.std(combined_samples),
        "ci_95": (np.percentile(combined_samples, 2.5), np.percentile(combined_samples, 97.5)),
    }

    # Stack chains into array for ArviZ
    chain_array = np.stack(chains)

    if ARVIZ_AVAILABLE and az is not None:
        try:
            idata = az.from_dict(
                {"posterior": {"mu": chain_array.reshape(n_chains, 1, n_samples).transpose(0, 2, 1)}},
                coords={"chain": np.arange(n_chains), "draw": np.arange(n_samples)},
            )
            summary = az.summary(idata)
            result["r_hat"] = float(summary["r_hat"].iloc[0])
            result["ess_bulk"] = float(summary["ess_bulk"].iloc[0])
            result["ess_tail"] = float(summary["ess_tail"].iloc[0])
            result["inference_data"] = idata
        except (ValueError, KeyError, TypeError):
            result["r_hat"] = _calculate_gelman_rubin(chains)
    else:
        result["r_hat"] = _calculate_gelman_rubin(chains)

    result["converged"] = result["r_hat"] < 1.001
    return result


def _calculate_gelman_rubin(chains: list) -> float:
    """
    Calculate R-hat (Gelman-Rubin) convergence diagnostic.

    R-hat < 1.1 indicates convergence.  Delegates to ``az.rhat()``
    (split-R-hat, more robust) when ArviZ is available.

    Parameters
    ----------
    chains : list
        List of MCMC sample arrays

    Returns
    -------
    float
        R-hat statistic
    """
    if ARVIZ_AVAILABLE and az is not None:
        try:
            chain_array = np.stack(chains).reshape(len(chains), -1)
            idata = az.from_dict({"posterior": {"x": chain_array[:, np.newaxis, :]}})
            return float(az.rhat(idata)["x"].values)
        except Exception:
            pass  # fall through to manual implementation

    m = len(chains)  # number of chains
    n = len(chains[0])  # samples per chain

    if m < 2 or n < 10:
        return float("inf")

    chain_means = np.array([np.mean(c) for c in chains])
    overall_mean = np.mean(chain_means)

    # Between-chain variance
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)

    # Within-chain variance
    W = np.mean([np.var(c, ddof=1) for c in chains])

    if W < 1e-10:
        return 1.0

    # Pooled variance estimate
    var_hat = (1 - 1 / n) * W + B / n

    return float(np.sqrt(var_hat / W))


def analyze_employee_productivity_frontier(
    df: pd.DataFrame, sector_col: str = "industry"
) -> pd.DataFrame:
    """
    Identify companies with superior human capital efficiency using industry-adjusted rankings.

    Features: profit_per_employee, ebitda_per_employee, revenue_per_employee, workforce_stability
    """
    metrics = ["profit_per_employee", "ebitda_per_employee", "revenue_per_employee"]

    # Filter for available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return df

    result = df.copy()

    # Calculate industry-adjusted scores
    for metric in available_metrics:
        # Normalize by sector (z-score)
        result[f"{metric}_sector_z"] = result.groupby(sector_col)[metric].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    # Calculate productivity frontier score (average of available z-scores)
    z_cols = [f"{m}_sector_z" for m in available_metrics]
    result["productivity_frontier_score"] = result[z_cols].mean(axis=1)

    # Add workforce stability if available
    if "workforce_stability" in df.columns:
        result["productivity_frontier_score"] += result["workforce_stability"] / 100

    # Rank companies
    result["productivity_rank"] = result.groupby(sector_col)["productivity_frontier_score"].rank(
        ascending=False
    )

    return result


def detect_accounting_anomalies(
    df: pd.DataFrame,
    anomaly_z_threshold: float | None = None,
    tier_bins: list[float] | None = None,
    tier_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Detect accounting anomalies using multi-layered statistical analysis.

    Enhanced from the original z-score approach with:
    - **Robust z-scores** using median/MAD instead of mean/std (outlier-resistant)
    - **Distribution fitting** — selects best-fit (Normal vs Student-t vs Laplace)
      per feature using AIC, then scores outliers against the fitted distribution
    - **Sector-relative scoring** — anomaly score adjusted for industry norms
    - **Feature-level flags** — individual feature anomaly flags (>2σ robust)
    - **Composite anomaly tier** — categorical risk label (Clean / Watch / Flag / Alert)
    - **Mahalanobis distance** — multivariate outlier detection across all features
    - **Benford's Law test** — digit distribution test for earnings manipulation signals

    Parameters
    ----------
    df : pd.DataFrame
        Financial data. Expected to include some or all of:
        - exceptional_items_frequency, non_operating_income_share
        - gaap_adj_eps_gap_pct, asset_sale_boost
        - ebitda_adjustment_ratio, eps_adjustment_ratio
        - exceptional_items_to_ebitda, restructuring_intensity
        - goodwill_change_rate
    anomaly_z_threshold : float or None
        Robust z-score threshold for flagging anomalies. None -> derived
        from the ``accounting_anomaly_score`` distribution (75th percentile
        mapped to a z-score cutoff); falls back to 2.5.
    tier_bins : list[float] or None
        Bin edges for anomaly tier classification. None -> derived from
        the score distribution quartiles; falls back to
        ``[-0.1, 25, 50, 75, 100.1]``.
    tier_labels : list[str] or None
        Labels for the tier bins. None -> ``['Clean', 'Watch', 'Flag', 'Alert']``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - accounting_anomaly_score: Composite score normalized to 0-100
        - accounting_anomaly_tier: 'Clean' / 'Watch' / 'Flag' / 'Alert'
        - {feature}_z_robust: Robust (MAD-based) z-score per feature
        - {feature}_anomaly_flag: Boolean flag if robust |z| > 2.5
        - {feature}_dist_name: Best-fit distribution name
        - {feature}_dist_pvalue: Goodness-of-fit p-value
        - anomaly_feature_count: Number of features flagged as anomalous
        - mahalanobis_distance: Multivariate outlier distance
        - sector_relative_anomaly: Anomaly score relative to sector median
        - benford_chi2_pvalue: Benford's Law chi-squared p-value (if applicable)
    """
    # ── Dynamic threshold computation ──
    if anomaly_z_threshold is None:
        specs: dict[str, dict] = {}
        if "accounting_anomaly_score" in df.columns:
            specs["accounting_anomaly_score"] = {
                "direction": "max",
                "percentile": 75,
                "fallback": 2.5,
            }
        if specs:
            dynamic = _compute_dynamic_thresholds(df, specs)
            # Map the 75th-pctl score to a z-score proxy (normalize to ~2-3 range)
            raw = dynamic.get("accounting_anomaly_score", 2.5)
            anomaly_z_threshold = max(1.5, min(raw / 10.0, 4.0)) if raw > 10 else 2.5
        else:
            anomaly_z_threshold = 2.5

    if tier_labels is None:
        tier_labels = ["Clean", "Watch", "Flag", "Alert"]

    if tier_bins is None:
        # Derive tier bins from data distribution if score column exists
        if "accounting_anomaly_score" in df.columns:
            score_data = df["accounting_anomaly_score"].dropna()
            if len(score_data) >= 20:
                q25 = float(score_data.quantile(0.25))
                q50 = float(score_data.quantile(0.50))
                q75 = float(score_data.quantile(0.75))
                tier_bins = [-0.1, q25, q50, q75, 100.1]
            else:
                tier_bins = [-0.1, 30, 60, 90, 100.1]
        else:
            tier_bins = [-0.1, 30, 60, 90, 100.1]
    features = [
        # ── Original features ──
        "exceptional_items_frequency",
        "gaap_adj_eps_gap_pct",
        "asset_sale_boost",
        "ebitda_adjustment_ratio",
        "eps_adjustment_ratio",
        "exceptional_items_to_ebitda",
        "restructuring_intensity",
        "goodwill_change_rate",
        # ── EPS adjustment features ──
        "eps_adj_ltm",
        "eps_adjustment_ratio_comp",
        "eps_adjustment_spread_ltm",
        "eps_adjustment_spread_fy",
        "eps_adjustment_pct",
        # ── Net income adjustment features ──
        "net_income_adjustment_ratio_ltm",
        "net_income_adjustment_ratio_fy",
        "net_income_adjustment_pct",
        # ── EBITDA / EBIT adjustment features ──
        "ebitda_adjustment_pct_ltm",
        "ebitda_adjustment_pct_fy",
        "ebit_adjustment_pct_ltm",
        "ebit_adjustment_pct_fy",
        # ── GAAP vs non-GAAP spread & revision features ──
        "forward_eps_gaap_adj_spread",
        "gaap_vs_norm_revision_spread",
        "gaap_revision_momentum",
        "gaap_revision_1m",
        "gaap_revision_3m",
        "gaap_revision_6m",
        "gaap_revision_1y",
        # ── Earnings quality & discontinuities ──
        "discontinued_ops_impact",
        "earnings_quality_warning",
        "revision_quality_divergence",
        # ── Surprise & growth acceleration ──
        "eps_growth_accel",
        "eps_surprise_pct",
        "revenue_surprise_pct",
        # ── Unusual Items (Cat 17) ──
        "total_unusual_items",
        "unusual_items_to_revenue",
        "unusual_items_to_ebitda",
        "earnings_quality_impact",
        # ── Asset Sale (Cat 21) ──
        "asset_sale_gain_loss_ltm",
        "asset_sale_frequency",
        "asset_sale_trend",
        # ── Tax Rate (Cat 19) ──
        "tax_rate_stability",
        "tax_rate_trend_4q",
        # ── Investment Income (Cat 24) ──
        "interest_income_to_revenue",
        "interest_income_to_revenue_trend",
        # ── Balance Sheet & Working Capital Flags (NEW v3.5) ──
        "accumulated_deficit_flag",
        "negative_wc_flag",
        "wc_deteriorating_flag",
        "intangibles_growth_flag",
        # ── Inventory Signals (NEW v3.5) ──
        "inventory_buildup_flag",
        "inventory_reduction_flag",
        # ── Impairment & Writedown Events (NEW v3.5) ──
        "has_goodwill_impairment",
        "has_asset_writedown",
        "has_restructuring",
        "has_goodwill_impairment_ltm",
        "impairment_risk_score",
        # ── Strategic & Operational Red Flags (NEW v3.5) ──
        "revenue_accelerating_flag",
        "overinvestment_flag",
        "recent_acquisition_flag",
        "high_rnd_intensity_flag",
        "has_unusual_items_flag",
        "low_tax_flag",
        "layoff_risk_flag",
        # ── External Signals (NEW v3.5) ──
        "analyst_bearish_pct",
        "debt_maturity_risk",
    ]

    # Feature importance weights (higher = more indicative of manipulation)
    feature_weights = {
        # ── Original features ──
        "exceptional_items_frequency": 1.0,
        "non_operating_income_share": 1.2,
        "gaap_adj_eps_gap_pct": 1.5,  # GAAP-adjusted gap is a strong signal
        "asset_sale_boost": 1.3,
        "ebitda_adjustment_ratio": 1.4,
        "eps_adjustment_ratio": 1.5,  # EPS adjustments are high-signal
        "exceptional_items_to_ebitda": 1.1,
        "restructuring_intensity": 0.8,  # Restructuring can be legitimate
        "goodwill_change_rate": 0.9,
        # ── EPS adjustment features ──
        "eps_adj_ltm": 1.3,  # Trailing EPS adjustment — direct quality signal
        "eps_adjustment_ratio_comp": 1.5,  # Comparative ratio amplifies divergence
        "eps_adjustment_spread_ltm": 1.6,  # Wide LTM spread = aggressive adjustments
        "eps_adjustment_spread_fy": 1.5,  # FY spread — forward-looking manipulation risk
        "eps_adjustment_pct": 1.4,  # Magnitude of EPS adjustments
        # ── Net income adjustment features ──
        "net_income_adjustment_ratio_ltm": 1.4,  # NI adjustments distort bottom line directly
        "net_income_adjustment_ratio_fy": 1.3,  # Forward NI adjustment — moderate signal
        "net_income_adjustment_pct": 1.4,  # Percentage-based NI adjustments
        # ── EBITDA / EBIT adjustment features ──
        "ebitda_adjustment_pct_ltm": 1.3,  # EBITDA adjustments — common in add-backs
        "ebitda_adjustment_pct_fy": 1.2,  # Forward EBITDA — slightly less actionable
        "ebit_adjustment_pct_ltm": 1.3,  # EBIT trailing adjustments
        "ebit_adjustment_pct_fy": 1.2,  # EBIT forward adjustments
        # ── GAAP vs non-GAAP spread & revision features ──
        "forward_eps_gaap_adj_spread": 1.6,  # Forward GAAP vs adjusted spread — very high signal
        "gaap_vs_norm_revision_spread": 1.7,  # Divergence between GAAP & normalized revisions — top signal
        "gaap_revision_momentum": 1.4,  # Accelerating GAAP revisions suggest deterioration
        "gaap_revision_1m": 1.0,  # Short-term revisions — noisy but timely
        "gaap_revision_3m": 1.1,  # Medium-term — more reliable
        "gaap_revision_6m": 1.2,  # Confirmed trend
        "gaap_revision_1y": 1.3,  # Long-term revision drift — persistent signal
        # ── Earnings quality & discontinuities ──
        "discontinued_ops_impact": 1.5,  # Discontinued operations mask ongoing performance
        "earnings_quality_warning": 1.8,  # Explicit quality flag — highest weight in group
        "revision_quality_divergence": 1.6,  # Analyst vs GAAP revision disagreement
        # ── Surprise & growth acceleration ──
        "eps_growth_accel": 0.7,  # Growth acceleration is often legitimate
        "eps_surprise_pct": 0.8,  # Surprises alone are weak anomaly signals
        "revenue_surprise_pct": 0.7,  # Revenue surprises — even more common legitimately
        # ── Unusual Items (Cat 17) ──
        "total_unusual_items": 1.4,
        "unusual_items_to_revenue": 1.5,
        "unusual_items_to_ebitda": 1.3,
        "earnings_quality_impact": 1.6,
        # ── Asset Sale (Cat 21) ──
        "asset_sale_gain_loss_ltm": 1.2,
        "asset_sale_frequency": 1.3,
        "asset_sale_trend": 1.1,
        # ── Tax Rate (Cat 19) ──
        "tax_rate_stability": 0.9,  # Low stability = volatile tax rate
        "tax_rate_trend_4q": 1.0,
        # ── Investment Income (Cat 24) ──
        "interest_income_to_revenue": 0.8,
        "interest_income_to_revenue_trend": 0.9,
        # ── Balance Sheet & Working Capital Flags (NEW v3.5) ──
        "accumulated_deficit_flag": 1.4,      # Persistent losses = manipulation pressure
        "negative_wc_flag": 1.3,              # Aggressive revenue recognition signal
        "wc_deteriorating_flag": 1.5,         # Deteriorating WC + stable earnings = red flag
        "intangibles_growth_flag": 1.2,       # Intangibles inflation risk
        # ── Inventory Signals (NEW v3.5) ──
        "inventory_buildup_flag": 1.6,        # Channel-stuffing / demand mismatch
        "inventory_reduction_flag": 1.1,      # Write-downs or demand collapse
        # ── Impairment & Writedown Events (NEW v3.5) ──
        "has_goodwill_impairment": 1.7,       # Direct impairment event — high signal
        "has_asset_writedown": 1.6,           # Prior asset overvaluation
        "has_restructuring": 1.3,             # Big-bath accounting risk
        "has_goodwill_impairment_ltm": 1.8,   # Recent impairment — very high signal
        "impairment_risk_score": 1.7,         # Composite impairment risk
        # ── Strategic & Operational Red Flags (NEW v3.5) ──
        "revenue_accelerating_flag": 0.8,     # Can be legitimate; mild signal alone
        "overinvestment_flag": 1.2,           # Capital misallocation
        "recent_acquisition_flag": 1.4,       # Enables goodwill/revenue inflation
        "high_rnd_intensity_flag": 0.9,       # R&D capitalization risk
        "has_unusual_items_flag": 1.5,        # Complements existing unusual items features
        "low_tax_flag": 1.3,                  # Aggressive tax strategy signal
        "layoff_risk_flag": 1.0,              # Cost deferral / restructuring masking
        # ── External Signals (NEW v3.5) ──
        "analyst_bearish_pct": 1.1,           # Sentiment divergence from fundamentals
        "debt_maturity_risk": 1.2,            # Manipulation incentive pressure
    }

    available = [f for f in features if f in df.columns]
    if not available:
        return df

    result = df.copy()
    result["accounting_anomaly_score"] = 0.0
    total_weight = 0.0

    # Collect per-feature columns to avoid DataFrame fragmentation (GH PerformanceWarning)
    _new_cols: dict[str, pd.Series | np.ndarray] = {}

    # ── Layer 1: Robust z-scores (Median / MAD) ──
    for feat in available:
        data = result[feat].dropna()
        if len(data) <= 10:
            continue

        median_val = data.median()
        mad = stats.median_abs_deviation(data, nan_policy="omit")
        # MAD-based robust z-score (scale factor 1.4826 for Normal consistency)
        mad_scaled = mad * 1.4826 if mad > 1e-10 else data.std()

        if mad_scaled > 1e-10:
            robust_z = (result[feat] - median_val) / mad_scaled
        else:
            robust_z = pd.Series(0.0, index=result.index)

        _new_cols[f"{feat}_z_robust"] = robust_z.abs()
        _new_cols[f"{feat}_anomaly_flag"] = robust_z.abs() > anomaly_z_threshold

        # ── Layer 2: Distribution fitting per feature ──
        weight = feature_weights.get(feat, 1.0)
        clean_data = data.values
        best_dist_name = "normal"
        best_pvalue = 0.0

        candidates = [
            ("normal", stats.norm),
            ("student_t", stats.t),
            ("laplace", stats.laplace),
        ]
        best_aic = np.inf
        best_fit_dist = None

        for dist_name, dist_obj in candidates:
            try:
                params = dist_obj.fit(clean_data)
                log_lik = dist_obj.logpdf(clean_data, *params).sum()
                k = len(params)
                aic = 2 * k - 2 * log_lik
                if aic < best_aic:
                    best_aic = aic
                    best_dist_name = dist_name
                    best_fit_dist = (dist_obj, params)
                    # KS goodness-of-fit test
                    ks_stat, ks_pval = stats.kstest(clean_data, dist_obj.cdf, args=params)
                    best_pvalue = float(ks_pval)
            except Exception:
                continue

        _new_cols[f"{feat}_dist_name"] = best_dist_name
        _new_cols[f"{feat}_dist_pvalue"] = best_pvalue

        # Score: use fitted distribution's survival function for tail probability
        if best_fit_dist is not None:
            dist_obj, params = best_fit_dist
            # Two-tailed: probability of observing a value as extreme or more
            cdf_vals = dist_obj.cdf(result[feat].values, *params)
            tail_prob = np.minimum(cdf_vals, 1 - cdf_vals) * 2
            # Convert to anomaly contribution: lower tail prob = higher anomaly
            feat_score = np.where(
                np.isnan(tail_prob), 0.0, -np.log10(np.clip(tail_prob, 1e-10, 1.0))
            )
        else:
            feat_score = _new_cols[f"{feat}_z_robust"].fillna(0).values

        result["accounting_anomaly_score"] += feat_score * weight
        total_weight += weight

    # Merge all per-feature columns at once to avoid fragmentation
    if _new_cols:
        result = pd.concat([result, pd.DataFrame(_new_cols, index=result.index)], axis=1)

    # ── Layer 3: Multivariate outlier detection (Mahalanobis distance) ──
    numeric_available = [f for f in available if f in result.columns]
    feature_matrix = result[numeric_available].copy()
    complete_mask = feature_matrix.notna().all(axis=1)

    if complete_mask.sum() > len(numeric_available) + 5:
        complete_data = feature_matrix.loc[complete_mask].values
        try:
            mean_vec = np.mean(complete_data, axis=0)
            cov_matrix = np.cov(complete_data, rowvar=False)

            # Regularise covariance to ensure invertibility
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov_matrix)

            # Compute Mahalanobis distance for all complete rows
            diff = complete_data - mean_vec
            mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

            result.loc[complete_mask, "mahalanobis_distance"] = mahal

            # Add Mahalanobis contribution to anomaly score (chi-squared tail prob)
            p_dims = len(numeric_available)
            mahal_pvalue = 1 - stats.chi2.cdf(mahal**2, df=p_dims)
            mahal_score = -np.log10(np.clip(mahal_pvalue, 1e-10, 1.0))
            result.loc[complete_mask, "accounting_anomaly_score"] += mahal_score
            total_weight += 1.0
        except np.linalg.LinAlgError:
            logger.debug("Mahalanobis distance skipped — singular covariance matrix")

    # ── Layer 4: Sector-relative scoring ──
    sector_col = "industry" if "industry" in result.columns else "sector"
    if sector_col in result.columns:
        sector_median = result.groupby(sector_col)["accounting_anomaly_score"].transform("median")
        sector_std = result.groupby(sector_col)["accounting_anomaly_score"].transform("std")
        result["sector_relative_anomaly"] = np.where(
            sector_std > 0,
            (result["accounting_anomaly_score"] - sector_median) / sector_std,
            0.0,
        )
    else:
        result["sector_relative_anomaly"] = 0.0

    # ── Layer 5: Benford's Law test (digit distribution) ──
    # Applied to gaap_adj_eps_gap_pct or eps_adjustment_ratio if available
    benford_col = next(
        (c for c in ["gaap_adj_eps_gap_pct", "eps_adjustment_ratio"] if c in result.columns),
        None,
    )
    if benford_col is not None:
        benford_data = result[benford_col].dropna().abs()
        benford_data = benford_data[benford_data > 0]
        if len(benford_data) > 50:
            leading_digits = benford_data.apply(
                lambda x: int(str(f"{abs(x):.10f}").lstrip("0").lstrip(".")[0]) if x != 0 else 0
            )
            leading_digits = leading_digits[leading_digits.between(1, 9)]
            if len(leading_digits) > 10:
                observed = leading_digits.value_counts().reindex(range(1, 10), fill_value=0)
                expected_benford = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
                expected_counts = expected_benford * len(leading_digits)
                chi2, p_val = stats.chisquare(observed.values, f_exp=expected_counts)
                result["benford_chi2_pvalue"] = float(p_val)
            else:
                result["benford_chi2_pvalue"] = np.nan
        else:
            result["benford_chi2_pvalue"] = np.nan
    else:
        result["benford_chi2_pvalue"] = np.nan

    # ── Normalize composite score to 0-100 ──
    max_score = result["accounting_anomaly_score"].max()
    if max_score > 0:
        result["accounting_anomaly_score"] = (result["accounting_anomaly_score"] / max_score) * 100

    # ── Anomaly feature count ──
    flag_cols = [c for c in result.columns if c.endswith("_anomaly_flag")]
    if flag_cols:
        result["anomaly_feature_count"] = result[flag_cols].sum(axis=1).astype(int)
    else:
        result["anomaly_feature_count"] = 0

    # ── Composite anomaly tier ──
    result["accounting_anomaly_tier"] = pd.cut(
        result["accounting_anomaly_score"],
        bins=tier_bins,
        labels=tier_labels,
    )

    # ── Quality frequency flags (v3.4 → v3.5) ──
    freq_cols = [
        "goodwill_impairment_frequency",
        "asset_writedown_frequency",
        "restructuring_frequency",
        "exceptional_items_frequency",
    ]
    event_cols = [
        "has_goodwill_impairment",
        "has_asset_writedown",
        "has_restructuring",
        "has_goodwill_impairment_ltm",
        "has_unusual_items_flag",
    ]
    available_freq = [c for c in freq_cols if c in df.columns]
    available_events = [c for c in event_cols if c in df.columns]
    if available_freq or available_events:
        result["quality_frequency_score"] = (
            df[available_freq].fillna(0).sum(axis=1) if available_freq else 0
        ) + (
            df[available_events].fillna(0).sum(axis=1) if available_events else 0
        )
        result["repeat_offender_flag"] = (result["quality_frequency_score"] >= 10).astype(int)

    return result


def analyze_reporting_lag_sentiment(df: pd.DataFrame) -> dict:
    """
    Test the "bad news travels slow" hypothesis: relationship between reporting_lag and earnings misses.

    Features: reporting_lag, eps_surprise_pct, days_to_earnings
    """
    if "reporting_lag" not in df.columns or "eps_surprise_pct" not in df.columns:
        return {
            "correlation": 0,
            "p_value": 1.0,
            "hypothesis_confirmed": False,
            "sample_size": 0,
        }

    data = df[["reporting_lag", "eps_surprise_pct"]].dropna()
    if len(data) < 5:
        return {
            "correlation": 0,
            "p_value": 1.0,
            "hypothesis_confirmed": False,
            "sample_size": len(data),
        }

    corr, p_val = stats.spearmanr(data["reporting_lag"], data["eps_surprise_pct"])

    # If correlation is negative and significant, hypothesis is confirmed
    # (Higher lag correlated with lower (negative) surprise)
    confirmed = corr < -0.1 and p_val < 0.05

    return {
        "correlation": float(corr),
        "p_value": float(p_val),
        "hypothesis_confirmed": bool(confirmed),
        "sample_size": int(len(data)),
    }


def analyze_accounting_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated: use AccountingAnomalyProbabilityModel.analyze_dataframe().

    Extended analytics layer on top of ``detect_accounting_anomalies`` output.
    This function now delegates to
    :class:`~analytics.probability_analytics.AccountingAnomalyProbabilityModel`.
    """
    import warnings

    warnings.warn(
        "analyze_accounting_anomalies() is deprecated. "
        "Use AccountingAnomalyProbabilityModel().analyze_dataframe(df) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from probabilistic_ml_model.statistical_functions.probability_models import (
        AccountingAnomalyProbabilityModel,
    )

    return AccountingAnomalyProbabilityModel().analyze_dataframe(df)


def run_category_probability_analytics(
    df: pd.DataFrame,
    category_name: str,
    features: list[str],
    n_simulations: int = 10000,
) -> dict:
    """
    Run comprehensive probability analytics for a feature category.

    Combines Bayesian analysis, distribution fitting, and conditional
    probability calculations for all features in a category.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature data
    category_name : str
        Name of the feature category (e.g., "Valuation Ratios")
    features : list[str]
        List of feature column names in this category
    n_simulations : int, default 10000
        Number of Monte Carlo simulations

    Returns
    -------
    dict
        Dictionary containing:
        - 'bayesian_results': Posterior distributions per feature
        - 'distribution_fits': Best-fit distributions with parameters
        - 'conditional_probs': P(Distress | Feature) analysis
        - 'summary_statistics': Descriptive statistics
    """
    available_features = [f for f in features if f in df.columns]

    results = {
        "category": category_name,
        "features_analyzed": len(available_features),
        "bayesian_results": {},
        "distribution_fits": {},
        "conditional_probs": {},
        "summary_statistics": {},
    }

    # 1. Bayesian parameter estimation
    bayesian = bayesian_category_analysis(df, category_name, available_features)
    results["bayesian_results"] = bayesian

    # 2. Distribution fitting
    dist_fits = fit_distributions_by_category(df, category_name, available_features, n_simulations)
    results["distribution_fits"] = dist_fits

    # 3. Conditional probabilities (if combined_distress_score available)
    if "combined_distress_score" in df.columns:
        cond_probs = calculate_conditional_probabilities(df, {category_name: available_features})
        results["conditional_probs"] = cond_probs

    # 4. Summary statistics
    for feat in available_features:
        data = pd.to_numeric(df[feat], errors="coerce").dropna()
        if len(data) > 0:
            # v3.5: Clip to prevent overflow in higher-moment calculations (Issue 12)
            data_safe = data.clip(-1e12, 1e12)
            results["summary_statistics"][feat] = {
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "skewness": float(data_safe.skew()),
                "kurtosis": float(data_safe.kurtosis()),
            }

    return results


def run_all_views_probability_analytics(
    views_dict: dict[str, pd.DataFrame],
    view_category_mapping: dict[str, str],
) -> dict[str, dict]:
    """
    Run probability analytics for all feature views.

    Parameters
    ----------
    views_dict : dict[str, pd.DataFrame]
        Dictionary of DataFrames keyed by view name
    view_category_mapping : dict[str, str]
        Mapping from view name to category name

    Returns
    -------
    dict[str, dict]
        Analytics results for each view
    """
    from probabilistic_ml_model.data_utils import get_identifier_cols_set

    all_results = {}
    identifier_cols = get_identifier_cols_set()

    for view_name, df_view in views_dict.items():
        if df_view.empty:
            continue

        category_name = view_category_mapping.get(view_name, view_name)
        feature_cols = [c for c in df_view.columns if c not in identifier_cols]

        logging.info("Running analytics for %s (%d features)", category_name, len(feature_cols))

        results = run_category_probability_analytics(df_view, category_name, feature_cols)
        all_results[view_name] = results

    return all_results


def export_probability_view_results(
    df: pd.DataFrame,
    view_name: str,
    feature_cols: list[str],
    identifier_cols: list[str] | None = None,
) -> int | None:
    """
    Export per-feature probability metrics to analytics prob_vw_features_* tables.

    Computes percentile, z-score, and P(above median) for each feature
    and writes to the corresponding analytics table in long format.
    Uses standardized identifier columns from vw_identifier_columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame with feature data
    view_name : str
        Source view name (e.g., 'vw_features_earnings')
    feature_cols : list[str]
        Feature columns to compute probabilities for
    identifier_cols : list[str], optional
        Identifier columns to include. If None, loads from
        vw_identifier_columns via data_utils.

    Returns
    -------
    int or None
        Number of rows exported
    """
    from scipy import stats as sp_stats

    from probabilistic_ml_model.data_utils import export_to_db, load_identifier_columns

    if identifier_cols is None:
        identifier_cols = load_identifier_columns()

    available_ids = [c for c in identifier_cols if c in df.columns]
    rows = []

    # Computes and appends feature statistics for each valid value
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        data = pd.to_numeric(df[feat], errors="coerce")
        valid = data.dropna()
        if len(valid) < 10:
            continue

        median_val = valid.median()
        mean_val = valid.mean()
        std_val = valid.std()

        for idx in df.index:
            val = data.loc[idx]
            if pd.isna(val):
                continue
            row = {c: df.loc[idx, c] for c in available_ids if c in df.columns}
            row["feature"] = feat
            row["value"] = float(val)
            row["percentile"] = float(sp_stats.percentileofscore(valid, val))
            row["z_score"] = float((val - mean_val) / std_val) if std_val > 0 else 0.0
            row["prob_above_median"] = 1.0 if val > median_val else 0.0
            rows.append(row)

    if not rows:
        return 0

    result_df = pd.DataFrame(rows)

    # Reorder columns: identifier cols first, then metric cols
    id_cols_ordered = [c for c in identifier_cols if c in result_df.columns]
    metric_cols = [c for c in result_df.columns if c not in id_cols_ordered]
    result_df = result_df[id_cols_ordered + metric_cols]

    table_name = f"prob_{view_name}"
    return export_to_db(result_df, table_name)


def bayesian_earnings_beat_model(df: pd.DataFrame, n_total: int = 5) -> pd.DataFrame:
    """
    Bayesian model for earnings beat probability.

    Uses EPS positive streak as prior evidence and updates posterior
    based on recent performance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - ticker, name, industry
        - eps_positive_streak (number of positive quarters in last n_total)
    n_total : int, default 5
        Total number of quarters in the observation window

    Returns
    -------
    pd.DataFrame
        DataFrame with Bayesian model results:
        - ticker, name, industry, eps_positive_streak
        - posterior_beat_prob, model_confidence, map_estimate
    """
    # Prior: Uniform belief across probability grid
    p_grid = np.linspace(0.01, 0.99, 200)  # Fine-grained grid for smooth posterior
    uniform_prior = 1 / len(p_grid)

    results = []

    streak_col = "eps_positive_streak"
    if streak_col not in df.columns:
        return pd.DataFrame()

    for _, row in df.dropna(subset=[streak_col]).iterrows():
        n_beats = int(row[streak_col])
        n_beats = min(n_beats, n_total)  # Cap at n_total

        # Compute likelihood: P(data | p) = p^k * (1-p)^(n-k)
        likelihoods = p_grid**n_beats * (1 - p_grid) ** (n_total - n_beats)

        # Unnormalized posterior
        posterior_unnorm = uniform_prior * likelihoods

        # Normalize
        posterior = posterior_unnorm / posterior_unnorm.sum()

        # Posterior predictive: P(beat next quarter) = sum(p * posterior(p))
        prob_beat_next = np.sum(p_grid * posterior)

        # Confidence (inverse entropy proxy)
        entropy = -np.sum(posterior * np.log(posterior + 1e-10))
        confidence = 1 - entropy / np.log(len(p_grid))

        results.append(
            {
                "isin": row.get("isin", ""),
                "name": row.get("name", ""),
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "region": row.get("region", ""),
                "country": row.get("country", ""),
                "exchange": row.get("exchange", ""),
                "eps_positive_streak": n_beats,
                "posterior_beat_prob": prob_beat_next,
                "model_confidence": confidence,
                "map_estimate": p_grid[np.argmax(posterior)],  # Maximum a posteriori
            }
        )

    return pd.DataFrame(results)


def analyze_distress_distribution(
    df: pd.DataFrame,
    high_risk_threshold: float | None = None,
    low_risk_threshold: float | None = None,
) -> go.Figure:
    """
    Analyze distress risk score distribution with tail risk metrics.

    Uses concepts from MCMC sampling to understand distribution shape.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - combined_distress_score
        - industry
    high_risk_threshold : float or None
        Score below which a stock is considered high risk. None -> derived
        from the distress score distribution (25th percentile); falls back
        to 30.
    low_risk_threshold : float or None
        Score above which a stock is considered low risk. None -> derived
        from the distress score distribution (75th percentile); falls back
        to 70.

    Returns
    -------
    Figure
        Plotly Figure with 4 panels:
        1. Distress risk score distribution with fitted normal
        2. Empirical CDF
        3. Q-Q plot vs normal
        4. Tail risk by industry
    """
    distress_data = df["combined_distress_score"].dropna()

    # ── Dynamic threshold computation ──
    if high_risk_threshold is None or low_risk_threshold is None:
        specs: dict[str, dict] = {}
        if high_risk_threshold is None:
            specs["combined_distress_risk_score_low"] = {
                "direction": "min",
                "percentile": 25,
                "fallback": 30,
            }
        if low_risk_threshold is None:
            specs["combined_distress_risk_score_high"] = {
                "direction": "min",
                "percentile": 75,
                "fallback": 70,
            }
        # Use the actual column for both specs
        if len(distress_data) >= 30:
            if high_risk_threshold is None:
                high_risk_threshold = float(distress_data.quantile(0.25))
            if low_risk_threshold is None:
                low_risk_threshold = float(distress_data.quantile(0.75))
        else:
            if high_risk_threshold is None:
                high_risk_threshold = 30
            if low_risk_threshold is None:
                low_risk_threshold = 70

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Distress Risk Score Distribution",
            "Empirical CDF",
            "Q-Q Plot vs Normal",
            "Tail Risk by Industry",
        ],
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # Panel 1: Histogram with fitted distribution
    fig.add_trace(
        go.Histogram(
            x=distress_data,
            nbinsx=50,
            name="Observed",
            marker_color="#3498db",
            opacity=0.7,
            histnorm="probability density",
        ),
        row=1,
        col=1,
    )

    # Fit normal for comparison
    mu, std = distress_data.mean(), distress_data.std()
    x_range = np.linspace(0, 100, 100)
    normal_pdf = stats.norm.pdf(x_range, mu, std)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_pdf,
            mode="lines",
            name="Normal Fit",
            line=dict(color="#e74c3c", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Panel 2: Empirical CDF
    sorted_data = np.sort(distress_data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig.add_trace(
        go.Scatter(x=sorted_data, y=ecdf, mode="lines", name="ECDF", line=dict(color="#00bc8c")),
        row=1,
        col=2,
    )
    # Add risk thresholds
    fig.add_vline(
        x=high_risk_threshold,
        line_dash="dot",
        line_color="#e74c3c",
        row=1,
        col=2,
        annotation_text=f"High Risk (<{high_risk_threshold:.0f})",
    )
    fig.add_vline(
        x=low_risk_threshold,
        line_dash="dot",
        line_color="#2ecc71",
        row=1,
        col=2,
        annotation_text=f"Low Risk (>{low_risk_threshold:.0f})",
    )

    # Panel 3: Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    empirical_quantiles = np.percentile(distress_data, np.linspace(1, 99, 100))
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=empirical_quantiles,
            mode="markers",
            marker=dict(size=4, color="#9b59b6"),
            name="Q-Q",
        ),
        row=2,
        col=1,
    )
    # Reference line
    fig.add_trace(
        go.Scatter(
            x=[-3, 3],
            y=[mu - 3 * std, mu + 3 * std],
            mode="lines",
            line=dict(dash="dash", color="white"),
            name="Normal Ref",
        ),
        row=2,
        col=1,
    )

    # Panel 4: Tail risk by industry (% below high_risk_threshold)
    if "industry" in df.columns:
        _hr_thresh = high_risk_threshold
        tail_risk = (
            df.groupby("industry")
            .apply(
                lambda x: (x["combined_distress_score"] < _hr_thresh).mean() * 100,
                include_groups=False,
            )
            .sort_values(ascending=False)
        )

        fig.add_trace(
            go.Bar(
                x=tail_risk.values[:15],
                y=tail_risk.index[:15],
                orientation="h",
                marker_color="#e74c3c",
                name="High Risk %",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=800,
        title_text="📉 Financial Distress Risk Distribution Analysis",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    # Compute tail risk metrics
    var_5 = np.percentile(distress_data, 5)
    var_1 = np.percentile(distress_data, 1)
    high_risk_pct = (distress_data < high_risk_threshold).mean() * 100

    # Add annotations
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.08,
        y=1.0,
        text=f"μ={mu:.1f}, σ={std:.1f}",
        showarrow=False,
        font=dict(size=12),
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.08,
        y=0.9,
        text=f"VaR(5%): {var_5:.1f}",
        showarrow=False,
        font=dict(size=12),
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.08,
        y=0.8,
        text=f"VaR(1%): {var_1:.1f}",
        showarrow=False,
        font=dict(size=12),
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.08,
        y=0.7,
        text=f"High Risk (<{high_risk_threshold:.0f}): {high_risk_pct:.1f}%",
        showarrow=False,
        font=dict(size=12),
    )

    return fig
