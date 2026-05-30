"""Helpers for the PyMC price-target notebook (``pymc_price_target.ipynb``).

Contains pure-NumPy/pandas building blocks that are independently testable
without requiring PyMC at import time:

* :func:`prepare_price_target_inputs` — derives ``target_vs_price_pct``,
  ``feat_conviction_ratio`` and ``feat_target_dispersion_cv`` from the raw
  ``pml.mv_pymc_price_target`` materialised view and returns the arrays the
  PyMC model consumes. Aligned with
  :class:`probabilistic_ml_model.pymc_models.PriceTargetModel.PriceTargetAchievement`.

* :func:`simulate_lagged_risk_adjusted_returns` — Monte-Carlo lagged
  risk-adjusted expected returns built on top of posterior draws of
  ``risk_adj_return``, ``sigma_isin`` and ``nu`` using a structural-TS
  AR(1) lag (per the issue spec).

* :func:`summarize_mc_returns` — per-ISIN summary DataFrame
  (``er_mean``, ``er_p05``, ``er_p50``, ``er_p95``, ``prob_pos``).

The PyMC model itself lives in the notebook (it depends on a sampler) but
the data preparation and the Monte-Carlo lag step are deterministic given
a ``random_seed`` and thus testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "PriceTargetInputs",
    "prepare_price_target_inputs",
    "simulate_lagged_risk_adjusted_returns",
    "summarize_mc_returns",
]


@dataclass(frozen=True)
class PriceTargetInputs:
    """Container of arrays consumed by the price-target PyMC model.

    Attributes
    ----------
    frame : pandas.DataFrame
        Filtered / augmented input frame (with derived ``target_vs_price_pct``,
        ``feat_conviction_ratio`` and ``feat_target_dispersion_cv``).
    isins, sectors : np.ndarray
        Per-row identifier / sector arrays.
    target_vs_price_pct : np.ndarray
        Observed target — ``(price_target_median - close) / close``.
    conviction_ratio : np.ndarray
        Analyst-buy ratio (drives heteroscedastic sigma prior).
    dispersion_cv : np.ndarray
        Coefficient of variation of analyst price targets.
    n_analysts : np.ndarray
        Floor-at-1 analyst count used for precision weighting.
    """

    frame: pd.DataFrame
    isins: np.ndarray
    sectors: np.ndarray
    target_vs_price_pct: np.ndarray
    conviction_ratio: np.ndarray
    dispersion_cv: np.ndarray
    n_analysts: np.ndarray


_PRICE_COL_ALIASES = ("last_price", "close")
_TARGET_AVG_ALIASES = ("price_target", "price_target_median")
_TARGET_NUM_ALIASES = ("n_analysts", "price_target_num")


def _resolve_alias(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    """Return the first column in *aliases* that exists in *df* (else None)."""
    for name in aliases:
        if name in df.columns:
            return name
    return None


def prepare_price_target_inputs(
        price_target_df: pd.DataFrame,
        *,
        price_col: Optional[str] = None,
        target_avg_col: Optional[str] = None,
        target_std_col: str = "price_target_stddev",
        target_num_col: Optional[str] = None,
        buys_col: str = "feat_net_buy_sentiment",
) -> PriceTargetInputs:
    """Schema-aligned data preparation for the price-target notebook.

    Drops rows missing the ``isin`` or ``sector`` identifiers or with a
    non-computable ``target_vs_price_pct`` (e.g. zero / NaN price).

    Column resolution is alias-aware so the helper works directly against the
    ``pml.mv_pymc_price_target`` materialised view (which exposes
    ``last_price``, ``price_target`` and ``n_analysts``) as well as legacy
    frames that use ``close`` / ``price_target_median`` / ``price_target_num``.
    Pre-computed ``feat_conviction_ratio`` and ``feat_target_dispersion_cv``
    columns (present in the MV) are used as-is when available; otherwise
    they are derived in-place.

    Parameters
    ----------
    price_target_df : pandas.DataFrame
        Frame backed by ``pml.mv_pymc_price_target``.
    price_col, target_avg_col, target_num_col : str, optional
        Explicit column-name overrides. When ``None`` (default) the helper
        auto-detects the column via the MV-first alias lists above.
    target_std_col, buys_col : str
        Std-of-targets and buy-count columns (MV-native defaults).

    Returns
    -------
    PriceTargetInputs
    """
    price_col = price_col or _resolve_alias(price_target_df, _PRICE_COL_ALIASES)
    target_avg_col = target_avg_col or _resolve_alias(price_target_df, _TARGET_AVG_ALIASES)
    target_num_col = target_num_col or _resolve_alias(price_target_df, _TARGET_NUM_ALIASES)

    required_logical = {
        "isin": "isin",
        "sector": "sector",
        "price (last_price/close)": price_col,
        "price-target avg (price_target/price_target_median)": target_avg_col,
        "price-target stddev": target_std_col if target_std_col in price_target_df.columns else None,
        "analyst count (n_analysts/price_target_num)": target_num_col,
        "buys count (feat_net_buy_sentiment)": buys_col if buys_col in price_target_df.columns else None,
    }
    missing = [label for label, col in required_logical.items()
               if col is None or col not in price_target_df.columns]
    # Also require base identifier columns
    for ident in ("isin", "sector"):
        if ident not in price_target_df.columns and ident not in missing:
            missing.append(ident)
    if missing:
        raise KeyError(f"price_target_df is missing required columns: {sorted(missing)}")

    df = price_target_df.dropna(subset=["isin", "sector"]).copy()

    # target_vs_price_pct = (price_target_median - close) / close
    close = df[price_col].astype("float64")
    df = df.loc[close.notna() & close.ne(0.0)].copy()
    close = df[price_col].astype("float64")
    df["target_vs_price_pct"] = (df[target_avg_col].astype("float64") - close) / close
    df = df.dropna(subset=["target_vs_price_pct"]).reset_index(drop=True)

    # Conviction proxy — prefer MV-precomputed `feat_conviction_ratio` if present,
    # otherwise derive as feat_net_buy_sentiment / max(n_analysts, 1).
    if "feat_conviction_ratio" in df.columns:
        df["feat_conviction_ratio"] = (
            df["feat_conviction_ratio"].astype("float64").fillna(0.0)
        )
    else:
        df["feat_conviction_ratio"] = (
                df[buys_col].astype("float64")
                / df[target_num_col].astype("float64").clip(lower=1)
        ).fillna(0.0)

    # Dispersion CV — prefer MV-precomputed `feat_target_dispersion_cv` if present.
    if "feat_target_dispersion_cv" in df.columns:
        df["feat_target_dispersion_cv"] = (
            df["feat_target_dispersion_cv"].astype("float64").fillna(0.0)
        )
    else:
        avg_abs = df[target_avg_col].astype("float64").abs().clip(lower=1e-6)
        df["feat_target_dispersion_cv"] = (
                df[target_std_col].astype("float64") / avg_abs
        ).fillna(0.0)

    return PriceTargetInputs(
        frame=df,
        isins=df["isin"].to_numpy(),
        sectors=df["sector"].to_numpy(),
        target_vs_price_pct=df["target_vs_price_pct"].to_numpy(dtype="float64"),
        conviction_ratio=df["feat_conviction_ratio"].to_numpy(dtype="float64"),
        dispersion_cv=df["feat_target_dispersion_cv"].to_numpy(dtype="float64"),
        n_analysts=df[target_num_col].astype("float64").clip(lower=1).to_numpy(),
    )


def simulate_lagged_risk_adjusted_returns(
        mu_draws: np.ndarray,
        sigma_draws: np.ndarray,
        nu_draws: np.ndarray,
        *,
        horizon: int = 4,
        rho: float = 0.85,
        random_seed: int = 42,
) -> np.ndarray:
    """Structural-TS style AR(1) Monte-Carlo simulation of risk-adjusted ERs.

    For each posterior draw ``s`` and ISIN ``i`` evolve

        r_{t+h} = rho * r_{t+h-1} + (1 - rho) * mu_i^s
                  + sigma_i^s * StudentT(df=max(nu^s, 3))

    over ``h = 0 .. horizon - 1``.

    Parameters
    ----------
    mu_draws : array of shape (n_isin, n_samples)
        Posterior draws of ``risk_adj_return`` (one row per ISIN).
    sigma_draws : array of shape (n_isin, n_samples)
        Posterior draws of ``sigma_isin``.
    nu_draws : array of shape (n_samples,) or scalar
        Posterior draws of the Student-t degrees of freedom ``nu``.
    horizon : int
        Number of forward periods to simulate (e.g. quarters).
    rho : float
        AR(1) persistence (0 <= rho < 1).
    random_seed : int
        Seed for the simulation's PRNG (reproducibility).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_isin, n_samples, horizon)`` of simulated ERs.
    """
    mu_draws = np.asarray(mu_draws, dtype="float64")
    sigma_draws = np.asarray(sigma_draws, dtype="float64")
    nu_draws = np.atleast_1d(np.asarray(nu_draws, dtype="float64"))

    if mu_draws.ndim != 2:
        raise ValueError("mu_draws must be 2-D (n_isin, n_samples).")
    if sigma_draws.shape != mu_draws.shape:
        raise ValueError("sigma_draws must match mu_draws shape.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0, 1).")

    n_isin, n_samples = mu_draws.shape
    if nu_draws.size not in (1, n_samples):
        raise ValueError(
            "nu_draws must be scalar or have length == mu_draws.shape[1]."
        )
    nu_row = np.broadcast_to(np.maximum(nu_draws, 3.0), (n_samples,))

    rng = np.random.default_rng(random_seed)
    mc = np.empty((n_isin, n_samples, horizon), dtype="float64")
    prev = mu_draws.copy()
    for h in range(horizon):
        eps = sigma_draws * rng.standard_t(df=nu_row, size=(n_isin, n_samples))
        prev = rho * prev + (1.0 - rho) * mu_draws + eps
        mc[:, :, h] = prev
    return mc


def summarize_mc_returns(
        mc: np.ndarray,
        isins: np.ndarray,
        *,
        quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95),
) -> pd.DataFrame:
    """Per-ISIN summary of the MC simulation output.

    Parameters
    ----------
    mc : array of shape (n_isin, n_samples, horizon)
        Output of :func:`simulate_lagged_risk_adjusted_returns`.
    isins : array of length ``n_isin``
        ISIN labels.
    quantiles : 3-tuple of float
        Lower / median / upper quantiles to report (default 5 / 50 / 95).

    Returns
    -------
    pandas.DataFrame
        Columns: ``isin, er_mean, er_p05, er_p50, er_p95, prob_pos``.
    """
    mc = np.asarray(mc, dtype="float64")
    if mc.ndim != 3:
        raise ValueError("mc must be 3-D (n_isin, n_samples, horizon).")
    if len(isins) != mc.shape[0]:
        raise ValueError("isins length must match mc.shape[0].")
    if len(quantiles) != 3:
        raise ValueError("quantiles must be a 3-tuple (lo, mid, hi).")

    lo, mid, hi = quantiles
    return pd.DataFrame(
        {
            "isin": np.asarray(isins),
            "er_mean": mc.mean(axis=(1, 2)),
            f"er_p{int(round(lo * 100)):02d}": np.quantile(mc, lo, axis=(1, 2)),
            f"er_p{int(round(mid * 100)):02d}": np.quantile(mc, mid, axis=(1, 2)),
            f"er_p{int(round(hi * 100)):02d}": np.quantile(mc, hi, axis=(1, 2)),
            "prob_pos": (mc > 0.0).mean(axis=(1, 2)),
        }
    )