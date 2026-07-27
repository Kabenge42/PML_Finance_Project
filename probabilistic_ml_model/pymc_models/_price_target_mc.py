"""Helpers for the PyMC price-target notebook (``pymc_price_target.ipynb``).

Contains pure-NumPy/pandas building blocks that are independently testable
without requiring PyMC at import time:

* :func:`prepare_price_target_inputs` — derives ``target_vs_price_pct``,
  ``feat_analyst_conviction`` and ``feat_target_dispersion_cv`` from the raw
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
    "PriceTargetPanelInputs",
    "prepare_price_target_inputs",
    "prepare_price_target_panel_inputs",
    "simulate_lagged_risk_adjusted_returns",
    "summarize_mc_returns",
    "PANEL_FISCAL_ANCHOR_COLS",
    "PANEL_DAY_COUNT_COLS",
    "PANEL_RESPONSE_COLS",
    "PANEL_CATEGORICAL_COORDS",
    "ANALYST_SENTIMENT_PCT_COLS",
]


# Normalised analyst-sentiment share columns added to ``pml.mv_pymc_price_target``
# by the pml-schema update. Each is a 0-100 percentage over all rating buckets:
#   feat_analyst_bullish_pct  — (strong buys + buys) share of all opinions
#   feat_analyst_bearish_pct  — (sells + strong sells) share of all opinions
#   feat_analyst_neutral_pct  — hold share of all opinions
# ``feat_analyst_conviction`` (|bullish% - bearish%|) is handled separately as a
# dedicated latent and rescaled to [0, 1] on read (see prepare_price_target_inputs).
# These three are coverage-invariant directional predictors; neutral_pct is the
# residual of the other two (+ no-opinion), so callers wiring a regression design
# matrix typically include bullish/bearish and drop neutral to avoid collinearity.
ANALYST_SENTIMENT_PCT_COLS: tuple[str, ...] = (
    "feat_analyst_bullish_pct",
    "feat_analyst_bearish_pct",
    "feat_analyst_neutral_pct",
)


@dataclass(frozen=True)
class PriceTargetInputs:
    """Container of arrays consumed by the price-target PyMC model.

    Attributes
    ----------
    frame : pandas.DataFrame
        Filtered / augmented input frame (with derived ``target_vs_price_pct``,
        ``feat_analyst_conviction`` and ``feat_target_dispersion_cv``).
    isins, sectors : np.ndarray
        Per-row identifier / sector arrays.
    target_vs_price_pct : np.ndarray
        Observed target — ``(price_target_median - close) / close``.
    conviction_ratio : np.ndarray
        Analyst conviction on a [0, 1] fraction scale (drives the
        heteroscedastic sigma prior and the ``exp(-risk_penalty * conviction)``
        risk adjustment). Sourced from the MV-native ``feat_analyst_conviction``
        percentage rescaled by 1/100, or the legacy buys/n_analysts fallback.
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
_TARGET_AVG_ALIASES = ("price_target", "price_target_median", "price_target_avg")
_TARGET_NUM_ALIASES = ("n_analysts", "price_target_num")
_BUYS_COL_ALIASES = ("feat_net_buy_sentiment", "num_buys_ratings", "num_strong_buys_ratings", "feat_analyst_bullish_pct")


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
        buys_col: Optional[str] = None,
) -> PriceTargetInputs:
    """Schema-aligned data preparation for the price-target notebook.

    Drops rows missing the ``isin`` or ``sector`` identifiers or with a
    non-computable ``target_vs_price_pct`` (e.g. zero / NaN price).

    Column resolution is alias-aware so the helper works directly against the
    ``pml.mv_pymc_price_target`` materialised view (which exposes
    ``last_price``, ``price_target`` and ``n_analysts``) as well as legacy
    frames that use ``close`` / ``price_target_median`` / ``price_target_num``.
    Pre-computed ``feat_analyst_conviction`` and ``feat_target_dispersion_cv``
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
    buys_col = buys_col or _resolve_alias(price_target_df, _BUYS_COL_ALIASES)

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

    # Conviction proxy — prefer MV-precomputed `feat_analyst_conviction` if present,
    # otherwise derive as feat_net_buy_sentiment / max(n_analysts, 1).
    #
    # NOTE (pml-schema update): MV-native ``feat_analyst_conviction`` is now a
    # 0-100 PERCENTAGE — ``ABS(bullish% - bearish%)`` over every rating bucket
    # (see pml.mv_pymc_price_target). The legacy fallback below is a ~[0, 1]
    # fraction, and every downstream consumer (the heteroscedastic ``sigma``
    # prior and the ``exp(-risk_penalty * conviction)`` risk adjustment) was
    # tuned for that [0, 1] scale — a raw 0-100 value would collapse
    # ``risk_adj_return`` to ~0 for high-conviction names. Rescale by 1/100 so
    # both paths (and the model priors) stay on the same footing.
    if "feat_analyst_conviction" in df.columns:
        df["feat_analyst_conviction"] = (
            df["feat_analyst_conviction"].astype("float64").div(100.0).fillna(0.0)
        )
    else:
        df["feat_analyst_conviction"] = (
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
        conviction_ratio=df["feat_analyst_conviction"].to_numpy(dtype="float64"),
        dispersion_cv=df["feat_target_dispersion_cv"].to_numpy(dtype="float64"),
        n_analysts=df[target_num_col].astype("float64").clip(lower=1).to_numpy(),
    )


# ---------------------------------------------------------------------------
# Panel (3-D MvGRW) data preparation — for the fused price-target model.
# ---------------------------------------------------------------------------

# Ordered fiscal-calendar date anchors emitted by ``pml.mv_pymc_price_target``.
# These define the MvGaussianRandomWalk time axis (T=6 steps). t0 == anchor[0].
PANEL_FISCAL_ANCHOR_COLS: tuple[str, ...] = (
    "income_statement_report_date",
    "next_earnings",
    "fy_end_date",
    "next_income_statement_report_date",
    "next_fy_end_date",
    "expected_report_date",
)

# Pre-computed numeric day-count horizons (NO ``feat_`` prefix — these mirror
# the ``pml.mv_pymc_price_target`` DDL aliases exactly; the notebook's old
# ``feat_days_*`` names were wrong and never matched the materialised view).
PANEL_DAY_COUNT_COLS: tuple[str, ...] = (
    "days_to_next_earnings",
    "days_since_last_report",
    "days_to_next_fy_end",
    "days_to_next_fiscal_quarter",
    "days_to_next_report",
    "days_to_expected_report",
    "days_since_fy_end",
)

# D-dimensional joint response series (D=8) broadcast across the T anchors.
PANEL_RESPONSE_COLS: tuple[str, ...] = (
    "feat_implied_upside",
    "feat_target_dispersion_cv",
    "observed_target_pct",
    "observed_target_pct_med",
    "price_target",
    "price_target_high",
    "price_target_low",
    "price_target_median",
)

# Categorical coords emitted by the MV (each becomes a model dim + ``_idx``).
PANEL_CATEGORICAL_COORDS: tuple[str, ...] = (
    "region",
    "country",
    "trading_country",
    "exchange",
    "unit",
    "style_class",
    "size_class",
    "sector",
    "industry",
)


@dataclass(frozen=True)
class PriceTargetPanelInputs:
    """Container of arrays consumed by the fused (3-D MvGRW) price-target model.

    Extends the 1-D :class:`PriceTargetInputs` payload with the panel tensors
    required by the Multivariate Gaussian Random Walk spine: the standardised
    ``(isin, time, y_series)`` response tensor, the ``(isin, time)`` fiscal-anchor
    time matrix, the NumPy-precomputed ``sqrt(n_analysts)`` (so the PyMC graph
    never applies an inplace ``Sqrt`` that NUTS rejects), and the per-coord
    unique-label vectors / integer index arrays.

    Attributes
    ----------
    frame : pandas.DataFrame
        Filtered modelling frame (response present & ``n_analysts > 0``).
    isins : np.ndarray
        Per-row ISIN identifiers (``len == n_isin``).
    Y : np.ndarray
        Standardised response tensor, shape ``(n_isin, T, D)``.
    t_scaled : np.ndarray
        Standardised fiscal-anchor time matrix, shape ``(n_isin, T)``.
    X_std : np.ndarray
        Standardised predictor matrix, shape ``(n_isin, n_pt_feature)``.
    n_analysts, sqrt_n_analysts : np.ndarray
        Floored analyst count and its NumPy-precomputed square root.
    target_vs_price_pct, conviction_ratio, dispersion_cv : np.ndarray
        Model-A latents reused for the conviction-aware baseline / σ.
    predictor_names, response_names : list[str]
        Column labels backing the ``pt_feature`` / ``y_series`` dims.
    coord_uniques : dict[str, np.ndarray]
        Per-coord unique label vectors.
    coord_idx : dict[str, np.ndarray]
        Per-coord ``int64`` index arrays (``len == n_isin``).
    """

    frame: pd.DataFrame
    isins: np.ndarray
    Y: np.ndarray
    t_scaled: np.ndarray
    X_std: np.ndarray
    n_analysts: np.ndarray
    sqrt_n_analysts: np.ndarray
    target_vs_price_pct: np.ndarray
    conviction_ratio: np.ndarray
    dispersion_cv: np.ndarray
    predictor_names: list[str]
    response_names: list[str]
    coord_uniques: dict[str, np.ndarray]
    coord_idx: dict[str, np.ndarray]


def prepare_price_target_panel_inputs(
        price_target_df: pd.DataFrame,
        *,
        fiscal_anchor_cols: tuple[str, ...] = PANEL_FISCAL_ANCHOR_COLS,
        response_cols: tuple[str, ...] = PANEL_RESPONSE_COLS,
        categorical_coords: tuple[str, ...] = PANEL_CATEGORICAL_COORDS,
        predictor_cols: Optional[list[str]] = None,
        max_features: int = 8,
        min_coverage: float = 0.70,
) -> PriceTargetPanelInputs:
    """Assemble the 3-D MvGRW panel tensors for the fused price-target model.

    Pure-NumPy/pandas (no PyMC import) so it is unit-testable in isolation.
    Rows missing the ``observed_target_pct`` response or with
    ``n_analysts <= 0`` are dropped. The Model-A latents
    (``target_vs_price_pct``, ``feat_analyst_conviction``,
    ``feat_target_dispersion_cv``) are derived via
    :func:`prepare_price_target_inputs` so the 1-D and 3-D paths stay aligned.

    Parameters
    ----------
    price_target_df : pandas.DataFrame
        Frame backed by ``pml.mv_pymc_price_target``.
    fiscal_anchor_cols : tuple[str, ...]
        Ordered date columns defining the T random-walk anchors.
    response_cols : tuple[str, ...]
        D response series columns (broadcast across the T anchors).
    categorical_coords : tuple[str, ...]
        Categorical coord columns present in the MV. Missing columns are
        skipped (empty-coord guard).
    predictor_cols : list[str], optional
        Explicit predictor columns. When ``None`` numeric ``feat_*`` / day-count
        columns with ``>= min_coverage`` non-null coverage are auto-selected
        (capped at ``max_features``).
    max_features : int
        Cap on auto-selected predictors.
    min_coverage : float
        Minimum non-null fraction for auto-selected predictors.

    Returns
    -------
    PriceTargetPanelInputs
    """
    if "observed_target_pct" not in price_target_df.columns:
        raise KeyError(
            "price_target_df is missing required column 'observed_target_pct'."
        )
    if "n_analysts" not in price_target_df.columns:
        raise KeyError("price_target_df is missing required column 'n_analysts'.")

    # Reuse the 1-D preparation for the Model-A latents + row filtering.
    base = prepare_price_target_inputs(price_target_df)
    df = base.frame.copy()

    # Keep only rows with a usable response and a positive analyst count.
    mask = df["observed_target_pct"].notna() & df["n_analysts"].fillna(0).gt(0) \
        if "observed_target_pct" in df.columns else None
    if mask is not None:
        df = df.loc[mask].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No modelling rows after filtering response / n_analysts.")

    isin_labels = df["isin"].astype(str).to_numpy()
    n_isin = len(df)

    # --- Fiscal-anchor time matrix (n_isin, T) --------------------------------
    anchors = [c for c in fiscal_anchor_cols if c in df.columns]
    if not anchors:
        raise KeyError(
            f"None of the fiscal-anchor columns {list(fiscal_anchor_cols)} present."
        )
    t0 = pd.to_datetime(df[anchors[0]], errors="coerce")
    t_days = np.stack(
        [
            (pd.to_datetime(df[c], errors="coerce") - t0).dt.days.to_numpy(dtype="float64")
            for c in anchors
        ],
        axis=1,
    )
    t_mean = np.nanmean(t_days) if np.isfinite(t_days).any() else 0.0
    t_std = np.nanstd(t_days)
    t_std = t_std if t_std and np.isfinite(t_std) else 1.0
    t_scaled = np.nan_to_num((t_days - t_mean) / t_std, nan=0.0)

    # --- Response tensor (n_isin, T, D) standardised --------------------------
    resp = [c for c in response_cols if c in df.columns]
    if not resp:
        raise KeyError(
            f"None of the response columns {list(response_cols)} present."
        )
    T = t_scaled.shape[1]
    D = len(resp)
    Y = np.stack(
        [np.tile(df[c].astype("float64").to_numpy()[:, None], (1, T)) for c in resp],
        axis=-1,
    )
    y_mean = Y.reshape(-1, D).mean(axis=0)
    y_std = Y.reshape(-1, D).std(axis=0)
    y_std = np.where(y_std > 1e-6, y_std, 1.0)
    Y_std = np.nan_to_num((Y - y_mean) / y_std, nan=0.0)

    # --- Categorical coords (empty-coord guard) -------------------------------
    coord_uniques: dict[str, np.ndarray] = {}
    coord_idx: dict[str, np.ndarray] = {}
    for col in categorical_coords:
        if col not in df.columns:
            continue
        labels = df[col].fillna("Unknown").astype(str).to_numpy()
        uniques, idx = np.unique(labels, return_inverse=True)
        coord_uniques[col] = uniques
        coord_idx[col] = idx.astype("int64")

    # --- Predictor matrix (n_isin, n_pt_feature) standardised -----------------
    if predictor_cols is None:
        candidates = [
            c
            for c in df.columns
            if (c.startswith("feat_") or c in PANEL_DAY_COUNT_COLS)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if candidates:
            coverage = df[candidates].notna().mean()
            predictors = (
                coverage[coverage >= min_coverage]
                .sort_values(ascending=False)
                .index.tolist()[:max_features]
            )
        else:
            predictors = []
        if not predictors:
            # Minimal fallback design matrix.
            predictors = [
                c
                for c in ("feat_target_dispersion_cv", "feat_analyst_conviction")
                if c in df.columns
            ]
    else:
        predictors = [c for c in predictor_cols if c in df.columns]

    if predictors:
        x_raw = df[predictors].astype("float64")
        x_std_df = (x_raw - x_raw.mean()) / x_raw.std(ddof=0).replace(0, 1.0)
        X_std = x_std_df.fillna(0.0).to_numpy()
    else:
        X_std = np.zeros((n_isin, 0), dtype="float64")

    n_analysts = df["n_analysts"].astype("float64").clip(lower=1).to_numpy()
    sqrt_n_analysts = np.sqrt(np.maximum(n_analysts, 1.0))

    return PriceTargetPanelInputs(
        frame=df,
        isins=isin_labels,
        Y=Y_std,
        t_scaled=t_scaled,
        X_std=X_std,
        n_analysts=n_analysts,
        sqrt_n_analysts=sqrt_n_analysts,
        target_vs_price_pct=df["target_vs_price_pct"].to_numpy(dtype="float64"),
        conviction_ratio=df["feat_analyst_conviction"].to_numpy(dtype="float64"),
        dispersion_cv=df["feat_target_dispersion_cv"].to_numpy(dtype="float64"),
        predictor_names=list(predictors),
        response_names=list(resp),
        coord_uniques=coord_uniques,
        coord_idx=coord_idx,
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
        Columns: ``isin, er_mean, er_sd, er_p05, er_p50, er_p95, prob_pos``.
        ``er_sd`` is the pooled standard deviation over the ``(sample, horizon)``
        draws — the forward-return volatility used as the denominator of the
        exported ``expected_sharpe_ratio``.
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
            "er_sd": mc.std(axis=(1, 2)),
            f"er_p{int(round(lo * 100)):02d}": np.quantile(mc, lo, axis=(1, 2)),
            f"er_p{int(round(mid * 100)):02d}": np.quantile(mc, mid, axis=(1, 2)),
            f"er_p{int(round(hi * 100)):02d}": np.quantile(mc, hi, axis=(1, 2)),
            "prob_pos": (mc > 0.0).mean(axis=(1, 2)),
        }
    )