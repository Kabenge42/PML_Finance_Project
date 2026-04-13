"""
ArviZ-backed diagnostic visualizations for the expected returns pipeline.

Provides convergence diagnostics, posterior comparisons, and hierarchical
shrinkage plots that consume InferenceData or raw DataFrames and produce
Matplotlib figures suitable for PNG export.

Functions
---------
Step 5d:
    build_screening_inference_data, create_screening_posterior_ridge,
    create_productivity_frontier_posterior
Step 5e:
    build_resampled_posterior_idata, create_resampled_posterior_diagnostics,
    create_resampled_sector_forest
Step 6:
    build_alignment_inference_data, create_model_alignment_arviz_panel,
    create_agreement_posterior_by_sector
Step 7:
    create_hierarchical_shrinkage_diagnostic, create_multi_level_mcmc_comparison
Step 7a:
    create_mcmc_convergence_panel_arviz
Step 7b:
    build_category_analytics_idata, create_category_posterior_diagnostics,
    create_cross_category_summary

See also ``probability_viz`` for MCMC-enhanced Plotly dashboards:
    create_mcmc_anomaly_posterior_chart, create_mcmc_credit_risk_chart,
    create_mcmc_dividend_cut_chart, create_mcmc_price_target_chart,
    create_mcmc_category_posterior_chart
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import arviz as az
    import xarray as xr

    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    az = None  # type: ignore[assignment]
    xr = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Step 5d — Stock Screening + Productivity Frontier
# ---------------------------------------------------------------------------


def build_screening_inference_data(
    screens: dict[str, pd.DataFrame],
    return_col: str = "expected_upside_pct",
    n_bootstrap: int = 4_000,
    n_chains: int = 8,
) -> "az.InferenceData":
    """
    Build InferenceData with posterior-like bootstrap distributions
    for each screening strategy's return profile.
    """
    if not ARVIZ_AVAILABLE:
        raise ImportError("arviz is required for build_screening_inference_data")

    posterior_dict: dict[str, tuple] = {}

    for screen_name, screen_df in screens.items():
        if screen_df.empty or return_col not in screen_df.columns:
            continue
        returns = screen_df[return_col].dropna().values
        if len(returns) < 10:
            continue

        rng = np.random.default_rng(42)
        samples = np.array(
            [
                [
                    rng.choice(returns, size=len(returns), replace=True).mean()
                    for _ in range(n_bootstrap)
                ]
                for _ in range(n_chains)
            ]
        )
        posterior_dict[screen_name] = samples

    if not posterior_dict:
        return az.InferenceData()

    data_vars = {name: (["chain", "draw"], samples) for name, samples in posterior_dict.items()}
    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_bootstrap),
    }
    posterior_ds = xr.Dataset(data_vars, coords=coords)
    return az.InferenceData(posterior=posterior_ds)


def create_screening_posterior_ridge(
    screens: dict[str, pd.DataFrame],
    return_col: str = "expected_upside_pct",
):
    """Ridge plot comparing posterior mean returns across screening strategies."""
    if not ARVIZ_AVAILABLE:
        return None

    idata = build_screening_inference_data(screens, return_col=return_col)
    if not hasattr(idata, "posterior"):
        return None

    axes = az.plot_forest(
        idata,
        kind="ridgeplot",
        combined=True,
        ridgeplot_overlap=0.7,
        hdi_prob=0.94,
        figsize=(12, max(6, len(screens) * 0.8)),
    )
    axes[0].set_title("Screening Strategy — Posterior Mean Return Distributions")
    axes[0].set_xlabel("Expected Upside (%)")
    return axes[0].get_figure()


def create_productivity_frontier_posterior(
    df: pd.DataFrame,
    productivity_col: str = "productivity_frontier_score",
    return_col: str = "expected_upside_pct",
    n_quantiles: int = 5,
):
    """
    ArviZ forest plot of expected returns by productivity frontier quintile.
    Tests whether high-productivity firms deliver systematically higher returns.
    """
    if not ARVIZ_AVAILABLE:
        return None
    if productivity_col not in df.columns or return_col not in df.columns:
        return None

    df = df.dropna(subset=[productivity_col, return_col])
    if len(df) < n_quantiles * 10:
        return None

    df["prod_quintile"] = pd.qcut(
        df[productivity_col],
        q=n_quantiles,
        labels=[f"Q{i + 1}" for i in range(n_quantiles)],
    )

    n_chains, n_draws = 8, 4_000
    rng = np.random.default_rng(42)
    posterior_dict: dict[str, tuple] = {}
    for q_label, group in df.groupby("prod_quintile", observed=True):
        returns = group[return_col].values
        if len(returns) < 10:
            continue
        samples = np.array(
            [
                [
                    rng.choice(returns, size=len(returns), replace=True).mean()
                    for _ in range(n_draws)
                ]
                for _ in range(n_chains)
            ]
        )
        posterior_dict[f"Productivity {q_label}"] = (["chain", "draw"], samples)

    if not posterior_dict:
        return None

    ds = xr.Dataset(posterior_dict, coords={"chain": range(n_chains), "draw": range(n_draws)})
    idata = az.InferenceData(posterior=ds)

    axes = az.plot_forest(
        idata,
        kind="ridgeplot",
        combined=True,
        hdi_prob=0.94,
        ridgeplot_overlap=0.5,
        figsize=(10, 6),
    )
    axes[0].set_title("Expected Returns by Productivity Frontier Quintile")
    return axes[0].get_figure()


# ---------------------------------------------------------------------------
# Step 5e — Resampled Bayesian Posterior Returns
# ---------------------------------------------------------------------------


def build_resampled_posterior_idata(
    resampled_df: pd.DataFrame,
    n_chains: int = 8,
    n_draws: int = 4_000,
) -> "az.InferenceData":
    """
    Build InferenceData from resampled posterior returns for ArviZ diagnostics.
    Simulates multi-chain structure for convergence checking.
    """
    if not ARVIZ_AVAILABLE:
        raise ImportError("arviz is required")

    if resampled_df.empty or "posterior_mean" not in resampled_df.columns:
        return az.InferenceData()

    values = resampled_df["posterior_mean"].dropna().values
    chain_len = len(values) // n_chains
    if chain_len < 2:
        return az.InferenceData()

    chains = np.array([values[i * chain_len : (i + 1) * chain_len] for i in range(n_chains)])

    posterior = xr.Dataset(
        {"posterior_return": (["chain", "draw"], chains)},
        coords={"chain": range(n_chains), "draw": range(chain_len)},
    )

    if "posterior_std" in resampled_df.columns:
        std_vals = resampled_df["posterior_std"].dropna().values[: n_chains * chain_len]
        if len(std_vals) >= n_chains * chain_len:
            std_chains = np.array(
                [std_vals[i * chain_len : (i + 1) * chain_len] for i in range(n_chains)]
            )
            posterior["posterior_uncertainty"] = (["chain", "draw"], std_chains)

    return az.InferenceData(posterior=posterior)


def create_resampled_posterior_diagnostics(
    resampled_df: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Generate ArviZ trace, rank, and HDI plots for resampled posteriors."""
    outputs: list[str] = []
    if not ARVIZ_AVAILABLE:
        return outputs

    idata = build_resampled_posterior_idata(resampled_df)
    if not hasattr(idata, "posterior"):
        return outputs

    # 1) Trace plot
    try:
        axes = az.plot_trace(idata, var_names=["posterior_return"], compact=True, figsize=(12, 4))
        fig = axes.ravel()[0].get_figure()
        path = output_dir / "er_resampled_trace.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("Resampled trace plot failed: %s", e)

    # 2) Rank plot
    try:
        axes = az.plot_rank(idata, var_names=["posterior_return"], figsize=(10, 4))
        fig = axes.ravel()[0].get_figure()
        path = output_dir / "er_resampled_rank.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("Resampled rank plot failed: %s", e)

    # 3) Posterior HDI plot
    try:
        axes = az.plot_posterior(
            idata,
            var_names=["posterior_return"],
            hdi_prob=0.94,
            point_estimate="median",
            figsize=(8, 4),
        )
        fig = axes.ravel()[0].get_figure() if hasattr(axes, "ravel") else axes.get_figure()
        path = output_dir / "er_resampled_posterior_hdi.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("Resampled posterior HDI plot failed: %s", e)

    # 4) R-hat summary
    try:
        rhat = az.rhat(idata)
        logger.info("Resampled posterior R̂: %s", rhat)
    except Exception as e:
        logger.debug("R-hat computation failed: %s", e)

    return outputs


def create_resampled_sector_forest(
    resampled_df: pd.DataFrame,
    df_source: pd.DataFrame,
    sector_col: str = "sector",
    top_n: int = 15,
):
    """
    Forest plot of sector-level resampled posterior return distributions
    with 94% HDI intervals — directly addresses the negative mean concern
    by showing which sectors drive the negative aggregate.
    """
    if not ARVIZ_AVAILABLE:
        return None

    if "ticker" not in resampled_df.columns or "ticker" not in df_source.columns:
        return None

    merged = resampled_df.merge(
        df_source[["ticker", sector_col]].drop_duplicates("ticker"),
        on="ticker",
        how="left",
    )
    if merged.empty or "posterior_mean" not in merged.columns:
        return None

    n_chains, n_draws = 4, 4_000
    rng = np.random.default_rng(42)
    posterior_dict: dict[str, tuple] = {}
    for sector, group in merged.groupby(sector_col):
        vals = group["posterior_mean"].dropna().values
        if len(vals) < 20:
            continue
        samples = np.array(
            [
                [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_draws)]
                for _ in range(n_chains)
            ]
        )
        posterior_dict[str(sector)] = (["chain", "draw"], samples)

    if not posterior_dict:
        return None

    ds = xr.Dataset(posterior_dict, coords={"chain": range(n_chains), "draw": range(n_draws)})
    idata = az.InferenceData(posterior=ds)

    axes = az.plot_forest(
        idata,
        kind="forestplot",
        combined=True,
        hdi_prob=0.94,
        figsize=(12, max(6, len(posterior_dict) * 0.5)),
    )
    axes[0].set_title("Sector Resampled Posterior Returns (94% HDI)")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5, label="Zero return")
    return axes[0].get_figure()


# ---------------------------------------------------------------------------
# Step 6 — Tri-Model & Quad-Model Alignment
# ---------------------------------------------------------------------------


def build_alignment_inference_data(
    summary: pd.DataFrame,
) -> "az.InferenceData":
    """
    Build InferenceData comparing the four model return distributions
    for cross-model posterior comparison plots.
    """
    if not ARVIZ_AVAILABLE:
        raise ImportError("arviz is required")

    model_cols = {
        "Monte Carlo": "expected_upside_pct",
        "Kalman Filter": "filtered_upside",
        "Price Target": "expected_return_prob_weighted",
    }

    n_chains, n_draws = 4, 5_000
    rng = np.random.default_rng(42)
    posterior_dict: dict[str, tuple] = {}
    for model_name, col in model_cols.items():
        if col not in summary.columns:
            continue
        vals = summary[col].dropna().values
        if len(vals) < 50:
            continue
        samples = np.array(
            [
                [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_draws)]
                for _ in range(n_chains)
            ]
        )
        posterior_dict[model_name] = (["chain", "draw"], samples)

    if not posterior_dict:
        return az.InferenceData()

    ds = xr.Dataset(posterior_dict, coords={"chain": range(n_chains), "draw": range(n_draws)})
    return az.InferenceData(posterior=ds)


def create_model_alignment_arviz_panel(
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """
    Generate ArviZ panel: forest + density + pair_plot for cross-model alignment.
    Directly surfaces the MC vs Kalman level discrepancy.
    """
    outputs: list[str] = []
    if not ARVIZ_AVAILABLE:
        return outputs

    idata = build_alignment_inference_data(summary)
    if not hasattr(idata, "posterior"):
        return outputs

    # 1) Forest plot
    try:
        axes = az.plot_forest(
            idata,
            kind="forestplot",
            combined=True,
            hdi_prob=0.94,
            figsize=(10, 5),
        )
        axes[0].set_title("Cross-Model Expected Return Posteriors (94% HDI)")
        path = output_dir / "er_model_alignment_forest.png"
        axes[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("Model alignment forest plot failed: %s", e)

    # 2) Density overlay
    try:
        axes = az.plot_density(
            [idata],
            var_names=list(idata.posterior.data_vars),
            hdi_prob=0.94,
            figsize=(10, 5),
            shade=0.3,
        )
        path = output_dir / "er_model_alignment_density.png"
        axes[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("Model alignment density plot failed: %s", e)

    # 3) Pair plot
    try:
        ax = az.plot_pair(
            idata,
            kind="kde",
            marginals=True,
            figsize=(10, 10),
        )
        path = output_dir / "er_model_alignment_pair.png"
        fig = ax.ravel()[0].get_figure() if hasattr(ax, "ravel") else ax[0][0].get_figure()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("Model alignment pair plot failed: %s", e)

    return outputs


def create_agreement_posterior_by_sector(
    summary: pd.DataFrame,
    sector_col: str = "sector",
):
    """
    ArviZ-style forest plot of agreement_score posterior by sector.
    Reveals which sectors have genuine consensus vs noisy alignment.
    """
    if not ARVIZ_AVAILABLE:
        return None
    if "agreement_score" not in summary.columns or sector_col not in summary.columns:
        return None

    n_chains, n_draws = 4, 4_000
    rng = np.random.default_rng(42)
    posterior_dict: dict[str, tuple] = {}

    for sector, group in summary.groupby(sector_col):
        scores = group["agreement_score"].dropna().values.astype(float)
        if len(scores) < 20:
            continue
        samples = np.array(
            [
                [rng.choice(scores, size=len(scores), replace=True).mean() for _ in range(n_draws)]
                for _ in range(n_chains)
            ]
        )
        posterior_dict[str(sector)] = (["chain", "draw"], samples)

    if not posterior_dict:
        return None

    ds = xr.Dataset(posterior_dict, coords={"chain": range(n_chains), "draw": range(n_draws)})
    idata = az.InferenceData(posterior=ds)

    axes = az.plot_forest(
        idata,
        kind="forestplot",
        combined=True,
        hdi_prob=0.94,
        figsize=(12, max(8, len(posterior_dict) * 0.5)),
    )
    axes[0].set_title("Model Agreement Score by Sector (94% HDI)")
    axes[0].axvline(3.0, color="green", linestyle="--", alpha=0.5, label="Strong Bullish threshold")
    return axes[0].get_figure()


# ---------------------------------------------------------------------------
# Step 7 — Expected Returns Summary + Hierarchical Sector MCMC
# ---------------------------------------------------------------------------


def create_hierarchical_shrinkage_diagnostic(
    summary: pd.DataFrame,
    return_col: str = "expected_upside_pct",
    sector_col: str = "industry",
):
    """
    ArviZ plot showing raw sector means vs hierarchical posterior means
    with shrinkage arrows and HDI intervals.
    """
    if not ARVIZ_AVAILABLE:
        return None

    from analytics.statistical_analysis import hierarchical_mcmc_by_sector

    hier = hierarchical_mcmc_by_sector(summary, return_col, sector_col=sector_col)
    if not isinstance(hier, dict):
        return None

    sectors_data = hier.get("sectors", hier)
    n_chains, n_draws = 4, 5_000
    rng = np.random.default_rng(42)
    posterior_dict: dict[str, tuple] = {}
    raw_means: dict[str, float] = {}

    for sector, info in sectors_data.items():
        if not isinstance(info, dict) or "posterior_mean" not in info:
            continue
        post_mean = info["posterior_mean"]
        post_std = info.get("posterior_std", info.get("raw_std", 5.0) * 0.5)
        raw_means[sector] = info.get("raw_mean", post_mean)

        samples = rng.normal(
            post_mean,
            max(post_std, 0.01),
            size=(n_chains, n_draws),
        )
        posterior_dict[sector] = (["chain", "draw"], samples)

    if not posterior_dict:
        return None

    ds = xr.Dataset(posterior_dict, coords={"chain": range(n_chains), "draw": range(n_draws)})
    idata = az.InferenceData(posterior=ds)

    axes = az.plot_forest(
        idata,
        kind="forestplot",
        combined=True,
        hdi_prob=0.94,
        figsize=(14, max(8, len(posterior_dict) * 0.45)),
    )
    ax = axes[0]
    ax.set_title(f"Hierarchical Sector MCMC — Posterior vs Raw Means ({sector_col})")

    # Overlay raw means as red triangles
    for i, sector in enumerate(posterior_dict.keys()):
        if sector in raw_means:
            ax.plot(raw_means[sector], i, "r^", markersize=6, alpha=0.7)

    if return_col in summary.columns:
        ax.axvline(
            summary[return_col].mean(),
            color="gray",
            linestyle=":",
            alpha=0.5,
            label="Grand mean",
        )
    return ax.get_figure()


def create_multi_level_mcmc_comparison(
    summary: pd.DataFrame,
    return_col: str = "expected_upside_pct",
    levels: Optional[list[str]] = None,
):
    """
    ArviZ density plot comparing posterior distributions across
    hierarchical category levels (region, sector, size_class, style_class).
    """
    if not ARVIZ_AVAILABLE:
        return None
    import matplotlib.pyplot as plt

    levels = levels or ["region", "sector", "size_class", "style_class"]
    n_chains, n_draws = 4, 5_000
    rng = np.random.default_rng(42)
    level_idatas = []
    level_labels = []

    for level_col in levels:
        if level_col not in summary.columns:
            continue
        posterior_dict: dict[str, tuple] = {}
        for group_name, group_df in summary.groupby(level_col):
            vals = group_df[return_col].dropna().values
            if len(vals) < 20:
                continue
            samples = np.array(
                [
                    [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_draws)]
                    for _ in range(n_chains)
                ]
            )
            posterior_dict[str(group_name)] = (["chain", "draw"], samples)

        if posterior_dict:
            ds = xr.Dataset(
                posterior_dict,
                coords={"chain": range(n_chains), "draw": range(n_draws)},
            )
            level_idatas.append(az.InferenceData(posterior=ds))
            level_labels.append(level_col)

    if not level_idatas:
        return None

    fig, axes_arr = plt.subplots(
        len(level_idatas),
        1,
        figsize=(14, 5 * len(level_idatas)),
        squeeze=False,
    )
    for i, (idata, label) in enumerate(zip(level_idatas, level_labels)):
        az.plot_forest(
            idata,
            kind="ridgeplot",
            combined=True,
            hdi_prob=0.94,
            ridgeplot_overlap=0.6,
            ax=axes_arr[i, 0],
        )
        axes_arr[i, 0].set_title(f"Posterior Returns by {label.replace('_', ' ').title()}")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Step 7a — Parallel MCMC with Gelman-Rubin Convergence
# ---------------------------------------------------------------------------


def create_mcmc_convergence_panel_arviz(
    mcmc_result: dict,
    output_dir: Path,
) -> list[str]:
    """
    Comprehensive ArviZ convergence diagnostics for parallel MCMC chains.
    Addresses the R̂=1.0000 result by providing visual verification.
    """
    outputs: list[str] = []
    if not ARVIZ_AVAILABLE:
        return outputs

    chains = mcmc_result.get("chain_samples")
    if chains is None or len(chains) < 2:
        return outputs

    n_chains = len(chains)
    min_len = min(len(c) for c in chains)
    if min_len < 2:
        return outputs
    chain_array = np.array([c[:min_len] for c in chains])

    posterior = xr.Dataset(
        {"expected_return_prob_weighted": (["chain", "draw"], chain_array)},
        coords={"chain": range(n_chains), "draw": range(min_len)},
    )
    idata = az.InferenceData(posterior=posterior)

    # 1) Trace plot
    try:
        axes = az.plot_trace(
            idata,
            var_names=["expected_return_prob_weighted"],
            kind="trace",
            compact=False,
            figsize=(14, 4),
        )
        path = output_dir / "er_mcmc_trace.png"
        axes.ravel()[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("MCMC trace plot failed: %s", e)

    # 2) Rank plot
    try:
        axes = az.plot_rank(
            idata,
            var_names=["expected_return_prob_weighted"],
            kind="bars",
            figsize=(12, 4),
        )
        path = output_dir / "er_mcmc_rank_bars.png"
        axes.ravel()[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("MCMC rank plot failed: %s", e)

    # 3) Autocorrelation plot
    try:
        axes = az.plot_autocorr(
            idata,
            var_names=["expected_return_prob_weighted"],
            max_lag=100,
            figsize=(12, 4),
        )
        path = output_dir / "er_mcmc_autocorr.png"
        axes.ravel()[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("MCMC autocorr plot failed: %s", e)

    # 4) ESS evolution plot
    try:
        axes = az.plot_ess(
            idata,
            var_names=["expected_return_prob_weighted"],
            kind="evolution",
            figsize=(10, 4),
        )
        path = output_dir / "er_mcmc_ess_evolution.png"
        axes.ravel()[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("MCMC ESS plot failed: %s", e)

    # 5) MCSE plot
    try:
        axes = az.plot_mcse(
            idata,
            var_names=["expected_return_prob_weighted"],
            figsize=(10, 4),
        )
        path = output_dir / "er_mcmc_mcse.png"
        axes.ravel()[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("MCMC MCSE plot failed: %s", e)

    # 6) Posterior with reference values
    try:
        axes = az.plot_posterior(
            idata,
            var_names=["expected_return_prob_weighted"],
            hdi_prob=0.94,
            point_estimate="median",
            ref_val=0,
            figsize=(8, 4),
        )
        path = output_dir / "er_mcmc_posterior.png"
        fig = axes.ravel()[0].get_figure() if hasattr(axes, "ravel") else axes.get_figure()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(str(path))
    except Exception as e:
        logger.debug("MCMC posterior plot failed: %s", e)

    # 7) Summary statistics
    try:
        summary_df = az.summary(idata, var_names=["expected_return_prob_weighted"], hdi_prob=0.94)
        logger.info("MCMC Summary:\n%s", summary_df.to_string())
    except Exception as e:
        logger.debug("MCMC summary failed: %s", e)

    return outputs


# ---------------------------------------------------------------------------
# Step 7b — Per-Category Bayesian Probability Analytics
# ---------------------------------------------------------------------------


def build_category_analytics_idata(
    category_analytics: dict[str, dict],
    df: pd.DataFrame,
) -> dict[str, "az.InferenceData"]:
    """
    Build per-category InferenceData from category probability analytics results.
    Each category gets its own InferenceData with feature-level posteriors.
    """
    if not ARVIZ_AVAILABLE:
        return {}

    category_idatas: dict[str, az.InferenceData] = {}
    n_chains, n_draws = 4, 4_000
    rng = np.random.default_rng(42)

    for cat_name, cat_result in category_analytics.items():
        bayesian = cat_result.get("bayesian_results", {})
        if not bayesian:
            continue

        posterior_dict: dict[str, tuple] = {}
        for feature_name, feat_info in bayesian.items():
            if not isinstance(feat_info, dict):
                continue
            post_mean = feat_info.get("posterior_mean")
            post_std = feat_info.get("posterior_std")
            if post_mean is None or post_std is None:
                continue

            samples = rng.normal(
                post_mean,
                max(post_std, 1e-6),
                size=(n_chains, n_draws),
            )
            short_name = feature_name[:30] if len(feature_name) > 30 else feature_name
            posterior_dict[short_name] = (["chain", "draw"], samples)

        if posterior_dict:
            ds = xr.Dataset(
                posterior_dict,
                coords={"chain": range(n_chains), "draw": range(n_draws)},
            )
            category_idatas[cat_name] = az.InferenceData(posterior=ds)

    return category_idatas


def create_category_posterior_diagnostics(
    category_analytics: dict[str, dict],
    df: pd.DataFrame,
    output_dir: Path,
    max_features_per_plot: int = 20,
) -> list[str]:
    """
    Generate ArviZ diagnostic plots for each category's Bayesian posteriors.
    """
    outputs: list[str] = []
    if not ARVIZ_AVAILABLE:
        return outputs

    category_idatas = build_category_analytics_idata(category_analytics, df)

    for cat_name, idata in category_idatas.items():
        safe_name = cat_name.lower().replace(" ", "_").replace("&", "and")
        var_names = list(idata.posterior.data_vars)[:max_features_per_plot]

        # 1) Ridge plot
        try:
            axes = az.plot_forest(
                idata,
                var_names=var_names,
                kind="ridgeplot",
                combined=True,
                hdi_prob=0.94,
                ridgeplot_overlap=0.5,
                figsize=(14, max(6, len(var_names) * 0.5)),
            )
            axes[0].set_title(f"{cat_name} — Feature Posterior Distributions")
            path = output_dir / f"er_category_{safe_name}_ridge.png"
            axes[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
            outputs.append(str(path))
        except Exception as e:
            logger.debug("Ridge plot failed for %s: %s", cat_name, e)

        # 2) Forest plot with HDI
        try:
            axes = az.plot_forest(
                idata,
                var_names=var_names,
                kind="forestplot",
                combined=True,
                hdi_prob=0.94,
                figsize=(12, max(6, len(var_names) * 0.4)),
            )
            axes[0].set_title(f"{cat_name} — Feature Posterior Forest (94% HDI)")
            path = output_dir / f"er_category_{safe_name}_forest.png"
            axes[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
            outputs.append(str(path))
        except Exception as e:
            logger.debug("Forest plot failed for %s: %s", cat_name, e)

        # 3) ESS summary
        try:
            ess = az.ess(idata)
            low_ess = {
                var: float(ess[var].values) for var in ess.data_vars if float(ess[var].values) < 400
            }
            if low_ess:
                logger.warning("%s: low ESS features (< 400): %s", cat_name, low_ess)
        except Exception:
            pass

    return outputs


def create_cross_category_summary(
    category_analytics: dict[str, dict],
    output_dir: Path,
) -> Optional[str]:
    """
    Single ArviZ plot comparing aggregate posterior means across all categories.
    Provides a birds-eye view of which feature domains drive return predictions.
    """
    if not ARVIZ_AVAILABLE:
        return None

    n_chains, n_draws = 4, 5_000
    rng = np.random.default_rng(42)
    posterior_dict: dict[str, tuple] = {}

    for cat_name, cat_result in category_analytics.items():
        bayesian = cat_result.get("bayesian_results", {})
        if not bayesian:
            continue

        post_means = [
            v["posterior_mean"]
            for v in bayesian.values()
            if isinstance(v, dict) and "posterior_mean" in v
        ]
        post_stds = [
            v["posterior_std"]
            for v in bayesian.values()
            if isinstance(v, dict) and "posterior_std" in v
        ]
        if not post_means:
            continue

        agg_mean = np.mean(post_means)
        agg_std = np.mean(post_stds) / np.sqrt(len(post_means))
        samples = rng.normal(agg_mean, max(agg_std, 1e-6), size=(n_chains, n_draws))
        posterior_dict[cat_name] = (["chain", "draw"], samples)

    if not posterior_dict:
        return None

    ds = xr.Dataset(posterior_dict, coords={"chain": range(n_chains), "draw": range(n_draws)})
    idata = az.InferenceData(posterior=ds)

    axes = az.plot_forest(
        idata,
        kind="ridgeplot",
        combined=True,
        hdi_prob=0.94,
        ridgeplot_overlap=0.6,
        figsize=(14, max(8, len(posterior_dict) * 0.6)),
    )
    axes[0].set_title("Cross-Category Bayesian Posterior Summary (Feature Domains)")
    path = output_dir / "er_cross_category_posterior.png"
    axes[0].get_figure().savefig(path, dpi=150, bbox_inches="tight")
    return str(path)
