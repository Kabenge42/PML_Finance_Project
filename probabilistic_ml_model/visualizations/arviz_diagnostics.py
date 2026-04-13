"""
ArviZ-backed diagnostic visualizations for the expected returns pipeline.

Provides convergence diagnostics, posterior comparisons, and hierarchical
shrinkage plots that consume DataTree (ArviZ 1.0) or raw DataFrames and
produce Matplotlib figures suitable for PNG export.

**ArviZ 1.0 migration notes** (see https://python.arviz.org/en/latest/user_guide/migration_guide.html):
- ``az.InferenceData`` replaced by ``xr.DataTree`` built via ``az.from_dict``.
- Plot functions return ``PlotCollection``; extract the figure with ``_fig_from_pc``.
- ``plot_forest(kind="ridgeplot")`` replaced by ``az.plot_ridge``.
- ``plot_posterior`` replaced by ``az.plot_dist``.
- ``plot_density`` removed; use ``az.plot_dist`` or ``az.plot_ridge``.

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probabilistic_ml_model.visualizations._shared import (
    apply_arviz_theme,
    _make_datatree,
    _fig_from_pc,
    _pc_add_title,
)

logger = logging.getLogger(__name__)

try:
    import arviz_plots as azp
    import arviz_stats as azs
    import arviz_base as azb
    import xarray as xr

    ARVIZ_AVAILABLE = True
except ImportError:
    try:
        import arviz as az
        import xarray as xr

        azp = az  # type: ignore[assignment]  # fallback
        azs = az  # type: ignore[assignment]
        azb = None  # type: ignore[assignment]
        ARVIZ_AVAILABLE = True
    except ImportError:
        ARVIZ_AVAILABLE = False
        azp = None  # type: ignore[assignment]
        azs = None  # type: ignore[assignment]
        azb = None  # type: ignore[assignment]
        xr = None  # type: ignore[assignment]

# Apply global dark theme for all ArviZ / Matplotlib figures in this module
apply_arviz_theme()


def _build_datatree(posterior_ds: "xr.Dataset") -> "xr.DataTree":
    """Build a DataTree with a ``posterior`` group from an xr.Dataset."""
    return _make_datatree(posterior=posterior_ds)


def _empty_datatree() -> "xr.DataTree":
    """Return an empty DataTree (replaces ``az.InferenceData()``)."""
    return _make_datatree()


def _has_posterior(dt) -> bool:
    """Check whether a DataTree has a ``posterior`` child group."""
    try:
        return "posterior" in dt.children
    except Exception:
        return False


def _save_fig(fig, path: Path, dpi: int = 150) -> str:
    """Save a matplotlib figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Step 5d — Stock Screening + Productivity Frontier
# ---------------------------------------------------------------------------


def build_screening_inference_data(
    screens: dict[str, pd.DataFrame],
    return_col: str = "implied_return_pt",
    n_bootstrap: int = 10_000,
    n_chains: int = 8,
) -> "xr.DataTree":
    """
    Build DataTree with posterior-like bootstrap distributions
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
        return _empty_datatree()

    data_vars = {name: (["chain", "draw"], samples) for name, samples in posterior_dict.items()}
    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_bootstrap),
    }
    posterior_ds = xr.Dataset(data_vars, coords=coords)
    return _build_datatree(posterior_ds)


def create_screening_posterior_ridge(
    screens: dict[str, pd.DataFrame],
    return_col: str = "implied_return_pt",
):
    """Ridge plot comparing posterior mean returns across screening strategies."""
    if not ARVIZ_AVAILABLE:
        return None

    dt = build_screening_inference_data(screens, return_col=return_col)
    if not _has_posterior(dt):
        return None

    pc = azp.plot_ridge(
        dt,
        combined=False,
        backend="matplotlib",
    )
    _pc_add_title(pc, "Screening Strategy — Posterior Mean Return Distributions")
    return _fig_from_pc(pc)


def create_productivity_frontier_posterior(
    df: pd.DataFrame,
    productivity_col: str = "productivity_frontier_score",
    return_col: str = "implied_return_pt",
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

    n_chains, n_draws = 8, 10_000
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
    dt = _build_datatree(ds)

    pc = azp.plot_ridge(
        dt,
        combined=False,
        backend="matplotlib",
    )
    _pc_add_title(pc, "Expected Returns by Productivity Frontier Quintile")
    return _fig_from_pc(pc)


# ---------------------------------------------------------------------------
# Step 5e — Resampled Bayesian Posterior Returns
# ---------------------------------------------------------------------------


def build_resampled_posterior_idata(
    resampled_df: pd.DataFrame,
    n_chains: int = 8,
    n_draws: int = 10_000,
) -> "xr.DataTree":
    """
    Build DataTree from resampled posterior returns for ArviZ diagnostics.
    Simulates multi-chain structure for convergence checking.
    """
    if not ARVIZ_AVAILABLE:
        raise ImportError("arviz is required")

    if resampled_df.empty or "posterior_mean" not in resampled_df.columns:
        return _empty_datatree()

    values = resampled_df["posterior_mean"].dropna().values
    chain_len = len(values) // n_chains
    if chain_len < 2:
        return _empty_datatree()

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

    return _build_datatree(posterior)


def create_resampled_posterior_diagnostics(
    resampled_df: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Generate ArviZ trace, rank, and distribution plots for resampled posteriors."""
    outputs: list[str] = []
    if not ARVIZ_AVAILABLE:
        return outputs

    dt = build_resampled_posterior_idata(resampled_df)
    if not _has_posterior(dt):
        return outputs

    # 1) Trace plot with title
    try:
        pc = azp.plot_trace(
            dt,
            var_names=["posterior_return"],
            backend="matplotlib",
        )
        _pc_add_title(pc, "MCMC Sampling Traces: Resampled Posterior Returns")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_resampled_trace.png"))
    except Exception as e:
        logger.debug("Resampled trace plot failed: %s", e)

    # 2) Rank plot
    try:
        pc = azp.plot_rank(
            dt,
            var_names=["posterior_return"],
            backend="matplotlib",
        )
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_resampled_rank.png"))
    except Exception as e:
        logger.debug("Resampled rank plot failed: %s", e)

    # 3) ECDF distribution with reference quantile lines (NEW)
    try:
        ref_ds = dt["posterior"].ds.quantile([0.5, 0.1, 0.9], dim=["chain", "draw"])
        pc = azp.plot_dist(
            dt,
            var_names=["posterior_return"],
            kind="ecdf",
            backend="matplotlib",
        )
        pc = azp.add_lines(
            pc,
            values=ref_ds,
            ref_dim="quantile",
            aes_by_visuals={"ref_line": ["color"]},
            color=["black", "gray", "gray"],
        )
        _pc_add_title(pc, "Resampled Posterior ECDF with Quantile References")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_resampled_ecdf.png"))
    except Exception as e:
        logger.debug("Resampled ECDF plot failed: %s", e)

    # 4) Dot plot (quantile dotplot — NEW in ArviZ 1.0)
    try:
        pc = azp.plot_dist(
            dt,
            var_names=["posterior_return"],
            kind="dot",
            visuals={"point_estimate_text":True},
            stats={"dist": {"nquantiles": 200}},
            backend="matplotlib",
        )
        _pc_add_title(pc, "Resampled Posterior Dot Plot (200 quantiles)")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_resampled_dotplot.png"))
    except Exception as e:
        logger.debug("Resampled dot plot failed: %s", e)

    # 5) R-hat summary
    try:
        rhat = azs.rhat(dt) if hasattr(azs, "rhat") else None
        if rhat is not None:
            logger.info("Resampled posterior R̂: %s", rhat)
    except Exception as e:
        logger.debug("R-hat computation failed: %s", e)

    return outputs


def create_resampled_sector_forest(
    resampled_df: pd.DataFrame,
    df_source: pd.DataFrame,
    sector_col: str = "industry",
    top_n: int = 50,
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

    n_chains, n_draws = 8, 10_000
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
    dt = _build_datatree(ds)

    pc = azp.plot_forest(
        dt,
        combined=True,
        backend="matplotlib",
    )
    _pc_add_title(pc, "Sector Resampled Posterior Returns (94% HDI)")
    return _fig_from_pc(pc)


# ---------------------------------------------------------------------------
# Step 6 — Tri-Model & Quad-Model Alignment
# ---------------------------------------------------------------------------


def build_alignment_inference_data(
    summary: pd.DataFrame,
) -> "xr.DataTree":
    """
    Build DataTree comparing the four model return distributions
    for cross-model posterior comparison plots.
    """
    if not ARVIZ_AVAILABLE:
        raise ImportError("arviz is required")

    model_cols = {
        "Monte Carlo": "implied_return_mc",
        "Kalman Filter": "implied_return_kalman",
        "Price Target": "implied_return_pt",
        "Price Target Fair Value": "price_target_prob_weighted",
        "MC Fair Value": "price_target_mc",
        "Kalman Fair Value": "price_target_kalman",
    }

    n_chains, n_draws = 8, 10_000
    rng = np.random.default_rng(42)
    posterior_dict: dict[str, tuple] = {}
    for model_name, col in model_cols.items():
        if col not in summary.columns:
            continue
        vals = summary[col].dropna().values
        if len(vals) < 100:
            continue
        samples = np.array(
            [
                [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_draws)]
                for _ in range(n_chains)
            ]
        )
        posterior_dict[model_name] = (["chain", "draw"], samples)

    if not posterior_dict:
        return _empty_datatree()

    ds = xr.Dataset(posterior_dict, coords={"chain": range(n_chains), "draw": range(n_draws)})
    return _build_datatree(ds)


def create_model_alignment_arviz_panel(
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """
    Generate ArviZ panel: forest + ridge + pair_plot for cross-model alignment.
    Directly surfaces the MC vs Kalman level discrepancy.
    """
    outputs: list[str] = []
    if not ARVIZ_AVAILABLE:
        return outputs

    dt = build_alignment_inference_data(summary)
    if not _has_posterior(dt):
        return outputs

    # 1) Forest plot with shade_label for model names
    try:
        pc = azp.plot_forest(
            dt,
            shade_label="model",
            combined=True,
            backend="matplotlib",
        )
        _pc_add_title(pc, "Cross-Model Expected Return Posteriors (94% HDI)")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_model_alignment_forest.png"))
    except Exception as e:
        logger.debug("Model alignment forest plot failed: %s", e)

    # 2) Ridge overlay
    try:
        pc = azp.plot_ridge(
            dt,
            combined=True,
            backend="matplotlib",
        )
        _pc_add_title(pc, "Cross-Model Posterior Density Overlay")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_model_alignment_density.png"))
    except Exception as e:
        logger.debug("Model alignment ridge plot failed: %s", e)

    # 3) Pair plot
    try:
        pc = azp.plot_pair(dt, backend="matplotlib")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_model_alignment_pair.png"))
    except Exception as e:
        logger.debug("Model alignment pair plot failed: %s", e)

    # 4) ECDF comparison with quantile references (NEW)
    try:
        ref_ds = dt["posterior"].ds.quantile([0.5, 0.05, 0.95], dim=["chain", "draw"])
        pc = azp.plot_dist(
            dt,
            kind="ecdf",
            backend="matplotlib",
        )
        pc = azp.add_lines(
            pc,
            values=ref_ds,
            ref_dim="quantile",
            aes_by_visuals={"ref_line": ["color"]},
            color=["black", "gray", "gray"],
        )
        _pc_add_title(pc, "Cross-Model ECDF with 5%/50%/95% Quantiles")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_model_alignment_ecdf.png"))
    except Exception as e:
        logger.debug("Model alignment ECDF failed: %s", e)

    return outputs


def create_agreement_posterior_by_sector(
    summary: pd.DataFrame,
    sector_col: str = "industry",
):
    """
    ArviZ-style forest plot of agreement_score posterior by sector.
    Reveals which sectors have genuine consensus vs noisy alignment.
    """
    if not ARVIZ_AVAILABLE:
        return None
    if "agreement_score" not in summary.columns or sector_col not in summary.columns:
        return None

    n_chains, n_draws = 8, 10_000
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
    dt = _build_datatree(ds)

    pc = azp.plot_forest(
        dt,
        combined=True,
        backend="matplotlib",
    )
    _pc_add_title(pc, "Model Agreement Score by Industry (94% HDI)")
    return _fig_from_pc(pc)


# ---------------------------------------------------------------------------
# Step 7 — Expected Returns Summary + Hierarchical Sector MCMC
# ---------------------------------------------------------------------------


def create_hierarchical_shrinkage_diagnostic(
    summary: pd.DataFrame,
    return_col: str = "implied_return_pt",
    sector_col: str = "industry",
):
    """
    ArviZ plot showing raw sector means vs hierarchical posterior means
    with shrinkage arrows and HDI intervals.
    """
    if not ARVIZ_AVAILABLE:
        return None

    from probabilistic_ml_model.statistical_functions.statistical_models import (
        hierarchical_mcmc_by_sector,
    )

    hier = hierarchical_mcmc_by_sector(summary, return_col, sector_col=sector_col)
    if not isinstance(hier, dict):
        return None

    sectors_data = hier.get("sectors", hier)
    n_chains, n_draws = 8, 10_000
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
    dt = _build_datatree(ds)

    pc = azp.plot_forest(
        dt,
        combined=True,
        backend="matplotlib",
    )
    fig = _fig_from_pc(pc)
    ax = fig.axes[0] if fig.axes else None
    if ax is not None:
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
    return fig


def create_multi_level_mcmc_comparison(
    summary: pd.DataFrame,
    return_col: str = "implied_return_pt",
    levels: Optional[list[str]] = None,
):
    """
    ArviZ ridge plot comparing posterior distributions across
    hierarchical category levels (region, sector, size_class, style_class).
    """
    if not ARVIZ_AVAILABLE:
        return None

    levels = levels or ["region","country","unit","exchange", "sector","industry", "size_class", "style_class"]
    n_chains, n_draws = 8, 10_000
    rng = np.random.default_rng(42)
    level_figs = []

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
            dt = _build_datatree(ds)
            try:
                pc = azp.plot_ridge(
                    dt,
                    combined=True,
                    backend="matplotlib",
                )
                fig = _fig_from_pc(pc)
                fig.suptitle(f"Posterior Returns by {level_col.replace('_', ' ').title()}")
                level_figs.append(fig)
            except Exception as e:
                logger.debug("Ridge plot for level %s failed: %s", level_col, e)

    if not level_figs:
        return None

    # Return the first figure if only one, otherwise combine
    if len(level_figs) == 1:
        return level_figs[0]

    # Combine into a single figure by returning the list; caller can handle
    # For backward compat, return the first figure
    return level_figs[0]


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
        {"implied_return_pt": (["chain", "draw"], chain_array)},
        coords={"chain": range(n_chains), "draw": range(min_len)},
    )
    dt = _build_datatree(posterior)

    # 1) Trace plot
    try:
        pc = azp.plot_trace(dt, var_names=["implied_return_pt"], backend="matplotlib")
        _pc_add_title(pc, "MCMC Sampling Traces: Expected Returns")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_mcmc_trace.png"))
    except Exception as e:
        logger.debug("MCMC trace plot failed: %s", e)

    # 2) Rank plot
    try:
        pc = azp.plot_rank(dt, var_names=["implied_return_pt"], backend="matplotlib")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_mcmc_rank_bars.png"))
    except Exception as e:
        logger.debug("MCMC rank plot failed: %s", e)

    # 3) Autocorrelation
    try:
        pc = azp.plot_autocorr(
            dt, var_names=["implied_return_pt"], backend="matplotlib"
        )
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_mcmc_autocorr.png"))
    except Exception as e:
        logger.debug("MCMC autocorr plot failed: %s", e)

    # 4) ESS evolution
    try:
        pc = azp.plot_ess(
            dt,
            var_names=["implied_return_pt"],
            kind="evolution",
            backend="matplotlib",
        )
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_mcmc_ess_evolution.png"))
    except Exception as e:
        logger.debug("MCMC ESS plot failed: %s", e)

    # 5) MCSE
    try:
        pc = azp.plot_mcse(dt, var_names=["implied_return_pt"], backend="matplotlib")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_mcmc_mcse.png"))
    except Exception as e:
        logger.debug("MCMC MCSE plot failed: %s", e)

    # 6) Dot plot — quantile representation (NEW)
    try:
        pc = azp.plot_dist(
            dt,
            var_names=["implied_return_pt"],
            kind="dot",
            visuals={"point_estimate_text": True},
            stats={"dist": {"nquantiles": 200}},
            backend="matplotlib",
        )
        _pc_add_title(pc, "MCMC Posterior Dot Plot (200 quantiles)")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_mcmc_dotplot.png"))
    except Exception as e:
        logger.debug("MCMC dot plot failed: %s", e)

    # 7) ECDF with reference quantiles (NEW)
    try:
        ref_ds = dt["posterior"].ds.quantile([0.5, 0.1, 0.9], dim=["chain", "draw"])
        pc = azp.plot_dist(dt, kind="ecdf", backend="matplotlib")
        pc = azp.add_lines(
            pc,
            values=ref_ds,
            ref_dim="quantile",
            aes_by_visuals={"ref_line": ["color"]},
            color=["black", "gray", "gray"],
        )
        _pc_add_title(pc, "MCMC Posterior ECDF with 10%/50%/90% Reference Lines")
        fig = _fig_from_pc(pc)
        outputs.append(_save_fig(fig, output_dir / "er_mcmc_ecdf.png"))
    except Exception as e:
        logger.debug("MCMC ECDF plot failed: %s", e)

    # 8) PPC Rootogram (NEW — if observed data available)
    observed = mcmc_result.get("observed_returns")
    if observed is not None:
        try:
            obs_ds = xr.Dataset(
                {"implied_return_pt": (["obs"], np.array(observed))},
                coords={"obs": range(len(observed))},
            )
            dt_ppc = _make_datatree(
                posterior_predictive=posterior,
                observed_data=obs_ds,
            )
            pc = azp.plot_ppc_rootogram(
                dt_ppc,
                aes={"color": ["__variable__"]},
                aes_by_visuals={"title": ["color"]},
                backend="matplotlib",
            )
            _pc_add_title(pc, "Posterior Predictive Rootogram: Expected Returns")
            fig = _fig_from_pc(pc)
            outputs.append(_save_fig(fig, output_dir / "er_mcmc_ppc_rootogram.png"))
        except Exception as e:
            logger.debug("PPC rootogram failed: %s", e)

    # 9) Summary statistics
    try:
        summary_df = azs.summary(dt, var_names=["implied_return_pt"])
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
) -> dict[str, "xr.DataTree"]:
    """
    Build per-category DataTree from category probability analytics results.
    Each category gets its own DataTree with feature-level posteriors.
    """
    if not ARVIZ_AVAILABLE:
        return {}

    category_datatrees: dict[str, xr.DataTree] = {}
    n_chains, n_draws = 8, 10_000
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
            category_datatrees[cat_name] = _build_datatree(ds)

    return category_datatrees


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

    category_datatrees = build_category_analytics_idata(category_analytics, df)

    for cat_name, dt in category_datatrees.items():
        safe_name = cat_name.lower().replace(" ", "_").replace("&", "and")
        var_names = list(dt["posterior"].ds.data_vars)[:max_features_per_plot]

        # 1) Ridge plot (replaces plot_forest kind="ridgeplot")
        try:
            pc = azp.plot_ridge(
                dt,
                var_names=var_names,
                combined=True,
                backend="matplotlib",
            )
            _pc_add_title(pc, f"{cat_name} — Feature Posterior Distributions")
            fig = _fig_from_pc(pc)
            outputs.append(_save_fig(fig, output_dir / f"er_category_{safe_name}_ridge.png"))
        except Exception as e:
            logger.debug("Ridge plot failed for %s: %s", cat_name, e)

        # 2) Forest plot with HDI
        try:
            pc = azp.plot_forest(
                dt,
                var_names=var_names,
                combined=True,
                backend="matplotlib",
            )
            _pc_add_title(pc, f"{cat_name} — Feature Posterior Forest (94% HDI)")
            fig = _fig_from_pc(pc)
            outputs.append(_save_fig(fig, output_dir / f"er_category_{safe_name}_forest.png"))
        except Exception as e:
            logger.debug("Forest plot failed for %s: %s", cat_name, e)

        # 3) ESS summary
        try:
            ess = azs.ess(dt) if hasattr(azs, "ess") else None
            if ess is not None:
                low_ess = {
                    var: float(ess[var].values)
                    for var in ess.data_vars
                    if float(ess[var].values) < 400
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

    n_chains, n_draws = 8, 10_000
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
    dt = _build_datatree(ds)

    pc = azp.plot_ridge(
        dt,
        combined=True,
        backend="matplotlib",
    )
    _pc_add_title(pc, "Cross-Category Bayesian Posterior Summary (Feature Domains)")
    fig = _fig_from_pc(pc)
    path = output_dir / "er_cross_category_posterior.png"
    return _save_fig(fig, path)


# ---------------------------------------------------------------------------
# New ArviZ 1.0 visualizations
# ---------------------------------------------------------------------------


def create_screening_ppc_rootogram(
    screens: dict[str, pd.DataFrame],
    return_col: str = "implied_return_pt",
) -> Optional[plt.Figure]:
    """PPC rootogram comparing predicted vs observed return distributions
    across screening strategies.

    Directly addresses whether screening-selected stocks' predicted returns
    match their observed return profiles.
    """
    if not ARVIZ_AVAILABLE:
        return None

    dt = build_screening_inference_data(screens, return_col=return_col)
    if not _has_posterior(dt):
        return None

    # Build observed from actual screen returns
    obs_dict = {}
    max_obs = 0
    for screen_name, screen_df in screens.items():
        if screen_df.empty or return_col not in screen_df.columns:
            continue
        vals = screen_df[return_col].dropna().values
        if len(vals) >= 10:
            obs_dict[screen_name] = vals
            max_obs = max(max_obs, len(vals))

    if not obs_dict:
        return None

    # Pad to uniform length
    padded = {}
    for name, vals in obs_dict.items():
        padded[name] = (["obs"], np.pad(vals, (0, max_obs - len(vals)), constant_values=np.nan))

    obs_ds = xr.Dataset(padded, coords={"obs": range(max_obs)})
    dt_ppc = _make_datatree(
        posterior_predictive=dt["posterior"].ds,
        observed_data=obs_ds,
    )

    try:
        pc = azp.plot_ppc_rootogram(
            dt_ppc,
            aes={"color": ["__variable__"]},
            aes_by_visuals={"title": ["color"]},
            backend="matplotlib",
        )
        _pc_add_title(pc, "Screening Strategy — PPC Rootogram")
        return _fig_from_pc(pc)
    except Exception:
        return None


def create_hierarchical_dot_comparison(
    summary: pd.DataFrame,
    return_col: str = "implied_return_pt",
    sector_col: str = "industry",
) -> Optional[plt.Figure]:
    """Dot plot comparing hierarchical posterior distributions by sector.

    Uses the ArviZ 1.0 quantile dot plot for intuitive visualization of
    sector-level expected return uncertainty — each dot represents a
    posterior quantile, making distribution shape immediately visible.
    """
    if not ARVIZ_AVAILABLE:
        return None

    from probabilistic_ml_model.statistical_functions.statistical_models import (
        hierarchical_mcmc_by_sector,
    )

    hier = hierarchical_mcmc_by_sector(summary, return_col, sector_col=sector_col)
    if not isinstance(hier, dict):
        return None

    sectors_data = hier.get("sectors", hier)
    n_chains, n_draws = 8, 10_000
    rng = np.random.default_rng(42)
    posterior_dict = {}

    for sector, info in sectors_data.items():
        if not isinstance(info, dict) or "posterior_mean" not in info:
            continue
        post_mean = info["posterior_mean"]
        post_std = info.get("posterior_std", 5.0)
        samples = rng.normal(post_mean, max(post_std, 0.01), size=(n_chains, n_draws))
        posterior_dict[sector] = (["chain", "draw"], samples)

    if not posterior_dict:
        return None

    ds = xr.Dataset(posterior_dict, coords={"chain": range(n_chains), "draw": range(n_draws)})
    dt = _make_datatree(posterior=ds)

    try:
        pc = azp.plot_dist(
            dt,
            kind="dot",
            visuals={"point_estimate_text": True},
            stats={"dist": {"nquantiles": 100}},
            backend="matplotlib",
        )
        _pc_add_title(pc, f"Sector Posterior Dot Plot — Hierarchical Shrinkage ({sector_col})")
        return _fig_from_pc(pc)
    except Exception:
        return None


def create_cross_model_ecdf_with_references(
    summary: pd.DataFrame,
    output_dir: Path,
) -> Optional[str]:
    """ECDF comparison of MC vs Kalman vs Price Target return distributions
    with reference quantile lines showing median and 90% credible interval.

    Surfaces the MC↔Kalman correlation from pipeline output and
    highlights where models diverge in the tails.
    """
    if not ARVIZ_AVAILABLE:
        return None

    dt = build_alignment_inference_data(summary)
    if not _has_posterior(dt):
        return None

    try:
        ref_ds = dt["posterior"].ds.quantile([0.5, 0.05, 0.95], dim=["chain", "draw"])
        pc = azp.plot_dist(dt, kind="ecdf", backend="matplotlib")
        pc = azp.add_lines(
            pc,
            values=ref_ds,
            ref_dim="quantile",
            aes_by_visuals={"ref_line": ["color"]},
            color=["black", "gray", "gray"],
        )
        _pc_add_title(pc, "Cross-Model Return ECDF — MC vs Kalman vs Price Target")
        fig = _fig_from_pc(pc)
        return _save_fig(fig, output_dir / "er_cross_model_ecdf.png")
    except Exception:
        return None
