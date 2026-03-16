"""
Probabilistic financial analysis visualizations (ArviZ-enhanced).

This module provides interactive ArviZ-backed plots for Bayesian
financial analysis, bridging the inference_schema.py InferenceData
objects and the analytics.* database tables into rich Plotly dashboards.

Functions:
- create_posterior_return_forest: Forest plot of posterior expected returns
- create_beat_probability_posterior: Posterior density of earnings beat probability
- create_ruin_probability_diagnostic: Ruin probability MCMC diagnostics
- create_mcse_convergence_panel: Monte Carlo Standard Error convergence
- create_bayesian_category_ridge: Ridge plot of posterior feature means
- create_tri_model_posterior_comparison: Overlaid posteriors from MC / Kalman / Achievement
- create_mcmc_anomaly_posterior_chart: MCMC anomaly score posterior with sector shrinkage
- create_mcmc_credit_risk_chart: MCMC credit risk posterior (heuristic vs MCMC)
- create_mcmc_dividend_cut_chart: MCMC dividend cut probability posterior
- create_mcmc_price_target_chart: MCMC price target achievement posterior with R-hat
- create_mcmc_category_posterior_chart: MCMC category feature posterior forest plot

Data sources (postgres.analytics):
    - monte_carlo_simulation
    - earnings_probability_analysis
    - eps_streak_analysis
    - expected_returns_tri_model
    - credit_risk_analysis
    - profitability_distributions
    - probability_analytics_summary

ArviZ / arviz-plots integration:
    When ``arviz`` is available the module generates InferenceData objects
    via inference_schema.py and overlays ArviZ diagnostics (R-hat, ESS,
    MCSE) onto the Plotly figures.  All functions degrade gracefully to
    pure Plotly/Scipy when ArviZ is not installed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots
from scipy import stats

from analytics.visualizations._shared import (
    PLOTLY_TEMPLATE,
    COLORS,
    create_no_data_figure,
)
from analytics.inference_schema import (
    FeatureViewSpec,
    IdentifierCoordinates,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_CI_LINE_WIDTH = 4
_MEDIAN_MARKER_SIZE = 8
_MIN_FOREST_HEIGHT = 500
_ROW_HEIGHT_PX = 22
_ZERO_LINE_COLOR = "red"
_ZERO_LINE_OPACITY = 0.6
_CI_HOVER_TEMPLATE = (
    "<b>{ticker}</b><br>" "{ci_pct:.0%} CI: [{lo:.1f}%, {hi:.1f}%]" "<extra></extra>"
)

# ---------------------------------------------------------------------------
# Lazy ArviZ import (matches project-wide pattern)
# ---------------------------------------------------------------------------
try:
    import arviz as az
    ARVIZ_AVAILABLE = hasattr(az, "InferenceData")
except ImportError:  # pragma: no cover
    az = None  # type: ignore[assignment]
    ARVIZ_AVAILABLE = False

def float_array(x) -> np.ndarray:
    """Convert to a float64 numpy array (handles xarray scalars gracefully)."""
    return np.asarray(x, dtype=np.float64)


def _ci_bounds(credible_interval: float) -> tuple[float, float]:
    """Return the (lower, upper) quantile fractions for a symmetric ETI.

    >>> _ci_bounds(0.94)
    (0.03, 0.97)
    """
    tail = (1.0 - credible_interval) / 2.0
    return tail, 1.0 - tail

# ═══════════════════════════════════════════════════════════════════════════
# 1. Posterior Return Forest Plot
# ═══════════════════════════════════════════════════════════════════════════


def create_posterior_return_forest(
    idata_or_df: "az.InferenceData | pd.DataFrame",
    var_name: str = "expected_return_prob_weighted",
    top_n: int = 30,
    credible_interval: float = 0.94,
    sort_by: str = "median",
    title: Optional[str] = None,
    view_spec: Optional[FeatureViewSpec] = None,
) -> go.Figure:
    """
    Forest plot of posterior expected returns per equity.

    When *idata_or_df* is an ArviZ InferenceData (from
    ``BayesianTechnicalResampler.build_inference_data`` or
    ``build_monte_carlo_inference_data``), the plot extracts
    posterior samples, computes credible intervals, and annotates R-hat / ESS.

    Falls back to a simple credible-interval bar chart when only a
    DataFrame (e.g. ``analytics.expected_returns_tri_model``) is given.

    Parameters
    ----------
    idata_or_df : arviz.InferenceData or pd.DataFrame
        Posterior samples or a summary DataFrame with columns:
        ``ticker``, ``expected_upside_pct``, ``prob_positive_upside``.
    var_name : str, default 'expected_return'
        Variable name inside InferenceData posterior group.
    top_n : int, default 30
        Number of equities to display (sorted by *sort_by*).
    credible_interval : float, default 0.94
        Width of the credible interval (ETI).
    sort_by : str, default 'median'
        Sort criterion: 'median', 'mean', or 'hdi_width'.
    title : str, optional
        Custom figure title.

    Returns
    -------
    go.Figure
    """
    if not 0 < credible_interval < 1:
        raise ValueError(f"credible_interval must be in (0, 1), got {credible_interval}")

    if title is None and view_spec is not None:
        title = f"Posterior Returns — {view_spec.category} Features"
    title = title or f"Posterior Expected Returns — Top {top_n} Equities"

    # ── ArviZ path ────────────────────────────────────────────────────────
    if ARVIZ_AVAILABLE and isinstance(idata_or_df, az.InferenceData):
        return _forest_from_idata(idata_or_df, var_name, top_n, credible_interval, sort_by, title)

    # ── DataFrame fallback ────────────────────────────────────────────────
    if isinstance(idata_or_df, pd.DataFrame):
        return _forest_from_dataframe(idata_or_df, top_n, credible_interval, title)

    return create_no_data_figure(title)


# ---------------------------------------------------------------------------
# Shared forest-plot figure builder
# ---------------------------------------------------------------------------


def _build_forest_figure(
    tickers: np.ndarray,
    medians: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    credible_interval: float,
    title: str,
) -> go.Figure:
    """Construct the Plotly forest-plot figure from pre-computed arrays."""
    fig = go.Figure()

    for i in range(len(tickers)):
        fig.add_trace(
            go.Scatter(
                x=[ci_lower[i], ci_upper[i]],
                y=[tickers[i], tickers[i]],
                mode="lines",
                line=dict(color=COLORS[0], width=_CI_LINE_WIDTH),
                showlegend=False,
                hovertemplate=_CI_HOVER_TEMPLATE.format(
                    ticker=tickers[i],
                    ci_pct=credible_interval,
                    lo=ci_lower[i],
                    hi=ci_upper[i],
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=medians,
            y=tickers,
            mode="markers",
            marker=dict(size=_MEDIAN_MARKER_SIZE, color=COLORS[2], symbol="diamond"),
            name="Median",
        )
    )

    fig.add_vline(x=0, line_dash="dash", line_color=_ZERO_LINE_COLOR, opacity=_ZERO_LINE_OPACITY)
    fig.update_layout(
        title=title,
        xaxis_title="Annualised Return (%)",
        yaxis=dict(autorange="reversed"),
        height=max(_MIN_FOREST_HEIGHT, len(tickers) * _ROW_HEIGHT_PX),
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# Posterior helpers
# ---------------------------------------------------------------------------


def _sort_and_slice_posterior(
    tickers: np.ndarray,
    medians: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    sort_by: str,
    top_n: int,
    sort_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort arrays by the chosen criterion and return the top-*n* slice.

    Parameters
    ----------
    sort_fallback : array, optional
        Values used when *sort_by* is neither ``"median"`` nor
        ``"hdi_width"`` (e.g. posterior means).

    Returns (tickers, medians, ci_lower, ci_upper) after sorting and slicing.
    """
    sort_key = {
        "median": medians,
        "hdi_width": ci_upper - ci_lower,
    }
    sort_vals = sort_key.get(sort_by, sort_fallback if sort_fallback is not None else medians)
    order = np.argsort(sort_vals)[::-1][:top_n]
    return tickers[order], medians[order], ci_lower[order], ci_upper[order]


def _compute_rhat_annotation(
    idata: "az.InferenceData",
    var_name: str,
    n_chains: int,
) -> str:
    """Return an R-hat annotation string, or empty if unavailable."""
    if n_chains <= 1:
        return ""
    try:
        rhat = az.rhat(idata)
        max_rhat = float(rhat[var_name].max().values)
        return f"  |  R̂ max = {max_rhat:.3f}"
    except Exception:
        return ""


def _forest_from_idata(
    idata: "az.InferenceData",
    var_name: str,
    top_n: int,
    credible_interval: float,
    sort_by: str,
    title: str,
) -> go.Figure:
    """Build forest plot directly from InferenceData posterior samples."""
    posterior = idata.posterior
    if var_name not in posterior.data_vars:
        return create_no_data_figure(f"{title} — variable '{var_name}' not found")

    samples = posterior[var_name]  # (chain, draw, equity)
    tickers = samples.coords["equity"].values

    medians = float_array(samples.median(dim=("chain", "draw")).values)
    means = float_array(samples.mean(dim=("chain", "draw")).values)
    q_lo, q_hi = _ci_bounds(credible_interval)
    ci_lower = float_array(samples.quantile(q_lo, dim=("chain", "draw")).values)
    ci_upper = float_array(samples.quantile(q_hi, dim=("chain", "draw")).values)

    tickers, medians, ci_lower, ci_upper = _sort_and_slice_posterior(
        tickers,
        medians,
        ci_lower,
        ci_upper,
        sort_by=sort_by,
        top_n=top_n,
        sort_fallback=means,
    )

    rhat_note = _compute_rhat_annotation(idata, var_name, samples.sizes["chain"])
    annotated_title = f"{title}{rhat_note}" if rhat_note else title

    return _build_forest_figure(
        tickers, medians, ci_lower, ci_upper, credible_interval, annotated_title
    )


def _forest_from_dataframe(
    df: pd.DataFrame,
    top_n: int,
    credible_interval: float,
    title: str,
) -> go.Figure:
    """Fallback forest plot from a summary DataFrame."""
    if "expected_upside_pct" not in df.columns or "ticker" not in df.columns:
        return create_no_data_figure(f"{title} — missing columns")

    plot_df = (
        df.dropna(subset=["expected_upside_pct"]).nlargest(top_n, "expected_upside_pct").copy()
    )
    if plot_df.empty:
        return create_no_data_figure(title)

    # Approximate CI from upside_std if available
    z = stats.norm.ppf(1 - (1 - credible_interval) / 2)
    if "upside_std" in plot_df.columns:
        plot_df["lo"] = plot_df["expected_upside_pct"] - z * plot_df["upside_std"]
        plot_df["hi"] = plot_df["expected_upside_pct"] + z * plot_df["upside_std"]
    else:
        # Additive fallback using overall std to avoid broken intervals
        # on negative or zero upside values
        overall_std = plot_df["expected_upside_pct"].std()
        fallback_spread = z * max(overall_std * 0.3, 1.0) if not pd.isna(overall_std) else z * 1.0
        plot_df["lo"] = plot_df["expected_upside_pct"] - fallback_spread
        plot_df["hi"] = plot_df["expected_upside_pct"] + fallback_spread

    fig = go.Figure()
    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["lo"], row["hi"]],
                y=[row["ticker"], row["ticker"]],
                mode="lines",
                line=dict(color=COLORS[0], width=_CI_LINE_WIDTH),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['ticker']}</b><br>"
                    f"Upside: {row['expected_upside_pct']:.1f}%<br>"
                    f"{credible_interval:.0%} CI: [{row['lo']:.1f}%, {row['hi']:.1f}%]<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=plot_df["expected_upside_pct"],
            y=plot_df["ticker"],
            mode="markers",
            marker=dict(size=_MEDIAN_MARKER_SIZE, color=COLORS[2], symbol="diamond"),
            name="Expected Upside",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)
    fig.update_layout(
        title=title,
        xaxis_title="Expected Upside (%)",
        yaxis=dict(autorange="reversed"),
        height=max(_MIN_FOREST_HEIGHT, top_n * _ROW_HEIGHT_PX),
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. Earnings Beat Probability — Posterior Density
# ═══════════════════════════════════════════════════════════════════════════


def create_beat_probability_posterior(
    idata_or_df: "az.InferenceData | pd.DataFrame",
    tickers: Optional[list[str]] = None,
    top_n: int = 12,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Posterior density plot for earnings beat probability per equity.

    Leverages ``build_beat_probability_inference_data`` output or
    the ``analytics.earnings_probability_analysis`` table directly.

    Parameters
    ----------
    idata_or_df : arviz.InferenceData or pd.DataFrame
        Beat probability posteriors.
    tickers : list[str], optional
        Specific tickers to plot. If None, picks top *top_n* by posterior.
    top_n : int, default 12
        Number of equities if *tickers* is None.
    title : str, optional
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or "Posterior Earnings Beat Probability"

    # ── ArviZ path ────────────────────────────────────────────────────────
    if ARVIZ_AVAILABLE and isinstance(idata_or_df, az.InferenceData):
        return _beat_density_from_idata(idata_or_df, tickers, top_n, title)

    # ── DataFrame fallback ────────────────────────────────────────────────
    if isinstance(idata_or_df, pd.DataFrame):
        return _beat_density_from_dataframe(idata_or_df, tickers, top_n, title)

    return create_no_data_figure(title)


def _beat_density_from_idata(idata, tickers, top_n, title) -> go.Figure:
    """KDE overlays from InferenceData posterior."""
    posterior = idata.posterior
    var_name = "beat_probability" if "beat_probability" in posterior.data_vars else None
    if var_name is None:
        return create_no_data_figure(f"{title} — no beat_probability variable")

    all_tickers = posterior.coords["equity"].values
    if tickers is None:
        medians = posterior[var_name].median(dim=("chain", "draw")).values
        order = np.argsort(medians)[::-1][:top_n]
        tickers = all_tickers[order]

    fig = go.Figure()
    for i, t in enumerate(tickers):
        idx = np.where(all_tickers == t)[0]
        if len(idx) == 0:
            continue
        samples = posterior[var_name].values[:, :, idx[0]].flatten()
        x_kde = np.linspace(0, 1, 200)
        try:
            kde = stats.gaussian_kde(samples)
            y_kde = kde(x_kde)
        except Exception:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde,
                mode="lines",
                name=str(t),
                line=dict(width=2),
                fill="tozeroy",
                opacity=0.3,
            )
        )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
    fig.update_layout(
        title=title,
        xaxis_title="P(Beat Next Quarter)",
        yaxis_title="Density",
        height=550,
        template=PLOTLY_TEMPLATE,
    )
    return fig


def _beat_density_from_dataframe(df, tickers, top_n, title) -> go.Figure:
    """Approximate Beta posterior from posterior_alpha / posterior_beta columns."""
    required = {"ticker", "posterior_alpha", "posterior_beta"}
    if not required.issubset(df.columns):
        # Try simpler columns
        if "posterior_beat_prob" in df.columns and "ticker" in df.columns:
            return _beat_bar_fallback(df, tickers, top_n, title)
        return create_no_data_figure(f"{title} — missing columns")

    if tickers is None:
        df_sorted = df.sort_values(
            "posterior_beat_prob" if "posterior_beat_prob" in df.columns else "posterior_alpha",
            ascending=False,
        )
        tickers = df_sorted["ticker"].head(top_n).tolist()

    fig = go.Figure()
    x_grid = np.linspace(0, 1, 200)
    for t in tickers:
        row = df[df["ticker"] == t]
        if row.empty:
            continue
        a = float(row["posterior_alpha"].iloc[0])
        b = float(row["posterior_beta"].iloc[0])
        if pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0:
            continue
        y = stats.beta.pdf(x_grid, a, b)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y,
                mode="lines",
                name=str(t),
                line=dict(width=2),
                fill="tozeroy",
                opacity=0.3,
            )
        )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
    fig.update_layout(
        title=title,
        xaxis_title="P(Beat Next Quarter)",
        yaxis_title="Beta Density",
        height=550,
        template=PLOTLY_TEMPLATE,
    )
    return fig


def _beat_bar_fallback(df, tickers, top_n, title) -> go.Figure:
    """Simple horizontal bar when only point estimates are available."""
    if tickers is None:
        plot_df = df.nlargest(top_n, "posterior_beat_prob")
    else:
        plot_df = df[df["ticker"].isin(tickers)]

    fig = go.Figure(
        go.Bar(
            y=plot_df["ticker"],
            x=plot_df["posterior_beat_prob"],
            orientation="h",
            marker_color=[
                COLORS[2] if v > 0.5 else COLORS[3] for v in plot_df["posterior_beat_prob"]
            ],
            hovertemplate="<b>%{y}</b><br>P(Beat): %{x:.2%}<extra></extra>",
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Posterior Beat Probability",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(plot_df) * 22),
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. Ruin Probability MCMC Diagnostics
# ═══════════════════════════════════════════════════════════════════════════


def create_ruin_probability_diagnostic(
    idata_or_df: "az.InferenceData | pd.DataFrame",
    top_n: int = 20,
    title: Optional[str] = None,
    identifier_coords: Optional[IdentifierCoordinates] = None,
) -> go.Figure:
    """
    Four-panel diagnostic dashboard for ruin probability posteriors.

    Panels:
    1. Posterior density of ruin probability (top-risk equities)
    2. Risk tier distribution (pie / bar)
    3. Ruin prob vs distress_risk_score scatter
    4. Convergence trace (if InferenceData with chains)

    Uses ``build_credit_risk_inference_data`` output or
    ``analytics.credit_risk_analysis`` directly.

    Parameters
    ----------
    idata_or_df : arviz.InferenceData or pd.DataFrame
        Ruin probability posteriors or credit risk summary.
    top_n : int, default 20
        Number of equities for density panel.
    title : str, optional
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or "Ruin Probability — Bayesian Diagnostic Dashboard"

    if isinstance(idata_or_df, pd.DataFrame):
        return _ruin_diagnostic_from_df(idata_or_df, top_n, title)

    if ARVIZ_AVAILABLE and isinstance(idata_or_df, az.InferenceData):
        return _ruin_diagnostic_from_idata(idata_or_df, top_n, title)

    return create_no_data_figure(title)


def _ruin_diagnostic_from_df(df: pd.DataFrame, top_n: int, title: str) -> go.Figure:
    """Build diagnostic from credit_risk_analysis or ruin probability DataFrame."""
    ruin_col = "ruin_probability" if "ruin_probability" in df.columns else "distress_probability"
    risk_col = "risk_tier" if "risk_tier" in df.columns else "risk_level"
    distress_col = (
        "distress_risk_score" if "distress_risk_score" in df.columns else "altman_z_score"
    )

    if ruin_col not in df.columns:
        return create_no_data_figure(f"{title} — no ruin/distress probability column")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Top {top_n} Highest Ruin Probability",
            "Risk Tier Distribution",
            (
                f"Ruin Probability vs {distress_col.replace('_', ' ').title()}"
                if distress_col in df.columns
                else "Ruin Probability Histogram"
            ),
            "Sector Median Ruin Probability",
        ),
        specs=[[{"type": "bar"}, {"type": "pie"}], [{"type": "scatter"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Panel 1: Top ruin probabilities
    top = df.nlargest(top_n, ruin_col)
    fig.add_trace(
        go.Bar(
            x=top["ticker"] if "ticker" in top.columns else top.index.astype(str),
            y=top[ruin_col],
            marker_color=[_ruin_color(v) for v in top[ruin_col]],
            hovertemplate="<b>%{x}</b><br>P(Ruin): %{y:.2%}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(tickangle=-45, row=1, col=1)

    # Panel 2: Risk tier pie
    if risk_col in df.columns:
        tier_counts = df[risk_col].value_counts()
        fig.add_trace(
            go.Pie(
                labels=tier_counts.index.astype(str),
                values=tier_counts.values,
                marker_colors=[_tier_color(t) for t in tier_counts.index],
                textinfo="label+percent",
            ),
            row=1,
            col=2,
        )

    # Panel 3: Scatter vs distress score
    if distress_col in df.columns:
        df_dropped = df.dropna(subset=[ruin_col, distress_col])
        sample = df_dropped.sample(
            min(2000, len(df_dropped)),
            random_state=42,
        )
        fig.add_trace(
            go.Scatter(
                x=sample[distress_col],
                y=sample[ruin_col],
                mode="markers",
                marker=dict(size=4, color=sample[ruin_col], colorscale="Reds", opacity=0.5),
                hovertemplate="%{text}<br>%{x:.1f} / %{y:.2%}<extra></extra>",
                text=sample.get("ticker", sample.index.astype(str)),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text=distress_col.replace("_", " ").title(), row=2, col=1)
        fig.update_yaxes(title_text="P(Ruin)", row=2, col=1)

    # Panel 4: Sector medians
    group_col = "sector" if "sector" in df.columns else "industry"
    if group_col in df.columns:
        sector_med = df.groupby(group_col)[ruin_col].median().sort_values(ascending=True).tail(20)
        fig.add_trace(
            go.Bar(
                x=sector_med.values,
                y=sector_med.index.astype(str),
                orientation="h",
                marker_color=[_ruin_color(v) for v in sector_med.values],
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=title,
        height=900,
        template=PLOTLY_TEMPLATE,
    )
    return fig


def _ruin_diagnostic_from_idata(idata, top_n, title) -> go.Figure:
    """Build diagnostic panels using ArviZ InferenceData."""
    posterior = idata.posterior
    var_name = "ruin_probability"
    if var_name not in posterior.data_vars:
        return create_no_data_figure(f"{title} — no ruin_probability in posterior")

    samples = posterior[var_name]  # (chain, draw, equity)
    tickers = samples.coords["equity"].values
    medians = float_array(samples.median(dim=("chain", "draw")).values)

    # Reuse the DataFrame path with computed stats
    summary_df = pd.DataFrame(
        {
            "ticker": tickers,
            "ruin_probability": medians,
        }
    )
    # Classify risk tiers
    summary_df["risk_tier"] = pd.cut(
        summary_df["ruin_probability"],
        bins=[0, 0.1, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
    )
    fig = _ruin_diagnostic_from_df(summary_df, top_n, title)

    # Annotate convergence
    if samples.sizes["chain"] > 1:
        try:
            rhat = az.rhat(idata)
            max_rhat = float(rhat[var_name].max().values)
            ess = az.ess(idata)
            min_ess = float(ess[var_name].min().values)
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.05,
                text=f"R̂ max: {max_rhat:.3f}  |  ESS min: {min_ess:.0f}",
                showarrow=False,
                font=dict(size=11, color="white"),
            )
        except Exception:
            pass

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. MCSE Convergence Panel
# ═══════════════════════════════════════════════════════════════════════════


def create_mcse_convergence_panel(
    idata: "az.InferenceData | pd.DataFrame",
    var_name: str = "expected_return_prob_weighted",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Monte Carlo Standard Error convergence panel.

    Shows MCSE as a function of draw count, annotated with ESS and R-hat.
    Requires ArviZ InferenceData; returns a no-data figure when the input
    is not an InferenceData object or ArviZ is unavailable.

    Parameters
    ----------
    idata : arviz.InferenceData or pd.DataFrame
        Must contain a posterior group with *var_name*.
        If a DataFrame is passed, a no-data placeholder is returned.
    var_name : str
        Variable to diagnose.
    title : str, optional
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or f"MCSE Convergence — {var_name}"

    if not ARVIZ_AVAILABLE:
        return create_no_data_figure(f"{title} — ArviZ required")
    if not isinstance(idata, az.InferenceData) or not hasattr(idata, "posterior"):
        return create_no_data_figure(f"{title} — no posterior group")
    if var_name not in idata.posterior.data_vars:
        return create_no_data_figure(f"{title} — variable not found")

    samples = idata.posterior[var_name]  # (chain, draw, equity)
    n_chains = samples.sizes["chain"]
    n_draws = samples.sizes["draw"]

    # Compute running MCSE across draws for each chain (mean across equities)
    draw_counts = np.arange(100, n_draws + 1, max(1, n_draws // 50))

    fig = go.Figure()
    for c in range(n_chains):
        chain_samples = samples.isel(chain=c).values  # (draw, equity)
        mcse_vals = []
        for d in draw_counts:
            sub = chain_samples[:d, :]
            se = sub.std(axis=0) / np.sqrt(d)
            mcse_vals.append(float(np.nanmean(se)))
        fig.add_trace(
            go.Scatter(
                x=draw_counts,
                y=mcse_vals,
                mode="lines",
                name=f"Chain {c}",
                line=dict(width=2),
            )
        )

    # Annotate ESS & R-hat
    try:
        ess = az.ess(idata)
        rhat = az.rhat(idata)
        min_ess = float(ess[var_name].min().values)
        mean_ess = float(ess[var_name].mean().values)
        max_rhat = float(rhat[var_name].max().values)
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            text=f"ESS min/mean: {min_ess:.0f}/{mean_ess:.0f}<br>R̂ max: {max_rhat:.4f}",
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="gray",
        )
    except Exception:
        pass

    fig.update_layout(
        title=title,
        xaxis_title="Number of Draws",
        yaxis_title="Mean MCSE (across equities)",
        height=450,
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. Bayesian Category Ridge Plot
# ═══════════════════════════════════════════════════════════════════════════


def create_bayesian_category_ridge(
    analysis_results: dict[str, dict],
    category_name: str = "Profitability",
    n_samples: int = 4000,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Ridge plot of posterior means for features within a category.

    Uses the output of ``bayesian_category_analysis()`` from
    ``statistical_analysis.py``.  Each feature's Normal posterior
    N(posterior_mean, posterior_std) is rendered as a filled KDE.

    Parameters
    ----------
    analysis_results : dict
        Output from ``bayesian_category_analysis()``.
    category_name : str
        Category label for the title.
    n_samples : int, default 4000
        Number of draws for KDE estimation.
    title : str, optional
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or f"Posterior Feature Distributions — {category_name}"

    features = list(analysis_results.keys())
    if not features:
        return create_no_data_figure(title)

    rng = np.random.default_rng(42)
    fig = go.Figure()

    for i, feat in enumerate(features):
        info = analysis_results[feat]
        pm = info.get("posterior_mean", 0)
        ps = info.get("posterior_std", 1)
        if ps <= 0:
            continue

        # Use ArviZ InferenceData samples when available (from updated
        # bayesian_category_analysis), otherwise fall back to parametric draws.
        idata = info.get("inference_data")
        if ARVIZ_AVAILABLE and idata is not None:
            try:
                samples = idata.posterior["mu"].values.flatten()
            except Exception:
                samples = rng.normal(pm, ps, n_samples)
        else:
            samples = rng.normal(pm, ps, n_samples)
        x_kde = np.linspace(pm - 4 * ps, pm + 4 * ps, 200)
        kde = stats.gaussian_kde(samples)
        y_kde = kde(x_kde)

        # Ridge offset
        offset = i * 0.8
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde + offset,
                mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                fill="tozeroy" if i == 0 else None,
                name=feat,
                hovertemplate=(
                    f"<b>{feat}</b><br>"
                    f"Posterior mean: {pm:.3f}<br>"
                    f"95% CI: [{info.get('ci_95_low', 0):.3f}, {info.get('ci_95_high', 0):.3f}]<br>"
                    f"P(>0): {info.get('prob_positive', 0):.1%}<extra></extra>"
                ),
            )
        )
        # Baseline for each ridge
        fig.add_hline(y=offset, line_dash="dot", line_color="gray", opacity=0.2)

    fig.update_layout(
        title=title,
        xaxis_title="Parameter Value",
        yaxis_title="Feature (stacked density)",
        yaxis=dict(
            tickvals=[i * 0.8 for i in range(len(features))],
            ticktext=features,
        ),
        height=max(450, len(features) * 80),
        template=PLOTLY_TEMPLATE,
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. Tri-Model Posterior Comparison
# ═══════════════════════════════════════════════════════════════════════════


def create_tri_model_posterior_comparison(
    tri_df: pd.DataFrame,
    tickers: Optional[list[str]] = None,
    top_n: int = 8,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Overlaid approximate posteriors from the three expected-return models.

    For each equity, draws Normal posteriors centred on:
    - Monte Carlo expected upside (spread from ``upside_std``)
    - Kalman filtered upside (spread estimated from MC std)
    - Price target achievement prob-weighted return

    Uses ``analytics.expected_returns_tri_model`` as data source.

    Parameters
    ----------
    tri_df : pd.DataFrame
        Tri-model alignment DataFrame (expected_upside_pct,
        filtered_upside, expected_return_prob_weighted, ticker).
    tickers : list[str], optional
        Specific tickers. If None, picks top *top_n* by agreement_score.
    top_n : int, default 8
        Equities to display.
    title : str, optional
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or "Tri-Model Posterior Return Comparison"

    required = {"ticker", "expected_upside_pct", "filtered_upside", "expected_return_prob_weighted"}
    if not required.issubset(tri_df.columns):
        return create_no_data_figure(f"{title} — missing columns")

    if tickers is None:
        sort_col = (
            "agreement_score" if "agreement_score" in tri_df.columns else "expected_upside_pct"
        )
        selected = tri_df.nlargest(top_n, sort_col)
        tickers = selected["ticker"].tolist()

    n = len(tickers)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[str(t) for t in tickers],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for idx, t in enumerate(tickers):
        row_df = tri_df[tri_df["ticker"] == t]
        if row_df.empty:
            continue
        r = row_df.iloc[0]
        r_idx = idx // cols + 1
        c_idx = idx % cols + 1

        # Shared spread (use upside_std if present, else 10% of upside)
        spread = abs(r.get("upside_std", abs(r["expected_upside_pct"]) * 0.15)) or 5.0
        x_range = np.linspace(
            min(r["expected_upside_pct"], r["filtered_upside"], r["expected_return_prob_weighted"])
            - 3 * spread,
            max(r["expected_upside_pct"], r["filtered_upside"], r["expected_return_prob_weighted"])
            + 3 * spread,
            200,
        )

        models = [
            ("MC Upside", r["expected_upside_pct"], spread, COLORS[0]),
            ("Kalman", r["filtered_upside"], spread * 0.8, COLORS[1]),
            ("Achievm.", r["expected_return_prob_weighted"], spread * 1.2, COLORS[2]),
        ]
        for name, mu, s, color in models:
            y = stats.norm.pdf(x_range, mu, s)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=name,
                    showlegend=(idx == 0),
                    legendgroup=name,
                ),
                row=r_idx,
                col=c_idx,
            )

        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.3, row=r_idx, col=c_idx)

    fig.update_layout(
        title=title,
        height=280 * rows,
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _ruin_color(p: float) -> str:
    """Map ruin probability to a traffic-light colour."""
    if p >= 0.6:
        return "#e74c3c"
    if p >= 0.3:
        return "#f39c12"
    if p >= 0.1:
        return "#3498db"
    return "#00bc8c"


def _tier_color(tier: str) -> str:
    """Map risk tier label to colour."""
    tier_lower = str(tier).lower()
    if "critical" in tier_lower:
        return "#e74c3c"
    if "high" in tier_lower:
        return "#f39c12"
    if "moderate" in tier_lower:
        return "#3498db"
    return "#00bc8c"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Feature View Posterior Panel
# ═══════════════════════════════════════════════════════════════════════════


def create_feature_view_posterior_panel(
    idata: "az.InferenceData | xr.Dataset",
    view_spec: FeatureViewSpec,
    top_n_features: int = 10,
    top_n_equities: int = 20,
    title: Optional[str] = None,
) -> go.Figure:
    """Multi-panel posterior visualization for a specific feature view.

    Creates one subplot per top feature showing equity-level posterior
    distributions.  Uses ``view_spec.feature_columns`` for axis labels.

    Parameters
    ----------
    idata : arviz.InferenceData or xr.Dataset
        Posterior samples (from ``build_feature_view_inference_data``).
    view_spec : FeatureViewSpec
        Feature view specification with category and column metadata.
    top_n_features : int, default 10
        Maximum number of feature subplots.
    top_n_equities : int, default 20
        Equities to display per feature panel.
    title : str, optional
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or f"{view_spec.category} — Feature Posterior Panel"

    # Extract the posterior dataset
    if ARVIZ_AVAILABLE and isinstance(idata, az.InferenceData):
        if hasattr(idata, "posterior"):
            ds = idata.posterior
        else:
            return create_no_data_figure(f"{title} — no posterior group")
    elif isinstance(idata, xr.Dataset):
        ds = idata
    else:
        return create_no_data_figure(title)

    # Select features present in both spec and dataset
    available = [c for c in view_spec.feature_columns if c in ds.data_vars]
    if not available:
        return create_no_data_figure(f"{title} — no matching features")

    features = available[:top_n_features]
    n_features = len(features)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=features,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    for idx, feat in enumerate(features):
        r_idx = idx // cols + 1
        c_idx = idx % cols + 1

        var_data = ds[feat]
        # Collapse chain/draw → per-equity medians and CIs
        if "chain" in var_data.dims and "draw" in var_data.dims:
            flat = var_data.stack(sample=("chain", "draw"))
            medians = float_array(flat.median(dim="sample"))
            q_lo = float_array(flat.quantile(0.03, dim="sample"))
            q_hi = float_array(flat.quantile(0.97, dim="sample"))
        elif "equity" in var_data.dims:
            medians = float_array(var_data)
            q_lo = medians
            q_hi = medians
        else:
            continue

        # Sort by median, take top_n_equities
        order = np.argsort(medians)[::-1][:top_n_equities]
        tickers = (
            np.array(ds.coords["equity"].values)[order]
            if "equity" in ds.coords
            else np.arange(len(medians))[order].astype(str)
        )

        for i, oi in enumerate(order):
            fig.add_trace(
                go.Scatter(
                    x=[q_lo[oi], q_hi[oi]],
                    y=[tickers[i], tickers[i]],
                    mode="lines",
                    line=dict(color=COLORS[0], width=3),
                    showlegend=False,
                ),
                row=r_idx,
                col=c_idx,
            )

        fig.add_trace(
            go.Scatter(
                x=medians[order],
                y=tickers,
                mode="markers",
                marker=dict(size=6, color=COLORS[2], symbol="diamond"),
                showlegend=False,
            ),
            row=r_idx,
            col=c_idx,
        )

    fig.update_layout(
        title=title,
        height=max(500, 250 * rows),
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 8. Accounting Anomaly Conditional Probability Visualization
# ═══════════════════════════════════════════════════════════════════════════


def create_anomaly_conditional_probability_chart(
    df: pd.DataFrame,
    cond_probs: pd.DataFrame | None = None,
    top_n: int = 20,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Four-panel dashboard for Bayesian conditional anomaly probabilities.

    Visualizes output from
    :meth:`AccountingAnomalyProbabilityModel.calculate_conditional_probabilities`
    and the per-row ``anomaly_conditional_probability`` column.

    Panels:
    1. Per-feature P(Anomaly|High) vs P(Anomaly|Low) — paired bar chart
    2. Feature separation scores (horizontal bar, descending)
    3. Lift ratios per feature (high vs low, grouped bar)
    4. Per-stock conditional P(Anomaly) histogram with tier overlay

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by ``AccountingAnomalyProbabilityModel.analyze_dataframe``.
        Must contain ``anomaly_conditional_probability``.
    cond_probs : pd.DataFrame or None
        Output of ``calculate_conditional_probabilities``.  If *None* the
        model will be instantiated to compute it from *df*.
    top_n : int, default 20
        Maximum number of features to display.
    title : str, optional
        Custom title.

    Returns
    -------
    go.Figure
        Plotly figure with a 4-panel conditional probability dashboard.
    """
    title = title or "Accounting Anomaly — Conditional Probability Analysis"

    if "anomaly_conditional_probability" not in df.columns:
        return create_no_data_figure(
            f"{title} — run AccountingAnomalyProbabilityModel.analyze_dataframe first"
        )

    # Compute conditional probabilities if not supplied
    if cond_probs is None:
        from analytics.probability_analytics import (
            AccountingAnomalyProbabilityModel,
        )
        cond_probs = AccountingAnomalyProbabilityModel().calculate_conditional_probabilities(df)

    has_tier = "accounting_anomaly_tier" in df.columns
    tier_colors = {
        "Clean": "#00A878",
        "Watch": "#FFD93D",
        "Flag": "#FF8C42",
        "Alert": "#E63946",
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "P(Anomaly | Feature High) vs P(Anomaly | Feature Low)",
            "Feature Separation Scores",
            "Lift Ratios by Feature",
            "Per-Stock Conditional P(Anomaly) Distribution",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    if not cond_probs.empty:
        cp = cond_probs.head(top_n)
        features = cp["feature"].tolist()

        # ── Panel 1: Paired bar — P(Anomaly|High) vs P(Anomaly|Low) ──
        fig.add_trace(
            go.Bar(
                y=features,
                x=cp["p_anomaly_high"],
                orientation="h",
                name="P(Anomaly|High)",
                marker_color="#E63946",
                opacity=0.8,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                y=features,
                x=cp["p_anomaly_low"],
                orientation="h",
                name="P(Anomaly|Low)",
                marker_color="#00A878",
                opacity=0.8,
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Probability", row=1, col=1)

        # ── Panel 2: Separation scores ──
        fig.add_trace(
            go.Bar(
                y=features,
                x=cp["separation"],
                orientation="h",
                marker_color="#6C63FF",
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Separation: %{x:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Separation", row=1, col=2)

        # ── Panel 3: Lift ratios (grouped bar) ──
        fig.add_trace(
            go.Bar(
                y=features,
                x=cp["lift_high"],
                orientation="h",
                name="Lift (High)",
                marker_color="#FF8C42",
                opacity=0.8,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                y=features,
                x=cp["lift_low"],
                orientation="h",
                name="Lift (Low)",
                marker_color="#0A7EA4",
                opacity=0.8,
            ),
            row=2,
            col=1,
        )
        # Add base-rate reference line
        if "base_anomaly_rate" in cp.columns and len(cp) > 0:
            fig.add_vline(
                x=1.0,
                line_dash="dash",
                line_color="grey",
                row=2,
                col=1,
            )
        fig.update_xaxes(title_text="Lift Ratio", row=2, col=1)

    # ── Panel 4: Per-stock conditional probability histogram ──
    if has_tier:
        for tier in ["Clean", "Watch", "Flag", "Alert"]:
            tier_data = df.loc[
                df["accounting_anomaly_tier"] == tier,
                "anomaly_conditional_probability",
            ].dropna()
            if len(tier_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=tier_data,
                        name=f"{tier}",
                        marker_color=tier_colors.get(tier, "#0A7EA4"),
                        opacity=0.65,
                    ),
                    row=2,
                    col=2,
                )
    else:
        fig.add_trace(
            go.Histogram(
                x=df["anomaly_conditional_probability"].dropna(),
                marker_color="#0A7EA4",
                opacity=0.7,
                name="P(Anomaly)",
            ),
            row=2,
            col=2,
        )
    fig.update_xaxes(title_text="Conditional P(Anomaly)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=1000,
        width=1200,
        showlegend=True,
        barmode="group",
        margin=dict(l=180, r=40, t=80, b=60),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 9. MCMC Anomaly Posterior Dashboard
# ═══════════════════════════════════════════════════════════════════════════


def create_mcmc_anomaly_posterior_chart(
    df: pd.DataFrame,
    top_n: int = 20,
    title: Optional[str] = None,
) -> go.Figure:
    """Dashboard for MCMC-enhanced anomaly posterior estimates.

    Visualizes the Student-t posterior and sector-level hierarchical
    shrinkage columns produced by
    ``AccountingAnomalyProbabilityModel._apply_mcmc_posteriors()``.

    Panels
    ------
    1. Top-N stocks by anomaly score with MCMC credible intervals
    2. Sector posterior means (hierarchical shrinkage) vs raw means

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``AccountingAnomalyProbabilityModel.analyze_dataframe()``
        with ``use_mcmc=True``.  Expected columns include
        ``accounting_anomaly_score``, ``anomaly_posterior_mean``,
        ``anomaly_ci_lower``, ``anomaly_ci_upper``, and optionally
        ``sector_posterior_mean``.
    top_n : int
        Number of stocks to display.
    title : str | None
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or "MCMC Anomaly Score Posterior"

    mcmc_cols = {"anomaly_posterior_mean", "anomaly_ci_lower", "anomaly_ci_upper"}
    if not isinstance(df, pd.DataFrame) or df.empty or not mcmc_cols.issubset(df.columns):
        return create_no_data_figure(f"{title} — missing MCMC posterior columns")

    has_sector = "sector_posterior_mean" in df.columns
    n_panels = 2 if has_sector else 1
    subplot_titles = ["Top Anomaly Scores with MCMC CI"]
    if has_sector:
        subplot_titles.append("Sector Posterior vs Raw Mean")

    fig = make_subplots(
        rows=1,
        cols=n_panels,
        subplot_titles=subplot_titles,
    )

    # Panel 1: top-N stocks by anomaly score with CI
    score_col = "accounting_anomaly_score"
    if score_col not in df.columns:
        return create_no_data_figure(f"{title} — missing {score_col}")

    plot_df = df.dropna(subset=[score_col]).nlargest(top_n, score_col)
    if plot_df.empty:
        return create_no_data_figure(f"{title} — no data after filtering")

    tickers = (
        plot_df["ticker"].values
        if "ticker" in plot_df.columns
        else plot_df.index.astype(str).values
    )
    scores = plot_df[score_col].values
    ci_lo = plot_df["anomaly_ci_lower"].values
    ci_hi = plot_df["anomaly_ci_upper"].values

    # CI error bars (symmetric around posterior mean)
    post_mean = plot_df["anomaly_posterior_mean"].values
    fig.add_trace(
        go.Bar(
            y=tickers,
            x=scores,
            orientation="h",
            name="Anomaly Score",
            marker_color=COLORS[0],
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    # Overlay posterior mean with CI
    fig.add_trace(
        go.Scatter(
            y=tickers,
            x=post_mean,
            mode="markers",
            name="Posterior Mean",
            marker=dict(color=COLORS[1], size=8, symbol="diamond"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=ci_hi - post_mean,
                arrayminus=post_mean - ci_lo,
                color=COLORS[1],
                thickness=1.5,
            ),
        ),
        row=1,
        col=1,
    )

    # Panel 2: sector posterior vs raw
    if has_sector:
        sector_col = None
        for c in ("industry", "sector"):
            if c in df.columns:
                sector_col = c
                break
        if sector_col:
            sector_df = (
                df.dropna(subset=[score_col, "sector_posterior_mean"])
                .groupby(sector_col)
                .agg(
                    raw_mean=(score_col, "mean"),
                    posterior_mean=("sector_posterior_mean", "first"),
                )
                .reset_index()
                .sort_values("raw_mean", ascending=True)
            )
            fig.add_trace(
                go.Bar(
                    y=sector_df[sector_col],
                    x=sector_df["raw_mean"],
                    orientation="h",
                    name="Raw Mean",
                    marker_color=COLORS[2],
                    opacity=0.7,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    y=sector_df[sector_col],
                    x=sector_df["posterior_mean"],
                    mode="markers",
                    name="Shrunk Posterior",
                    marker=dict(color=COLORS[3], size=10, symbol="diamond"),
                ),
                row=1,
                col=2,
            )

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=max(_MIN_FOREST_HEIGHT, top_n * _ROW_HEIGHT_PX + 120),
        showlegend=True,
        margin=dict(l=160, r=40, t=80, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 10. MCMC Credit Risk Posterior Dashboard
# ═══════════════════════════════════════════════════════════════════════════


def create_mcmc_credit_risk_chart(
    df: pd.DataFrame,
    top_n: int = 20,
    title: Optional[str] = None,
) -> go.Figure:
    """Dashboard for MCMC-enhanced credit risk posterior estimates.

    Visualizes the Metropolis-Hastings and Student-t posterior columns
    produced by ``CreditRiskProbabilityModel._apply_mcmc_posteriors()``.

    Panels
    ------
    1. Heuristic vs MCMC distress probability (scatter)
    2. Sector Z-score posterior means (hierarchical shrinkage)

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``CreditRiskProbabilityModel.analyze_dataframe()``
        with ``use_mcmc=True``.
    top_n : int
        Number of stocks to display.
    title : str | None
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or "MCMC Credit Risk Posterior"

    if not isinstance(df, pd.DataFrame) or df.empty:
        return create_no_data_figure(f"{title} — no data")

    has_mcmc = "mcmc_distress_probability" in df.columns
    has_heuristic = "distress_probability" in df.columns
    has_sector = "sector_z_posterior_mean" in df.columns
    # NEW: v3.4 enrichment columns
    has_bs_distress = "balance_sheet_strength" in df.columns and "distress_risk_score" in df.columns

    if not has_mcmc:
        return create_no_data_figure(f"{title} — missing mcmc_distress_probability")

    n_panels = 1 + int(has_sector) + int(has_bs_distress)
    subplot_titles = ["Heuristic vs MCMC Distress Probability"]
    if has_sector:
        subplot_titles.append("Sector Z-Score Posterior (Hierarchical)")
    if has_bs_distress:
        subplot_titles.append("Balance Sheet Strength vs Distress Risk")

    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=subplot_titles)

    # Panel 1: scatter heuristic vs MCMC
    plot_df = df.dropna(subset=["mcmc_distress_probability"]).head(top_n * 5)
    x_col = "distress_probability" if has_heuristic else "mcmc_distress_probability"
    tickers = (
        plot_df["ticker"].values
        if "ticker" in plot_df.columns
        else plot_df.index.astype(str).values
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df["mcmc_distress_probability"],
            mode="markers",
            text=tickers,
            marker=dict(
                color=plot_df["mcmc_distress_probability"],
                colorscale="RdYlGn_r",
                size=8,
                showscale=True,
                colorbar=dict(title="MCMC P(distress)"),
            ),
            hovertemplate="<b>%{text}</b><br>Heuristic: %{x:.2%}<br>MCMC: %{y:.2%}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    # 45-degree reference line
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash", color="grey", width=1),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Heuristic P(distress)", row=1, col=1)
    fig.update_yaxes(title_text="MCMC P(distress)", row=1, col=1)

    # Panel 2: sector posterior
    if has_sector:
        sector_col = None
        for c in ("industry", "sector"):
            if c in df.columns:
                sector_col = c
                break
        if sector_col:
            sector_df = (
                df.dropna(subset=["sector_z_posterior_mean"])
                .groupby(sector_col)
                .agg(posterior_mean=("sector_z_posterior_mean", "first"))
                .reset_index()
                .sort_values("posterior_mean", ascending=True)
            )
            fig.add_trace(
                go.Bar(
                    y=sector_df[sector_col],
                    x=sector_df["posterior_mean"],
                    orientation="h",
                    marker_color=COLORS[0],
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.update_xaxes(title_text="Posterior Z-Score Mean", row=1, col=2)

    # Panel 3 (optional): Balance Sheet Strength vs Distress Risk Score
    if has_bs_distress:
        bs_col_idx = n_panels  # last panel
        bs_df = df.dropna(subset=["balance_sheet_strength", "distress_risk_score"]).head(top_n * 5)
        if not bs_df.empty:
            bs_tickers = (
                bs_df["ticker"].values
                if "ticker" in bs_df.columns
                else bs_df.index.astype(str).values
            )
            fig.add_trace(
                go.Scatter(
                    x=bs_df["balance_sheet_strength"],
                    y=bs_df["distress_risk_score"],
                    mode="markers",
                    text=bs_tickers,
                    marker=dict(
                        color=bs_df.get("mcmc_distress_probability", bs_df["distress_risk_score"]),
                        colorscale="RdYlGn_r",
                        size=7,
                        showscale=False,
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "BS Strength: %{x:.0f}<br>"
                        "Distress Risk: %{y:.0f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=bs_col_idx,
            )
            fig.update_xaxes(title_text="Balance Sheet Strength", row=1, col=bs_col_idx)
            fig.update_yaxes(title_text="Distress Risk Score", row=1, col=bs_col_idx)

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=600,
        showlegend=True,
        margin=dict(l=120, r=40, t=80, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 11. MCMC Dividend Cut Posterior Dashboard
# ═══════════════════════════════════════════════════════════════════════════


def create_mcmc_dividend_cut_chart(
    df: pd.DataFrame,
    top_n: int = 20,
    title: Optional[str] = None,
) -> go.Figure:
    """Dashboard for MCMC-enhanced dividend cut probability estimates.

    Visualizes the composite posterior from FCF coverage and payout ratio
    produced by ``DividendCutProbabilityModel._apply_mcmc_posteriors()``.

    Panels
    ------
    1. Top-N riskiest stocks: heuristic vs MCMC cut probability
    2. FCF coverage posterior CI

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``DividendCutProbabilityModel.analyze_dataframe()``
        with ``use_mcmc=True``.
    top_n : int
        Number of stocks to display.
    title : str | None
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or "MCMC Dividend Cut Probability"

    if not isinstance(df, pd.DataFrame) or df.empty:
        return create_no_data_figure(f"{title} — no data")

    has_mcmc = "mcmc_cut_probability" in df.columns
    if not has_mcmc:
        return create_no_data_figure(f"{title} — missing mcmc_cut_probability")

    has_heuristic = "dividend_cut_probability" in df.columns
    has_ci = {"mcmc_ci_lower", "mcmc_ci_upper"}.issubset(df.columns)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Heuristic vs MCMC Cut Probability",
            "FCF Coverage Posterior CI",
        ],
    )

    # Panel 1: grouped bar — heuristic vs MCMC
    sort_col = "mcmc_cut_probability"
    plot_df = df.dropna(subset=[sort_col]).nlargest(top_n, sort_col)
    tickers = (
        plot_df["ticker"].values
        if "ticker" in plot_df.columns
        else plot_df.index.astype(str).values
    )

    if has_heuristic:
        fig.add_trace(
            go.Bar(
                y=tickers,
                x=plot_df["dividend_cut_probability"],
                orientation="h",
                name="Heuristic",
                marker_color=COLORS[2],
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            y=tickers,
            x=plot_df["mcmc_cut_probability"],
            orientation="h",
            name="MCMC Composite",
            marker_color=COLORS[1],
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="P(Dividend Cut)", row=1, col=1)

    # Panel 2: CI from FCF posterior
    if has_ci:
        ci_df = plot_df.dropna(subset=["mcmc_ci_lower", "mcmc_ci_upper"])
        if not ci_df.empty:
            ci_tickers = (
                ci_df["ticker"].values
                if "ticker" in ci_df.columns
                else ci_df.index.astype(str).values
            )
            ci_lo = ci_df["mcmc_ci_lower"].values
            ci_hi = ci_df["mcmc_ci_upper"].values
            mid = (ci_lo + ci_hi) / 2
            fig.add_trace(
                go.Scatter(
                    y=ci_tickers,
                    x=mid,
                    mode="markers",
                    name="FCF Posterior Midpoint",
                    marker=dict(color=COLORS[0], size=8),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=ci_hi - mid,
                        arrayminus=mid - ci_lo,
                        color=COLORS[0],
                        thickness=1.5,
                    ),
                ),
                row=1,
                col=2,
            )
            fig.update_xaxes(title_text="FCF Coverage (95% CI)", row=1, col=2)

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=max(_MIN_FOREST_HEIGHT, top_n * _ROW_HEIGHT_PX + 120),
        showlegend=True,
        barmode="group",
        margin=dict(l=160, r=40, t=80, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 12. MCMC Price Target Achievement Posterior Dashboard
# ═══════════════════════════════════════════════════════════════════════════


def create_mcmc_price_target_chart(
    df: pd.DataFrame,
    top_n: int = 20,
    title: Optional[str] = None,
) -> go.Figure:
    """Dashboard for MCMC-enhanced price target achievement estimates.

    Visualizes the Student-t posterior and parallel MCMC convergence
    diagnostics produced by
    ``PriceTargetAchievementModel._apply_mcmc_posteriors()``.

    Panels
    ------
    1. Heuristic vs MCMC achievement probability (scatter)
    2. MCMC posterior expected return (prob-weighted) with CI

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``PriceTargetAchievementModel.analyze_dataframe()``
        with ``use_mcmc=True``.
    top_n : int
        Number of stocks to display.
    title : str | None
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or "MCMC Price Target Achievement Posterior"

    if not isinstance(df, pd.DataFrame) or df.empty:
        return create_no_data_figure(f"{title} — no data")

    has_mcmc = "mcmc_achievement_probability" in df.columns
    if not has_mcmc:
        return create_no_data_figure(f"{title} — missing mcmc_achievement_probability")

    has_heuristic = "achievement_probability" in df.columns
    has_weighted = "mcmc_expected_return_prob_weighted" in df.columns
    has_ci = {"mcmc_ci_lower", "mcmc_ci_upper"}.issubset(df.columns)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Heuristic vs MCMC Achievement Probability",
            "MCMC Posterior Expected Return (Prob-Weighted)",
        ],
    )

    # Panel 1: scatter
    plot_df = df.dropna(subset=["mcmc_achievement_probability"])
    x_col = "achievement_probability" if has_heuristic else "mcmc_achievement_probability"
    tickers = (
        plot_df["ticker"].values
        if "ticker" in plot_df.columns
        else plot_df.index.astype(str).values
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df["mcmc_achievement_probability"],
            mode="markers",
            text=tickers,
            marker=dict(
                color=plot_df["mcmc_achievement_probability"],
                colorscale="Viridis",
                size=8,
                showscale=True,
                colorbar=dict(title="MCMC P(achieve)"),
            ),
            hovertemplate="<b>%{text}</b><br>Heuristic: %{x:.2%}<br>MCMC: %{y:.2%}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash", color="grey", width=1),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Heuristic P(achieve)", row=1, col=1)
    fig.update_yaxes(title_text="MCMC P(achieve)", row=1, col=1)

    # Panel 2: prob-weighted return with CI
    if has_weighted:
        top_df = plot_df.nlargest(top_n, "mcmc_expected_return_prob_weighted")
        t_tickers = (
            top_df["ticker"].values
            if "ticker" in top_df.columns
            else top_df.index.astype(str).values
        )
        weighted = top_df["mcmc_expected_return_prob_weighted"].values

        error_kwargs = {}
        if has_ci:
            ci_lo = top_df["mcmc_ci_lower"].values
            ci_hi = top_df["mcmc_ci_upper"].values
            error_kwargs = dict(
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=ci_hi - weighted,
                    arrayminus=weighted - ci_lo,
                    color=COLORS[1],
                    thickness=1.5,
                ),
            )

        fig.add_trace(
            go.Scatter(
                y=t_tickers,
                x=weighted,
                mode="markers",
                name="MCMC E[R] × P(achieve)",
                marker=dict(color=COLORS[0], size=8),
                **error_kwargs,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Prob-Weighted Return (%)", row=1, col=2)

    # Annotate Gelman-Rubin if available
    if "mcmc_gelman_rubin" in df.columns:
        rhat = df["mcmc_gelman_rubin"].dropna()
        if len(rhat) > 0:
            rhat_val = rhat.iloc[0]
            fig.add_annotation(
                text=f"R̂ = {rhat_val:.4f}",
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.02,
                showarrow=False,
                font=dict(size=11, color="green" if rhat_val < 1.1 else "red"),
            )

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=600,
        showlegend=True,
        margin=dict(l=120, r=40, t=80, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 13. MCMC Category Probability Posterior Dashboard
# ═══════════════════════════════════════════════════════════════════════════


def create_mcmc_category_posterior_chart(
    analytics: dict,
    category_name: str = "Category",
    title: Optional[str] = None,
) -> go.Figure:
    """Dashboard for MCMC-enhanced category probability posteriors.

    Visualizes the output of ``run_category_probability_analytics()``
    or ``CategoryProbabilityAnalyzer.analyze_view()`` with MCMC enabled.

    Each feature's posterior mean and 95% CI are shown as a forest plot.

    Parameters
    ----------
    analytics : dict
        Mapping of ``feature_name → {posterior_mean, posterior_std,
        ci_95_low / ci_lower_95, ci_95_high / ci_upper_95, ...}``.
    category_name : str
        Category label for the title.
    title : str | None
        Custom title.

    Returns
    -------
    go.Figure
    """
    title = title or f"MCMC Category Posterior — {category_name}"

    if not isinstance(analytics, dict) or not analytics:
        return create_no_data_figure(f"{title} — no analytics data")

    features = []
    means = []
    ci_lo = []
    ci_hi = []

    for feat, info in analytics.items():
        if not isinstance(info, dict):
            continue
        pm = info.get("posterior_mean")
        if pm is None:
            continue
        ps = info.get("posterior_std", 0)
        lo = info.get("ci_95_low", info.get("ci_lower_95", pm - 1.96 * ps))
        hi = info.get("ci_95_high", info.get("ci_upper_95", pm + 1.96 * ps))
        features.append(feat)
        means.append(pm)
        ci_lo.append(lo)
        ci_hi.append(hi)

    if not features:
        return create_no_data_figure(f"{title} — no valid posterior data")

    means_arr = np.array(means)
    ci_lo_arr = np.array(ci_lo)
    ci_hi_arr = np.array(ci_hi)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=features,
            x=means_arr,
            mode="markers",
            name="Posterior Mean",
            marker=dict(color=COLORS[0], size=10, symbol="diamond"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=ci_hi_arr - means_arr,
                arrayminus=means_arr - ci_lo_arr,
                color=COLORS[0],
                thickness=2,
            ),
            hovertemplate="<b>%{y}</b><br>Mean: %{x:.3f}<br>CI: [%{error_x.arrayminus:.3f}, +%{error_x.array:.3f}]<extra></extra>",
        )
    )

    # Zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=max(_MIN_FOREST_HEIGHT, len(features) * _ROW_HEIGHT_PX + 120),
        xaxis_title="Posterior Mean",
        showlegend=True,
        margin=dict(l=200, r=40, t=80, b=60),
    )
    return fig
