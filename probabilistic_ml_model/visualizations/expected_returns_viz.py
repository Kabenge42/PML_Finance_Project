"""
Expected returns visualization functions extracted from expected_returns_v3.py.

Provides Plotly-based charts for Monte Carlo distributions, sector heatmaps,
model agreement, VaR analysis, and cross-model dispersion dashboards.

These functions were originally inline in the expected_returns_v3 pipeline
and have been extracted here to separate visualization from orchestration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from probabilistic_ml_model.visualizations._shared import (
    PLOTLY_TEMPLATE,
    COLORS,
    create_no_data_figure,
)
from probabilistic_ml_model.data_utils.inference_schema import (
    IdentifierCoordinates,
    EquitiesMaterializedViewSpec,
    EquitiesSchemaMetadata,
)
from probabilistic_ml_model.data_utils.feature_catalog import columns_for_viz

# Signal labels for tri-model agreement (duplicated from expected_returns_v3
# to avoid circular imports)
_SIGNAL_LABELS = {
    0: "Strong Bearish (0/3)",
    1: "Bearish (1/3)",
    2: "Bullish (2/3)",
    3: "Strong Bullish (3/3)",
}


def create_mc_return_distribution(mc: pd.DataFrame) -> go.Figure:
    """Two-panel figure: expected upside histogram + P(positive) bar chart."""
    if mc.empty or "implied_return_mc" not in mc.columns:
        return create_no_data_figure("MC Return Distribution — No Data")

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Expected Upside Distribution",
            "Probability of Positive Return",
        ),
        vertical_spacing=0.12,
    )

    upside = mc["implied_return_mc"].clip(-100, 300)
    fig.add_trace(
        go.Histogram(
            x=upside,
            nbinsx=80,
            marker_color=COLORS[0],
            opacity=0.75,
            name="Expected Upside %",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    median_val = mc["implied_return_mc"].median()
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: {median_val:.1f}%",
        row=1,
        col=1,
    )

    prob_bins = pd.cut(
        mc["prob_positive_upside"],
        bins=[0, 25, 50, 75, 100],
        labels=["0–25%", "25–50%", "50–75%", "75–100%"],
    )
    prob_counts = prob_bins.value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=prob_counts.index.astype(str),
            y=prob_counts.values,
            marker_color=[COLORS[3], COLORS[1], COLORS[0], COLORS[2]],
            name="Stock Count",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Monte Carlo Simulation: Return Distribution Overview",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Expected Upside (%)", row=1, col=1)
    fig.update_xaxes(title_text="P(Positive Return) Bucket", row=2, col=1)
    fig.update_yaxes(title_text="Number of Stocks", row=1, col=1)
    fig.update_yaxes(title_text="Number of Stocks", row=2, col=1)
    return fig


def create_sector_risk_reward_scatter(
    mc: pd.DataFrame,
    identifier_coords: Optional[IdentifierCoordinates] = None,
) -> go.Figure:
    """Sector-level bubble scatter with optional region/trading_country grouping."""
    if identifier_coords is not None and len(identifier_coords.regions) > 0:
        mc = mc.copy()
        if "region" not in mc.columns:
            ticker_to_region = dict(zip(identifier_coords.tickers, identifier_coords.regions))
            mc["region"] = mc["ticker"].map(ticker_to_region)
    sector = (
        mc.groupby("industry")
        .agg(
            mean_upside=("implied_return_mc", "mean"),
            mean_var5=("var_5_pct", "mean"),
            mean_prob=("prob_positive_upside", "mean"),
            count=("ticker", "count"),
        )
        .reset_index()
    )
    fig = px.scatter(
        sector,
        x="mean_var5",
        y="mean_upside",
        size="count",
        color="industry",
        hover_name="industry",
        hover_data={"mean_prob": ":.1f", "count": True},
        title="Industry Risk-Reward: Expected Upside vs VaR 5%",
        labels={
            "mean_var5": "Mean VaR 5% (%)",
            "mean_upside": "Mean Expected Upside (%)",
        },
        template=PLOTLY_TEMPLATE,
        height=550,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def create_kalman_vs_raw_scatter(kal: pd.DataFrame) -> go.Figure:
    """Scatter of Kalman-filtered vs raw analyst upside (log-transformed)."""
    plot_df = kal.copy()
    if "raw_upside" not in plot_df.columns:
        plot_df["raw_upside"] = (
            (plot_df["original_target"] - plot_df["original_price"])
            / plot_df["original_price"]
            * 100
        )

    sample = plot_df.sample(min(2000, len(plot_df)), random_state=42).copy()
    sample["filtered_log"] = np.sign(sample["implied_return_kalman"]) * np.log1p(
        np.abs(sample["implied_return_kalman"])
    )
    sample["raw_log"] = np.sign(sample["raw_upside"]) * np.log1p(np.abs(sample["raw_upside"]))

    # v3.5: Support both 'ticker' and 'isin' as identifiers for hover data
    id_col = "ticker" if "ticker" in sample.columns else "isin"
    hover_cols = [id_col, "implied_return_kalman", "raw_upside"]
    hover_cols = [c for c in hover_cols if c in sample.columns]

    fig = px.scatter(
        sample,
        x="raw_log",
        y="filtered_log",
        color="industry" if "industry" in sample.columns else None,
        hover_data=hover_cols,
        title="Kalman Filtered vs Raw Analyst Upside (log scale)",
        labels={
            "raw_log": "Raw Upside (signed log)",
            "filtered_log": "Kalman Filtered (signed log)",
        },
        template=PLOTLY_TEMPLATE,
        height=550,
        opacity=0.5,
    )
    log_max = max(
        sample["filtered_log"].abs().quantile(0.99),
        sample["raw_log"].abs().quantile(0.99),
    )
    fig.add_shape(
        type="line",
        x0=-log_max,
        y0=-log_max,
        x1=log_max,
        y1=log_max,
        line=dict(color="gray", dash="dash", width=1),
    )
    return fig


def create_tri_model_agreement_histogram(tri: pd.DataFrame) -> go.Figure:
    """Histogram of tri-model signal agreement (0/3 → 3/3)."""
    color_map = {
        v: [COLORS[3], COLORS[1], COLORS[0], COLORS[2]][k] for k, v in _SIGNAL_LABELS.items()
    }
    fig = px.histogram(
        tri,
        x="signal",
        color="signal",
        title="Tri-Model Signal Agreement (MC + Kalman + Achievement)",
        labels={"signal": "Model Agreement"},
        color_discrete_map=color_map,
        category_orders={"signal": list(_SIGNAL_LABELS.values())},
        template=PLOTLY_TEMPLATE,
        height=420,
    )
    fig.update_layout(showlegend=False)
    return fig


def create_strong_consensus_bar(strong: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: MC / Kalman / Achievement returns for top picks."""
    if strong.empty:
        return create_no_data_figure("Strong Consensus Picks — No Data")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=strong["ticker"],
            y=strong["implied_return_mc"],
            name="MC Expected Upside",
            marker_color=COLORS[0],
        )
    )
    fig.add_trace(
        go.Bar(
            x=strong["ticker"],
            y=strong["implied_return_kalman"],
            name="Kalman Filtered Upside",
            marker_color=COLORS[1],
        )
    )
    fig.add_trace(
        go.Bar(
            x=strong["ticker"],
            y=strong["implied_return_pt"],
            name="Prob-Weighted Return",
            marker_color=COLORS[2],
        )
    )
    if "price_target_mc" in strong.columns:
        fig.add_trace(
            go.Bar(
                x=strong["ticker"],
                y=strong["price_target_mc"],
                name="MC Fair Value ($)",
                marker_color=COLORS[3] if len(COLORS) > 3 else "#636EFA",
            )
        )
    if "price_target_kalman" in strong.columns:
        fig.add_trace(
            go.Bar(
                x=strong["ticker"],
                y=strong["price_target_kalman"],
                name="Kalman Fair Value ($)",
                marker_color=COLORS[4] if len(COLORS) > 4 else "#EF553B",
            )
        )
    fig.update_layout(
        title=f"Top {len(strong)} Strong Consensus Picks (All 3 Models Bullish)",
        yaxis_title="Expected Return (%)",
        barmode="group",
        template=PLOTLY_TEMPLATE,
        height=500,
        xaxis_tickangle=-45,
    )
    return fig


def create_sector_heatmap(
    tri: pd.DataFrame,
    compute_sector_fn=None,
    schema_metadata: Optional[
        EquitiesSchemaMetadata
    ] = None,  # noqa: ARG001 — reserved for column validation
) -> go.Figure:
    """Sector expected returns heatmap across all models.

    Parameters
    ----------
    tri : pd.DataFrame
        Tri-model alignment DataFrame.
    compute_sector_fn : callable, optional
        Function to compute sector expected returns from *tri*.
        When ``None``, a lightweight fallback aggregation is used.
    schema_metadata : EquitiesSchemaMetadata, optional
        Schema metadata for column validation.
    """
    if callable(compute_sector_fn):
        sector = compute_sector_fn(tri)
    else:
        # Lightweight fallback aggregation
        sector = _default_sector_aggregation(tri)

    if sector.empty:
        return create_no_data_figure("Sector Heatmap — No Data")

    heatmap_data = sector.set_index("industry")[
        [
            "mc_mean",
            "mc_median",
            "kalman_mean",
            "kalman_median",
            "pt_mean",
            "pt_median",
            "pct_bullish",
        ]
    ].rename(
        columns={
            "mc_mean": "MC Mean",
            "mc_median": "MC Median",
            "kalman_mean": "Kalman Mean",
            "kalman_median": "Kalman Median",
            "pt_mean": "Achiev. Mean",
            "pt_median": "Achiev. Median",
            "pct_bullish": "% All Bullish",
        }
    )

    fig = px.imshow(
        heatmap_data.round(1),
        color_continuous_scale="RdYlGn",
        text_auto=True,
        aspect="auto",
        title="Industry Expected Returns Heatmap (All Models)",
        labels={"color": "Value"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=max(600, len(sector) * 22))
    return fig


def _default_sector_aggregation(tri: pd.DataFrame) -> pd.DataFrame:
    """Lightweight sector aggregation when compute_sector_expected_returns is unavailable."""
    if tri.empty or "industry" not in tri.columns:
        return pd.DataFrame()

    agg_map = {}
    col_pairs = [
        ("implied_return_mc", "mc"),
        ("price_target_mc", "mc_price"),
        ("implied_return_kalman", "kalman"),
        ("price_target_kalman", "kalman_price"),
        ("implied_return_pt", "pt"),
    ]
    for col, prefix in col_pairs:
        if col in tri.columns:
            agg_map[f"{prefix}_mean"] = (col, "mean")
            agg_map[f"{prefix}_median"] = (col, "median")

    if not agg_map:
        return pd.DataFrame()

    sector = tri.groupby("industry").agg(**agg_map).reset_index()
    if "agreement_score" in tri.columns:
        bullish = (
            tri.groupby("industry")["agreement_score"]
            .apply(lambda x: (x == 3).mean() * 100)
            .reset_index(name="pct_bullish")
        )
        sector = sector.merge(bullish, on="industry", how="left")
    else:
        sector["pct_bullish"] = 0.0
    return sector


def create_var_analysis(mc: pd.DataFrame) -> go.Figure:
    """Two-panel VaR analysis: distribution + VaR vs upside scatter."""
    if mc.empty or "var_5_pct" not in mc.columns or "implied_return_mc" not in mc.columns:
        return create_no_data_figure("VaR Analysis — No Data")

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("VaR 5% Distribution", "VaR 5% vs Expected Upside"),
        vertical_spacing=0.12,
    )

    var_clipped = mc["var_5_pct"].clip(-150, 300)
    fig.add_trace(
        go.Histogram(
            x=var_clipped,
            nbinsx=80,
            marker_color=COLORS[3],
            opacity=0.75,
            name="VaR 5%",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="blue", row=1, col=1)

    sample = mc.sample(min(2000, len(mc)), random_state=42)
    fig.add_trace(
        go.Scatter(
            x=sample["var_5_pct"],
            y=sample["implied_return_mc"],
            mode="markers",
            marker=dict(
                size=4,
                color=sample["prob_positive_upside"],
                colorscale="RdYlGn",
                colorbar=dict(title="P(+)"),
                opacity=0.5,
            ),
            text=sample.get("name"),
            hovertemplate="%{text}<br>VaR 5%: %{x:.1f}%<br>Upside: %{y:.1f}%<extra></extra>",
            name="Stocks",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Value-at-Risk (5%) Analysis",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=False,
    )
    fig.update_xaxes(title_text="VaR 5% (%)", row=1, col=1)
    fig.update_xaxes(title_text="VaR 5% (%)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Expected Upside (%)", row=2, col=1)
    return fig


def create_beat_vs_achievement_scatter(beat: pd.DataFrame, pt: pd.DataFrame) -> go.Figure:
    """Scatter: P(Beat) vs P(Reach Price Target) coloured by return."""
    if beat.empty or pt.empty:
        return create_no_data_figure("Beat vs Achievement — Insufficient Data")

    merged = beat[["ticker", "posterior_beat_prob", "confidence_score"]].merge(
        pt[["ticker", "achievement_probability", "implied_return_pt"]],
        on="ticker",
        how="inner",
    )
    if merged.empty:
        return create_no_data_figure("Beat vs Achievement — No Overlapping Tickers")

    # v3.5: Support both 'ticker' and 'isin' as identifiers
    id_col = "ticker" if "ticker" in merged.columns else "isin"

    fig = px.scatter(
        merged,
        x="posterior_beat_prob",
        y="achievement_probability",
        color="implied_return_pt",
        size="confidence_score",
        hover_data=[id_col] if id_col in merged.columns else None,
        title="P(Beat Earnings) vs P(Reach Price Target)",
        labels={
            "posterior_beat_prob": "P(Beat Next Quarter)",
            "achievement_probability": "P(Reach Price Target)",
        },
        color_continuous_scale="RdYlGn",
        template=PLOTLY_TEMPLATE,
        height=500,
    )
    return fig


def create_model_dispersion_dashboard(summary: pd.DataFrame) -> go.Figure:
    """Four-panel dashboard showing inter-model agreement/dispersion analytics."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Inter-Model Dispersion Distribution",
            "Discrete vs Weighted Agreement",
            "Sector Consensus Rate (% All Bullish)",
            "Highest Model Disagreement Stocks",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    _catalog_return_cols = columns_for_viz("expected_returns")
    return_cols = [
        c
        for c in _catalog_return_cols
        if c in (
            "implied_return_mc",
            "price_target_mc",
            "implied_return_kalman",
            "price_target_kalman",
            "implied_return_pt",
        )
    ] or [
        "implied_return_mc",
        "price_target_mc",
        "implied_return_kalman",
        "price_target_kalman",
        "implied_return_pt",
    ]
    available = [c for c in return_cols if c in summary.columns]
    dispersion = pd.Series(dtype=float)

    if len(available) >= 2:
        returns_df = summary[available].dropna()
        row_mean = returns_df.mean(axis=1)
        dispersion = returns_df.sub(row_mean, axis=0).abs().mean(axis=1)

        fig.add_trace(
            go.Histogram(
                x=dispersion,
                nbinsx=60,
                marker_color=COLORS[0],
                opacity=0.75,
                name="Model Dispersion",
            ),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=dispersion.median(),
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {dispersion.median():.1f}",
            row=1,
            col=1,
        )

    if "agreement_score" in summary.columns and "weighted_agreement" in summary.columns:
        sample = summary.dropna(subset=["agreement_score", "weighted_agreement"]).sample(
            min(2000, len(summary)),
            random_state=42,
        )
        fig.add_trace(
            go.Scatter(
                x=sample["agreement_score"],
                y=sample["weighted_agreement"],
                mode="markers",
                marker=dict(size=4, opacity=0.4, color=COLORS[1]),
                name="Stocks",
                hovertemplate="Score: %{x}<br>Weighted: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    group_col = "industry" if "industry" in summary.columns else "sector"
    if group_col in summary.columns and "agreement_score" in summary.columns:
        consensus = (
            summary.groupby(group_col)["agreement_score"]
            .apply(lambda x: (x == 3).mean() * 100)
            .sort_values(ascending=True)
            .tail(20)
        )
        fig.add_trace(
            go.Bar(
                x=consensus.values,
                y=consensus.index.astype(str),
                orientation="h",
                marker_color=COLORS[2],
                name="% Full Consensus",
            ),
            row=2,
            col=1,
        )

    if len(available) >= 2 and "ticker" in summary.columns and not dispersion.empty:
        summary_disp = summary.copy()
        summary_disp["_dispersion"] = dispersion
        top_disagree = summary_disp.nlargest(15, "_dispersion")
        fig.add_trace(
            go.Bar(
                x=top_disagree["_dispersion"],
                y=top_disagree["ticker"],
                orientation="h",
                marker_color=COLORS[3],
                name="Dispersion",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title="Cross-Model Dispersion & Agreement Dashboard",
        template=PLOTLY_TEMPLATE,
        height=900,
        showlegend=False,
    )
    return fig


def create_return_distribution_fit_chart(mc: pd.DataFrame) -> go.Figure:
    """Overlay histogram of MC returns with fitted parametric distribution."""
    upside = mc["implied_return_mc"].dropna().clip(-100, 300)
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=upside,
            nbinsx=100,
            histnorm="probability density",
            marker_color=COLORS[0],
            opacity=0.6,
            name="Observed",
        )
    )

    x_range = np.linspace(upside.min(), upside.max(), 300)
    for dist, color, label in [
        (sp_stats.norm, COLORS[1], "Normal"),
        (sp_stats.t, COLORS[2], "Student-t"),
        (sp_stats.skewnorm, COLORS[3], "Skew-Normal"),
    ]:
        try:
            params = dist.fit(upside)
            pdf = dist.pdf(x_range, *params)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=pdf,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=label,
                )
            )
        except (ValueError, RuntimeError, TypeError):
            continue

    var_5 = float(np.percentile(upside, 5))
    cvar_5 = float(upside[upside <= var_5].mean()) if (upside <= var_5).any() else var_5
    fig.add_vline(
        x=var_5,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR 5%: {var_5:.1f}%",
    )
    fig.add_vline(
        x=cvar_5,
        line_dash="dot",
        line_color="darkred",
        annotation_text=f"CVaR 5%: {cvar_5:.1f}%",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="MC Return Distribution — Parametric Fit Overlay",
        xaxis_title="Expected Upside (%)",
        yaxis_title="Density",
        template=PLOTLY_TEMPLATE,
        height=550,
    )
    return fig


def create_sector_return_analytics_heatmap(sector_analytics: pd.DataFrame) -> go.Figure:
    """Enhanced sector heatmap with confidence intervals, consensus rates, and beat probs."""
    group_col = "industry" if "industry" in sector_analytics.columns else "sector"
    if sector_analytics.empty or group_col not in sector_analytics.columns:
        return create_no_data_figure("Sector Return Analytics — No Data")

    display_cols = [
        c
        for c in [
            "mc_mean",
            "mc_median",
            "kalman_mean",
            "pt_mean",
            "pct_full_consensus",
            "mean_weighted_agreement",
            "risk_adjusted_return",
            "mean_beat_prob",
        ]
        if c in sector_analytics.columns
    ]
    if not display_cols:
        return go.Figure()

    heatmap_data = sector_analytics.set_index(group_col)[display_cols]
    rename_map = {
        "mc_mean": "MC Mean %",
        "mc_median": "MC Median %",
        "kalman_mean": "Kalman Mean %",
        "pt_mean": "Achiev. Mean %",
        "pct_full_consensus": "Full Consensus %",
        "mean_weighted_agreement": "Wtd Agreement",
        "risk_adjusted_return": "Risk-Adj Return",
        "mean_beat_prob": "Mean P(Beat)",
    }
    heatmap_data = heatmap_data.rename(
        columns={k: v for k, v in rename_map.items() if k in heatmap_data.columns}
    )

    fig = px.imshow(
        heatmap_data.round(2),
        color_continuous_scale="RdYlGn",
        text_auto=True,
        aspect="auto",
        title="Sector Expected Returns — Enhanced Analytics Heatmap",
        labels={"color": "Value"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=max(600, len(sector_analytics) * 28))
    return fig


def create_screening_summary_chart(screens: dict[str, pd.DataFrame]) -> go.Figure:
    """Bar chart summarizing stock counts from each screening strategy."""
    names = []
    counts = []
    for name, df in screens.items():
        if not df.empty:
            names.append(name.replace("_", " ").title())
            counts.append(len(df))

    if not names:
        return create_no_data_figure("Stock Screening Results — No Data")

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker_color=COLORS[: len(names)],
            text=counts,
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Stock Screening Results Summary",
        xaxis_title="Number of Stocks Passing",
        template=PLOTLY_TEMPLATE,
        height=max(400, len(names) * 40),
    )
    return fig


def create_price_target_drift_dashboard(
    mv_equities_df: pd.DataFrame,
    mv_spec: Optional[EquitiesMaterializedViewSpec] = None,
) -> go.Figure:
    """Dashboard showing price target drift across historical snapshots.

    Uses ``mv_spec.price_target_columns`` (when available) to identify
    price-target horizons and creates a multi-line chart showing
    PT median/high/low drift over 1W/1M/3M/6M/1Y.

    Parameters
    ----------
    mv_equities_df : pd.DataFrame
        Data from mv_equities (or equivalent).
    mv_spec : EquitiesMaterializedViewSpec, optional
        Materialized view specification for column discovery.

    Returns
    -------
    go.Figure
    """
    if mv_equities_df.empty:
        return create_no_data_figure("Price Target Drift — No Data")

    # Discover price target columns
    if mv_spec is not None:
        pt_cols = mv_spec.price_target_columns
    else:
        pt_cols = [c for c in mv_equities_df.columns if c.startswith("price_target")]

    if not pt_cols:
        return create_no_data_figure("Price Target Drift — No PT Columns")

    # Identify horizon suffixes
    horizon_order = ["1w_ago", "1m_ago", "3m_ago", "6m_ago", "1y_ago"]
    horizon_labels = ["1W Ago", "1M Ago", "3M Ago", "6M Ago", "1Y Ago"]

    # Find the base PT column (current, no _ago suffix)
    base_cols = [c for c in pt_cols if not any(h in c for h in horizon_order)]
    if not base_cols:
        base_cols = pt_cols[:1]

    fig = go.Figure()
    color_idx = 0

    for base in base_cols[:5]:  # limit to 5 base metrics
        # Current value
        if base in mv_equities_df.columns:
            median_val = mv_equities_df[base].median()
            points = [("Current", median_val)]

            for suffix, label in zip(horizon_order, horizon_labels):
                hist_col = (
                    f"{base}_{suffix}" if f"{base}_{suffix}" in mv_equities_df.columns else None
                )
                if hist_col is None:
                    # Try matching pattern
                    candidates = [
                        c for c in pt_cols if base.replace("price_target", "") in c and suffix in c
                    ]
                    hist_col = candidates[0] if candidates else None
                if hist_col and hist_col in mv_equities_df.columns:
                    points.append((label, mv_equities_df[hist_col].median()))

            if len(points) > 1:
                labels, values = zip(*points)
                fig.add_trace(
                    go.Scatter(
                        x=list(labels),
                        y=list(values),
                        mode="lines+markers",
                        name=base.replace("_", " ").title(),
                        line=dict(color=COLORS[color_idx % len(COLORS)], width=2),
                        marker=dict(size=8),
                    )
                )
                color_idx += 1

    if not fig.data:
        return create_no_data_figure("Price Target Drift — Insufficient Historical Data")

    fig.update_layout(
        title="Price Target Drift Across Historical Snapshots",
        xaxis_title="Snapshot Horizon",
        yaxis_title="Median Price Target",
        template=PLOTLY_TEMPLATE,
        height=500,
    )
    return fig
