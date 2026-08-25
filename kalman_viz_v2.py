"""Figures and statistics tables for the Kalman price-target workflow, v2.

v2 shipped with no visualizations at all. Every chart in the post-run analysis
was therefore drawn by hand, and hand-drawn charts go stale silently: the
published decay ladder still describes run ``37e6d8966250`` and understates the
current trail asymptote by about 0.026, which is only known because someone
happened to re-read the gate. Everything here is generated from the run that
produced it, so a figure cannot describe a different fit than its caption claims.

What this module is *not*
-------------------------
It does not compute anything the workflow already gates. Every panel takes an
object the run has already produced -- ``idata``, ``KalmanPanelV2``, the §4b
audit dict, the §8 PPC dict, the §9 diagnostics frame, the screen, the risk book
-- and draws it. Recomputing a statistic beside the gate that owns it is how two
readings of "the same" number end up in one report; the discipline the post-run
skill states ("do not recompute its statistics by hand") applies to the pipeline
that feeds it too.

Where the panels come from
--------------------------
Three groups, and only the first mirrors v1:

**Workflow stages.** Prior predictive, posterior predictive, diagnostics, screen
and risk book, matching v1 section for section so the two workflows can be read
side by side.

**v2-only structure.** The decay ladder, the variance simplex, the ``sigma_time``
staleness calibration and the drift forest have no v1 equivalent, because v1 has
no correlated trail and no Dirichlet variance split. These are the panels that
show what v2 exists to model.

**Two questions the export could not answer.** ``plot_rank_correlations``
measures how far each exported column departs from a consensus sort -- the table
the post-run analysis rebuilds every edition -- and ``plot_er_sd_calibration``
scores the forward-return second moment against realised volatility, which is
what gives ``tail_risk_vol_floor_k`` a measured rationale rather than a value.

Payload budget
--------------
Non-negotiable, and enforced by using the shared primitives rather than by
review: pre-binned densities (:func:`_add_binned_density`), gridded ECDFs
(:func:`_ecdf_xy`), decimated scatters (:func:`_decimate_frame`) with the
sampled count in the title and summary statistics computed on the FULL frame,
and :func:`_azp_backend` for every arviz-plots call. A v1 notebook reached
233 MB -- 207.7 MB of it one prior-predictive figure -- by ignoring exactly
these. Per-point hover identity strings are left off deliberately: they are the
bulk of a large scatter's payload and unreadable in a dense cloud.

Usage
-----
.. code-block:: python

    import kalman_viz_v2 as viz

    viz.install(run_cfg)                      # once, before any panel
    with viz.section('08_ppc'):
        viz.plot_ppc_overlay(ppc, panel)

``install`` points the shared figure layer at ``KalmanRunConfigV2`` and turns
artifact export on. Every panel is safe to call with missing inputs -- it
returns ``None`` and logs -- because a figure failure must never cost a run the
export it has already paid a fit for.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import pandas as pd

from probabilistic_ml_model.visualizations.kalman_shared import (
    C_ACCENT, C_FORECAST, C_MUTED, C_OBSERVED, C_POSTERIOR, C_REF,
    CS_DIV, CS_SEQ,
    _PPC_ECDF_GRID, _SCREEN_SCATTER_MAX_POINTS,
    _add_binned_density, _add_ref_line, _azp_backend,
    _decimate_frame, _ecdf_xy, _export_path, _fmt_axis, _forest_height_px,
    _hex_to_rgba, _next_stem, _render_plotly, _safe_show,
    enable_artifact_export, export_section, get_export_state,
    set_viz_config_resolver, setup_plotting,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover - optional dependency
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]
    _HAS_PLOTLY = False

logger = logging.getLogger(__name__)

#: Panel heights, in px. Named rather than inlined so a row of panels cannot
#: drift a hundred pixels apart across sections.
H_SHORT = 380
H_PANEL = 520
H_TALL = 700

#: Marker edge colour on the dark panel background.
C_PANEL_EDGE = "#1e1e1e"

#: Draws to thin a posterior to before a per-name scatter. The panels below plot
#: posterior *summaries* per name, so the full ``chain x draw`` grid buys
#: nothing; where a raw draw cloud is genuinely wanted it is binned instead.
_MAX_SCATTER_DRAWS = 400

__all__ = [
    "render_run",
    "install",
    "section",
    "write_table",
    # §4b
    "plot_panel_audit",
    "plot_decay_ladder",
    # §6
    "plot_prior_predictive",
    # §8
    "plot_ppc_overlay",
    "plot_ppc_calibration",
    "plot_ppc_decay",
    # §9
    "plot_rhat_ess",
    "plot_trace_worst",
    "plot_energy",
    "plot_variance_legs",
    "plot_sigma_time_calibration",
    "plot_drift_forest",
    # §10 / §10b
    "plot_screen_overview",
    "plot_rank_correlations",
    "plot_er_sd_calibration",
    "plot_risk_book",
]


# =========================================================================== #
#  Wiring                                                                     #
# =========================================================================== #
def install(run_cfg: Any, *, enable: bool = True) -> None:
    """Point the shared figure layer at this run and turn artifact export on.

    Parameters
    ----------
    run_cfg
        A :class:`KalmanRunConfigV2`. Read for ``fig_width_px``, ``random_seed``
        and ``results_dir``.
    enable
        ``False`` installs the resolver but leaves export off, which is what a
        ``--no-figures`` run wants: panels called anyway become cheap no-ops
        rather than needing a guard at every call site.

    Notes
    -----
    The resolver is installed as a CALLABLE returning ``run_cfg``, matching how
    v1 hands over ``get_run_config``. ``KalmanRunConfigV2`` is frozen, so a
    snapshot would be safe here -- but keeping both workflows on the same
    mechanism means there is one contract to remember, not two.
    """
    set_viz_config_resolver(lambda: run_cfg)
    setup_plotting()
    enable_artifact_export(enable)


@contextmanager
def section(label: str):
    """Scope artifact filenames to a workflow section (see ``export_section``)."""
    with export_section(label):
        yield


def write_table(frame: pd.DataFrame, stem: str) -> Optional[str]:
    """Write a statistics table beside the figure it explains.

    Every panel that carries a number exports the table behind it, so the
    post-run extractor can read the value instead of re-deriving it -- and so a
    number in a caption can be checked against the frame that produced it.

    Returns
    -------
    str or None
        The written path, or ``None`` when export is off or the frame is empty.
    """
    if frame is None or not len(frame) or not get_export_state().enabled:
        return None
    try:
        path = _export_path(_next_stem(stem), "csv")
        frame.to_csv(path, index=False)
        logger.info("wrote %s (%d rows)", path, len(frame))
        return str(path)
    except Exception as exc:  # pragma: no cover - export is best-effort
        logger.warning("table export failed for %s: %s", stem, exc)
        return None


def _post(idata: Any) -> Any:
    """Return ``idata.posterior`` or ``None``, without raising on a bad handle."""
    try:
        return idata.posterior
    except Exception:  # pragma: no cover - defensive
        return None


def _flat(post: Any, name: str) -> Optional[np.ndarray]:
    """Return a posterior variable flattened over (chain, draw), or ``None``."""
    if post is None or name not in post:
        return None
    try:
        arr = np.asarray(post[name])
        return arr.reshape(-1, *arr.shape[2:]) if arr.ndim >= 2 else arr.ravel()
    except Exception:  # pragma: no cover - defensive
        return None


def _requires_plotly(fn_name: str) -> bool:
    if not _HAS_PLOTLY:
        logger.warning("%s skipped: plotly is not installed", fn_name)
        return False
    return True


def _decay_curve(rho_inf: float, ell: float, gaps: np.ndarray) -> np.ndarray:
    """The model's own kernel ``r(d) = rho_inf + (1 - rho_inf) exp(-d / ell)``."""
    ell = max(float(ell), 1e-9)
    return rho_inf + (1.0 - rho_inf) * np.exp(-np.asarray(gaps, dtype="float64") / ell)


# =========================================================================== #
#  §4b  Panel information audit                                               #
# =========================================================================== #
def plot_panel_audit(panel: Any, audit: dict) -> Optional[Any]:
    """Draw what the panel carries before anything is fitted.

    Three questions in one figure, all answered by :func:`run_panel_diagnostics`
    and none of them by v1: how many independent observations each name really
    has (``T_eff``), how the response correlates across the lookback grid, and
    how much of the grid is observed at all.

    ``T_eff`` is the one to read first. Close to 1 means the trail is a
    near-duplicate of the snapshot, and a likelihood treating it as ``T``
    independent reads over-counts the evidence -- which is the failure this whole
    stage exists to catch.
    """
    if not _requires_plotly("plot_panel_audit"):
        return None
    corr = np.asarray(audit.get("corr", np.empty((0, 0))), dtype="float64")
    cov = np.asarray(audit.get("per_step_coverage", []), dtype="float64")
    if corr.size == 0:
        logger.warning("plot_panel_audit skipped: no correlation matrix in the audit")
        return None

    labels = [f"{int(d)}d" for d in np.asarray(panel.time_days, dtype="float64")]
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.13,
        column_widths=[0.55, 0.45],
        subplot_titles=(
            "Response correlation across the lookback grid",
            "Share of names observed at each lookback",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=corr, x=labels, y=labels, colorscale=CS_DIV, zmid=0.0, zmin=-1.0, zmax=1.0,
            colorbar=dict(title="r", len=0.9, x=0.46),
            hovertemplate="r(%{y}, %{x}) = %{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    if cov.size:
        fig.add_trace(
            go.Bar(
                x=labels[: cov.size], y=cov * 100.0,
                marker=dict(color=_hex_to_rgba(C_POSTERIOR, 0.85)),
                hovertemplate="%{x}: %{y:.1f}% observed<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )
        _fmt_axis(fig, y="names observed (%)", y_kind="pct", row=1, col=2)
        fig.update_yaxes(range=[0, 105], row=1, col=2)

    t_eff = float(audit.get("t_eff", float("nan")))
    fig.update_layout(
        title=(
            f"Panel information audit — T_eff {t_eff:.2f} of T = {panel.n_time}"
            f"  ·  {panel.n_isin:,} names"
        ),
    )
    _render_plotly(fig, height=H_PANEL, label="04b_panel_audit")

    write_table(
        pd.DataFrame(
            {
                "lookback": labels[: cov.size] if cov.size else labels,
                "gap_days": np.asarray(panel.time_days, dtype="float64")[
                    : cov.size if cov.size else len(labels)
                ],
                "share_observed": cov if cov.size else np.full(len(labels), np.nan),
            }
        ),
        "04b_panel_coverage_table",
    )
    return fig


def plot_decay_ladder(
    panel: Any,
    audit: dict,
    idata: Optional[Any] = None,
    *,
    residual_kernel: Optional[dict] = None,
) -> Optional[Any]:
    """The measurement v2 exists because of, drawn from the run that produced it.

    The correlation between a name's log-uplift at two lookbacks decays with the
    calendar gap toward a non-zero asymptote. That asymptote -- ``rho_inf`` -- is
    a permanent per-name level the v1 factorised likelihood could not see, and
    splitting it from the decaying state is the entire premise of the redesign.

    Three curves, and the gap between them is the point:

    * **Raw trail** -- the empirical kernel on the response panel, from the
      §4b audit. Its asymptote is large (~0.42-0.45).
    * **After the fitted mean** -- the same kernel on the residual. ``mu_reg`` is
      constant in ``t``, and a between-name quantity that does not vary with time
      is exactly what a permanent level is, so the two are confounded up to the
      prior. Most of the asymptote goes here.
    * **Posterior** -- ``rho_inf_implied``, which has read 0.005 on five
      consecutive fits while the raw asymptote moved by 0.037.

    Passing ``idata`` adds the posterior curve; ``residual_kernel`` (the
    ``decay_residual`` entry of the §8 PPC dict, or any ``{'rho_inf', 'ell_days'}``
    mapping) adds the middle one. Both are optional -- the raw curve alone is
    still the honest version of the published figure.
    """
    if not _requires_plotly("plot_decay_ladder"):
        return None
    corr = np.asarray(audit.get("corr", np.empty((0, 0))), dtype="float64")
    days = np.asarray(panel.time_days, dtype="float64")
    if corr.size == 0 or days.size != corr.shape[0]:
        logger.warning("plot_decay_ladder skipped: audit carries no usable corr matrix")
        return None

    # The measured points: every unordered pair of lookbacks, at its calendar gap.
    iu, ju = np.triu_indices(corr.shape[0], k=1)
    gaps = np.abs(days[iu] - days[ju])
    vals = corr[iu, ju]
    order = np.argsort(gaps)
    gaps, vals = gaps[order], vals[order]

    grid = np.linspace(0.0, float(max(gaps.max(), 1.0)) * 1.05, 200)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gaps, y=vals, mode="markers", name="measured trail correlation",
            marker=dict(color=C_OBSERVED, size=10, line=dict(width=1, color=C_PANEL_EDGE)),
            hovertemplate="gap %{x:.0f}d → r = %{y:.3f}<extra></extra>",
        )
    )

    rows: list[dict] = []
    raw_rho = float(audit.get("rho_inf", float("nan")))
    raw_ell = float(audit.get("ell_days", float("nan")))
    if np.isfinite(raw_rho) and np.isfinite(raw_ell):
        fig.add_trace(
            go.Scatter(
                x=grid, y=_decay_curve(raw_rho, raw_ell, grid), mode="lines",
                name=f"raw trail · rho_inf {raw_rho:.3f}, ell {raw_ell:.0f}d",
                line=dict(color=C_OBSERVED, width=2.4),
                hovertemplate="raw: r(%{x:.0f}d) = %{y:.3f}<extra></extra>",
            )
        )
        _add_ref_line(fig, y=raw_rho, kind="anchor",
                      annotation_text=f"raw asymptote {raw_rho:.3f}")
        rows.append({"curve": "raw_trail", "rho_inf": raw_rho, "ell_days": raw_ell,
                     "source": "panel audit (this run)"})

    if residual_kernel:
        r_rho = float(residual_kernel.get("observed_rho_inf",
                                          residual_kernel.get("rho_inf", np.nan)))
        r_ell = float(residual_kernel.get("observed_ell_days",
                                          residual_kernel.get("ell_days", np.nan)))
        if np.isfinite(r_rho) and np.isfinite(r_ell):
            fig.add_trace(
                go.Scatter(
                    x=grid, y=_decay_curve(r_rho, r_ell, grid), mode="lines",
                    name=f"after the fitted mean · rho_inf {r_rho:.3f}, ell {r_ell:.0f}d",
                    line=dict(color=C_FORECAST, width=2.4, dash="dash"),
                    hovertemplate="residual: r(%{x:.0f}d) = %{y:.3f}<extra></extra>",
                )
            )
            rows.append({"curve": "after_fitted_mean", "rho_inf": r_rho,
                         "ell_days": r_ell, "source": "ppc decay_residual (this run)"})

    post = _post(idata) if idata is not None else None
    p_rho = _flat(post, "rho_inf_implied")
    p_ell = _flat(post, "ou_length_scale_days")
    if p_rho is not None and p_ell is not None and p_rho.size and p_ell.size:
        pr, pe = float(np.mean(p_rho)), float(np.mean(p_ell))
        fig.add_trace(
            go.Scatter(
                x=grid, y=_decay_curve(pr, pe, grid), mode="lines",
                name=f"posterior · rho_inf {pr:.4f}, ell {pe:.0f}d",
                line=dict(color=C_POSTERIOR, width=2.8),
                hovertemplate="posterior: r(%{x:.0f}d) = %{y:.3f}<extra></extra>",
            )
        )
        rows.append({"curve": "posterior", "rho_inf": pr, "ell_days": pe,
                     "source": "rho_inf_implied / ou_length_scale_days"})

    _add_ref_line(fig, y=0.0, kind="zero")
    _fmt_axis(fig, x="calendar gap between lookbacks (days)", y="correlation")
    fig.update_yaxes(range=[-0.05, 1.02])
    fig.update_layout(
        title=(
            "The decay ladder — how much of the trail correlation is a "
            "permanent level<br><sub>Points are measured at the "
            f"{gaps.size} calendar gaps this grid provides. Curves are "
            "r(d) = rho_inf + (1 − rho_inf)·exp(−d/ell).</sub>"
        ),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    _render_plotly(fig, height=H_PANEL, label="04b_decay_ladder")

    measured = pd.DataFrame({"gap_days": gaps, "correlation": vals})
    write_table(measured, "04b_decay_measured_table")
    if rows:
        write_table(pd.DataFrame(rows), "04b_decay_kernels_table")
    return fig


# =========================================================================== #
#  §6  Prior predictive                                                       #
# =========================================================================== #
def plot_prior_predictive(prior_idata: Any, panel: Any) -> Optional[Any]:
    """Check the prior on the scale a person can judge, not in standardised units.

    The gate (``prior_scale``) reduces this to one ratio: the prior 90% upside
    width over the empirical one, wanting 1x-10x. That number moved 4.8x -> 7.1x
    between two runs *without the prior changing* -- the empirical denominator
    moved with the data refresh. A ratio whose denominator moves can leave its
    band with nothing wrong with the model, and the only way to see which half
    moved is to draw both distributions.

    De-standardises through ``expm1`` with the same log-space clip the decision
    layer uses. Clipping before ``expm1`` rather than after is load-bearing: a
    Student-t prior with a free scale produces draws whose exponential overflows
    to ``inf``, and a handful of infinities otherwise define the percentiles.
    """
    if not _requires_plotly("plot_prior_predictive"):
        return None
    try:
        pp = prior_idata.prior_predictive
        key = next((k for k in pp.data_vars if str(k).startswith("target_pct_obs")), None)
        if key is None:
            raise KeyError("no target_pct_obs* variable in the prior predictive")
        rep = np.asarray(pp[key]).ravel()
    except Exception as exc:
        logger.warning("plot_prior_predictive skipped: %s", exc)
        return None

    lo_clip, hi_clip = -5.0, 5.0  # matches LOG_UPLIFT_CLIP_{LO,HI}
    prior_up = np.expm1(np.clip(rep * panel.response_std + panel.response_mean,
                                lo_clip, hi_clip))
    obs_std = panel.Y[np.isfinite(panel.Y)]
    emp_up = np.expm1(np.clip(obs_std * panel.response_std + panel.response_mean,
                              lo_clip, hi_clip))

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.10,
        subplot_titles=("Prior vs empirical implied upside",
                        "Prior scale parameters"),
    )
    # PRE-BINNED. As raw go.Histogram traces this single figure serialised
    # prior_draws x n_isin float64 three times over and weighed 207.7 MB in the
    # v1 notebook, against ~120 KB for a PNG of the same figure.
    _add_binned_density(
        fig, prior_up * 100.0, row=1, col=1, bins=80, clip=(-100.0, 300.0),
        color=C_POSTERIOR, alpha=0.6, name="prior",
        hovertemplate="prior upside = %{x:.0f}%<extra></extra>",
    )
    _add_binned_density(
        fig, emp_up * 100.0, row=1, col=1, bins=80, clip=(-100.0, 300.0),
        color=C_OBSERVED, fill=False, name="empirical",
        hovertemplate="empirical upside = %{x:.0f}%<extra></extra>",
    )
    _add_ref_line(fig, x=0, kind="zero", row=1, col=1)
    _fmt_axis(fig, x="implied upside (%)", y="density", x_kind="pct", row=1, col=1)

    for name, colour in (("sigma_level", C_ACCENT), ("sigma_state", C_FORECAST)):
        vals = _flat(getattr(prior_idata, "prior", None), name)
        if vals is not None and vals.size:
            v = np.asarray(vals, dtype="float64").ravel()
            v = v[np.isfinite(v)]
            if v.size:
                _add_binned_density(
                    fig, v, row=1, col=2, bins=60, color=colour, alpha=0.6, name=name,
                    clip=(0.0, float(np.nanpercentile(v, 99))),
                    hovertemplate=f"{name} = %{{x:.3f}}<extra></extra>",
                )
    _fmt_axis(fig, x="standardised scale", y="density", row=1, col=2)

    p_lo, p_hi = np.nanpercentile(prior_up, [5, 95])
    o_lo, o_hi = np.nanpercentile(emp_up, [5, 95])
    ratio = (p_hi - p_lo) / max(o_hi - o_lo, 1e-12)
    fig.update_layout(
        title=(
            f"Prior predictive — prior/empirical 90% width {ratio:.1f}x "
            "<sub>(gate wants 1x–10x)</sub>"
        ),
        legend=dict(font_size=11),
    )
    _render_plotly(fig, height=H_SHORT, label="06_prior_predictive")

    write_table(
        pd.DataFrame(
            [
                {"series": "prior", "p05": p_lo, "p50": float(np.nanmedian(prior_up)),
                 "p95": p_hi, "width_90": p_hi - p_lo, "n": int(prior_up.size)},
                {"series": "empirical", "p05": o_lo, "p50": float(np.nanmedian(emp_up)),
                 "p95": o_hi, "width_90": o_hi - o_lo, "n": int(emp_up.size)},
                {"series": "ratio", "p05": np.nan, "p50": np.nan, "p95": np.nan,
                 "width_90": ratio, "n": np.nan},
            ]
        ),
        "06_prior_scale_table",
    )
    return fig


# =========================================================================== #
#  §8  Posterior predictive                                                   #
# =========================================================================== #
def plot_ppc_overlay(ppc_out: dict, panel: Any, ppc_idata: Optional[Any] = None) -> Optional[Any]:
    """Replicated vs observed response: a density overlay and a gridded ECDF.

    The ECDF is the half that earns its place. A density overlay hides exactly
    the disagreement the ``ppc_t_spread`` gate exists to catch -- two
    distributions with visibly similar peaks can differ by 5% in IQR -- and the
    cumulative view puts the gap where a reader can measure it.

    Drawn on :data:`_PPC_ECDF_GRID` points, not one per observation: the same
    curve at full resolution measured 42.2 MB against 0.86 MB here.
    """
    if not _requires_plotly("plot_ppc_overlay"):
        return None
    obs = panel.Y[panel.observed_mask]
    obs = obs[np.isfinite(obs)]
    rep = None
    if ppc_idata is not None:
        try:
            pp = ppc_idata.posterior_predictive
            key = next((k for k in pp.data_vars if str(k).startswith("target_pct_obs")), None)
            if key is not None:
                rep = np.asarray(pp[key]).ravel()
                rep = rep[np.isfinite(rep)]
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.debug("ppc replicates unavailable for the overlay: %s", exc)
    if obs.size == 0:
        logger.warning("plot_ppc_overlay skipped: no finite observations")
        return None

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.10,
        subplot_titles=("Density — replicated vs observed",
                        "ECDF — where the two actually differ"),
    )
    _add_binned_density(fig, obs, row=1, col=1, bins=80, color=C_OBSERVED,
                        fill=False, name="observed",
                        hovertemplate="observed = %{x:.2f}<extra></extra>")
    if rep is not None and rep.size:
        _add_binned_density(fig, rep, row=1, col=1, bins=80, color=C_POSTERIOR,
                            alpha=0.5, name="replicated",
                            hovertemplate="replicated = %{x:.2f}<extra></extra>")
    _fmt_axis(fig, x="standardised log-uplift", y="density", row=1, col=1)

    ox, oy = _ecdf_xy(obs, n=_PPC_ECDF_GRID)
    fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", name="observed ECDF",
                             line=dict(color=C_OBSERVED, width=2.2),
                             showlegend=False), row=1, col=2)
    if rep is not None and rep.size:
        rx, ry = _ecdf_xy(rep, n=_PPC_ECDF_GRID)
        fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", name="replicated ECDF",
                                 line=dict(color=C_POSTERIOR, width=2.2),
                                 showlegend=False), row=1, col=2)
    _fmt_axis(fig, x="standardised log-uplift", y="F(x)", y_kind="prob", row=1, col=2)

    ts = ppc_out.get("t_spread", {}) or {}
    obs_iqr = ts.get("observed_iqr", float("nan"))
    fig.update_layout(
        title=(
            f"Posterior predictive — observed IQR {obs_iqr:.3f} in "
            f"[{ts.get('lo', float('nan')):.3f}, {ts.get('hi', float('nan')):.3f}]"
            f"  ·  {obs.size:,} observed cells"
        )
    )
    _render_plotly(fig, height=H_SHORT, label="08_ppc_overlay")
    return fig


def plot_ppc_calibration(ppc_out: dict, panel: Any, run_cfg: Any) -> Optional[Any]:
    """Per-lookback coverage against its target.

    Over-shooting is a failure, not a safety margin: a model whose 94% interval
    contains 99% of the data is not being careful, it is being wide, and every
    downstream risk column inherits the width. The band is the tolerance the
    ``ppc_coverage`` gate applies.
    """
    if not _requires_plotly("plot_ppc_calibration"):
        return None
    cov = np.asarray(ppc_out.get("coverage", []), dtype="float64")
    if cov.size == 0:
        logger.warning("plot_ppc_calibration skipped: no coverage in the ppc dict")
        return None
    days = np.asarray(panel.time_days, dtype="float64")[: cov.size]
    labels = [f"{int(d)}d" for d in days]
    target = float(getattr(run_cfg, "gate_coverage_target", 0.94))
    tol = float(getattr(run_cfg, "gate_coverage_tol", 0.02))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=cov, marker=dict(color=_hex_to_rgba(C_POSTERIOR, 0.85)),
        hovertemplate="%{x}: %{y:.3f} covered<extra></extra>", showlegend=False,
    ))
    _add_ref_line(fig, y=target, kind="anchor", annotation_text=f"target {target:.2f}")
    _add_ref_line(fig, y=target + tol, kind="emphasis")
    _add_ref_line(fig, y=target - tol, kind="emphasis")
    _fmt_axis(fig, x="lookback (gap to now)", y="share of cells inside the interval",
              y_kind="prob")
    fig.update_yaxes(range=[min(0.80, float(cov.min()) - 0.02), 1.0])
    worst = float(np.max(np.abs(cov - target)))
    fig.update_layout(
        title=(
            f"Per-lookback predictive coverage — worst deviation {worst:.3f} "
            f"<sub>(tolerance ±{tol})</sub>"
        )
    )
    _render_plotly(fig, height=H_SHORT, label="08_ppc_coverage")

    rows = [{"lookback": lab, "gap_days": float(d), "coverage": float(c),
             "target": target, "deviation": float(c - target)}
            for lab, d, c in zip(labels, days, cov)]
    ts = ppc_out.get("t_spread", {}) or {}
    for k, v in ts.items():
        rows.append({"lookback": f"t_spread.{k}", "gap_days": np.nan,
                     "coverage": float(v) if np.isscalar(v) else np.nan,
                     "target": np.nan, "deviation": np.nan})
    write_table(pd.DataFrame(rows), "08_ppc_calibration_table")
    return fig


def plot_ppc_decay(ppc_out: dict) -> Optional[Any]:
    """The statistic no v1 gate could see.

    A factorised likelihood can match the marginal variance of a panel exactly
    while producing replicates with no time structure at all -- which is what v1
    did. This compares the replicated ``rho_inf`` interval against the observed
    value, on the raw response and again on the residual.

    Reading the pair: failing both points at the covariance; passing the residual
    while the raw fails points at the mean.
    """
    if not _requires_plotly("plot_ppc_decay"):
        return None
    parts = [(k, ppc_out.get(k)) for k in ("decay", "decay_residual")]
    parts = [(k, v) for k, v in parts if isinstance(v, dict) and v]
    if not parts:
        logger.warning("plot_ppc_decay skipped: no decay entries in the ppc dict")
        return None

    fig = go.Figure()
    rows = []
    for i, (name, d) in enumerate(parts):
        lo, hi = float(d["replicated_lo"]), float(d["replicated_hi"])
        obs = float(d["observed_rho_inf"])
        label = "raw response" if name == "decay" else "residual (mean removed)"
        colour = C_POSTERIOR if name == "decay" else C_FORECAST
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[i, i], mode="lines",
            line=dict(color=_hex_to_rgba(colour, 0.55), width=14),
            name=f"{label} — replicated 94%",
            hovertemplate=f"{label}: replicated [{lo:.3f}, {hi:.3f}]<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[obs], y=[i], mode="markers", showlegend=False,
            marker=dict(color=C_OBSERVED, size=15, symbol="diamond",
                        line=dict(width=1.4, color=C_PANEL_EDGE)),
            hovertemplate=f"{label}: observed {obs:.3f}<extra></extra>",
        ))
        rows.append({"statistic": name, "observed_rho_inf": obs,
                     "replicated_lo": lo, "replicated_hi": hi,
                     "inside": bool(lo <= obs <= hi),
                     "observed_ell_days": float(d.get("observed_ell_days", np.nan)),
                     "n_replicates": float(d.get("n_replicates", np.nan))})

    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(parts))),
        ticktext=["raw response" if k == "decay" else "residual" for k, _ in parts],
        range=[-0.6, len(parts) - 0.4],
    )
    _add_ref_line(fig, x=0.0, kind="zero")
    _fmt_axis(fig, x="rho_inf (permanent share of trail correlation)")
    share = ppc_out.get("decay_mean_share")
    sub = f"  ·  mean explains {share:.0%} of the raw asymptote" \
        if isinstance(share, (int, float)) and np.isfinite(share) else ""
    fig.update_layout(
        title=f"Correlation decay — observed against replicated{sub}",
        legend=dict(font_size=11),
    )
    _render_plotly(fig, height=H_SHORT, label="08_ppc_decay")
    write_table(pd.DataFrame(rows), "08_ppc_decay_table")
    return fig


# =========================================================================== #
#  §9  Diagnostics                                                            #
# =========================================================================== #
def plot_rhat_ess(diag: pd.DataFrame, run_cfg: Any) -> Optional[Any]:
    """Every monitored global against the two gates, with the margin visible.

    A gate verdict says pass or fail; this says by how much and on which
    parameter. Both matter: the worst-R-hat parameter has been a different one on
    each of three fits of one specification, which is the shape of a statistic
    with nothing left to report -- and that is only legible when the whole
    distribution is drawn rather than its maximum.
    """
    if not _requires_plotly("plot_rhat_ess") or diag is None or not len(diag):
        return None
    df = diag.reset_index() if diag.index.name else diag.copy()
    name_col = next((c for c in ("index", "parameter", "var_name") if c in df.columns), None)
    if name_col is None or "r_hat" not in df.columns or "ess_bulk" not in df.columns:
        logger.warning("plot_rhat_ess skipped: diagnostics frame lacks r_hat/ess_bulk")
        return None
    d = df[[name_col, "r_hat", "ess_bulk"]].dropna()
    if not len(d):
        logger.warning("plot_rhat_ess skipped: no finite r_hat/ess_bulk rows")
        return None

    rhat_gate = float(getattr(run_cfg, "gate_r_hat_max", 1.01))
    ess_gate = float(getattr(run_cfg, "gate_ess_min", 400))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["r_hat"], y=d["ess_bulk"], mode="markers", showlegend=False,
        marker=dict(color=d["ess_bulk"], colorscale=CS_SEQ, size=9,
                    line=dict(width=0.8, color=C_PANEL_EDGE),
                    colorbar=dict(title="ESS bulk")),
        text=d[name_col].astype(str),
        hovertemplate="%{text}<br>R-hat %{x:.4f} · ESS %{y:,.0f}<extra></extra>",
    ))
    _add_ref_line(fig, x=rhat_gate, kind="anchor", annotation_text=f"R-hat gate {rhat_gate}")
    _add_ref_line(fig, y=ess_gate, kind="anchor", annotation_text=f"ESS gate {ess_gate:.0f}")
    _fmt_axis(fig, x="R-hat", y="bulk ESS")
    fig.update_yaxes(type="log")
    worst = d.loc[d["r_hat"].idxmax()]
    thin = d.loc[d["ess_bulk"].idxmin()]
    fig.update_layout(
        title=(
            f"Convergence — worst R-hat {worst['r_hat']:.4f} "
            f"({worst[name_col]}), lowest ESS {thin['ess_bulk']:,.0f} "
            f"({thin[name_col]}) over {len(d)} globals"
        )
    )
    _render_plotly(fig, height=H_PANEL, label="09_rhat_ess")

    write_table(
        d.assign(
            rhat_margin=rhat_gate - d["r_hat"],
            ess_margin_x=d["ess_bulk"] / max(ess_gate, 1.0),
        ).sort_values("r_hat", ascending=False).head(25),
        "09_diagnostics_worst_table",
    )
    return fig


def plot_trace_worst(idata: Any, diag: pd.DataFrame, *, n: int = 4) -> Optional[Any]:
    """Trace and rank plots for the parameters actually closest to their gate.

    Selected from the §9 frame rather than by a hand-kept name list, so the panel
    follows the model instead of needing an edit each time the binding parameter
    changes -- which it has on every fit.

    Routed through the HEAVY backend. These fan one facet per vector element and
    draw every chain's full draw sequence in each; as Plotly they are tens of
    megabytes of JSON for a panel that gets read, not interrogated.
    """
    if diag is None or not len(diag):
        return None
    try:
        import arviz_plots as azp
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("plot_trace_worst skipped: arviz_plots is unavailable")
        return None
    df = diag.reset_index() if diag.index.name else diag.copy()
    name_col = next((c for c in ("index", "parameter", "var_name") if c in df.columns), None)
    if name_col is None or "ess_bulk" not in df.columns:
        return None
    post = _post(idata)
    if post is None:
        return None

    # Base names: a diagnostics row may be `beta[feat_x]`, and the posterior
    # holds `beta`. Deduplicate so one wide vector does not consume every slot.
    picks: list[str] = []
    for raw in df.sort_values("ess_bulk").loc[:, name_col].astype(str):
        base = raw.split("[", 1)[0]
        if base in post and base not in picks:
            picks.append(base)
        if len(picks) >= n:
            break
    if not picks:
        return None
    try:
        pc = azp.plot_trace(idata, var_names=picks, backend=_azp_backend(heavy=True))
        _safe_show(pc, label="09_trace_worst")
        return pc
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("trace panel unavailable: %s", exc)
        return None


def plot_energy(idata: Any) -> Optional[Any]:
    """Energy plot: the geometry check R-hat and ESS cannot make.

    Zero divergences with a marginal energy distribution far heavier than the
    transition distribution is the signature of a posterior NUTS is exploring
    slowly rather than badly -- exactly the state five v2 runs have reported.
    """
    try:
        import arviz_plots as azp
    except ImportError:  # pragma: no cover
        return None
    try:
        pc = azp.plot_energy(idata, backend=_azp_backend(heavy=True))
        _safe_show(pc, label="09_energy")
        return pc
    except Exception as exc:  # pragma: no cover - sample_stats may be absent
        logger.warning("energy panel unavailable: %s", exc)
        return None


def plot_variance_legs(idata: Any) -> Optional[Any]:
    """The Dirichlet split of one total scale into level, state and observation.

    This is recommendation 05 drawn. ``w_level`` has read 0.55%-0.60% on five
    consecutive fits, near zero and confidently rather than uncertainly -- and it
    did not follow the empirical trail asymptote when that moved by 0.037, which
    is the evidence that the near-zero posterior is a property of the model's
    decomposition rather than of one snapshot.

    The intervals are the point. A leg that is small AND tight is a different
    claim from a leg that is small and uncertain, and only the second is an
    argument for more data.
    """
    if not _requires_plotly("plot_variance_legs"):
        return None
    post = _post(idata)
    w = _flat(post, "variance_weights")
    if w is None or w.ndim != 2 or w.shape[1] < 3:
        logger.warning("plot_variance_legs skipped: variance_weights absent")
        return None
    names = ["level", "state", "observation"]
    mean = w.mean(axis=0)
    lo = np.percentile(w, 5.5, axis=0)
    hi = np.percentile(w, 94.5, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=mean * 100.0,
        marker=dict(color=[_hex_to_rgba(c, 0.85) for c in (C_ACCENT, C_POSTERIOR, C_MUTED)]),
        error_y=dict(type="data", symmetric=False,
                     array=(hi - mean) * 100.0, arrayminus=(mean - lo) * 100.0,
                     color=C_REF, thickness=1.6),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>", showlegend=False,
    ))
    _fmt_axis(fig, x="variance component", y="share of response variance (%)", y_kind="pct")
    fig.update_yaxes(type="log")

    extras = []
    for nm in ("sigma_total", "ou_length_scale_days", "obs_share", "nu_tail",
               "rho_inf_implied"):
        v = _flat(post, nm)
        if v is not None and v.size:
            extras.append(f"{nm} {float(np.mean(v)):.4g}")
    fig.update_layout(
        title=(
            f"Variance legs — level carries {mean[0] * 100:.2f}% "
            f"<sub>89% ETI [{lo[0] * 100:.2f}%, {hi[0] * 100:.2f}%]</sub>"
            + (f"<br><sub>{' · '.join(extras)}</sub>" if extras else "")
        )
    )
    _render_plotly(fig, height=H_SHORT, label="09_variance_legs")

    rows = [{"component": n, "mean": float(m), "eti89_lo": float(a), "eti89_hi": float(b)}
            for n, m, a, b in zip(names, mean, lo, hi)]
    for nm in ("sigma_total", "sigma_level", "sigma_state", "sigma_obs_base",
               "ou_length_scale_days", "obs_share", "nu_tail", "rho_inf_implied",
               "signal_exponent"):
        v = _flat(post, nm)
        if v is not None and v.size:
            vv = np.asarray(v, dtype="float64").ravel()
            rows.append({"component": nm, "mean": float(vv.mean()),
                         "eti89_lo": float(np.percentile(vv, 5.5)),
                         "eti89_hi": float(np.percentile(vv, 94.5))})
    write_table(pd.DataFrame(rows), "09_variance_legs_table")
    return fig


def plot_sigma_time_calibration(idata: Any, panel: Any) -> Optional[Any]:
    """Per-lookback scale against measured staleness.

    ``sigma_time`` is free per column, and the question is whether it tracks how
    much *worse* a stale column actually is. The reference is an ordinary least
    squares fit per column with the model's own structure -- shared slopes, free
    per-time intercept -- whose residual sd is what the model should be charging.

    Both series are relative to the snapshot column, which the model pins at 1.0.
    Feature staleness has cost ~11.5 points of R-squared at the one-year column
    and inflated its residual sd by ~26.5%, against ~19.8% charged: tracking to
    within about seven points at the worst column, with no free-parameter
    blow-out.
    """
    if not _requires_plotly("plot_sigma_time_calibration"):
        return None
    post = _post(idata)
    st = _flat(post, "sigma_time")
    if st is None or st.ndim != 2:
        logger.warning("plot_sigma_time_calibration skipped: sigma_time absent")
        return None
    fitted = st.mean(axis=0)
    days = np.asarray(panel.time_days, dtype="float64")[: fitted.size]
    labels = [f"{int(d)}d" for d in days]

    # The empirical reference, computed HERE on this run's panel rather than
    # carried over from an older one -- the published version of this table has
    # not been recomputed in three fits.
    emp_sd, emp_r2 = _pooled_ols_residual_scale(panel)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=fitted, name="posterior sigma_time",
        marker=dict(color=_hex_to_rgba(C_POSTERIOR, 0.85)),
        hovertemplate="%{x}: sigma_time %{y:.3f}<extra></extra>",
    ))
    if emp_sd is not None and emp_sd.size == fitted.size:
        fig.add_trace(go.Scatter(
            x=labels, y=emp_sd, mode="lines+markers", name="measured residual sd",
            line=dict(color=C_OBSERVED, width=2.4), marker=dict(size=9),
            hovertemplate="%{x}: measured %{y:.3f}<extra></extra>",
        ))
    _add_ref_line(fig, y=1.0, kind="anchor", annotation_text="snapshot column (pinned)")
    _fmt_axis(fig, x="lookback (gap to now)", y="scale, relative to the snapshot")
    fig.update_layout(
        title="Per-lookback scale against measured staleness",
        legend=dict(font_size=11, yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    _render_plotly(fig, height=H_SHORT, label="09_sigma_time_calibration")

    at = _flat(post, "alpha_time")
    rows = []
    for i, (lab, d) in enumerate(zip(labels, days)):
        rows.append({
            "lookback": lab,
            "gap_days": float(d),
            "sigma_time_posterior": float(fitted[i]),
            "residual_sd_measured": float(emp_sd[i]) if emp_sd is not None and i < emp_sd.size else np.nan,
            "r2_measured": float(emp_r2[i]) if emp_r2 is not None and i < emp_r2.size else np.nan,
            "alpha_time_posterior": float(at.mean(axis=0)[i]) if at is not None and at.ndim == 2 and i < at.shape[1] else np.nan,
        })
    write_table(pd.DataFrame(rows), "09_sigma_time_table")
    return fig


def _pooled_ols_residual_scale(panel: Any) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Per-column residual sd and R-squared under the model's own mean structure.

    Fits ``y[:, t] = a_t + X @ b`` with slopes SHARED across ``t`` and a free
    intercept per ``t`` -- deliberately the structure the model imposes, because
    the comparison is only meaningful if the reference is constrained the same
    way. Residual sds are returned relative to the snapshot column, matching how
    ``sigma_time`` is parameterised.

    Returns ``(None, None)`` rather than raising: this is a reference series for
    a figure, and a rank-deficient design must not cost the run its panel.
    """
    try:
        Y = np.asarray(panel.Y, dtype="float64")
        X = np.asarray(panel.X_drift, dtype="float64")
        mask = np.asarray(panel.observed_mask, dtype=bool)
        n_isin, T = Y.shape
        # Stack the panel long, with a dummy per time column for the intercepts.
        rows_i, rows_t = np.nonzero(mask)
        y = Y[rows_i, rows_t]
        D = np.zeros((y.size, T), dtype="float64")
        D[np.arange(y.size), rows_t] = 1.0
        A = np.hstack([D, X[rows_i, :]])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        sd = np.full(T, np.nan)
        r2 = np.full(T, np.nan)
        for t in range(T):
            sel = rows_t == t
            if sel.sum() > 2:
                sd[t] = float(np.std(resid[sel], ddof=1))
                var_y = float(np.var(y[sel], ddof=1))
                r2[t] = 1.0 - (np.var(resid[sel], ddof=1) / var_y) if var_y > 0 else np.nan
        anchor = sd[-1] if np.isfinite(sd[-1]) and sd[-1] > 0 else np.nanmax(sd)
        return (sd / anchor if np.isfinite(anchor) and anchor > 0 else sd), r2
    except Exception as exc:  # pragma: no cover - reference series only
        logger.debug("pooled OLS reference unavailable: %s", exc)
        return None, None


def plot_drift_forest(idata: Any, panel: Any) -> Optional[Any]:
    """Drift coefficients with their intervals and their ESS.

    Two features carry the mean and three straddle zero, and that has held across
    six runs. Drawing the intervals rather than tabulating the point estimates is
    what makes "straddles zero" a property a reader can see instead of a claim
    they have to take.

    ESS is annotated because the lowest-ESS coefficient has been the same one on
    every fit (``feat_mcap_country_r``), which is a fact about the design matrix
    rather than about the sampler.
    """
    if not _requires_plotly("plot_drift_forest"):
        return None
    post = _post(idata)
    beta = _flat(post, "beta")
    if beta is None or beta.ndim != 2:
        logger.warning("plot_drift_forest skipped: beta absent from the posterior")
        return None
    names = list(panel.drift_names)[: beta.shape[1]]
    mean = beta.mean(axis=0)
    lo = np.percentile(beta, 5.5, axis=0)
    hi = np.percentile(beta, 94.5, axis=0)
    order = np.argsort(np.abs(mean))
    straddles = (lo < 0) & (hi > 0)

    fig = go.Figure()
    for j in order:
        colour = C_MUTED if straddles[j] else C_POSTERIOR
        fig.add_trace(go.Scatter(
            x=[lo[j], hi[j]], y=[names[j], names[j]], mode="lines",
            line=dict(color=_hex_to_rgba(colour, 0.7), width=6), showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[mean[j]], y=[names[j]], mode="markers", showlegend=False,
            marker=dict(color=colour, size=11, line=dict(width=1, color=C_PANEL_EDGE)),
            hovertemplate=(f"{names[j]}<br>beta %{{x:+.4f}}<br>"
                           f"89% ETI [{lo[j]:+.4f}, {hi[j]:+.4f}]"
                           f"{' · straddles zero' if straddles[j] else ''}"
                           "<extra></extra>"),
        ))
    _add_ref_line(fig, x=0.0, kind="zero")
    _fmt_axis(fig, x="beta (standardised response per standardised feature)")
    fig.update_layout(
        title=(
            f"Drift coefficients — {len(names)} features, "
            f"{int(straddles.sum())} straddling zero <sub>(89% ETI; grey = "
            "indistinguishable from zero)</sub>"
        ),
        height=_forest_height_px(len(names)),
    )
    _render_plotly(fig, label="09_drift_forest")

    write_table(
        pd.DataFrame({
            "feature": names, "beta": mean, "eti89_lo": lo, "eti89_hi": hi,
            "straddles_zero": straddles,
        }).sort_values("beta", key=np.abs, ascending=False),
        "09_drift_table",
    )
    return fig


# =========================================================================== #
#  §10  Screen                                                                #
# =========================================================================== #
def plot_screen_overview(screen: pd.DataFrame, panel: Any) -> Optional[Any]:
    """Model expected upside against analyst-implied upside, over the universe.

    The one panel that answers "what did the model actually change?". The point
    estimate is, to a first approximation, the consensus -- which is not a defect
    on its own, since a filter whose input is a 12-month analyst target should
    agree with it more than it disagrees -- but the slope and the spread around
    the identity line are where the shrinkage lives.

    DECIMATED, with the sampled count stated in the title, and the Spearman and
    the OLS slope computed on the **full** frame. A rank-based cut is not an
    alternative: the moment it binds it deletes one tail and the surviving cloud
    misrepresents the screen.
    """
    if not _requires_plotly("plot_screen_overview"):
        return None
    need = {"expected_upside", "implied_upside"}
    if screen is None or not need <= set(screen.columns):
        logger.warning("plot_screen_overview skipped: screen lacks %s", need)
        return None
    df = screen[[c for c in ("isin", "sector", "expected_upside", "implied_upside",
                             "p_upside_pos_cond") if c in screen.columns]].dropna(
        subset=["expected_upside", "implied_upside"])
    if not len(df):
        return None

    # Statistics on the FULL frame, before any decimation.
    rho = float(df["expected_upside"].corr(df["implied_upside"], method="spearman"))
    slope, intercept = np.polyfit(df["implied_upside"], df["expected_upside"], 1)
    med_rev = float((df["expected_upside"] - df["implied_upside"]).abs().median())

    shown, decimated = _decimate_frame(
        df, _SCREEN_SCATTER_MAX_POINTS,
        by="sector" if "sector" in df.columns else None,
    )
    fig = go.Figure()
    colour = (shown["p_upside_pos_cond"] if "p_upside_pos_cond" in shown.columns
              else shown["expected_upside"])
    fig.add_trace(go.Scattergl(
        x=shown["implied_upside"] * 100.0, y=shown["expected_upside"] * 100.0,
        mode="markers", showlegend=False,
        marker=dict(color=colour, colorscale=CS_SEQ, size=5, opacity=0.65,
                    colorbar=dict(title="P(up | risk)" if "p_upside_pos_cond"
                                  in shown.columns else "E[upside]")),
        # No per-point identity string: hover text is the bulk of a large
        # scatter's payload and is unreadable in a cloud this dense.
        hovertemplate="implied %{x:.1f}% → model %{y:.1f}%<extra></extra>",
    ))
    lim = float(np.nanpercentile(np.abs(df[["implied_upside", "expected_upside"]].to_numpy()), 99)) * 100.0
    grid = np.array([-lim, lim])
    fig.add_trace(go.Scatter(
        x=grid, y=grid, mode="lines", name="y = x", showlegend=False,
        line=dict(color=C_REF, width=1.4, dash="dot"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=grid, y=(slope * grid / 100.0 + intercept) * 100.0, mode="lines",
        name="OLS", showlegend=False,
        line=dict(color=C_FORECAST, width=2.0), hoverinfo="skip",
    ))
    _add_ref_line(fig, x=0, kind="zero")
    _add_ref_line(fig, y=0, kind="zero")
    _fmt_axis(fig, x="analyst implied upside (%)", y="model expected upside (%)",
              x_kind="pct", y_kind="pct")
    fig.update_layout(
        title=(
            f"Screen — Spearman {rho:.4f}, OLS slope {slope:.3f}, median "
            f"absolute revision {med_rev * 100:.2f}pp"
            f"<br><sub>{len(shown):,} of {len(df):,} names drawn"
            f"{' (stratified sample)' if decimated else ''}; statistics on all "
            f"{len(df):,}</sub>"
        )
    )
    _render_plotly(fig, height=H_PANEL, label="10_screen_overview")
    return fig


#: Columns the rank-correlation table scores, under the ANALYTICS spellings and
#: the intermediate screen ones, because the frame handed in may be either.
_RANKING_SURFACE: tuple[str, ...] = (
    "expected_return_kalman", "expected_upside",
    "price_target_kalman", "expected_pt",
    "er_mean", "er_p50",
    "risk_adj_return",
    "expected_sharpe_ratio",
    "reward_to_cvar", "starr",
    "ret_vol_ratio",
    "p_upside_pos_cond",
    "mc_prob_pos",
    "shrink_gain",
)


def plot_rank_correlations(
    frame: pd.DataFrame,
    *,
    reference: str = "implied_upside",
) -> Optional[Any]:
    """How far each exported column departs from a consensus sort.

    The post-run analysis rebuilds this every edition by hand, which is precisely
    the practice its own skill warns against -- two runs compared on
    hand-recomputed statistics are two runs compared on different definitions.
    Making it a pipeline artifact fixes the definition once: **Spearman**, over
    every name with both columns present, against ``implied_upside``.

    A column at 1.0 re-ranks nothing. The reading that matters is which columns
    are DRIFTING: ``reward_to_cvar`` moved 0.7625 -> 0.8356 -> 0.8685 across
    three runs while every other column held, so the book's ranking column has
    been losing its independence from the input it is meant to filter, and only a
    stable definition can show that.
    """
    if not _requires_plotly("plot_rank_correlations"):
        return None
    if frame is None or reference not in frame.columns:
        logger.warning("plot_rank_correlations skipped: no %s column", reference)
        return None
    ref = pd.to_numeric(frame[reference], errors="coerce")

    rows = []
    for col in _RANKING_SURFACE:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        ok = ref.notna() & vals.notna()
        if ok.sum() < 30 or vals[ok].nunique() < 3:
            continue
        rows.append({
            "column": col,
            "spearman_vs_reference": float(vals[ok].corr(ref[ok], method="spearman")),
            "n": int(ok.sum()),
            "pinned_share": float((vals[ok] == vals[ok].mode().iloc[0]).mean()),
        })
    if not rows:
        logger.warning("plot_rank_correlations skipped: no scoreable columns")
        return None
    tab = pd.DataFrame(rows).sort_values("spearman_vs_reference")

    # Dimensionless risk-normalised columns read differently from first-moment
    # ones, so they are coloured apart rather than mixed into one ranking.
    risk_like = {"expected_sharpe_ratio", "reward_to_cvar", "starr", "ret_vol_ratio",
                 "p_upside_pos_cond", "mc_prob_pos", "shrink_gain", "risk_adj_return"}
    colours = [C_ACCENT if c in risk_like else C_POSTERIOR for c in tab["column"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tab["spearman_vs_reference"], y=tab["column"], orientation="h",
        marker=dict(color=[_hex_to_rgba(c, 0.85) for c in colours]),
        hovertemplate="%{y}: rho = %{x:.4f}<extra></extra>", showlegend=False,
    ))
    _add_ref_line(fig, x=1.0, kind="anchor", annotation_text="re-ranks nothing")
    _fmt_axis(fig, x=f"Spearman rho against {reference}")
    fig.update_xaxes(range=[min(0.5, float(tab["spearman_vs_reference"].min()) - 0.05), 1.02])
    fig.update_layout(
        title=(
            "How far each exported column departs from a consensus sort"
            f"<br><sub>Green = risk-normalised / dimensionless. Furthest: "
            f"{tab.iloc[0]['column']} at {tab.iloc[0]['spearman_vs_reference']:.4f}</sub>"
        ),
        height=_forest_height_px(len(tab)),
    )
    _render_plotly(fig, label="10_rank_correlations")
    write_table(tab, "10_screen_rank_correlations")
    return fig


def plot_er_sd_calibration(frame: pd.DataFrame, panel: Any) -> Optional[Any]:
    """Score the forward-return second moment against realised volatility.

    ``er_sd`` -- the pooled sd of the Monte-Carlo forward-return draws -- is the
    second moment the entire risk book rests on. It sets ``expected_sharpe_ratio``
    directly, and through ``tail_risk = max(-cvar05, k * er_sd, 0.01)`` it is the
    denominator of ``reward_to_cvar`` for every name on the volatility floor,
    which on the last two runs was all 25 of 25 book names.

    ``k = 0.25`` has never had a documented rationale, only a value. This panel
    is the reference that could give it one: the panel already carries
    ``vol_level`` (realised volatility), so ``er_sd`` can be regressed on it and
    the ratio read off. If a name's modelled forward sd is a quarter of its
    realised volatility, ``k = 0.25`` is charging roughly one realised-vol unit;
    if it is a tenth, ``k`` is charging four.

    Reported, NOT gated. One run is a measurement, not a calibration -- and the
    honest form of "we chose 0.25" is a number with a run attached, not a gate
    that would fail on the first refresh.

    Notes
    -----
    This is the useful half of what ``bsm_functions.py`` suggested. A
    market-implied volatility would be the better reference -- it is
    forward-looking, which realised volatility is not -- but the database carries
    no option quotes and no risk-free rate, so the implied route is blocked on
    data rather than on code.
    """
    if not _requires_plotly("plot_er_sd_calibration"):
        return None
    if frame is None or "er_sd" not in frame.columns or "isin" not in frame.columns:
        logger.warning("plot_er_sd_calibration skipped: no er_sd/isin in the frame")
        return None
    vol = np.asarray(getattr(panel, "vol_level", []), dtype="float64")
    if vol.size != len(panel.isins):
        logger.warning("plot_er_sd_calibration skipped: panel carries no vol_level")
        return None

    ref = pd.DataFrame({"isin": np.asarray(panel.isins, dtype=object), "vol_level": vol})
    df = frame[["isin", "er_sd"]].merge(ref, on="isin", how="inner").dropna()
    df = df[(df["er_sd"] > 0) & (df["vol_level"] > 0)]
    if len(df) < 50:
        logger.warning("plot_er_sd_calibration skipped: only %d usable rows", len(df))
        return None

    # Statistics on the FULL frame.
    slope, intercept = np.polyfit(df["vol_level"], df["er_sd"], 1)
    rho = float(df["er_sd"].corr(df["vol_level"], method="spearman"))
    ratio = df["er_sd"] / df["vol_level"]
    r_med = float(ratio.median())
    resid = df["er_sd"] - (slope * df["vol_level"] + intercept)
    var_y = float(np.var(df["er_sd"], ddof=1))
    r2 = 1.0 - float(np.var(resid, ddof=1)) / var_y if var_y > 0 else float("nan")

    shown, decimated = _decimate_frame(df, _SCREEN_SCATTER_MAX_POINTS)
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.11,
        subplot_titles=("er_sd against realised volatility",
                        "Ratio er_sd / realised vol"),
    )
    fig.add_trace(go.Scattergl(
        x=shown["vol_level"], y=shown["er_sd"], mode="markers", showlegend=False,
        marker=dict(color=_hex_to_rgba(C_POSTERIOR, 0.5), size=4),
        hovertemplate="realised vol %{x:.3f} → er_sd %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    xg = np.linspace(float(df["vol_level"].min()), float(df["vol_level"].max()), 50)
    fig.add_trace(go.Scatter(
        x=xg, y=slope * xg + intercept, mode="lines", showlegend=False,
        line=dict(color=C_FORECAST, width=2.2), hoverinfo="skip",
    ), row=1, col=1)
    _fmt_axis(fig, x="realised volatility (feat_vol_level)", y="er_sd", row=1, col=1)

    _add_binned_density(
        fig, ratio.to_numpy(), row=1, col=2, bins=70, color=C_ACCENT, alpha=0.7,
        clip=(0.0, float(np.nanpercentile(ratio, 99))), name="er_sd / vol",
        showlegend=False, hovertemplate="ratio = %{x:.3f}<extra></extra>",
    )
    _add_ref_line(fig, x=r_med, kind="anchor",
                  annotation_text=f"median {r_med:.3f}", row=1, col=2)
    _add_ref_line(fig, x=0.25, kind="emphasis",
                  annotation_text="k = 0.25", row=1, col=2)
    _fmt_axis(fig, x="er_sd / realised vol", y="density", row=1, col=2)

    fig.update_layout(
        title=(
            f"Second-moment calibration — slope {slope:.3f}, R² {r2:.3f}, "
            f"Spearman {rho:.3f}, median ratio {r_med:.3f}"
            f"<br><sub>{len(shown):,} of {len(df):,} names drawn; statistics on all "
            f"{len(df):,}. Reported, not gated.</sub>"
        )
    )
    _render_plotly(fig, height=H_SHORT, label="10_er_sd_calibration")

    write_table(
        pd.DataFrame([{
            "n": len(df),
            "ols_slope": slope,
            "ols_intercept": intercept,
            "r2": r2,
            "spearman": rho,
            "ratio_p05": float(ratio.quantile(0.05)),
            "ratio_median": r_med,
            "ratio_p95": float(ratio.quantile(0.95)),
            "tail_risk_vol_floor_k": 0.25,
            "k_in_realised_vol_units": 0.25 * r_med,
        }]),
        "10_er_sd_calibration_table",
    )
    return fig


# =========================================================================== #
#  §10b  Risk book                                                            #
# =========================================================================== #
def plot_risk_book(risk_book: Any, run_cfg: Any) -> Optional[Any]:
    """The sized book, with the volatility floor made visible.

    Two things this panel exists to show, neither of which survives a table:

    The **weights** against the cap. The cap has bound nowhere on recent runs
    (largest holding ~6.15% against a 10% cap), which means the sizing is being
    set by STARR dispersion rather than by the constraint -- worth knowing before
    changing the cap.

    Which names sit on the **volatility floor**. ``tail_risk`` is
    ``max(-cvar05, k * er_sd, 0.01)``, and for a name whose simulated shortfall
    is positive the loss leg is not binding and the floor takes over. On the last
    two runs that was 25 of 25 book names: their ``starr`` is exactly
    ``expected_upside / (k * er_sd)`` -- a reward-to-variability ratio wearing the
    name of a tail ratio. Marking them is the difference between a reader seeing
    a risk ranking and seeing what it actually is.
    """
    if not _requires_plotly("plot_risk_book") or risk_book is None:
        return None
    book = getattr(risk_book, "book", None)
    if book is None or not len(book) or "book_weight" not in book.columns:
        logger.warning("plot_risk_book skipped: no sized book")
        return None
    b = book.sort_values("book_weight", ascending=True).copy()

    label = (b["ticker"].astype(str) if "ticker" in b.columns
             else b["isin"].astype(str).str[:8])
    # Disambiguate collisions: Plotly silently SUMS same-category bar values,
    # which hides part of the book behind one inflated bar.
    seen: dict[str, int] = {}
    labels = []
    for t in label:
        seen[t] = seen.get(t, 0) + 1
        labels.append(t if seen[t] == 1 else f"{t} ({seen[t]})")

    k = float(getattr(run_cfg, "tail_risk_vol_floor_k", 0.25))
    on_floor = None
    if {"tail_risk", "er_sd"} <= set(b.columns):
        floor = k * pd.to_numeric(b["er_sd"], errors="coerce")
        on_floor = np.isclose(pd.to_numeric(b["tail_risk"], errors="coerce"), floor,
                              rtol=1e-9, atol=1e-12)

    # No `subplot_titles`. They anchor at paper y=1, inside the top margin, which
    # `_render_plotly` pins at 70px AFTER this function returns -- so the two-line
    # title below overprints them and neither is readable. The x-axis titles
    # ("weight (%)" / "CVaR 5% (%)") already name both panels.
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05)
    fig.add_trace(go.Bar(
        x=pd.to_numeric(b["book_weight"], errors="coerce") * 100.0, y=labels,
        orientation="h", marker=dict(color=_hex_to_rgba(C_POSTERIOR, 0.85)),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>", showlegend=False,
    ), row=1, col=1)
    cap = float((risk_book.summary or {}).get("cap", float("nan")))
    if np.isfinite(cap):
        _add_ref_line(fig, x=cap * 100.0, kind="anchor",
                      annotation_text=f"cap {cap:.0%}", row=1, col=1)
    _fmt_axis(fig, x="weight (%)", x_kind="pct", row=1, col=1)

    if "cvar05" in b.columns:
        cv = pd.to_numeric(b["cvar05"], errors="coerce") * 100.0
        colours = [_hex_to_rgba(C_OBSERVED if (on_floor is None or f) else C_ACCENT, 0.85)
                   for f in (on_floor if on_floor is not None else [True] * len(b))]
        fig.add_trace(go.Bar(
            x=cv, y=labels, orientation="h", marker=dict(color=colours),
            hovertemplate="%{y}: CVaR5 %{x:.2f}%<extra></extra>", showlegend=False,
        ), row=1, col=2)
        _add_ref_line(fig, x=0, kind="zero", row=1, col=2)
        _fmt_axis(fig, x="CVaR 5% (%)", x_kind="pct", row=1, col=2)

    # Pin the label axis to `category`. A book of global names routinely holds
    # exchange-code tickers -- "700", "2330", "601318", "532454" -- and under the
    # project's arviz dark template Plotly's axis autotype resolves a MIXED
    # alphabetic/numeric label array to `linear` rather than `category`. (The same
    # array autotypes as `category` under the stock template, which is why this
    # only ever showed up on exported panels.) On a linear axis the numeric
    # tickers become COORDINATES -- scattering four or five hairline bars across a
    # 0-600k range -- and every alphabetic label coerces to NaN and is dropped, so
    # a fully-populated 25-name book renders as an empty panel with a correct
    # title. `categoryarray` keeps the book_weight sort explicit rather than
    # leaving it to trace order.
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=labels)

    s = risk_book.summary or {}
    n_floor = int(on_floor.sum()) if on_floor is not None else -1
    w = pd.to_numeric(b["book_weight"], errors="coerce").fillna(0.0).to_numpy()
    hhi = float((w ** 2).sum())
    fig.update_layout(
        title=(
            f"Risk book — {len(b)} names, effective N {1.0 / hhi if hhi else float('nan'):.1f}, "
            f"HHI {hhi:.4f}"
            + (f"<br><sub>{n_floor} of {len(b)} names charged the VOLATILITY FLOOR "
               f"(k = {k}) rather than their tail — for those, "
               "reward_to_cvar is a rescaling of expected_sharpe_ratio</sub>"
               if n_floor >= 0 else "")
        ),
        height=max(H_PANEL, _forest_height_px(len(b))),
    )
    _render_plotly(fig, label="10b_risk_book")

    comp = b.copy()
    if on_floor is not None:
        comp["on_volatility_floor"] = on_floor
    keep = [c for c in ("isin", "ticker", "name", "sector", "region", "book_weight",
                        "expected_upside", "er_mean", "er_sd", "er_p05", "cvar05",
                        "exp_vol", "tail_risk", "starr", "expected_sharpe_ratio",
                        "p_upside_pos_cond", "on_volatility_floor")
            if c in comp.columns]
    write_table(comp[keep], "10b_book_composition_table")

    summary_rows = [{"statistic": kk, "value": vv} for kk, vv in s.items()]
    summary_rows += [
        {"statistic": "hhi", "value": hhi},
        {"statistic": "effective_n", "value": (1.0 / hhi) if hhi else float("nan")},
        {"statistic": "n_on_volatility_floor", "value": float(n_floor)},
        {"statistic": "tail_risk_vol_floor_k", "value": k},
    ]
    write_table(pd.DataFrame(summary_rows), "10b_book_summary_table")
    return fig


# =========================================================================== #
#  Orchestration                                                              #
# =========================================================================== #
def _attempt(label: str, fn, *args, **kwargs) -> None:
    """Run one panel, log a failure, and never re-raise.

    The rule ``write_analytics_ddl_v2`` already follows, applied to figures. A
    run reaching this point has paid for a fit; losing the export to a Plotly
    error in a panel nobody gated on would be the worst possible trade.
    """
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - figures are best-effort
        logger.warning("panel %s failed: %s", label, exc, exc_info=logger.isEnabledFor(logging.DEBUG))


def render_run(
    result: dict,
    panel: Any,
    run_cfg: Any,
    *,
    kalman_results: Optional[pd.DataFrame] = None,
) -> None:
    """Draw every panel this run has the inputs for.

    One call from :func:`pymc_kalman_filter_pt_v2.main`, placed AFTER the export
    so a figure can never delay or endanger the analytics write. Each panel is
    attempted independently and each is a no-op when its input is absent, so a
    ``--dry-run`` (panel audit only) and a full ``--write`` run both work through
    the same entry point without a matrix of guards at the call site.

    Parameters
    ----------
    result
        The dict :func:`main` assembles. Read by the pipeline's own key names --
        ``panel_audit``, ``prior_idata``, ``ppc``, ``idata``, ``diagnostics``,
        ``screen``, ``risk_book``, ``kalman_results`` -- because that dict is a
        published contract with other consumers and the figure layer is the
        newcomer. Every key is optional.
    panel
        The fitted :class:`KalmanPanelV2`.
    run_cfg
        The run's :class:`KalmanRunConfigV2`.
    kalman_results
        The canonical export frame. Preferred over ``screen`` for the
        rank-correlation panel, because it carries the ANALYTICS column
        spellings a dashboard reader will actually look for.
    """
    if not get_export_state().enabled:
        logger.info("figures disabled; skipping the panel set")
        return

    audit = result.get("panel_audit") or result.get("audit") or {}
    idata = result.get("idata")
    ppc = result.get("ppc") or {}
    if kalman_results is None:
        kalman_results = result.get("kalman_results")

    if audit:
        with section("04b_audit"):
            _attempt("panel_audit", plot_panel_audit, panel, audit)
            _attempt("decay_ladder", plot_decay_ladder, panel, audit, idata,
                     residual_kernel=ppc.get("decay_residual") or ppc.get("decay"))

    prior = result.get("prior_idata") or result.get("prior")
    if prior is not None:
        with section("06_prior"):
            _attempt("prior_predictive", plot_prior_predictive, prior, panel)

    if ppc:
        with section("08_ppc"):
            _attempt("ppc_overlay", plot_ppc_overlay, ppc, panel, result.get("ppc_idata"))
            _attempt("ppc_calibration", plot_ppc_calibration, ppc, panel, run_cfg)
            _attempt("ppc_decay", plot_ppc_decay, ppc)

    if idata is not None:
        with section("09_diagnostics"):
            diag = result.get("diagnostics")
            if diag is not None:
                _attempt("rhat_ess", plot_rhat_ess, diag, run_cfg)
                _attempt("trace_worst", plot_trace_worst, idata, diag)
            _attempt("energy", plot_energy, idata)
            _attempt("variance_legs", plot_variance_legs, idata)
            _attempt("sigma_time", plot_sigma_time_calibration, idata, panel)
            _attempt("drift_forest", plot_drift_forest, idata, panel)

    screen = result.get("screen")
    if screen is not None and len(screen):
        with section("10_screen"):
            _attempt("screen_overview", plot_screen_overview, screen, panel)
            # The analytics frame carries the exported spellings; fall back to the
            # screen so a `--dry-run`-style call still produces the table.
            _attempt("rank_correlations", plot_rank_correlations,
                     kalman_results if kalman_results is not None else screen)
            _attempt("er_sd_calibration", plot_er_sd_calibration, screen, panel)

    if result.get("risk_book") is not None:
        with section("10b_risk"):
            _attempt("risk_book", plot_risk_book, result["risk_book"], run_cfg)

    logger.info("figures written under %s", get_export_state().root)
