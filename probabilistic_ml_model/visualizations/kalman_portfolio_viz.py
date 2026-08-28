"""Figures for the forecast + decision replay (``kalman_portfolio.py``).

What these panels are for
-------------------------
The replay produces two things a table reads badly: **two books from one posterior
that share a minority of their names**, and **two priors nothing identifies** that
between them determine the whole departure from analyst consensus. Both are questions
about shape, and both have been argued in prose for five editions of the post-run
analysis without a picture. That is what this module is for.

The suite's argument, in two panels
-----------------------------------
:func:`plot_denominator_sanity` and :func:`plot_shrinkage_contrast` are meant to be
read together, and they are the reason the recommendation layer is in this pipeline at
all. The first shows what a reward-to-risk ratio does with thin evidence: it *inflates*
the score, so a book ends up holding names whose modelled downside sits two orders of
magnitude below the universe median. The second shows the only place in the pipeline
where thin evidence does the opposite -- ``lambda_g`` pulls a poorly-resolved group's
signal *toward zero* rather than to an extreme. Same input condition, opposite
treatment, drawn side by side.

Colour
------
The design system is ``kalman_shared``'s semantic roles, not a second palette. Its
categorical slots were run through the data-viz validator against the real dark
surface ``#1e1e1e`` with ``--pairs all``:

- Four slots FAIL: ``C_DRAWS`` vs ``C_OBSERVED`` at CVD ΔE 7.6 (protan), inside the
  6-8 band that is legal only with secondary encoding.
- Three slots -- ``C_POSTERIOR`` ``#56b4e9``, ``C_OBSERVED`` ``#ffb000``,
  ``C_FORECAST`` ``#cc79a7`` -- PASS every separation gate: worst all-pairs CVD ΔE 9.6,
  normal-vision ΔE 20.0, contrast >= 3:1 on all three.

So :data:`ARM_COLORS` is capped at three, which is exactly the three ranking arms. A
fourth series folds into "Other" or facets; it is never a generated hue.

One documented deviation: the validator's lightness-band check FAILs on all three.
The palette is Okabe-Ito lifted for a dark surface and sits brighter than the
validator's dark band. Contrast and separation pass, and re-stepping would break every
existing v2 figure, so the deviation is recorded here rather than silently ignored.

Sign is never a categorical slot. Over/under-zero uses the diverging rule --
:data:`CS_DIV` for a scale, ``_add_ref_line(kind='zero')`` for the axis -- and the
three-state verdict is carried by position and label, because a verdict on a signed
axis painted in three hues encodes the same thing twice and reads as neither.

Payload budget
--------------
Every panel here draws over thousands of names by thousands of scenarios, so the
budget is the design constraint rather than an afterthought: pre-binned densities
(:func:`_add_binned_density`), gridded ECDFs (:func:`_ecdf_xy`), decimated scatters
(:func:`_decimate_frame`) with the sampled count in the title and every summary
statistic computed on the FULL frame. No per-point hover identity strings -- they are
the bulk of a large scatter's payload and unreadable in a dense cloud.

Usage
-----
.. code-block:: python

    from probabilistic_ml_model.visualizations import kalman_portfolio_viz as pviz

    pviz.install(cfg)
    pviz.render_replay(result)      # one call; every panel is a no-op without input
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .kalman_shared import (
    C_ACCENT, C_FORECAST, C_HIGHLIGHT, C_MUTED, C_OBSERVED, C_POSTERIOR, C_REF,
    CS_DIV, CS_SEQ,
    _PPC_ECDF_GRID, _SCREEN_SCATTER_MAX_POINTS,
    _add_binned_density, _add_ref_line, _decimate_frame, _ecdf_xy, _export_path,
    _fmt_axis, _forest_height_px, _hex_to_rgba, _next_stem, _safe_show,
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

__all__ = [
    "ARM_COLORS",
    "install",
    "section",
    "write_table",
    "render_replay",
    "plot_engine_contrast",
    "plot_factor_sweep",
    "plot_multiplier_sweep",
    "plot_two_books",
    "plot_sector_mix",
    "plot_action_ladder",
    "plot_consensus_gap",
    "plot_kelly_pin",
    "plot_risk_ladder",
    "plot_denominator_sanity",
    "plot_rank_agreement",
    "plot_ergodicity",
    "plot_group_signal_forest",
    "plot_shrinkage_contrast",
    "plot_size_down_overlap",
]

H_SHORT = 380
H_PANEL = 520
H_TALL = 700

C_PANEL_EDGE = "#1e1e1e"

#: The validated three-slot categorical set, in fixed order. Assigned to arms by
#: NAME, never cycled by position, so a run that drops an arm does not repaint the
#: survivors. See the module docstring for the validator output.
ARM_COLORS: dict[str, str] = {
    "reward_to_downside": C_POSTERIOR,   # #56b4e9
    "reward_to_cvar": C_OBSERVED,        # #ffb000
    "p_upside_pos_cond": C_FORECAST,     # #cc79a7
}

#: Sectors shown before the rest fold into "Other". Eleven sectors would need eleven
#: hues, and the palette validates three.
_MAX_SECTORS = 5

#: Rows above which a per-name panel switches to a summary rather than a label per row.
_MAX_LABELLED_ROWS = 60


def _arm_color(arm: str, fallback: str = C_MUTED) -> str:
    """Colour for a ranking arm, by name. Unknown arms are muted, not invented."""
    return ARM_COLORS.get(arm, fallback)


# =========================================================================== #
#  Wiring — identical in shape to kalman_viz_v2, deliberately                 #
# =========================================================================== #


def install(cfg: Any, *, enable: bool = True) -> None:
    """Point the shared figure layer at this replay and turn artifact export on."""
    set_viz_config_resolver(lambda: cfg)
    setup_plotting()
    enable_artifact_export(enable)


@contextmanager
def section(label: str):
    """Scope artifact filenames to a section (see ``export_section``)."""
    with export_section(label):
        yield


def write_table(frame: Optional[pd.DataFrame], stem: str) -> Optional[str]:
    """Write the table behind a panel, so a caption's number can be checked."""
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


def _requires_plotly(name: str) -> bool:
    if not _HAS_PLOTLY:
        logger.info("plotly is not installed; %s skipped", name)
        return True
    return False


def _base_layout(fig: Any, title: str, height: int) -> Any:
    fig.update_layout(title=title, height=height, margin=dict(l=70, r=30, t=70, b=60))
    return fig


def _pct_format(values: Any) -> str:
    """Percent tick format with enough precision for the values actually plotted.

    ``kalman_shared``'s ``x_kind='prob'`` is a fixed ``.0%``, which is right for a
    weight or a probability and wrong for a shrunk group excess: a column whose
    entries are all a few tenths of a percent renders as a row of identical "0%"
    ticks, so the axis carries no information at all. Chosen from the data's span.
    """
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return ".0%"
    span = float(np.nanmax(np.abs(arr)))
    if span >= 0.10:
        return ".0%"
    if span >= 0.01:
        return ".1%"
    if span >= 0.001:
        return ".2%"
    if span >= 0.0001:
        return ".3%"
    # Below a thousandth of a percent, percent notation has stopped carrying the
    # value: every tick rounds to the same string and the axis says nothing. Fall
    # back to scientific, which is ugly and legible, rather than pretty and blank.
    return ".1e"


# =========================================================================== #
#  §15  The forecast                                                          #
# =========================================================================== #


def plot_engine_contrast(engines: Optional[pd.DataFrame]) -> Optional[Any]:
    """Forward dispersion, this engine against the shipped AR simulator.

    *Job: agreement.* A y=x anchor and a decimated cloud. The two engines decay
    differently -- a fitted OU kernel against a hand-set ``rho=0.85`` -- so the spread
    around the diagonal is the size of that modelling choice, not noise.

    Spearman is computed on the **full** frame and stated in the title beside the
    sampled point count, because a statistic annotated on a decimated panel that was
    computed on the decimation is a different statistic.
    """
    if engines is None or not len(engines) or _requires_plotly("engine_contrast"):
        return None
    need = {"er_sd_fc", "er_sd_ar"}
    if not need <= set(engines.columns):
        logger.info("engine_contrast needs %s; skipped", sorted(need))
        return None

    full = engines.dropna(subset=["er_sd_fc", "er_sd_ar"])
    if not len(full):
        return None
    rho = float(full["er_sd_fc"].corr(full["er_sd_ar"], method="spearman"))
    shown, _cut = _decimate_frame(full, _SCREEN_SCATTER_MAX_POINTS)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=shown["er_sd_ar"], y=shown["er_sd_fc"], mode="markers",
        marker=dict(size=5, color=C_POSTERIOR, opacity=0.55,
                    line=dict(width=0.5, color=C_PANEL_EDGE)),
        name="name", hoverinfo="skip",
    ))
    lo = float(min(full["er_sd_ar"].min(), full["er_sd_fc"].min()))
    hi = float(max(full["er_sd_ar"].max(), full["er_sd_fc"].max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
        line=dict(color=C_REF, dash="dot", width=1.2),
    ))
    _base_layout(
        fig,
        f"Forward dispersion: forecast layer vs AR simulator<br>"
        f"<sub>Spearman ρ = {rho:.4f} on all {len(full):,} matched names; "
        f"{len(shown):,} plotted. Joined by ISIN.</sub>",
        H_PANEL,
    )
    _fmt_axis(fig, x="AR simulator er_sd", y="forecast layer er_sd")
    _safe_show(fig, label="engine_contrast")
    write_table(pd.DataFrame([{"spearman": rho, "n_matched": len(full),
                               "median_sd_ratio": float(full.get(
                                   "sd_ratio", pd.Series(dtype=float)).median())}]),
                "engine_contrast")
    return fig


def _sweep_panel(
    frame: pd.DataFrame, x: str, measures: Sequence[tuple[str, str]],
    title: str, anchor: Optional[float], anchor_label: str,
) -> Optional[Any]:
    """Small multiples over a swept knob. One measure per panel, one axis each."""
    present = [(c, lab) for c, lab in measures if c in frame.columns
               and frame[c].notna().any()]
    if not present:
        return None
    fig = make_subplots(rows=1, cols=len(present), shared_xaxes=False,
                        subplot_titles=[lab for _, lab in present])
    for i, (col, _lab) in enumerate(present, start=1):
        fig.add_trace(go.Scatter(
            x=frame[x], y=frame[col], mode="lines+markers",
            line=dict(color=C_POSTERIOR, width=2),
            marker=dict(size=8, line=dict(width=0.5, color=C_PANEL_EDGE)),
            showlegend=False, name=_lab,
        ), row=1, col=i)
        if anchor is not None:
            _add_ref_line(fig, x=anchor, kind="emphasis", row=1, col=i,
                          annotation_text=anchor_label)
        fig.update_xaxes(title_text=x, row=1, col=i)
    _base_layout(fig, title, H_SHORT)
    return fig


def plot_factor_sweep(sweep: Optional[pd.DataFrame],
                      shipped: float = 0.35) -> Optional[Any]:
    """What ``factor_share`` moves, and what it provably does not.

    *Job: change over a knob.* Three panels, one measure each, never a second y-axis.
    ``er_sd_max_abs_diff`` is on the far right on purpose: the split is
    variance-preserving, so per-name marginals are invariant and that panel should be
    flat at Monte-Carlo noise. A curve there would mean the invariance had broken and
    ``exp_vol`` -- and every ratio built on it -- had quietly moved.
    """
    if sweep is None or not len(sweep) or _requires_plotly("factor_sweep"):
        return None
    fig = _sweep_panel(
        sweep, "factor_share",
        [("book_sd_ratio", "Book sd, relative to shipped"),
         ("top_k_overlap", "Top-k membership overlap"),
         ("er_sd_max_abs_diff", "Max per-name er_sd move (invariance check)")],
        "factor_share is a prior: what it moves, and what it cannot<br>"
        "<sub>The variance split is preserving, so only the JOINT distribution "
        "moves — harmless for the screen, decisive for the book.</sub>",
        shipped, f"shipped {shipped:g}",
    )
    if fig is None:
        return None
    _safe_show(fig, label="factor_sweep")
    write_table(sweep, "factor_share_sweep")
    return fig


def plot_multiplier_sweep(sweep: Optional[pd.DataFrame],
                          shipped: float = 1.0) -> Optional[Any]:
    """The panel that is the argument for the whole layer.

    Without the forecast-error term the screen reproduces analyst consensus at
    Spearman 0.999995 -- run ``49e84d7e9d59``, the anchor of every table in the
    post-run analysis. The multiplier is what buys the departure from consensus, and
    it is a **prior chosen from a feasible band**, not an estimate: the panel cannot
    identify it, because ``obs_share`` measures how noisily a target is *republished*,
    not how far consensus sits from fair value.

    Drawn with the pass-through at ``kappa = 0`` marked as the anchor, so the reader
    sees how much of the model's distinctiveness is a function of one chosen number.
    """
    if sweep is None or not len(sweep) or _requires_plotly("multiplier_sweep"):
        return None
    fig = _sweep_panel(
        sweep, "forecast_error_multiplier",
        [("cross_sectional_sd", "Cross-sectional sd of the latent"),
         ("spearman_vs_unshrunk", "Spearman vs the unshrunk latent"),
         ("mean_abs_revision", "Mean |revision| vs unshrunk")],
        "forecast_error_multiplier is a prior, and it buys the whole departure "
        "from consensus<br><sub>At κ = 0 the latent is unshrunk and the screen "
        "reproduces consensus. Nothing in the panel identifies this knob.</sub>",
        shipped, f"shipped {shipped:g}",
    )
    if fig is None:
        return None
    if "forecast_error_multiplier" in sweep.columns:
        _add_ref_line(fig, x=0.0, kind="anchor", row=1, col=1,
                      annotation_text="pass-through")
    _safe_show(fig, label="multiplier_sweep")
    write_table(sweep, "multiplier_sweep")
    return fig


# =========================================================================== #
#  §15e  The decision layer                                                   #
# =========================================================================== #


def plot_two_books(books: Optional[pd.DataFrame],
                   arms: Optional[Sequence[str]] = None) -> Optional[Any]:
    """The suite's headline: two books from one posterior, side by side.

    *Job: identity plus magnitude of a set difference.* One row per name held by
    either arm, weights mirrored left and right of a zero line, shared names picked
    out in :data:`C_HIGHLIGHT`. The overlap count and the effective-N gap are readable
    without a legend lookup, which is the whole reason this is a diverging bar chart
    and not two ranked lists.

    Parameters
    ----------
    books
        Long frame with ``isin``, ``rank_by``, ``weight`` and optionally ``name`` --
        the ``15e_decision_books`` export.
    arms
        The two arms to contrast. Defaults to the first two present.
    """
    if books is None or not len(books) or _requires_plotly("two_books"):
        return None
    need = {"isin", "rank_by", "weight"}
    if not need <= set(books.columns):
        logger.info("two_books needs %s; skipped", sorted(need))
        return None

    held = books[books["weight"] > 0]
    present = list(dict.fromkeys(held["rank_by"]))
    pair = list(arms) if arms else present[:2]
    pair = [a for a in pair if a in present]
    if len(pair) < 2:
        logger.info("two_books needs two arms with a sized book; have %s", present)
        return None
    left, right = pair[0], pair[1]

    wl = held[held["rank_by"] == left].set_index("isin")["weight"]
    wr = held[held["rank_by"] == right].set_index("isin")["weight"]
    names = sorted(set(wl.index) | set(wr.index),
                   key=lambda i: -(float(wl.get(i, 0)) + float(wr.get(i, 0))))
    shared = set(wl.index) & set(wr.index)

    label_by = {}
    if "name" in held.columns:
        label_by = held.drop_duplicates("isin").set_index("isin")["name"].to_dict()
    # Sharedness is carried in the LABEL and by a hatch, never by colour. Colour is
    # already spent on arm identity, and the palette's amber slots -- C_HIGHLIGHT
    # #e69f00 against C_OBSERVED #ffb000 -- are close enough that a third amber makes
    # the legend unreadable. Identity is never colour-alone here twice over.
    labels = [f"{'● ' if i in shared else '   '}{str(label_by.get(i, i))[:26]}"
              for i in names]

    fig = go.Figure()
    for arm, series, sign in ((left, wl, -1.0), (right, wr, +1.0)):
        values = np.array([sign * float(series.get(i, 0.0)) for i in names])
        color = _arm_color(arm)
        fig.add_trace(go.Bar(
            y=labels, x=values, orientation="h", name=arm,
            marker=dict(
                color=color,
                # Sharedness as an OUTLINE, not a hatch: a hatch draws its foreground
                # over the fill and swallows the arm colour, which is the one thing
                # this panel's colour is spent on.
                line=dict(
                    width=[2.5 if i in shared else 1.0 for i in names],
                    color=[C_REF if i in shared else C_PANEL_EDGE for i in names],
                ),
            ),
            hovertemplate="%{y}<br>" + arm + " weight %{customdata:.2%}<extra></extra>",
            customdata=np.abs(values),
        ))
    _add_ref_line(fig, x=0.0, kind="zero")
    _base_layout(
        fig,
        f"Two books, one posterior — {len(shared)} of {len(names)} names shared"
        f"<br><sub>{left} (left) vs {right} (right). Outlined and marked ● = held by "
        f"both. Weights are absolute; the sign only separates the arms.</sub>",
        _forest_height_px(len(names), per_row=20, base=220),
    )
    fig.update_layout(barmode="overlay", bargap=0.25,
                      yaxis=dict(autorange="reversed"),
                      # Long company names need room; the default 70px clips them.
                      margin=dict(l=190, r=30, t=80, b=60))
    _fmt_axis(fig, x="weight", y="", x_kind="prob")
    _safe_show(fig, label="two_books")

    write_table(pd.DataFrame({
        "isin": names,
        "label": labels,
        f"weight_{left}": [float(wl.get(i, 0.0)) for i in names],
        f"weight_{right}": [float(wr.get(i, 0.0)) for i in names],
        "shared": [i in shared for i in names],
    }), "two_books")
    return fig


def plot_sector_mix(books: Optional[pd.DataFrame],
                    sector_col: str = "sector",
                    sector_cap: Optional[float] = None) -> Optional[Any]:
    """Sector weight per arm, top five plus "Other".

    Eleven sectors would need eleven hues and the palette validates three, so the tail
    folds into "Other" rather than becoming generated colour. Segments carry a 2px
    surface gap and direct labels above 5%, so identity is never colour alone.
    """
    if books is None or not len(books) or _requires_plotly("sector_mix"):
        return None
    if sector_col not in books.columns or "weight" not in books.columns:
        logger.info("sector_mix needs %r and 'weight'; skipped", sector_col)
        return None

    held = books[books["weight"] > 0]
    if not len(held):
        return None
    mix = held.groupby(["rank_by", sector_col])["weight"].sum().reset_index()
    top = (mix.groupby(sector_col)["weight"].sum()
           .nlargest(_MAX_SECTORS).index.tolist())
    mix["group"] = np.where(mix[sector_col].isin(top), mix[sector_col], "Other")
    mix = mix.groupby(["rank_by", "group"])["weight"].sum().reset_index()

    order = top + ["Other"]
    # A magnitude ramp, not a categorical one: sectors are ordered by total weight,
    # so the encoding is ordinal and one hue's steps say so.
    ramp = _sector_steps(len(order))
    fig = go.Figure()
    for group, color in zip(order, ramp):
        block = mix[mix["group"] == group]
        if not len(block):
            continue
        fig.add_trace(go.Bar(
            x=block["rank_by"], y=block["weight"], name=str(group),
            marker=dict(color=color, line=dict(width=2, color=C_PANEL_EDGE)),
            text=[f"{group}<br>{v:.0%}" if v >= 0.05 else "" for v in block["weight"]],
            textposition="inside", insidetextanchor="middle",
            hovertemplate="%{x}<br>" + str(group) + " %{y:.1%}<extra></extra>",
        ))
    if sector_cap is not None:
        _add_ref_line(fig, y=float(sector_cap), kind="emphasis",
                      annotation_text=f"cap {sector_cap:.0%}")
    _base_layout(
        fig,
        "Sector mix by ranking arm<br><sub>Top five sectors; the tail folds into "
        "Other rather than becoming a generated hue.</sub>",
        H_PANEL,
    )
    fig.update_layout(barmode="stack")
    _fmt_axis(fig, x="ranking arm", y="book weight", y_kind="prob")
    _safe_show(fig, label="sector_mix")
    write_table(mix, "sector_mix")
    return fig


def _sector_steps(n: int) -> list[str]:
    """``n`` steps of the sequential ramp, light to dark, plus a muted "Other"."""
    try:
        import plotly.express as px

        scale = px.colors.sample_colorscale(CS_SEQ, [i / max(n - 1, 1) for i in range(n)])
    except Exception:  # pragma: no cover - defensive
        scale = [C_POSTERIOR] * n
    if n:
        scale[-1] = C_MUTED
    return scale


def plot_kelly_pin(decision: Optional[pd.DataFrame]) -> Optional[Any]:
    """The Kelly pin, as an ECDF with the mass at 1.0 called out.

    *Job: a single headline plus a distribution.* A pinned fraction is the bisection
    reporting that ``E[log(1+f·r)]`` never turned over inside the feasible interval --
    i.e. that no simulated scenario loses money. It is a statement about the forward
    simulation's left tail, not a sizing recommendation, and a column reading 1.000 for
    nine names in ten must not be readable as the latter.

    Interiority is carried by a **second trace with its own dash**, not a fourth hue.
    """
    if decision is None or not len(decision) or _requires_plotly("kelly_pin"):
        return None
    if "kelly_fraction" not in decision.columns:
        logger.info("kelly_pin needs 'kelly_fraction'; skipped", )
        return None

    arm = decision["rank_by"].iloc[0] if "rank_by" in decision.columns else None
    frame = decision if arm is None else decision[decision["rank_by"] == arm]
    vals = pd.to_numeric(frame["kelly_fraction"], errors="coerce").dropna()
    if not len(vals):
        return None
    pinned = float((vals >= 1.0 - 1e-9).mean())
    interior = (float(frame["kelly_interior"].mean())
                if "kelly_interior" in frame.columns else float("nan"))
    strictly_inside = float(((vals > 1e-9) & (vals < 1.0 - 1e-9)).mean())

    x, y = _ecdf_xy(vals.to_numpy(), n=_PPC_ECDF_GRID)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="all names",
                             line=dict(color=C_POSTERIOR, width=2)))
    if "kelly_interior" in frame.columns:
        sub = pd.to_numeric(
            frame.loc[frame["kelly_interior"].astype(bool), "kelly_fraction"],
            errors="coerce").dropna()
        if len(sub) > 1:
            xi, yi = _ecdf_xy(sub.to_numpy(), n=_PPC_ECDF_GRID)
            fig.add_trace(go.Scatter(
                x=xi, y=yi, mode="lines", name="interior solution",
                line=dict(color=C_POSTERIOR, width=2, dash="dash"),
            ))
    _add_ref_line(fig, x=1.0, kind="emphasis", annotation_text="cap")
    _base_layout(
        fig,
        f"Kelly fraction — {pinned:.1%} of the universe pinned at exactly 1.0"
        f"<br><sub>Only {strictly_inside:.1%} land strictly inside (0, 1). A pin "
        f"means E[log(1+f·r)] never turned over: no draw loses money.</sub>",
        H_PANEL,
    )
    _fmt_axis(fig, x="Kelly fraction", y="cumulative share", y_kind="prob")
    _safe_show(fig, label="kelly_pin")
    write_table(pd.DataFrame([{
        "arm": arm, "n": len(vals), "share_pinned_at_1": pinned,
        "share_strictly_interior": strictly_inside, "share_kelly_interior": interior,
    }]), "kelly_pin")
    return fig


def plot_risk_ladder(decision: Optional[pd.DataFrame],
                     arm: Optional[str] = None) -> Optional[Any]:
    """``gvar >= ges >= gtr`` per book name, against zero.

    *Job: polarity.* The post-run analysis's central finding is that the 95%-worst
    modelled terminal outcome is a **gain** for every name in the book. That is a claim
    about sign, so it is drawn against a zero line with the three statistics on one row
    per name, rather than tabulated.
    """
    if decision is None or not len(decision) or _requires_plotly("risk_ladder"):
        return None
    need = {"gvar", "ges", "gtr", "weight"}
    if not need <= set(decision.columns):
        logger.info("risk_ladder needs %s; skipped", sorted(need))
        return None
    frame = decision
    if arm is not None and "rank_by" in frame.columns:
        frame = frame[frame["rank_by"] == arm]
    elif "rank_by" in frame.columns:
        arm = frame["rank_by"].iloc[0]
        frame = frame[frame["rank_by"] == arm]
    held = frame[frame["weight"] > 0].sort_values("gvar")
    if not len(held):
        return None

    labels = held["name"].astype(str).str[:24] if "name" in held.columns \
        else held["isin"].astype(str)
    fig = go.Figure()
    for col, color, name in (("gvar", C_POSTERIOR, "GVaR (95%-worst)"),
                             ("ges", C_OBSERVED, "GES (mean of the tail)"),
                             ("gtr", C_FORECAST, "GTR (deepest quantile)")):
        fig.add_trace(go.Scatter(
            x=held[col], y=labels, mode="markers", name=name,
            marker=dict(size=9, color=color, line=dict(width=1, color=C_PANEL_EDGE)),
            hovertemplate="%{y}<br>" + name + " %{x:.2%}<extra></extra>",
        ))
    _add_ref_line(fig, x=0.0, kind="zero")
    above = int((held["gvar"] > 0).sum())
    _base_layout(
        fig,
        f"Generative risk ladder [{arm}] — GVaR above zero for {above} of {len(held)}"
        f"<br><sub>Ordered gvar ≥ ges ≥ gtr by construction. A point right of zero "
        f"means the modelled bad case is a gain.</sub>",
        _forest_height_px(len(held), per_row=22, base=220),
    )
    fig.update_layout(margin=dict(l=190, r=30, t=80, b=60))
    _fmt_axis(fig, x="terminal return", y="")
    fig.update_xaxes(tickformat=_pct_format(held[["gvar", "ges", "gtr"]].to_numpy()))
    _safe_show(fig, label="risk_ladder")
    write_table(held[["isin", "gvar", "ges", "gtr", "weight"]], "risk_ladder")
    return fig


def plot_denominator_sanity(decision: Optional[pd.DataFrame],
                            arm: Optional[str] = None) -> Optional[Any]:
    """Where the book's ranking denominators sit in the universe's distribution.

    *Job: magnitude against a reference distribution.* Half of the suite's argument.
    A reward-to-risk ratio lets **thin evidence inflate a score**, so a book selected
    on it ends up holding names whose modelled downside is orders of magnitude below
    the universe median. Drawn as a pre-binned universe density with the selected
    names as a rug and the median marked. Read beside
    :func:`plot_shrinkage_contrast`, which shows the opposite treatment of the same
    input condition.
    """
    if decision is None or not len(decision) or _requires_plotly("denominator_sanity"):
        return None
    # `rank_denominator` was a per-name copy of whichever column ranked and was
    # retired on 2026-08-27 as a duplicate; the arm's own denominator is read
    # directly. The order matches RANKING_RULES' default-first ordering.
    col = next((c for c in ("downside_dev", "tail_risk")
                if c in decision.columns), None)
    if col is None or "weight" not in decision.columns:
        logger.info("denominator_sanity found no denominator column; skipped")
        return None
    frame = decision
    if "rank_by" in frame.columns:
        arm = arm or frame["rank_by"].iloc[0]
        frame = frame[frame["rank_by"] == arm]
    universe = pd.to_numeric(frame[col], errors="coerce").dropna()
    held = pd.to_numeric(frame.loc[frame["weight"] > 0, col], errors="coerce").dropna()
    if not len(universe) or not len(held):
        logger.info("denominator_sanity: nothing to draw for arm %r", arm)
        return None

    median = float(universe.median())
    fig = go.Figure()
    _add_binned_density(fig, universe.to_numpy(), bins=90, color=C_POSTERIOR,
                        name="universe", density=True)
    fig.add_trace(go.Scatter(
        x=held.to_numpy(), y=np.zeros(len(held)), mode="markers",
        name=f"book ({len(held)})",
        marker=dict(symbol="line-ns-open", size=14, color=C_HIGHLIGHT,
                    line=dict(width=2)),
        hovertemplate=col + " %{x:.3g}<extra></extra>",
    ))
    _add_ref_line(fig, x=median, kind="anchor",
                  annotation_text=f"universe median {median:.4g}")
    ratio = median / max(float(held.max()), 1e-300)
    # The headline states what THIS run measured, and only calls it a finding when it
    # is one. A caption that repeats a conclusion from another run is precisely the
    # failure this module was written to stop: a figure describing a fit that is not
    # the one it was generated from.
    if ratio >= 10.0:
        headline = (f"every selected name sits at least {ratio:,.0f}x below the "
                    f"universe median")
        note = ("Ranking on reward-per-risk selects on the ABSENCE of modelled risk "
                "when the denominator can vanish.")
    else:
        headline = (f"the book's denominators reach {ratio:.2f}x the universe median")
        note = ("The book is NOT concentrated in the vanishing-denominator region on "
                "this run. That is the check passing, not the finding.")
    _base_layout(
        fig,
        f"Ranking denominator [{arm}] — {headline}"
        f"<br><sub>{col}, {len(universe):,} eligible names. {note}</sub>",
        H_PANEL,
    )
    # Bound the log axis to the data. Left to itself Plotly runs the decade out to 1
    # and the whole distribution collapses into the left tenth of the panel.
    lo = float(np.log10(max(float(universe.min()), 1e-12))) - 0.15
    hi = float(np.log10(max(float(universe.max()), 1e-12))) + 0.15
    fig.update_xaxes(type="log", range=[lo, hi])
    _fmt_axis(fig, x=f"{col} (log scale)", y="density")
    _safe_show(fig, label="denominator_sanity")
    write_table(pd.DataFrame([{
        "arm": arm, "column": col, "universe_median": median,
        "book_min": float(held.min()), "book_max": float(held.max()),
        "ratio_median_to_book_max": ratio,
    }]), "denominator_sanity")
    return fig


def plot_rank_agreement(agreement: Optional[pd.DataFrame]) -> Optional[Any]:
    """Pairwise membership overlap between the ranking arms.

    *Job: magnitude on an ordered pair grid.* A **sequential single hue**, light to
    dark — never a rainbow, and never a diverging scale, because an overlap count has
    no meaningful midpoint.

    The diagonal is each arm's OWN book size, read from ``n_a``/``n_b``, and the
    headline is Jaccard rather than a raw count. Both follow from breadth becoming
    an output on 2026-08-28: the arms no longer hold the same number of names, so
    there is no single ``k`` to divide by, and a raw overlap makes a small
    disciplined book look like it disagreed with a large one when it may be a
    subset of it. This panel read a ``k_book`` column that ``_book_agreement``
    stopped emitting at that change, and rendered "of None names".
    """
    if agreement is None or not len(agreement) or _requires_plotly("rank_agreement"):
        return None
    need = {"arm_a", "arm_b", "overlap"}
    if not need <= set(agreement.columns):
        return None
    arms = sorted(set(agreement["arm_a"]) | set(agreement["arm_b"]))

    # Each arm's book size, from whichever side of a pair it appears on.
    sizes: dict[str, float] = {}
    for col, n_col in (("arm_a", "n_a"), ("arm_b", "n_b")):
        if n_col in agreement.columns:
            for arm, n in zip(agreement[col], agreement[n_col]):
                if pd.notna(n):
                    sizes[str(arm)] = float(n)

    grid = pd.DataFrame(np.nan, index=arms, columns=arms, dtype="float64")
    for _, r in agreement.iterrows():
        grid.loc[r["arm_a"], r["arm_b"]] = r["overlap"]
        grid.loc[r["arm_b"], r["arm_a"]] = r["overlap"]
    for a in arms:
        grid.loc[a, a] = sizes.get(a, np.nan)

    zmax = float(np.nanmax(grid.to_numpy())) if np.isfinite(grid.to_numpy()).any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(), x=arms, y=arms, colorscale=CS_SEQ,
        zmin=0, zmax=zmax,
        text=[[("" if np.isnan(v) else f"{int(v)}") for v in row]
              for row in grid.to_numpy()],
        texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z} shared<extra></extra>",
        colorbar=dict(title="shared"),
    ))
    # Worst pair by Jaccard where we have it — the size-invariant measure — and by
    # raw overlap only as a fallback.
    if "jaccard" in agreement.columns and agreement["jaccard"].notna().any():
        worst = agreement.loc[agreement["jaccard"].idxmin()]
        headline = (f"Jaccard as low as {float(worst['jaccard']):.2f} — "
                    f"{int(worst['overlap'])} names shared of "
                    f"{int(worst['n_a'])}/{int(worst['n_b'])}")
    else:
        worst = agreement.loc[agreement["overlap"].idxmin()]
        headline = f"as few as {int(worst['overlap'])} names shared"
    _base_layout(
        fig,
        f"Ranking-arm agreement — {headline}"
        f"<br><sub>One posterior, {len(arms)} arms; the diagonal is each arm's own "
        f"book size. A low overlap says the ranking choice is underdetermined, not "
        f"that either arm is wrong.</sub>",
        H_PANEL,
    )
    _safe_show(fig, label="rank_agreement")
    write_table(agreement, "rank_agreement")
    return fig


def plot_ergodicity(curve: Optional[Any],
                    kelly: Optional[float] = None) -> Optional[Any]:
    """Terminal wealth against the bet fraction, with the Kelly peak and 3× ruin.

    One line, so no legend box — the title names it.
    """
    if curve is None or _requires_plotly("ergodicity"):
        return None
    frame = curve if isinstance(curve, pd.DataFrame) else pd.DataFrame(curve)
    xcol = next((c for c in ("fraction", "f", "kelly_fraction") if c in frame.columns), None)
    ycol = next((c for c in ("log_growth", "terminal_wealth", "growth", "wealth")
                 if c in frame.columns), None)
    if xcol is None or ycol is None or not len(frame):
        logger.info("ergodicity: no (fraction, growth) columns in the curve; skipped")
        return None

    fig = go.Figure(go.Scatter(
        x=frame[xcol], y=frame[ycol], mode="lines",
        line=dict(color=C_POSTERIOR, width=2), showlegend=False,
    ))
    peak = float(frame.loc[frame[ycol].idxmax(), xcol])
    _add_ref_line(fig, x=peak, kind="emphasis", annotation_text=f"peak f={peak:.2f}")
    if peak > 0:
        _add_ref_line(fig, x=3.0 * peak, kind="anchor", annotation_text="3× peak")
    _add_ref_line(fig, y=0.0, kind="zero")
    _base_layout(
        fig,
        "Growth against bet fraction — the peak is Kelly, and 3× it is ruin"
        "<br><sub>Overbetting is the asymmetric error: past the optimum growth "
        "falls and keeps falling.</sub>",
        H_SHORT,
    )
    _fmt_axis(fig, x="fraction of capital", y="expected log growth")
    _safe_show(fig, label="ergodicity")
    return fig


# =========================================================================== #
#  §14b  The recommendation layer                                             #
# =========================================================================== #


def plot_group_signal_forest(signals: Optional[pd.DataFrame]) -> Optional[Any]:
    """Shrunk group excess with the ±1-sd band, faceted by coordinate.

    Verdict is carried by **position and label**, and sign by the diverging rule --
    not by three categorical hues. A three-state verdict painted on a signed axis
    encodes the same information twice and reads as neither.
    """
    if signals is None or not len(signals) or _requires_plotly("group_forest"):
        return None
    need = {"level", "group", "excess_shrunk", "band", "verdict"}
    if not need <= set(signals.columns):
        return None

    levels = list(dict.fromkeys(signals["level"]))
    fig = make_subplots(rows=len(levels), cols=1, shared_xaxes=True,
                        subplot_titles=[l.replace("_", " ") for l in levels],
                        vertical_spacing=0.06)
    total_rows = 0
    for i, level in enumerate(levels, start=1):
        block = signals[signals["level"] == level].sort_values("excess_shrunk")
        total_rows += len(block)
        labels = [f"{g}  [{v[:2]}]" for g, v in zip(block["group"], block["verdict"])]
        fig.add_trace(go.Bar(
            y=labels, x=block["excess_shrunk"], orientation="h",
            marker=dict(color=block["excess_shrunk"], colorscale=CS_DIV,
                        cmid=0.0, line=dict(width=1, color=C_PANEL_EDGE)),
            showlegend=False,
            customdata=np.stack([block["excess_raw"], block["lambda_g"],
                                 block["n"]], axis=-1),
            hovertemplate=("%{y}<br>shrunk %{x:.2%} (raw %{customdata[0]:.2%}, "
                           "λ %{customdata[1]:.2f}, n %{customdata[2]})<extra></extra>"),
        ), row=i, col=1)
        band = float(block["band"].iloc[0])
        for sign in (-1.0, 1.0):
            _add_ref_line(fig, x=sign * band, kind="anchor", row=i, col=1)
        _add_ref_line(fig, x=0.0, kind="zero", row=i, col=1)

    _base_layout(
        fig,
        "Group posture — excess over the universe, shrunk by λ = τ²/(τ²+s²)"
        "<br><sub>OW/UW band is ±1 cross-group sd of the SHRUNK excess. Verdict is "
        "in the label; colour carries sign only.</sub>",
        _forest_height_px(total_rows, per_row=22, base=140 + 60 * len(levels)),
    )
    fig.update_layout(margin=dict(l=200, r=30, t=90, b=60))
    # Title the BOTTOM x-axis only. `_fmt_axis` sets every subplot's, and on a
    # shared-x stack the upper ones render inside the figure, over the panel below.
    fig.update_xaxes(tickformat=_pct_format(signals["excess_shrunk"]))
    fig.update_xaxes(title_text="shrunk excess", row=len(levels), col=1)
    _safe_show(fig, label="group_signal_forest")
    write_table(signals, "group_signals")
    return fig


def plot_shrinkage_contrast(signals: Optional[pd.DataFrame]) -> Optional[Any]:
    """Raw group excess against its shrunk value — the other half of the argument.

    *Job: the effect of an operation.* Every point lies between the y=x line and zero,
    and how far it falls short of y=x is exactly ``lambda_g``. This is the one place in
    the pipeline where **thin evidence pulls a signal toward zero** rather than
    inflating it. Read beside :func:`plot_denominator_sanity`, where the same input
    condition — a group or name the data barely resolves — produces the opposite
    result.
    """
    if signals is None or not len(signals) or _requires_plotly("shrinkage_contrast"):
        return None
    need = {"excess_raw", "excess_shrunk", "lambda_g"}
    if not need <= set(signals.columns):
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=signals["excess_raw"], y=signals["excess_shrunk"], mode="markers",
        marker=dict(size=11, color=signals["lambda_g"], colorscale=CS_SEQ,
                    cmin=0, cmax=1, line=dict(width=1, color=C_PANEL_EDGE),
                    colorbar=dict(title="λ")),
        showlegend=False,
        customdata=np.stack([signals["group"], signals["n"]], axis=-1),
        hovertemplate=("%{customdata[0]} (n %{customdata[1]})<br>"
                       "raw %{x:.2%} → shrunk %{y:.2%}<extra></extra>"),
    ))
    lim = float(np.nanmax(np.abs(signals["excess_raw"]))) * 1.1
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines", showlegend=False,
        line=dict(color=C_REF, dash="dot", width=1.2),
    ))
    _add_ref_line(fig, y=0.0, kind="zero")
    worst = signals.loc[signals["lambda_g"].idxmin()]
    _base_layout(
        fig,
        f"Shrinkage pulls thin evidence toward zero — λ as low as "
        f"{float(worst['lambda_g']):.2f} for {worst['group']}"
        f"<br><sub>Distance below y = x is the shrinkage. The opposite of what a "
        f"reward-per-risk ratio does with the same condition.</sub>",
        H_PANEL,
    )
    _fmt_axis(fig, x="raw excess", y="shrunk excess")
    _tick = _pct_format(pd.concat([signals["excess_raw"], signals["excess_shrunk"]]))
    fig.update_xaxes(tickformat=_tick)
    fig.update_yaxes(tickformat=_tick)
    _safe_show(fig, label="shrinkage_contrast")
    return fig


def plot_size_down_overlap(watch: Optional[pd.DataFrame],
                           books: Optional[pd.DataFrame]) -> Optional[Any]:
    """Which book names the size-down watch flags, at what weight, in which arm.

    A small set, so a labelled dot plot rather than a density. The veto is orthogonal
    to the ranking: a wide posterior or an analyst panel of two is what produces a
    forward simulation with no credible left tail, which is what both books select on.
    """
    if (watch is None or books is None or not len(watch) or not len(books)
            or _requires_plotly("size_down_overlap")):
        return None
    if "isin" not in watch.columns or "weight" not in books.columns:
        return None
    flagged = set(watch["isin"].astype(str))
    held = books[(books["weight"] > 0) & books["isin"].astype(str).isin(flagged)]
    if not len(held):
        logger.info("size_down_overlap: no book name is flagged; nothing to draw")
        return None

    labels = held["name"].astype(str).str[:24] if "name" in held.columns \
        else held["isin"].astype(str)
    fig = go.Figure()
    for arm in dict.fromkeys(held["rank_by"]) if "rank_by" in held.columns else [None]:
        block = held if arm is None else held[held["rank_by"] == arm]
        fig.add_trace(go.Scatter(
            x=block["weight"],
            y=(labels.loc[block.index] if arm is not None else labels),
            mode="markers", name=str(arm),
            marker=dict(size=11, color=_arm_color(str(arm)),
                        line=dict(width=1, color=C_PANEL_EDGE)),
            hovertemplate="%{y}<br>weight %{x:.2%}<extra></extra>",
        ))
    _base_layout(
        fig,
        f"Size-down watch ∩ sized book — {held['isin'].nunique()} names flagged"
        f"<br><sub>Wide posterior or an analyst panel of two. A veto orthogonal to "
        f"the ranking, reported rather than applied.</sub>",
        _forest_height_px(held["isin"].nunique(), per_row=24, base=200),
    )
    fig.update_layout(margin=dict(l=190, r=30, t=80, b=60))
    _fmt_axis(fig, x="book weight", y="")
    fig.update_xaxes(tickformat=_pct_format(held["weight"]))
    _safe_show(fig, label="size_down_overlap")
    write_table(held, "size_down_overlap")
    return fig


# =========================================================================== #
#  Wiring                                                                     #
# =========================================================================== #



#: The action ladder, most bullish first, with the colour each rung takes.
#: Ordered here rather than sorted at draw time so a rung that is EMPTY on a run
#: still occupies its slot -- a five-point scale that silently renders as three
#: bars is the same illegibility the wider vocabulary was meant to remove.
_ACTION_LADDER: tuple[tuple[str, str], ...] = (
    ("STRONG BUY", C_POSTERIOR),
    ("BUY", C_FORECAST),
    ("HOLD", C_MUTED),
    ("SELL", C_OBSERVED),
    ("STRONG SELL", C_HIGHLIGHT),
)


def plot_action_ladder(actions: Optional[pd.DataFrame]) -> Optional[Any]:
    """The five-rung action distribution, and where its gates fall.

    *Job: part-to-whole over an ORDERED category.* A single horizontal bar per
    rung, in ladder order, on a diverging bull-to-bear ramp -- not a pie, which
    cannot show order, and not a sorted bar chart, which would destroy it.

    **What this panel is for.** The three-valued list it replaces returned 83.5 %
    ``BUY`` on run ``807df55e7158`` and nothing said so; it took a human reading
    a table. Five rungs do not fix that by themselves, because the gates are
    scaled by the universe-mean confidence and a low mean pulls the STRONG
    threshold down onto the ordinary one. So the gate positions are annotated on
    the panel: a top rung holding most of the universe, with its gate sitting at
    the 28th percentile of the probability distribution, is a statement about the
    forward simulation's left tail rather than about conviction.
    """
    if actions is None or not len(actions) or _requires_plotly("action_ladder"):
        return None
    if "action" not in actions.columns:
        logger.info("action_ladder needs 'action'; skipped")
        return None

    counts = actions["action"].value_counts()
    total = int(len(actions))
    labels = [a for a, _ in _ACTION_LADDER]
    values = [int(counts.get(a, 0)) for a in labels]
    colors = [c for _, c in _ACTION_LADDER]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:,} ({v / total:.1%})" for v in values],
        textposition="outside",
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.update_yaxes(autorange="reversed")  # most bullish at the top

    gate_bits = [
        f"{c}={float(actions[c].iloc[0]):.3f}"
        for c in ("gate_strong_hi", "gate_hi", "gate_lo", "gate_strong_lo")
        if c in actions.columns and pd.notna(actions[c].iloc[0])
    ]
    subtitle = f"{total:,} names"
    if gate_bits:
        subtitle += "   ·   " + "  ".join(gate_bits)
    if "consensus_gap" in actions.columns:
        gap = pd.to_numeric(actions["consensus_gap"], errors="coerce")
        if gap.notna().any():
            subtitle += (f"   ·   median gap vs analyst consensus "
                         f"{gap.median():+.2f} on the 1-5 scale")

    _base_layout(
        fig,
        f"Name actions — {subtitle}",
        height=340,
    )
    fig.update_xaxes(title_text="names")
    _safe_show(fig, label="action_ladder")

    table = pd.DataFrame({
        "action": labels,
        "n": values,
        "share": [v / total for v in values],
    })
    write_table(table, "action_ladder")
    return fig


def plot_consensus_gap(actions: Optional[pd.DataFrame]) -> Optional[Any]:
    """How far the model's action departs from the analyst panel's own rating.

    *Job: distribution about a meaningful zero.* A binned density (never
    ``go.Histogram`` — it ships every raw value into the notebook) with the
    zero line marked as reference geometry.

    Zero means "this model agrees with consensus". The screen reproduces the
    consensus ORDERING at Spearman 0.992, so a gap distribution tightly centred
    on zero says the extra structure is risk, not a different view of value —
    which is the question §4 of the published analysis could not put a number on.
    """
    if actions is None or not len(actions) or _requires_plotly("consensus_gap"):
        return None
    if "consensus_gap" not in actions.columns:
        logger.info(
            "consensus_gap absent: the screen carries no analyst rating. "
            "Re-export the screen, or run the replay with the panel frame present."
        )
        return None
    gap = pd.to_numeric(actions["consensus_gap"], errors="coerce").dropna()
    if not len(gap):
        return None

    fig = go.Figure()
    _add_binned_density(fig, gap.to_numpy(), name="consensus gap",
                        color=C_POSTERIOR, density=False)
    _add_ref_line(fig, x=0.0, kind="zero",
                  annotation_text="agrees with consensus")
    _base_layout(
        fig,
        f"Model action minus analyst rating — median {gap.median():+.2f}, "
        f"{float((gap > 0).mean()):.0%} more bullish than the panel",
        height=380,
    )
    fig.update_xaxes(title_text="action_score − analyst_rating (1–5 scale)")
    fig.update_yaxes(title_text="names")
    _safe_show(fig, label="consensus_gap")
    return fig

def _attempt(label: str, fn, *args, **kwargs) -> None:
    """Run one panel, log a failure, and never re-raise."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - figures are best-effort
        logger.warning("panel %s failed: %s", label, exc,
                       exc_info=logger.isEnabledFor(logging.DEBUG))


def render_replay(result: dict[str, Any], cfg: Any = None) -> None:
    """Draw every panel this replay has the inputs for.

    One call, placed AFTER the export so a figure can never delay or endanger it.
    Each panel is attempted independently and is a no-op when its input is absent, so
    a partial replay and a full one both work through this entry point without a
    matrix of guards at the call site.

    Parameters
    ----------
    result
        The dict :func:`kalman_portfolio.main` returns.
    cfg
        The :class:`KalmanPortfolioConfig`, read only for ``sector_cap`` and the
        shipped prior values that anchor the sweep panels.
    """
    if not get_export_state().enabled:
        logger.info("figures disabled; skipping the panel set")
        return

    forecast = result.get("forecast") or {}
    sweeps = result.get("sweeps") or {}
    decision = result.get("decision") or {}
    recommendations = result.get("recommendations") or {}
    books = decision.get("decision_frame")
    shipped_share = getattr(cfg, "factor_share", 0.35)
    sector_cap = getattr(cfg, "sector_cap", None)

    # The REPLAY's own sections, not the fit's. `15_forecast` and `15b_decision`
    # belong to the v2 workflow, which writes them once per fit; this script runs
    # many times over one fit, so its panels landing there would overwrite the
    # fit's own artifacts with a replay's -- and a reader browsing the tree could
    # not tell which had produced what. The figures now sit beside the frames they
    # were drawn from.
    with section("15c_forecast"):
        _attempt("engine_contrast", plot_engine_contrast, forecast.get("engines"))

    with section("15d_sweeps"):
        _attempt("factor_sweep", plot_factor_sweep, sweeps.get("factor_share"),
                 shipped_share)
        _attempt("multiplier_sweep", plot_multiplier_sweep, sweeps.get("multiplier"))

    with section("15e_books"):
        _attempt("two_books", plot_two_books, books)
        _attempt("sector_mix", plot_sector_mix, books, "sector", sector_cap)
        _attempt("kelly_pin", plot_kelly_pin, books)
        _attempt("risk_ladder", plot_risk_ladder, books)
        _attempt("denominator_sanity", plot_denominator_sanity, books)
        _attempt("rank_agreement", plot_rank_agreement, decision.get("agreement"))
        _attempt("ergodicity", plot_ergodicity, decision.get("wealth_curve"))

    with section("14b_recommendations"):
        signals = recommendations.get("group_signals")
        _attempt("group_signal_forest", plot_group_signal_forest, signals)
        _attempt("shrinkage_contrast", plot_shrinkage_contrast, signals)
        _attempt("size_down_overlap", plot_size_down_overlap,
                 recommendations.get("watch"), books)
        actions = recommendations.get("actions")
        _attempt("action_ladder", plot_action_ladder, actions)
        _attempt("consensus_gap", plot_consensus_gap, actions)

    logger.info("figures written under %s", get_export_state().root)
