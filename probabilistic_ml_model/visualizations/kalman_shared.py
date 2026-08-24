"""Shared figure primitives for the Kalman price-target workflows.

The single source of truth for how a Kalman figure is themed, budgeted, drawn
and filed. Both ``pymc_kalman_filter_pt.py`` (v1) and ``kalman_viz_v2.py`` (v2)
render through the helpers here, so a panel cannot look one way in one workflow
and another way in the other, and a payload rule cannot be honoured in one and
forgotten in the next.

Extracted from v1 on 2026-08-24 as a **move**, not a rewrite: every definition
below is the v1 definition, and v1 imports them straight back under the same
private names. A behavioural difference between this module and the v1 it came
from is a defect, not an improvement.

Three SSOTs live here, and CLAUDE.md treats each as normative:

``_PLOTLY_TEMPLATE`` / :func:`_apply_dark_template`
    One template, applied in exactly one place -- the :func:`_safe_show` funnel
    -- so a displayed figure and its exported PNG cannot diverge.

The figure payload budget
    Plotly serialises every coordinate it draws. Use
    :func:`_binned_density_trace` / :func:`_add_binned_density` for densities
    (measured 87.6 MB vs 9.0 KB for 6.5 M values against a raw ``go.Histogram``),
    :func:`_ecdf_xy` for ECDFs (42.2 MB vs 0.86 MB), :func:`_decimate_frame` for
    full-universe scatters, and :func:`_azp_backend` -- never a ``backend=``
    literal -- for arviz-plots. A v1 notebook once reached 233 MB, 207.7 MB of it
    in a single prior-predictive figure, by ignoring these.

``_REF_LINE_KINDS`` / :func:`_add_ref_line` / :func:`_add_ref_band`
    Reference geometry is keyed on a ROLE (``zero`` / ``anchor`` / ``emphasis``),
    never drawn with a bare ``add_hline`` / ``add_vline`` / ``add_vrect``.

Configuration seam
------------------
Three helpers need run-level settings -- :func:`_display_width_px` (figure
width), :func:`_decimate_frame` (RNG seed) and :func:`get_export_state`
(artifact root). In v1 these read the ``KalmanRunConfig`` singleton directly,
which is what makes ``main(config=...)`` actually redirect artifacts. That
coupling is preserved rather than copied: :func:`set_viz_config_resolver`
installs a *callable* returning the live config, so v1 hands over
``get_run_config`` itself and behaves exactly as before, while v2 hands over its
own ``KalmanRunConfigV2`` accessor. Copying values at import time would have
silently broken ``main(config=...)``.

Notes
-----
Import-time side effects are deliberately limited to the Matplotlib backend
selection, which no-ops inside a notebook kernel, under an explicit
``MPLBACKEND``, and where no display is available -- v2 is a headless CLI and
must not open a Tk window on a server.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import sys
import uuid
import warnings
from dataclasses import dataclass, field
import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import matplotlib
import matplotlib.figure


def _in_ipython_kernel() -> bool:
    """True when running inside a Jupyter / IPython kernel (not a plain script).

    ``matplotlib.use("TkAgg")`` below is right for the script path -- a native
    window beats PyCharm's plot tool -- but inside a notebook kernel it HIJACKS
    the inline backend: figures open in an off-screen Tk window and the cell
    renders nothing. That is invisible while every panel is Plotly, and becomes a
    hard blocker the moment an arviz-plots panel is routed to the matplotlib
    backend (see :data:`_AZP_BACKEND_HEAVY`), so the switch is conditional.
    """
    if 'ipykernel' in sys.modules:
        return True
    ipy = sys.modules.get('IPython')
    return ipy is not None and getattr(ipy, 'get_ipython', lambda: None)() is not None


def _want_interactive_backend() -> bool:
    """True when a native Matplotlib window is the right default.

    Three ways it is not, and v2 hits the last two routinely:

    * inside a notebook kernel -- see :func:`_in_ipython_kernel`;
    * when ``MPLBACKEND`` is set, because the caller has already chosen;
    * with no display available (a headless CLI or a server), where ``TkAgg``
      raises at first use rather than at selection, so the failure surfaces
      minutes into a run instead of at import.
    """
    if _in_ipython_kernel() or os.environ.get('MPLBACKEND'):
        return False
    if sys.platform.startswith('win') or sys.platform == 'darwin':
        return True
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


if _want_interactive_backend():
    with contextlib.suppress(Exception):
        matplotlib.use("TkAgg")  # native window instead of PyCharm's plot tool

import matplotlib.colors as _mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import arviz_plots as azp
import arviz_stats as azs
from arviz_base import rcParams as _az_rcparams
from cycler import cycler as _cycler

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover - optional dependency
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]
    _HAS_PLOTLY = False

logger = logging.getLogger(__name__)


# --- Configuration seam ------------------------------------------------------
class VizConfigLike(Protocol):
    """The three run-level settings the figure layer reads.

    Structural, not nominal: ``KalmanRunConfig`` and ``KalmanRunConfigV2`` both
    satisfy it without either importing this module.
    """

    fig_width_px: int
    random_seed: int
    results_dir: Optional[str]


@dataclass(frozen=True)
class VizConfig:
    """Fallback settings used until a resolver is installed."""

    fig_width_px: int = 1000
    random_seed: int = 42
    results_dir: Optional[str] = None


_viz_config_resolver: Optional[Callable[[], Any]] = None


def set_viz_config_resolver(resolver: Optional[Callable[[], Any]]) -> None:
    """Install the callable this module reads run settings from.

    Parameters
    ----------
    resolver
        Zero-argument callable returning an object with ``fig_width_px``,
        ``random_seed`` and ``results_dir``. Pass ``None`` to fall back to
        :class:`VizConfig` defaults.

    Notes
    -----
    A CALLABLE, not a value. v1's ``KalmanRunConfig`` is a live singleton that
    ``main(config=...)`` replaces, so reading it once at import would freeze the
    figure width and the artifact root at their defaults and quietly ignore every
    later override. v1 installs its own ``get_run_config`` here and therefore
    behaves exactly as it did before the extraction.
    """
    global _viz_config_resolver
    _viz_config_resolver = resolver


def _project_root() -> Path:
    """Return the repository root.

    This module sits at ``probabilistic_ml_model/visualizations/``, two levels
    below the root that both workflows resolve relative paths against. Every
    caller that used to write ``Path(__file__).resolve().parent`` in a root-level
    script must come here instead, or a relative ``KALMAN_PT_RESULTS_DIR`` lands
    inside the package.
    """
    return Path(__file__).resolve().parents[2]


def get_viz_config() -> Any:
    """Return the live run settings, or :class:`VizConfig` defaults."""
    if _viz_config_resolver is not None:
        try:
            cfg = _viz_config_resolver()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug('viz config resolver failed (%s); using defaults', exc)
        else:
            if cfg is not None:
                return cfg
    return VizConfig()


# --- Figure sizing bounds ----------------------------------------------------
# Clamp for :func:`_display_width_px`. Declared as a tuple assignment in v1,
# which is how the extraction's dependency walk missed them.
_FIG_WIDTH_MIN_PX, _FIG_WIDTH_MAX_PX = 700, 2200

# --- Semantic series palette (single source of truth for every figure) --------
# An Okabe-Ito-ish convention grown organically across v1's sections; declared
# here so every panel in BOTH workflows keys colours on ROLE, not on a
# per-section hex. Reference geometry (0-lines, y=x guides, now-boundaries) is
# always C_REF and is drawn through :func:`_add_ref_line`.
C_POSTERIOR = '#56b4e9'    # posterior / model / expected series
C_OBSERVED = '#ffb000'     # observed / raw consensus series
C_FORECAST = '#cc79a7'     # forward-looking forecast bands / lines
C_VOL = '#ff7f0e'          # volatility paths (sigma_obs, SV panels)
C_DRAWS = '#4daf4a'        # posterior-draw spaghetti / clouds
C_ACCENT = '#2ca02c'       # secondary accent (scale panels, tertiary KDE)
C_MUTED = '#7f7f7f'        # de-emphasised context points

# --- Continuous colour ramps -------------------------------------------------
# One sequential and one diverging ramp for both workflows: v1's panels had
# drifted across Viridis / Magma / flare (sequential) and vlag (diverging), so
# two views of the same quantity read as different measurements. ``*_MPL`` are
# the matplotlib / seaborn spellings of the same two ramps.
CS_SEQ = 'Viridis'         # magnitude-only quantities (beta, STARR, kalman gain)
CS_DIV = 'RdBu'            # signed quantities centred on zero
CS_SEQ_MPL = 'viridis'
CS_DIV_MPL = 'vlag'

# --- Payload caps (CLAUDE.md figure budget) ----------------------------------
# A full-universe scatter must be decimated with :func:`_decimate_frame` and the
# sampled count stated in the title. A rank-based cut is NOT an alternative:
# when it binds it deletes one tail, and the surviving cloud misrepresents the
# screen.
_EDA_SCATTER_MAX_POINTS = 1200
_SCREEN_SCATTER_MAX_POINTS = 2500
#: Grid size for :func:`_ecdf_xy`. One point per observation costs 42.2 MB
#: against 0.86 MB for the same curve on this grid.
_PPC_ECDF_GRID = 512


# --- Front-end display shim --------------------------------------------------
# ``display`` exists only in IPython; fall back to ``print`` for plain-script
# runs. v1 wraps this in its own ``display()`` that additionally snapshots
# DataFrames to CSV; that wrapper stays in v1 because it reaches into the
# analytics exporter, and only pandas objects are captured by it -- so a figure
# routed through either lands identically.
try:  # pragma: no cover - depends on runtime
    from IPython.display import display as _display_impl
except ImportError:  # pragma: no cover
    def _display_impl(obj: object) -> None:
        print(obj)


# =============================================================================
# 1. Plotting setup + reusable helpers
# =============================================================================
# --- Dark theme (single source of truth) --------------------------------------
# One Plotly template governs every figure in this module. It is applied in
# exactly one place at render time — :func:`_apply_dark_template`, called from
# the :func:`_safe_show` funnel — so a displayed figure and its exported PNG can
# never diverge. arviz-plots 1.x renamed the dark style, hence the candidate
# list; :func:`setup_plotting` warns when neither resolves instead of silently
# leaving PlotCollection figures on the light default.
_PLOTLY_TEMPLATE = 'arviz-tumma'


_ARVIZ_STYLE_CANDIDATES: tuple[str, ...] = ('arviz-tumma', 'arviz-variat')


C_PANEL_BG = '#1e1e1e'     # figure / paper background (matplotlib + marker edges)


C_AXES_BG = '#2a2a2a'      # axes (plot area) background


# --- arviz-plots backend policy (single source of truth) ----------------------
# Plotly serialises every coordinate it draws into the notebook. That is a fair
# price for a panel a reader hovers over, and a ruinous one for the diagnostic
# facet grids, which fan ONE FACET PER VECTOR ELEMENT (``beta`` is ~17-40) and
# draw every chain's full draw sequence in each — a grid budgeted up to 6200 px
# by :func:`_facet_grid_height_px`. Those panels are read, not interrogated, so
# they render through the matplotlib backend and land in the notebook as a single
# raster instead of tens of megabytes of JSON.
#
# Resolve every ``azp.plot_*`` backend through :func:`_azp_backend` rather than a
# literal, so the split is one decision in one place. ``PML_AZP_HEAVY_BACKEND``
# overrides it (set to ``plotly`` to get interactivity back for a debugging run).
_AZP_BACKEND_LIGHT = 'plotly'        # bounded, hover-worthy panels


_AZP_BACKEND_HEAVY = 'matplotlib'    # facet grids / draw-dense panels


_AZP_HEAVY_BACKEND_ENV = 'PML_AZP_HEAVY_BACKEND'


def _azp_backend(*, heavy: bool = False) -> str:
    """Resolve the arviz-plots backend for a panel.

    Parameters
    ----------
    heavy
        ``True`` for facet grids and draw-dense panels (trace, rank-dist,
        prior-posterior, ESS evolution, PPC t-stat) — anything whose payload
        scales with ``chains x draws x n_elements``. ``False`` for bounded
        panels, which stay interactive.

    Returns
    -------
    str
        ``'matplotlib'`` or ``'plotly'``.
    """
    if not heavy:
        return _AZP_BACKEND_LIGHT
    override = os.environ.get(_AZP_HEAVY_BACKEND_ENV, '').strip().lower()
    return override if override in ('plotly', 'matplotlib', 'none') else _AZP_BACKEND_HEAVY


def setup_plotting() -> None:
    """Set the default arviz-plots backend and install the dark notebook theme.

    ArviZ figures render through **Plotly by default and Matplotlib where the panel
    is draw-dense** — every ``azp.plot_*`` call resolves its backend through
    :func:`_azp_backend`, which is the SSOT for that split (see
    :data:`_AZP_BACKEND_HEAVY` for why the heavy panels are raster). The
    ``arviz_base.rcParams['plot.backend']`` default set below is the fallback for
    any call that passes no ``backend`` at all. (Assigning ``azp.backend`` has no
    effect: ``arviz_plots.backend`` is a subpackage, so the attribute assignment
    merely shadows the module and never reaches the plotting layer.) The dark
    ``arviz-tumma`` Plotly template is registered as the default so composed
    collections inherit it.

    Notes
    -----
    The Matplotlib dark theme installed below is therefore no longer only for the
    residual hand-built Matplotlib / Seaborn panels (e.g.
    :func:`plot_kalman_forecast`, the §2.4 observation-noise density panels and the
    per-sector error-bar comparisons) — the heavy arviz-plots grids land on it too,
    and :func:`_apply_mpl_dark` re-asserts it at display time for collections that
    arviz composes against its own style. seaborn installs its colour cycle as RGB
    tuples; those are re-expressed as hex strings so any Matplotlib artist that
    reshapes the active cycle behaves under the theme.
    """
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Fallback for any azp.plot_* / PlotCollection call that omits ``backend``.
    # Call sites that care resolve it explicitly through _azp_backend().
    _az_rcparams['plot.backend'] = _AZP_BACKEND_LIGHT
    try:
        import plotly.io as _pio
        _pio.templates.default = _PLOTLY_TEMPLATE  # dark ArviZ Plotly template
    except Exception:  # pragma: no cover - plotly optional / renderer-less env
        pass

    # Matplotlib / Seaborn dark theme for the residual hand-built (non-ArviZ) panels.
    plt.style.use('dark_background')
    # A silent failure here used to leave arviz_plots on its light default while
    # everything else went dark -- warn instead, and accept the 1.x rename.
    for _style in _ARVIZ_STYLE_CANDIDATES:
        try:
            azp.style.use(_style)
            break
        except (OSError, ValueError, AttributeError):
            continue
    else:
        logger.warning(
            "No arviz-plots dark style available (tried %s); PlotCollection "
            "figures fall back to the library default and are re-themed at "
            "display time by _apply_dark_template.",
            ', '.join(_ARVIZ_STYLE_CANDIDATES))
    sns.set_theme(style='darkgrid', context='notebook',
                  rc={
                      'figure.facecolor': C_PANEL_BG,
                      'axes.facecolor': C_AXES_BG,
                      'savefig.facecolor': C_PANEL_BG,
                      'axes.edgecolor': '#cccccc',
                      'axes.labelcolor': '#e6e6e6',
                      'xtick.color': '#e6e6e6',
                      'ytick.color': '#e6e6e6',
                      'text.color': '#e6e6e6',
                      'grid.color': '#555555',
                  })

    _cycle_cols = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if _cycle_cols and not all(isinstance(_c, str) for _c in _cycle_cols):
        plt.rcParams['axes.prop_cycle'] = _cycler(
            color=[_mcolors.to_hex(_c) for _c in _cycle_cols]
        )
    plt.rcParams['figure.dpi'] = 110


# --- Figure sizing (single source of truth for every panel) ------------------
# Target display width in pixels. Override with ``PML_FIG_WIDTH_PX`` to match
# the editor / notebook pane. Every ArviZ (arviz_plots), Plotly and matplotlib
# panel derives its size from this one knob; Plotly panels additionally
# autosize to the rendered container width (see :func:`_autosize_plotly`), so
# the pixel width is only the static-export / fallback size.
_FIG_WIDTH_DEFAULT_PX = 1150


C_REF = '#bbbbbb'          # reference lines (zero, y=x, anchors)


C_HIGHLIGHT = '#e69f00'    # emphasised subset (held book, key feature)


def _display_width_px() -> int:
    """Resolve the target figure width (px) from the run config (``PML_FIG_WIDTH_PX``).

    Falls back to :data:`_FIG_WIDTH_DEFAULT_PX` and clamps to
    ``[_FIG_WIDTH_MIN_PX, _FIG_WIDTH_MAX_PX]`` so a typo cannot produce an
    unreadable sliver or a multi-screen banner.
    """
    width = get_viz_config().fig_width_px
    if width is None:
        width = _FIG_WIDTH_DEFAULT_PX
    return int(np.clip(width, _FIG_WIDTH_MIN_PX, _FIG_WIDTH_MAX_PX))


def _azp_figure_kwargs(height_px: float, *, width_frac: float = 1.0) -> dict[str, Any]:
    """``figure_kwargs`` for ``azp.plot_*`` with sizes declared in **dots**.

    Passing ``figsize_units='dots'`` silences the arviz-plots Plotly backend's
    ``"Assuming dpi=100"`` UserWarning (its default interprets ``figsize`` in
    inches) and pins an explicit pixel geometry. ``width_frac`` shrinks narrow
    single-axis panels (KDEs, PIT ECDFs) below the full display width.
    """
    width = int(round(_display_width_px() * float(np.clip(width_frac, 0.2, 1.0))))
    return {'figsize': (width, int(round(height_px))), 'figsize_units': 'dots'}


def _forest_height_px(n_rows: int, *, per_row: int = 26, base: int = 190,
                      min_px: int = 320, max_px: int = 1650) -> int:
    """Height (px) for row-oriented panels (forest / ridge) from their row count.

    Guarantees a readable per-row band so long forests scroll instead of
    overlapping their labels, while degenerate 1–2 row panels keep a sane
    minimum aspect.
    """
    return int(np.clip(max(int(n_rows), 1) * per_row + base, min_px, max_px))


# --- Diagnostic facet-grid sizing (SSOT for the §9 run_diagnostics panels) ----
# arviz-plots fans a vector variable (e.g. the ~40-element ``beta`` drift-slope
# vector) into one facet per element and wraps the facets into a grid, so panel
# heights must budget for the FACET count, not the variable count. Every §9
# grid (trace / rank-dist / prior-posterior / ESS evolution) derives its
# geometry from these two knobs so the diagnostics share one dynamic rule.
_DIAG_FACET_COLS = 4  # facet-wrap width of the wrapped diagnostic grids


_DIAG_ROW_PX = 260    # vertical budget per grid row (panel + axes + title)


def _n_facets(posterior, var_names: Sequence[str]) -> int:
    """Total non-sample elements (= plot facets) across ``var_names``.

    Scalars count 1; a vector variable contributes one facet per element of
    its non-sample dims (e.g. ``beta`` over ``drift_feature``). Names absent
    from ``posterior`` are skipped.
    """
    total = 0
    for v in var_names:
        if v not in posterior.data_vars:
            continue
        da = posterior[v]
        total += int(np.prod([da.sizes[d] for d in da.dims
                              if d not in ('chain', 'draw')], initial=1))
    return total


def _facet_grid_height_px(n_facets: int, *, cols: int = _DIAG_FACET_COLS,
                          per_row: int = _DIAG_ROW_PX, base: int = 170,
                          min_px: int = 420, max_px: int = 6200) -> int:
    """Height (px) for a wrapped facet grid from its facet count.

    ``cols`` is the facet-wrap width. The raised ``max_px`` (vs
    :func:`_forest_height_px`) lets long grids scroll in the notebook instead
    of compressing dozens of facets into a fixed-height banner.
    """
    n_rows = int(np.ceil(max(int(n_facets), 1) / max(int(cols), 1)))
    return int(np.clip(n_rows * per_row + base, min_px, max_px))


def _diag_figure_kwargs(posterior, var_names: Sequence[str],
                        *, cols: int = _DIAG_FACET_COLS,
                        per_row: int = _DIAG_ROW_PX) -> dict[str, Any]:
    """Standardised ``figure_kwargs`` for the §9 diagnostic facet grids."""
    return _azp_figure_kwargs(
        _facet_grid_height_px(_n_facets(posterior, var_names),
                              cols=cols, per_row=per_row))


_RANK_DIST_ROW_PX = 220  # vertical budget per compact rank-dist row (dist | rank)


def _rank_dist_figure_kwargs(var_names: Sequence[str]) -> dict[str, Any]:
    """``figure_kwargs`` for the §9.3b compact ``plot_rank_dist`` panels.

    ``azp.plot_rank_dist(compact=True)`` pins exactly one grid row per
    *variable* (``rows=['__variable__']``, dist | rank columns) and overlays a
    vector variable's elements within that single row — unlike ``plot_trace``,
    which fans elements into wrapped facets. Height must therefore budget the
    variable count, not the element count :func:`_n_facets` reports (element
    sizing inflated a one-row vector panel to ~2800 px).
    """
    return _azp_figure_kwargs(
        _facet_grid_height_px(len(var_names), cols=1,
                              per_row=_RANK_DIST_ROW_PX, base=150, min_px=360))


def _polish_facet_axes(pc) -> None:
    """Re-enable the tick labels Plotly culls on small facets; pad the margins.

    Best-effort cosmetic pass shared by the §9 grids so every facet keeps
    visible x/y axes regardless of how many panels the grid wraps.
    """
    fig = _plotly_figure_of(pc)
    if fig is None:
        return
    fig.update_xaxes(showticklabels=True, tickfont=dict(size=9))
    fig.update_yaxes(showticklabels=True, tickfont=dict(size=9))
    fig.update_layout(margin=dict(t=70, b=50))


def _mpl_figsize(height_frac: float = 0.45, *, width_frac: float = 1.0) -> tuple[float, float]:
    """Matplotlib ``figsize`` (inches) derived from the shared pixel width.

    ``height_frac`` is the height:width aspect ratio; the pixel width comes from
    :func:`_display_width_px` divided by the active ``figure.dpi`` so mpl panels
    track the same editor-width knob as the Plotly ones.
    """
    dpi = float(plt.rcParams.get('figure.dpi', 100.0)) or 100.0
    w_in = _display_width_px() * float(np.clip(width_frac, 0.2, 1.0)) / dpi
    return w_in, max(2.0, w_in * float(height_frac))


def _plotly_figure_of(obj) -> Optional[object]:
    """Best-effort: the underlying Plotly figure of ``obj``.

    Accepts a raw :class:`plotly.graph_objects.Figure` (returned as-is) or an
    ``arviz_plots.PlotCollection``, whose ``viz`` DataTree stores either a
    root-level ``figure`` node or per-variable ``plot`` targets carrying a
    ``.figure`` attribute (the ``PlotlyPlot`` wrapper). Returns ``None`` when no
    figure can be resolved — callers must treat that as a silent no-op.
    """
    if hasattr(obj, 'update_layout'):
        return obj
    viz = getattr(obj, 'viz', None)
    if viz is None:
        return None
    for key in ('figure', 'chart'):
        try:
            fig = viz[key].item()
            if hasattr(fig, 'update_layout'):
                return fig
        except Exception:
            continue
    try:
        targets = np.asarray(viz['plot'].values).ravel()
        for target in targets:
            fig = getattr(target, 'figure', None)
            if hasattr(fig, 'update_layout'):
                return fig
    except Exception:
        pass
    return None


def _mpl_figure_of(obj) -> Optional[matplotlib.figure.Figure]:
    """Best-effort: the underlying Matplotlib figure of ``obj``.

    The Matplotlib mirror of :func:`_plotly_figure_of`. An ``arviz_plots``
    PlotCollection built on the matplotlib backend stores its ``Figure`` in the
    same ``viz['figure']`` node the Plotly backend uses for its own figure, so the
    two resolvers differ only in the attribute they duck-type on (``savefig`` /
    ``add_subplot`` rather than ``update_layout``). Returns ``None`` when ``obj``
    is not — and does not wrap — a matplotlib figure.
    """
    if hasattr(obj, 'savefig') and hasattr(obj, 'add_subplot'):
        return obj
    viz = getattr(obj, 'viz', None)
    if viz is None:
        return None
    for key in ('figure', 'chart'):
        try:
            fig = viz[key].item()
            if hasattr(fig, 'savefig') and hasattr(fig, 'add_subplot'):
                return fig
        except Exception:
            continue
    try:
        targets = np.asarray(viz['plot'].values).ravel()
        for target in targets:
            fig = getattr(target, 'figure', None)
            if hasattr(fig, 'savefig') and hasattr(fig, 'add_subplot'):
                return fig
    except Exception:
        pass
    return None


def _apply_mpl_dark(fig) -> None:
    """Stamp the dark panel/axes colours onto a Matplotlib figure.

    ``setup_plotting`` installs the dark theme through rcParams, but an
    ``arviz_plots`` matplotlib PlotCollection composes some figures against its own
    style, so a panel can reach the display funnel with a white canvas — the exact
    divergence :func:`_apply_dark_template` exists to prevent on the Plotly side.
    Best-effort: any failure leaves the figure as built.
    """
    try:
        fig.patch.set_facecolor(C_PANEL_BG)
        for ax in fig.get_axes():
            ax.set_facecolor(C_AXES_BG)
    except Exception:  # pragma: no cover - cosmetic only
        pass


def _autosize_plotly(obj) -> None:
    """Let a Plotly figure / PlotCollection stretch to the notebook/editor width.

    Plotly treats an explicit ``layout.width`` as fixed, so the width stamped by
    arviz-plots (or a hand-built panel) freezes the figure regardless of the
    rendering pane. Clearing it and setting ``autosize=True`` makes the HTML
    renderer track the container width while the explicit **height** (which
    encodes per-row readability) is preserved. Best-effort: any failure leaves
    the figure exactly as built.
    """
    try:
        fig = _plotly_figure_of(obj)
        if fig is None:
            return
        fig.layout.width = None
        fig.update_layout(autosize=True)
    except Exception:  # pragma: no cover - cosmetic only
        pass


def _apply_dark_template(obj: object) -> None:
    """Stamp the dark :data:`_PLOTLY_TEMPLATE` onto a figure or PlotCollection.

    ``plotly.io.templates.default`` only applies to figures built *after* it is
    pinned, and ``arviz_plots`` composes some collections against its own
    default — so figures reaching the display funnel can still carry the stock
    light template. Applying it here, in the one funnel every figure passes
    through, is what keeps displayed and exported output identical. Best-effort:
    any failure leaves the figure exactly as built.
    """
    try:
        fig = _plotly_figure_of(obj)
        if fig is not None:
            fig.update_layout(template=_PLOTLY_TEMPLATE)
            return
    except Exception:  # pragma: no cover - cosmetic only
        pass
    # Matplotlib-backed panels (the heavy arviz-plots grids) get the equivalent
    # treatment, so the funnel's displayed-==-exported invariant holds for both
    # backends rather than silently no-opping on one of them.
    mfig = _mpl_figure_of(obj)
    if mfig is not None:
        _apply_mpl_dark(mfig)


def _render_plotly(fig: object, *, height: Optional[int] = None,
                   label: Optional[str] = None,
                   hovermode: Optional[str] = None) -> None:
    """Render a Plotly figure in the notebook (side-effecting; dark theme).

    Sets the shared margins and forwards to :func:`_safe_show`, which applies the
    dark template (:func:`_apply_dark_template`) for display *and* export alike.
    Mirrors the ``pc.show()`` convention used throughout the module; falls back to
    the front-end display shim and is a silent no-op when no renderer is
    available, so a headless / plain-script run never raises. ``label`` is forwarded to the
    artifact exporter via :func:`_safe_show`.

    ``hovermode`` overrides Plotly's default ``'closest'`` — pass ``'x unified'``
    for time-series panels so all series report at the hovered date.
    """
    if fig is None:
        return
    fig.update_layout(margin=dict(l=60, r=30, t=70, b=50))
    if height is not None:
        fig.update_layout(height=height)
    if hovermode is not None:
        fig.update_layout(hovermode=hovermode)
    _autosize_plotly(fig)
    # `_display_impl`, not v1's `display()`. That wrapper additionally snapshots
    # DataFrames to CSV through the analytics exporter, and captures PANDAS
    # OBJECTS ONLY -- so for a figure it has always been a pass-through to this
    # same shim. Calling the shim directly is exactly equivalent here and keeps
    # the figure layer free of the analytics exporter, which stays in v1.
    _safe_show(fig, label=label, _fallback=_display_impl)


def _safe_show(obj: object, *, label: Optional[str] = None,
               _fallback: Optional[object] = None) -> None:
    """Display any Plotly figure or ``arviz_plots`` PlotCollection, swallowing transport errors.

    IDE-managed display back-ends (e.g. PyCharm's DataLore/Kaleido helper) push the
    rendered image to a local HTTP server; when that socket is aborted the underlying
    ``show()`` raises ``ConnectionAbortedError`` (WinError 10053) mid-run. Rendering is a
    side-effect, never part of the model result, so a failure here must never abort the
    workflow. Handles both :class:`arviz_plots.PlotCollection` (``.show()``) and Plotly
    figures; ``pc.show()`` and raw ``fig.show()`` both route through this one guard.

    This is also the single auto-capture point for artifact export: every displayed
    figure is handed to :func:`_export_figure` *before* ``show()``, so a display
    transport failure still yields the exported file. The dark template is applied
    here too — before both display and export — so the two can never diverge
    (``arviz_plots`` PlotCollections reach this funnel still carrying the light
    default, which is why several exported panels used to render white).

    Parameters
    ----------
    obj
        The figure or PlotCollection to display. ``None`` is a no-op.
    label
        Optional filename slug for the exported artifact; auto-derived from the
        figure title (or just auto-numbered) when omitted.
    _fallback
        Optional callable tried when ``obj.show()`` raises — used by
        :func:`_render_plotly` to fall back to :func:`display` for a returned figure.
    """
    if obj is None:
        return
    _apply_dark_template(obj)  # displayed == exported theme
    _autosize_plotly(obj)  # width tracks the editor pane; heights stay explicit
    _export_figure(obj, label=label)

    # Matplotlib-backed panels must NOT take the ``obj.show()`` path: under an
    # interactive backend that blocks on a native window, and under the notebook
    # inline backend a PlotCollection's ``show()`` does not emit the figure as a
    # cell output. Hand it to IPython.display instead, then close it so the
    # figure is not re-emitted by the kernel's own end-of-cell flush (which would
    # duplicate every heavy raster in the notebook).
    mfig = _mpl_figure_of(obj)
    if mfig is not None:
        # Outside a kernel there is nothing to display INTO: the PNG is already on
        # disk from _export_figure above, and the alternatives both misbehave —
        # ``plt.show()`` blocks the headless export script on a Tk window, and the
        # non-IPython ``_display_impl`` fallback is ``print``, which would emit
        # "Figure(1400x780)" into the log.
        if _in_ipython_kernel():
            try:
                _display_impl(mfig)
            except Exception as exc:  # pragma: no cover - renderer-dependent
                logger.debug("Matplotlib display skipped: %r", exc)
        with contextlib.suppress(Exception):
            plt.close(mfig)
        return

    try:
        obj.show()
    except Exception as exc:  # pragma: no cover - display transport is environment-dependent
        logger.debug("Figure display skipped (renderer/transport failure): %r", exc)
        if _fallback is not None:
            try:
                _fallback
            except Exception:
                pass


# --- Hover / axis formatting conventions -------------------------------------
# Every hand-built trace gets a ``name=`` (or an explicit ``hoverinfo='skip'``
# for guides and band edges); every hovertemplate ends in ``<extra></extra>``;
# percent-scaled series format hover values as ``.1f`` + '%', decimal
# probabilities as ``.0%``; band+line pairs share a ``legendgroup`` with
# ``showlegend`` only on the line.
_LEGEND_FONT_SIZE = 9


def _hover_pct(label: str, *, axis: str = 'y', lead_text: bool = False,
               date_x: bool = False) -> str:
    """Hovertemplate for a percent-scaled series (values pre-scaled ×100)."""
    lead = '%{text}<br>' if lead_text else ('%{x|%Y-%m-%d}<br>' if date_x else '')
    return f'{lead}{label} = %{{{axis}:.1f}}%<extra></extra>'


def _hover_price(label: str, *, date_x: bool = True, nd: int = 2) -> str:
    """Hovertemplate for a price-space series along a date axis."""
    lead = '%{x|%Y-%m-%d}<br>' if date_x else ''
    return f'{lead}{label}: %{{y:.{nd}f}}<extra></extra>'


def _hover_prob(label: str, *, axis: str = 'y') -> str:
    """Hovertemplate for a decimal probability in [0, 1]."""
    return f'{label} = %{{{axis}:.0%}}<extra></extra>'


# --- Reference geometry (single source of truth for every guide line) ---------
# Zero lines, break-even markers, y=x guides, now-boundaries and horizon markers
# all key on a ROLE here rather than on per-call-site hexes and widths. Every
# guide draws with ``layer='below'`` so it sits behind the data it annotates.
#
# ``zero``     — the invariant of the panel: 0 return, break-even, y=x, p=0.5.
# ``anchor``   — a datum from the data: last price, now-boundary, horizon edge.
# ``emphasis`` — a highlighted cohort/universe statistic worth reading off.
_REF_LINE_KINDS: dict[str, dict[str, Any]] = {
    'zero': {'color': C_REF, 'dash': 'dash', 'width': 1.0},
    'anchor': {'color': C_REF, 'dash': 'dot', 'width': 1.2},
    'emphasis': {'color': C_HIGHLIGHT, 'dash': 'dash', 'width': 1.4},
}


_REF_LINE_OPACITY = 0.7


_REF_BAND_ALPHA = 0.08


def _add_ref_line(fig, *, x: Optional[float] = None, y: Optional[float] = None,
                  kind: str = 'zero', color: Optional[str] = None,
                  annotation_text: Optional[str] = None,
                  row: Optional[int] = None, col: Optional[int] = None,
                  **kwargs: Any) -> None:
    """Draw a horizontal / vertical reference line under the shared spec.

    Replaces the ad-hoc ``add_hline`` / ``add_vline`` calls that had grown seven
    different widths, three dash treatments and two Plotly APIs for what is
    semantically one piece of geometry.

    Parameters
    ----------
    x, y
        Vertical (``x``) or horizontal (``y``) position. Exactly one is given.
    kind
        Role key into :data:`_REF_LINE_KINDS` — ``'zero'``, ``'anchor'`` or
        ``'emphasis'``.
    color
        Overrides the role colour (e.g. a per-series median marker).
    annotation_text
        Optional label, rendered at :data:`_LEGEND_FONT_SIZE` in the line colour.
    row, col
        Subplot target; both omitted applies to the whole figure.
    **kwargs
        Forwarded to Plotly (``opacity``, ``annotation_position``, …).

    Raises
    ------
    ValueError
        If neither or both of ``x`` / ``y`` are supplied, or ``kind`` is unknown.
    """
    if (x is None) == (y is None):
        raise ValueError("Pass exactly one of x= / y= to _add_ref_line.")
    if kind not in _REF_LINE_KINDS:
        raise ValueError(f"Unknown reference-line kind {kind!r}. "
                         f"Valid: {sorted(_REF_LINE_KINDS)}")

    spec = dict(_REF_LINE_KINDS[kind])
    if color is not None:
        spec['color'] = color
    opts: dict[str, Any] = {
        'line': spec,
        'layer': 'below',
        'opacity': kwargs.pop('opacity', _REF_LINE_OPACITY),
    }
    if annotation_text is not None:
        opts['annotation_text'] = annotation_text
        opts['annotation_font'] = dict(size=_LEGEND_FONT_SIZE, color=spec['color'])
    if row is not None or col is not None:
        opts['row'], opts['col'] = row, col
    opts.update(kwargs)

    if y is not None:
        fig.add_hline(y=y, **opts)
    else:
        fig.add_vline(x=x, **opts)


def _add_ref_band(fig, *, x0: float, x1: float, color: str = C_REF,
                  alpha: float = _REF_BAND_ALPHA, **kwargs: Any) -> None:
    """Shade a vertical reference band (e.g. a tolerance corridor around zero)."""
    fig.add_vrect(x0=x0, x1=x1, fillcolor=_hex_to_rgba(color, alpha),
                  line_width=0, layer='below', **kwargs)


def _fmt_axis(fig, *, x: Optional[str] = None, y: Optional[str] = None,
              x_kind: str = 'linear', y_kind: str = 'linear',
              row: Optional[int] = None, col: Optional[int] = None) -> None:
    """Set axis titles + unit formatting in one call.

    ``*_kind``: ``'linear'`` (no formatting), ``'pct'`` (``ticksuffix='%'`` for
    percent-scaled values), ``'prob'`` (``tickformat='.0%'`` for decimal
    probabilities), ``'date'`` (``tickformat='%b %y'``).
    """
    def _apply(update, title, kind):
        kw: dict[str, Any] = {}
        if title is not None:
            kw['title_text'] = title
        if kind == 'pct':
            kw['ticksuffix'] = '%'
        elif kind == 'prob':
            kw['tickformat'] = '.0%'
        elif kind == 'date':
            kw['tickformat'] = '%b %y'
        if kw:
            if row is not None or col is not None:
                update(row=row, col=col, **kw)
            else:
                update(**kw)
    _apply(fig.update_xaxes, x, x_kind)
    _apply(fig.update_yaxes, y, y_kind)


def _show_fig(fig, *, label: Optional[str] = None) -> None:
    """Surface a Plotly :class:`~plotly.graph_objects.Figure` in any front-end.

    Thin wrapper over :func:`_render_plotly` (dark ``arviz-tumma`` template + a silent
    no-op when no renderer is available) for figures that are *returned* rather than
    self-shown -- e.g. the :func:`plot_kalman_forecast` structural forecasts. Mirrors
    the ``pc.show()`` convention used elsewhere in this module.

    Parameters
    ----------
    fig
        The Plotly figure to surface (typically the first element of a
        :func:`plot_kalman_forecast` return tuple).
    label
        Optional filename slug forwarded to the artifact exporter.
    """
    _render_plotly(fig, label=label)


def _present_vars(idata, candidates: Sequence[str]) -> list[str]:
    """Return the subset of ``candidates`` actually in ``idata.posterior``.

    Which posterior variables exist depends on how ``KalmanFilterPriceTarget.fit`` was
    parameterized (marginalized → ``log_state_init``; non-centred → ``z_innov``;
    stochastic-volatility → ``vol_step_size`` / ``nu_obs`` / ``vol_anchor_offset``), so
    summaries filter to what is present rather than hard-coding names.
    """
    post = getattr(idata, 'posterior', idata)
    have = set(post.data_vars)
    return [v for v in candidates if v in have]


@contextlib.contextmanager
def _quiet_degenerate_density():
    """Silence arviz-stats' single-value/non-finite KDE ``UserWarning`` while plotting.

    A stuck chain (poorly-mixed scale) or a structurally-constant slice (e.g. a
    one-level :class:`pm.ZeroSumNormal` effect, which is identically 0) makes
    arviz-stats fall back from a smooth KDE to a delta spike and emit
    ``arviz_stats/base/density.py:672`` *once per affected (chain, slice)* — the
    ``"Your data appears to have a single value or no finite values"`` warnings.
    The plot itself is still valid (the spike correctly depicts a pinned
    marginal), so we suppress only that one message rather than letting it flood
    the diagnostics log. Convergence problems are surfaced numerically by the
    R-hat / ESS report instead, not by raw KDE warnings.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='Your data appears to have a single value or no finite values',
            category=UserWarning,
        )
        yield


def _degenerate_posterior_vars(idata, var_names: Sequence[str],
                               *, atol: float = 1e-8) -> list[str]:
    """Return ``var_names`` whose pooled posterior draws are constant / non-finite.

    A variable is flagged when *every* non-sample slice is either all-non-finite
    or has a (chain+draw) spread ``<= atol``. Such variables produce only the
    delta-spike KDE that triggers ``density.py:672``; reporting them tells the
    reader *which* parameter collapsed (a real modelling signal) rather than
    leaving an anonymous warning in the log.

    Notes
    -----
    The all-finite fast path is not a micro-optimisation at this scale. The
    masked route allocates a full float64 copy per variable via ``np.where``, and
    this is called over the WHOLE posterior — which carries ``state_path``
    (``chains x draws x n_isin x T``, ~1.7 GB at the current budget) plus sixteen
    ``dims="isin"`` deterministics. Non-finite draws are the rare case, so
    probing for them first and reducing in place keeps the common path free of
    multi-GB temporaries. The masked branch is retained verbatim because ``inf``
    must be excluded from the spread and ``nanmax`` alone would not do it.
    """
    post = getattr(idata, 'posterior', idata)
    flagged: list[str] = []
    for v in var_names:
        if v not in post.data_vars:
            continue
        da = post[v]
        arr = np.asarray(da.values, dtype='float64')
        sample_axes = tuple(i for i, d in enumerate(da.dims) if d in ('chain', 'draw'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            if np.isfinite(arr).all():
                spread = (np.max(arr, axis=sample_axes)
                          - np.min(arr, axis=sample_axes))
            else:
                finite = np.isfinite(arr)
                if not finite.any():
                    flagged.append(v)
                    continue
                masked = np.where(finite, arr, np.nan)
                spread = (np.nanmax(masked, axis=sample_axes)
                          - np.nanmin(masked, axis=sample_axes))
        if not np.isfinite(spread).any() or float(np.nanmax(spread)) <= atol:
            flagged.append(v)
    return flagged


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` hex string to a Plotly ``rgba(r,g,b,a)`` string.

    Plotly fills/markers take alpha through the colour string (``fillcolor`` /
    ``rgba``) rather than a separate ``alpha`` argument, so the hand-built panels
    that shade credible bands express transparency this way.
    """
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


def _xval(v):
    """Coerce a scalar to a Plotly-friendly x value for shapes/annotations.

    Datetimes become a native :class:`datetime.datetime` (JSON-serialisable, so
    ``write_image`` / ``write_html`` exports succeed — a bare ``pandas.Timestamp`` is
    rejected by Plotly's encoder); everything else becomes a float.
    """
    if isinstance(v, np.datetime64) or np.issubdtype(np.asarray(v).dtype, np.datetime64):
        return pd.Timestamp(v).to_pydatetime()
    return float(v)


def _plotly_band(fig, x, lo, hi, *, color, alpha, name=None, row=None, col=None,
                 showlegend=False):
    """Add a shaded credible band (``fill='tonexty'`` between ``lo`` and ``hi``).

    The upper edge is drawn first as an invisible line, then the lower edge fills up
    to it — the Plotly idiom for the matplotlib ``fill_between`` the panels used.
    """
    fig.add_trace(go.Scatter(x=x, y=hi, mode='lines', line=dict(width=0),
                             hoverinfo='skip', showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=lo, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor=_hex_to_rgba(color, alpha),
                             name=name, hoverinfo='skip',
                             showlegend=bool(showlegend and name)), row=row, col=col)


def _kde_xy(values, *, clip_low: Optional[float] = None, n: int = 200,
            bw=None) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Gaussian-KDE (x, density) for the hand-built Plotly density panels.

    Plotly has no native KDE, so the panels that used ``seaborn.kdeplot`` evaluate a
    :class:`scipy.stats.gaussian_kde` on a robust 0.5–99.5 pct window. Returns
    ``(None, None)`` when the sample is too small or constant (a KDE would be a spike).
    """
    v = np.asarray(values, dtype='float64')
    v = v[np.isfinite(v)]
    if clip_low is not None:
        v = v[v >= clip_low]
    if v.size < 5 or np.allclose(v, v[0]):
        return None, None
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(v, bw_method=bw)
    lo, hi = np.nanpercentile(v, [0.5, 99.5])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    xs = np.linspace(lo, hi, n)
    return xs, kde(xs)


def _binned_density_trace(values, *, bins: int = 80, color: str,
                          name: Optional[str] = None,
                          hovertemplate: Optional[str] = None,
                          fill: bool = True, alpha: float = 0.6,
                          width: float = 1.5, clip: Optional[tuple] = None,
                          density: bool = True,
                          showlegend: bool = True, **scatter_kwargs):
    """Pre-binned density trace: the small-payload replacement for ``go.Histogram``.

    ``go.Histogram`` serialises **every raw value** into the figure and bins them in
    the browser. For posterior/prior arrays that is ruinous: the §6 prior-predictive
    panel shipped three traces of ``prior_draws x n_isin`` (~6.5 M float64 each) and
    weighed **207.7 MB** in the notebook, against a 122 KB PNG of the same figure.

    Binning here instead sends ``bins`` numbers rather than millions, at identical
    visual fidelity — a density plot never needed the raw sample. The stepped
    ``shape='hvh'`` line is the matplotlib ``histtype='step'`` analogue already used
    for the §6 empirical overlay; this helper is that pattern lifted to an SSOT.

    Parameters
    ----------
    values
        Raw sample; flattened, non-finite entries dropped.
    bins
        Histogram bin count (the payload size, near enough).
    color
        Hex series colour; the fill re-expresses it via :func:`_hex_to_rgba`.
    fill
        Fill to zero (histogram-like) rather than drawing an outline only.
    clip
        Optional ``(lo, hi)`` applied before binning, matching the callers that
        winsorise for readability.
    density
        Normalise to a probability density (the ``histnorm='probability
        density'`` equivalent). Pass ``False`` for raw counts so a panel whose
        y-axis reads "count" keeps saying something true.

    Returns
    -------
    plotly.graph_objects.Scatter or None
        ``None`` when fewer than two finite values survive — the caller skips the
        trace rather than emitting a degenerate spike.
    """
    v = np.asarray(values, dtype='float64').reshape(-1)
    v = v[np.isfinite(v)]
    if clip is not None:
        v = np.clip(v, clip[0], clip[1])
    if v.size < 2:
        return None
    dens, edges = np.histogram(v, bins=bins, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    trace_kwargs: dict[str, Any] = dict(
        x=centers, y=dens, mode='lines',
        line=dict(color=color, width=width, shape='hvh'),
        name=name, showlegend=bool(showlegend and name),
    )
    if fill:
        trace_kwargs['fill'] = 'tozeroy'
        trace_kwargs['fillcolor'] = _hex_to_rgba(color, alpha)
    if hovertemplate is not None:
        trace_kwargs['hovertemplate'] = hovertemplate
    trace_kwargs.update(scatter_kwargs)
    return go.Scatter(**trace_kwargs)


def _decimate_frame(df: pd.DataFrame, max_points: int, *,
                    by: Optional[str] = None,
                    seed: Optional[int] = None) -> tuple[pd.DataFrame, bool]:
    """Uniformly subsample ``df`` to at most ``max_points`` rows for a scatter.

    Plotly serialises every marker — coordinates, colour, size and each
    ``hover_data`` field — into the notebook, so a full-universe scatter costs
    megabytes per panel and a 17-facet grid costs tens of them. Beyond a few
    thousand markers the plot is an opaque cloud anyway, so the points past that
    buy overplotting rather than information.

    The sample is **uniform**, not a top-N truncation. A rank-based cut (the
    previous ``nlargest`` in :func:`plot_risk_return_scatter`) is fine as a no-op
    guard set above the universe size, but the moment it binds it deletes one
    tail of the distribution and the surviving cloud misrepresents the screen.
    ``by`` stratifies so small sectors keep representation.

    Parameters
    ----------
    df
        Frame to subsample. Returned unchanged when already small enough.
    max_points
        Row cap; ``<= 0`` disables decimation.
    by
        Optional column to stratify on (proportional allocation).
    seed
        RNG seed; defaults to the run config's ``random_seed`` so a panel is
        reproducible across runs.

    Returns
    -------
    tuple[pandas.DataFrame, bool]
        The (possibly subsampled) frame and whether decimation actually applied —
        callers annotate the figure when it did, so a thinned panel never reads
        as the full universe.
    """
    if max_points is None or max_points <= 0 or len(df) <= max_points:
        return df, False
    if seed is None:
        seed = get_viz_config().random_seed
    if by is not None and by in df.columns:
        # Select INDEX LABELS per group rather than ``groupby.apply``: pandas 3
        # excludes the grouping columns from the frame handed to ``apply``, so a
        # lambda returning ``g`` silently drops the stratifying column itself.
        frac = float(max_points) / float(len(df))
        keep: list = []
        for _, idx in df.groupby(by, observed=True).groups.items():
            take = max(1, int(round(len(idx) * frac)))
            if take >= len(idx):
                keep.extend(list(idx))
            else:
                keep.extend(list(pd.Index(idx).to_series()
                                  .sample(take, random_state=seed)))
        out = df.loc[keep]
        if len(out) > max_points:
            out = out.sample(max_points, random_state=seed)
        return out, True
    return df.sample(max_points, random_state=seed), True


def _add_binned_density(fig, values, *, row=None, col=None, **kwargs) -> None:
    """Add a :func:`_binned_density_trace` to ``fig``, skipping degenerate samples.

    Wraps the ``None`` return so callers never hand ``fig.add_trace(None)`` a
    degenerate series.
    """
    trace = _binned_density_trace(values, **kwargs)
    if trace is not None:
        fig.add_trace(trace, row=row, col=col)


def _ecdf_xy(values, *, n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """ECDF evaluated on a fixed ``n``-point probability grid.

    A literal ECDF plots one point per observation, so overlaying 60 predictive
    draws over a 25,948-cell response tensor emits ~1.6 M points into a single
    figure. Sampling the quantile function on a fixed grid is visually
    indistinguishable at any realistic figure width and bounds the payload at
    ``n`` points per curve regardless of sample size.
    """
    v = np.asarray(values, dtype='float64').reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.empty(0), np.empty(0)
    q = np.linspace(0.0, 1.0, int(n))
    return np.quantile(v, q), q


def _resolve_env_setting(key: str, env_file: str = 'environment_variables.txt',
                         default: Optional[str] = None) -> Optional[str]:
    """Return ``key`` from ``os.environ``, falling back to environment_variables.txt.

    The process may have been started without sourcing ``set_env.ps1``, in which case
    ``os.environ`` lacks ``key``; we then parse the ``KEY=VALUE`` lines of the project's
    ``environment_variables.txt`` as a fallback before returning ``default``.

    Parameters
    ----------
    key
        Environment variable name to resolve (e.g. ``'DB_URL'``).
    env_file
        Name of the dotenv-style file searched upward from the CWD.
    default
        Value returned when ``key`` is set neither in the environment nor the file.

    Returns
    -------
    Optional[str]
        The resolved value, or ``default`` when not found.
    """
    val = os.environ.get(key)
    if val:
        return val

    here = Path.cwd()
    for base in (here, *here.parents):
        candidate = base / env_file
        if candidate.is_file():
            for raw in candidate.read_text(encoding='utf-8').splitlines():
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, _, value = line.partition('=')
                if k.strip() == key:
                    return value.strip().strip('"').strip("'")
            break
    return default


# =============================================================================
# 1c. Artifact export — figures / tables / DataTrees -> KALMAN_PT_RESULTS_DIR
# =============================================================================
# Every figure shown through the :func:`_safe_show` funnel (which also serves
# :func:`_render_plotly` / :func:`_show_fig`), every DataFrame routed through
# :func:`display`, and the bulk data artifacts returned by :func:`main` are
# persisted under ``KALMAN_PT_RESULTS_DIR`` (env / environment_variables.txt,
# default ``pymc_kalman_filter_pt_results`` next to this script), in a
# **per-section subdirectory** derived from the artifact stem (see
# :data:`_EXPORT_SECTION_DIRS` / :func:`_export_dir_for`):
#
# * figures       -> PNG via plotly/kaleido or matplotlib; self-contained HTML
#                    fallback when kaleido / Chromium is unavailable
# * DataFrames    -> the curated bulk frames in :data:`_SQL_EXPORT_ARTIFACTS`
#                    become ``analytics.<stem>`` tables plus a generated
#                    ``{stem}.sql`` DDL file; every other frame (the
#                    auto-numbered ``display()`` snapshots) stays CSV
# * DataTrees     -> NetCDF (h5netcdf) + compact per-group JSON summary
#
# Exports are side effects in the same sense as rendering: every writer is
# guarded and logs a warning instead of raising, so a failed export can never
# abort the workflow (mirrors the ``_safe_show`` philosophy). The SQL sink
# additionally falls back to CSV when the database is unreachable, so an
# offline run never loses a frame.
_DEFAULT_RESULTS_DIRNAME = 'pymc_kalman_filter_pt_results'


_DEFAULT_EXPORT_PNG_WIDTH = 1400


_DEFAULT_EXPORT_PNG_HEIGHT = 780


_DEFAULT_EXPORT_PNG_SCALE = 2.0


_DEFAULT_EXPORT_DPI = 150


_EXPORT_SLUG_MAXLEN = 48


# Results-tree subdirectories, matched against an artifact stem by longest
# prefix. The ``export_section`` labels in :func:`main` and the hard-coded bulk
# stems in :func:`export_all_artifacts` both key off this one tuple. Two entries
# deliberately differ from their section label so the bulk stems land correctly:
# ``04_panel`` catches ``04_panel_frame`` (exported outside any section) and
# ``10b_risk`` catches ``10b_risk_analytics`` / ``_book`` / ``_summary``
# alongside the ``10b_risk_book_NN_*`` section stems.
_EXPORT_SECTION_DIRS: tuple[str, ...] = (
    # `04b_audit` is v2's panel-information audit (`run_panel_diagnostics`), the
    # stage v1 has no equivalent of. Safe beside `04_panel` because
    # `_export_dir_for` resolves ties with `max(matches, key=len)`, so the longer
    # prefix wins; without the entry every §4b artifact silently lands in
    # `00_misc`, which is how the decay ladder would have gone missing.
    '01_data', '02_eda', '03_features', '04_panel', '04b_audit',
    '06_prior', '07_posterior',
    '08_ppc', '09_diagnostics', '09b_comparison', '10_screen', '10b_risk',
    '10c_analytics',
    '10k_universe', '11_single_isin', '11b_single_sv', '12_mingled',
    '12b_mingled_sv', '13_forest', '13b_further_views', '14_summary',
    '14b_recommendations', '00_misc',
)


_EXPORT_MISC_DIR = '00_misc'


# Bulk stems whose prefix does not match their section directory. ``10c`` is the
# only genuine mismatch: the section is ``10c_analytics`` but its bulk artifact
# is ``10c_kalman_results``, so a plain prefix scan would file it under 00_misc.
_EXPORT_DIR_ALIASES: dict[str, str] = {'10c_kalman': '10c_analytics'}


@dataclass
class _ExportState:
    """Mutable module-level artifact-export state (lazy singleton).

    Attributes
    ----------
    root
        Resolved output directory (``KALMAN_PT_RESULTS_DIR``).
    enabled
        Master switch; all export helpers no-op while ``False``.
    section
        Active filename prefix, managed by :func:`export_section`.
    counters
        Per-section running counter for auto-numbered artifact stems.
    png_ok
        Memo flag — flips ``False`` after the first kaleido/Chromium failure so
        every subsequent Plotly figure goes straight to the HTML fallback.
    sql_ok
        Memo flag — flips ``False`` after the first analytics-schema write
        failure so every subsequent curated frame goes straight to the CSV
        fallback instead of re-attempting a doomed connection.
    analytics_written
        Set by :func:`note_analytics_written` once
        ``analytics.kalman_filtered_price_targets`` has been written this run;
        suppresses the redundant ``10c_kalman_results`` table.
    cleaned
        Section directories already purged this run (``KALMAN_PT_CLEAN_RESULTS``).
    run_id
        Stable identifier for this workflow run, stamped onto every frame the
        SQL sink writes (see :func:`stamp_export_provenance`). Assigned once, at
        state construction, so every section of one run shares it — that is the
        whole point. Before 2026-08-16 nothing recorded which run a row came
        from, and the analytics schema silently served two vintages at once:
        ``09_diagnostics_01_table`` / ``kalman_filtered_price_targets`` from the
        2026-08-15 fit alongside five bulk frames from 2026-08-14, with 6 425 of
        6 427 joinable ISINs carrying a different ``er_mean``. Nothing in the
        schema made that visible; it took a value-level diff to find.
    started_at
        UTC timestamp of state construction, exported as ``exported_at``.
    """

    root: Path
    enabled: bool = False
    section: str = _EXPORT_MISC_DIR
    counters: dict[str, int] = field(default_factory=dict)
    png_ok: bool = True
    sql_ok: bool = True
    analytics_written: bool = False
    cleaned: set[str] = field(default_factory=set)
    started_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


_export_state_instance: Optional[_ExportState] = None


def get_export_state() -> _ExportState:
    """Return the lazy export-state singleton, resolving the output directory once.

    Resolution order: :attr:`KalmanRunConfig.results_dir` (so ``main(config=…)``
    actually redirects artifacts) → ``KALMAN_PT_RESULTS_DIR`` via
    :func:`_resolve_env_setting` (env → ``environment_variables.txt``) → the
    :data:`_DEFAULT_RESULTS_DIRNAME` default. A relative value is anchored at this
    script's directory, so the default lands at
    ``<project root>/pymc_kalman_filter_pt_results`` regardless of the CWD.
    """
    global _export_state_instance
    if _export_state_instance is None:
        raw: Optional[str] = None
        with contextlib.suppress(Exception):
            raw = get_viz_config().results_dir
        if not raw:
            raw = _resolve_env_setting('KALMAN_PT_RESULTS_DIR',
                                       default=_DEFAULT_RESULTS_DIRNAME)
        root = Path(raw or _DEFAULT_RESULTS_DIRNAME)
        if not root.is_absolute():
            # PROJECT root, not this module's directory. In v1 these helpers
            # lived in a root-level script, so `Path(__file__).parent` WAS the
            # project root; here it is `probabilistic_ml_model/visualizations/`
            # and the same expression would bury every artifact three levels
            # deep inside the package. `_project_root()` reproduces the old
            # anchor exactly.
            root = _project_root() / root
        _export_state_instance = _ExportState(root=root)
    return _export_state_instance


def reset_export_state() -> None:
    """Reset the export-state singleton (re-reads ``KALMAN_PT_RESULTS_DIR``)."""
    global _export_state_instance
    _export_state_instance = None


def enable_artifact_export(enabled: bool = True) -> None:
    """Turn artifact export on (or off) for the current process."""
    get_export_state().enabled = enabled


def set_export_section(label: str) -> str:
    """Set the artifact-export section without a ``with`` block.

    :func:`export_section` is the right tool inside :func:`main`, where sections
    nest and unwind. A notebook has no enclosing block per cell, so this setter
    lets each cell declare which section its figures and tables belong to —
    otherwise every notebook artifact would land in ``00_misc``.

    Parameters
    ----------
    label
        Section prefix, e.g. ``'09_diagnostics'`` (see
        :data:`_EXPORT_SECTION_DIRS`).

    Returns
    -------
    str
        The previous section, so a caller can restore it.
    """
    state = get_export_state()
    prev = state.section
    state.section = label
    _clean_section_dir(label)
    return prev


@contextlib.contextmanager
def export_section(label: str):
    """Scope auto-generated artifact filenames to a workflow section.

    Sets the ``{section}`` prefix used by the auto-capture hooks in
    :func:`_safe_show` and :func:`display` while the block is active; restores
    the previous section on exit (nestable). Entering a section never touches
    the filesystem.

    Parameters
    ----------
    label
        Section prefix, e.g. ``'09_diagnostics'``.
    """
    state = get_export_state()
    prev = state.section
    state.section = label
    _clean_section_dir(label)
    try:
        yield
    finally:
        state.section = prev


def _slugify(raw: str) -> str:
    """Compress ``raw`` to a lowercase ``[a-z0-9_]`` filename slug."""
    parts = re.findall(r'[a-z0-9]+', str(raw).lower())
    return '_'.join(parts)[:_EXPORT_SLUG_MAXLEN].strip('_')


def _export_dir_for(stem: str) -> str:
    """Return the results subdirectory owning ``stem`` (longest-prefix match).

    Matching is done on the **stem** rather than the active
    :func:`export_section`, because :func:`export_all_artifacts` writes its bulk
    stems after every section context has exited.

    Parameters
    ----------
    stem
        Artifact filename stem, e.g. ``'10b_risk_book'`` or ``'02_eda_07'``.

    Returns
    -------
    str
        A member of :data:`_EXPORT_SECTION_DIRS`; :data:`_EXPORT_MISC_DIR` when
        no prefix matches.
    """
    for prefix, directory in _EXPORT_DIR_ALIASES.items():
        if stem == prefix or stem.startswith(f'{prefix}_'):
            return directory
    matches = [d for d in _EXPORT_SECTION_DIRS if stem == d or stem.startswith(f'{d}_')]
    if not matches:
        return _EXPORT_MISC_DIR
    return max(matches, key=len)


def _clean_section_dir(label: str) -> None:
    """Purge a section's subdirectory once per run when ``KALMAN_PT_CLEAN_RESULTS=1``.

    Per-section counters restart on every run while title slugs drift, so stale
    artifacts otherwise interleave with current ones under shifted indices. Now
    that sections are partitioned into their own directories this is a safe,
    surgical reset. Off by default so an interrupted run never destroys the
    previous run's output.
    """
    state = get_export_state()
    if not state.enabled:
        return
    if _resolve_env_setting('KALMAN_PT_CLEAN_RESULTS', default='0') != '1':
        return
    directory = _export_dir_for(label)
    if directory in state.cleaned:
        return
    state.cleaned.add(directory)
    target = state.root / directory
    if not target.is_dir():
        return
    try:
        shutil.rmtree(target)
        logger.info("Cleared stale artifacts in %s", target)
    except Exception as exc:  # pragma: no cover - best-effort hygiene
        logger.warning("Could not clear %s: %r", target, exc)


def _export_path(stem: str, ext: str) -> Path:
    """Return ``root/<section-dir>/stem.ext``, creating the directory on first use.

    The single filesystem touch point for every artifact writer, so the
    per-section tree is applied uniformly to PNG / HTML / CSV / SQL / JSON /
    NetCDF output without touching individual call sites.
    """
    state = get_export_state()
    directory = state.root / _export_dir_for(stem)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}.{ext}"


def _next_stem(label: Optional[str] = None, obj: object = None) -> str:
    """Return the next auto-numbered ``{section}_{NN}[_{slug}]`` filename stem.

    The slug comes from ``label`` when given, else (best-effort) from the figure
    title of ``obj`` — meaningful names for free on most panels without touching
    call sites. Both backends are probed: Plotly stores the title on
    ``layout.title.text``, Matplotlib on the figure ``suptitle`` (which is what
    ``PlotCollection.add_title`` sets there), so routing a panel to the matplotlib
    backend does not silently degrade its filename to a bare counter.
    """
    state = get_export_state()
    n = state.counters.get(state.section, 0) + 1
    state.counters[state.section] = n
    slug = label
    if slug is None and obj is not None:
        try:
            slug = obj.layout.title.text  # type: ignore[attr-defined]
        except Exception:
            slug = None
    if slug is None and obj is not None:
        mfig = _mpl_figure_of(obj)
        if mfig is not None:
            with contextlib.suppress(Exception):
                sup = mfig._suptitle  # type: ignore[attr-defined]
                slug = sup.get_text() if sup is not None else None
            if not slug:
                with contextlib.suppress(Exception):
                    axes = mfig.get_axes()
                    slug = axes[0].get_title() if axes else None
    stem = f"{state.section}_{n:02d}"
    if slug:
        slug = _slugify(slug)
        if slug:
            stem = f"{stem}_{slug}"
    return stem


def _write_plotly_figure(fig: object, stem: str) -> None:
    """Write a Plotly figure as PNG, falling back to self-contained HTML.

    PNG needs ``kaleido`` + a Chromium binary (``plotly_get_chrome -y``). The
    first failure flips :attr:`_ExportState.png_ok` so later figures skip the
    doomed ``write_image`` attempt. ``_autosize_plotly`` clears ``layout.width``
    before display, so an explicit pixel width is always passed.
    """
    state = get_export_state()
    if state.png_ok:
        try:
            _apply_dark_template(fig)  # idempotent; guards the direct-call path
            fig.write_image(
                str(_export_path(stem, 'png')),
                width=int(fig.layout.width or _DEFAULT_EXPORT_PNG_WIDTH),
                height=int(fig.layout.height or _DEFAULT_EXPORT_PNG_HEIGHT),
                scale=_DEFAULT_EXPORT_PNG_SCALE,
            )
            return
        except Exception as exc:
            state.png_ok = False
            logger.warning(
                "Plotly PNG export unavailable (install kaleido + run "
                "`plotly_get_chrome -y`): %r -- falling back to HTML", exc)
    fig.write_html(str(_export_path(stem, 'html')),
                   include_plotlyjs=True, full_html=True)


def _export_figure(obj: object, *, label: Optional[str] = None) -> None:
    """Persist any displayed figure object under the active export section.

    Handles a raw Plotly figure or an ``arviz_plots`` PlotCollection (via
    :func:`_plotly_figure_of`) and matplotlib figures, raw or wrapped (via
    :func:`_mpl_figure_of`). The matplotlib branch pins the figure's own facecolor
    explicitly rather than relying on the seaborn ``savefig.facecolor`` rc
    surviving — ``bbox_inches='tight'`` otherwise lets a light background leak into
    the exported PNG.

    Resolution order matters: a matplotlib-backed PlotCollection satisfies both
    ``hasattr(obj, 'viz')`` and ``hasattr(obj, 'savefig')``, so it used to fall
    into the untyped collection branch and export at the library's default dpi
    with no facecolor. Resolving the real Figure first keeps every matplotlib
    panel on one export path. No-op while export is disabled; never raises.
    """
    state = get_export_state()
    if not state.enabled or obj is None:
        return
    try:
        fig = _plotly_figure_of(obj)
        mfig = None if fig is not None else _mpl_figure_of(obj)
        stem = _next_stem(label, fig if fig is not None else obj)
        if fig is not None:
            _write_plotly_figure(fig, stem)
        elif mfig is not None:  # matplotlib Figure, raw or PlotCollection-wrapped
            mfig.savefig(_export_path(stem, 'png'),
                         dpi=_DEFAULT_EXPORT_DPI, bbox_inches='tight',
                         facecolor=mfig.get_facecolor(), edgecolor='none')
        elif hasattr(obj, 'viz') and hasattr(obj, 'savefig'):
            # PlotCollection whose figure node could not be resolved either way.
            ext = 'png' if state.png_ok else 'html'
            obj.savefig(str(_export_path(stem, ext)))
        else:
            logger.debug("Figure export skipped (unrecognised type %s)", type(obj))
    except Exception as exc:
        logger.warning("Figure export skipped (%s): %r", label or '?', exc)
