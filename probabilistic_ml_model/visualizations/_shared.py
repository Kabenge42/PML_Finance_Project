"""
Shared utilities for visualization modules.

Centralizes common constants, helper functions, and column resolution logic
to avoid duplication across visualization modules.

Column alias resolution is driven by the ``FeatureViewCatalog`` in
``feature_catalog.py``.  The legacy ``MV_COLUMN_ALIASES`` dict is now
built from the catalog's ``VIZ_REQUIREMENTS`` at import time, ensuring
a single source of truth.
"""

from __future__ import annotations

import logging

import plotly.graph_objects as go

from probabilistic_ml_model.data_utils.feature_catalog import (
    get_column_aliases,
    resolve_column_from_catalog,
    columns_for_viz,
    VIZ_REQUIREMENTS,
)

# Dark theme for Plotly (single source of truth)
PLOTLY_TEMPLATE = "plotly_dark"

# Dark theme for ArviZ / Matplotlib (single source of truth)
# ArviZ 1.0 style system: "arviz-variat" (updated from "arviz-tumma")
ARVIZ_TEMPLATE = "arviz-variat"

logger = logging.getLogger(__name__)


def apply_arviz_theme() -> None:
    """Apply the global ArviZ 1.0 theme using arviz_plots.

    Tries ``arviz_plots`` first, then legacy ``arviz``, falling back to
    ``matplotlib.pyplot.style.use("dark_background")`` otherwise.
    Call this once at import time in modules that produce ArviZ figures.
    """
    try:
        import arviz_plots as azp

        azp.style.use(ARVIZ_TEMPLATE)
    except (ImportError, ValueError, TypeError):
        try:
            import arviz as az

            az.style.use(ARVIZ_TEMPLATE)
        except (ImportError, ValueError, TypeError):
            try:
                import matplotlib.pyplot as plt

                plt.style.use("dark_background")
            except ImportError:
                pass


# ---------------------------------------------------------------------------
# ArviZ 1.0 compatibility helpers
# ---------------------------------------------------------------------------


def _make_datatree(**groups):
    """Build an ``xr.DataTree`` from keyword groups (ArviZ 1.0 compat).

    Replaces the old ``az.InferenceData(posterior=ds, ...)`` constructor.
    Each keyword is a group name and its value an ``xr.Dataset``.
    An empty call returns an empty DataTree.
    """
    import xarray as xr

    if not groups:
        return xr.DataTree()
    return xr.DataTree.from_dict(
        {name: xr.DataTree(ds) for name, ds in groups.items() if ds is not None}
    )


def _fig_from_pc(pc):
    """Extract a matplotlib ``Figure`` from an ArviZ 1.0 ``PlotCollection``.

    ArviZ 1.0 plot functions return ``PlotCollection`` instead of axes.
    The figure is stored in ``pc.viz.ds['figure']``.
    Handles both PlotCollection and raw Figure objects gracefully.
    Returns ``None`` for ``None`` input.
    """
    if pc is None:
        return None
    # PlotCollection path
    try:
        return pc.viz.ds["figure"].item()
    except (AttributeError, KeyError, TypeError):
        pass
    # Direct figure passthrough
    try:
        import matplotlib.pyplot as plt

        if isinstance(pc, plt.Figure):
            return pc
    except ImportError:
        pass
    return pc


def _pc_add_title(pc, title: str):
    """Safely add title to a PlotCollection (ArviZ 1.0)."""
    try:
        pc.add_title(title)
    except (AttributeError, TypeError):
        fig = _fig_from_pc(pc)
        if fig and hasattr(fig, "suptitle"):
            fig.suptitle(title)
    return pc


# Shared color scheme
COLORS = [
    "#0A7EA4",
    "#00A878",
    "#6C63FF",
    "#FF6B6B",
    "#4ECDC4",
    "#FFD93D",
    "#95E1D3",
    "#F38181",
]

# Column alias map — now derived from FeatureViewCatalog VIZ_REQUIREMENTS.
# The catalog merges all VisualizationFeatureRequirement.column_aliases into
# a single dict at module load time.  Additional aliases that are not part of
# any visualization requirement are appended here to maintain backward compat.
MV_COLUMN_ALIASES: dict[str, list[str]] = get_column_aliases()

# Append legacy aliases that aren't covered by VIZ_REQUIREMENTS yet
_LEGACY_EXTRAS: dict[str, list[str]] = {
    "inventory_turnover": [
        "inventory_turnover_itf",
        "inventory_turnover_mv",
        "inventory_turnover",
    ],
    "beneish_m_score": ["accounting_quality_score", "accruals_quality"],
    "eps_qoq_growth": ["eps_qoq_growth"],
    "quality_composite_score": ["quality_momentum_score", "quality_composite_score"],
    "rnd_intensity": ["rnd_intensity", "rnd_intensity_ltm"],
    "asset_turnover": ["asset_turnover", "total_asset_turnover"],
}
for _key, _aliases in _LEGACY_EXTRAS.items():
    if _key not in MV_COLUMN_ALIASES:
        MV_COLUMN_ALIASES[_key] = _aliases
    else:
        for _a in _aliases:
            if _a not in MV_COLUMN_ALIASES[_key]:
                MV_COLUMN_ALIASES[_key].append(_a)

# ---------------------------------------------------------------------------
# Ensemble risk-adjusted return columns (from build_quad_model_alignment)
# ---------------------------------------------------------------------------
ENSEMBLE_RETURN_COLS: list[str] = [
    "ensemble_return",
    "ensemble_return_shrunk",
    "risk_adj_return",
    "mcmc_shrinkage",
]

RISK_DISCOUNT_MAP: dict[int, float] = {0: 0.70, 1: 0.85, 2: 0.95, 3: 1.00}

RISK_QUALITY_LABELS: dict[int, str] = {
    0: "High Risk",
    1: "Elevated Risk",
    2: "Moderate Risk",
    3: "Low Risk",
}


def resolve_column(df, logical_name: str) -> str | None:
    """Resolve a logical column name to an actual DataFrame column.

    Delegates to the catalog-driven ``resolve_column_from_catalog`` and
    falls back to ``MV_COLUMN_ALIASES`` for legacy aliases not yet
    registered in ``VIZ_REQUIREMENTS``.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check columns against.
    logical_name : str
        The logical / expected column name.

    Returns
    -------
    str | None
        The resolved column name present in *df*, or ``None``.
    """
    # Try catalog-driven resolution first
    result = resolve_column_from_catalog(df, logical_name)
    if result is not None:
        return result
    # Fall back to full MV_COLUMN_ALIASES (includes legacy extras)
    for alias in MV_COLUMN_ALIASES.get(logical_name, []):
        if alias in df.columns:
            return alias
    return None


def create_no_data_figure(title: str) -> go.Figure:
    """Create a placeholder figure when no data is available."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(title=title, template=PLOTLY_TEMPLATE)
    return fig
