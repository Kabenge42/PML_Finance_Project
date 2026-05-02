"""Multi-level hierarchical shrinkage infrastructure for PyMC models.

This module is the *single source of truth* for the canonical category
hierarchy used by every PyMC model in :mod:`probabilistic_ml_model.pymc_models`.
It mirrors the semantics of
:func:`probabilistic_ml_model.statistical_functions.statistical_models.hierarchical_mcmc_multi_level`
(see ``_HIERARCHICAL_CATEGORY_COLS`` and the inline ``_PARENT_MAP`` in that
function) and exposes three reusable building blocks:

* :data:`HIERARCHICAL_CATEGORY_COLS` / :data:`PARENT_MAP` — canonical names
  and parent-of-child relationships.
* :func:`build_hierarchy_indices` — pure-NumPy helper that returns
  per-level metadata (unique labels, isin → level idx, level → parent idx).
* :func:`build_nested_logit_normal_rates` — PyMC helper that wires nested
  non-centred logit-Normal rates following the parent-of relations from
  :data:`PARENT_MAP`.

Plus :func:`_resolve_prior_sigma` — a small helper implementing
recommendation §12.3 #2 (calculation-type-driven prior sigmas).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------
HIERARCHICAL_CATEGORY_COLS: tuple[str, ...] = (
    "region",
    "country",
    "trading_country",
    "exchange",
    "unit",
    "sector",
    "industry",
    "style_class",
    "size_class",
)

PARENT_MAP: dict[str, Optional[str]] = {
    "region": None,
    "country": "region",
    "unit": None,
    "trading_country": None,
    "exchange": "country",
    "sector": "exchange",
    "industry": "sector",
    "style_class": None,
    "size_class": "style_class",
}


# ---------------------------------------------------------------------------
# Calculation-type-driven prior sigma (recommendation §12.3 #2)
# ---------------------------------------------------------------------------
_DATA_TYPE_BASE_SIGMA: dict[str, float] = {
    "pct": 0.05,
    "ratio": 0.10,
    "level": 0.20,
    "score": 0.50,
    "flag": 0.25,
}

_CALC_TYPE_MULTIPLIER: dict[str, float] = {
    "growth": 0.5,  # tighter for growth aliases
    "ttm": 0.8,
    "log": 0.6,
    "delta": 0.7,
}


def _resolve_prior_sigma(
    data_type: Optional[str] = None,
    calculation_type: Optional[str] = None,
    *,
    default: float = 0.10,
) -> float:
    """Resolve a calculation-type-aware prior sigma.

    Parameters
    ----------
    data_type
        Catalogue ``feature_catalogue.data_type`` (``pct``, ``ratio``,
        ``level``, ``score``, ``flag``...). Drives the base sigma.
    calculation_type
        Catalogue ``feature_catalogue.calculation_type`` (``growth``,
        ``ttm``, ``log``, ``delta``...). Multiplies the base sigma.
    default
        Fallback sigma when ``data_type`` is unknown.

    Returns
    -------
    float
        Strictly-positive sigma.
    """
    base = _DATA_TYPE_BASE_SIGMA.get((data_type or "").lower(), default)
    mult = _CALC_TYPE_MULTIPLIER.get((calculation_type or "").lower(), 1.0)
    return float(max(base * mult, 1e-6))


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------
def build_hierarchy_indices(
    df: pd.DataFrame,
    isins: np.ndarray,
    levels: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Return per-level metadata for nested hierarchical shrinkage.

    Output schema per level::

        {
          "labels":       np.ndarray[str]            # coord values
          "idx":          np.ndarray[int32, n_isin]  # isin -> level index
          "parent_label": str | None                 # parent column (or None)
          "parent_of":    np.ndarray[int32, n_level] # level idx -> parent idx
        }

    Parameters
    ----------
    df
        Categorical frame indexed by ISIN with one column per level. Missing
        rows / columns are filled with ``"_unknown"``.
    isins
        ISIN array used to align ``df`` rows; the resulting ``idx`` arrays
        will have shape ``(len(isins),)`` in the same order.
    levels
        Subset of :data:`HIERARCHICAL_CATEGORY_COLS` to materialise. Defaults
        to every column from :data:`HIERARCHICAL_CATEGORY_COLS` that is
        present in ``df``.
    """
    isins = np.asarray(isins)
    if levels is None:
        levels = [c for c in HIERARCHICAL_CATEGORY_COLS if c in df.columns]
    else:
        unknown = [lv for lv in levels if lv not in HIERARCHICAL_CATEGORY_COLS]
        if unknown:
            raise ValueError(f"hierarchy_levels {unknown!r} not in HIERARCHICAL_CATEGORY_COLS")

    # Reindex to the supplied isin order; fill missing with sentinel
    aligned = df.reindex(index=isins).copy()
    for lv in levels:
        if lv not in aligned.columns:
            aligned[lv] = "_unknown"
        aligned[lv] = aligned[lv].astype(object).fillna("_unknown").astype(str)

    out: dict[str, dict[str, Any]] = {}
    for lv in levels:
        labels, idx = np.unique(aligned[lv].to_numpy(), return_inverse=True)
        labels = labels.astype(str)
        idx = idx.astype(np.int32)

        parent = PARENT_MAP.get(lv)
        parent_of: Optional[np.ndarray] = None
        parent_label: Optional[str] = None
        if parent is not None and parent in out:
            parent_label = parent
            parent_labels = out[parent]["labels"]
            parent_of = np.empty(labels.shape, dtype=np.int32)
            for i, lbl in enumerate(labels):
                # Mode parent label across all rows assigned to this child label.
                mask = aligned[lv].to_numpy() == lbl
                if not mask.any():
                    parent_of[i] = 0
                    continue
                parent_vals = aligned.loc[mask, parent].to_numpy()
                # mode (first most-frequent)
                vals, counts = np.unique(parent_vals, return_counts=True)
                mode_val = vals[np.argmax(counts)]
                where = np.where(parent_labels == mode_val)[0]
                parent_of[i] = int(where[0]) if where.size else 0

        out[lv] = {
            "labels": labels,
            "idx": idx,
            "parent_label": parent_label,
            "parent_of": parent_of,
        }
    return out


# ---------------------------------------------------------------------------
# Backward-compat shim used by every model's fit(...)
# ---------------------------------------------------------------------------
def coerce_categories(
    isins: np.ndarray,
    *,
    sectors: Optional[np.ndarray] = None,
    categories_df: Optional[pd.DataFrame] = None,
    hierarchy_levels: Optional[list[str]] = None,
) -> tuple[Optional[pd.DataFrame], Optional[list[str]]]:
    """Resolve the ``(categories_df, hierarchy_levels)`` pair from legacy /
    new fit-signature inputs.

    * If ``categories_df`` is supplied, it is used as-is (re-indexed by
      ``isins``); ``hierarchy_levels`` defaults to its columns.
    * Else if ``sectors`` is supplied (legacy), it is wrapped as a single-
      column ``categories_df`` with ``hierarchy_levels=["sector"]``.
    * Else returns ``(None, None)``.
    """
    if categories_df is not None:
        df = categories_df.copy()
        if df.index.name != "isin" and "isin" in df.columns:
            df = df.drop_duplicates("isin").set_index("isin")
        df = df.reindex(index=np.asarray(isins))
        levels = (
            list(hierarchy_levels)
            if hierarchy_levels
            else [c for c in HIERARCHICAL_CATEGORY_COLS if c in df.columns]
        )
        return df, levels
    if sectors is not None:
        df = pd.DataFrame({"sector": np.asarray(sectors)}, index=np.asarray(isins))
        return df, ["sector"]
    return None, None


# ---------------------------------------------------------------------------
# Out-of-sample (pm.set_data) helpers
# ---------------------------------------------------------------------------
def find_leaf_idx_var(idata) -> Optional[str]:
    """Return the ``{leaf_level}_idx`` variable name registered on
    ``idata.constant_data`` by :func:`build_nested_logit_normal_rates`,
    or ``None`` if no such container exists.

    The helper looks for any data_var whose name ends with ``"_idx"``;
    by construction only the leaf level is registered as ``pm.Data``, so
    a single match is expected.
    """
    if not hasattr(idata, "constant_data"):
        return None
    return next(
        (str(v) for v in idata.constant_data.data_vars if str(v).endswith("_idx")),
        None,
    )


def resolve_leaf_train_labels(idata, leaf_level: str) -> Optional[np.ndarray]:
    """Return the trained labels for ``leaf_level`` from a fitted ``idata``.

    The hierarchy-level coords (e.g. ``industry``, ``sector``, ``exchange``)
    live on ``idata.posterior.coords`` — only ``isin`` and the per-model
    ``<model>_feature`` coord land in ``idata.constant_data``. To make the
    OOS swap robust, this helper inspects both groups (``constant_data``
    first, then ``posterior``) and returns the first match as a 1-D
    ``np.ndarray`` of strings, or ``None`` if the level coord is not found.
    """
    for group_name in ("constant_data", "posterior"):
        group = getattr(idata, group_name, None)
        if group is None:
            continue
        coords = getattr(group, "coords", {})
        if leaf_level in coords:
            return np.asarray(coords[leaf_level].values).astype(str)
    return None


def build_oos_setdata(
    idata,
    categories_df: Optional[pd.DataFrame],
    new_isins: np.ndarray,
    *,
    extra_data: Optional[dict] = None,
) -> dict:
    """Build a ``pm.set_data`` payload for an out-of-sample cohort.

    Centralises the foot-gun-prone leaf-index lookup that every model's
    OOS cell needed to repeat. Returns a dict suitable for direct use as
    ``pm.set_data(<dict>, coords={"isin": new_isins})`` inside the fitted
    ``pm.Model`` context.

    Behaviour
    ---------
    * Looks up ``{leaf_level}_idx`` via :func:`find_leaf_idx_var`. If no
      hierarchy was registered on ``idata`` the returned dict only carries
      ``extra_data``.
    * Resolves the trained leaf-level labels via
      :func:`resolve_leaf_train_labels` (checks both ``constant_data`` and
      ``posterior`` coords, mirroring where the hierarchy helper actually
      stores them).
    * Maps each ``new_isins`` row to its leaf-level idx via
      ``categories_df[leaf_level]``; unknown labels are mapped to ``0`` so
      the swap never raises a ``KeyError``.

    Parameters
    ----------
    idata
        ArviZ ``InferenceData`` returned by a fitted PyMC model.
    categories_df
        DataFrame indexed by ISIN with at least the leaf-level column
        (e.g. ``industry``). Must already be sliced/aligned to ``new_isins``
        — the helper only takes the last ``len(new_isins)`` rows.
    new_isins
        1-D array of holdout ISINs that will become the new ``isin`` coord.
    extra_data
        Optional mapping merged into the returned dict. Use this to pass
        the model-specific data containers (``n_total``, ``payout_data``,
        ``z_score_data`` etc.) alongside the leaf-idx swap.

    Returns
    -------
    dict
        Payload for ``pm.set_data(...)``. Always includes ``extra_data``
        entries; additionally includes ``{leaf_level}_idx`` whenever the
        hierarchy level is resolvable.
    """
    payload: dict = dict(extra_data or {})

    leaf_idx_name = find_leaf_idx_var(idata)
    if leaf_idx_name is None or categories_df is None:
        return payload

    leaf_level = leaf_idx_name[: -len("_idx")]
    if leaf_level not in categories_df.columns:
        return payload

    train_labels = resolve_leaf_train_labels(idata, leaf_level)
    if train_labels is None:
        return payload

    label_to_idx = {lbl: i for i, lbl in enumerate(train_labels)}
    holdout_n = len(new_isins)
    holdout_labels = categories_df[leaf_level].astype(str).to_numpy()[-holdout_n:]
    payload[leaf_idx_name] = np.array(
        [label_to_idx.get(lbl, 0) for lbl in holdout_labels],
        dtype="int32",
    )
    return payload


# ---------------------------------------------------------------------------
# Nested non-centred logit-Normal builder (PyMC)
# ---------------------------------------------------------------------------
def build_nested_logit_normal_rates(
    hierarchy: dict[str, dict[str, Any]],
    *,
    leaf_dim: str = "isin",
    name: str = "rate",
    global_mu_sigma: float = 1.5,
    level_sigma_scale: float = 1.0,
):
    """Build nested non-centred logit-Normal rates per level inside a
    ``pm.Model`` context.

    For each level ``L`` with parent ``P``::

        mu_L[g]   = mu_P[parent_of(g)] + sigma_L * z_L[g]
        rate_L[g] = sigmoid(mu_L[g])

    Top-level (parent is ``None``) anchors at a global
    ``mu_logit ~ Normal(0, global_mu_sigma)``.

    Returns a dict with::

        {
          "level_logits":  {level: tt}    # logit per level (dims=level)
          "level_rates":   {level: tt}    # sigmoid(logit) per level (dims=level)
          "leaf_rate":     tt              # leaf rate broadcast to leaf_dim
          "leaf_level":    str             # name of the leaf level
        }

    Requires PyMC; raises ``ImportError`` otherwise.
    """
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - exercised when pymc missing
        raise ImportError("pymc required for build_nested_logit_normal_rates") from exc

    if not hierarchy:
        raise ValueError("hierarchy must contain at least one level")

    levels = list(hierarchy.keys())
    leaf_level = levels[-1]

    level_logits: dict[str, Any] = {}
    level_rates: dict[str, Any] = {}

    # Global anchor (used by every top-level branch with parent=None)
    global_mu = pm.Normal(f"{name}_mu_logit", 0.0, global_mu_sigma)

    for lv in levels:
        parent = hierarchy[lv].get("parent_label")
        sigma = pm.HalfNormal(f"sigma_{lv}", level_sigma_scale)
        z = pm.Normal(f"z_{lv}", 0.0, 1.0, dims=lv)
        if parent is None or parent not in level_logits:
            mu = pm.Deterministic(f"{name}_logit_{lv}", global_mu + sigma * z, dims=lv)
        else:
            parent_of = hierarchy[lv]["parent_of"]
            parent_logit = level_logits[parent]
            mu = pm.Deterministic(
                f"{name}_logit_{lv}",
                parent_logit[parent_of] + sigma * z,
                dims=lv,
            )
        rate = pm.Deterministic(f"{name}_{lv}", pm.math.sigmoid(mu), dims=lv)
        level_logits[lv] = mu
        level_rates[lv] = rate

    leaf_idx = hierarchy[leaf_level]["idx"]
    # Register the index as Data so it appears in constant_data
    leaf_idx_data = pm.Data(f"{leaf_level}_idx", leaf_idx, dims=leaf_dim)
    leaf_rate = pm.Deterministic(
        f"{name}_leaf",
        level_rates[leaf_level][leaf_idx_data],
        dims=leaf_dim,
    )

    return {
        "level_logits": level_logits,
        "level_rates": level_rates,
        "leaf_rate": leaf_rate,
        "leaf_level": leaf_level,
    }
