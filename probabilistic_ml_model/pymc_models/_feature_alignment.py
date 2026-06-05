"""
Shared helpers for ``feature_catalogue``-aligned feature handling across all
PyMC model classes in :mod:`probabilistic_ml_model.pymc_models`.

Implements the actionable recommendations §12.3 from
``pymc_expected_returns_model.ipynb``:

* **Rec #1 — Type-aware coercion** (``coerce_by_data_type``): drives per-column
  dtype + bounded clipping from ``feature_catalogue.data_type``
  (``numeric`` / ``pct`` / ``flag`` / ``score`` / …) instead of a uniform
  ``astype('float64').fillna(0.0)``.
* **Rec #3 — Source-function provenance** (``stamp_feature_provenance``):
  copies ``feature_catalogue.source_function`` / ``calculation_type`` /
  ``data_type`` onto ``idata.constant_data[var].attrs`` so downstream lineage
  tooling can map every posterior coordinate back to the SQL/Python function
  that materialised it.
* **Rec #4 — Category-conflict guard** (``assert_disjoint_features``):
  optional ``strict=True`` mode that asserts a model's materialised
  ``feature_alias`` set is disjoint from any other model's previously
  attached set on the same ``InferenceData``.
* **Rec #7 — OOS shape contract** (``validate_oos_shape``): asserts that a
  ``new_arr`` passed into ``pm.set_data({"<model>_features": new_arr})`` has
  ``shape[1] == len(feature_aliases)`` before the swap, with the canonical
  feature-alias order reflecting ``feature_catalogue``.

The helpers are deliberately additive — every existing
``_align_*_features`` static method on the per-model classes still works
unchanged. New call-sites can opt-in via ``coerce_by_data_type(...)`` to get
the type-aware behaviour.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default per-data_type clipping / dtype rules. Aligned with the canonical
# values used in ``pml.vw_pymc_feature_catalogue.data_type``
# (sourced from ``pml.pml_df_metadata.data_type``).
_DTYPE_RULES: dict[str, dict[str, Any]] = {
    "pct": {"dtype": "float64", "clip": (-1.0, 1.0), "fill": 0.0},
    "ratio": {"dtype": "float64", "clip": (-1e6, 1e6), "fill": 0.0},
    "zscore": {"dtype": "float64", "clip": (-10.0, 10.0), "fill": 0.0},
    "score": {"dtype": "float64", "clip": (0.0, 100.0), "fill": 0.0},
    "flag": {"dtype": "int8", "clip": (0, 1), "fill": 0},
    "boolean": {"dtype": "int8", "clip": (0, 1), "fill": 0},
    "count": {"dtype": "int32", "clip": (0, None), "fill": 0},
    "numeric": {"dtype": "float64", "clip": None, "fill": 0.0},
    "level": {"dtype": "float64", "clip": None, "fill": 0.0},
    "growth": {"dtype": "float64", "clip": (-5.0, 5.0), "fill": 0.0},
}


@lru_cache(maxsize=4)
def load_feature_metadata_from_db(
    connection_string: Optional[str] = None,
) -> dict[str, dict[str, Optional[str]]]:
    """Load ``(category, calculation_type, data_type, source_function)``
    metadata per ``feature_alias`` from the pml single-source-of-truth
    ``pml.vw_pymc_feature_catalogue`` (backed by ``pml.pml_df_metadata``).

    The pml schema does not carry ``calculation_type`` / ``source_function``
    columns, so those keys are returned as ``None`` to preserve the dict
    contract consumed by :func:`coerce_by_data_type` (which only needs
    ``data_type``) and :func:`stamp_feature_provenance`.

    Returns an empty dict on any failure so callers can fall back gracefully
    to untyped behaviour.
    """
    if connection_string is None:
        connection_string = os.environ.get("DB_URL")
    if not connection_string:
        return {}
    try:
        from sqlalchemy import create_engine, text  # local import — optional dep
    except Exception:  # pragma: no cover
        return {}

    query = text("""
        SELECT DISTINCT feature_alias,
               category,
               NULL::text                     AS calculation_type,
               COALESCE(data_type, 'numeric') AS data_type,
               NULL::text                     AS source_function
        FROM pml.vw_pymc_feature_catalogue
        WHERE feature_alias IS NOT NULL
        """)
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            rows = conn.execute(query).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("load_feature_metadata_from_db failed: %s", exc)
        return {}

    out: dict[str, dict[str, Optional[str]]] = {}
    for r in rows:
        alias = r[0]
        if alias is None or alias in out:
            continue
        out[alias] = {
            "category": r[1],
            "calculation_type": r[2],
            "data_type": r[3],
            "source_function": r[4],
        }
    return out


def coerce_by_data_type(
    df: pd.DataFrame,
    feature_aliases: list[str],
    metadata: Optional[Mapping[str, Mapping[str, Optional[str]]]] = None,
    *,
    default_dtype: str = "float64",
    default_fill: float = 0.0,
) -> np.ndarray:
    """Type-aware analogue of ``df.reindex(columns=...).astype('float64').fillna(0.0)``.

    Parameters
    ----------
    df
        Input DataFrame indexed by ``isin`` (or any per-row id).
    feature_aliases
        Canonical column order — drives ``df.reindex`` and the returned
        matrix's column axis.
    metadata
        Mapping ``feature_alias -> {data_type, ...}``; typically the result
        of :func:`load_feature_metadata_from_db`. When ``None`` or missing
        for a given alias, falls back to ``default_dtype`` / ``default_fill``
        (legacy behaviour).

    Returns
    -------
    np.ndarray
        ``(n_rows, len(feature_aliases))`` matrix with per-column dtype +
        clipping derived from ``metadata[alias]["data_type"]``. Always
        returns ``float64`` so downstream PyMC ``pm.Data`` containers stay
        homogenous; the dtype rule is used to drive *clipping* and
        ``NaN``/inf handling, not the final dtype.
    """
    if not feature_aliases:
        return np.zeros((len(df), 0), dtype="float64")
    aligned = df.reindex(columns=feature_aliases)

    cols: list[np.ndarray] = []
    for alias in feature_aliases:
        rule_key = None
        if metadata and alias in metadata:
            rule_key = (metadata[alias].get("data_type") or "").lower() or None
        rule = _DTYPE_RULES.get(rule_key or "", None)

        col = aligned[alias].to_numpy()
        # Ensure float for clipping / NaN handling.
        col = np.asarray(col, dtype="float64")
        col = np.where(np.isfinite(col), col, np.nan)
        fill = rule["fill"] if rule else default_fill
        col = np.where(np.isnan(col), fill, col)
        if rule and rule.get("clip") is not None:
            lo, hi = rule["clip"]
            if lo is not None:
                col = np.maximum(col, lo)
            if hi is not None:
                col = np.minimum(col, hi)
        cols.append(col)
    return np.column_stack(cols).astype(default_dtype, copy=False)


def stamp_feature_provenance(
    idata: Any,
    var_name: str,
    feature_aliases: Iterable[str],
    metadata: Optional[Mapping[str, Mapping[str, Optional[str]]]] = None,
) -> None:
    """Stamp ``feature_catalogue`` provenance onto
    ``idata.constant_data[var_name].attrs``.

    Records (a) the canonical ``feature_alias`` order, and (b) compact
    ``source_function`` / ``calculation_type`` / ``data_type`` lookups so
    lineage tooling can resolve every posterior coordinate back to its SQL
    materialisation. No-op if ``idata`` lacks ``constant_data`` or the
    target variable.
    """
    try:
        cd = idata.constant_data  # type: ignore[attr-defined]
    except Exception:
        return
    if var_name not in cd.data_vars:
        return
    aliases = list(feature_aliases)
    attrs: dict[str, Any] = {"feature_aliases": aliases}
    if metadata:
        attrs["source_function"] = [metadata.get(a, {}).get("source_function") for a in aliases]
        attrs["calculation_type"] = [metadata.get(a, {}).get("calculation_type") for a in aliases]
        attrs["data_type"] = [metadata.get(a, {}).get("data_type") for a in aliases]
    cd[var_name].attrs.update(attrs)


def validate_oos_shape(
    new_arr: np.ndarray,
    feature_aliases: list[str],
    *,
    var_name: str = "<features>",
) -> None:
    """Assert ``new_arr.shape[1] == len(feature_aliases)`` for ``pm.set_data``
    swaps. Raises :class:`ValueError` with a descriptive message otherwise.
    """
    arr = np.asarray(new_arr)
    if arr.ndim != 2:
        raise ValueError(f"OOS swap for '{var_name}' expects a 2-D matrix, got shape {arr.shape}.")
    if arr.shape[1] != len(feature_aliases):
        raise ValueError(
            f"OOS swap for '{var_name}' shape mismatch: got {arr.shape[1]} cols, "
            f"expected {len(feature_aliases)} (catalogue feature_alias count)."
        )


def assert_disjoint_features(
    idata: Any,
    new_aliases: Iterable[str],
    *,
    new_var_name: str,
) -> None:
    """Recommendation #4 — category-conflict guard.

    Raises :class:`ValueError` if any alias in ``new_aliases`` already
    appears under a *different* ``constant_data`` variable on ``idata``
    (i.e. a previously-attached model's feature matrix).
    """
    try:
        cd = idata.constant_data  # type: ignore[attr-defined]
    except Exception:
        return
    new_set = set(new_aliases)
    conflicts: dict[str, list[str]] = {}
    for var in cd.data_vars:
        if var == new_var_name:
            continue
        existing = list(cd[var].attrs.get("feature_aliases", []))
        if not existing:
            # Try to recover from coords (any *_feature dim).
            for d in cd[var].dims:
                if d.endswith("_feature") or d == "feature":
                    existing = list(cd[var].coords[d].values)
                    break
        overlap = sorted(new_set.intersection(existing))
        if overlap:
            conflicts[var] = overlap
    if conflicts:
        raise ValueError(
            f"Category-conflict guard: aliases {sorted(new_set)} for "
            f"'{new_var_name}' overlap with: {conflicts}"
        )
