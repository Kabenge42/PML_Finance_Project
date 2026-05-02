"""
Compatibility shim: patch ``arviz`` so that PyMC 5.x can import against arviz >= 1.0.

arviz 1.0 removed ``InferenceData``, ``concat``, and reorganised submodules
(``arviz.data.base``, ``arviz.stats``) that PyMC's backends rely on.
This module injects lightweight replacements into ``arviz`` **before** PyMC is
imported, so that ``from arviz import InferenceData, concat`` succeeds.

Usage -- call ``patch()`` once at process start, before any ``import pymc``::

    from probabilistic_ml_model._pymc_arviz_compat import patch
    patch()
    import pymc as pm  # now works
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Dict, Optional, Sequence, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)

_PATCHED = False




# ---------------------------------------------------------------------------
# InferenceData shim
# ---------------------------------------------------------------------------


class InferenceData:
    """Minimal ``arviz.InferenceData`` replacement.

    Accepts keyword arguments whose values are ``xarray.Dataset`` objects
    (one per group: posterior, sample_stats, observed_data, ...).
    """

    def __init__(self, *, save_warmup: bool = False, attrs: dict[str, Any] | None = None, **groups: Any) -> None:
        from xarray import Dataset

        self._groups: dict[str, Any] = {}
        self.save_warmup = save_warmup
        self.attrs = attrs or {}

        warmup_prefix = "warmup_"
        for name, ds in groups.items():
            if ds is None:
                continue
            # PyMC's backends return (data, warmup_data) tuples for some
            # groups (posterior, sample_stats).  Unpack them so that each
            # group stores only an xarray.Dataset.
            if isinstance(ds, tuple) and len(ds) == 2:
                main_ds, warmup_ds = ds
                # store warmup counterpart when requested
                if save_warmup and warmup_ds is not None:
                    warmup_name = f"{warmup_prefix}{name}"
                    if isinstance(warmup_ds, Dataset):
                        self._groups[warmup_name] = warmup_ds
                    else:
                        try:
                            self._groups[warmup_name] = Dataset(warmup_ds)
                        except Exception:
                            self._groups[warmup_name] = warmup_ds
                ds = main_ds
                if ds is None:
                    continue
            if not save_warmup and name.startswith(warmup_prefix):
                continue
            if not isinstance(ds, Dataset):
                try:
                    ds = Dataset(ds)
                except Exception:
                    pass
            self._groups[name] = ds

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._groups[name]
        except KeyError:
            raise AttributeError(f"'InferenceData' object has no group '{name}'") from None

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like ``.get()`` used by PyMC convergence checks."""
        return self._groups.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow ``idata['posterior']`` access used by PyMC convergence checks."""
        try:
            return self._groups[key]
        except KeyError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._groups

    def items(self):
        return self._groups.items()

    def keys(self):
        return self._groups.keys()

    def values(self):
        return self._groups.values()

    def __iter__(self):
        return iter(self._groups)

    def __len__(self):
        return len(self._groups)

    def add_groups(self, groups, coords=None, dims=None, **kwargs):
        """Minimal ``InferenceData.add_groups`` replacement."""
        for name, ds in groups.items():
            self._groups[name] = ds

    def __repr__(self) -> str:
        groups = ", ".join(self._groups.keys())
        return f"InferenceData({groups})"

    def extend(self, other: "InferenceData", join: str = "left") -> None:
        if other is None:
            return
        for name, ds in other._groups.items():
            if name not in self._groups:
                self._groups[name] = ds

    def to_dict(self) -> dict:
        return dict(self._groups)

    def groups(self) -> list[str]:
        return list(self._groups.keys())

    @property
    def posterior(self):
        return self._groups.get("posterior")

    @posterior.setter
    def posterior(self, value):
        self._groups["posterior"] = value


# ---------------------------------------------------------------------------
# concat shim
# ---------------------------------------------------------------------------


def concat(
    ids: Sequence[InferenceData],
    *,
    dim: Optional[str] = None,
    copy: bool = True,
    inplace: bool = False,
) -> InferenceData:
    """Minimal ``arviz.concat`` replacement."""
    if not ids:
        return InferenceData()

    import xarray as xr

    base = ids[0] if inplace else InferenceData(**ids[0].to_dict())
    for other in ids[1:]:
        if other is None:
            continue
        for group_name in other.groups():
            other_ds = getattr(other, group_name)
            if group_name in base._groups:
                existing = base._groups[group_name]
                if dim and isinstance(existing, xr.Dataset) and isinstance(other_ds, xr.Dataset):
                    try:
                        base._groups[group_name] = xr.concat([existing, other_ds], dim=dim)
                    except Exception:
                        base._groups[group_name] = other_ds
            else:
                base._groups[group_name] = other_ds
    return base


# ---------------------------------------------------------------------------
# arviz.data.base shim types
# ---------------------------------------------------------------------------

CoordSpec = Dict[str, Any]
DimSpec = Dict[str, Sequence[str]]
WARMUP_TAG = "warmup_"


def requires(group_names):
    """Decorator for ``arviz.data.base.requires``.

    Skips the decorated method (returns ``None``) when any of the
    required attributes on *self* is ``None``.  This replicates the
    guard behaviour of legacy arviz 0.x that PyMC 5.x relies on.
    """
    if isinstance(group_names, str):
        group_names = [group_names]

    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for grp in group_names:
                if getattr(self, grp, None) is None:
                    return None
            return func(self, *args, **kwargs)

        wrapper.__requires__ = group_names
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# patch()
# ---------------------------------------------------------------------------


def _ensure_module(fqn: str, parent_attr: str | None = None, parent=None) -> types.ModuleType:
    """Return (and register) a synthetic module if *fqn* is not yet importable."""
    mod = sys.modules.get(fqn)
    if mod is None:
        mod = types.ModuleType(fqn)
        mod.__package__ = fqn.rsplit(".", 1)[0] if "." in fqn else fqn
        mod.__path__ = []
        sys.modules[fqn] = mod
    if parent is not None and parent_attr:
        setattr(parent, parent_attr, mod)
    return mod


def patch() -> None:
    """Monkey-patch ``arviz`` so that PyMC 5.x can import successfully."""
    global _PATCHED
    if _PATCHED:
        return

    import arviz

    # ── Top-level symbols ────────────────────────────────────────────────
    if not hasattr(arviz, "InferenceData"):
        arviz.InferenceData = InferenceData  # type: ignore[attr-defined]
    if not hasattr(arviz, "concat"):
        arviz.concat = concat  # type: ignore[attr-defined]

    # ── arviz.data  / arviz.data.base  / arviz.data.inference_data ───────
    arviz_data = _ensure_module("arviz.data", "data", arviz)

    arviz_data_base = _ensure_module("arviz.data.base", "base", arviz_data)
    from arviz_base.base import dict_to_dataset as _real_dict_to_dataset
    from arviz_base.base import make_attrs as _real_make_attrs

    arviz_data_base.CoordSpec = CoordSpec  # type: ignore[attr-defined]
    arviz_data_base.DimSpec = DimSpec  # type: ignore[attr-defined]
    arviz_data_base.requires = requires  # type: ignore[attr-defined]

    # Wrapper for make_attrs to handle both 'library' and 'inference_library'
    # arviz 0.23.4 (internals) calls it with 'library',
    # but arviz_base.make_attrs expects 'inference_library'.
    def _make_attrs_wrapper(attrs=None, library=None, inference_library=None):
        if inference_library is None and library is not None:
            inference_library = library
        return _real_make_attrs(attrs=attrs, inference_library=inference_library)

    # Wrapper for dict_to_dataset to handle 'library' -> 'inference_library'
    # and 'default_dims' -> 'sample_dims' (renamed in arviz_base >= 1.0)
    def _dict_to_dataset_wrapper(data, *, library=None, inference_library=None, **kwargs):
        if inference_library is None and library is not None:
            inference_library = library
        # PyMC 5.x passes 'default_dims' which was renamed to 'sample_dims' in arviz_base
        if "default_dims" in kwargs:
            default_dims = kwargs.pop("default_dims")
            if "sample_dims" not in kwargs:
                kwargs["sample_dims"] = default_dims
        return _real_dict_to_dataset(data, inference_library=inference_library, **kwargs)

    arviz_data_base.make_attrs = _make_attrs_wrapper  # type: ignore[attr-defined]
    arviz_data_base.dict_to_dataset = _dict_to_dataset_wrapper  # type: ignore[attr-defined]

    arviz_data_id = _ensure_module("arviz.data.inference_data", "inference_data", arviz_data)
    arviz_data_id.WARMUP_TAG = WARMUP_TAG  # type: ignore[attr-defined]
    arviz_data_id.InferenceData = InferenceData  # type: ignore[attr-defined]

    # ── arviz.stats  (PyMC iterates az.stats.__all__) ────────────────────
    try:
        import arviz_stats as _real_stats

        arviz_stats_mod = _ensure_module("arviz.stats", "stats", arviz)
        # Copy all public callables from arviz_stats
        _all_names = []
        for name in dir(_real_stats):
            if name.startswith("_"):
                continue
            obj = getattr(_real_stats, name)
            setattr(arviz_stats_mod, name, obj)
            _all_names.append(name)
        arviz_stats_mod.__all__ = _all_names  # type: ignore[attr-defined]

        # Also expose common functions at arviz top-level (PyMC may access them)
        for name in ("summary", "ess", "rhat", "mcse", "hdi", "loo", "waic", "bfmi"):
            if hasattr(_real_stats, name) and not hasattr(arviz, name):
                setattr(arviz, name, getattr(_real_stats, name))
    except ImportError:
        logger.warning("arviz_stats not available; arviz.stats shim will be empty")
        arviz_stats_mod = _ensure_module("arviz.stats", "stats", arviz)
        arviz_stats_mod.__all__ = []  # type: ignore[attr-defined]

    # ── arviz.plots (PyMC iterates az.plots.__all__) ───────────────────
    try:
        import arviz_plots as _real_plots

        arviz_plots_mod = _ensure_module("arviz.plots", "plots", arviz)
        _all_plot_names = []
        for name in dir(_real_plots):
            if name.startswith("_"):
                continue
            setattr(arviz_plots_mod, name, getattr(_real_plots, name))
            _all_plot_names.append(name)
        arviz_plots_mod.__all__ = _all_plot_names  # type: ignore[attr-defined]
    except ImportError:
        logger.warning("arviz_plots not available; arviz.plots shim will be empty")
        arviz_plots_mod = _ensure_module("arviz.plots", "plots", arviz)
        arviz_plots_mod.__all__ = []  # type: ignore[attr-defined]

    # ── arviz.labels (some PyMC code may reference it) ───────────────────
    try:
        from arviz_base import labels as _real_labels

        arviz_labels = _ensure_module("arviz.labels", "labels", arviz)
        for name in dir(_real_labels):
            if not name.startswith("_"):
                setattr(arviz_labels, name, getattr(_real_labels, name))
    except ImportError:
        pass

    # ── arviz.sel_utils ──────────────────────────────────────────────────
    try:
        from arviz_base import sel_utils as _real_sel

        _ensure_module("arviz.sel_utils", "sel_utils", arviz)
        sys.modules["arviz.sel_utils"] = _real_sel
    except ImportError:
        pass

    _PATCHED = True
    logger.debug("arviz patched for PyMC 5.x compatibility with arviz >= 1.0")


# ---------------------------------------------------------------------------
# Typing alias — ArviZ migration (classic InferenceData <-> arviz-base DataTree)
# ---------------------------------------------------------------------------
# Per the ArviZ migration guide
# (https://python.arviz.org/en/latest/user_guide/migration_guide.html),
# arviz-base 1.x replaces ``arviz.InferenceData`` with ``xarray.DataTree``.
# PyMC 5.x still emits ``arviz.InferenceData`` from ``pm.sample()``, so this
# alias lets downstream type annotations accept either object. Runtime
# behaviour is unchanged: PyMC returns InferenceData, the shim above keeps
# that working against arviz-base 1.x, and downstream callers can transparently
# consume ``xarray.DataTree`` instances when produced (e.g. via xarray I/O).
if TYPE_CHECKING:
    import xarray as _xr
    import arviz as _az

    InferenceLike = Union[_az.InferenceData, _xr.DataTree]
else:
    try:
        import xarray as _xr  # noqa: F401

        _DataTree = getattr(_xr, "DataTree", None)
        try:
            import arviz as _az

            _AzInferenceData = getattr(_az, "InferenceData", InferenceData)
        except Exception:
            _AzInferenceData = InferenceData
        if _DataTree is not None:
            InferenceLike = Union[_AzInferenceData, _DataTree]  # type: ignore[misc]
        else:
            InferenceLike = _AzInferenceData  # type: ignore[misc]
    except Exception:  # pragma: no cover - defensive
        InferenceLike = Any  # type: ignore[misc]


def to_datatree(idata: Any) -> Any:
    """Best-effort conversion of an ``arviz.InferenceData`` to ``xarray.DataTree``.

    Returns ``idata`` unchanged when conversion is not possible (e.g. arviz-base
    not installed, or ``idata`` is already a ``DataTree``). PyMC's ``pm.sample()``
    currently returns ``InferenceData``; downstream code that prefers the new
    ``DataTree`` interface (per the arviz-base 1.x migration) can route results
    through this helper without breaking the legacy code path.
    """
    try:
        import xarray as xr
    except Exception:
        return idata
    DataTree = getattr(xr, "DataTree", None)
    if DataTree is None:
        return idata
    if isinstance(idata, DataTree):
        return idata
    # InferenceData is dict-like ({group_name: Dataset}); DataTree.from_dict
    # accepts exactly that mapping.
    try:
        groups = {name: getattr(idata, name) for name in idata.groups()}
        return DataTree.from_dict(groups)
    except Exception:  # pragma: no cover - defensive
        return idata
