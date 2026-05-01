import importlib as _importlib
import warnings as _warnings

# ── Suppress ArviZ 1.0 MigrationWarning for ``arviz.InferenceData`` ─────────
# ArviZ 1.0 migrated ``InferenceData`` to ``xarray.DataTree``.  PyMC 5.x and
# legacy downstream code may still reference ``arviz.InferenceData`` (e.g.
# via ``hasattr``/type-hint resolution), which triggers a noisy
# ``MigrationWarning`` emitted from ``arviz/__init__.py`` on every attribute
# access.  Filter it package-wide; the project's compat shim
# (``_pymc_arviz_compat``) injects a functional ``InferenceData`` replacement
# so behaviour is preserved.
_warnings.filterwarnings(
    "ignore",
    message=r".*arviz\.InferenceData is no longer available.*",
)

# ── Patch arviz for PyMC 5.x compatibility with arviz >= 1.0 ─────────────────
try:
    from probabilistic_ml_model._pymc_arviz_compat import patch as _patch_arviz

    _patch_arviz()
except Exception:
    pass

# ── Lazy-loaded PML model aliases ─────────────────────────────────────────────
# Maps public alias → (relative module path, attribute name inside that module).
# Each entry is resolved on first access via __getattr__.
_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "ProbabilisticLinearRegression": (
        ".pymc_models.ProbabilisticLinearRegressionModel",
        "ProbabilisticLinearRegression",
    ),
    "KalmanFilterPriceTarget": (".pymc_models.KalmanFilterModel", "KalmanFilterPriceTarget"),
    "DCFPriceTarget": (".pymc_models.DCF_PriceTargetModel", "DCFPriceTarget"),
    "EarningsBeatBayesian": (".pymc_models.EarningsBeatModel", "EarningsBeatBayesian"),
    "DividendSafetyBayesian": (".pymc_models.DividendSafetyModel", "DividendSafetyBayesian"),
    "PriceTargetAchievement": (".pymc_models.PriceTargetModel", "PriceTargetAchievement"),
    "AccountingAnomalyBayesian": (
        ".pymc_models.AccountingAnomalyModel",
        "AccountingAnomalyBayesian",
    ),
    "CreditRiskBayesian": (".pymc_models.CreditRiskModel", "CreditRiskBayesian"),
    "MonteCarloReturnSimulation": (
        ".pymc_models.MonteCarloSimulation",
        "MonteCarloReturnSimulation",
    ),
    # Top-level helper exposed alongside MonteCarloReturnSimulation
    "monte_carlo_fit": (".pymc_models.MonteCarloSimulation", "fit"),
    # Baseline orchestration entry points
    "baseline_main": (".pymc_models.BaselineProbabilityModel", "main"),
    "PipelineConfig": (".pymc_models.BaselineProbabilityModel", "PipelineConfig"),
    # Shared feature-alignment helpers (used by every per-model `_align_*_features`
    # and by §13 of `pymc_expected_returns_model.ipynb`).
    "assert_disjoint_features": (
        ".pymc_models._feature_alignment",
        "assert_disjoint_features",
    ),
    "coerce_by_data_type": (
        ".pymc_models._feature_alignment",
        "coerce_by_data_type",
    ),
    "load_feature_metadata_from_db": (
        ".pymc_models._feature_alignment",
        "load_feature_metadata_from_db",
    ),
    "stamp_feature_provenance": (
        ".pymc_models._feature_alignment",
        "stamp_feature_provenance",
    ),
    "validate_oos_shape": (
        ".pymc_models._feature_alignment",
        "validate_oos_shape",
    ),
    # PyTensor compile-kwargs helper used internally by every model's `fit(...)`.
    "get_pytensor_compile_kwargs": (
        ".pymc_models._pytensor_compat",
        "get_pytensor_compile_kwargs",
    ),
}

# Lazy imports that use an absolute module path + identical attribute name.
# ⚠️ fragile — only works if repo root is on sys.path
_LAZY_IMPORT_DIRECT: dict[str, str] = {
    "BaselinePipeline": "expected_returns_v4",
}

# Subpackages advertised in __all__; resolved via normal import when accessed.
_SUBPACKAGES: set[str] = {
    "data_utils",
    "config",
    "utils",
    "logging_config",
    "optimized_ops",
    "statistical_functions",
    "visualizations",
    "pymc_models",
    "pipeline_runners",
}


def _lazy_import(name: str) -> object:
    """Resolve a lazily-imported attribute by *name*."""
    if name in _LAZY_IMPORT_MAP:
        mod_path, attr_name = _LAZY_IMPORT_MAP[name]
        module = _importlib.import_module(mod_path, __package__)
        return getattr(module, attr_name)

    if name in _LAZY_IMPORT_DIRECT:
        module = _importlib.import_module(_LAZY_IMPORT_DIRECT[name])
        return getattr(module, name)

    if name in _SUBPACKAGES:
        return _importlib.import_module(f".{name}", __package__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> object:
    """Lazy imports to avoid hard dependency on pymc/arviz at import time."""
    obj = _lazy_import(name)
    globals()[name] = obj  # cache so __getattr__ isn't called again
    return obj


__all__ = [
    # PML model public aliases (match __getattr__ keys)
    *_LAZY_IMPORT_MAP,
    *_LAZY_IMPORT_DIRECT,
    # Subpackages
    *_SUBPACKAGES,
]
