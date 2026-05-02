import importlib as _importlib

# Maps public alias → (relative module path, attribute name inside that module).
# Module paths mirror the on-disk layout of probabilistic_ml_model/pymc_models/.
_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "ProbabilisticLinearRegression": (
        ".ProbabilisticLinearRegressionModel",
        "ProbabilisticLinearRegression",
    ),
    "KalmanFilterPriceTarget": (".KalmanFilterModel", "KalmanFilterPriceTarget"),
    "DCFPriceTarget": (".DCF_PriceTargetModel", "DCFPriceTarget"),
    "EarningsBeatBayesian": (".EarningsBeatModel", "EarningsBeatBayesian"),
    "DividendSafetyBayesian": (".DividendSafetyModel", "DividendSafetyBayesian"),
    "PriceTargetAchievement": (".PriceTargetModel", "PriceTargetAchievement"),
    "AccountingAnomalyBayesian": (".AccountingAnomalyModel", "AccountingAnomalyBayesian"),
    "CreditRiskBayesian": (".CreditRiskModel", "CreditRiskBayesian"),
    "MonteCarloReturnSimulation": (".MonteCarloSimulation", "MonteCarloReturnSimulation"),
    "monte_carlo_fit": (".MonteCarloSimulation", "fit"),
    "baseline_main": (".BaselineProbabilityModel", "main"),
    "PipelineConfig": (".BaselineProbabilityModel", "PipelineConfig"),
    "assert_disjoint_features": ("._feature_alignment", "assert_disjoint_features"),
    "coerce_by_data_type": ("._feature_alignment", "coerce_by_data_type"),
    "load_feature_metadata_from_db": ("._feature_alignment", "load_feature_metadata_from_db"),
    "stamp_feature_provenance": ("._feature_alignment", "stamp_feature_provenance"),
    "validate_oos_shape": ("._feature_alignment", "validate_oos_shape"),
    "get_pytensor_compile_kwargs": ("._pytensor_compat", "get_pytensor_compile_kwargs"),
    # Multi-level hierarchical shrinkage infrastructure (see _hierarchy.py)
    "HIERARCHICAL_CATEGORY_COLS": ("._hierarchy", "HIERARCHICAL_CATEGORY_COLS"),
    "PARENT_MAP": ("._hierarchy", "PARENT_MAP"),
    "build_hierarchy_indices": ("._hierarchy", "build_hierarchy_indices"),
    "build_nested_logit_normal_rates": ("._hierarchy", "build_nested_logit_normal_rates"),
    "coerce_categories": ("._hierarchy", "coerce_categories"),
    "_resolve_prior_sigma": ("._hierarchy", "_resolve_prior_sigma"),
}

# Correct the leading-underscore module paths above (they collapsed to "__"
# in the table for visual alignment). The on-disk modules are
# `_feature_alignment.py` and `_pytensor_compat.py` (single leading underscore).


def __getattr__(name: str) -> object:
    """Lazy imports to avoid hard dependency on pymc/arviz at import time."""
    if name in _LAZY_IMPORT_MAP:
        mod_path, attr_name = _LAZY_IMPORT_MAP[name]
        module = _importlib.import_module(mod_path, __package__)
        obj = getattr(module, attr_name)
        globals()[name] = obj  # cache so __getattr__ isn't called again
        return obj
    if name == "BaselinePipeline":
        # ⚠️ fragile — only works if repo root is on sys.path
        from expected_returns_v4 import BaselinePipeline

        globals()[name] = BaselinePipeline
        return BaselinePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_LAZY_IMPORT_MAP,
]
