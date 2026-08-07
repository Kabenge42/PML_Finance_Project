import importlib as _importlib

# Maps public alias → (relative module path, attribute name inside that module).
# Module paths mirror the on-disk layout of probabilistic_ml_model/pymc_models/.
_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "ProbabilisticLinearRegression": (
        ".ProbabilisticLinearRegressionModel",
        "ProbabilisticLinearRegression",
    ),
    "KalmanFilterPriceTarget": (".KalmanFilterModel", "KalmanFilterPriceTarget"),
    "KalmanPanelInputs": (".KalmanFilterModel", "KalmanPanelInputs"),
    "build_fused_kalman_pt_model": (".KalmanFilterModel", "build_fused_kalman_pt_model"),
    "KALMAN_DRIFT_EXCLUDED_FEATURES": (
        ".KalmanFilterModel",
        "KALMAN_DRIFT_EXCLUDED_FEATURES",
    ),
    "KALMAN_TIME_COVARIATE_PREFIX": (".KalmanFilterModel", "KALMAN_TIME_COVARIATE_PREFIX"),
    "KALMAN_NOISE_WIDENER_FEATURES": (".KalmanFilterModel", "KALMAN_NOISE_WIDENER_FEATURES"),
    "KALMAN_RANGE_WIDENER_FEATURE": (".KalmanFilterModel", "KALMAN_RANGE_WIDENER_FEATURE"),
    "KALMAN_CONSENSUS_SIGMA_FEATURE": (".KalmanFilterModel", "KALMAN_CONSENSUS_SIGMA_FEATURE"),
    "KALMAN_VOL_DRIFT_FEATURE": (".KalmanFilterModel", "KALMAN_VOL_DRIFT_FEATURE"),
    "KALMAN_TILT_FEATURES": (".KalmanFilterModel", "KALMAN_TILT_FEATURES"),
    "KALMAN_TILT_FEATURE_ORDER": (".KalmanFilterModel", "KALMAN_TILT_FEATURE_ORDER"),
    "FISCAL_HORIZONS": (".KalmanFilterModel", "FISCAL_HORIZONS"),
    "FiscalHorizon": (".KalmanFilterModel", "FiscalHorizon"),
    "AGO_HISTORY_RE": (".KalmanFilterModel", "AGO_HISTORY_RE"),
    "AGO_SUFFIX_PATTERN": (".KalmanFilterModel", "AGO_SUFFIX_PATTERN"),
    "DCFPriceTarget": (".DCF_PriceTargetModel", "DCFPriceTarget"),
    "EarningsBeatBayesian": (".EarningsBeatModel", "EarningsBeatBayesian"),
    "DividendSafetyBayesian": (".DividendSafetyModel", "DividendSafetyBayesian"),
    "PriceTargetAchievement": (".PriceTargetModel", "PriceTargetAchievement"),
    "build_fused_price_target_model": (".PriceTargetModel", "build_fused_price_target_model"),
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
    # Bayesian-workflow stage helpers (see _workflow.py)
    "MIN_ESS_GATE": ("._workflow", "MIN_ESS_GATE"),
    "attach_log_likelihood": ("._workflow", "attach_log_likelihood"),
    "build_sample_kwargs": ("._workflow", "build_sample_kwargs"),
    "log_sample_diagnostics": ("._workflow", "log_sample_diagnostics"),
    "posterior_dataset": ("._workflow", "posterior_dataset"),
    "posterior_predictive_check": ("._workflow", "posterior_predictive_check"),
    "prior_predictive_check": ("._workflow", "prior_predictive_check"),
    # Decision analysis — CVaR-aware risk book (see RiskBookModel.py)
    "RiskBook": (".RiskBookModel", "RiskBook"),
    "compute_cvar_aware_book": (".RiskBookModel", "compute_cvar_aware_book"),
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