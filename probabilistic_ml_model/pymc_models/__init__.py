import importlib as _importlib

# Maps public alias → (relative module path, attribute name inside that module).
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
}


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
