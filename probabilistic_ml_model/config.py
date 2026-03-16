"""
Configuration management module for PML Finance Project.

This module provides configuration management via YAML/JSON files
and environment variables, with sensible defaults.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PMLConfig:
    """Configuration dataclass for Finance ML platform."""

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    model_dir: Path = field(default_factory=lambda: Path("../outputs/regression"))
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @property
    def analytics_dir(self) -> Path:
        """Get analytics output directory."""
        path = self.output_dir / "analytics"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models_output_dir(self) -> Path:
        """Get regression output directory."""
        path = self.output_dir / "regression"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def eda_dir(self) -> Path:
        """Get base EDA output directory."""
        path = self.output_dir / "eda"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def eda_with_importance_dir(self) -> Path:
        """Get EDA with importance output directory."""
        path = self.output_dir / "eda" / "eda_with_importance"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def eda_with_multivariate_dir(self) -> Path:
        """Get EDA with multivariate analysis output directory."""
        path = self.output_dir / "eda" / "eda_with_multivariate"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def enhanced_eda_dir(self) -> Path:
        """Get enhanced EDA output directory."""
        path = self.output_dir / "eda" / "enhanced_eda"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def financial_data_quality_reports_dir(self) -> Path:
        """Get financial data quality reporting output directory."""
        path = self.output_dir / "eda" / "financial_data_quality_reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def catalog_dir(self) -> Path:
        """Get data catalog output directory (Phase 9.1)."""
        path = self.output_dir / "catalog"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def preprocessing_dir(self) -> Path:
        """Get preprocessing output directory (Phase 9.1)."""
        path = self.output_dir / "preprocessing"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def features_dir(self) -> Path:
        """Get features output directory (Phase 9.3)."""
        path = self.output_dir / "features"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def classification_dir(self) -> Path:
        """Get classification output directory (Phase 9.4)."""
        path = self.output_dir / "classification"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def evaluation_dir(self) -> Path:
        """Get evaluation output directory (Phase 9.6)."""
        path = self.output_dir / "evaluation"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def plots_dir(self) -> Path:
        """Get plots output directory."""
        path = self.output_dir / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def reporting_dir(self) -> Path:
        """Get reporting output directory (Phase 9.8)."""
        path = self.output_dir / "reporting"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dashboards_dir(self) -> Path:
        """Get dashboards output directory."""
        path = self.output_dir / "dashboards"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_output_structure(self) -> None:
        """Create all Phase 9.1-9.8 output subdirectories."""
        # Create base output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create Phase 9.1-9.8 aligned subdirectories
        self.catalog_dir.mkdir(parents=True, exist_ok=True)  # Phase 9.1
        self.preprocessing_dir.mkdir(parents=True, exist_ok=True)  # Phase 9.1
        self.eda_dir.mkdir(parents=True, exist_ok=True)  # Phase 9.2
        self.features_dir.mkdir(parents=True, exist_ok=True)  # Phase 9.3
        self.classification_dir.mkdir(parents=True, exist_ok=True)  # Phase 9.4
        self.models_output_dir.mkdir(parents=True, exist_ok=True)  # Phase 9.5 (regression)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)  # Phase 9.6
        self.analytics_dir.mkdir(parents=True, exist_ok=True)  # Phase 9.7
        self.reporting_dir.mkdir(parents=True, exist_ok=True)  # Phase 9.8
        self.plots_dir.mkdir(parents=True, exist_ok=True)  # Visualizations
        self.dashboards_dir.mkdir(parents=True, exist_ok=True)  # Dashboard data

        # Create EDA subdirectories
        self.eda_with_importance_dir.mkdir(parents=True, exist_ok=True)
        self.eda_with_multivariate_dir.mkdir(parents=True, exist_ok=True)
        self.enhanced_eda_dir.mkdir(parents=True, exist_ok=True)
        self.financial_data_quality_reports_dir.mkdir(parents=True, exist_ok=True)

    # Database configuration
    db_url: Optional[str] = "postgresql+psycopg2://postgres:@localhost:5432/postgres"
    DB_EQUITIES_SCHEMA: str = "public"
    db_table: str = "equities"

    # Model configuration
    model_version: str = "v9_10"
    random_seed: int = 42
    n_jobs: int = -1

    # Memory and performance
    memory_limit: Optional[str] = None

    # Feature engineering
    feature_engineering_enabled: bool = True
    target_column: str = "price_target"

    # Logging
    log_level: str = "INFO"
    tf_cpp_min_log_level: str = "2"

    def __post_init__(self):
        """Convert string paths to Path objects."""
        for attr in ["data_dir", "model_dir", "cache_dir", "output_dir"]:
            value = getattr(self, attr)
            if isinstance(value, str):
                setattr(self, attr, Path(value))

    @classmethod
    def from_env(cls) -> PMLConfig:
        """Create configuration from environment variables."""
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            model_dir=Path(os.getenv("MODEL_DIR", "../outputs/regression")),
            cache_dir=Path(os.getenv("CACHE_DIR", ".cache")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
            db_url=os.getenv("DB_URL"),
            DB_EQUITIES_SCHEMA=os.getenv("DB_EQUITIES_SCHEMA", "public"),
            db_table=os.getenv("DB_TABLE", "equities"),
            model_version=os.getenv("MODEL_VERSION", "v9_10"),
            random_seed=int(os.getenv("RANDOM_SEED", "42")),
            n_jobs=int(os.getenv("N_JOBS", "-1")),
            memory_limit=os.getenv("MEMORY_LIMIT"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            tf_cpp_min_log_level=os.getenv("TF_CPP_MIN_LOG_LEVEL", "2"),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> PMLConfig:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PMLConfig:
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config. Install with: pip install pyyaml"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def to_json(self, path: Path | str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to {path}")

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config. Install with: pip install pyyaml"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

        logger.info(f"Configuration saved to {path}")

    def apply_to_env(self) -> None:
        """Apply configuration to environment variables."""
        os.environ["DATA_DIR"] = str(self.data_dir)
        os.environ["MODEL_DIR"] = str(self.model_dir)
        os.environ["CACHE_DIR"] = str(self.cache_dir)
        os.environ["OUTPUT_DIR"] = str(self.output_dir)

        if self.db_url:
            os.environ["DB_URL"] = self.db_url

        os.environ["DB_TABLE"] = self.db_table
        os.environ["DB_EQUITIES_SCHEMA"] = self.DB_EQUITIES_SCHEMA
        os.environ["MODEL_VERSION"] = self.model_version
        os.environ["RANDOM_SEED"] = str(self.random_seed)
        os.environ["N_JOBS"] = str(self.n_jobs)
        os.environ["LOG_LEVEL"] = self.log_level
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = self.tf_cpp_min_log_level

        if self.memory_limit:
            os.environ["MEMORY_LIMIT"] = self.memory_limit


def load_config(
    config_path: Optional[Path | str] = None,
    use_env: bool = True,
    output_dir: Optional[Path | str] = None,
) -> PMLConfig:
    """
    Load configuration from file or environment variables.

    Args:
        config_path: Path to JSON or YAML config file (optional)
        use_env: If True, load from environment variables when config_path is None
        output_dir: Optional output directory to override default/env value (avoids config mutation)

    Returns:
        PMLConfig instance

    Examples:
        >>> # Load from environment variables
        >>> config = load_config()

        >>> # Load from JSON file
        >>> config = load_config("config.json")

        >>> # Load from YAML file
        >>> config = load_config("config.yaml")

        >>> # Load with custom output directory
        >>> config = load_config(output_dir="custom_outputs")
    """
    if config_path is not None:
        config_path = Path(config_path)

        if config_path.suffix == ".json":
            config = PMLConfig.from_json(config_path)
        elif config_path.suffix in [".yaml", ".yml"]:
            config = PMLConfig.from_yaml(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    elif use_env:
        config = PMLConfig.from_env()
    else:
        config = PMLConfig()

    # Override output_dir if provided (avoids config mutation anti-pattern)
    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        config.output_dir = output_dir

    return config


# Global configuration instance (lazy-loaded)
_global_config: Optional[PMLConfig] = None


def get_config() -> PMLConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: PMLConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to None (will be reloaded on next get_config)."""
    global _global_config
    _global_config = None
