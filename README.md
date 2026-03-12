# PML Finance Project

A comprehensive platform for probabilistic equity screening, feature engineering, and machine learning modeling across global financial markets.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![Package Version](https://img.shields.io/badge/version-0.9.5-green)](pyproject.toml)

## Overview

The PML Finance Project is a robust solution for financial data analysis, probabilistic machine learning modeling, and portfolio optimization. It implements a structured **8-Phase ML Workflow** (Phases 9.1–9.8) for data quality, advanced feature engineering, and reliable model evaluation, followed by a **7-Phase Portfolio Optimization** module.

### Key Features

- **Probabilistic ML Models**: Monte Carlo simulation, Kalman filtering, DCF price-target regression, Bayesian earnings-beat analysis, credit risk estimation, dividend safety scoring, and accounting anomaly detection.
- **8-Phase ML Workflow**: Data ingestion, preprocessing (winsorization, 6-step imputation), EDA, advanced feature engineering, feature selection, model training (regression/classification), and error analysis.
- **7-Phase Portfolio Optimization**: Stock selection, return prediction, risk-adjusted optimization (Efficient Frontier), backtesting, and interactive dashboards.
- **Analytics Module**: 15 stock screeners, Bayesian/MCMC statistical analysis, Kalman/Copula methods, and interactive Plotly visualizations.
- **Unified Schema Module**: Single source of truth for financial columns, ensuring alignment between SQL databases and Python data structures.
- **Flexible ETL Pipeline**: Decoupled configuration handling multiple data sources (CSV, SQL) with built-in currency conversion and data validation.
- **Interactive Dashboards**: Integrated Streamlit, Dash, and Plotly applications for market monitoring, earnings analytics, and portfolio visualization.

### Core Models (`probabilistic_ml_model/`)

- **Monte Carlo Simulation** — probabilistic upside/downside distributions
- **Price Target Achievement** — probability-weighted expected returns
- **Kalman Filtered Targets** — noise-reduced price target signals
- **Earnings Beat Analysis** — three-layer Bayesian beat probability
- **Credit Risk Analysis** — Bayesian distress estimation
- **Dividend Safety Analysis** — dividend cut probability with FCF coverage
- **Accounting Anomaly Detection** — multi-layered statistical anomaly detection
- **DCF Price Target Model** — discounted cash flow regression
- **Probabilistic Linear Regression** — Bayesian linear regression model

## Tech Stack

| Category            | Technologies                                                                       |
|:--------------------|:-----------------------------------------------------------------------------------|
| **Language**        | Python 3.12 / 3.13 / 3.14                                                          |
| **Package Manager** | `pip`, `pipenv` (`Pipfile`), `setuptools` (`pyproject.toml`)                       |
| **ML Frameworks**   | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `SHAP`, `TensorFlow` |
| **Bayesian**        | `PyMC`, `PyTensor`, `ArviZ`, `xarray`                                              |
| **Data Processing** | `pandas`, `NumPy`, `SciPy`, `statsmodels`, `numba`, `imbalanced-learn`             |
| **Visualization**   | `Plotly`, `Matplotlib`, `Seaborn`                                                  |
| **Dashboards**      | `Streamlit`, `Dash`, `dash-bootstrap-components`                                   |
| **Database**        | `PostgreSQL` (`psycopg2`), `SQLAlchemy`, `SQLite`                                  |
| **Testing**         | `pytest`, `unittest`                                                               |
| **Code Quality**    | `Black`, `Flake8`, `isort`, `Mypy`                                                 |
| **Utilities**       | `tqdm`, `joblib`, `xlsxwriter`, `psutil`, `forex-python`, `python-dotenv`          |

## Requirements

- **Python**: Version 3.12, 3.13, or 3.14 (`>=3.12,<3.15`).
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Dependencies**: Managed via `requirements.txt`, `Pipfile`, or `pyproject.toml`.

### Python-Version-Gated Dependencies

Some packages are restricted to `python_version < '3.14'`:
`catboost`, `shap`, `streamlit`, `tensorflow`, `scikeras`, `numba`.

## Setup

### Quick Setup

```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables for the current session
. .\set_env.ps1
```

### Editable Install (Recommended for Development)

```powershell
# Install the package in editable mode
pip install -e .

# Install with optional dependency groups
pip install -e ".[dev,dashboards,database,performance,tensorflow]"
```

### Optional Dependency Groups (`pyproject.toml`)

| Group          | Packages                                                  |
|:---------------|:----------------------------------------------------------|
| `dev`          | pytest, pytest-cov, black, flake8, mypy, isort, pip-tools |
| `dashboards`   | streamlit, dash                                           |
| `database`     | psycopg2-binary, SQLAlchemy                               |
| `tensorflow`   | tensorflow, scikeras                                      |
| `performance`  | numba                                                     |

### Key Configuration Files

| File                       | Purpose                                               |
|:---------------------------|:------------------------------------------------------|
| `pyproject.toml`           | Build system, project metadata, tool configs (v0.9.5) |
| `requirements.txt`         | Full dependency list (core + optional)                |
| `Pipfile`                  | Pipenv dependency management                          |
| `environment_variables.txt`| Reference for all environment variables               |
| `set_env.ps1`              | PowerShell script to set env vars for a session       |
| `.gitignore`               | Git ignore rules                                      |

## Execution & Entry Points

### Main Pipeline

```powershell
python expected_returns_v3.py
```

Runs the complete expected-returns pipeline (8-phase ML workflow + portfolio optimization).

### CLI Entry Points

The package provides command-line entry points (defined in `pyproject.toml`):

| Command               | Target           | Description                          |
|:----------------------|:-----------------|:-------------------------------------|
| `finance-ml`          | `cli:main`       | Main analysis pipeline               |
| `finance-ml-analyze`  | `cli:analyze`    | Quick data analysis and exploration  |
| `finance-ml-validate` | `cli:validate`   | Data validation and schema check     |

### Interactive Dashboards

| Dashboard              | Run Command                                                    |
|:-----------------------|:---------------------------------------------------------------|
| **Streamlit App**      | `streamlit run finance_ml\dashboards\streamlit_app.py`         |
| **Equities Dashboard** | `python finance_ml\dashboards\equities_dashboard_app.py`       |
| **Dash App**           | `python finance_ml\dashboards\dash_app.py`                     |
| **GEIB Dashboard**     | `python dashboards\geib_dash_app.py`                           |

## Environment Variables

Set via `set_env.ps1` (dot-source to persist in session: `. .\set_env.ps1`).
Reference values are listed in `environment_variables.txt`.

| Variable                   | Default / Example                                            | Description                          |
|:---------------------------|:-------------------------------------------------------------|:-------------------------------------|
| `LOG_LEVEL`                | `INFO`                                                       | Python logging level                 |
| `DATA_DIR`                 | `data`                                                       | Local data storage                   |
| `MODEL_DIR`                | `regression`                                                 | Saved model artifacts                |
| `CACHE_DIR`                | `.cache`                                                     | Cache directory                      |
| `OUTPUT_DIR`               | `outputs`                                                    | Generated reports / visualizations   |
| `DB_URL`                   | `postgresql+psycopg2://postgres:...@localhost:5432/postgres` | SQLAlchemy DB connection URL         |
| `MODEL_VERSION`            | `v9_10`                                                      | Active model version tag             |
| `RANDOM_SEED`              | `42`                                                         | Reproducibility seed                 |
| `N_JOBS`                   | `4`                                                          | Parallel job count                   |
| `GEIB_DASHBOARD`           | `true`                                                       | Enable equities dashboard            |
| `ENABLE_INTERACTIVE_PLOTS` | `true`                                                       | Toggle interactive visualizations    |

## Testing

The project uses `pytest` as the primary testing framework (configured in `pyproject.toml`), with `unittest.TestCase` style also supported.

```powershell
# Full suite
pytest

# Verbose
pytest -v

# Specific file
pytest tests\test_screening.py

# Fast subset (avoids heavy ML training)
python tools\run_fast_tests.py
```

### Test Coverage Summary

| Test File                              | Tests     | Coverage                       |
|:---------------------------------------|:----------|:-------------------------------|
| `test_screening.py`                    | 42 tests  | Screening functions            |
| `test_data_utils.py`                   | 38 tests  | Data loading and preprocessing |
| `test_statistical_analysis.py`         | 39 tests  | Bayesian, MCMC, distributions  |
| `test_market_analytics_integration.py` | 18 tests  | Cross-module workflows         |
| `test_visualizations.py`              | 35 tests  | All 12 visualization functions |
| `test_enhanced_statistics.py`          | 40 tests  | Kalman, Copula, parallel MCMC  |
| **Total**                              | **212**   | All modules covered            |

### Adding New Tests

- Place tests in the `tests/` directory; name files `test_*.py`.
- Use small, deterministic samples; mock external services.
- Aim for ≥ 80% coverage on new code.

## Project Structure

```text
PML_Finance_Project/
├── analytics/                  # Feature analytics, screening, statistics, visualizations
│   ├── __init__.py             # Package exports & optional-dep stubs
│   ├── data_utils.py           # Data loading, preprocessing, export framework
│   ├── statistical_analysis.py # Bayesian, MCMC, Kalman, Copula
│   ├── screening.py            # 15 stock screeners
│   ├── feature_analytics.py    # Visualization dashboards
│   ├── probability_analytics.py# Probability models
│   ├── inference_schema.py     # ArviZ / xarray InferenceData bridge
│   ├── optimized_ops.py        # Performance optimizations
│   └── visualizations/         # Modular visualization sub-package
├── probabilistic_ml_model/     # Probabilistic ML models
│   ├── pml_models/             # Model implementations
│   │   ├── BaselineProbabilityModel.py
│   │   ├── ProbabilisticLinearRegressionModel.py
│   │   ├── KalmanFilterModel.py
│   │   ├── DCF_PriceTargetModel.py
│   │   ├── EarningsBeatModel.py
│   │   ├── DividendSafetyModel.py
│   │   ├── PriceTargetModel.py
│   │   ├── AccountingAnomalyModel.py
│   │   ├── CreditRiskModel.py
│   │   └── MonteCarloSimulation.py
│   ├── data_utils/
│   ├── statistical_functions/
│   └── visualizations/
├── finance_ml/                 # Core ML package (8-phase workflow)
│   ├── core/                   # Shared constants & unified schema
│   ├── etl/                    # ETL pipeline
│   ├── features/               # Feature engineering
│   ├── ml_workflow/            # ML phases (preprocessing → reporting)
│   ├── dashboards/             # Interactive dashboards
│   └── cli.py                  # CLI entry points
├── dashboards/                 # Standalone dashboards
│   └── geib_dash_app.py        # GEIB equities dashboard
├── feature_factory/            # Feature calculation utilities (Beta/CAPM, DCF, Monte Carlo)
├── sql_scripts/                # SQL setup and migration scripts
├── expected_returns_v3.py      # Automated expected-returns pipeline v3.1
├── tests/                      # Unit and integration tests
├── tools/                      # Utility scripts (setup, fast tests)
├── data/                       # Local data storage
├── outputs/                    # Reports and visualizations
├── docs/                       # Documentation
├── pyproject.toml
├── Pipfile
├── requirements.txt
├── environment_variables.txt
└── set_env.ps1
```

### Critical Imports

```python
# Analytics
from analytics.data_utils import load_feature_data_from_db, load_all_feature_views, ExportConfig
from analytics.screening import create_enhanced_screener
from analytics.statistical_analysis import bayesian_category_analysis, hierarchical_mcmc_multi_level

# Schema (finance_ml)
from finance_ml.core.schema import COLUMN_SCHEMA, normalize_column_name, list_price_cols

# ETL
from finance_ml.etl import run_etl_pipeline, ETLConfig

# Features
from finance_ml.ml_workflow.features.api import build_features
```

### Data Sources (v3.4)

| Source                              | Access Function                     |
|:------------------------------------|:------------------------------------|
| `public.mv_equities`               | `load_equities_data_from_db`        |
| `public.vw_features_*` (17 views)  | `load_all_feature_views`            |
| `public.mv_all_stock_features`     | `load_feature_data_from_db`         |
| `public.equities_schema_metadata`  | `get_equities_schema`               |
| `public.calculated_features_registry` | `load_feature_categories_from_db` |

### Design Principles

- **Unified Schema**: `finance_ml.core.schema` (`COLUMN_SCHEMA`) is the single source of truth for column definitions.
- **Optional-dependency stubs**: `analytics/__init__.py` generates stub functions for unavailable optional modules (ArviZ, probability_analytics) so imports never fail at the package level.
- **Modular ETL**: Configuration and pipeline stages are decoupled in `finance_ml/etl/`.

## 8-Phase ML Workflow

| Phase   | Description                                      | Key Module                              |
|:--------|:-------------------------------------------------|:----------------------------------------|
| **9.1** | Loading and preprocessing with 6-step imputation | `finance_ml.etl`                        |
| **9.2** | Enhanced exploratory data analysis               | `finance_ml.ml_workflow.eda`            |
| **9.3** | Advanced feature engineering                     | `finance_ml.ml_workflow.features`       |
| **9.4** | Multi-class event classification                 | `finance_ml.ml_workflow.classification` |
| **9.5** | Sector-optimized regression with quantile models | `finance_ml.ml_workflow.regression`     |
| **9.6** | Model evaluation and error analysis              | `finance_ml.ml_workflow.evaluation`     |
| **9.7** | Identification of under/overvalued stocks        | `finance_ml.ml_workflow.analytics`      |
| **9.8** | Comprehensive analytics and reporting            | `finance_ml.ml_workflow.reporting`      |

## Code Style

Enforced by the following tools (configured in `pyproject.toml`):

| Tool    | Purpose              | Key Setting                              |
|:--------|:---------------------|:-----------------------------------------|
| Black   | Code formatting      | `line-length = 100`, target py312/py313  |
| isort   | Import sorting       | `profile = "black"`, `line_length = 100` |
| Flake8  | Linting              | Standard rules                           |
| Mypy    | Static type checking | `python_version = "3.12"`                |

## Future Enhancements

### Probabilistic ML Integration (`probabilistic_ml_model/`)

- PyMC / PyTensor integration for full Bayesian inference.
- SQL schema-driven data pipeline with SQLAlchemy, xarray, ArviZ.
- New DCF price-target regression model (`DCF_PriceTargetModel.py`).
- Feature importance analysis.

### Real-time Dashboard / Data Integration

- Live market data feeds.
- Streaming analytics.
- Alert system for screening triggers.

## License

<!-- TODO: Add a LICENSE file to the project root. -->
This project is intended to be licensed under the MIT License.
