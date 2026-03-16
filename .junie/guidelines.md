# PML Finance Project — Development Guidelines

## 1. Build / Configuration Instructions

### Prerequisites

- **Python**: 3.12, 3.13, or 3.14 (`requires-python = ">=3.12,<3.15"`).
- **OS**: Windows (primary, PowerShell), Linux, macOS.
- **Dependencies**: Managed via `requirements.txt`, `Pipfile` (pipenv), and `pyproject.toml` (setuptools).

### Quick Setup

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables for the current session
. .\set_env.ps1
```

### Key Configuration Files

| File                      | Purpose                                              |
|:--------------------------|:-----------------------------------------------------|
| `pyproject.toml`          | Build system, project metadata, tool configs (v0.9.5)|
| `requirements.txt`        | Full dependency list (core + optional)               |
| `Pipfile`                 | Pipenv dependency management                         |
| `environment_variables.txt`| Reference for all environment variables              |
| `set_env.ps1`             | PowerShell script to set env vars for a session      |
| `.gitignore`              | Git ignore rules                                     |

### Environment Variables

Set via `set_env.ps1` (dot-source to persist in session: `. .\set_env.ps1`):

| Variable              | Default / Example                                                  | Description                          |
|:----------------------|:-------------------------------------------------------------------|:-------------------------------------|
| `LOG_LEVEL`           | `INFO`                                                             | Python logging level                 |
| `DATA_DIR`            | `data`                                                             | Local data storage                   |
| `MODEL_DIR`           | `regression`                                                       | Saved model artifacts                |
| `CACHE_DIR`           | `.cache`                                                           | Cache directory                      |
| `OUTPUT_DIR`          | `outputs`                                                          | Generated reports / visualizations   |
| `DB_URL`              | `postgresql+psycopg2://postgres:...@localhost:5432/postgres`       | SQLAlchemy DB connection URL         |
| `MODEL_VERSION`       | `v9_10`                                                            | Active model version tag             |
| `RANDOM_SEED`         | `42`                                                               | Reproducibility seed                 |
| `N_JOBS`              | `4`                                                                | Parallel job count                   |
| `GEIB_DASHBOARD`      | `true`                                                             | Enable equities dashboard            |
| `ENABLE_INTERACTIVE_PLOTS` | `true`                                                        | Toggle interactive visualizations    |

### Python-Version–Gated Dependencies

Some packages are restricted to `python_version < '3.14'`:
`catboost`, `shap`, `streamlit`, `tensorflow`, `scikeras`, `numba`.

### Optional Dependency Groups (pyproject.toml)

| Group          | Packages                                                 |
|:---------------|:---------------------------------------------------------|
| `dev`          | pytest, pytest-cov, black, flake8, mypy, isort, pip-tools|
| `dashboards`   | streamlit, dash                                          |
| `database`     | psycopg2-binary, SQLAlchemy                              |
| `tensorflow`   | tensorflow, scikeras                                     |
| `performance`  | numba                                                    |

Install an optional group: `pip install -e ".[dev,database]"`

---

## 2. Testing

### Framework

- **Primary**: `pytest` (configured in `pyproject.toml` under `[tool.pytest.ini_options]`).
- **Secondary**: `unittest.TestCase` style also supported.
- Test discovery paths: `tests/`, files matching `test_*.py`, classes `Test*`, functions `test_*`.

### Running Tests

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

### Unit Test Example

```python
# tests/test_screening.py
import pandas as pd
from analytics.screening import create_enhanced_screener


def test_enhanced_screener():
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'piotroski_f_score': [8, 6],
        'distress_risk_score': [85, 70],
        'eps_trajectory_score': [75, 60],
        'fcf_positive_years': [5, 3],
    })

    result = create_enhanced_screener(df, min_fscore=7)
    assert len(result) == 1
    assert result.iloc[0]['ticker'] == 'AAPL'
```

### Integration Test Example

```python
# tests/test_integration.py
from analytics.data_utils import load_feature_data_from_db
from analytics.screening import create_enhanced_screener


def test_full_workflow():
    df = load_feature_data_from_db(limit=100)
    quality_stocks = create_enhanced_screener(df, min_fscore=5)
    assert len(quality_stocks) > 0
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

Run all analytics tests:

```powershell
pytest tests\test_screening.py tests\test_data_utils.py tests\test_statistical_analysis.py tests\test_market_analytics_integration.py tests\test_visualizations.py tests\test_enhanced_statistics.py -v
```

### Guidelines for New Tests

- Place tests in the `tests/` directory; name files `test_*.py`.
- Use small, deterministic samples; mock external services.
- Aim for ≥ 80 % coverage on new code.

---

## 3. Code Style

Enforced by the following tools (configured in `pyproject.toml`):

| Tool    | Purpose              | Key Setting                     |
|:--------|:---------------------|:--------------------------------|
| Black   | Code formatting      | `line-length = 100`, target py312/py313 |
| isort   | Import sorting       | `profile = "black"`, `line_length = 100` |
| Flake8  | Linting              | Standard rules                  |
| Mypy    | Static type checking | `python_version = "3.12"`       |

---

## 4. Entry Points & Execution

### CLI Commands (pyproject.toml `[project.scripts]`)

| Command               | Target           | Description                          |
|:----------------------|:-----------------|:-------------------------------------|
| `finance-ml`          | `cli:main`       | Main analysis pipeline               |
| `finance-ml-analyze`  | `cli:analyze`    | Quick data analysis                  |
| `finance-ml-validate` | `cli:validate`   | Data validation / schema check       |

### Main Pipeline

```powershell
python expected_returns_v3.py
```

Runs the complete expected-returns pipeline (8-phase ML workflow + portfolio optimization).

### Dashboards

| Dashboard              | Command                                                  |
|:-----------------------|:---------------------------------------------------------|
| Streamlit App          | `streamlit run finance_ml\dashboards\streamlit_app.py`   |
| Equities Dashboard     | `python finance_ml\dashboards\equities_dashboard_app.py` |
| Dash App               | `python finance_ml\dashboards\dash_app.py`               |

---

## 5. Architecture Overview

### Project Structure

```
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
├── probabilistic_ml_model/     # New PML models (in development)
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
├── expected_returns_v3.py      # Automated expected-returns pipeline v3.1
├── tests/                      # Unit and integration tests
├── tools/                      # Utility scripts (setup, fast tests)
├── data/                       # Local data storage
├── models/                     # Saved model artifacts
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
from core.schema import COLUMN_SCHEMA, normalize_column_name, list_price_cols

# ETL
from etl import run_etl_pipeline, ETLConfig

# Features
from ml_workflow.features.api import build_features
```

### Data Sources (v3.4)

| Source                             | Access Function                    |
|:-----------------------------------|:-----------------------------------|
| `public.mv_equities`              | `load_equities_data_from_db`       |
| `public.vw_features_*` (17 views) | `load_all_feature_views`           |
| `public.mv_all_stock_features`    | `load_feature_data_from_db`        |
| `public.equities_schema_metadata` | `get_equities_schema`              |
| `public.calculated_features_registry` | `load_feature_categories_from_db` |

### Core Models

- **Monte Carlo Simulation** — probabilistic upside/downside distributions
- **Price Target Achievement** — probability-weighted expected returns
- **Kalman Filtered Targets** — noise-reduced price target signals
- **Earnings Beat Analysis** — three-layer Bayesian beat probability
- **Credit Risk Analysis** — Bayesian distress estimation
- **Dividend Safety Analysis** — dividend cut probability with FCF coverage
- **Accounting Anomaly Detection** — multi-layered statistical anomaly detection

### Design Principles

- **Unified Schema**: `core.schema` (`COLUMN_SCHEMA`) is the single source of truth for column definitions.
- **Optional-dependency stubs**: `analytics/__init__.py` generates stub functions for unavailable optional modules (ArviZ, probability_analytics) so imports never fail at the package level.
- **Modular ETL**: Configuration and pipeline stages are decoupled in `finance_ml/etl/`.

---

## 6. Future Enhancements

### Probabilistic ML Integration (probabilistic_ml_model/)

- PyMC / PyTensor integration for full Bayesian inference.
- SQL schema-driven data pipeline with SQLAlchemy, xarray, ArviZ.
- New DCF price-target regression model (`DCF_PriceTargetModel.py`).
- Feature importance analysis.

### Real-time Dashboard / Data Integration

- Live market data feeds.
- Streaming analytics.
- Alert system for screening triggers.
