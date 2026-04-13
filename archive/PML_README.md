# PML Finance Project

A comprehensive platform for probabilistic equity screening, feature engineering, and machine learning modeling across
global financial markets.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![Package Version](https://img.shields.io/badge/version-0.9.5-green)](pyproject.toml)

## Overview

The PML Finance Project is a robust solution for financial data analysis, probabilistic machine learning modeling, and
portfolio optimization. It implements a structured **8-Phase ML Workflow** (Phases 9.1–9.8) for data quality, advanced
feature engineering, and reliable model evaluation, followed by a **7-Phase Portfolio Optimization** module.

### Key Features

- **Probabilistic ML Models**: Monte Carlo simulation, Kalman filtering, DCF price-target regression, Bayesian
  earnings-beat analysis, credit risk estimation, dividend safety scoring, and accounting anomaly detection.
- **Analytics Module**: 15 stock screeners, Bayesian/MCMC statistical analysis, Kalman/Copula methods, and interactive
  Plotly visualizations.
- ** SQL Database**: Centralized data storage and management.
- **7-Phase Portfolio Optimization**: Stock selection, return prediction, risk-adjusted optimization (Efficient
  Frontier), backtesting, and interactive dashboards.
- **Interactive Dashboards**: Integrated Streamlit, Dash, and Plotly applications for market monitoring, earnings
  analytics, and portfolio visualization.

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

| Category            | Technologies                                                                      |
|:--------------------|:----------------------------------------------------------------------------------|
| **Language**        | Python 3.12 / 3.13 / 3.14                                                         |
| **Package Manager** | `pip`, `pipenv` (`Pipfile`), `setuptools` (`pyproject.toml`)                      |
| **ML Frameworks**   | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `SHAP`, `TensorFlow` |
| **Bayesian**        | `PyMC`, `PyTensor`, `ArviZ`, `xarray`                                             |
| **Data Processing** | `pandas`, `NumPy`, `SciPy`, `statsmodels`, `numba`, `imbalanced-learn`            |
| **Visualization**   | `Plotly`, `Matplotlib`, `Seaborn`                                                 |
| **Dashboards**      | `Streamlit`, `Dash`, `dash-bootstrap-components`                                  |
| **Database**        | `PostgreSQL` (`psycopg2`), `SQLAlchemy`, `SQLite`                                 |
| **Testing**         | `pytest`, `unittest`                                                              |
| **Code Quality**    | `Black`, `Flake8`, `isort`, `Mypy`                                                |
| **Utilities**       | `tqdm`, `joblib`, `xlsxwriter`, `psutil`, `forex-python`, `python-dotenv`         |

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

| Group         | Packages                                                  |
|:--------------|:----------------------------------------------------------|
| `dev`         | pytest, pytest-cov, black, flake8, mypy, isort, pip-tools |
| `dashboards`  | streamlit, dash                                           |
| `database`    | psycopg2-binary, SQLAlchemy                               |
| `tensorflow`  | tensorflow, scikeras                                      |
| `performance` | numba                                                     |

### Key Configuration Files

| File                        | Purpose                                               |
|:----------------------------|:------------------------------------------------------|
| `pyproject.toml`            | Build system, project metadata, tool configs (v0.9.5) |
| `requirements.txt`          | Full dependency list (core + optional)                |
| `Pipfile`                   | Pipenv dependency management                          |
| `environment_variables.txt` | Reference for all environment variables               |
| `set_env.ps1`               | PowerShell script to set env vars for a session       |
| `.gitignore`                | Git ignore rules                                      |

## Execution & Entry Points

### Main Pipeline

```powershell
python expected_returns_v3.py
```

Runs the complete expected-returns pipeline (8-phase ML workflow + portfolio optimization).

### CLI Entry Points

The package provides command-line entry points (defined in `pyproject.toml`):

| Command               | Target         | Description                         |
|:----------------------|:---------------|:------------------------------------|
| `finance-ml`          | `cli:main`     | Main analysis pipeline              |
| `finance-ml-analyze`  | `cli:analyze`  | Quick data analysis and exploration |
| `finance-ml-validate` | `cli:validate` | Data validation and schema check    |

### Interactive Dashboards

| Dashboard              | Run Command                                              |
|:-----------------------|:---------------------------------------------------------|
| **Streamlit App**      | `streamlit run finance_ml\dashboards\streamlit_app.py`   |
| **Equities Dashboard** | `python finance_ml\dashboards\equities_dashboard_app.py` |
| **Dash App**           | `python finance_ml\dashboards\dash_app.py`               |
| **GEIB Dashboard**     | `python dashboards\geib_dash_app.py`                     |

## Environment Variables

Set via `set_env.ps1` (dot-source to persist in session: `. .\set_env.ps1`).
Reference values are listed in `environment_variables.txt`.

| Variable                   | Default / Example                                            | Description                        |
|:---------------------------|:-------------------------------------------------------------|:-----------------------------------|
| `LOG_LEVEL`                | `INFO`                                                       | Python logging level               |
| `DATA_DIR`                 | `data`                                                       | Local data storage                 |
| `MODEL_DIR`                | `regression`                                                 | Saved model artifacts              |
| `CACHE_DIR`                | `.cache`                                                     | Cache directory                    |
| `OUTPUT_DIR`               | `outputs`                                                    | Generated reports / visualizations |
| `DB_URL`                   | `postgresql+psycopg2://postgres:...@localhost:5432/postgres` | SQLAlchemy DB connection URL       |
| `MODEL_VERSION`            | `v9_10`                                                      | Active model version tag           |
| `RANDOM_SEED`              | `42`                                                         | Reproducibility seed               |
| `N_JOBS`                   | `4`                                                          | Parallel job count                 |
| `GEIB_DASHBOARD`           | `true`                                                       | Enable equities dashboard          |
| `ENABLE_INTERACTIVE_PLOTS` | `true`                                                       | Toggle interactive visualizations  |

## Testing

The project uses `pytest` as the primary testing framework (configured in `pyproject.toml`), with `unittest.TestCase`
style also supported.

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

| Test File                              | Tests    | Coverage                       |
|:---------------------------------------|:---------|:-------------------------------|
| `test_screening.py`                    | 42 tests | Screening functions            |
| `test_data_utils.py`                   | 38 tests | Data loading and preprocessing |
| `test_statistical_analysis.py`         | 39 tests | Bayesian, MCMC, distributions  |
| `test_market_analytics_integration.py` | 18 tests | Cross-module workflows         |
| `test_visualizations.py`               | 35 tests | All 12 visualization functions |
| `test_enhanced_statistics.py`          | 40 tests | Kalman, Copula, parallel MCMC  |
| **Total**                              | **212**  | All modules covered            |

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
from probabilistic_ml_model.data_utils import load_feature_data_from_db, load_all_feature_views, ExportConfig
from probabilistic_ml_model.statistical_functions.screening import create_enhanced_screener
from probabilistic_ml_model.statistical_functions.statistical_analysis import bayesian_category_analysis, hierarchical_mcmc_multi_level


```

### Data Sources (v3.4)

| Source                                | Access Function                   |
|:--------------------------------------|:----------------------------------|
| `public.mv_equities`                  | `load_equities_data_from_db`      |
| `public.vw_features_*` (17 views)     | `load_all_feature_views`          |
| `public.mv_all_stock_features`        | `load_feature_data_from_db`       |
| `public.equities_schema_metadata`     | `get_equities_schema`             |
| `public.calculated_features_registry` | `load_feature_categories_from_db` |

### Design Principles

- **Unified Schema**: `core.schema` (`COLUMN_SCHEMA`) is the single source of truth for column definitions.
- **Optional-dependency stubs**: `probabilistic_ml_model/__init__.py` generates stub functions for unavailable optional modules (
  ArviZ, probability_analytics) so imports never fail at the package level.
- **Modular ETL**: Configuration and pipeline stages are decoupled in `finance_ml/etl/`.

## Probabilistic Machine Learning (PML) Workflow (v4.0)

### Phase 1 — Data Ingestion & Enrichment

    1.  Load equities data from `mv_equities` via `probabilistic_ml_model.data_utils.load_equities_data_from_db`
    2.  Load feature views from `vw_features_*` (17 views) via `probabilistic_ml_model.data_utils.load_all_feature_views`
    3.  Load full feature superset from `mv_all_stock_features` via `probabilistic_ml_model.data_utils.load_feature_data_from_db`
    4.  Load schema metadata (`equities_schema_metadata`) and feature registry (`calculated_features_registry`)
    5.  Apply column backfill and Kalman momentum smoothing
    6.  Pre-compute historical target drift enrichment (consensus drift, spread evolution, price anchor)

### Phase 2 — Core PML Model Execution (`pml_models/`)

    7.  Run `MonteCarloSimulation` — triangular distribution sampling with historical target drift priors
    8.  Run `PriceTargetModel` — probability-weighted expected returns with analyst sentiment & risk adjustment
    9.  Run `KalmanFilterModel` — noise-reduced price target signals with momentum-informed priors
    10. Run `EarningsBeatModel` — three-layer Bayesian beat probability (+ EPS streak & resampled priors)
    11. Run `AccountingAnomalyModel` — multi-layered statistical anomaly detection with Mahalanobis distance
    12. Run `CreditRiskModel` — Bayesian distress estimation with debt trajectory & balance sheet strength
    13. Run `DividendSafetyModel` — dividend cut probability with FCF coverage & leverage signals

### Phase 3 — Probabilistic Linear Market Model (`pml_models/`)

    14. Run `ProbabilisticLinearRegressionModel` — Bayesian linear regression with posterior inference
    15. Run `DCF_PriceTargetModel` — discounted cash flow regression with probabilistic fair value bands

### Phase 4 — Statistical Functions & Screening (`statistical_functions/`)

    16. Run stock screening strategies (15 screeners + productivity frontier & reporting lag enrichment)
    17. Compute resampled Bayesian posterior returns from historical price snapshots
    18. Run per-category Bayesian probability analytics (distribution fitting, conditional probabilities)
    19. Run parallel MCMC return analysis with Gelman-Rubin convergence diagnostics

### Phase 5 — Ensemble Alignment & Summary

    20. Build tri-model alignment (MC × Kalman × Price Target) with direction agreement scores
    21. Build quad-model alignment (+ Earnings Beat signals)
    22. Build `expected_returns_summary` with cross-model diagnostics & hierarchical sector MCMC
    23. Compute multi-level hierarchical MCMC posteriors (region, country, sector, industry, style, size)
    24. Merge ProbabilisticLinearRegression & DCF posteriors into ensemble summary

### Phase 6 — Posterior Inference & InferenceData (`data_utils/inference_schema.py`)

    25. Build per-model `InferenceData` (ArviZ) with `EquityCoordinates` dimensions
    26. Build per-feature-view `InferenceData` with ArviZ diagnostics
    27. Compute MCMC convergence diagnostics (R-hat, ESS, MCSE) across all model posteriors
    28. Assemble unified posterior: P(a, b, e | X, Y) from prior × likelihood / marginal

### Phase 7 — Visualization (`visualizations/`)

    29. Generate expected returns visualizations (MC distribution, Kalman, tri-model, sector heatmaps)
    30. Generate earnings quality & accounting anomaly visualizations
    31. Generate ArviZ diagnostic plots (trace, forest, ridge, convergence panels)
    32. Generate feature view posterior panels and cross-category summaries

### Phase 8 — Export & Reporting

    33. Export all model results to analytics schema (parallelized, deduplicated, hash-gated)
    34. Export probability analytics and screening results
    35. Generate pipeline summary report with timing and convergence diagnostics

## Code Style

Enforced by the following tools (configured in `pyproject.toml`):

| Tool   | Purpose              | Key Setting                              |
|:-------|:---------------------|:-----------------------------------------|
| Black  | Code formatting      | `line-length = 100`, target py312/py313  |
| isort  | Import sorting       | `profile = "black"`, `line_length = 100` |
| Flake8 | Linting              | Standard rules                           |
| Mypy   | Static type checking | `python_version = "3.12"`                |

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
