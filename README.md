# PML Finance Project

A comprehensive platform for probabilistic equity screening, feature engineering, and machine learning modeling across global financial markets.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![Package Version](https://img.shields.io/badge/version-0.9.8.5-green)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The PML Finance Project is a robust solution for financial data analysis, probabilistic machine learning modeling, and
portfolio optimization. It implements a multi-modal **Probabilistic ML Workflow** (Phases 1-8) for data quality,
advanced
feature engineering, and reliable model evaluation, followed by a **Portfolio Optimization** module.

### Key Features

- **Probabilistic ML Models**: Monte Carlo simulation, Kalman filtering, DCF price-target regression, Bayesian earnings-beat analysis, credit risk estimation, dividend safety scoring, and accounting anomaly detection.
- **Analytics Module**: 15 stock screeners, Bayesian/MCMC statistical analysis, Kalman/Copula methods, and interactive
  Plotly visualizations.
- **SQL Database Integration**: Centralized data storage using PostgreSQL with schema-driven feature catalogs.
- **7-Phase Portfolio Optimization**: Stock selection, return prediction, risk-adjusted optimization (Efficient Frontier), backtesting, and interactive dashboards.
- **Interactive Dashboards**: Integrated Streamlit, Dash, and Plotly applications for market monitoring, earnings analytics, and portfolio visualization.

### PyMC Models Notebook (`pymc_expected_returns_model.ipynb`)

End-to-end PyMC + ArviZ workflow that fits the seven Bayesian models from
`probabilistic_ml_model/pymc_models/` (EarningsBeat, PriceTarget, Kalman,
DCF, DividendSafety, CreditRisk, AccountingAnomaly). The notebook shares a
single `MODEL_FEATURE_CONTAINERS` registry and a model-aware
`attach_features(idata, features_df, model_name)` helper that appends a
catalogue-aligned `(isin × feature)` matrix to `idata.constant_data`.
Both the `pm.Data` dim name (`pt_feature`, `dcf_feature`,
`earnings_feature`, …) and the canonical feature_alias list are derived
from the same registry source of truth used by each model's
`_resolve_*_feature_aliases()` helper, so the notebook helper cannot
drift out of sync with the per-model `with pm.Model(coords=...)` block.

Each model section ends with a runnable *Out-of-sample prediction via
`pm.set_data`* code cell (§5, §6.6, §7.6, §8.6, §9.6, §10.6, §11.6). The
cells build a 50-row holdout slice from the model's already-prepared
inputs, re-align the auxiliary feature matrix via the corresponding
`_align_*_features` helper, call `pm.set_data({...}, coords={'isin': ...})`
inside the fitted `pm.Model` context, and visualise the resulting
posterior predictive distribution (histogram, scatter, or fan chart
depending on the model's likelihood).

A new **§13 "Implementation — Actionable Recommendations from §12.3"**
turns each `feature_catalogue`-aligned recommendation into a runnable
code snippet (type-aware coercion, calculation-type-driven prior σ
table, `source_function` provenance attrs, strict `attach_features`
mode, catalogue-driven coverage check, per-`category` hyperprior PyMC
sketch for `AccountingAnomalyBayesian`, OOS shape contract). The
notebook helpers are backed by a new shared module
`probabilistic_ml_model/pymc_models/_feature_alignment.py`
(`coerce_by_data_type`, `stamp_feature_provenance`,
`assert_disjoint_features`, `validate_oos_shape`,
`load_feature_metadata_from_db`) used by all seven model
`_align_*_features(use_typed_coercion=True)` paths and stamped onto
`idata.constant_data[...].attrs` after every `*Model.fit(...)` call.

### Core Models (`probabilistic_ml_model/pml_models/`)

Managed via a unified `PipelineRunner` in `probabilistic_ml_model/pipeline_runners.py`:

- **Monte Carlo Simulation** — probabilistic upside/downside distributions.
- **Price Target Achievement** — probability-weighted expected returns.
- **Kalman Filtered Targets** — noise-reduced price target signals.
- **Earnings Beat Analysis** — three-layer Bayesian beat probability.
- **Credit Risk Analysis** — Bayesian distress estimation.
- **Dividend Safety Analysis** — dividend cut probability with FCF coverage.
- **Accounting Anomaly Detection** — multi-layered statistical anomaly detection.
- **DCF Price Target Model** — discounted cash flow regression.
- **Probabilistic Linear Regression** — Bayesian linear regression model.

## Tech Stack

| Category            | Technologies                                                                           |
|:--------------------|:---------------------------------------------------------------------------------------|
| **Language**        | Python 3.12 / 3.13 / 3.14                                                              |
| **Package Manager** | `pip`, `pipenv` (`Pipfile`), `setuptools` (`pyproject.toml`)                           |
| **ML Frameworks**   | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `SHAP`, `TensorFlow`      |
| **Bayesian**        | `PyMC`, `PyTensor`, `ArviZ` 1.0 (`arviz-base`, `arviz-stats`, `arviz-plots`), `xarray` |
| **Data Processing** | `pandas`, `NumPy`, `SciPy`, `statsmodels`, `numba`, `imbalanced-learn`                 |
| **Visualization**   | `Plotly`, `Matplotlib`, `Seaborn`                                                      |
| **Dashboards**      | `Streamlit`, `Dash`, `dash-bootstrap-components`                                       |
| **Database**        | `PostgreSQL` (`psycopg2`), `SQLAlchemy`, `SQLite`                                      |
| **Testing**         | `pytest`, `unittest`                                                                   |
| **Code Quality**    | `Black`, `Flake8`, `isort`, `Mypy`                                                     |
| **Utilities**       | `tqdm`, `joblib`, `xlsxwriter`, `psutil`, `forex-python`, `python-dotenv`              |

## Requirements

- **Python**: Version 3.12, 3.13, or 3.14 (`>=3.12,<3.15`).
- **Operating System**: Windows (primary support via PowerShell), Linux, or macOS.
- **Database**: PostgreSQL (for full feature set).

### Python-Version-Gated Dependencies

Some packages are restricted to `python_version < '3.14'`:
`catboost`, `shap`, `streamlit`, `tensorflow`, `scikeras`, `numba`.

## Setup

### Quick Setup (Windows PowerShell)

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
pip install -e ".[dev,dashboards,database,performance,tensorflow,notebooks,extras]"
```

### Optional Dependency Groups (`pyproject.toml`)

| Group         | Packages                                                        |
|:--------------|:----------------------------------------------------------------|
| `dev`         | pytest, pytest-cov, black, flake8, mypy, isort, pip-tools       |
| `dashboards`  | streamlit, dash, dash-bootstrap-components, flask               |
| `database`    | psycopg2-binary, SQLAlchemy                                     |
| `tensorflow`  | tensorflow, scikeras                                            |
| `performance` | numba                                                           |
| `notebooks`   | jupyter, notebook, ipykernel, ipython                           |
| `extras`      | openpyxl, boruta, networkx, Pillow, forex-python, python-dotenv |

### Key Configuration Files

| File                        | Purpose                                               |
|:----------------------------|:------------------------------------------------------|
| `pyproject.toml`            | Build system, project metadata, tool configs (v0.9.5) |
| `CHANGELOG.md`              | Release notes (Keep a Changelog / SemVer)             |
| `requirements.txt`          | Full dependency list (core + optional)                |
| `Pipfile`                   | Pipenv dependency management                          |
| `environment_variables.txt` | Reference for all environment variables               |
| `set_env.ps1`               | PowerShell script to set env vars for a session       |
| `.gitignore`                | Git ignore rules                                      |

## Execution & Entry Points

### Main Pipelines

```powershell
# v3 pipeline (8-phase ML workflow + portfolio optimization)
python expected_returns_v3.py

# v4 pipeline
python expected_returns_v4.py
```

### CLI Entry Points

The package provides command-line entry points (defined in `pyproject.toml`):

| Command               | Target                     | Description                         |
|:----------------------|:---------------------------|:------------------------------------|
| `finance-ml`          | `cli:main`                 | Main analysis pipeline              |
| `finance-ml-analyze`  | `cli:analyze`              | Quick data analysis and exploration |
| `finance-ml-validate` | `cli:validate`             | Data validation and schema check    |
| `finance-ml-v4`       | `expected_returns_v4:main` | v4 expected-returns pipeline        |

> **Note**: The `cli.py` module for `finance-ml`, `finance-ml-analyze`, and `finance-ml-validate` is currently under
> development.

### Interactive Dashboards

| Dashboard          | Run Command                          | Status |
|:-------------------|:-------------------------------------|:-------|
| **GEIB Dashboard** | `python dashboards\geib_dash_app.py` | ✅ Live |

<!-- TODO: Add Streamlit / Equities / Dash apps under finance_ml/dashboards/ (not yet implemented). -->

## Scripts & Utilities

| Script / File            | Description                                        |
|:-------------------------|:---------------------------------------------------|
| `expected_returns_v3.py` | Automated expected-returns pipeline v3.1           |
| `expected_returns_v4.py` | Next-generation expected-returns pipeline          |
| `set_env.ps1`            | Set environment variables for a PowerShell session |

### Feature Factory (`feature_factory/`)

Standalone feature calculation utilities (Beta/CAPM, DCF, Monte Carlo, Efficient Frontier, etc.).

### SQL Scripts (`sql_scripts/`)

Database setup and migration scripts for PostgreSQL (schema creation, materialized views, feature registry, data
import).

## Environment Variables

Set via `set_env.ps1` (dot-source to persist in session: `. .\set_env.ps1`). Reference values are listed in
`environment_variables.txt`.

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

The project uses `pytest` as the primary testing framework (configured in `pyproject.toml`), with `unittest.TestCase` style also supported.

```powershell
# Full suite
pytest

# Verbose
pytest -v

# Specific file
pytest tests\test_pml_workflow_v4.py

# With coverage (requires the [dev] extra)
pytest --cov=probabilistic_ml_model --cov-report=term-missing
```

### Test Coverage Summary

| Test File                                 | Tests   | Coverage                                     |
|:------------------------------------------|:--------|:---------------------------------------------|
| `test_pml_workflow_v4.py`                 | 55      | v4 pipeline functions, models, screening     |
| `test_arviz_refactoring.py`               | 54      | ArviZ 1.0 migration, new viz types, registry |
| `test_viz_catalog_integration.py`         | 46      | Visualization–catalog integration            |
| `test_safe_get_output_values.py`          | 40      | Safe output value extraction                 |
| `test_cache_optimization.py`              | 39      | Cache optimization & invalidation            |
| `test_downstream_risk_adj_integration.py` | 30      | Downstream risk-adjusted integration         |
| `test_data_loading_refactoring.py`        | 27      | Data loading & preprocessing refactoring     |
| `test_catalog_consolidation.py`           | 26      | Catalog consolidation & consistency          |
| `test_feature_catalog.py`                 | 26      | FeatureViewCatalog registry & resolution     |
| `test_v35_earnings_beat.py`               | 22      | v3.5 earnings beat model tests               |
| `test_ensemble_risk_adj_return.py`        | 21      | Ensemble risk-adjusted return scoring        |
| `test_arviz_migration.py`                 | 18      | ArviZ 1.0 API migration                      |
| `test_arviz_improvements.py`              | 17      | ArviZ diagnostic improvements                |
| `test_pipeline_statistical_fixes.py`      | 17      | Pipeline statistical fix validations         |
| `test_hierarchical_mcmc_refactor.py`      | 11      | Hierarchical MCMC refactoring                |
| `test_new_columns.py`                     | 9       | New column definitions & schema              |
| `test_idata_shim_smoke.py`                | 9       | InferenceData shim smoke tests               |
| `test_v35_anomaly_enhancements.py`        | 7       | v3.5 accounting anomaly enhancements         |
| `test_distribution_fitting.py`            | 6       | Distribution fitting models                  |
| `test_plr_idata_kwargs.py`                | 3       | ProbabilisticLinearRegression idata args     |
| **Total**                                 | **483** | All modules covered                          |

### Adding New Tests

- Place tests in the `tests/` directory; name files `test_*.py`.
- Use small, deterministic samples; mock external services.
- Aim for ≥ 80% coverage on new code.

## Project Structure

```text
PML_Finance_Project/
├── probabilistic_ml_model/         # Probabilistic ML models & pipeline
│   ├── __init__.py                 # Package init & optional-dep stubs
│   ├── config.py                   # Pipeline configuration
│   ├── pipeline_runners.py         # Orchestration for model execution
│   ├── utils.py                    # Shared utilities
│   ├── logging_config.py           # Logging setup
│   ├── optimized_ops.py            # Performance optimizations
│   ├── _pymc_arviz_compat.py       # PyMC / ArviZ compatibility shim
│   ├── pml_models/                 # Model implementations
│   │   ├── BaselineProbabilityModel.py
│   │   ├── ProbabilisticLinearRegressionModel.py
│   │   ├── KalmanFilterModel.py
│   │   ├── DCF_PriceTargetModel.py
│   │   ├── EarningsBeatModel.py
│   │   ├── DividendSafetyModel.py
│   │   ├── PriceTargetModel.py
│   │   ├── AccountingAnomalyModel.py
│   │   ├── CreditRiskModel.py
│   │   ├── MonteCarloSimulation.py
│   │   └── _pytensor_compat.py     # PyTensor compatibility shim
│   ├── data_utils/                 # Data loading & inference schema
│   │   ├── data_utils.py           # DB loading, preprocessing, export
│   │   ├── feature_catalog.py      # FeatureViewCatalog registry
│   │   └── inference_schema.py     # ArviZ / xarray InferenceData bridge
│   ├── statistical_functions/      # Screening, statistics, ensemble, probability analytics
│   │   ├── screening.py            # Stock screeners
│   │   ├── statistical_models.py   # Bayesian & MCMC models
│   │   ├── ensemble_models.py      # Ensemble alignment helpers
│   │   └── probability_models.py   # Distribution fitting & probability analytics
│   └── visualizations/             # Model-specific visualizations (8 modules + _shared, ArviZ 1.0)
│       ├── expected_returns_viz.py
│       ├── earnings_quality.py
│       ├── arviz_diagnostics.py
│       ├── convergence_diagnostics.py
│       ├── growth_analysis.py
│       ├── probability_viz.py
│       ├── quality_risk.py
│       ├── valuation.py
│       └── _shared.py              # Shared visualization utilities
├── finance_ml/                     # Core ML package (workflow modules)
│   └── ml_workflow/                # ML workflow phases
│       └── v3/                     # v3 workflow utilities (cache, config, enrichment, I/O)
├── dashboards/                     # Standalone dashboards
│   └── geib_dash_app.py            # GEIB equities dashboard (live)
├── feature_factory/                # Feature calculation utilities (Beta/CAPM, DCF, Monte Carlo)
├── sql_scripts/                    # SQL setup and migration scripts
│   ├── analytics/                  # Analytics schema SQL
│   ├── information_schema/         # Information schema queries
│   └── public/                     # Public schema SQL
├── expected_returns_v3.py          # Automated expected-returns pipeline v3.1
├── expected_returns_v4.py          # Next-generation expected-returns pipeline
├── tests/                          # Unit and integration tests (483 tests)
├── data/                           # Local data storage
├── outputs/                        # Reports and visualizations
├── logs/                           # Pipeline execution logs
├── archive/                        # Archived files and prior outputs
├── docs/                           # Documentation
├── pyproject.toml
├── Pipfile
├── requirements.txt
├── environment_variables.txt
└── set_env.ps1
```

## Probabilistic Machine Learning (PML) Workflow (v4.0)

### Phase 1 — Data Ingestion & Enrichment

1. Load equities data from `mv_equities` via `probabilistic_ml_model.data_utils.load_equities_data_from_db`
2. Load feature views from `vw_features_*` (17 views) via `probabilistic_ml_model.data_utils.load_all_feature_views`
3. Load full feature superset from `mv_all_stock_features` via
   `probabilistic_ml_model.data_utils.load_feature_data_from_db`
4. Load schema metadata (`equities_schema_metadata`) and feature registry (`calculated_features_registry`)
5. Apply column backfill and Kalman momentum smoothing
6. Pre-compute historical target drift enrichment (consensus drift, spread evolution, price anchor)

### Phase 2 — Core PML Model Execution (`pml_models/`)

7. Run `MonteCarloSimulation` — triangular distribution sampling with historical target drift priors
8. Run `PriceTargetModel` — probability-weighted expected returns with analyst sentiment & risk adjustment
9. Run `KalmanFilterModel` — noise-reduced price target signals with momentum-informed priors
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
31. Generate ArviZ 1.0 diagnostic plots (trace, forest, ridge, convergence panels, ECDF, dot plots, PPC rootogram)
32. Generate feature view posterior panels and cross-category summaries
33. Generate unified convergence dashboard across all pipeline MCMC outputs
34. Generate hierarchical dot comparison and cross-model ECDF with reference quantiles

### Phase 8 — Export & Reporting

35. Export all model results to analytics schema (parallelized, deduplicated, hash-gated)
36. Export probability analytics and screening results
37. Generate pipeline summary report with timing and convergence diagnostics

## Code Style

Enforced by the following tools (configured in `pyproject.toml`):

| Tool   | Purpose              | Key Setting                              |
|:-------|:---------------------|:-----------------------------------------|
| Black  | Code formatting      | `line-length = 100`, target py314        |
| isort  | Import sorting       | `profile = "black"`, `line_length = 100` |
| Flake8 | Linting              | Standard rules                           |
| Mypy   | Static type checking | `python_version = "3.14"`                |

## Contributing

- Format with `black .` and `isort .`; lint with `flake8`; type-check with `mypy`.
- Run the test suite (`pytest`) before submitting changes.
- See `CHANGELOG.md` for release history (Keep a Changelog / SemVer).

## License

This project is licensed under the **MIT License** (declared in `pyproject.toml`).

<!-- TODO: Add a top-level LICENSE file with the full MIT license text. -->
<!-- TODO: Implement cli.py module for CLI entry points (finance-ml, finance-ml-analyze, finance-ml-validate). -->
