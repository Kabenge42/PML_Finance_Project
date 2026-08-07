# PML Finance Project

A comprehensive platform for probabilistic equity screening, feature engineering, and machine learning modeling across global financial markets.

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![Package Version](https://img.shields.io/badge/version-0.9.9.5-green)](pyproject.toml)
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
- **Interactive Dashboards**: Dash + Plotly GEIB dashboard for market monitoring, risk metrics, and portfolio visualization (a Streamlit extra is declared but no Streamlit app ships yet).

### Quick Start

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
. .\set_env.ps1

# Full expected-returns pipeline (8-phase ML workflow + portfolio optimization)
python expected_returns_v3.py

# Or the Bayesian Kalman price-target workflow
python pymc_kalman_filter_pt.py
```

Both pipelines read from PostgreSQL, so a reachable `DB_URL` is required (see
[Environment Variables](#environment-variables)).

See [Setup](#setup), [Execution & Entry Points](#execution--entry-points), and [Testing](#testing) for details.

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

### Kalman Price-Target Workflow (`pymc_kalman_filter_pt.py`)

The largest standalone workflow in the repository (~6.4k lines) and the script
counterpart of `pymc_kalman_filter_pt.ipynb` / `pymc_kalman_filter_pt_v2.ipynb`.
It runs an end-to-end fused MvGRW + volatility-conditioned panel model built on
`probabilistic_ml_model/pymc_models/KalmanFilterModel.py`
(`KalmanFilterPriceTarget`, `KalmanPanelInputs`, `build_fused_kalman_pt_model`):

1. §1 data load from PostgreSQL + feature-catalogue role resolution
2. §2 EDA panels (Plotly / matplotlib / `arviz-plots`)
3. §3–4 state-space feature mapping and panel container construction
4. §5b–8 fused panel model → prior predictive → NUTS posterior → posterior predictive
5. §9–10c diagnostics, cross-sectional screen, CVaR-aware risk book, analytics export
6. §10K–13 universe consensus fit, single-ISIN filter, mingled cohort, granular forest (all with optional stochastic-volatility twins)
7. §14 summary and actionable recommendations

Run it directly, or import `main()` for programmatic control:

```powershell
python pymc_kalman_filter_pt.py
```

```python
from pymc_kalman_filter_pt import main

artifacts = main(run_eda_section=True, write_analytics=True, robust=False, export_results=True)
# -> {'idata', 'results', 'kalman_results', 'panel', 'screen', 'universe_fit'}
```

Artifacts are written to `KALMAN_PT_RESULTS_DIR` in a **per-section
subdirectory** (`01_data/`, `02_eda/`, … `14b_recommendations/`, with `00_misc/`
as the fallback):

| Artifact    | Destination                                                                        |
|-------------|------------------------------------------------------------------------------------|
| Figures     | PNG via kaleido; self-contained HTML when kaleido/Chromium is unavailable           |
| DataFrames  | curated bulk frames → `analytics."<stem>"` **plus** a generated `<stem>.sql` DDL; all other frames → CSV |
| DataTrees   | NetCDF (h5netcdf) + a compact per-group JSON summary                               |

The curated set is `04_panel_frame`, `09_diagnostics_01_table`,
`10_screen_results`, `10_screen_mc_summary`, `10b_risk_analytics`,
`10b_risk_book`, `10c_kalman_results`. Set `KALMAN_PT_SQL_EXPORT=0` to emit DDL
and CSV without touching the database — the same fallback happens automatically
when the database is unreachable, so no frame is ever lost.

The script resolves `DB_URL` from the environment and falls back to parsing
`environment_variables.txt`. Set `KALMAN_PT_EXPORT_DRAWS=1` to additionally
bundle the raw `eu` / `ept` posterior draws (large, ~200 MB per array),
`KALMAN_PT_CLEAN_RESULTS=1` to purge each section subdirectory on first entry,
and `PML_FIG_WIDTH_PX` to match figure width to your display.

To re-file a pre-0.9.9.13 flat results directory into the tree:

```powershell
python pymc_kalman_filter_pt.py --migrate-layout           # dry run
python pymc_kalman_filter_pt.py --migrate-layout --apply
```

### Multi-Level Hierarchical Shrinkage (`probabilistic_ml_model/pymc_models/_hierarchy.py`)

Every PyMC model in `probabilistic_ml_model/pymc_models/` now shares a
single, canonical category hierarchy defined in `_hierarchy.py`:

- `HIERARCHICAL_CATEGORY_COLS` — `(region, country, trading_country,
  exchange, unit, sector, industry, style_class, size_class)`.
- `PARENT_MAP` — parent-of-child relationships
  (`region → country → exchange → sector → industry`, plus the
  independent `style_class → size_class` and `unit` / `trading_country`
  branches). Re-imported by `statistical_functions/statistical_models.py`
  so PyMC models and `hierarchical_mcmc_multi_level` share one source of
  truth (recommendation §12.4 #1).
- `build_hierarchy_indices(df, isins, levels=None)` — pure-NumPy helper
  returning per-level metadata (unique labels, `isin → level idx`,
  `level idx → parent idx`).
- `build_nested_logit_normal_rates(hierarchy, ...)` — PyMC helper that
  wires nested non-centred logit-Normal rates
  (`mu_L[g] = mu_P[parent_of(g)] + sigma_L * z_L[g]`) and returns the
  leaf rate broadcast to the `isin` dim.

Every model's `fit(...)` accepts a unified
`categories_df` + `hierarchy_levels` pair (back-compat: legacy
`sectors=` is auto-wrapped into a single-level
`categories_df({"sector": sectors})` with
`hierarchy_levels=["sector"]`).

```python
from probabilistic_ml_model.pymc_models import EarningsBeatBayesian

model = EarningsBeatBayesian()
idata, _ = model.fit(n_beats, n_total, isins, categories_df=df_categories,
                     hierarchy_levels=["exchange", "sector", "industry"])
```

### Core Models (`probabilistic_ml_model/pymc_models/`)

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

Some packages are restricted to `python_version < '3.14'` because they do not yet ship Python 3.14 wheels:
`streamlit`, `tensorflow`, `scikeras`.

`catboost` (>=1.2.10), `shap` (>=0.52.0), and `numba` (>=0.65.0) are now Python 3.14-compatible and are no longer gated.

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

| File                        | Purpose                                                 |
|:----------------------------|:--------------------------------------------------------|
| `pyproject.toml`            | Build system, project metadata, tool configs (v0.9.9.5) |
| `CHANGELOG.md`              | Release notes (Keep a Changelog / SemVer)               |
| `requirements.txt`          | Full dependency list (core + optional)                  |
| `Pipfile`                   | Pipenv dependency management                            |
| `environment_variables.txt` | Reference for all environment variables                 |
| `set_env.ps1`               | PowerShell script to set env vars for a session         |
| `.gitignore`                | Git ignore rules                                        |

## Execution & Entry Points

### Main Pipelines

```powershell
# v3 pipeline (8-phase ML workflow + portfolio optimization)
python expected_returns_v3.py

# Bayesian Kalman price-target workflow (fused panel model + screening + export)
python pymc_kalman_filter_pt.py
```

<!-- TODO: A v4 pipeline script (expected_returns_v4.py) is declared as the
     `finance-ml-v4` entry point in pyproject.toml but is not present in the
     repository yet. -->

### CLI Entry Points

The package provides command-line entry points (defined in `pyproject.toml`):

| Command               | Target                     | Description                         |
|:----------------------|:---------------------------|:------------------------------------|
| `finance-ml`          | `cli:main`                 | Main analysis pipeline              |
| `finance-ml-analyze`  | `cli:analyze`              | Quick data analysis and exploration |
| `finance-ml-validate` | `cli:validate`             | Data validation and schema check    |
| `finance-ml-v4`       | `expected_returns_v4:main` | v4 expected-returns pipeline        |

> **Note**: The `cli.py` module (backing `finance-ml`, `finance-ml-analyze`,
> and `finance-ml-validate`) and the `expected_returns_v4.py` module (backing
> `finance-ml-v4`) are declared in `pyproject.toml` but are not yet present in
> the repository. These entry points will not resolve until those modules are
> added.

### Interactive Dashboards

| Dashboard                        | Run Command                                              | Status    |
|:---------------------------------|:---------------------------------------------------------|:----------|
| **GEIB Dashboard** (modular)     | `python dashboards\geib_dash_app.py`                     | ✅ Live    |
| **GEIB Dashboard** (single-file) | `python dashboards\global_equity_investment_dashboard.py` | ⚠️ Legacy |

The modular app lives in `dashboards/geib/` (`app.py`, `data.py`, `metrics.py`,
`theme.py`, plus `charts/` and `components/` sub-packages); `geib_dash_app.py`
is its thin launcher. Set `GEIB_DASHBOARD=true` before launching.

<!-- TODO: No Streamlit / Dash apps exist under finance_ml/dashboards/ — remove
     or implement if the Streamlit entry point is still planned. -->

## Scripts & Utilities

| Script / File            | Description                                        |
|:-------------------------|:---------------------------------------------------|
| `expected_returns_v3.py`                | Automated expected-returns pipeline (module header: v3.6) |
| `pymc_kalman_filter_pt.py`              | Kalman price-target workflow (fused panel model, ~6.4k lines) |
| `dashboards\geib_dash_app.py`           | GEIB equities dashboard launcher (live)            |
| `feature_factory\eda_visualizations.py` | Ad-hoc EDA visualization helpers                   |
| `feature_factory\dcf_calculator.py`     | Standalone DCF feature calculator                  |
| `screening_etf_dw_transformations.py`   | ETF / data-warehouse screening transformations     |
| `set_env.ps1`                           | Set environment variables for a PowerShell session |
| `main.py`                               | PyCharm-generated sample stub (no project logic)   |

<!-- TODO: `main.py` is still the default PyCharm "print_hi" template — either
     implement a real top-level entry point or delete it. -->

### Feature Factory (`feature_factory/`)

Standalone feature calculation utilities (Beta/CAPM, DCF, Monte Carlo, Efficient Frontier, etc.).

### SQL Scripts (`sql_scripts/` and repo root)

Database setup and migration scripts for PostgreSQL (schema creation, materialized views, feature registry, data
import). Key root-level SQL files:

| File                                                   | Purpose                                      |
|:-------------------------------------------------------|:---------------------------------------------|
| `create_equities_schema.sql`                           | Equities table / schema creation             |
| `equities_schema_metadata_setup.sql`                   | Populates `public.equities_schema_metadata`  |
| `mv_equities.sql`                                      | Materialized view `mv_equities`              |
| `mv_expected_returns.sql` / `mv_exp_returns.sql`       | Expected-returns materialized views          |
| `mv_dcf.sql`                                           | DCF materialized view                        |
| `feature_registry.sql`                                 | `calculated_features_registry` setup         |
| `pml_df_metadata.sql` / `pml_df_metadata_populate.sql` | `pml.pml_df_metadata` DDL + population       |
| `pml_feature_catalogue.sql`                            | PML feature catalogue (PyMC alignment layer) |
| `pml_cohorts.sql`                                      | Cohort definitions for the PML pipeline      |
| `import_equities_data.sql` / `import_pml_data.sql`     | Bulk import scripts                          |
| `currency_import.sql`                                  | Currency reference data import               |
| `create_helper_functions.sql`                          | Shared PL/pgSQL helper functions             |

### Jupyter Notebooks

Exploratory and reproducible analysis notebooks live at the repository root. Install the `notebooks` extra to run them:
`pip install -e ".[notebooks]"`.

| Notebook                                      | Purpose                                                    |
|:----------------------------------------------|:-----------------------------------------------------------|
| `pymc_expected_returns_model.ipynb`           | End-to-end PyMC + ArviZ workflow for the 7 Bayesian models |
| `pymc_expected_returns_v2.ipynb`              | v2 PyMC expected-returns experiments                       |
| `pymc_kalman_filter_pt.ipynb`                 | Kalman price-target workflow (notebook twin of the script) |
| `pymc_kalman_filter_pt_v2.ipynb`              | v2 Kalman price-target workflow                            |
| `pymc_pml_model.ipynb`                        | PyMC PML model exploration                                 |
| `pymc_dcf.ipynb`                              | PyMC DCF price-target model walkthrough                    |
| `pymc_earnings_beat.ipynb`                    | PyMC earnings-beat model walkthrough                       |
| `pymc_price_target.ipynb`                     | PyMC price-target model walkthrough                        |
| `pymc_price_target_v2.ipynb` / `_v3.ipynb`    | Later price-target model iterations                        |
| `bayesian_expected_returns_var_model.ipynb`   | Bayesian VaR / expected-returns model                      |
| `pml_df_eda.ipynb`                            | Exploratory analysis of the PML dataframe                  |
| `expected_returns_v3.ipynb`                   | v3 expected-returns pipeline notebook companion            |
| `exp_returns_v3_analytics.ipynb`              | v3 analytics exploration                                   |
| `expected_returns_analytics.ipynb`            | Expected-returns analytics exploration                     |
| `pml_finance_model.ipynb`                     | PML finance model exploration                              |
| `pml_model_analysis.ipynb`                    | PML model analysis & diagnostics                           |
| `pml_bonds.ipynb`                             | PML bonds exploration                                      |
| `ExpectedReturnsAnalytics.ipynb`              | Expected-returns analytics                                 |
| `financial_market_statistical_analysis.ipynb` | Financial market statistical analysis                      |

## Environment Variables

Set via `set_env.ps1` (dot-source to persist in session: `. .\set_env.ps1`). Reference values are listed in
`environment_variables.txt`.

| Variable                   | Default / Example                                            | Description                                                       |
|:---------------------------|:-------------------------------------------------------------|:------------------------------------------------------------------|
| `LOG_LEVEL`                | `INFO`                                                       | Python logging level                                              |
| `TF_CPP_MIN_LOG_LEVEL`     | `2`                                                          | TensorFlow log level (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)        |
| `NO_COLOR`                 | `1`                                                          | Disable ANSI-colored console output (set by `set_env.ps1`)        |
| `DATA_DIR`                 | `data`                                                       | Local data storage                                                |
| `MODEL_DIR`                | `models`                                                     | Saved model artifacts (`set_env.ps1` currently sets `regression`) |
| `CACHE_DIR`                | `.cache`                                                     | Cache directory                                                   |
| `OUTPUT_DIR`               | `outputs`                                                    | Generated reports / visualizations                                |
| `KALMAN_PT_RESULTS_DIR`    | `pymc_kalman_filter_pt_results`                             | Kalman artifact-export root; artifacts land in per-section subdirectories |
| `KALMAN_PT_EXPORT_DRAWS`   | `0`                                                          | `1` also exports the raw `eu` / `ept` posterior draws as NetCDF (large) |
| `KALMAN_PT_SQL_EXPORT`     | `1`                                                          | `0` skips the analytics-schema write for curated frames (DDL + CSV only) |
| `KALMAN_PT_CLEAN_RESULTS`  | `0`                                                          | `1` purges each section subdirectory on first entry (no cross-run interleaving) |
| `DB_ANALYTICS_OWNER`       | `postgres`                                                   | Owner emitted in the generated analytics DDL                      |
| `PML_FIG_WIDTH_PX`         | *(unset)*                                                    | Target Plotly / matplotlib figure width (px) for the Kalman panels |
| `PML_STRICT_STREAK_MERGE`  | *(unset)*                                                    | Truthy = fail fast on missing EPS streak-merge columns (CI guard) |
| `DB_URL`                   | `postgresql+psycopg2://postgres:...@localhost:5432/postgres` | SQLAlchemy DB connection URL                                      |
| `DB_EQUITIES_SCHEMA`       | `public`                                                     | PostgreSQL schema for equities tables                             |
| `DB_TABLE`                 | `equities`                                                   | Source equities table name                                        |
| `DB_PML_SCHEMA`            | `pml`                                                        | PostgreSQL schema for PML tables                                  |
| `DB_ANALYTICS_SCHEMA`      | `analytics`                                                  | PostgreSQL schema for analytics outputs                           |
| `MODEL_VERSION`            | `v9_10` (`set_env.ps1`) / `v9_11` (`environment_variables.txt`) | Active model version tag                                      |
| `RANDOM_SEED`              | `42`                                                        | Reproducibility seed                                              |
| `N_JOBS`                   | `-1`                                                         | Parallel job count (`-1` = all cores)                             |
| `PML_ENABLE_PYTENSOR_C`    | `1`                                                         | Opt-in flag to enable PyTensor's g++ C backend (MSYS2 UCRT64)     |
| `PYTENSOR_FLAGS`           | `floatX=float64,cxx=`                                       | PyTensor configuration (default: pure-Python/numba VM; C backend enabled only when `PML_ENABLE_PYTENSOR_C=1`) |
| `GEIB_DASHBOARD`           | `true`                                                       | Enable GEIB equities dashboard                                    |
| `ENABLE_INTERACTIVE_PLOTS` | `true`                                                       | Toggle interactive visualizations                                 |

## Testing

The project uses `pytest` as the primary testing framework (configured in `pyproject.toml`), with `unittest.TestCase` style also supported.

```powershell
# Full suite
pytest

# Verbose
pytest -v

# Specific file
pytest tests\test_pml_workflow_v4.py

# Single test
pytest tests\test_kalman_filter_pt.py -k panel

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
| `test_kalman_filter_pt.py`                | 24      | Kalman price-target workflow & panel model   |
| `test_v35_earnings_beat.py`               | 22      | v3.5 earnings beat model tests               |
| `test_ensemble_risk_adj_return.py`        | 21      | Ensemble risk-adjusted return scoring        |
| `test_arviz_migration.py`                 | 18      | ArviZ 1.0 API migration                      |
| `test_price_target_mc.py`                 | 18      | Price-target Monte Carlo helpers             |
| `test_arviz_improvements.py`              | 17      | ArviZ diagnostic improvements                |
| `test_pipeline_statistical_fixes.py`      | 17      | Pipeline statistical fix validations         |
| `test_dcf_pt_nb_integration.py`           | 15      | DCF price-target notebook integration        |
| `test_price_target_panel.py`              | 14      | Price-target panel inputs & coords           |
| `test_hierarchical_mcmc_refactor.py`      | 11      | Hierarchical MCMC refactoring                |
| `test_hierarchical_pymc_models.py`        | 11      | Hierarchical PyMC model shrinkage            |
| `test_new_columns.py`                     | 9       | New column definitions & schema              |
| `test_idata_shim_smoke.py`                | 9       | InferenceData shim smoke tests               |
| `test_v35_anomaly_enhancements.py`        | 7       | v3.5 accounting anomaly enhancements         |
| `test_distribution_fitting.py`            | 6       | Distribution fitting models                  |
| `test_plr_idata_kwargs.py`                | 3       | ProbabilisticLinearRegression idata args     |
| **Total**                                 | **565** | All 25 test modules covered                  |

> Counts are `def test_*` functions per module (parametrized cases expand at
> runtime). Tests that need PostgreSQL are skipped when `DB_URL` is unreachable.

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
│   ├── pymc_models/                # PyMC model implementations
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
│   │   ├── _hierarchy.py           # Canonical category hierarchy (shrinkage)
│   │   ├── _feature_alignment.py   # Typed feature coercion & provenance helpers
│   │   ├── _price_target_mc.py     # Price-target Monte Carlo helpers
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
│   ├── geib_dash_app.py            # GEIB equities dashboard launcher (live)
│   ├── global_equity_investment_dashboard.py  # Legacy single-file dashboard
│   └── geib/                       # Modular GEIB app (app, data, metrics, theme)
│       ├── charts/                 # Plotly chart builders (CAPM, MC, VaR, Kelly, …)
│       ├── components/             # Reusable Dash components
│       └── assets/                 # CSS assets
├── feature_factory/                # Feature calculation utilities (Beta/CAPM, DCF, Monte Carlo, EDA viz)
├── analytics/                      # Legacy analytics helpers (screening, statistics, viz)
├── sql_scripts/                    # SQL setup and migration scripts
│   ├── analytics/                  # Analytics schema SQL
│   ├── information_schema/         # Information schema queries
│   ├── pml/                        # PML schema SQL
│   └── public/                     # Public schema SQL
├── expected_returns_v3.py          # Automated expected-returns pipeline (v3.6)
├── pymc_kalman_filter_pt.py        # Kalman price-target workflow (fused panel model)
├── pymc_kalman_filter_pt_results/  # Kalman artifact exports, one subdirectory per workflow step
│   ├── 01_data/ 02_eda/ …          #   (through 14b_recommendations/, 00_misc/ fallback)
├── main.py                         # PyCharm sample stub (see TODO above)
├── *.sql                           # Root-level schema / materialized-view / metadata SQL
├── *.ipynb                         # Exploratory / reproducible analysis notebooks
├── tests/                          # Unit and integration tests (565 tests, 25 modules)
├── data/                           # Local data storage
├── outputs/                        # Reports and visualizations
├── logs/                           # Pipeline execution logs
├── archive/                        # Archived files and prior outputs
├── docs/                           # Documentation (architecture, ArviZ migration, PyMC guides)
├── reference material/             # External reference notebooks and datasets
├── CLAUDE.md                       # AI-assistant contributor guide
├── CHANGELOG.md                    # Release history (Keep a Changelog / SemVer)
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

### Phase 2 — Core PML Model Execution (`pymc_models/`)

7. Run `MonteCarloSimulation` — triangular distribution sampling with historical target drift priors
8. Run `PriceTargetModel` — probability-weighted expected returns with analyst sentiment & risk adjustment
9. Run `KalmanFilterModel` — noise-reduced price target signals with momentum-informed priors
10. Run `EarningsBeatModel` — three-layer Bayesian beat probability (+ EPS streak & resampled priors)
11. Run `AccountingAnomalyModel` — multi-layered statistical anomaly detection with Mahalanobis distance
12. Run `CreditRiskModel` — Bayesian distress estimation with debt trajectory & balance sheet strength
13. Run `DividendSafetyModel` — dividend cut probability with FCF coverage & leverage signals

### Phase 3 — Probabilistic Linear Market Model (`pymc_models/`)

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

## Documentation

Longer-form documentation lives in `docs/`:

| Document                                | Contents                                              |
|:----------------------------------------|:-------------------------------------------------------|
| `docs/ARCHITECTURE.md`                  | System architecture overview                           |
| `docs/ArviZ 1.0 migration.md`           | ArviZ 1.x migration notes                              |
| `docs/pymc_kalman_filter_pt.md`         | Kalman price-target workflow walkthrough               |
| `docs/PyMC_overview.md` / `PyMC_glossary.md` | PyMC concepts and terminology                     |
| `docs/pml_features.md`                  | Feature catalogue reference                            |
| `docs/pml_sql_queries_updates.md`       | SQL query / schema change log                          |
| `docs/global_equity_investment_dashboard.md` | GEIB dashboard design notes                       |
| `docs/expected_returns_v5_dev.md`       | Planned v5 pipeline design                             |
| `CLAUDE.md`                             | Contributor guide for AI assistants (deep-dive on conventions) |
| `finance_ml_analytics_guide.md`         | Analytics platform usage guide                         |

## Contributing

- Format with `black .` and `isort .`; lint with `flake8`; type-check with `mypy`.
- Run the test suite (`pytest`) before submitting changes.
- See `CHANGELOG.md` for release history (Keep a Changelog / SemVer).

## License

This project is licensed under the **MIT License**, declared in `pyproject.toml` (`license = { text = "MIT" }`).

No top-level `LICENSE` file is present in the repository.

### Open TODOs

- **TODO**: Add a top-level `LICENSE` file with the full MIT license text.
- **TODO**: Confirm and document the copyright holder / authors (currently `Finance ML Team` in `pyproject.toml`).
- **TODO**: Implement the `cli.py` module backing the `finance-ml`, `finance-ml-analyze`, and `finance-ml-validate`
  entry points declared in `pyproject.toml`.
- **TODO**: Add `expected_returns_v4.py` backing the `finance-ml-v4` entry point, or drop the entry point.
- **TODO**: Replace or remove the placeholder `main.py`.
- **TODO**: Align `MODEL_DIR` between `set_env.ps1` (`regression`) and `environment_variables.txt` (`models`), and
  `MODEL_VERSION` (`v9_10` vs `v9_11`).
- **TODO**: `environment_variables.txt` contains a committed database password — rotate it and move secrets to an
  untracked `.env`.