# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PML Finance Project** is a comprehensive platform for probabilistic equity screening, feature engineering, and machine
learning modeling for financial markets. The core system implements an 8-phase workflow combining Bayesian models (via
PyMC 6.0), statistical analysis, and portfolio optimization.

**Key Technologies:**

- Python 3.12–3.14
- **PyMC 6.2** + **PyTensor 3.2** + **ArviZ 1.1** — Bayesian inference and diagnostics
    - `pymc` ↔ `pytensor` are a coupled pair (pymc 6.2 requires pytensor >=3.2.2,<3.3) — bump them together.
    - ArviZ 1.x ships as three packages: `arviz-base` (data containers), `arviz-stats` (diagnostics), `arviz-plots` (
      visualization). The top-level `arviz` meta-package re-exports all three for backward-compatible imports.
    - ArviZ 1.x replaces `arviz.InferenceData` with `xarray.DataTree` as the canonical output type. Use the
      `InferenceLike` alias (defined in `probabilistic_ml_model/_pymc_arviz_compat.py`) for type annotations:
      `Union[arviz.InferenceData, xarray.DataTree]`.
    - **nutpie 0.16+** — default high-performance sampler (numba backend)
    - **JAX 0.11+ / jaxlib 0.11+** (mutually pinned pair), **blackjax 1.6+**, **numpyro 0.18+** — alternative JAX-based samplers
    - **bambi 0.19+** — formula-based GLM interface on top of PyMC
    - **Blocked upgrades** (documented in-line in the dependency files): `numpy` stays <2.5 (numba 0.65/0.66 both
      require it) — numba is the project's default PyTensor backend. numba 0.66 + llvmlite 0.48 are allowed since
      pytensor 3.2.4 raised its numba cap to <=0.66.0 (0.65.1 was the ceiling under pytensor <=3.2.3).
- PostgreSQL — centralized data storage with 17 feature views
- pandas/NumPy/SciPy — data processing
- scikit-learn, XGBoost, LightGBM, CatBoost — classical ML
- Plotly, Matplotlib, Seaborn — visualization
- Streamlit (Python < 3.14 only), Dash — interactive dashboards
- pytest — 569 test cases across 25 test modules

## Development Setup

### Initial Environment Setup

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
. .\set_env.ps1
```

Key environment variables (full list in `environment_variables.txt`):

| Variable                                                | Purpose                                                          |
|---------------------------------------------------------|------------------------------------------------------------------|
| `DB_URL`                                                | SQLAlchemy PostgreSQL connection URL                             |
| `DB_EQUITIES_SCHEMA` / `DB_PML_SCHEMA` / `DB_ANALYTICS_SCHEMA` | Source / feature / output schema names (`public` / `pml` / `analytics`) |
| `DB_TABLE`                                              | Source equities table                                            |
| `PYTENSOR_FLAGS`                                        | PyTensor backend; on Windows points `cxx` at the MSYS2 g++       |
| `JAX_PLATFORM_NAME`                                     | `cpu` / `gpu` for blackjax / numpyro samplers                    |
| `LOG_LEVEL` / `TF_CPP_MIN_LOG_LEVEL`                    | Python / TensorFlow logging verbosity                            |
| `DATA_DIR` / `MODEL_DIR` / `CACHE_DIR` / `OUTPUT_DIR`   | Artifact directories                                             |
| `MODEL_VERSION` / `RANDOM_SEED`                         | Model run identifier / RNG seed                                  |
| `N_JOBS`                                                | Parallel job count (`-1` = all cores)                            |
| `PML_STRICT_STREAK_MERGE`                               | Fail-fast on missing EPS streak-merge columns (CI/regression)    |
| `PML_ENABLE_PYTENSOR_C`                                 | `1` opts back into the PyTensor C backend (default: numba/py VM) |
| `PML_FIG_WIDTH_PX`                                      | Target Plotly/mpl figure width (px) for the Kalman notebook panels |
| `KALMAN_PT_RESULTS_DIR` / `KALMAN_PT_EXPORT_DRAWS`      | Kalman artifact-export root (per-section subdirectories; PNG/CSV/SQL/JSON/NetCDF) / `1` also exports raw eu/ept draws |
| `KALMAN_PT_SQL_EXPORT` / `KALMAN_PT_CLEAN_RESULTS`      | `0` skips the analytics-schema write (DDL + CSV only) / `1` purges each section subdirectory on first entry |
| `DB_ANALYTICS_OWNER`                                    | Owner emitted in generated analytics DDL (default `postgres`)     |

### Code Quality & Testing

```powershell
# Format code
black .
isort .

# Lint
flake8 probabilistic_ml_model/ tests/ --max-line-length=100

# Type checking
mypy probabilistic_ml_model/

# Run all tests
pytest -v

# Run single test file
pytest tests/test_pml_workflow_v4.py -v

# Show coverage
pytest --cov=probabilistic_ml_model --cov-report=term-missing tests/
```

## Project Structure

```
PML_Finance_Project/
├── probabilistic_ml_model/      # Core package (lazy-loaded PyMC/ArviZ)
│   ├── pymc_models/             # 7 Bayesian models + _workflow / _hierarchy / _feature_alignment
│   │                            #   + RiskBookModel / _price_target_mc (decision analysis)
│   ├── statistical_functions/   # Hierarchical MCMC, probability & ensemble models
│   ├── data_utils/              # DB loading, feature_catalog, inference_schema, export
│   ├── visualizations/          # Per-model plot modules + ArviZ diagnostics
│   ├── pipeline_runners.py      # 8-phase orchestration via PipelineConfig
│   ├── _pytensor_env.py         # Forces the PyTensor VM before any pytensor import
│   └── _pymc_arviz_compat.py    # InferenceLike type alias (arviz.InferenceData | xarray.DataTree)
├── tests/                       # 569 pytest cases across 25 modules
├── sql_scripts/
│   ├── pml/                     # pg_dump EXTRACT — tables + vw_pml_df_* only; MV/function files are stubs
│   ├── analytics/               # Output analytics tables/screens (kalman_filtered_price_targets, screens)
│   └── public/                  # Legacy public-schema views
├── dashboards/                  # geib/ package (Dash GEIB board, :8050) + launcher; legacy geib_dash_app.py
├── feature_factory/             # Ad-hoc feature/cohort SQL + plotting
├── docs/                        # Architecture guides (PyMC, ArviZ 1.0, SQL)
├── data/                        # Regional PML / screening CSV snapshots
├── reference material/          # MyST / notebook reference material
├── archive/                     # Archived scripts/notebooks (expected_returns_v4.py, pml_workflow_v4.ipynb, …)
├── *.ipynb                      # PyMC model + analytics notebooks (see Key Notebooks)
├── pml_feature_catalogue.sql    # SSOT: pml.* functions, all 7 mv_pymc_* MVs, catalogue views, coverage check
├── pml_df_metadata.sql          # SSOT: metadata/alias table DDL + CHECK-enforced vocabularies
├── pml_df_metadata_populate.sql # SSOT: pymc_role / model_targets assignment (+ §7i coverage reconciliation)
├── *.sql                        # Other root-level schema/import SQL (import_pml_data.sql, …)
├── expected_returns_v3.py       # Main v3 pipeline entry point
├── pymc_kalman_filter_pt.py     # Kalman price-target workflow (~8.1k lines; fused panel model + screen + analytics export)
├── pyproject.toml / Pipfile / requirements.txt   # Dependency definitions (keep in sync)
├── set_env.ps1 / environment_variables.txt       # Environment configuration
├── CHANGELOG.md                 # Release notes (authoritative version source)
└── CLAUDE.md                    # This file
```

## Architecture & Core Abstractions

### 1. Feature Catalog (data_utils/feature_catalog.py)

Single source of truth for which columns each PyMC model needs.

```python
from probabilistic_ml_model.data_utils.feature_catalog import (
    get_feature_catalog,
    FEATURE_VIEW_REGISTRY,  # 17 views
)

catalog = get_feature_catalog()
features = catalog.get_features_for_model("EarningsBeatBayesian")
```

Key data sources:

- `pml.pml_df` — core equities source table
- `pml.vw_pymc_feature_catalogue` pymc model feature catalogue
- `pml.pml_df_metadata` — feature → category → pymc_model mappings

### 2. Hierarchical Shrinkage (pymc_models/_hierarchy.py)

Canonical multi-level category hierarchy shared by all 7 PyMC models:

```python
from probabilistic_ml_model import (
    HIERARCHICAL_CATEGORY_COLS,
    PARENT_MAP,  # region → country → exchange → sector → industry
    build_hierarchy_indices,
)

idata, _ = model.fit(data, categories_df=df_categories, hierarchy_levels=["exchange", "sector", "industry"])
```

### 3. Feature Alignment & ArviZ (pymc_models/_feature_alignment.py)

Column names passed to `coerce_by_data_type()` must match `pml.pml_df` column names exactly — derive them from
`pml.vw_pymc_feature_catalogue` (SQL), not from Python variable names. Every model stamps feature provenance onto
`idata.constant_data`:

```python
from probabilistic_ml_model import (
    coerce_by_data_type,
    stamp_feature_provenance,
    load_feature_metadata_from_db,
)
```

Two helpers in this module are **exported but called by nothing** — treat them as the intended contract for new code,
not as dead weight to imitate:

| Helper                                   | What it guards                                                              |
|------------------------------------------|-----------------------------------------------------------------------------|
| `validate_oos_shape(new_arr, aliases)`   | `new_arr.shape[1] == len(feature_aliases)` before any `pm.set_data(...)` swap |
| `assert_disjoint_features(idata, …)`     | a model's feature set does not collide with another model's on the same idata |

### 4. PyMC Models (pymc_models/)

7 Bayesian models with unified interface:

| Model                     | Purpose                                                                                                                                                                                                                                              |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| EarningsBeatBayesian      | Beat probability                                                                                                                                                                                                                                     |
| PriceTargetAchievement    | Return expectation                                                                                                                                                                                                                                   |
| KalmanFilterPriceTarget   | Smoothed signals. Single-response hierarchical panel with learned risk/size/volume tilts. On a `T > 1` panel it carries a **per-ISIN random intercept** (`sigma_isin_level`) — restored because repeated observations identify it, which the T=1 cross-section could not. `state_now` is the decision latent — resolve it via `KALMAN_SCREEN_LATENT` / `resolve_screen_latent`, never by reading `risk_adj_return` directly. **The genuine `(isin, time)` T=4 log-uplift panel is the DEFAULT** — built from the `price_target_{6m,3m,1m}_ago` / `price_{6m,3m,1m}_ago` trails via `KalmanRunConfig.panel_lookbacks = ('6m','3m','1m')`. Collapse to the T=1 MV snapshot with `replace(cfg, panel_lookbacks=())`; an opt-in AR(1) state layer sits behind `state_innovation_scale` (off by default — see the config table). See `build_fused_kalman_pt_model` |
| DCFPriceTarget            | Fair-value bands                                                                                                                                                                                                                                     |
| DividendSafetyBayesian    | Cut probability                                                                                                                                                                                                                                      |
| CreditRiskBayesian        | Distress risk                                                                                                                                                                                                                                        |
| AccountingAnomalyBayesian | Quality flags                                                                                                                                                                                                                                        |

Each returns `InferenceLike` (i.e. `arviz.InferenceData | xarray.DataTree`) with posterior, constant_data (features +
provenance attrs), and diagnostics. Use the compat shim in `_pymc_arviz_compat.py` for type annotations instead of
importing `arviz.InferenceData` directly, as ArviZ 1.x uses `xarray.DataTree` internally.

Also in this package, but **not** `fit()`-style models:

| Module                             | Role                                                                    |
|------------------------------------|-------------------------------------------------------------------------|
| `_workflow.py`                     | Bayesian-workflow helper SSOT (§9 below)                                |
| `_hierarchy.py` / `_feature_alignment.py` / `_pytensor_compat.py` | Shared model plumbing            |
| `_price_target_mc.py`              | Decision analysis (§10 below)                                           |
| `RiskBookModel.py`                 | Decision analysis — CVaR-aware sizing (§11 below)                       |
| `MonteCarloSimulation.py`          | Module-level `fit()` + `MonteCarloReturnSimulation`                     |
| `ProbabilisticLinearRegressionModel.py` | Bayesian linear regression                                         |
| `BaselineProbabilityModel.py`      | **Not a PyMC model** — a `PipelineConfig` dataclass + `main()` orchestrator |
| `PortfolioOptimizationModel.py`    | 0-byte stub, unimplemented since 2025-07-02                             |

### 5. Pipeline Runner (pipeline_runners.py)

Orchestrates all 8 phases via `PipelineConfig` (`pipeline_runners.py:34-224`). The field groups, rather than a
count:

| Group                | Representative fields                                                                            |
|----------------------|--------------------------------------------------------------------------------------------------|
| MC / MCMC budget     | `mc_simulations=10_000`, `mc_max_stocks`, `mcmc_chains=8`, `mcmc_samples=5_000`, `mcmc_burn_in`  |
| Likelihood & tails   | `use_student_t`, `student_t_df_floor`, `use_mixture_likelihood`, `tail_risk_metric`, `cvar_alpha` |
| Volatility           | `use_garch_volatility`, `garch_p/q`, `use_stochastic_vol`, `vol_regime_window`                   |
| Ensemble & BMA       | `use_bayesian_model_averaging`, `bma_prior_weights`, `bma_log_score_window`, `ensemble_shrinkage_kappa` |
| Macro                | `use_macro_covariates`, `macro_covariates`, `macro_hierarchy_level`                              |
| Rolling backtest     | `enable_rolling_backtest`, `backtest_window_months`, `backtest_step_months`, `ci_coverage_target` |
| Screening thresholds | `screening_min_pct`, `screening_quality_roe_min`, `screening_quality_piotroski_min`, …           |
| Cache / perf         | `n_jobs`, `enable_result_caching`, `enable_mcmc_caching`, `cache_ttl_hours`, `export_max_workers` |

> **Trap:** `PipelineConfig.from_env()` (`:127`) does **not** reproduce the dataclass defaults. `ER_MC_SIMULATIONS`
> defaults to `50_000` (vs `10_000`) and `ER_MCMC_SAMPLES` to `10_000` (vs `5_000`), so a config built from the
> environment samples ~2–5× more than one built directly. Check which constructor you are on before comparing runs.

BMA weighting and `compute_cross_model_correlation()` (`:2136`) are the *only* cross-model machinery here — there is
no ELPD/LOO comparison (see the Bayesian Workflow section).

### 9. Workflow Helpers (pymc_models/_workflow.py)

The SSOT for the sampling / diagnostics / predictive-check stages shared by every model — see
**The Bayesian Workflow — Stage Contract** below for what each one is for.

```python
from probabilistic_ml_model.pymc_models._workflow import (
    MIN_ESS_GATE,             # 400 — project convergence gate
    build_sample_kwargs,      # canonical pm.sample() kwargs assembly
    log_sample_diagnostics,   # divergences + bulk-ESS warnings from code
    prior_predictive_check,   # pm.sample_prior_predictive wrapper
    posterior_predictive_check,  # pm.sample_posterior_predictive wrapper
    attach_log_likelihood,    # post-hoc log_likelihood -> enables az.loo/az.compare
    posterior_dataset,        # DataTree/InferenceData -> flat xarray.Dataset
)
```

### 10. Price-Target Monte Carlo (pymc_models/_price_target_mc.py)

The forward-return decision layer feeding both the analytics export and the GEIB dashboard:

```python
from probabilistic_ml_model.pymc_models._price_target_mc import (
    prepare_price_target_inputs, prepare_price_target_panel_inputs,
    simulate_lagged_risk_adjusted_returns,  # AR-damped structural-TS forward draws
    summarize_mc_returns,                   # -> er_mean, er_sd, er_p05, er_p50, er_p95, prob_pos
)
```

`summarize_mc_returns` produces the `er_*` / `prob_pos` columns persisted in
`analytics.kalman_filtered_price_targets`; `expected_sharpe_ratio = er_mean / er_sd`.

### 11. CVaR Risk Book (pymc_models/RiskBookModel.py)

```python
from probabilistic_ml_model.pymc_models.RiskBookModel import RiskBook, compute_cvar_aware_book

rb = compute_cvar_aware_book(idata, screen.eu, results, alpha=0.05, cap=0.08, k_book=25)
rb.analytics   # per-name risk columns incl. cvar05, exp_vol, starr, book_weight
rb.book        # STARR-ranked, cap-and-spill sized long book (weights sum to 1)
rb.summary     # port_up, port_cvar, wavg_cvar, starr_book, div, n_book, sizing params
```

`pymc_kalman_filter_pt.compute_cvar_aware_book(idata, panel, screen, results, config=…)` is a thin wrapper that
resolves the sizing knobs from `KalmanRunConfig` and delegates here. All columns are raw decimals.

### 6. Data Loading (data_utils.py)

Dynamic column discovery via PostgreSQL:

```python
from probabilistic_ml_model.data_utils import (
    load_equities_data_from_db,
    load_all_feature_views,
    get_equities_schema,
)

equities_df = load_equities_data_from_db()
feature_dfs = load_all_feature_views()  # parallel load
```

### 7. Statistical Analysis (statistical_functions/statistical_models.py)

Per-category Bayesian inference and hierarchical MCMC:

```python
from probabilistic_ml_model.statistical_functions import (
    hierarchical_mcmc_multi_level,
    fit_distributions_by_category,
)

posteriors = hierarchical_mcmc_multi_level(
    df,
    response='expected_return',
    category_cols=['region', 'sector', 'industry'],
)
```

### 8. Visualization (visualizations/)

8 model-specific modules plus ArviZ diagnostics:

```python
from probabilistic_ml_model.visualizations import (
    plot_expected_returns,
    plot_convergence_diagnostics,
)
```

PPC plotting exists (`create_screening_ppc_rootogram`, `create_screening_ppc_continuous`, plus `azp.plot_ppc_dist` /
`plot_ppc_tstat` call sites in `arviz_diagnostics.py`). There is **no** prior-predictive plot helper and **no**
LOO/WAIC/`az.compare` plot anywhere in `visualizations/`.

> `visualizations/__init__.py:60-176` builds `__all__` by dynamic import and **swallows `ImportError`**, so a broken
> submodule silently disappears instead of failing. `earnings_quality.py` has no `_IMPORT_REGISTRY` entry at all —
> import it directly from the submodule.

## The Bayesian Workflow — Stage Contract

The project follows the PyMC *Bayesian workflow*. This section is normative: new and modified models are expected to
reach every stage, using the shared helper for each rather than a local reimplementation.

The reference implementation is **`pymc_kalman_filter_pt.py`**, not the model package — §6 prior predictive, §8
posterior predictive and §9 diagnostics there are the standard to match.

### Stage → canonical API

| Stage                        | Use this                                                                                          | Reference                                                        |
|------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| Conceptual model building    | `pml.vw_pymc_feature_catalogue` → `select_drift_features()` / `_resolve_*_feature_aliases()`. Never hand-list columns | `KalmanFilterModel.select_drift_features` + `KALMAN_DRIFT_EXCLUDED_FEATURES` |
| Prior predictive             | `_workflow.prior_predictive_check(model, var_names=…, draws=…)`, then de-standardise onto an interpretable scale and compare to the empirical distribution | `run_prior_predictive` (`pymc_kalman_filter_pt.py:4009`)          |
| Computational implementation | `pm.Model(coords=…)` + `pm.Data`; `_hierarchy` helpers; `coerce_by_data_type`                     | `build_fused_kalman_pt_model`                                     |
| Fitting & diagnostics        | `_workflow.build_sample_kwargs()` → `pm.sample` → `_workflow.log_sample_diagnostics()`            | `run_diagnostics` (`:4378`); gate `MIN_ESS_GATE = 400`            |
| Model evaluation (PPC)       | `_workflow.posterior_predictive_check()` **plus ≥1 calibration statistic** (ECDF, t-stat, coverage, PIT) | `run_posterior_predictive` (`:4237`)                        |
| Model comparison             | `_workflow.attach_log_likelihood(idata, model)` → `az.compare` — see below                        | `run_model_comparison` (`pymc_kalman_filter_pt.py`, §9b)          |
| Expansion / simplification   | Record it in the builder docstring: what was tried, its divergence / R-hat counts, why it was dropped | `KalmanFilterModel.py:2134-2145`, `:2373`, `:2450`, `:2535`   |
| Decision analysis            | `_price_target_mc.summarize_mc_returns`; `RiskBookModel.compute_cvar_aware_book`                  | §10 screen / §10b risk book                                       |

The artifact tree mirrors the stages: `_EXPORT_SECTION_DIRS` **is** the workflow (`06_prior`, `08_ppc`,
`09_diagnostics`, …). A new stage means a new section directory resolved through `_export_dir_for` — never a
hand-built path.

### Current coverage

✅ implemented · ⚠️ partial · ❌ absent

| Module                          | Prior pred. | PPC                        | Diagnostics | Comparison | Decision analysis |
|---------------------------------|-------------|----------------------------|-------------|------------|-------------------|
| `pymc_kalman_filter_pt.py`      | ✅ `run_prior_predictive` | ✅ `run_posterior_predictive` (per-`y_series` **and** per-time coverage + PIT) | ✅ `run_diagnostics` | ✅ `run_model_comparison` (§9b, opt-in) | ✅ screen + risk book |
| KalmanFilterPriceTarget         | ❌          | ⚠️ `forecast()` hand-rolls predictions | ✅ `log_sample_diagnostics` | ❌ | ✅ `implied_upside_from_state` |
| EarningsBeatBayesian            | ❌          | ❌                         | ✅          | ❌         | ❌                |
| PriceTargetAchievement          | ❌          | ❌                         | ✅          | ❌         | ⚠️ `achieve_prob` latent |
| DCFPriceTarget                  | ❌          | ❌                         | ✅          | ❌         | ❌                |
| DividendSafetyBayesian          | ❌          | ❌                         | ✅          | ❌         | ❌                |
| CreditRiskBayesian              | ❌          | ❌                         | ✅          | ❌         | ❌                |
| AccountingAnomalyBayesian       | ❌          | ❌                         | ✅          | ❌         | ⚠️ `threshold`   |
| MonteCarloSimulation            | ❌          | ⚠️ post-hoc NumPy group    | ✅          | ❌         | ⚠️ `sim_returns` |
| ProbabilisticLinearRegression   | ❌          | ⚠️ opt-in, default off     | ✅          | ❌         | ❌                |

Diagnostics became universal when `log_sample_diagnostics` was lifted into `_workflow.py`; **prior predictive is the
largest remaining gap** in the model package — `prior_predictive_check()` now makes it a one-liner.

### `log_likelihood` and why comparison is opt-in

`log_likelihood` is **off by default** everywhere. It roughly doubles `InferenceData` size and adds materially to
wall-clock on a ~5k-ISIN cross-section, and no *production* pipeline path consumes it. That is a deliberate
trade-off — but the consequence is that **`az.loo` / `az.waic` / `az.compare` raise on every idata this repo
produces by default**.

The one place that opts back in is `run_model_comparison` (`pymc_kalman_filter_pt.py` §9b, gated by
`KalmanRunConfig.enable_model_comparison`), which attaches the group post-hoc with `attach_log_likelihood` and
subsamples the ISIN axis to `comparison_max_isins` — the group is `chains × draws × n_isin × T × D` floats
(~820 MB per arm at full T=4 panel size), and it is paid once per arm being compared. Follow that pattern rather
than enabling `log_likelihood` on a production fit.

Two escape hatches, one of which is a trap:

```python
# (1) Sampler-dependent — works for pymc/numpyro/blackjax, SILENTLY IGNORED under nutpie,
#     and DEPRECATED by PyMC itself (FutureWarning: "Passing `log_likelihood` via
#     `idata_kwargs` is deprecated ... Call `pm.compute_log_likelihood(idata)` instead").
idata, model = MyModel().fit(..., nuts_sampler="pymc", idata_kwargs={"log_likelihood": True})

# (2) Sampler-independent — the recommended route, and what PyMC now steers you to.
from probabilistic_ml_model.pymc_models._workflow import attach_log_likelihood
attach_log_likelihood(idata, model)          # wraps pm.compute_log_likelihood
az.compare({"a": idata_a, "b": idata_b})
```

**Reading the result:** ArviZ 1.x `ELPDData` exposes the value as **`.elpd`** — `.elpd_loo`, `.p_loo` and `.loo` are
all gone, even though the object's `repr` still prints the row label `elpd_loo`. Reading the old attribute yields a
silent `nan` via `getattr(loo, "elpd_loo", nan)` rather than an error:

```python
loo = az.loo(idata)
print(loo.elpd, loo.se)      # -122.38  5.13   ✅
print(loo.elpd_loo)          # AttributeError  ❌
```

Why (1) is a trap: `build_sample_kwargs` layers kwargs as `defaults → nuts_sampler → setdefault(idata_kwargs) →
update(sample_kwargs) → nutpie strip`. The `setdefault`-before-`update` is what lets your override win; the nutpie
strip afterwards discards `idata_kwargs` wholesale, because nutpie ignores it and warns. nutpie is the project
default sampler, so on the default path your override vanishes. `build_sample_kwargs` logs an INFO line when it
detects this.

On the script path, `sample_posterior` (`pymc_kalman_filter_pt.py:4095`) hard-codes `log_likelihood: False` at `:4165`
and takes no `**sample_kwargs` at all — use `attach_log_likelihood` on the returned idata.

### Checklist for a new or modified model

1. Resolve features from `pml.vw_pymc_feature_catalogue`, not Python literals.
2. `coerce_by_data_type()`; `assert_disjoint_features()` when combining feature sets.
3. Build with `pm.Data` + `coords`; hierarchy via `_hierarchy.py`.
4. Ship a **prior predictive check** on an interpretable scale before any posterior run.
5. Sample via `_workflow.build_sample_kwargs()` — never re-copy the kwargs block.
6. Call `_workflow.log_sample_diagnostics()` — warn from code, never rely on console scraping.
7. Ship a **posterior predictive check** with at least one calibration statistic.
8. `stamp_feature_provenance()` after `pm.sample()`; `validate_oos_shape()` on any `pm.set_data` path.
9. Docstring states what decision the model serves, and what alternatives were rejected and why.

## Entry Points & Workflows

### Main Pipeline

```powershell
# v3 pipeline (8 phases: data load → models → ensemble → MCMC → viz → export)
python expected_returns_v3.py

# Bayesian Kalman price-target workflow (fused panel model → screen → risk book → analytics export)
python pymc_kalman_filter_pt.py
```

`pymc_kalman_filter_pt.py` also exposes an importable `main()`:

```python
main(*, run_eda_section=True, write_analytics=True, robust=False,
     volume_penalty=0.25, export_results=True, config=None) -> dict[str, Any]
# -> {'idata', 'prior_idata', 'results', 'kalman_results',
#     'panel', 'screen', 'risk_book', 'universe_fit'}       # 8 keys
```

`volume_penalty=0.25` overrides `build_fused_kalman_pt_model`'s own `0.2` default; `0.0` disables the tilt.

Workflow knobs (NUTS budget, screen/risk-book parameters, panel lookbacks, universe-query dates) live on the frozen
`KalmanRunConfig` dataclass, passed via `main(config=…)`. **`from_env()` reads only five variables** —
`RANDOM_SEED`, `KALMAN_PT_RESULTS_DIR`, `KALMAN_PT_EXPORT_DRAWS`, `PML_FIG_WIDTH_PX`, `LOG_LEVEL`. Everything else
keeps its dataclass default and is overridden programmatically:

```python
from dataclasses import replace
cfg = replace(get_run_config(), panel_lookbacks=(), chains=8)   # T=1 cross-section, 8 chains
cfg = replace(get_run_config(), state_innovation_scale=0.1)     # enable the opt-in AR(1) state
cfg = replace(get_run_config(), enable_model_comparison=True)   # run §9b (≈3× sampling cost)
cfg = replace(get_run_config(), panel_response_extra=('pt_dispersion',))  # D=2, activates the ICM
```

Local-level state / comparison knobs on `KalmanRunConfig`:

| Field                     | Default | Purpose                                                                          |
|---------------------------|---------|----------------------------------------------------------------------------------|
| `draws` / `tune`          | `2000` / `4000` | Raised 2000/1000 → 4000/4000 on 2026-08-17, then **split asymmetrically to 2000/4000** in 0.9.9.17. The log-linear `sigma_isin` model mixes harder than the two-term scale it replaced: at 2000/1000 the full fit gave max global R-hat 1.0134 / min bulk ESS 460 at **zero divergences** — slow mixing, not bad geometry. At 4000/4000: **R-hat 1.0063, min ESS 884**. The two knobs are **not interchangeable** — tune bought the R-hat, draws bought the ESS — so the cut goes where the headroom is: ESS 884 against a 400 gate is ~2× margin and scales about linearly in draws, while R-hat had none to give. Cutting *tune* instead attacks the half that carried the gain. Both `validate_kalman_state.py` and `export_kalman_analytics.py` read this, so clearing the gate only via a `--draws` override would certify a model the export never fits. |
| `ppc_draws`               | `1000`  | Predictive draws retained for §8 (total across chains; `0` disables). `pm.sample_posterior_predictive` replicates once per posterior sample and takes no draw count, so an un-thinned call replays the whole `chains × draws` grid for statistics that are averages over ~26k observations. See `thin_posterior`. |
| `cores` (CLI)             | `4`     | `KalmanRunConfig.cores = 1` is a **notebook** constraint (nutpie's parallel native workers crash an IDE-managed Jupyter kernel on Windows), but `main()` and both scripts inherited it, so four chains ran sequentially even headless. `main(cores=…)` and `--cores` (default `_CLI_DEFAULT_CORES = 4`) override it on the script paths. Wall-clock only — chains, seeds and posterior are identical. |
| `state_innovation_scale`  | `0.0`   | **AR(1) time-varying state — OFF by default.** Tried and rejected at T=4: it bought +0.013 recovery correlation for min ESS 14 vs 69, with `sigma_state`/`rho` drifting between draw budgets. The per-ISIN intercept below carries the panel. Set to `0.1` to enable; revisit on a longer panel. |
| `isin_level_scale` (builder arg) | `0.10` | **FIXED scale (a constant, not a prior on a sampled parameter)** of the per-ISIN `ZeroSumNormal` intercept on `T > 1`. `sigma_isin_level` is `pm.Deterministic(pt.constant(...))`, so it reports sd 0, ESS == total draws and R-hat NaN by construction — do not read it as fitted. **Raising it to 0.40 was tried on 2026-08-16 and reverted**: the measured between-name level sd is 0.4718 (0.3854 after group effects), but at 0.40 `sigma_base` rose *with* the level instead of falling, total predictive scale went 0.2168 → 0.2517 and coverage over-shot to 98.4% against a 92% target. The arithmetic measures how much level *exists*, not how much the model absorbs without the rest of the scale expanding. `0.0` pins the layer off. |
| `enable_model_comparison` | `False` | Run §9b. Refits both arms and computes a pointwise `log_likelihood` per arm (~820 MB each at full panel size), so ≈3× the sampling cost. |
| `comparison_max_isins`    | `800`   | ISIN subsample used by §9b; the retained fraction is logged.                      |
| `panel_response_extra`    | `()`    | Keys of `KALMAN_PANEL_RESPONSE_EXTRA` promoting a second response series (`D > 1`), which is what activates the otherwise-dormant rank-1 ICM. |

**Section map** (the file is ~8.1k lines; sections are keyed to the Bayesian-workflow stages):

| §     | Line   | Content                                        |
|-------|--------|------------------------------------------------|
| 1/1c  | `:470` / `:1894` | Plot helpers · artifact export           |
| 2     | `:2817` | EDA panels                                    |
| 3     | `:3331` | State-space feature mapping                   |
| 5b    | `:3437` | Fused MvGRW panel model (Model A / Model B)   |
| **6** | `:4007` | **Prior predictive checks**                   |
| 7     | `:4093` | Posterior inference (NUTS)                    |
| **8** | `:4235` | **Posterior predictive checks** (per-`y_series` **and** per-time coverage, PIT) |
| **9** | `:4376` | **MCMC diagnostics**                          |
| **9b**| —       | **Model comparison** — `run_model_comparison`: local-level vs static arm on ELPD; opt-in via `enable_model_comparison` |
| 10 / 10b / 10c | `:4662` / `:5027` / `:5527` | Screen · CVaR risk book · analytics export |
| 10K–13 | `:5880`–`:6765` | Universe fit, single-ISIN, mingled cohort, forest (± SV twins) |
| 14    | `:7029` | Summary + recommendations                     |

There is **no §4 or §5** — `:3439` records that the legacy single-observation model was replaced by the §5b fused
panel path.

> **Split contract, know which side you are on.** `KalmanPanelInputs` (the dataclass) lives in the package at
> `pymc_models/KalmanFilterModel.py`, but its constructor `prepare_kalman_panel_inputs(...)` — along with
> `KALMAN_PANEL_RESPONSE_COLS`, `KALMAN_RESPONSE_COVERAGE_MIN`, `FeatureRoles` and `build_noise_wideners` — lives in
> `pymc_kalman_filter_pt.py:3500`. Import the preparer from the script, not the package. Consolidating the two sides
> is a known follow-up; it was left alone here because the move drags four coupled symbols across the boundary.

**Artifact export (since 0.9.9.13).** Artifacts go to `KALMAN_PT_RESULTS_DIR` in a **per-section subdirectory**
(`01_data/`, `02_eda/`, `03_features/`, `04_panel/`, `06_prior/`, `07_posterior/`, `08_ppc/`, `09_diagnostics/`,
`09b_comparison/`,
`10_screen/`, `10b_risk/`, `10c_analytics/`, `10k_universe/`, `11_single_isin/`, `11b_single_sv/`, `12_mingled/`,
`12b_mingled_sv/`, `13_forest/`, `13b_further_views/`, `14_summary/`, `14b_recommendations/`, `00_misc/`). The
directory is resolved from the artifact stem by `_export_dir_for` against the `_EXPORT_SECTION_DIRS` SSOT — do not
build result paths by hand.

- Figures → PNG (kaleido), self-contained HTML fallback.
- DataFrames → the curated bulk frames in `_SQL_EXPORT_ARTIFACTS` (`04_panel_frame`, `09_diagnostics_01_table`,
  `10_screen_results`, `10_screen_mc_summary`, `10b_risk_analytics`, `10b_risk_book`, `10c_kalman_results`) become
  `analytics."<stem>"` tables **plus** a generated `<stem>.sql` DDL file; all other frames stay CSV. `KALMAN_PT_SQL_EXPORT=0`
  — or an unreachable database — falls back to CSV while still emitting the DDL.
- DataTrees → NetCDF + per-group JSON summary.

Migrate a pre-0.9.9.13 flat results directory with `python pymc_kalman_filter_pt.py --migrate-layout` (dry run),
then `--migrate-layout --apply`. Notebooks call `enable_artifact_export()` once and `set_export_section('<step>')`
per cell (there is no enclosing `with` block per cell).

**Figure theming.** One template (`_PLOTLY_TEMPLATE = 'arviz-tumma'`, with the ArviZ 1.x `arviz-variat` rename as
fallback) applied in exactly one place — `_apply_dark_template`, called from the `_safe_show` funnel — so displayed
and exported figures cannot diverge. Reference geometry (zero lines, break-even markers, y=x guides, now-boundaries,
horizon markers) goes through `_add_ref_line` / `_add_ref_band` keyed on a role (`zero` / `anchor` / `emphasis`) from
`_REF_LINE_KINDS`; never call `add_hline` / `add_vline` / `add_vrect` directly. `_safe_show` handles **both** backends:
matplotlib figures (raw or `PlotCollection`-wrapped, resolved by `_mpl_figure_of`) go to `IPython.display` and are
closed; Plotly figures take `.show()`.

**Figure payload budget (since 0.9.9.17).** Plotly serialises every coordinate it draws into the notebook, so a panel's
data volume is a design constraint, not an afterthought. Three SSOTs govern it — a figure that ignores them is how the
v4 notebook reached **233 MB, 207.7 MB of it in a single prior-predictive figure**:

| Concern | Use | Never |
|---|---|---|
| Densities / histograms | `_binned_density_trace` / `_add_binned_density` (`density=False` for count axes) | `go.Histogram` — it ships every raw value and bins client-side (measured 87.6 MB vs 9.0 KB for 6.5 M values) |
| ECDF curves | `_ecdf_xy(values, n=_PPC_ECDF_GRID)` | one point per observation (42.2 MB vs 0.86 MB for the §8 overlay) |
| Full-universe scatters | `_decimate_frame(df, cap, by=…)` + state the sampled count in the title; caps are `_EDA_SCATTER_MAX_POINTS` / `_SCREEN_SCATTER_MAX_POINTS` | a rank-based `nlargest` cut — when it binds it deletes one tail and the surviving cloud misrepresents the screen |
| arviz-plots backend | `_azp_backend(heavy=True)` for facet grids and draw-dense panels (`PML_AZP_HEAVY_BACKEND` overrides) | a `backend='plotly'` literal |

Per-point `hover_data` identity strings (ticker / name) are the bulk of a large scatter's payload and are unreadable in
a dense cloud — leave them off. Summary statistics annotated on a decimated panel (Spearman ρ, OLS trendlines) must be
computed on the **full** frame.

Console-script entry points declared in `pyproject.toml` `[project.scripts]`:
`finance-ml`, `finance-ml-analyze`, `finance-ml-validate` (→ `cli:*`) and
`finance-ml-v4` (→ `expected_returns_v4:main`). **Note:** neither `cli.py` nor a root-level
`expected_returns_v4.py` exists yet (the v4 script lives in `archive/`), so these entry points do not currently
resolve — see the README TODOs.

### Key Notebooks

- `pymc_expected_returns_model.ipynb` — End-to-end PyMC + ArviZ
- `pymc_earnings_beat.ipynb` / `pymc_price_target_v3.ipynb` / `pymc_dcf.ipynb` — per-model PyMC workflows
- `pymc_kalman_filter_pt_v3.ipynb` — Kalman price-target panel model (notebook twin of `pymc_kalman_filter_pt.py`; KalmanRunConfig-driven, T=4 opt-in toggle, return-space forecasts. Supersedes `pymc_kalman_filter_pt_v2.ipynb`)
- `pml_model_analysis.ipynb` — Diagnostics
- `pml_workflow_v4.ipynb` — v4 pipeline (archived under `archive/`)

### Dashboards

The **GEIB** (Global Equity Investment Board) dashboard lives in the
`dashboards/geib/` package (`app.py`, `charts/`, `components/`, `data.py`,
`metrics.py`, `theme.py`). It is driven by the single analytics table
`analytics.kalman_filtered_price_targets` (DDL in
`sql_scripts/analytics/kalman_filtered_price_targets.sql`, exported by
`pymc_kalman_filter_pt.py`). Each chart module exposes a `component()` factory and
self-registers its Dash `@callback` on import. Cards include efficient frontier,
CVaR-aware Kelly sizing, Monte Carlo (+ return forecast by name), Sharpe / VaR-CVaR,
PT convergence, and high-conviction picks.

**Unit convention (since 0.9.9.7):** all persistent Kalman-pipeline frames
(`screen.results`, `RiskBook.analytics` / `.book`, the `kalman_results` export)
and `analytics.kalman_filtered_price_targets` store **raw decimal returns**
(0.25 = +25%) — including `cvar_5pct_kalman` and `expected_vol_kalman`; percent
scaling happens only at visualization / print boundaries. Per-column units are
documented via `COMMENT ON COLUMN` in the analytics DDL.
`expected_sharpe_ratio` = `er_mean / er_sd` (pooled std of the structural-TS
Monte-Carlo forward-return draws; `er_sd` is itself an exported column). Unit or
schema changes to the export must ship as a pair: re-run
`export_analytics(write=True)` **and** deploy the updated GEIB dashboard.

> **0.9.9.14 changes the exported VALUES, not the schema.** Two corrections
> compound: the de-standardisation fix removes a **+1.5 to +2.3 pp** upside
> overstatement that affected every T>1 run, and the local-level state widens the
> `expected_upside` HDIs (the previous build pseudo-replicated each name's `T`
> serially-correlated observations as iid, shrinking the per-name posterior sd by
> ~√T). The 82-column layout and the raw-decimal convention are unchanged, so no
> DDL migration is needed — but run `scripts/validate_kalman_state.py` before
> `export_analytics(write=True)`, and treat the export + dashboard deploy as the
> usual pair.

```powershell
. .\set_env.ps1   # sets DB_URL / DB_ANALYTICS_SCHEMA
python dashboards/global_equity_investment_dashboard.py
# Open http://localhost:8050  (env: GEIB_DEBUG=true, GEIB_PORT to override)
```

The monolithic `dashboards/geib_dash_app.py` is the legacy single-file version,
superseded by the `geib/` package above.

## SQL Schema — Authoritative Column & Dataframe Reference

**The SQL DDL is the single source of truth for all column names, data types, and feature definitions.** When writing
Python that references dataframe columns, derive the names from SQL — not from Python variable names or notebook
outputs.

**Which file, though — this matters.** `sql_scripts/pml/` is a pg_dump-style *extract*, not a source: 48 of its 64
files carry a `-- missing source code` body, including **all 7 `mv_pymc_*.sql`, all 4 `vw_pymc_*.sql` and all 37
function files**. They cannot recreate anything. The real SSOT files live at the repo root:

| File                            | Authoritative for                                                                                     |
|---------------------------------|--------------------------------------------------------------------------------------------------------|
| `pml_feature_catalogue.sql`     | `pml.*` helper functions · **all 7 `mv_pymc_*` MV definitions** · catalogue views · coverage check · refresh procedure |
| `pml_df_metadata.sql`           | `pml_df_metadata` / `pml_df_feature_alias` DDL + the CHECK-enforced vocabularies                        |
| `pml_df_metadata_populate.sql`  | `pymc_role` / `model_targets` / alias assignment (incl. §7i coverage reconciliation)                   |
| `sql_scripts/pml/pml_df.sql`, `staging.sql`, `vw_pml_df_*.sql` | The base tables and the five `vw_pml_df_*` views — the extract's valid part |

Recreating `pml.pml_df` cascade-drops every dependent `mv_pymc_*`, so a `pml_df` rebuild **must** be followed by
re-running `pml_feature_catalogue.sql`.

### How a column becomes a model feature

This resolution chain is what Python actually depends on, and every link can fail silently:

```
pml_df_metadata            (global pymc_role, model_targets[])
  -> pml_df_feature_alias  (per-model feature_alias AND pymc_role override)
  -> vw_pymc_feature_catalogue     COALESCE(fa.pymc_role, md.pymc_role); 'excluded' filtered out
  -> vw_pymc_feature_aliases       the arrays _resolve_<model>_feature_aliases() consumes
```

An unrecognised role neither raises nor matches a consumer — the column simply vanishes from the model's feature list
and is later zero-filled by the alignment layer. `pml_df_metadata.sql` now CHECK-constrains all three vocabularies at
write time, and `pml.assert_pymc_catalogue_coverage()` catches MV↔catalogue divergence after the fact.

### Core Tables (pml schema)

| Table                      | Purpose                                                                                                                                 |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `pml.pml_df`               | Master ~676-column denormalized equity dataframe. All numeric columns are `double precision`; identifiers are `text`; dates are `date`. |
| `pml.staging`              | Raw CSV/vendor landing zone with original vendor column names (mixed case). Mirrors `pml_df` structure. `import_pml_data.sql` filters ISIN-less vendor rows at `\copy` time and asserts none reach `pml_df`. |
| `pml.pml_df_metadata`      | Feature registry: one row per `pml_df` column with `pymc_role`, `feature_role`, `category`, `data_type`, `model_targets[]`.             |
| `pml.pml_df_feature_alias` | Per-model alias overrides: `(column_name, model_target) → feature_alias`.                                                               |

### pml_df Column Categories

**Identifiers / Coords** (used as PyMC coordinates):

```
isin, ticker, name, region, country, trading_country, exchange,
unit, sector, industry, style_class, size_class
```

**Earnings timing**:

```
income_statement_report_date, fy_end_date, next_earnings,
next_earnings_when, next_earnings_status, expected_report_date,
days_to_earnings, earnings_report_recency,
next_income_statement_report_date, next_fy_end_date
```

**Valuation**:

```
market_cap, enterprise_value, last_price,
market_cap_neg{1..4}f{q,y}, market_cap_{3,5}yavg, enterprise_value_{3,5}yavg,
price_target, price_target_low, price_target_median, price_target_high,
price_target_num, price_target_num_6m_ago, p_e_ntm, p_e_ltm, analyst_rating,
altman_z_score_{fy,fq,ltm}, beta_{1y,2y,5y}
```

**Fundamentals** — all carry `_ltm`, `_fy`, `_fq`, `_neg1fy`, `_neg1fq` variants:

```
eps_{adj,gaap,diluted}_*, ebitda_*, revenue_*, gross_profit_*,
fcf_*, cfo_*, cfi_*, cff_*, capital_expenditure_*,
roa_*, roe_*, gpm_*, ev_ebitda_*, ev_sales_*, pe_*, pb_*,
return_on_assets_roa_pct_*, asset_turnover_*, quick_ratio_*,
current_ratio_*, long_term_debt_equity_*, net_income_*
```

The last six families (levels plus `neg1..neg4` fy/fqfq lags) are the raw inputs to the `pml.piotroski_f_score()`
9-signal composite consumed by `mv_pymc_kalman_pt`.

**Time series / momentum** — lookback suffixes `_5d`, `_1w`, `_1m`, `_3m`, `_6m`, `_1y`, `_3y`, `_5y`:

```
price_{n}_ago, price_target_{n}_ago, volatility_{n},
total_return_{ytd,5y,10y}, tot_return_pct_cagr_*,
ema_{fast,slow}, w_52high_adj, w_52low_adj
```

**Analyst estimates** — suffixes `_fy1e`, `_fy2e` ... `_fy5e`, `_fq1e`, `_fq2e`:

```
eps_{adj,gaap}_est_avg_fy*, fcf_est_avg_fy*,
*_surprise_pct, *_estimate, *_actual
```

**Dividends**:

```
div_yield_{ltm,fwd}, dividend_per_share_*, dividend_streak,
dividend_record_{currency,amount,frequency,announce_date,payable_date,
record_date,ex_date}
```

### Column Naming Conventions

All column names are `snake_case`. Suffixes are standardised:

| Suffix                                                | Meaning                                         |
|-------------------------------------------------------|-------------------------------------------------|
| `_ltm`                                                | Last Twelve Months                              |
| `_fy` / `_fq`                                         | Current fiscal year / quarter                   |
| `_neg1fy` / `_neg1fq`                                 | Prior fiscal year / quarter                     |
| `_ntm`                                                | Next Twelve Months                              |
| `_fy1e` … `_fy5e`                                     | Fiscal year estimate (consensus)                |
| `_1w` / `_1m` / `_3m` / `_6m` / `_1y` / `_3y` / `_5y` | Lookback window                                 |
| `_ytd` / `_mtd` / `_qtd`                              | Year/month/quarter-to-date                      |
| `_est_avg`                                            | Analyst consensus average                       |
| `_surprise_pct`                                       | (Actual − Estimate) / \|Estimate\| × 100        |
| `pct_` prefix                                         | Percentage value                                |
| `feat_` prefix                                        | Engineered feature (only in materialized views) |
| `observed_` prefix                                    | PyMC observed/target variable (only in MVs)     |
| `n_` prefix                                           | Integer count (PyMC `constant_data`)            |

Common abbreviations: `eps`, `fcf`, `cfo`, `cfi`, `cff`, `ebitda`, `gpm`, `roa`, `roe`, `ev`, `pe`, `pb`, `dps`, `ema`,
`shrs`.

### pymc_role Enum (pml_df_metadata)

Drives all Python feature selection — do not invent new values:

| pymc_role           | Meaning                                              |
|---------------------|------------------------------------------------------|
| `coord`             | Categorical index (isin, sector, region …)           |
| `index`             | Panel time index                                     |
| `observed`          | Target / response variable                           |
| `mutable_predictor` | Trainable feature (pm.Data, updated at predict time) |
| `constant_data`     | Fixed prior / metadata (pm.ConstantData)             |
| `derived_input`     | Computed from raw columns before model entry         |
| `excluded`          | Omitted from all PyMC models                         |

Query features for a model:

```sql
SELECT column_name, feature_alias, data_type
FROM pml.vw_pymc_feature_catalogue
WHERE model_target = 'earnings_beat'
  AND pymc_role = 'mutable_predictor'
ORDER BY ordinal_position;
```

### Materialized Views — Per-Model Feature Matrices (pml schema)

Each MV is indexed on `isin` (UNIQUE). All use `feat_` prefix for engineered columns. Definitions live in
`pml_feature_catalogue.sql`. Refresh with:

```sql
CALL pml.refresh_pymc_materialized_views(
         use_concurrently => TRUE,    -- REFRESH ... CONCURRENTLY
         assert_coverage  => FALSE);  -- run pml.assert_pymc_catalogue_coverage() after
```

Both arguments are easy to miss. `assert_coverage => TRUE` fails the refresh loudly if any MV `feat_`/`observed_`/`n_`
column is unregistered, duplicated or phantom in the catalogue — see *Refreshing MVs* under Common Development Tasks.

Six of the seven MVs carry a shared market-cap/EV size-&-trend trio:
`feat_mcap_trend_1y`, `feat_mcap_vs_3yavg`, `feat_ev_vs_3yavg` (derived from the
`market_cap_neg{1..4}f{q,y}` lags and `market_cap`/`enterprise_value` `_{3,5}yavg`
columns added to `pml_df`).

> **`mv_pymc_kalman_pt` is the exception (since 0.9.9.15).** It dropped the trio,
> `feat_mv_ev_drift` and the eleven raw `market_cap_ev*` columns in favour of an EPS
> family (`feat_net_eps_drift` + `_n`, `feat_last_{q,y}_surprise`,
> `feat_eps_beat_rate{,_annual}`). The trio is *not* deleted — only the `kalman_pt`
> `model_target` was `array_remove`d (`pml_df_metadata_populate.sql` §7j), because
> `vw_pymc_feature_catalogue` is `metadata CROSS JOIN LATERAL UNNEST(model_targets)
> LEFT JOIN feature_alias`: an MV that stops emitting a still-tagged column raises
> `PHANTOM_CATALOGUE_ALIAS`. `feat_mcap_country_r` stays — it is the size-tilt
> `pm.Data` container, not a drift predictor.

> **`mv_pymc_kalman_pt` is not reproducible across refresh dates.** Its seven `days_*` horizons
> (`days_to_next_earnings`, `days_since_last_report`, …) are computed against `CURRENT_DATE`, so refreshing on a
> different day silently shifts every one. Fine for the live screen; unusable as-is for a point-in-time backtest,
> which would need an as-of date parameter. It is also part of why the `days_*` family is excluded from the drift
> matrix (`KALMAN_TIME_COVARIATE_PREFIX`).

Since 0.9.9.6 `mv_pymc_kalman_pt` replaces the raw `feat_vol_{1m,3m,6m,1y}`
columns with the winsorised realized-vol term-structure drift
`feat_vol_drift` (+ `feat_vol_drift_n` valid-pair counter), adds the analyst
rating-mix / PT-achievement features copied from `mv_pymc_price_target`, and
emits the raw observed trails (`price_{1d,mtd,ytd}_ago`,
`price_target_stddev_*_ago`, `price_target_num_6m_ago`). Several `kalman_pt`
catalogue roles are flipped via per-model overrides in
`pml.pml_df_feature_alias` (e.g. `last_price`, `feat_pt_noise_sigma` and the
`total_return_*` aliases → `observed`) — check
`pml.vw_pymc_feature_catalogue` rather than assuming the base-row role.

Since 0.9.9.9 `mv_pymc_kalman_pt` additionally computes four per-fiscal-year **Piotroski F-score** composites
(`feat_piotroski_f_score_{fy,neg1fy,neg2fy,neg3fy}`, via `pml.piotroski_f_score()` over consecutive lag pairs) plus
their median `feat_median_piotroski_f_score`. Only the median enters the fused drift design matrix as the
fundamental-quality predictor; the four components are collinear with it and are excluded via
`KALMAN_PIOTROSKI_COMPONENT_FEATURES` ⊂ `KALMAN_DRIFT_EXCLUDED_FEATURES` in
`probabilistic_ml_model/pymc_models/KalmanFilterModel.py` (`select_drift_features()` applies the SSOT partition:
leakage, noise wideners, tilt drivers, support counters, rating counts, collinear composition leg, Piotroski
components, and `days_*` time covariates all stay out of the drift matrix — EDA / analytics export only).

| MV                           | Observed column                                                                | Key `feat_` columns                                                                                                                                       |
|------------------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mv_pymc_earnings_beat`      | `n_total`, `n_beats`, `n_total_annual`, `n_beats_annual`                       | `feat_logit_beat_rate`, `feat_eps_fy1e`, `feat_rev_{1w,1m,3m,6m,1y}`, `feat_rev_accel_1m_6m`, `feat_last_q_surprise`                                      |
| `mv_pymc_price_target`       | `observed_target_pct`, `observed_target_pct_med`, `price_target`, `n_analysts` | `feat_net_buy_sentiment`, `feat_implied_upside`, `feat_target_range_width`, `feat_pt_momentum_3m`, `feat_target_dispersion_cv`, `feat_52w_range_position` |
| `mv_pymc_kalman_pt`          | `observed_pt`, `last_price`, `n_analysts`                                      | `feat_log_uplift` (the panel response), `feat_pt_drift(_n)`, `feat_price_drift(_n)`, `feat_pt_{high,low,median,noise}_drift`, `feat_coverage_drift`, `feat_pt_noise_sigma`, `feat_pt_range_norm`, `feat_vol_drift(_n)`, `feat_analyst_{bullish,bearish,neutral}_pct`, `feat_analyst_conviction`, `feat_analyst_rating`, `feat_{holds,buys,sells,no_opinion}`, `feat_pt_achievement_1y`, `feat_pt_accuracy_1y`, `feat_pt_range_hit_rate`, `feat_rel_volume`, `feat_avg_beta`, `feat_mcap_country_r`, `feat_vol_level`, `feat_log_mcap`, `feat_net_eps_drift(_n)`, `feat_last_{q,y}_surprise`, `feat_eps_beat_rate(_annual)`, `feat_one_day_return`, `feat_price_chg_pct_3m`, `feat_total_return_*` (14 windows), `feat_tr_cagr_{1y,3y,5y,10y}`, `feat_piotroski_f_score_{fy,neg1fy,neg2fy,neg3fy}`, `feat_median_piotroski_f_score`, plus raw `days_*` horizons |
| `mv_pymc_dcf_pt`             | `observed_pt`                                                                  | `feat_fcf_growth_{1y,2y}`, `feat_fcf_terminal_growth`, `feat_reinvest_rate`, `feat_capex_to_fcf`, `feat_tr_cagr_{3y,10y}`                                 |
| `mv_pymc_dividend_safety`    | `observed_div_yield`                                                           | `feat_fcf_coverage`, `feat_cfo_coverage`, `feat_eps_payout_ratio`, `feat_dps_growth_{1y,3y,5y}`, `feat_yield_spread_vs_5y`                                |
| `mv_pymc_credit_risk`        | `observed_altman_z`                                                            | `feat_distress_zone`, `feat_z_trend_{1y,3y}`, `feat_cfo_capex_cov`, `feat_fcf_yield`, `feat_beta_2y`                                                      |
| `mv_pymc_accounting_anomaly` | `observed_eps_adj`                                                             | `feat_accruals_ratio`, `feat_gpm_change_1y`, `feat_eps_adj_gap`, `feat_cfi_to_cfo`, `feat_fcfps_vs_eps_gap`                                               |

`feat_mcap_country_r = (100 - market_cap_country_r) / 100` — a **ratio where ≈0 means largest in country**. It is easy
to invert by mistake; it drives `KalmanRunConfig.mcap_country_r_max` (0.02 keeps roughly the top 2% per country).

### Metadata & Catalogue Views (pml schema)

| View                        | Purpose                                                                             |
|-----------------------------|-------------------------------------------------------------------------------------|
| `vw_pml_df_predictors`      | All `pymc_role = 'mutable_predictor'` columns                                       |
| `vw_pml_df_observed`        | All `pymc_role = 'observed'` columns                                                |
| `vw_pml_df_coords`          | All `pymc_role = 'coord'` columns                                                   |
| `vw_pml_df_derived_inputs`  | All `pymc_role = 'derived_input'` columns                                           |
| `vw_pml_df_pymc_features`   | All PyMC-relevant columns with `model_name`                                         |
| `vw_pymc_feature_catalogue` | Master 1-row-per `(model_target, pymc_role, column_name)` with alias fallback chain |
| `vw_pymc_feature_aliases`   | Aggregated alias arrays per model                                                   |
| `vw_pymc_feature_coverage`  | Diagnostic: count of columns per `(model_target, pymc_role)`                        |
| `vw_pymc_catalogue_coverage_check` | Diagnostic: per `(model_target, feat_name)` catalogue-row status (backs `assert_pymc_catalogue_coverage()`) |

### SQL Helper Functions (pml schema)

Defined in `pml_feature_catalogue.sql`. **Most** are `IMMUTABLE PARALLEL SAFE` with paired `NUMERIC` +
`DOUBLE PRECISION` overloads — but not all: `pml.calc_piotroski_f_score` is `STABLE` and single-overload (it reads
`pml_df`), as are the `country_name` / `currency_name` / `exchange_name` lookups. Check the definition before
assuming immutability in an index or generated column.

```sql
-- Arithmetic
pml.safe_divide(numerator, denominator)            -- NULLIF-safe division
pml.pct_change(current_val, previous_val)          -- (cur - prev) / prev * 100
pml.calc_change_ratio(current_val, previous_val)   -- (cur - prev) / prev
pml.target_drift(arr DOUBLE PRECISION[])           -- AVG of consecutive calc_change_ratio
pml.target_drift(arr DOUBLE PRECISION[], min_points INT)  -- same, NULL unless >= min_points pairs
pml.target_drift_n(arr DOUBLE PRECISION[]) → INT   -- count of valid consecutive pairs in target_drift
pml.signed_drift(arr DOUBLE PRECISION[])           -- as target_drift, but denominator is ABS(prev)
pml.signed_drift(arr DOUBLE PRECISION[], min_points INT)  -- sign-preserving; use for series that
                                                   -- cross zero (EPS, net income, FCF). target_drift
                                                   -- INVERTS those: -2.00 → -1.00 scores -0.5.
                                                   -- No signed_drift_n — reuse target_drift_n.

-- Transforms
pml.clamp_score(val, min DEFAULT 0, max DEFAULT 100)
pml.safe_logit(p, eps DEFAULT 1e-6)                -- LN(p / (1 - p)) with epsilon clip
pml.zscore(val, mu, sigma)
pml.winsorise(val, lo, hi)                         -- clip to [lo, hi]

-- Domain
pml.beat_counts(surprises DOUBLE PRECISION[])      -- → TABLE(n_total INT, n_beats INT)
pml.coef_var(mu, sigma)                            -- sigma / ABS(mu)
pml.fcf_dividend_coverage(fcf, dividends_paid)
pml.altman_zone(z) → INT                           -- 1=distress (<1.81), 2=grey, 3=safe
pml.accruals_ratio(ni, cfo, scale)                 -- (ni - cfo) / scale
pml.piotroski_f_score(roa, roa_prev, cfo, ni, ltde, ltde_prev, cr, cr_prev,
                      shrs, shrs_prev, gpm, gpm_prev, at, at_prev) → INT
                                                   -- 9-signal 0-9 composite; NULL comparisons score 0
pml.calc_piotroski_f_score(p_isin DEFAULT NULL)    -- → TABLE(isin, piotroski_f_score): LTM screener wrapper

-- Date / fiscal
pml.frequency_to_months(frequency TEXT, fy_end_date, next_fy_end_date) → INT
pml.calculate_next_fiscal_quarter(next_earnings_date, ...) → INT    -- returns 1-4
pml.calculate_next_fiscal_quarter_date(income_statement_report_date) → DATE  -- +3 months
pml.ema_crossover_signal(fast_ema, slow_ema) → INT  -- 1 / -1 / 0

-- Date / fiscal (continued)
pml.calculate_fiscal_info(reference_date, fy_end_date, frequency DEFAULT NULL)
                        → record  -- OUT fiscal_month/quarter/year, next_quarter(_year),
                                  --     reporting_interval, earnings_report_frequency,
                                  --     next_earnings_report_type
pml.calculate_expected_report_date(period_end_date, earnings_report_frequency) → DATE
pml.calculate_next_income_statement_report_date(report_date, frequency) → DATE
pml.calculate_next_fy_end_date(fy_end_date) → DATE
pml.calculate_reporting_lag(next_earnings, report_date, frequency DEFAULT 'Quarterly') → INT
pml.get_expected_reporting_lag_days(earnings_report_frequency) → INT
pml.derive_earnings_report_frequency(report_date, fy_end_date) → TEXT
pml.months_to_frequency(interval_months) → TEXT
pml.month_abbrev_to_number(month_abbrev) → INT
pml.parse_fiscal_year_end_date(fy_end_text) → DATE
pml.validate_fiscal_dates(fy_end_date, report_date, reference_date DEFAULT CURRENT_DATE)
                        → TABLE(issue TEXT, severity TEXT)

-- Parsing / lookup (STABLE, single-overload)
pml.text_to_date_safe(input_text, date_format DEFAULT 'AUTO') → DATE
pml.text_to_numeric_safe(input_text) → NUMERIC
pml.country_name(code_text) / pml.currency_name(code) / pml.exchange_name(code) → TEXT

-- Catalogue integrity
pml.assert_pymc_catalogue_coverage() → VOID   -- RAISES on any MV <-> catalogue divergence
CALL pml.refresh_pymc_materialized_views(use_concurrently DEFAULT TRUE,
                                         assert_coverage  DEFAULT FALSE);
```

### Analytics Schema (pipeline outputs)

The Kalman workflow writes seven curated frames (`_SQL_EXPORT_ARTIFACTS`), each landing as an `analytics."<stem>"`
table **and** a generated `sql_scripts/analytics/<stem>.sql` DDL file:

| Stem                      | Notable columns                                                                       |
|---------------------------|----------------------------------------------------------------------------------------|
| `04_panel_frame`          | the full 185-column model input frame — the de-facto reference for the Kalman MV surface |
| `09_diagnostics_01_table` | `mean`, `sd`, `eti89_lb/ub`, `ess_bulk`, `ess_tail`, `r_hat`, `mcse_mean`, `mcse_sd`   |
| `10_screen_results`       | per-ISIN screen: identity/geo → `observed_pt`, `expected_pt(_hdi_lo/hi)`, `expected_upside`, `risk_adj_return`, `prob_pos` |
| `10_screen_mc_summary`    | `isin`, `er_mean`, `er_sd`, `er_p05`, `er_p50`, `er_p95`, `prob_pos`                   |
| `10b_risk_analytics` / `10b_risk_book` | + `p_upside_pos(_cond)`, `band_width`, `kalman_gain`, `cvar05`, `exp_vol`, `ret_vol_ratio`, `expected_sharpe`, `tail_risk`, `starr`, `book_weight`, `weight` |
| `10c_kalman_results`      | feeds `analytics.kalman_filtered_price_targets`                                        |

`analytics.kalman_filtered_price_targets` (**102 columns** — 100 plus the `run_id` / `exported_at` provenance pair) is the GEIB dashboard's only source and the **only** file in
`sql_scripts/analytics/` carrying `COMMENT ON COLUMN` documentation and the raw-decimal unit header — keep it that way.
The other ~45 files there are hand-written screen/analysis scripts unmanaged by the pipeline.

## Key Architectural Patterns

### 1. Single Source of Truth (SSOT)

- Features: `feature_catalog.py` synced with SQL registry
- Hierarchy: `_hierarchy.py` shared by all models
- Identifiers: `DEFAULT_IDENTIFIER_COLUMNS` in feature_catalog.py
- Schema: `pml.pml_df_metadata` (DDL `pml_df_metadata.sql`, population `pml_df_metadata_populate.sql`)
- Functions / MVs / catalogue views: `pml_feature_catalogue.sql`
- Sampling & diagnostics: `pymc_models/_workflow.py`
- Drift-feature exclusions: `KALMAN_DRIFT_EXCLUDED_FEATURES` in `KalmanFilterModel.py` — 16 surviving columns
  (cond 19.7, max VIF 4.25). Exclusions live HERE, never as a `pymc_role='excluded'` flip in SQL: that drops the row
  from `vw_pymc_feature_catalogue` while the MV still emits the column, so `assert_pymc_catalogue_coverage()` raises
  MISSING_FROM_CATALOGUE. Collapsed families keep one representative each (`feat_pt_drift`, `feat_analyst_rating`).
  **Excluding ≠ removing:** the price-derived market-cap/EV four left the MV *and* the catalogue in 0.9.9.15, so they
  carry no exclusion entry; only `feat_net_eps_drift_n` (a support counter) was added to the union.
- Observation-scale model: `sigma_isin` in `build_fused_kalman_pt_model` is **log-linear** since 0.9.9.16, and it is
  the only place the measurement scale is defined:

  ```
  log sigma_isin = log sigma_base + log1p(cv)
                 + delta_vol*z(log1p(feat_vol_level)) + delta_mcap*z(feat_log_mcap)
                 + delta_range*z(feat_pt_range_norm)
                 - sigma_n_exponent*log(precision_weight) + sector_offset
  ```

  `build_noise_wideners` is **not** a second source of truth despite its former docstring: its `multiplier` key is an
  EDA display quantity the likelihood has never applied. `feat_vol_drift` is a provenance container only —
  it correlates −0.035 with `log|residual|` against +0.19 for the `feat_vol_level` that 0.9.9.6 removed.
- Artifact sections: `_EXPORT_SECTION_DIRS` in `pymc_kalman_filter_pt.py`
- Export provenance: `PROVENANCE_COLUMNS` / `stamp_export_provenance` / `check_export_vintage` in
  `pymc_kalman_filter_pt.py`. Every `_SQL_EXPORT_ARTIFACTS` frame and the analytics table carry `run_id` /
  `exported_at`. Resolve which tables are stamped from `information_schema` **first** — a speculative
  `SELECT run_id` aborts the PostgreSQL transaction and poisons every later query on that connection.
- Reward/risk ratio floor: `MIN_RATIO_DENOMINATOR` in `RiskBookModel.py`. A bare `> 0` guard is not enough; a
  denormal `er_sd` passes it and publishes a 1e15 ratio.
- Kalman decision latent: `KALMAN_SCREEN_LATENT` / `resolve_screen_latent` in `pymc_kalman_filter_pt.py` — every
  consumer (screen, price-target MC, risk book, analytics export, §13b plots, prior predictive) resolves the
  per-ISIN latent through it, so the decision quantity has exactly one name.
- Optional second response series: `KALMAN_PANEL_RESPONSE_EXTRA` in `pymc_kalman_filter_pt.py`
- Reference geometry: `_REF_LINE_KINDS` (`_add_ref_line` / `_add_ref_band`)

### 2. Lazy Loading

`probabilistic_ml_model/__init__.py` uses `__getattr__` to defer PyMC/ArviZ imports.

### 3. Type-Aware Feature Coercion

Models call `coerce_by_data_type()` to match declared types (pct, ratio, level, score, flag).

### 4. ArviZ InferenceData / DataTree as Output

Every PyMC model returns `InferenceLike` (`arviz.InferenceData | xarray.DataTree`) with:

- posterior
- constant_data (features + provenance attrs)
- diagnostics (R-hat, ESS, trace)

Import the type alias from the project compat shim, not directly from arviz:

```python
from probabilistic_ml_model._pymc_arviz_compat import InferenceLike
```

### 5. Parallel Data Loading

`load_all_feature_views()` uses joblib.Parallel.

### 6. Configuration as Dataclass

`PipelineConfig` and `KalmanRunConfig` centralize magic numbers for CLI/env override. Both expose `from_env()`; both
are overridden programmatically with `dataclasses.replace(...)` rather than by mutation (`KalmanRunConfig` is frozen).

### 7. Workflow Stage = Export Section

The artifact tree is the workflow. Resolve every result path through `_export_dir_for` against the
`_EXPORT_SECTION_DIRS` SSOT; scripts use `with export_section('08_ppc'):`, notebooks call
`enable_artifact_export()` once then `set_export_section('<step>')` per cell. Never build a result path by hand.

### 8. Diagnostics as Code Gates

Fit quality is *self-reported*, never scraped from console output. `log_sample_diagnostics()` warns on divergences
and on bulk-ESS below `MIN_ESS_GATE = 400`; `build_sample_kwargs()` warns when the effective chain count is `< 2`,
because r-hat and between-chain ESS are undefined for a single chain and come back NaN.

### 9. Layered Sample Kwargs

`build_sample_kwargs()` composes in a load-bearing order:

```
defaults (incl. compile_kwargs) -> nuts_sampler/cores -> setdefault(idata_kwargs)
  -> update(sample_kwargs)      -> nutpie idata_kwargs strip
```

`setdefault` before `update` is what lets a caller override `log_likelihood`; the nutpie strip afterwards is what
silently defeats it on the default sampler. Do not reorder — and prefer `attach_log_likelihood()` (see the Bayesian
Workflow section).

## Code Guidelines

These patterns are derived from the actual `probabilistic_ml_model/` source and must be followed consistently.

### Lazy Imports

All `pymc`, `arviz`, and `pytensor` imports are deferred via `__getattr__` in `__init__.py`. Never import them at module
level — use the lazy registry instead:

```python
# pymc_models/__init__.py pattern
_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "MyModel": (".MyModelModule", "MyModel"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORT_MAP:
        mod_path, attr_name = _LAZY_IMPORT_MAP[name]
        module = _importlib.import_module(mod_path, __package__)
        obj = getattr(module, attr_name)
        globals()[name] = obj  # cache after first access
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Inside model files, guard optional deps:

```python
try:
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:
    pm = None  # type: ignore[assignment]
    pt = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import pymc as pm_typing
```

Raise a clear error if the dep is missing when `fit()` is called:

```python
if pm is None:
    raise ImportError("PyMC is not available. Install pymc to use MyModel.")
```

### Type Annotations

- Use `InferenceLike` (from `_pymc_arviz_compat`) for all inference result types — never `arviz.InferenceData` directly.
- Use `Optional[Type]` not `Type | None`.
- Use lowercase generics: `dict[str, Type]`, `list[str]`, `tuple[str, ...]`, `frozenset[str]`.
- Reference pymc/arviz types only inside `TYPE_CHECKING` blocks (`pm_typing.Model`, `az_typing.InferenceData`).

### Model `fit()` Signature

All PyMC model `fit()` methods follow this parameter order:

```python
def fit(
        self,
        # 1. Core data arrays (positional)
        z_scores: np.ndarray,
        isins: np.ndarray,
        # 2. Optional categorical / hierarchical data
        categories_df: Optional[pd.DataFrame] = None,
        hierarchy_levels: Optional[list[str]] = None,
        # 3. MCMC sampling params (always include all of these)
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
        nuts_sampler: Optional[str] = None,  # e.g. "nutpie", "blackjax", "numpyro"
        **sample_kwargs: Any,
) -> tuple[InferenceLike, "pm_typing.Model"]:
```

A `fit()` **must** carry a return annotation. Known deviations from the defaults above:
`EarningsBeatBayesian` uses `samples=500`, and `cores` appears only on `PriceTargetAchievement` / `DCFPriceTarget`.

**Never re-copy the sampling boilerplate.** Build the kwargs with the shared helper, then self-report quality:

```python
from probabilistic_ml_model.pymc_models._workflow import (
    build_sample_kwargs, log_sample_diagnostics,
)

idata = pm.sample(
    **build_sample_kwargs(
        samples=samples, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=random_seed,
        nuts_sampler=nuts_sampler, sample_kwargs=sample_kwargs,
        model_name="MyModelBayesian",
    )
)
log_sample_diagnostics(idata, model_name="MyModelBayesian")
```

`build_sample_kwargs` already supplies `compile_kwargs=get_pytensor_compile_kwargs()`, the `log_likelihood` policy,
the nutpie `idata_kwargs` strip and the `chains < 2` warning. Hand-rolling the block is how five of the eight
modules ended up missing the nutpie strip (and emitting a spurious `UserWarning`) before it was centralized.

### PyMC Model Building

Use `pm.Data()` containers for all observed data:

```python
with pm.Model(coords=coords) as model:
    x_data = pm.Data("x_data", x_array, dims="isin")
```

Hierarchical shrinkage uses these three helpers from `_hierarchy.py`:

```python
from probabilistic_ml_model.pymc_models._hierarchy import (
    build_hierarchy_indices, build_nested_logit_normal_rates, coerce_categories,
)

cats_df, levels = coerce_categories(isins, categories_df=categories_df, hierarchy_levels=hierarchy_levels)
if cats_df is not None:
    meta = build_hierarchy_indices(cats_df, isins, levels=levels)
    nested = build_nested_logit_normal_rates(meta, leaf_dim="isin", name="rate")
    leaf_rate = nested["leaf_rate"]
```

### Naming Conventions

| Kind                        | Style              | Example                                    |
|-----------------------------|--------------------|--------------------------------------------|
| Classes                     | `CapitalCamelCase` | `CreditRiskBayesian`                       |
| Functions / methods         | `snake_case`       | `build_hierarchy_indices`                  |
| Private modules / functions | `_snake_case`      | `_hierarchy.py`, `_DEFAULT_SAMPLES`        |
| Constants                   | `UPPER_SNAKE_CASE` | `HIERARCHICAL_CATEGORY_COLS`, `PARENT_MAP` |
| Module-level defaults       | `_DEFAULT_*`       | `_DEFAULT_CHAINS = 4`                      |

### Docstrings

Use **NumPy docstring format** (Parameters / Returns / Raises / Notes):

```python
def build_hierarchy_indices(
        df: pd.DataFrame,
        isins: np.ndarray,
        levels: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Return per-level metadata for nested hierarchical shrinkage.

    Parameters
    ----------
    df
        Categorical frame indexed by ISIN.
    isins
        ISIN array used to align ``df`` rows.
    levels
        Subset of :data:`HIERARCHICAL_CATEGORY_COLS` to materialise.

    Returns
    -------
    dict[str, dict[str, Any]]
        Nested structure per level.
    """
```

### Error Handling & Logging

```python
logger = logging.getLogger(__name__)

# Validation errors: use !r and descriptive messages
raise ValueError(f"Unknown hierarchy levels {unknown!r}. Valid: {HIERARCHICAL_CATEGORY_COLS}")

# Logging: use %-style, not f-strings
logger.warning("Could not load feature categories: %s", exc)
logger.info("Catalog loaded: %d categories, %d views", len(cats), len(views))
```

### Global Singletons & Caching

Use the lazy-singleton + reset-helper pattern for module-level state:

```python
_catalog_instance: FeatureViewCatalog | None = None


def get_feature_catalog(...) -> FeatureViewCatalog:
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = FeatureViewCatalog()
        _catalog_instance.load_from_db(...)
    return _catalog_instance


def reset_feature_catalog() -> None:
    global _catalog_instance
    _catalog_instance = None
```

Use `@lru_cache` for repeated DB lookups:

```python
@lru_cache(maxsize=4)
def _resolve_feature_aliases(connection_string: Optional[str] = None) -> tuple[str, ...]:
    ...
```

Use `threading.Lock` (via `field(default_factory=threading.Lock, repr=False, compare=False)`) when a dataclass may be
accessed from multiple threads.

### Dataclasses

- Registry/config entries that should be immutable: `@dataclass(frozen=True)`
- Mutable state containers: `@dataclass` with `field(default_factory=...)` for mutable defaults
- Always expose a `from_env()` classmethod on config dataclasses

## Common Development Tasks

### Adding a New PyMC Model

Follow the nine-point checklist in **The Bayesian Workflow — Stage Contract**. Mechanically:

1. Create `probabilistic_ml_model/pymc_models/MyModelName.py`:
    - `fit(...)` with optional `categories_df` + `hierarchy_levels`, and a return annotation
    - Features from `pml.vw_pymc_feature_catalogue` via `_resolve_*_feature_aliases()`; `coerce_by_data_type()`
    - `build_sample_kwargs()` → `pm.sample()` → `log_sample_diagnostics()`
    - `stamp_feature_provenance(...)` after sampling
    - A `prior_predictive_check()` and a `posterior_predictive_check()` with ≥1 calibration statistic
    - Return `(idata, model)`

2. Update `pymc_models/__init__.py`:
   ```python
   _LAZY_IMPORT_MAP["MyModel"] = (".pymc_models.MyModuleName", "MyModel")
   ```

3. Register the model's columns in SQL (next task) — the catalogue, not Python, decides its feature list.

### Enabling Model Comparison for a Fit

```python
from probabilistic_ml_model.pymc_models._workflow import attach_log_likelihood
import arviz as az

idata_a, model_a = ModelA().fit(...)
idata_b, model_b = ModelB().fit(...)
attach_log_likelihood(idata_a, model_a)      # post-hoc; works under nutpie
attach_log_likelihood(idata_b, model_b)
az.compare({"a": idata_a, "b": idata_b})
```

Do **not** rely on `fit(..., idata_kwargs={"log_likelihood": True})` alone — nutpie discards it (see the Bayesian
Workflow section). Cost scales with `chains × draws × n_observations` and can exceed the fit itself, so keep it opt-in.

### Auditing a Model Against the Bayesian Workflow

1. Find its row in the coverage matrix; every ❌ is the work.
2. Run its prior predictive and check the implied observations are on a plausible scale.
3. Confirm `log_sample_diagnostics` output appears in the logs (divergences, bulk ESS vs `MIN_ESS_GATE`).
4. Confirm the PPC ships at least one calibration statistic, not just a density overlay.

### Adding a New Feature View or MV Column

1. Add column to `pml.pml_df` DDL and `pml.staging` (same column, same `double precision` type).
2. Register in `pml.pml_df_metadata` via `pml_df_metadata_populate.sql`: `column_name`, `pymc_role`, `feature_role`,
   `category`, `data_type`, `model_targets[]`. All three vocabularies are CHECK-constrained — an invalid value now
   fails the insert instead of silently dropping the column.
3. If it's a PyMC feature, add the `feat_` column to the relevant `pml.mv_pymc_*` definition in
   **`pml_feature_catalogue.sql`** (not `sql_scripts/pml/`, whose MV files are stubs) using a `pml.*` helper.
4. The MV column name and the catalogue `feature_alias` **must match exactly**, or the coverage check flags it.
5. Refresh and verify: `CALL pml.refresh_pymc_materialized_views(assert_coverage => TRUE);`
6. Update `FEATURE_VIEW_REGISTRY` in `feature_catalog.py` to match.
7. Column name in Python must exactly match the SQL column name (no renaming at the Python layer).

### Refreshing MVs and Checking Catalogue Coverage

```sql
CALL pml.refresh_pymc_materialized_views(use_concurrently => TRUE, assert_coverage => TRUE);

-- If it raises, enumerate what diverged:
SELECT model_target, status, count(*),
       string_agg(feat_name, ', ' ORDER BY feat_name)
FROM pml.vw_pymc_catalogue_coverage_check
WHERE status <> 'OK' GROUP BY 1, 2 ORDER BY 1, 2;
```

Three failure classes, all fixed in `pml_df_metadata_populate.sql` §7i:

| Status                      | Meaning                                            | Fix                                                        |
|-----------------------------|----------------------------------------------------|-------------------------------------------------------------|
| `MISSING_FROM_CATALOGUE`    | MV emits it, catalogue doesn't know it — **the dangerous one**: the model reindexes the column to 0.0 | Add the catalogue row / extend `model_targets` |
| `PHANTOM_CATALOGUE_ALIAS`   | Catalogue claims an alias the MV never emits       | Drop the stray model tag, or add the column to the MV      |
| `DUPLICATE_CATALOGUE_ALIAS` | Two rows claim one alias — makes alias resolution order-dependent | Remove the non-canonical row                |

`assert_coverage` still defaults to `FALSE`; flip it to `TRUE` in `pml_feature_catalogue.sql` once your database has
the §7i reconciliation applied and the check returns clean.

### Modifying the Hierarchy

1. Update `HIERARCHICAL_CATEGORY_COLS` and `PARENT_MAP` in _hierarchy.py
2. Ensure no cycles in PARENT_MAP
3. Re-run model `fit()` calls with new hierarchy_levels

### Running Tests

```powershell
pytest tests/test_pml_workflow_v4.py -v
pytest tests/test_pml_workflow_v4.py::TestClass::test_method -v
pytest --cov=probabilistic_ml_model --cov-report=term-missing tests/
```

### Cutting a Release

1. Add a new dated section at the top of `CHANGELOG.md` (Keep a Changelog + SemVer format).
2. Bump `version` in `pyproject.toml` and the README badge to match (these have historically lagged the CHANGELOG — see
   the recurring follow-up note in each release entry).
3. Sync dependency pins across `pyproject.toml`, `Pipfile`, and `requirements.txt` if they changed.
4. Re-run `pipenv lock` / `pip-compile` to regenerate lockfiles against the aligned version windows.

## Debugging & Troubleshooting

### PyTensor Compilation (Windows)

PyMC 6.2 + PyTensor 3.2 uses **numba** as the default backend (via nutpie). C++ compilation (`cxx`) is no longer the
primary backend and is not required for normal operation.

**The C backend is disabled project-wide by default.** Importing `probabilistic_ml_model` runs
`probabilistic_ml_model/_pytensor_env.py` (`force_python_vm()`), which normalises `PYTENSOR_FLAGS` to `cxx=` **before
any
`pytensor` import** — *stripping* any inherited `cxx=<g++ path>` (a naive `"cxx=" in flags` check is defeated by a
persistent `PYTENSOR_FLAGS=...,cxx=<g++>` User env var). The silent failure mode (**empty stderr, `status=1`** on a
fresh compile) was root-caused 2026-07-10: the `g++.exe` driver loads its DLLs from its own directory, but the actual
compiler `cc1plus.exe` lives in `ucrt64\lib\gcc\...` and resolves its DLLs (libgmp, libmpfr, libwinpthread, zstd, …)
via `PATH`. If `PATH` lacks the `C:\msys64\ucrt64\bin` **directory** (the System PATH had the g++.exe **file** path as
an entry instead), `cc1plus` dies on startup with `STATUS_DLL_NOT_FOUND` (0xC0000135) and zero output — while
`g++ --version` (driver only) still works. Symptoms are `CompileError: Compilation failed (return status=1)` with no
diagnostics and NUTS samplers failing at "Compiling new CVM". Earlier apparent successes were served from PyTensor's
compile **cache**. With `C:\msys64\ucrt64\bin` on `PATH`, the UCRT64 toolchain (runtime-matched to MSVC-built CPython)
is verified working on Python 3.14.6 + g++ 15.2.0 against a fresh compile cache.

Opt back into the C backend by setting `PML_ENABLE_PYTENSOR_C=1` **before** import, and ensure
`C:\msys64\ucrt64\bin` is first on `PATH` (`set_env.ps1` does both and sanity-compiles a probe before enabling;
without the flag it falls back to the pure-Python/numba VM). Model
`fit()` / sampling paths should also pass `compile_kwargs=get_pytensor_compile_kwargs()` (from
`probabilistic_ml_model.pymc_models._pytensor_compat`), which forces the `Mode(linker="py")` VM at the call site.

> **`environment_variables.txt` ships with the C backend OFF (`PML_ENABLE_PYTENSOR_C=0`, `cxx=`) as of
> 2026-08-07.** It previously shipped `=1` with the UCRT64 `cxx` path, and — because it is the `ENV_FILES` entry of
> the PyCharm run configurations (`pymc_kalman_filter_pt`, `expected_returns_v3`,
> `global_equity_investment_dashboard`, see `.idea/workspace.xml`) — that re-armed the C backend on every IDE run no
> matter what `set_env.ps1` did in a terminal. The PATH prerequisite had meanwhile been lost: `C:\msys64\ucrt64\bin`
> was absent from the process, User **and** Machine `PATH`, so `cc1plus` could not load its DLLs and
> `g++ -march=native` failed, producing
> `INFO pytensor.link.c.cmodule: Call to 'g++ -march=native' failed` followed by a `CompileError` on the first
> un-versioned op (`ExpandDims{axis=0}`) and an empty-stderr `pytensor_compilation_error_*` file in `%TEMP%`.
> **Flip the env file, not just the shell** — and re-add the `ucrt64\bin` directory to `PATH` before ever setting it
> back to `1`. Note that a cached compile can mask a broken toolchain: re-testing an op that PyTensor already has in
> its compiledir succeeds without invoking `g++` at all.

> **This opt-in was inert before 2026-08-06.** `_pytensor_env.py:27` defined
> `ENABLE_C_ENV_VAR = "PYTENSOR_FLAGS"`, so the guard compared the *flag string* against `"1"` and never matched:
> importing the package stripped a perfectly valid `cxx=<g++>` on every run, and no amount of `set_env.ps1`
> configuration could re-enable the C backend. Fixed to `"PML_ENABLE_PYTENSOR_C"`. Verify with:
> ```powershell
> python -c "import probabilistic_ml_model, os; print(os.environ['PYTENSOR_FLAGS'])"
> ```
> With the flag set the `cxx=` path must survive; with it unset the value must end in a bare `cxx=`.

**Use FORWARD slashes in the `cxx` path.** PyTensor parses `PYTENSOR_FLAGS` with posix shlex, which treats `\` as an
escape character and strips it — `C:\msys64\ucrt64\bin\g++.exe` is mangled into the non-existent
`C:msys64ucrt64bing++.exe`. Forward slashes survive the parser and g++/Windows accept them. (`device=` is also
dropped: PyTensor does not recognise it and defaults to CPU.)

If you see `FileNotFoundError: cxx not found` from a legacy code path:

```powershell
$env:PYTENSOR_FLAGS = "floatX=float64,cxx=C:/msys64/ucrt64/bin/g++.exe"
```

Disable C++ compilation entirely (forces pure Python / numba path):

```powershell
$env:PYTENSOR_FLAGS = "floatX=float64,cxx="
```

For JAX backend (blackjax / numpyro samplers):

```powershell
$env:JAX_PLATFORM_NAME = "cpu"   # or "gpu" if CUDA available
```

### Sampling & Diagnostics Gotchas

| Symptom                                                        | Cause / fix                                                                                                   |
|----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `az.loo` / `az.waic` / `az.compare` raise "log likelihood not found" | Expected: `log_likelihood` is off project-wide. Call `attach_log_likelihood(idata, model)` — see the Bayesian Workflow section. |
| `az.loo(...)` result reads as `nan`                            | You read `.elpd_loo`. ArviZ 1.x renamed the accessor to **`.elpd`**; the old name is absent, so `getattr(loo, "elpd_loo", nan)` silently yields `nan`. The `repr` still prints the `elpd_loo` label — don't be fooled by it. |
| `idata_kwargs` you passed had no effect                        | nutpie (the default sampler) ignores `idata_kwargs`, and `build_sample_kwargs` strips it to suppress the resulting `UserWarning`. Use `nuts_sampler="pymc"`, or the post-hoc helper. |
| `r_hat` / `ess_bulk` come back `NaN`                           | Fewer than 2 chains. `cores=1` is fine (chains run sequentially); `chains=1` is not — both statistics are between-chain. `build_sample_kwargs` warns. |
| Jupyter kernel dies with "Connection to IDE-Managed Server is lost" | nutpie's parallel native workers crash an IDE-embedded kernel on Windows. Keep `cores=1` in notebooks; raise it only on the CLI path. |
| A `visualizations` function vanished from `__all__`            | `visualizations/__init__.py:60-176` swallows `ImportError`, so a broken submodule silently disappears (exactly how a Python-2 `except` in `_shared.py` hid until 0.9.9.13). Import the submodule directly to see the real traceback. `earnings_quality.py` is never registered — always import it directly. |
| `nu` sits pinned at its 2.5 floor                              | It is absorbing scale mis-specification, not measuring tail weight. Check `sigma_isin` before touching `nu` — **never relax the floor** (`KalmanFilterModel.py` records the improper-density corner it removes). On 2026-08-16, fixing the scale model and `isin_level_scale` lifted it 2.53 → 7.50 with no change to the likelihood family. |
| PPC replicates over-disperse (T=std fails, coverage > target)   | Almost always the scale, not the likelihood. Diagnose by fitting a pooled OLS with shared slopes and free per-time intercepts — the exact structure the model imposes — and compare its BETWEEN-name and WITHIN-name residual sd against `isin_level_scale` and `sigma_isin`. Tightening the response clip and switching to a mixture likelihood were both tried in 0.9.9.16 and neither helped. |
| A redirected run dies mid-way with `UnicodeEncodeError`         | Windows stdout falls back to cp1252 when piped to a file, and `run_eda` prints `Spearman ρ` (U+03C1). **`export PYTHONIOENCODING=utf-8`** before any redirected run. cp1252 encodes the em-dashes and arrows fine, so most output survives and only the Greek glyphs raise — after minutes of figure rendering. `validate_kalman_state.py` prints none, so it redirects cleanly and gives false reassurance. |
| An MV "has 0 columns" per `information_schema`                  | PostgreSQL does not list **materialized view** columns in `information_schema.columns`. Use `pg_attribute` joined to `pg_class`/`pg_namespace`, or just `SELECT` the column. |

### Database Connection

```python
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DB_URL"])
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.fetchone())
```

**`psql` is not a shortcut here.** It cannot parse the SQLAlchemy-style `DB_URL`
(`postgresql+psycopg2://…`) and will hang on a password prompt unless `PGPASSWORD` is set. For ad-hoc queries use a
one-off SQLAlchemy script like the above rather than fighting the CLI.

### Feature Catalog Cache

```python
from probabilistic_ml_model.data_utils.feature_catalog import get_feature_catalog

catalog = get_feature_catalog(force_reload=True)
```

## References

- **README.md** — Full feature list, setup, CLI entry points
- **docs/** — Architecture guides (PyMC, ArviZ 1.0, SQL)
- **CHANGELOG.md** — Release notes

---

**Version:** 0.9.9.17 (CHANGELOG; `pyproject.toml` and README badge lag at 0.9.9.5 pending the next packaging bump) |
**Python:** 3.12–3.14 | **PyMC:** >=6.2,<7 | **PyTensor:** >=3.2.2,<4 | **ArviZ:** >=1.1,<2 (arviz-base + arviz-stats +
arviz-plots) | **JAX:** >=0.11,<1 | **License:** MIT | **DB:** PostgreSQL