# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PML Finance Project** is a comprehensive platform for probabilistic equity screening, feature engineering, and machine learning modeling for financial markets. The core system implements an 8-phase workflow combining Bayesian models (via PyMC 6.0), statistical analysis, and portfolio optimization.

**Key Technologies:**
- Python 3.12–3.14
- **PyMC 6.0** + **PyTensor 3.0** + **ArviZ 1.0** — Bayesian inference and diagnostics
  - ArviZ 1.x ships as three packages: `arviz-base` (data containers), `arviz-stats` (diagnostics), `arviz-plots` (visualization). The top-level `arviz` meta-package re-exports all three for backward-compatible imports.
  - ArviZ 1.x replaces `arviz.InferenceData` with `xarray.DataTree` as the canonical output type. Use the `InferenceLike` alias (defined in `probabilistic_ml_model/_pymc_arviz_compat.py`) for type annotations: `Union[arviz.InferenceData, xarray.DataTree]`.
  - **nutpie 0.14+** — default high-performance sampler for PyMC 6.0 (numba backend)
  - **JAX 0.4.30+ / jaxlib 0.4.30+**, **blackjax 1.2+**, **numpyro 0.16+** — alternative JAX-based samplers
  - **bambi 0.16+** — formula-based GLM interface on top of PyMC 6.0
- PostgreSQL — centralized data storage with 17 feature views
- pandas/NumPy/SciPy — data processing
- scikit-learn, XGBoost, LightGBM, CatBoost — classical ML
- Plotly, Matplotlib, Seaborn — visualization
- Streamlit (Python < 3.14 only), Dash — interactive dashboards
- pytest — 483 test cases across 23 test modules

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

Key environment variables (see `environment_variables.txt`):
- `DB_URL` — PostgreSQL connection
- `PYTENSOR_FLAGS` — PyTensor backend (Windows: set C++ compiler path)
- `LOG_LEVEL` — Python logging level
- `OUTPUT_DIR` — analytics artifact directory

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
- `public.mv_equities` — core equity metadata
- `public.vw_features_*` (17 views) — categorical feature groups
- `public.calculated_features_registry` — feature → category mappings

### 2. Hierarchical Shrinkage (pymc_models/_hierarchy.py)

Canonical multi-level category hierarchy shared by all 7 PyMC models:

```python
from probabilistic_ml_model import (
    HIERARCHICAL_CATEGORY_COLS,
    PARENT_MAP,  # region → country → exchange → sector → industry
    build_hierarchy_indices,
)

idata, _ = model.fit(
    data,
    categories_df=df_categories,
    hierarchy_levels=["exchange", "sector", "industry"],
)
```

### 3. Feature Alignment & ArviZ (pymc_models/_feature_alignment.py)

Column names passed to `coerce_by_data_type()` must match `pml.pml_df` column names exactly — derive them from `pml.vw_pymc_feature_catalogue` (SQL), not from Python variable names. Every model stamps feature provenance onto `idata.constant_data`:

```python
from probabilistic_ml_model import (
    coerce_by_data_type,
    stamp_feature_provenance,
    load_feature_metadata_from_db,
)
```

### 4. PyMC Models (pymc_models/)

7 Bayesian models with unified interface:

| Model | Purpose |
|-------|---------|
| EarningsBeatBayesian | Beat probability |
| PriceTargetAchievement | Return expectation |
| KalmanFilterPriceTarget | Smoothed signals |
| DCFPriceTarget | Fair-value bands |
| DividendSafetyBayesian | Cut probability |
| CreditRiskBayesian | Distress risk |
| AccountingAnomalyBayesian | Quality flags |

Each returns `InferenceLike` (i.e. `arviz.InferenceData | xarray.DataTree`) with posterior, constant_data (features + provenance attrs), and diagnostics. Use the compat shim in `_pymc_arviz_compat.py` for type annotations instead of importing `arviz.InferenceData` directly, as ArviZ 1.x uses `xarray.DataTree` internally.

### 5. Pipeline Runner (pipeline_runners.py)

Orchestrates all 8 phases via PipelineConfig:

```python
@dataclass
class PipelineConfig:
    mc_simulations: int = 10_000
    mcmc_chains: int = 8
    use_bayesian_model_averaging: bool = True
    # 20+ more tunable parameters
```


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

## Entry Points & Workflows

### Main Pipeline

```powershell
python expected_returns_v3.py
```

8 phases: data load → models → ensemble → MCMC → viz → export

### Key Notebooks

- `pymc_expected_returns_model.ipynb` — End-to-end PyMC + ArviZ
- `pml_workflow_v4.ipynb` — v4 pipeline
- `pml_model_analysis.ipynb` — Diagnostics

### Dashboards

```powershell
python dashboards/geib_dash_app.py
# Open http://localhost:8050
```


## SQL Schema — Authoritative Column & Dataframe Reference

**The SQL DDL files in `sql_scripts/pml/` are the single source of truth for all column names, data types, and feature definitions.** When writing Python code that references dataframe columns, always derive column names from the SQL schema — not from Python variable names or notebook outputs.

### Core Tables (pml schema)

| Table | Purpose |
|-------|---------|
| `pml.pml_df` | Master 578-column denormalized equity dataframe. All numeric columns are `double precision`; identifiers are `text`; dates are `date`. |
| `pml.staging` | Raw CSV/vendor landing zone with original vendor column names (mixed case). Mirrors `pml_df` structure. |
| `pml.pml_df_metadata` | Feature registry: one row per `pml_df` column with `pymc_role`, `feature_role`, `category`, `data_type`, `model_targets[]`. |
| `pml.pml_df_feature_alias` | Per-model alias overrides: `(column_name, model_target) → feature_alias`. |

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
price_target, price_target_low, price_target_median, price_target_high,
price_target_num, p_e_ntm, p_e_ltm, analyst_rating,
altman_z_score_{fy,fq,ltm}, beta_{1y,2y,5y}
```

**Fundamentals** — all carry `_ltm`, `_fy`, `_fq`, `_neg1fy`, `_neg1fq` variants:
```
eps_{adj,gaap,diluted}_*, ebitda_*, revenue_*, gross_profit_*,
fcf_*, cfo_*, cfi_*, cff_*, capital_expenditure_*,
roa_*, roe_*, gpm_*, ev_ebitda_*, ev_sales_*, pe_*, pb_*
```

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

| Suffix | Meaning |
|--------|---------|
| `_ltm` | Last Twelve Months |
| `_fy` / `_fq` | Current fiscal year / quarter |
| `_neg1fy` / `_neg1fq` | Prior fiscal year / quarter |
| `_ntm` | Next Twelve Months |
| `_fy1e` … `_fy5e` | Fiscal year estimate (consensus) |
| `_1w` / `_1m` / `_3m` / `_6m` / `_1y` / `_3y` / `_5y` | Lookback window |
| `_ytd` / `_mtd` / `_qtd` | Year/month/quarter-to-date |
| `_est_avg` | Analyst consensus average |
| `_surprise_pct` | (Actual − Estimate) / \|Estimate\| × 100 |
| `pct_` prefix | Percentage value |
| `feat_` prefix | Engineered feature (only in materialized views) |
| `observed_` prefix | PyMC observed/target variable (only in MVs) |
| `n_` prefix | Integer count (PyMC `constant_data`) |

Common abbreviations: `eps`, `fcf`, `cfo`, `cfi`, `cff`, `ebitda`, `gpm`, `roa`, `roe`, `ev`, `pe`, `pb`, `dps`, `ema`, `shrs`.

### pymc_role Enum (pml_df_metadata)

Drives all Python feature selection — do not invent new values:

| pymc_role | Meaning |
|-----------|---------|
| `coord` | Categorical index (isin, sector, region …) |
| `index` | Panel time index |
| `observed` | Target / response variable |
| `mutable_predictor` | Trainable feature (pm.Data, updated at predict time) |
| `constant_data` | Fixed prior / metadata (pm.ConstantData) |
| `derived_input` | Computed from raw columns before model entry |
| `excluded` | Omitted from all PyMC models |

Query features for a model:
```sql
SELECT column_name, feature_alias, data_type
FROM pml.vw_pymc_feature_catalogue
WHERE model_target = 'earnings_beat'
  AND pymc_role = 'mutable_predictor'
ORDER BY ordinal_position;
```

### Materialized Views — Per-Model Feature Matrices (pml schema)

Each MV is indexed on `isin` (UNIQUE). All use `feat_` prefix for engineered columns. Refresh with `CALL pml.refresh_pymc_materialized_views();`

| MV | Observed column | Key `feat_` columns |
|----|----------------|---------------------|
| `mv_pymc_earnings_beat` | `n_total`, `n_beats`, `n_total_annual`, `n_beats_annual` | `feat_logit_beat_rate`, `feat_eps_fy1e`, `feat_rev_{1w,1m,3m,6m,1y}`, `feat_rev_accel_1m_6m`, `feat_last_q_surprise` |
| `mv_pymc_price_target` | `observed_target_pct`, `observed_target_pct_med`, `price_target`, `n_analysts` | `feat_net_buy_sentiment`, `feat_implied_upside`, `feat_target_range_width`, `feat_pt_momentum_3m`, `feat_target_dispersion_cv`, `feat_52w_range_position` |
| `mv_pymc_kalman_pt` | `observed_pt`, `last_price`, `n_analysts` | `feat_pt_drift`, `feat_price_drift`, `feat_pt_noise_sigma`, `feat_pt_range_norm`, `feat_vol_{1m,3m,6m,1y}` |
| `mv_pymc_dcf_pt` | `observed_pt` | `feat_fcf_growth_{1y,2y}`, `feat_fcf_terminal_growth`, `feat_reinvest_rate`, `feat_capex_to_fcf`, `feat_tr_cagr_{3y,10y}` |
| `mv_pymc_dividend_safety` | `observed_div_yield` | `feat_fcf_coverage`, `feat_cfo_coverage`, `feat_eps_payout_ratio`, `feat_dps_growth_{1y,3y,5y}`, `feat_yield_spread_vs_5y` |
| `mv_pymc_credit_risk` | `observed_altman_z` | `feat_distress_zone`, `feat_z_trend_{1y,3y}`, `feat_cfo_capex_cov`, `feat_fcf_yield`, `feat_beta_2y` |
| `mv_pymc_accounting_anomaly` | `observed_eps_adj` | `feat_accruals_ratio`, `feat_gpm_change_1y`, `feat_eps_adj_gap`, `feat_cfi_to_cfo`, `feat_fcfps_vs_eps_gap` |

### Metadata & Catalogue Views (pml schema)

| View | Purpose |
|------|---------|
| `vw_pml_df_predictors` | All `pymc_role = 'mutable_predictor'` columns |
| `vw_pml_df_observed` | All `pymc_role = 'observed'` columns |
| `vw_pml_df_coords` | All `pymc_role = 'coord'` columns |
| `vw_pml_df_derived_inputs` | All `pymc_role = 'derived_input'` columns |
| `vw_pml_df_pymc_features` | All PyMC-relevant columns with `model_name` |
| `vw_pymc_feature_catalogue` | Master 1-row-per `(model_target, pymc_role, column_name)` with alias fallback chain |
| `vw_pymc_feature_aliases` | Aggregated alias arrays per model |
| `vw_pymc_feature_coverage` | Diagnostic: count of columns per `(model_target, pymc_role)` |

### SQL Helper Functions (pml schema)

All functions are `IMMUTABLE PARALLEL SAFE` with both `NUMERIC` and `DOUBLE PRECISION` overloads.

```sql
-- Arithmetic
pml.safe_divide(numerator, denominator)            -- NULLIF-safe division
pml.pct_change(current_val, previous_val)          -- (cur - prev) / prev * 100
pml.calc_change_ratio(current_val, previous_val)   -- (cur - prev) / prev
pml.target_drift(arr DOUBLE PRECISION[])           -- AVG of consecutive calc_change_ratio

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

-- Date / fiscal
pml.frequency_to_months(frequency text, fy_end_date, next_fy_end_date) → INT
pml.calculate_next_fiscal_quarter(next_earnings_date, ...) → INT    -- returns 1-4
pml.ema_crossover_signal(fast_ema, slow_ema) → INT  -- 1 / -1 / 0
```

## Key Architectural Patterns

### 1. Single Source of Truth (SSOT)

- Features: `feature_catalog.py` synced with SQL registry
- Hierarchy: `_hierarchy.py` shared by all models
- Identifiers: `DEFAULT_IDENTIFIER_COLUMNS` in feature_catalog.py
- Schema: `public.equities_schema_metadata`

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

`PipelineConfig` centralizes magic numbers for CLI/env override.


## Code Guidelines

These patterns are derived from the actual `probabilistic_ml_model/` source and must be followed consistently.

### Lazy Imports

All `pymc`, `arviz`, and `pytensor` imports are deferred via `__getattr__` in `__init__.py`. Never import them at module level — use the lazy registry instead:

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
    nuts_sampler: Optional[str] = None,   # e.g. "nutpie", "blackjax", "numpyro"
    **sample_kwargs: Any,
) -> tuple[InferenceLike, "pm_typing.Model"]:
```

Always pass `compile_kwargs=get_pytensor_compile_kwargs()` to `pm.sample()`:
```python
from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs

idata = pm.sample(
    samples, tune=tune, chains=chains, target_accept=target_accept,
    compile_kwargs=get_pytensor_compile_kwargs(),
    nuts_sampler=nuts_sampler,
    **sample_kwargs,
)
```

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

| Kind | Style | Example |
|------|-------|---------|
| Classes | `CapitalCamelCase` | `CreditRiskBayesian` |
| Functions / methods | `snake_case` | `build_hierarchy_indices` |
| Private modules / functions | `_snake_case` | `_hierarchy.py`, `_DEFAULT_SAMPLES` |
| Constants | `UPPER_SNAKE_CASE` | `HIERARCHICAL_CATEGORY_COLS`, `PARENT_MAP` |
| Module-level defaults | `_DEFAULT_*` | `_DEFAULT_CHAINS = 4` |

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

Use `threading.Lock` (via `field(default_factory=threading.Lock, repr=False, compare=False)`) when a dataclass may be accessed from multiple threads.

### Dataclasses

- Registry/config entries that should be immutable: `@dataclass(frozen=True)`
- Mutable state containers: `@dataclass` with `field(default_factory=...)` for mutable defaults
- Always expose a `from_env()` classmethod on config dataclasses

## Common Development Tasks

### Adding a New PyMC Model

1. Create `probabilistic_ml_model/pymc_models/MyModelName.py`:
   - `fit(...)` method with optional `categories_df` + `hierarchy_levels`
   - Return `(idata, posterior_pred_or_none)`
   - Call `stamp_feature_provenance(...)` to attach metadata

2. Update `pymc_models/__init__.py`:
   ```python
   _LAZY_IMPORT_MAP["MyModel"] = (".pymc_models.MyModuleName", "MyModel")
   ```

### Adding a New Feature View or MV Column

1. Add column to `pml.pml_df` DDL and `pml.staging` (same column, same `double precision` type).
2. Register in `pml.pml_df_metadata`: insert a row with `column_name`, `pymc_role`, `feature_role`, `category`, `data_type`, and `model_targets[]`.
3. If it's a PyMC feature, add `feat_` column to the relevant `pml.mv_pymc_*` MV definition in `sql_scripts/pml/mv_pymc_*.sql` using a `pml.*` helper function.
4. Refresh: `CALL pml.refresh_pymc_materialized_views();`
5. Update `FEATURE_VIEW_REGISTRY` in `feature_catalog.py` to match.
6. Column name in Python must exactly match the SQL column name (no renaming at the Python layer).

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

## Debugging & Troubleshooting

### PyTensor Compilation (Windows)

PyMC 6.0 + PyTensor 3.0 uses **numba** as the default backend (via nutpie). C++ compilation (`cxx`) is no longer the primary backend and is not required for normal operation.

If you see `FileNotFoundError: cxx not found` from a legacy code path:
```powershell
$env:PYTENSOR_FLAGS = "device=cpu,floatX=float64,cxx=C:\msys64\ucrt64\bin\g++.exe"
```

Disable C++ compilation entirely (forces pure Python / numba path):
```powershell
$env:PYTENSOR_FLAGS = "device=cpu,floatX=float64,cxx="
```

For JAX backend (blackjax / numpyro samplers):
```powershell
$env:JAX_PLATFORM_NAME = "cpu"   # or "gpu" if CUDA available
```

### Database Connection

```python
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DB_URL"])
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.fetchone())
```

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

**Version:** 0.9.9.3 | **Python:** 3.12–3.14 | **PyMC:** >=6.0,<7 | **PyTensor:** >=3.0,<4 | **ArviZ:** >=1.0,<2 (arviz-base + arviz-stats + arviz-plots) | **JAX:** >=0.4.30 | **DB:** PostgreSQL
