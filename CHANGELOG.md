# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.9.4] - 2026-05-30

### Added

- **`probabilistic_ml_model/pymc_models/_price_target_mc.py`** — new shared
  price-target Monte-Carlo helper module backing the refactored
  PyMC price-target / DCF workflow
  ([`bf35a96`](https://github.com/Kabenge42/PML_Finance_Project/commit/bf35a96)).
- **Test coverage for the price-target stack** — added
  `tests/test_price_target_mc.py` and `tests/test_dcf_pt_nb_integration.py`
  exercising the new helper and the DCF price-target notebook integration
  ([`bf35a96`](https://github.com/Kabenge42/PML_Finance_Project/commit/bf35a96)).
- **`CLAUDE.md`** — project guidance for Claude Code, plus the `pt_model`
  reference artefact and the `pymc_price_target_v2.ipynb` experiments
  notebook
  ([`53ca0cd`](https://github.com/Kabenge42/PML_Finance_Project/commit/53ca0cd)).
- **`pml`-schema SQL surface** — added the namespaced `sql_scripts/pml/*`
  function/view library, `pml_df_metadata.sql`,
  `pml_df_metadata_populate.sql`, `pml_cohorts.sql`,
  `pml_feature_catalogue.sql`, `pml_features.md`, `pml_mpc.md` and
  `docs/pml_sql_queries_updates.md`
  ([`331e2f6`](https://github.com/Kabenge42/PML_Finance_Project/commit/331e2f6)).

### Changed

- **SQL functions namespaced under the `pml` schema** — every
  `sql_scripts/pml/*` helper and materialized-view definition (and
  `import_pml_data.sql`) was migrated to the `pml` schema, and unnecessary
  metadata / cell outputs were stripped from the PyMC notebooks
  (`pymc_earnings_beat.ipynb`, `pymc_price_target.ipynb`)
  ([`99c7547`](https://github.com/Kabenge42/PML_Finance_Project/commit/99c7547)).
- **`DCF_PriceTargetModel.py`** refactored to route through the new
  `_price_target_mc.py` helper, alongside refreshed
  `pymc_dcf.ipynb` / `pymc_price_target.ipynb` notebooks and `README.md`
  ([`bf35a96`](https://github.com/Kabenge42/PML_Finance_Project/commit/bf35a96)).
- **`probabilistic_ml_model/_pymc_arviz_compat.py`**,
  **`_pytensor_compat.py`** and **`EarningsBeatModel.py`** received
  compatibility / behaviour touch-ups in step with the `pml`-schema
  migration
  ([`331e2f6`](https://github.com/Kabenge42/PML_Finance_Project/commit/331e2f6),
  [`99c7547`](https://github.com/Kabenge42/PML_Finance_Project/commit/99c7547)).
- **Environment / project config** — refreshed `set_env.ps1`,
  `environment_variables.txt`, `Pipfile`, `pyproject.toml`,
  `requirements.txt`, `README.md` and `.idea/*` settings to match the
  new schema and notebook layout
  ([`331e2f6`](https://github.com/Kabenge42/PML_Finance_Project/commit/331e2f6),
  [`99c7547`](https://github.com/Kabenge42/PML_Finance_Project/commit/99c7547)).
- **Data snapshots** — regenerated the regional PML / screening CSVs
  (`data/pml/pml_{us,eu,apac,rotw}.csv`,
  `data/screening_{us,eu,apac,rotw}.csv`) against the updated `pml`
  feature surface
  ([`331e2f6`](https://github.com/Kabenge42/PML_Finance_Project/commit/331e2f6),
  [`bf35a96`](https://github.com/Kabenge42/PML_Finance_Project/commit/bf35a96)).

### Removed

- **`pml_features.sql`** — the monolithic legacy feature script was
  removed in favour of the modular `sql_scripts/pml/*` library and
  `pml_feature_catalogue.sql`
  ([`331e2f6`](https://github.com/Kabenge42/PML_Finance_Project/commit/331e2f6)).
- **`pml_df_new.ipynb`** and the bloated `pymc_expected_returns_v2.ipynb`
  cell outputs were dropped during the notebook/schema cleanup
  ([`331e2f6`](https://github.com/Kabenge42/PML_Finance_Project/commit/331e2f6)).

### Notes

- Patch-level bump (`0.9.9.3 → 0.9.9.4`) — the release packages a `pml`
  SQL-schema migration, data-snapshot refresh, new price-target
  Monte-Carlo helper with tests, and project-guidance docs; no breaking
  public-API changes.
- Follow-up: bump `pyproject.toml` `version` and the README badge to
  `0.9.9.4` in the next packaging pass (left untouched here to keep this
  CHANGELOG refresh purely a documentation change).

[0.9.9.4]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.9.3...v0.9.9.4

## [0.9.9.3] - 2026-05-15

### Added

- **`pml_df_new.ipynb`** — new comprehensive analysis notebook for
  probabilistic PyMC models across the multi-feature PML dataset, used as
  the working surface for the §13 actionable-recommendation walkthroughs
  introduced in 0.9.8.5
  ([`3e39257`](https://github.com/Kabenge42/PML_Finance_Project/commit/3e39257)).
- **`pymc_expected_returns_v2.ipynb`** — v2 PyMC expected-returns
  experiments notebook split out from the main
  `pymc_expected_returns_model.ipynb` to keep the canonical end-to-end
  workflow lean while iterating on alternative model variants
  ([`3e39257`](https://github.com/Kabenge42/PML_Finance_Project/commit/3e39257)).
- **`pml_feature_catalogue.sql`** — consolidated PML feature SQL surface
  (registry-aligned column definitions and helper views) added under the
  repo root to support the new notebook workflow
  ([`3e39257`](https://github.com/Kabenge42/PML_Finance_Project/commit/3e39257)).

### Changed

- **`pymc_expected_returns_model.ipynb`** — large refactor that trims
  the notebook to the canonical end-to-end PyMC + ArviZ workflow and
  realigns the §1–§13 cells with the shared
  `MODEL_FEATURE_CONTAINERS` registry and `_hierarchy.py` /
  `_feature_alignment.py` helpers
  ([`3e39257`](https://github.com/Kabenge42/PML_Finance_Project/commit/3e39257)).
- **`probabilistic_ml_model/pymc_models/*`** — minor touch-ups across
  all seven PyMC model modules (`EarningsBeatModel`, `PriceTargetModel`,
  `DCF_PriceTargetModel`, `DividendSafetyModel`, `CreditRiskModel`,
  `KalmanFilterModel`, `AccountingAnomalyModel`,
  `MonteCarloSimulation`, `ProbabilisticLinearRegressionModel`) plus
  `statistical_functions/statistical_models.py`, keeping the
  hierarchical-shrinkage and feature-alignment contracts in sync with
  the refreshed notebook surface
  ([`3e39257`](https://github.com/Kabenge42/PML_Finance_Project/commit/3e39257)).
- **`probabilistic_ml_model/_pymc_arviz_compat.py`** and
  **`probabilistic_ml_model/data_utils/inference_schema.py`** — small
  compatibility-shim updates to match the latest ArviZ 1.0 /
  `xarray.DataTree` API surface used by the new notebooks
  ([`3e39257`](https://github.com/Kabenge42/PML_Finance_Project/commit/3e39257)).
- **SQL artefacts** (`feature_registry.sql`, `import_pml_data.sql`,
  `equities_schema_metadata_setup.sql`, `mv_equities.sql`,
  `sql_scripts/pml/pml_df.sql`,
  `sql_scripts/public/feature_catalogue.sql`) regenerated to match the
  updated PML feature surface
  ([`3e39257`](https://github.com/Kabenge42/PML_Finance_Project/commit/3e39257)).
- **Data snapshots** — refreshed regional PML / screening CSVs
  (`data/pml/pml_{us,eu,apac,rotw}.csv`,
  `data/screening_{us,eu,apac,rotw}.csv`) to reflect the latest
  upstream import run
  ([`3e39257`](https://github.com/Kabenge42/PML_Finance_Project/commit/3e39257)).
- **Tooling / IDE config** — added `.idea/IntelliLang.xml` (Selenium
  injection patterns) and `.idea/python-terminal.xml`
  (`virtualEnvActivate = false`); refreshed
  `.idea/data_source_mapping.xml`, `.idea/sqldialects.xml`,
  `.idea/finance-ml-analytics-platform.iml` and removed stale
  `.idea/csv-editor.xml` / `.idea/externalDependencies.xml` entries
  ([`3e39257`](https://github.com/Kabenge42/PML_Finance_Project/commit/3e39257)).

### Notes

- Patch-level bump (`0.9.9.2 → 0.9.9.3`) — no public-API or
  dependency-pin changes; the release packages a notebook split,
  data-snapshot refresh, and IDE-config tidy-up alongside non-behavioural
  touch-ups in the PyMC model modules.
- Follow-up: bump `pyproject.toml` `version` and the README badge to
  `0.9.9.3` in the next packaging pass (left untouched here to keep this
  CHANGELOG refresh purely a documentation change).

[0.9.9.3]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.9.2...v0.9.9.3

## [0.9.9.2] - 2026-05-09

### Changed — Dependency alignment across `Pipfile`, `pyproject.toml`, `requirements.txt`

Audit-driven cleanup that resolves the cross-file version drift identified
in the v0.9.9.1 dependency review and refreshes Python 3.14.4 compatibility
gating against PyPI / upstream release notes (May 2026).

- **Hard conflict fixed in `Pipfile`** — `pytensor` was pinned at
  `>=2.38.0,<5.0.0`, which is incompatible with the cross-file
  `pymc>=5.28,<5.29` pin (PyMC 5.28 requires `pytensor<2.34`). Bumped
  down to `pytensor>=2.31,<3` to match `pyproject.toml` and
  `requirements.txt`.
- **Aligned upper bounds across all three files**:
    - `numpy`: Pipfile `<4.0.0` → `<3.0.0`
    - `pymc`: Pipfile `>=5.0.0,<8.0.0` → `>=5.28,<5.29`
    - `xarray`: Pipfile `>=2025.0.0,<2027.0.0` → `>=2024.7.0,<2027.0.0`
    - `streamlit`: requirements.txt `<4.0.0` → `<6.0.0`
    - `dash`: requirements.txt `<5.0.0` → `<7.0.0`
- **Unified Python-version markers** to the canonical
  `python_version < '3.14'` form across all three files. The previous
  `python_version <= '3.14'` form in `requirements.txt` and the
  `tensorflow` / `streamlit` / `numba` entries in
  `pyproject.toml` silently broke on Python 3.14.x because no wheels
  exist for that interpreter — using the strict-less-than form makes
  pip skip those packages on 3.14 instead of failing the install.
- **Ungated `shap` and `numba` for Python 3.14**:
    - `shap>=0.50.0,<1.0.0` — SHAP 0.50.0 (Nov 2025) added Python 3.14
      test coverage; the marker is now removed from all three files.
    - `numba>=0.63.0,<1.0.0` — Numba 0.63.0 (Dec 2025) is the first
      release with official Python 3.14 + free-threaded support;
      minimum version bumped from `0.60.0` to `0.63.0`.
- **Still gated to `python_version < '3.14'`** (no 3.14 wheels yet on
  PyPI as of May 2026): `catboost`, `streamlit`, `tensorflow`,
  `scikeras`.
- **Missing core deps added to `Pipfile`** to bring it in line with
  `pyproject.toml` / `requirements.txt`:
  `cython>=3.0.0`, `pyyaml>=6.0.0,<7.0.0`, `bambi>=0.15.0,<1.0.0`,
  `nutpie>=0.13.0,<1.0.0`, `jax>=0.4.30,<1.0.0`,
  `jaxlib>=0.4.30,<1.0.0`.
- **Refreshed `requirements.txt` header** date to `2026-05-09` and the
  `Aligned with pyproject.toml` line to `v0.9.9.2`.

### Changed — Documentation

- **`pyproject.toml` `version`** bumped from `0.9.5` to `0.9.9.2` so it
  matches the README badge and CHANGELOG entries (the previous
  mismatch between `pyproject.toml` v0.9.5 and the README v0.9.8.5
  badge is resolved).
- **`README.md`** — updated the version badge to `0.9.9.2`, refreshed
  the "Python-Version-Gated Dependencies" section to reflect the
  current gate set (`catboost`, `streamlit`, `tensorflow`, `scikeras`)
  and explicitly note that `shap` and `numba` are no longer gated, and
  bumped the `pyproject.toml` configuration-files row to v0.9.9.2.

### Notes

- No source code changes were required — the dependency-marker
  cleanup is purely a packaging / install-time concern. Existing
  `import shap` / `import numba` call sites continue to work unchanged
  on Python 3.12 / 3.13 / 3.14.
- Follow-up: re-run `pipenv lock` and refresh
  `requirements.txt` from `pip-compile` once the new `Pipfile` lands
  in CI to regenerate `Pipfile.lock` against the aligned version
  windows.

[0.9.9.2]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.9.1...v0.9.9.2

## [0.9.9.1] - 2026-05-02

### Added — Centralised OOS leaf-index resolution helpers in `_hierarchy.py`

- **`probabilistic_ml_model/pymc_models/_hierarchy.py`** — added three new
  helpers that centralise the foot-gun-prone leaf-index swap every PyMC
  model's out-of-sample (`pm.set_data`) cell needed to repeat:
  - `find_leaf_idx_var(idata)` — returns the `{leaf_level}_idx` data_var
    name registered on `idata.constant_data` by
    `build_nested_logit_normal_rates`, or `None`.
  - `resolve_leaf_train_labels(idata, leaf_level)` — returns the trained
    labels for `leaf_level` from a fitted `idata`. Inspects **both**
    `constant_data.coords` and `posterior.coords` (the level coords
    typically live on `posterior`, not `constant_data` — this was the
    missing piece for the §9.6 DividendSafety crash).
  - `build_oos_setdata(idata, categories_df, new_isins, *, extra_data)`
    — builds a `pm.set_data` payload that auto-includes the
    `{leaf_level}_idx` swap whenever a hierarchy was registered, with
    unknown labels mapped to `0` for `KeyError` safety.
- **`pymc_expected_returns_model.ipynb`** — refactored the §5
  EarningsBeat, §9.6 DividendSafety and §10.6 CreditRisk OOS cells to
  delegate the leaf-idx lookup to `build_oos_setdata`. The
  DividendSafety cell previously omitted the leaf-idx swap entirely,
  causing the (2072,) vs (50,) broadcast failure inside
  `risk_adj` / `expected_coverage`; the new helper restores the shape
  contract end-to-end.

## [0.9.9.0] - 2026-05-02

### Added — Multi-Level Hierarchical Shrinkage for PyMC Models (§12 / §13 plan)

- **New `probabilistic_ml_model/pymc_models/_hierarchy.py`** — shared
  infrastructure (single source of truth) for the canonical category
  hierarchy used by every PyMC model:
  - `HIERARCHICAL_CATEGORY_COLS` and `PARENT_MAP` — canonical column tuple
    and parent-of-child relationships (region → country → exchange →
    sector → industry, plus independent `style_class` / `size_class`
    and `unit` / `trading_country` branches). Now re-imported by
    `statistical_functions/statistical_models.py` so PyMC models and the
    multi-level shrinkage helper share one definition.
  - `build_hierarchy_indices(df, isins, levels=None)` — pure-NumPy helper
    returning per-level metadata (unique labels, isin → level idx, level
    → parent idx). Generalises the previous flat
    `np.unique(sectors, return_inverse=True)` block to N nested levels.
  - `build_nested_logit_normal_rates(hierarchy, ...)` — PyMC helper that
    materialises a nested non-centred logit-Normal chain
    `mu_L[g] = mu_P[parent_of(g)] + sigma_L * z_L[g]`, returning the
    leaf-level rate broadcast to `isin`.
  - `_resolve_prior_sigma(data_type, calculation_type)` — calculation-
    type-driven prior sigma helper (recommendation §12.3 #2).
  - `coerce_categories(...)` — backward-compat shim that wraps the legacy
    `sectors=` argument into a single-level `categories_df` with
    `hierarchy_levels=["sector"]`.
- **All seven PyMC models** (`EarningsBeatModel`, `PriceTargetModel`,
  `DCF_PriceTargetModel`, `AccountingAnomalyModel`, `DividendSafetyModel`,
  `KalmanFilterModel`, `CreditRiskModel`) now accept a unified
  `categories_df` + `hierarchy_levels` pair on `fit(...)` while preserving
  the legacy `sectors=` argument. Default `hierarchy_levels`:
  - EarningsBeat: `["exchange", "sector", "industry"]`
  - PriceTarget: `["exchange", "sector", "industry", "size_class"]`
  - DividendSafety: `["region", "country", "sector", "industry"]`
  - CreditRisk: `["region", "country", "exchange", "sector", "industry"]`
  - AccountingAnomaly / DCF: `["sector", "industry"]`
- **EarningsBeat / CreditRisk / PriceTarget / DividendSafety** route the
  nested non-centred logit-Normal chain through
  `build_nested_logit_normal_rates`, so the leaf rate inherits shrinkage
  from every parent level instead of a single flat `sector` layer.
- **Tests** — `tests/test_hierarchical_pymc_models.py` adds 11 tests
  covering the canonical constants, `build_hierarchy_indices` shapes /
  parent-of consistency, `_resolve_prior_sigma` bounds, and the
  backward-compat shim (legacy `sectors=` still produces a `sector`
  coord on EarningsBeat and CreditRisk; multi-level `categories_df`
  registers every level as a coord).
- **`pymc_expected_returns_model.ipynb`** — every model `fit(...)` cell
  (EarningsBeat, PriceTarget, DCF, DividendSafety, CreditRisk,
  AccountingAnomaly) now passes `categories_df=` + `hierarchy_levels=`
  built from `HIERARCHICAL_CATEGORY_COLS` instead of the legacy flat
  `sectors=` argument. A small `_build_categories_df(df, isins)` helper
  is introduced in §1 (Earnings Beat preparation) and reused by every
  later section to align the notebook with the new shared infra.

## [0.9.8.5] - 2026-04-30

### Added — `feature_catalogue`-aligned model improvements (recommendations from §12.3)

- **`probabilistic_ml_model/pymc_models/_feature_alignment.py`** — new
  shared helper module implementing the actionable items from §12.3 of
  `pymc_expected_returns_model.ipynb`:
  - `load_feature_metadata_from_db(connection_string)` — loads the full
    `(category, calculation_type, data_type, source_function)` tuple per
    `feature_alias` from `public.calculated_features_registry` (LRU-cached).
  - `coerce_by_data_type(df, feature_aliases, metadata)` — type-aware
    analogue of `df.reindex(columns=...).astype('float64').fillna(0.0)`
    that drives per-column dtype + bounded clipping from
    `feature_catalogue.data_type` (`pct ∈ [-1, 1]`, `flag → int8`,
    `score ∈ [0, 100]`, `growth ∈ [-5, 5]`, `zscore ∈ [-10, 10]`, …).
    Implements **rec #1**.
  - `stamp_feature_provenance(idata, var_name, aliases, metadata)` —
    copies `feature_alias` / `source_function` / `calculation_type` /
    `data_type` onto `idata.constant_data[var_name].attrs` for downstream
    lineage tooling. Implements **rec #3**.
  - `assert_disjoint_features(idata, new_aliases, new_var_name=...)` —
    category-conflict guard that raises `ValueError` if the new alias
    set overlaps with any previously attached `constant_data` variable's
    alias set. Implements **rec #4**.
  - `validate_oos_shape(new_arr, feature_aliases, var_name=...)` —
    asserts `new_arr.shape[1] == len(feature_aliases)` before a
    `pm.set_data({...})` swap. Implements **rec #7**.
- **All seven PyMC models updated** (`PriceTargetModel.py`,
  `EarningsBeatModel.py`, `DCF_PriceTargetModel.py`,
  `DividendSafetyModel.py`, `CreditRiskModel.py`,
  `AccountingAnomalyModel.py`, `KalmanFilterModel.py`):
  - Each `_align_*_features(...)` static method gained an opt-in
    `use_typed_coercion: bool = False` keyword (plus `connection_string`)
    that routes through `coerce_by_data_type` when enabled.
    Default behaviour unchanged for backwards compatibility.
  - The six isin-keyed model `fit(...)` methods (all except Kalman, whose
    `fit` is keyed on `time` rather than `isin`) now call
    `stamp_feature_provenance(idata, "<model>_features", aliases, metadata)`
    immediately after `pm.sample(...)`, so every fitted `idata.constant_data`
    carries `source_function` / `calculation_type` / `data_type` lineage.
- **`pymc_expected_returns_model.ipynb`** — appended a new **§13
  "Implementation — Actionable Recommendations from §12.3"** with one
  runnable code cell per recommendation (#1 type-aware coercion,
  #2 calc-type-driven prior σ table, #3 provenance attrs check,
  #4 strict `attach_features_strict` wrapper, #5 catalogue-driven
  coverage check across all seven `_resolve_*_feature_aliases`,
  #6 per-`category` hyperprior PyMC block sketch for
  `AccountingAnomalyBayesian`, #7 OOS shape contract via
  `validate_oos_shape`). 15 new cells (1 intro + 7 markdown +
  7 code).

### Notes

- **Rec #2 (calc-type-driven priors)** and **rec #6 (per-category
  hyperprior)** are surfaced as ready-to-use code snippets in §13.2 /
  §13.6 of the notebook but are *not* wired into the per-model `fit(...)`
  bodies in this release — those changes affect the sampler's parameter
  count and posterior shape, so they are deferred to a follow-up version
  alongside regression fixtures.
- **Rec #5 (catalogue-driven test coverage)** is currently the §13.5
  notebook cell; promotion to `tests/test_feature_catalogue_coverage.py`
  is a follow-up.

[0.9.8.5]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.8.4...v0.9.8.5

## [0.9.8.4] - 2026-04-30

### Added — Out-of-sample PPC code cells in `pymc_expected_returns_model.ipynb`

- Replaced the seven *illustration-only* "Out-of-sample prediction via
  `pm.set_data`" markdown sections (§5, §6.6, §7.6, §8.6, §9.6, §10.6,
  §11.6) with runnable PyMC code cells. Each new cell now:
  1. Builds a 50-row holdout slice from the corresponding model's
     already-prepared `*_df` / arrays.
  2. Re-aligns the auxiliary feature matrix via the per-model
     `_align_*_features` static helper against the
     `<model>_feature` coordinate stored in `idata.constant_data`,
     so the swap respects the catalogue-aligned dim order.
  3. Calls `pm.set_data({...}, coords={'isin': ...})` and
     `pm.sample_posterior_predictive(idata, var_names=[...])` inside
     the model context returned by `*Model.fit(...)`.
  4. Visualises the predictive distribution: histogram of mean
     predicted beat-rate (EarningsBeat), histogram of upside (PriceTarget),
     time-series fan chart with 90 % PI (Kalman), observed-vs-PPC
     scatter (DCF, DividendSafety), distress-probability histogram
     (CreditRisk), and anomaly-probability histogram recomputed from
     posterior `feature_scale` × `threshold` (AccountingAnomaly).
- The CreditRisk cell additionally swaps the data-vector
  `zone_adj` (precomputed Altman-zone factor) and `distress_target`
  alongside `z_score_data`, `de_data`, `sector_idx`, and
  `credit_features`, matching the full set of `pm.Data` containers
  declared inside `CreditRiskBayesian.fit`.
- The Kalman cell additionally re-sets `log_price_target`
  (the log-space observation container) so the `obs` likelihood
  remains shape-aligned along the `time` dim.
- No source edits required in `probabilistic_ml_model/pymc_models/*` —
  the existing `pm.Data(..., dims=("isin", "<dim>"))` containers and
  `_align_*_features` helpers already expose the surface needed for
  out-of-sample swaps.

[0.9.8.4]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.8.3...v0.9.8.4

## [0.9.8.3] - 2026-04-30

### Changed — `attach_features` notebook helper made model-aware

- **`pymc_expected_returns_model.ipynb`** — refactored the `attach_features`
  helper used to append catalogue-aligned (isin × feature) matrices to
  `idata.constant_data` for models that don't natively persist the
  auxiliary feature matrix (`PriceTargetAchievement`,
  `DividendSafetyBayesian`, `CreditRiskBayesian`, `DCFPriceTarget`).
  The new signature is
  `attach_features(idata, features_df, model_name, *, dim_name=None, var_name=None)`
  with `model_name` keyed into `MODEL_FEATURE_CONTAINERS`. The helper
  now derives the canonical `feature_alias` list from
  `spec["observed"]` (registry-resolved) with fallback to
  `spec["features"]` (`FEATURE_CATEGORIES`-derived), and the `pm.Data`
  dim name from a new `_MODEL_FEATURE_DIM` mapping
  (`EarningsBeat→earnings_feature`, `PriceTarget→pt_feature`,
  `Kalman→kalman_feature`, `DCF→dcf_feature`,
  `DividendSafety→dividend_feature`, `CreditRisk→credit_feature`,
  `AccountingAnomaly→anomaly_feature`) so the notebook helper can
  never drift out of sync with the per-model
  `_resolve_*_feature_aliases()` contract declared inside each
  `pymc_models/*.py` `with pm.Model(coords=...)` block. The data-var
  name defaults to `f"{dim}s"` (e.g. `earnings_features`,
  `dcf_features`), matching what each model already writes into
  `constant_data`.
- **Missing columns/rows behaviour** — features absent from
  `features_df` are reindexed and filled with `0.0` (mirroring the
  per-model `_align_*_features` static methods on the model classes)
  instead of raising `KeyError`, and the `xr.Dataset.merge(ds)` call
  now passes `compat="override"` so the helper can be re-attached
  without conflicts on overlapping coord values.
- **Validation** — `attach_features` raises `KeyError` listing the
  valid `MODEL_FEATURE_CONTAINERS` keys when an unknown
  `model_name` is supplied.
- No source edits required in `probabilistic_ml_model/pymc_models/*` —
  the per-model `pm.Data(..., dims=("isin", "<dim>"))` containers
  (e.g. `pt_features`, `dcf_features`, `dividend_features`,
  `credit_features`, `earnings_features`, `kalman_feature`,
  `anomaly_features`) and their `_resolve_*_feature_aliases()` helpers
  already match the new `_MODEL_FEATURE_DIM` mapping and naming
  convention.

[0.9.8.3]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.8.2...v0.9.8.3

## [0.9.8.2] - 2026-04-23

### Fixed — EPS streak pre-merge ordering (v3.10 §12.5 orchestration fix)

- **`probabilistic_ml_model/pipeline_runners.py::run_earnings_beat_analysis`**
  and **`expected_returns_v3.py::run_earnings_beat_analysis`** reordered so
  that `EPSStreakAnalyzer.analyze_dataframe` runs *before*
  `EarningsBeatProbabilityModel.analyze_dataframe_enhanced` and
  `ResampledBeatProbabilityModel.analyze_dataframe`. The streak posterior
  columns (`map_estimate`, `model_confidence`) are now merged onto both
  `beat_df` (primary enhanced-analyzer input) and the source `df`
  subsequently consumed by the resampled wrapper (which re-invokes the
  base enhanced analyzer internally per chain/seed). This eliminates
  the four `logger.warning` lines previously emitted on every pipeline
  run ("expected streak-merge columns ['map_estimate',
  'model_confidence'] not found in DataFrame ...") and restores the
  Bayesian momentum-prior tilt that was silently skipped for ~15 % of
  the universe per v3.8 logs.
- **New `strict_streak_merge` keyword argument** added to both
  `run_earnings_beat_analysis` implementations. When `True` the
  underlying `analyze_dataframe_enhanced` raises `KeyError` if the
  streak-merge columns are missing, so regressions that re-introduce
  the silent drop fail fast instead of degrading silently.
- **`expected_returns_v3._step_earnings_beat`** opts into
  `strict_streak_merge=True` when the `PML_STRICT_STREAK_MERGE` env
  var is truthy (`1` / `true` / `yes`), intended for CI/regression
  runs. Default behaviour unchanged for interactive pipeline runs.

### Notes

- No schema or downstream-merge changes required — the existing
  `[c for c in resampled_df.columns if c != "isin" and c not in beat.columns]`
  merge shape introduced in 0.9.8.1 already propagates the resampled
  diagnostic columns correctly once the streak pre-merge is in place.
- `EarningsBeatProbabilityModel.analyze_dataframe_enhanced` and
  `EPSStreakAnalyzer.analyze_dataframe` are unchanged; the fix is
  purely an orchestration-ordering refactor.

[0.9.8.2]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.8.1...v0.9.8.2

## [0.9.8.1] - 2026-04-22

### Fixed — ResampledBeat null-column population (v3.10 patch)

- **`ResampledBeatProbabilityModel._run_analysis`**
  (`probabilistic_ml_model/statistical_functions/probability_models.py`)
  now populates the Part-2 `ResampledBeatEstimate` fields that were
  previously silently left as NaN: `posterior_std` (closed-form Beta
  variance), `hdi_low` / `hdi_high` (94 % symmetric quantile
  approximation), `n_effective_samples` (α + β concentration proxy),
  `volatility_regime` ("low" / "normal" / "high" label derived from
  the continuous `vol_regime` score) and `schema_version`.
  `chain_rhat` / `chain_ess_bulk` / `chain_ess_tail` are seeded as NaN
  sentinels and subsequently overwritten inside
  `ResampledBeatProbabilityModel.analyze_dataframe` from the ArviZ
  summary (same pass that already populated the legacy `ess_bulk` /
  `r_hat` columns) so they are non-NaN whenever ArviZ chains build
  successfully.
- **`expected_returns_v3.build_expected_returns_summary` earn
  allow-list** (lines 3458–3491) extended to propagate the new
  ResampledBeat posterior-spread / chain-diagnostic columns
  (`hdi_low`, `hdi_high`, `chain_rhat`, `chain_ess_bulk`,
  `chain_ess_tail`, `n_effective_samples`, `volatility_regime`) into
  the unified expected-returns summary.
- **`ensemble_models.build_expected_returns_summary` earn merge**
  refactored to the same conditional-inclusion pattern as the main
  pipeline (`[c for c in [...] if c in earn.columns]`) and extended
  with the same seven ResampledBeat diagnostic columns. Backwards
  compatible — missing columns are now silently skipped instead of
  raising `KeyError`.
- `probabilistic_ml_model/pipeline_runners.py::run_earnings_beat_analysis`
  already merges resampled columns via
  `[c for c in resampled_df.columns if c != "isin" and c not in beat.columns]`,
  so the new diagnostic columns flow through automatically without
  source edits.

## [0.9.8] - 2026-04-21

### Added — Probability Models v3.10 (Part 2 of the model-improvement plan)

Part 2 extends the prior diagnostic/dataclass foundation with targeted API
additions on the Accounting Anomaly, Earnings Beat, Model Confidence and
Resampled Beat stacks, and centralises the BMA log-score weight
computation inside `ModelConfidenceEstimator`.

- **§10 / T-F `BeatProbabilityEstimate` hardening**
  (`probabilistic_ml_model/statistical_functions/probability_models.py`).
  Kept as a `TypedDict` for backwards compatibility with the existing
  ``return {...}`` call sites, but added required `ci_low` / `ci_high`
  keys (§10.2), a sibling frozen `BeatProbabilityEstimateDC` dataclass
  with `__post_init__` validation (§10.1), and a standalone
  `validate_beat_probability_estimate()` helper that enforces the same
  contract at runtime without breaking legacy dict-style consumers.
- **§9 `AccountingAnomalyResult` extensions.** Added decoupled component
  fields (`flag_count_posterior_mean`, `flag_count_ci_low/high`,
  `magnitude_posterior_mean`, `combined_anomaly_score`,
  `dominant_flag_category`) per §9.1 and diagnostic-parity fields
  (`tail_df`, `cond_volatility`, `r_hat`, `ess_bulk`, `ess_tail`,
  `schema_version`) per §9.2.
- **§11 `BeatProbabilityResult` extensions.** Added per-layer
  contribution attribution (`prior_contribution`,
  `likelihood_contribution`, `momentum_tilt`, `quality_discount`,
  `macro_tilt`) per §11.1 and sector-prior provenance
  (`sector_prior_key`, `sector_prior_alpha`, `sector_prior_beta`,
  `used_default_prior`) per §11.2; plus diagnostic parity fields
  (`tail_df`, `cond_volatility`, `ess_tail`, `schema_version`).
- **§14 `ModelConfidenceResult` extensions.** Added calibration
  artefacts (`reliability_curve`, `ece_ci_low/high`, `brier_ci_low/high`,
  `log_score`, `auroc`, `n_samples`, `schema_version`) per §14.1 and a
  `passes_calibration(tol=0.05)` gate per §14.2 that short-circuits BMA
  log-score weighting when a sub-model is mis-calibrated.
- **§15 `ResampledBeatEstimate` extensions.** Added posterior spread
  and per-chain diagnostics (`posterior_std`, `hdi_low`/`hdi_high`,
  `chain_rhat`, `chain_ess_bulk`, `chain_ess_tail`,
  `n_effective_samples`, `volatility_regime`) per §15.1 and versioned
  `to_dict` / `from_dict` serialisation per §15.2.
- **§13 `ModelConfidenceEstimator` calibration overhaul.**
  `compute_calibration_error` now supports quantile bins via
  `pd.qcut(duplicates='drop')` with empty-bin drop (§13.1);
  `compute_confidence_metrics` emits a `reliability_curve` list of
  `(bin_mid, empirical_rate, n)` tuples and bootstrap 95 % CIs on
  ECE / Brier / log-score seeded from `RANDOM_SEED` env var (§13.2);
  `_compute_ci_coverage` is rewritten around Wilson-score intervals
  and emits coverage curves at 50 / 80 / 90 / 95 % (§13.3); new
  `compute_relative_confidence(dict[str, (probs, outcomes)])` method
  returns a DataFrame with Brier / ECE / log-score / AUROC / n_samples
  / `passes_calibration` / softmax BMA weights per model (§13.4 / T-E).
- **§12 `EarningsBeatProbabilityModel` targeted additions.** New
  `macro_prior_betas` constructor kwarg + `_apply_macro_logit_tilt`
  method closes the `[PENDING]` macro-covariate item from the v3.8
  findings (§12.3) — applies a `Δ_logit = Σ β_k · z(x_k)` tilt in
  logit space with Normal(0, 0.3) shrinkage defaults
  (`{"vix": -0.20, "yield_10y2y": 0.10}`). `analyze_dataframe_enhanced`
  now surfaces missing streak-merge columns (`map_estimate` /
  `model_confidence`) via `logger.warning` (or `KeyError` when
  `strict_streak_merge=True`), preventing silent drops that degraded
  the momentum prior adjustment for ~15 % of the universe (§12.5).
  New static `fit_priors_empirical_bayes(df)` method-of-moments helper
  returns a `dict[str, PriorParameters]` suitable for the
  `sector_priors=` kwarg — operationalises cross-cutting T-D for the
  beat model (§12.6).
- **§16 `ResampledBeatProbabilityModel` stability.** New `fit_weights`
  method learns per-sector `(momentum_weight, volatility_weight)` via
  non-negative least-squares with sum-to-≤-1 constraint (§16.1). New
  `sector_shrinkage_tau` constructor kwarg enables credibility
  shrinkage in `_adjust_prior` — blends the tilted prior with the
  base-model sector prior via `κ = n_sector / (n_sector + τ)`, directly
  addressing small-sample extreme tilts that drive MXN / TRY drift
  (§16.4). New `stability_report(df, seeds=[42, 7, 99])` runs
  multi-seed resamples and flags tickers with ticker-level std > 2 pp
  as seed-unstable (§16.5).
- **§8 `AccountingAnomalyProbabilityModel` API + decoupling.** Added
  `use_student_t_likelihood` / `use_garch_volatility` /
  `student_t_df_floor` / `sector_priors` / `flag_halflife_years`
  constructor kwargs (§8.2 / §8.3 / §8.5 API surface). The GARCH term
  is currently wired at API level only — sampler term is deferred.
  `analyze_dataframe` now populates the §8.1 / §9.1 decomposition
  component columns (`flag_count_posterior_mean`,
  `flag_count_ci_low/high`, `magnitude_posterior_mean`,
  `combined_anomaly_score`) via a Jeffreys-prior Beta-Binomial on the
  flag-count channel (§8.4 Jeffreys step) and method-of-moments
  shrinkage on the magnitude channel, plus emits a per-row `tail_df`
  column so the T-A per-stock tail-df wiring picks up the anomaly
  model's Student-t df.

### Changed

- **T-E BMA centralisation in `ensemble_models.build_quad_model_alignment`.**
  Optional new `validation_outcomes` kwarg
  (`dict[str, (probs, outcomes)]`) triggers
  `ModelConfidenceEstimator.compute_relative_confidence` and overrides
  the static `bma_weights` dict with log-score-derived softmax weights
  merged per model name. Fully backwards compatible: unchanged
  behaviour when the kwarg is not provided.

### Notes — deferred Part-2 tasks (follow-up work)

The following items from the Priority / Sequencing section of the
Part-2 plan are **not** included in 0.9.8 because they require deeper
MCMC sampler rewrites, dedicated validation fixtures or per-chain
ArviZ posterior-group construction that can't safely be landed
alongside the API / dataclass additions in a single session:

- **§8.3 (sampler side)** Add the Student-t + GARCH residual-z term
  inside `AccountingAnomalyProbabilityModel._apply_mcmc_posteriors`
  (API flags are exposed and consumed by the `tail_df` column export
  but the PyMC / NumPy sampler block is not yet touched).
- **§8.4 (full vectorisation)** `calculate_conditional_probabilities`
  still uses a per-group loop; the Jeffreys-prior step is now applied
  only in the flag-count channel on `analyze_dataframe`.
- **§8.5 (flag time-decay)** `flag_halflife_years` kwarg is exposed
  but not yet applied to the aggregation step.
- **§12.1** Replace the multiplicative tilt stack in
  `_apply_momentum_prior_adjustment` / `_tilt_prior_mean` with a full
  log-odds accumulator (`logit(p) + Σ βᵢ·xᵢ`).
- **§12.2** Promote `_apply_quality_discount` to a Beta(αq, βq)
  shrinkage factor on α / β rather than a point-wise MAP discount.
- **§12.4** Vectorise the `analyze_dataframe` 280-line loop via
  `scipy.stats.beta.ppf` on arrays.
- **§16.2** Swap `_compute_volatility_regime` for the GARCH σ_t from
  `CreditRiskProbabilityModel` / `PriceTargetAchievementModel`.
- **§16.3** Emit per-chain `(chain, draw)` posterior traces from
  `build_inference_data` — the chain-diagnostic fields are already
  present on `ResampledBeatEstimate` but populated by point-estimate
  proxies until the ArviZ `posterior` group is constructed per chain.
- **T-C (extended)** Regression fixtures for accounting anomaly and
  resampled beat pinned within ±2 % across runs.
- **T-D (extended)** Wire `fit_priors_empirical_bayes` + an analogous
  accounting-anomaly EB bootstrap into a standalone quarterly refresh
  script (`tools/refresh_priors.py`).

### Aligned — expected_returns_v3 pipeline & notebook

- **`expected_returns_v3.py`** — extended the export allow-lists so the new
  v3.9 / v3.10 diagnostic and decomposition columns survive the
  row-size-limited PostgreSQL export path:
  - `_trim_credit_for_export`: added `tail_df`, `cond_volatility`,
    `cvar_5`, `posterior_ess_bulk`, `posterior_ess_tail`, `r_hat`,
    `schema_version` and the four `macro_loading_*` columns
    (`yield_curve_10y2y`, `vix`, `dxy`, `hy_oas`) so Credit §2.1 / §2.2
    diagnostics and §1.3 macro loadings reach `analytics.credit_risk_analysis`.
  - `_trim_anomaly_for_export`: added the §8.1 / §9.1 decomposition
    components (`flag_count_posterior_mean`, `flag_count_ci_low/high`,
    `magnitude_posterior_mean`, `combined_anomaly_score`,
    `dominant_flag_category`) and §9.2 diagnostic parity fields
    (`tail_df`, `cond_volatility`, `r_hat`, `ess_bulk`, `ess_tail`,
    `schema_version`) so the anomaly export table exposes both channels
    of the decoupled flag / magnitude posteriors.
  - `build_tri_model_alignment` already auto-detects the per-stock
    `tail_df` column added in 0.9.7 (T-A); no further edits required.
  - Dividend-safety and earnings-beat exports use the generic
    `reorder_with_identifiers` + `export_to_db` path, so the new
    `cut_probability_1y/3y`, `payout_sustainability_score`,
    `fcf_coverage_posterior_mean`, `continuation_probability` and
    `hazard_rate_next_quarter` columns flow through unchanged.
- **`exp_returns_v3_analytics.ipynb`** — no cell edits required: the
  notebook reads the exported analytics tables generically via
  `pd.read_sql` / the shared viz modules, which now resolve the new
  columns through the `_LEGACY_EXTRAS` alias map (see "Aligned —
  data_utils & visualizations modules" below). Re-running the notebook
  against a pipeline build from 0.9.7 / 0.9.8 picks up the new columns
  automatically.

### Aligned — data_utils & visualizations modules

To surface the v3.9 / v3.10 probability-model diagnostic and
decomposition columns across the notebook- / report-facing layer:

- **`probabilistic_ml_model/visualizations/_shared.py`** — extended the
  `_LEGACY_EXTRAS` alias map so `resolve_column()` now recognises the
  new Part-1 / Part-2 columns without breaking legacy consumers:
  diagnostic parity (`tail_df`, `cond_volatility`); accounting anomaly
  decomposition (`combined_anomaly_score`, `flag_count_posterior_mean`,
  `magnitude_posterior_mean`); dividend safety decomposition
  (`cut_probability_1y`, `cut_probability_3y`,
  `payout_sustainability_score`, `fcf_coverage_posterior_mean`); EPS
  streak posterior (`continuation_probability`,
  `hazard_rate_next_quarter`).
- **`probabilistic_ml_model/visualizations/quality_risk.py`** —
  `create_accounting_anomaly_dashboard` now prefers the new
  `combined_anomaly_score` column when present (§9.1), falling back to
  the legacy `accounting_anomaly_score`. All downstream panels continue
  to work via a normalised column name.
- **`probabilistic_ml_model/data_utils/`** — no source edits required:
  the new probability-model result fields surface as additional
  DataFrame columns that flow through `ExportConfig` /
  `export_to_db` / `export_to_csv` / `export_to_json` and are picked
  up by `reorder_with_identifiers` automatically.
  `aggregate_probability_results` is per-feature so new diagnostic
  columns pass through unchanged.
- Other viz modules (`expected_returns_viz`, `probability_viz`,
  `arviz_diagnostics`, `earnings_quality`, `growth_analysis`,
  `valuation`, `convergence_diagnostics`) already consume columns via
  `resolve_column()` / presence checks, so they automatically pick up
  the new columns via the alias-map update above — no targeted edits
  required.

[0.9.8]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.7...v0.9.8

## [0.9.7] - 2026-04-21

### Added — Probability Models v3.9 foundation (Part 1 of the model-improvement plan)

- **Cross-cutting T-B — `PosteriorDiagnostics` mixin** (
  `probabilistic_ml_model/statistical_functions/probability_models.py`).
  New shared dataclass centralising `tail_df`, `cond_volatility`, `cvar_5`,
  `r_hat`, `ess_bulk`, `ess_tail`, `divergences` so that Credit, Dividend,
  Beat and Streak result types report a consistent diagnostic schema.
- **§2.1 / §2.2 `CreditRiskResult` extensions.** Added `tail_df`,
  `cond_volatility`, `cvar_5`, `macro_loading`, `posterior_ess_bulk`,
  `posterior_ess_tail`, `r_hat`, `schema_version` fields plus `to_dict()`
  / `from_dict()` serialisation contract (validated against
  `analytics.credit_risk_analysis` export columns).
- **§4.1 / §4.2 `DividendSafetyResult` extensions.** Added FCF coverage
  decomposition (`fcf_coverage_posterior_mean`, `fcf_coverage_ci_low/high`,
  `cut_probability_1y`, `cut_probability_3y`, `payout_sustainability_score`,
  `stress_scenario_cut_prob`) and tail/vol parity fields aligned with
  `CreditRiskResult`.
- **§5.1 Bayesian streak continuation posterior** on `EPSStreakAnalyzer`.
  New method `compute_bayesian_continuation_posterior` replaces the
  heuristic `0.65 × 0.85^streak` point-estimate with a proper
  Beta-Binomial posterior keyed on
  `ReportedEPSHistory.count_yoy_improvements()`. Exposes
  `posterior_alpha/beta`, `continuation_prob_ci_low/high`,
  `expected_streak_length_years`, `hazard_rate_next_quarter`,
  `effective_sample_size` on `EPSStreakResult` (§6.1). Optional §5.2
  forward-revision momentum tilt (Student-t logit-space shrinkage,
  strength controlled by `revision_tilt_strength`).
- **§6.1 / §6.2 `EPSStreakResult` extensions.** New posterior / CI /
  hazard fields plus `ess_bulk` / `r_hat` / `schema_version` to align
  naming with `BeatProbabilityEstimate`.
- **§7.1 / §7.4 `ReportedEPSHistory` hardening.** Added
  `MIN_REPORTS_FOR_STREAK = 4` gate on
  `count_quarterly_beats_vs_estimate`, `has_sufficient_streak_history`
  helper, and `unique_quarterly_series` (adjacent-duplicate dedup
  proxy for restatement collapse — dataclass slots are positional so
  true `report_date` dedup is deferred pending upstream schema change).
- **§3.1 `DividendCutProbabilityModel` GARCH/df-floor parity.** Added
  `use_garch_volatility` and `student_t_df_floor` constructor flags so
  the Dividend model mirrors the Credit / PriceTarget API surface.
  Structured `logger.debug` emission of the active config.
- **§1.2 `CreditRiskProbabilityModel` validation + logging.** Fail-fast
  `ValueError` if `student_t_df_floor < 2.0`; structured `logger.info`
  emission on every instantiation surfacing `student_t`, `garch`,
  `macro`, `df_floor`, `n_samples`, `burn_in` so pipeline diagnostics
  can confirm the tail-aware configuration is actually consumed.

### Changed

- **T-A Per-stock tail-df wiring.** `build_tri_model_alignment` in both
  `probabilistic_ml_model/statistical_functions/ensemble_models.py` and
  the mirrored implementation in `expected_returns_v3.py` now auto-detect
  a `tail_df` column on the PT / MC / Kalman inputs and compute a
  **per-stock** `tail_penalty` (0.5 / 0.75 / 1.0 buckets by df ≤ 3 /
  5 / ∞). Output frames gain a new `tail_df` column. Falls back to the
  prior global `student_t_df` scalar when no per-stock column is
  present (backwards compatible).

### Notes — deferred Part-1 tasks (to be addressed in follow-up work)

The following items from the Priority 1–5 sequencing are **not**
included in 0.9.7 because they require deeper MCMC sampler rewrites
and dedicated validation fixtures; they are explicitly flagged for
follow-up:

- **§1.1** Hierarchical region/sector distress priors seeded from
  `.cache/mcmc_results/credit_risk/*.json` (empirical Bayes).
- **§1.3** MCMC regression on `{yield_curve_10y2y, vix, dxy, hy_oas}`
  inside `CreditRiskProbabilityModel._apply_mcmc_posteriors` (the
  `use_macro_covariates` flag is wired end-to-end at API level but
  the sampler term is not yet added).
- **§1.4 / §5.3** Rolling-window `backtest()` methods on
  `CreditRiskProbabilityModel` and `EPSStreakAnalyzer`.
- **§3.2 / §3.3 / §3.4** FCF × leverage interaction, mixture
  likelihood for structural cuts, and policy-stickiness Markov prior
  on `DividendCutProbabilityModel`.
- **§7.2 / §7.3** Reporting-frequency inference and FX normalisation
  against the `currencies` table (requires upstream schema wiring
  of `report_date` onto `ReportedEPSHistory`).
- **T-C** Regression test fixtures under
  `tests/test_credit_dividend_streak.py` pinning posterior means
  vs cached `.cache/mcmc_results/*` JSON.
- **T-D** Empirical-Bayes bootstrap script for quarterly prior
  refresh.

[0.9.7]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.6...v0.9.7

## [0.9.6] - 2026-04-16

### Changed

- Migrated to ArviZ 1.0: fixed `az.from_dict()` usage across the package, updated `Pipfile` dependencies, and resolved
  compatibility bugs ([`3a06d8c`](https://github.com/Kabenge42/PML_Finance_Project/commit/3a06d8c))
- Migrated analytics modules to the new `probabilistic_ml_model` package ([
  `b157963`](https://github.com/Kabenge42/PML_Finance_Project/commit/b157963))

### Removed

- Deprecated inspection profiles and statistical-functions modules from `probabilistic_ml_model` ([
  `fec6015`](https://github.com/Kabenge42/PML_Finance_Project/commit/fec6015))

### Added

- Expected-returns analytics log file for pipeline tracking ([
  `b157963`](https://github.com/Kabenge42/PML_Finance_Project/commit/b157963))

[0.9.6]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.5...v0.9.6