The current pipeline loads data from **three separate sources** in `_step_load_data` (line 3983), but
`mv_all_stock_features` — which is the **unified superset** of all feature data — is underutilized. Here's a detailed
breakdown of the issues and recommended refactoring.

---

### Current Data Loading Flow (Problem)

In `_step_load_data` (lines 3983–4064):

1. **`r.df`** ← `load_expected_returns_data()` → queries `mv_equities` (core equities only, ~90 columns)
2. **`r.df_all`** ← `load_all_stock_features()` → merges 17 individual `vw_features_*` views (~400+ columns)
3. **`r.df_features`** ← `load_analytics_table()` → queries `mv_all_stock_features` (~700+ columns, the **full superset
   **)

**Key problem**: `r.df_features` (from `mv_all_stock_features`) is only used once — as a fallback `_enrichment_source`
at line 4704:

```python
_enrichment_source = (
    df_features if not df_features.empty else df_all if not df_all.empty else df
)
```

Meanwhile, **all model runners** pass `r.df_all` (from the 17 views) as `feature_df`:

| Step    | Function                                      | `feature_df=` |
|:--------|:----------------------------------------------|:--------------|
| Step 3  | `run_price_target_achievement` (line 4132)    | `r.df_all`    |
| Step 5  | `run_earnings_beat_analysis` (line 4198)      | `r.df_all`    |
| Step 5b | `run_accounting_anomaly_analysis` (line 4234) | `r.df_all`    |
| Step 5c | `run_credit_risk_analysis` (line 4300)        | `r.df_all`    |
| Step 5c | `run_dividend_safety_analysis` (line 4335)    | `r.df_all`    |
| Step 5d | `run_stock_screening` (line 4379)             | `r.df_all`    |

This means the richer `mv_all_stock_features` data (which includes Enhancement 1–12 columns, composite scores, forward
consensus, etc.) is **not being fed to any model**.

---

### Recommended Refactoring

#### 1. Make `mv_all_stock_features` the primary dataset

Replace the three-load pattern with `mv_all_stock_features` as the single source of truth:

```python
def _step_load_data(cfg: PipelineConfig) -> PipelineResult:
    r = PipelineResult()
    catalog = get_feature_catalog()

    # Primary load: mv_all_stock_features (full superset)
    r.df_features = load_analytics_table()
    if not r.df_features.empty:
        r.df_features = _apply_backfill_and_kalman(r.df_features)
        r.df_features = r.df_features.fillna(0)
        _log_and_print(f"✓ Loaded mv_all_stock_features: {len(r.df_features):,} stocks × {len(r.df_features.columns)} features")

        # Use mv_all_stock_features as the main working datasets
        r.df = r.df_features  # replaces mv_equities load
        r.df_all = r.df_features  # replaces 17-view merge
    else:
        # Fallback: load from mv_equities + feature views separately
        r.df, r.id_coords = load_expected_returns_data()
        r.df_all, r.view_specs = load_all_stock_features()
        if r.df_all.empty:
            r.df_all = r.df.copy()

    # ... rest of step (schema metadata, historical drift, etc.)
```

#### 2. Update `load_analytics_table` to apply backfill & Kalman

Currently `load_analytics_table` (line 1392) only does `fillna(0)`. It should also apply `_apply_backfill_and_kalman`:

```python
def load_analytics_table(...) -> pd.DataFrame:
    # ... existing load logic ...
    if df is not None and not df.empty:
        df = _apply_backfill_and_kalman(df)  # ADD THIS
        df = df.fillna(0)
        # ... build IdentifierCoordinates, reconcile feature categories ...
```

#### 3. Update model runner `feature_df` parameters

Change all model step functions to pass `r.df_features` instead of `r.df_all`:

```python
# _step_price_target (line 4132)
r.pt = run_price_target_achievement(r.df_enriched, feature_df=r.df_features)

# _step_earnings_beat (line 4197-4198)
r.beat = run_earnings_beat_analysis(
    r.df_features if not r.df_features.empty else r.df,
    feature_df=r.df_features
)

# _step_anomaly_detection (line 4234)
r.anomaly_results = run_accounting_anomaly_analysis(r.df, feature_df=r.df_features, ...)

# _step_credit_dividend (lines 4298-4335)
r.credit = run_credit_risk_analysis(r.df, feature_df=r.df_features, ...)
r.div_safety = run_dividend_safety_analysis(r.df_features, feature_df=r.df_features, ...)

# _step_screening (line 4379)
r.screens = run_stock_screening(r.df_features, min_pct=cfg.screening_min_pct)
```

#### 4. Update `_enrichment_source` to prefer `df_features` explicitly

Line 4704 already does this, but make it the **only** path:

```python
_enrichment_source = r.df_features if not r.df_features.empty else r.df_all
```

#### 5. Reconcile feature categories against `mv_all_stock_features` columns

In `load_expected_returns_data` (line 1311), the feature categories are reconciled against `r.df` (mv_equities). This
should be done against `r.df_features` instead, since `mv_all_stock_features` has the full column set:

```python
feature_categories = reconcile_feature_categories(feature_categories, set(df_features.columns))
```

#### 6. Column references that need verification

The following column references in the pipeline should be verified against the `mv_all_stock_features` schema:

| Pipeline Location                   | Column Reference                                                      | Status in `mv_all_stock_features`                                  |
|:------------------------------------|:----------------------------------------------------------------------|:-------------------------------------------------------------------|
| `_step_monte_carlo` (line 4081)     | `last_price`, `price_target`, `price_target_high`, `price_target_low` | ✅ Present                                                          |
| `_step_kalman` (line 4178)          | `filtered_upside`                                                     | Model output, not source column                                    |
| `_step_earnings_beat` (line 4209)   | `eps_surprise_pct`, `eps_revision_momentum`, `analyst_conviction`     | ✅ Present                                                          |
| `_step_credit_dividend` (line 4306) | `distress_risk_score`, `debt_to_equity`, `altman_z_score`             | ✅ Present                                                          |
| `_step_screening` (line 4357)       | `piotroski_f_score`, `fcf_positive_years`, `eps_trajectory_score`     | ✅ Present                                                          |
| `build_expected_returns_summary`    | `sector`, `industry`, `region`, `country`                             | ✅ Present                                                          |
| Historical drift (line 4057)        | `price_target_*_ago` columns                                          | ❌ Not in `mv_all_stock_features` — these are in `mv_equities` only |

**Critical note**: Historical price snapshot columns (`price_5d_ago`, `price_1m_ago`, `price_target_1w_ago`, etc.) are
in `mv_equities` but **not** in `mv_all_stock_features`. The historical drift enrichment step (
`_enrich_with_historical_target_drift`, line 4059) still needs `r.df` from `mv_equities` for these columns. You should
either:

- Keep a lightweight `mv_equities` load just for historical columns, then merge into `df_features`
- Or add historical snapshot columns to `mv_all_stock_features` SQL definition

#### 7. Eliminate redundant `load_all_stock_features` call

Since `mv_all_stock_features` is built from the same 17 `vw_features_*` views plus equities data, loading both is
redundant. The `load_all_stock_features()` call (line 4004) can be removed entirely when `mv_all_stock_features` loads
successfully.

#### 8. Update `PipelineConfig` and `PipelineResult`

The `PipelineResult` dataclass (line 596) already has `df_features` field. Consider adding a flag to `PipelineConfig` to
control the data source strategy:

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    prefer_materialized_view: bool = True  # Use mv_all_stock_features as primary
```

---

### Summary of Changes

| Change                                                      | File                     | Lines                         |
|:------------------------------------------------------------|:-------------------------|:------------------------------|
| Make `mv_all_stock_features` the primary load               | `expected_returns_v3.py` | `_step_load_data` (3983–4064) |
| Add backfill+Kalman to `load_analytics_table`               | `expected_returns_v3.py` | 1392–1444                     |
| Update all `feature_df=` params to use `r.df_features`      | `expected_returns_v3.py` | 4132, 4198, 4234, 4300, 4335  |
| Reconcile categories against `df_features`                  | `expected_returns_v3.py` | 1311                          |
| Keep `mv_equities` load for historical snapshot cols only   | `expected_returns_v3.py` | 3997                          |
| Remove redundant `load_all_feature_views` call              | `expected_returns_v3.py` | 4004                          |
| Merge historical cols from `mv_equities` into `df_features` | `expected_returns_v3.py` | New code in `_step_load_data` |

All 700+ columns in `mv_all_stock_features` (including Enhancement 1–12 features, composite scores, forward consensus,
volatility surface, tax rate, OpEx temporal, asset sale, FCF estimate curve, dividend history, investment income
temporal, share dilution tracking) are already snake_case aliased and match the column names expected by the pipeline's
model runners and screening functions.