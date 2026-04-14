#### Bug 1 — Leading space in column name (Line 3121)

```python
_ANOMALY_COLS = [
    " gross_profit_margin_pct_fy",  # ← leading space!
    "gross_profit_margin_pct_ltm",
    ...
    ]
```

The string `" gross_profit_margin_pct_fy"` has a leading space, so it will **never match** an actual DataFrame column.
This means `gross_profit_margin_pct_fy` is silently excluded from the anomaly enrichment merge in
`build_expected_returns_summary`.

**Fix:** Remove the leading space → `"gross_profit_margin_pct_fy"`.

---

#### Bug 2 — `cfg` (PipelineConfig) shadowed by `ExportConfig` (Line 5578)

```python
cfg = ExportConfig(table_name=f"prob_{cat_name.lower().replace(' ', '_')}")
export_to_db(aggregated, cfg)
```

Inside `main()`, the local variable `cfg` is the `PipelineConfig` instance used throughout the pipeline. This line *
*reassigns** `cfg` to an `ExportConfig` object inside the loop. If any code after this point references `cfg` expecting
a `PipelineConfig` (e.g., `cfg.export_max_workers`), it will fail or behave incorrectly. Even though no such reference
currently exists after the loop, this is a latent bug waiting to happen with any future addition.

**Fix:** Use a different variable name, e.g., `export_cfg = ExportConfig(...)`.

---

#### Bug 3 — Docstring/signature mismatch on `clear_cache` (Lines 547–553)

```python
def clear_cache(self, *, expired_only: bool = True) -> int:
    """
    ...
    expired_only : bool, default False    # ← docstring says False
    ...
    """
```

The signature defaults `expired_only` to `True`, but the docstring says `default False`. This misleads callers who read
the docstring.

**Fix:** Update the docstring to say `default True`.

---

#### Bug 4 — Inconsistent caching defaults between `from_env()` and dataclass (Lines 502–503 vs 538–539)

```python
# Dataclass defaults:
enable_result_caching: bool = True
enable_mcmc_caching: bool = True

# from_env() defaults (when env vars are unset):
enable_result_caching = os.environ.get("ER_ENABLE_CACHING", "false").lower() == "true",  # → False
enable_mcmc_caching = os.environ.get("ER_ENABLE_MCMC_CACHING", "false").lower() == "true",  # → False
```

`PipelineConfig()` enables caching by default, but `PipelineConfig.from_env()` (used by `main()`) disables it. This
means the production pipeline **never caches** unless the env var is explicitly set, contradicting the dataclass
declaration. This is confusing and likely unintentional.

**Fix:** Align the `from_env` fallback strings to `"true"` to match the dataclass defaults, or change the dataclass
defaults to `False` if caching-off is the intended default.

---

#### Bug 5 — Local aliases in `main()` diverge from `PipelineResult` (Lines 4734–4745 → 5076–5077)

```python
# Line 4734-4738: aliases created from r
mc, pt, kal, beat, credit, div_safety = r.mc, r.pt, r.kal, r.beat, r.credit, r.div_safety
...
df, df_all, df_features = r.df, r.df_all, r.df_features

# Line 5076-5077: local aliases are mutated (enriched)
df = _enrich_dataframe(df, _viz_source, _viz_needed_cols, "df (mv_equities)")
beat = _enrich_dataframe(beat, _viz_source, _viz_needed_cols, "beat")
```

After Step 8b, the local `df` and `beat` are **replaced** with enriched copies (since `_enrich_dataframe` likely returns
a new DataFrame). But `r.df` and `r.beat` still point to the **original** un-enriched DataFrames. The sync-back at line
5641 (`r.beat = beat`) happens at the very end, but any step function called between enrichment and sync-back that reads
from `r` would get stale data.

Similarly, `summary` is reassigned at line 4752 and further modified (lines 4791, 4799) but `r.summary` is only updated
at line 5643. If any intermediate code accesses `r.summary`, it gets the empty default.

**Recommendation:** Write enriched values back to `r` immediately after mutation, or operate directly on `r.xxx`
attributes instead of local aliases.

---

#### Bug 6 — `quad_agreement == 4` hardcoded in final summary (Line 5620)

```python
if not quad.empty:
    _log_and_print(f"    Quad-model full consensus: {(quad['quad_agreement'] == 4).sum()}")
```

The quad-model alignment supports 4–7 models, so full consensus could be 5, 6, or 7. This line hardcodes `== 4`, which *
*undercounts** full consensus when credit, dividend safety, or anomaly models are active. Earlier in the code (line
2977), `n_models` is correctly computed dynamically.

**Fix:** Use `quad['quad_agreement'].max()` or compute `n_models` dynamically as done in `build_quad_model_alignment`.

---

#### Bug 7 — `summary["agreement_score"] == 4` in final summary ignores ensemble models (Line 5622)

```python
full_consensus = (summary["agreement_score"] == 4).sum()
```

When 7 models are active, `summary` may have `quad_agreement` as the correct consensus column (range 0–7), but this line
checks `agreement_score` (range 0–4). Earlier in Step 7 (line 4777), the code correctly picks between `quad_agreement`
and `agreement_score`, but the final summary section doesn't.

**Fix:** Use the same `_consensus_col` / `_max_agreement` logic from Step 7.

```