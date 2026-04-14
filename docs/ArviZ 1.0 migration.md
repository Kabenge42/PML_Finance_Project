Based on the ArviZ 1.0 migration guide and analysis of the project, here are all the changes needed across configuration
files and the `probabilistic_ml_model` package.

---

### Configuration File Updates

#### `Pipfile` — Line 28 (Primary Issue)

The `Pipfile` is the **only configuration file still pinned to legacy ArviZ**. It needs to be updated to match
`requirements.txt` and `pyproject.toml`.

**Current (outdated):**

```toml
arviz = ">=0.19.0,<1.0.0"
```

**Required update — replace line 28 and add sub-packages:**

```toml
# --- Bayesian / Probabilistic ---
arviz = ">=1.0.0,<2.0.0"
arviz-base = ">=1.0.0,<2.0.0"
arviz-stats = ">=1.0.0,<2.0.0"
arviz-plots = ">=1.0.0,<2.0.0"
```

#### `requirements.txt` — Already Correct ✅

Lines 41–44 already specify the ArviZ 1.0 packages:

```
arviz>=1.0.0,<2.0.0
arviz-base>=1.0.0,<2.0.0
arviz-stats>=1.0.0,<2.0.0
arviz-plots>=1.0.0,<2.0.0
```

#### `pyproject.toml` — Already Correct ✅

Lines 51–54 already specify:

```toml
"arviz>=1.0.0,<2.0.0",
"arviz-base>=1.0.0,<2.0.0",
"arviz-stats>=1.0.0,<2.0.0",
"arviz-plots>=1.0.0,<2.0.0",
```

---

### Code Fixes in `probabilistic_ml_model` Package

#### Fix 1: `probability_models.py` — Lines 4219–4237 (Nested Dict Bug)

The `EarningsBeatBayesianModel.build_inference_data()` method passes a nested dict as the first positional argument to
`az.from_dict()`. This is the **same bug** that was previously fixed in `inference_schema.py` line 385.

**Current (buggy):**

```python
return az.from_dict(
    {
        "posterior": {"beat_probability": posterior_samples},
        "posterior_predictive": {"beat_outcome": pp_samples},
        "observed_data": {
            "base_posterior_mean": result_df["base_posterior_mean"].values,
            "momentum_signal": result_df["momentum_signal"].values,
        },
        "constant_data": {
            "momentum_weight": np.array([self.momentum_weight]),
            "volatility_weight": np.array([self.volatility_weight]),
        },
    },
    coords=coords,
    dims={...},
)
```

**Fix — use explicit keyword arguments:**

```python
return az.from_dict(
    posterior={"beat_probability": posterior_samples},
    posterior_predictive={"beat_outcome": pp_samples},
    observed_data={
        "base_posterior_mean": result_df["base_posterior_mean"].values,
        "momentum_signal": result_df["momentum_signal"].values,
    },
    constant_data={
        "momentum_weight": np.array([self.momentum_weight]),
        "volatility_weight": np.array([self.volatility_weight]),
    },
    coords=coords,
    dims={
        "beat_probability": ["chain", "draw", "equity"],
        "beat_outcome": ["chain", "draw", "equity"],
    },
)
```

#### Fix 2: `statistical_models.py` — Lines 1133–1136 (Nested Dict Bug)

The `hierarchical_mcmc_multi_level()` function has the same nested dict pattern.

**Current (buggy):**

```python
idata = az.from_dict(
    {"posterior": all_samples_for_idata},
    coords=all_coords_for_idata,
    dims={k: [k.replace("_mu", "")] for k in all_samples_for_idata},
)
```

**Fix:**

```python
idata = az.from_dict(
    posterior=all_samples_for_idata,
    coords=all_coords_for_idata,
    dims={k: [k.replace("_mu", "")] for k in all_samples_for_idata},
)
```

#### Fix 3: `inference_schema.py` — Lines 1606–1609 (Nested Dict Bug — Missed in Prior Fix)

The `build_feature_view_inference_data()` function at the bottom of the file was **missed** when the earlier fix was
applied at line 385.

**Current (buggy):**

```python
return az.from_dict(
    {"posterior": {v: posterior_ds[v].values for v in posterior_ds.data_vars}},
    coords={k: v.values for k, v in posterior_ds.coords.items()},
)
```

**Fix:**

```python
return az.from_dict(
    posterior={v: posterior_ds[v].values for v in posterior_ds.data_vars},
    coords={k: v.values for k, v in posterior_ds.coords.items()},
)
```

#### Fix 4: `probability_models.py` — Line 224 (ARVIZ_AVAILABLE Check)

The availability check only looks for `InferenceData`, which is **removed** in ArviZ 1.0 (replaced by `xr.DataTree`).
The compat shim injects it, but for robustness, align with the pattern in `inference_schema.py` line 41.

**Current:**

```python
ARVIZ_AVAILABLE = hasattr(az, "InferenceData")
```

**Fix — also accept `from_dict` (ArviZ 1.0 native API):**

```python
ARVIZ_AVAILABLE = hasattr(az, "from_dict") or hasattr(az, "InferenceData")
```

---

### Why the Nested Dict Pattern Fails

In ArviZ 0.23.4 (legacy), `az.from_dict()` has a signature where the first positional argument is `posterior`. When a
nested dict like `{"posterior": {...}, "observed_data": {...}}` is passed positionally, ArviZ interprets the **entire
nested dict** as a single posterior variable, causing dimension conflicts:

```
conflicting sizes for dimension 'draw': length 1 on the data but length 25000 on coordinate 'draw'
```

Using explicit keyword arguments (`posterior=`, `posterior_predictive=`, etc.) works correctly in **both** ArviZ 0.23.4
and ArviZ 1.0.

---

### No Changes Needed

| File                                  | Status    | Notes                                                                                                         |
|:--------------------------------------|:----------|:--------------------------------------------------------------------------------------------------------------|
| `_pymc_arviz_compat.py`               | ✅ Correct | Bridges arviz 1.0 sub-packages into `arviz.*` namespace for PyMC 5.x                                          |
| `__init__.py`                         | ✅ Correct | Calls `_patch_arviz()` before any model imports                                                               |
| `visualizations/_shared.py`           | ✅ Correct | Uses `_make_datatree()` (ArviZ 1.0 `xr.DataTree` pattern)                                                     |
| `visualizations/arviz_diagnostics.py` | ✅ Correct | Try `arviz_plots`/`arviz_stats`/`arviz_base` first, fallback to legacy `arviz`                                |
| `pml_models/*.py`                     | ✅ Correct | Try `import arviz`, fallback to `arviz_base`; use PyMC's `pm.sample()` which returns ArviZ-compatible objects |
| `requirements.txt`                    | ✅ Correct | Already specifies `arviz>=1.0.0` + sub-packages                                                               |
| `pyproject.toml`                      | ✅ Correct | Already specifies `arviz>=1.0.0` + sub-packages                                                               |

---

### Installation Note

After updating the `Pipfile`, upgrade the installed `arviz` package:

```powershell
pip install arviz==1.0.0
```

The `arviz` 1.0.0 meta-package on PyPI is a thin wrapper that depends on `arviz-base>=1.0.0,<1.1.0`,
`arviz-stats>=1.0.0,<1.1.0`, and `arviz-plots>=1.0.0,<1.1.0` (all already installed). The `_pymc_arviz_compat.py` shim
ensures PyMC 5.x can still import `from arviz import InferenceData, concat` even though ArviZ 1.0 removed those symbols.
