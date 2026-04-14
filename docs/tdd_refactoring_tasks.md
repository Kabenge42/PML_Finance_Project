# TDD Refactoring Tasks — Expected Returns Pipeline Statistical Fixes

Generated: 2026-04-14
Test file: `tests/test_pipeline_statistical_fixes.py`
Status: **6 RED / 11 GREEN** — production fixes required to turn RED tests GREEN.

---

## Task Overview

| # | Priority    | Issue                             | Test Class                           | RED Tests | Target Files                                    |
|---|-------------|-----------------------------------|--------------------------------------|-----------|-------------------------------------------------|
| 1 | 🔴 Critical | MC return winsorization           | `TestMCReturnWinsorization`          | 1         | `pipeline_runners.py`, `expected_returns_v3.py` |
| 2 | 🔴 Critical | Scale-aware agreement thresholds  | `TestScaleAwareAgreement`            | 1         | `ensemble_models.py`                            |
| 3 | 🟠 High     | Heavy-tail risk metric clipping   | `TestHeavyTailClipping`              | 1         | `ensemble_models.py`                            |
| 4 | 🟠 High     | Bayesian prior–likelihood balance | `TestBayesianPriorLikelihoodBalance` | 0 (GREEN) | Already at 0.3 — verify IQR in production       |
| 5 | 🟡 Medium   | Degenerate distress rescaling     | `TestDegenerateDistressRescaling`    | 0 (GREEN) | `pipeline_runners.py`, `expected_returns_v3.py` |
| 6 | 🟡 Medium   | Quantile-based quality tier bins  | `TestQuantileQualityTierBins`        | 1         | `expected_returns_v3.py`, `pipeline_runners.py` |
| 7 | 🟢 Low      | MC coverage gap logging           | `TestMCCoverageGapLogging`           | 1         | `pipeline_runners.py`, `expected_returns_v3.py` |
| 8 | —           | v3/pipeline_runners consistency   | `TestV3PipelineRunnerConsistency`    | 1         | Both modules                                    |

---

## Task 1 — MC Return Winsorization (Critical)

**Problem:** Monte Carlo `implied_return_mc` has max 897%, skew 5.24, kurtosis 58.8 — not
winsorized, unlike Kalman which clips at 1st/99th percentile.

**RED test:** `test_mc_returns_clipped_after_run_monte_carlo_analysis`

**Fix location:**

- `probabilistic_ml_model/pipeline_runners.py` → `run_monte_carlo_analysis()` (line ~625)
- `expected_returns_v3.py` → `run_monte_carlo_analysis()` (line ~1509)

**Implementation:**

```python
# After mc = monte_carlo_price_target_simulation(...)
if not mc.empty and "implied_return_mc" in mc.columns:
    lower, upper = mc["implied_return_mc"].quantile([0.01, 0.99])
    mc["implied_return_mc"] = mc["implied_return_mc"].clip(lower, upper)
```

**Acceptance:** `test_mc_returns_clipped_after_run_monte_carlo_analysis` and
`test_both_modules_winsorize_mc` turn GREEN.

---

## Task 2 — Scale-Aware Weighted Agreement (Critical)

**Problem:** Uniform `bullish_return_threshold=10.0` applied to MC (mean 28%), Kalman (mean 26%),
and PT (mean 7%) creates systematic bias — MC/Kalman almost always bullish, PT rarely.

**RED test:** `test_build_tri_model_uses_scale_aware_thresholds`

**Fix location:**

- `probabilistic_ml_model/statistical_functions/ensemble_models.py` → `build_tri_model_alignment()` (line ~141)
- `expected_returns_v3.py` → `build_tri_model_alignment()` (corresponding location)

**Implementation:**

```python
# Replace uniform threshold with model-specific percentile-based thresholds
mc_threshold = max(bullish_return_threshold, float(tri["implied_return_mc"].quantile(0.40)))
kal_threshold = max(bullish_return_threshold, float(tri["implied_return_kalman"].quantile(0.40)))
pt_threshold = max(bullish_return_threshold * 0.3, float(tri["implied_return_pt"].quantile(0.40)))

tri["mc_bullish"] = tri["implied_return_mc"] > mc_threshold
tri["kal_bullish"] = tri["implied_return_kalman"] > kal_threshold
tri["pt_bullish"] = tri["implied_return_pt"] > pt_threshold
```

**Acceptance:** Bullish rate gap between MC and PT < 30 percentage points.

---

## Task 3 — Heavy-Tail Risk Metric Clipping (High)

**Problem:** `pt_spread` (kurtosis 806) and `risk_reward_ratio` (kurtosis 1148) have extreme
outliers that dominate composite scoring and visualizations.

**RED test:** `test_heavy_tail_cols_clipped_in_summary`

**Fix location:**

- `probabilistic_ml_model/statistical_functions/ensemble_models.py` → `build_expected_returns_summary()` (after
  market-data merge, ~line 657)

**Implementation:**

```python
# After merging market-data columns, before downstream scoring
_HEAVY_TAIL_COLS = ["pt_spread", "risk_reward_ratio", "upside_std"]
for col in _HEAVY_TAIL_COLS:
    if col in summary.columns:
        lo, hi = summary[col].quantile(0.01), summary[col].quantile(0.99)
        summary[col] = summary[col].clip(lo, hi)
```

**Acceptance:** Kurtosis of clipped columns < 50 in test.

---

## Task 4 — Bayesian Prior–Likelihood Balance (High)

**Problem:** `resampled_posterior_mean` IQR of only 0.077 — prior overwhelms likelihood.

**Status:** ✅ GREEN — `momentum_prior_strength` is already 0.3 in the current codebase.

**Follow-up:** Monitor production IQR of `resampled_posterior_mean`. If still < 0.10, consider
increasing `n_posterior_samples` to 10,000 and `n_chains` to 12 in `run_resampled_posterior_analysis`.

---

## Task 5 — Degenerate Distress/Safety Score Rescaling (Medium)

**Problem:** `distress_risk_score` median is 100 (maximum), `safety_score` median is 5 (minimum) —
over half the universe gets identical scores with no discriminative power.

**Status:** ✅ GREEN (unit tests pass with manual rescaling logic).

**Integration fix needed in:**

- `pipeline_runners.py` → `run_credit_risk_analysis()` (after `credit_model.analyze_dataframe()`)
- `expected_returns_v3.py` → `run_credit_risk_analysis()` (same location)

**Implementation:**

```python
if "distress_risk_score" in credit.columns:
    at_max = (credit["distress_risk_score"] >= credit["distress_risk_score"].max()).mean()
    if at_max > 0.40:
        logger.warning(
            "distress_risk_score degenerate: %.0f%% at maximum — applying percentile rescaling",
            at_max * 100,
        )
        credit["distress_risk_score"] = credit["distress_risk_score"].rank(pct=True) * 100
```

---

## Task 6 — Quantile-Based Quality Tier Bins (Medium)

**Problem:** Fixed bins `[18, 25, 35, 45, 55, 60, 75]` produce highly uneven tier sizes given
the composite score distribution (mean 44.8, std 9.6).

**RED test:** `test_filter_quality_stocks_uses_adaptive_bins`

**Fix location:**

- `expected_returns_v3.py` → `filter_quality_stocks()` (line ~2656)
- `probabilistic_ml_model/pipeline_runners.py` → `filter_quality_stocks()` (if present)

**Implementation:**

```python
valid_scores = summary["composite_score"].dropna()
if len(valid_scores) > 100:
    q_bins = [
        valid_scores.min() - 0.01,
        valid_scores.quantile(0.10),
        valid_scores.quantile(0.30),
        valid_scores.quantile(0.50),
        valid_scores.quantile(0.70),
        valid_scores.quantile(0.90),
        valid_scores.max() + 0.01,
    ]
    q_bins = sorted(set(q_bins))
    if len(q_bins) >= 7:
        tier_labels = ["Very Low", "Low", "Below Avg", "Above Avg", "High", "Premium"]
    else:
        q_bins = [18, 25, 35, 45, 55, 60, 75]
        tier_labels = ["Very Low", "Low", "Below Avg", "Above Avg", "High", "Premium"]
else:
    q_bins = [18, 25, 35, 45, 55, 60, 75]
    tier_labels = ["Very Low", "Low", "Below Avg", "Above Avg", "High", "Premium"]

summary["quality_tier"] = pd.cut(
    summary["composite_score"],
    bins=q_bins,
    labels=tier_labels[:len(q_bins) - 1],
)
```

**Acceptance:** Max/min tier bucket ratio < 5.

---

## Task 7 — MC Coverage Gap Diagnostic Logging (Low)

**Problem:** 876 stocks (13%) missing from MC output with no diagnostic logging.

**RED test:** `test_coverage_gap_warning_logged`

**Fix location:**

- `probabilistic_ml_model/pipeline_runners.py` → `run_monte_carlo_analysis()` (after MC call)
- `expected_returns_v3.py` → `run_monte_carlo_analysis()` (same location)

**Implementation:**

```python
input_count = len(sim_df)
output_count = len(mc)
if output_count < input_count * 0.90:
    logger.warning(
        "MC coverage gap: %d/%d stocks (%.1f%%) processed — "
        "%d stocks likely missing required price target columns",
        output_count, input_count,
        output_count / input_count * 100,
        input_count - output_count,
    )
```

**Acceptance:** `test_coverage_gap_warning_logged` turns GREEN.

---

## Running the Tests

```powershell
# Run all TDD refactoring tests
pytest tests/test_pipeline_statistical_fixes.py -v

# Run a specific issue's tests
pytest tests/test_pipeline_statistical_fixes.py::TestMCReturnWinsorization -v
pytest tests/test_pipeline_statistical_fixes.py::TestScaleAwareAgreement -v
pytest tests/test_pipeline_statistical_fixes.py::TestHeavyTailClipping -v
pytest tests/test_pipeline_statistical_fixes.py::TestQuantileQualityTierBins -v
pytest tests/test_pipeline_statistical_fixes.py::TestMCCoverageGapLogging -v
pytest tests/test_pipeline_statistical_fixes.py::TestV3PipelineRunnerConsistency -v
```

## Verification Checklist

- [ ] All 6 RED tests turn GREEN after production fixes
- [ ] All 11 existing GREEN tests remain GREEN (no regressions)
- [ ] Existing test suite (`pytest tests/`) passes without regressions
- [ ] Pipeline log output shows new diagnostic warnings where expected
