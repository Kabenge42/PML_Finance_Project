Based on the pipeline logs and code analysis, here are the key issues and recommended improvements:

---

### Issue 1: Ensemble Alignment Treats Risk Filters as Directional Models

**Problem:** In `build_quad_model_alignment()` (line 2972–2981), all 7 signals are summed equally. However, MC/Kalman/PT
are **directional return models** (bullish = positive return), while credit/dividend/anomaly are **risk filters** (
pass/fail quality gates). Treating them identically inflates the denominator and makes full consensus nearly
impossible — only **34/5523** stocks achieve 7/7.

**Evidence from logs:**

- Credit safe: 1,007/5,523 (18.2%)
- Div safe: 1,082/5,523 (19.6%)
- Anomaly clean: 1,118/5,523 (20.2%)
- Full 7/7 consensus: only 34

**Refactoring task:**

- Separate the ensemble into a **two-tier scoring system**: a "directional agreement" score (MC + Kalman + PT +
  Earnings, 0–4) and a "risk quality" score (credit + dividend + anomaly, 0–3).
- Define `full_consensus` as directional agreement = 4/4 **AND** risk quality ≥ 2/3 (pass at least 2 of 3 risk gates),
  rather than requiring all 7/7.
- Add a `directional_agreement` and `risk_quality_score` column to the quad DataFrame instead of a single
  `quad_agreement`.

```python
# Proposed structure in build_quad_model_alignment():
quad["directional_agreement"] = (
    quad["mc_bullish"].astype(int)
    + quad["kal_bullish"].astype(int)
    + quad["pt_bullish"].astype(int)
    + quad["beat_bullish"]
)
quad["risk_quality_score"] = (
    quad["credit_safe"] + quad["div_safe"] + quad["anomaly_clean"]
)
quad["full_consensus"] = (quad["directional_agreement"] == 4) & (quad["risk_quality_score"] >= 2)
```

---

### Issue 2: Overly Aggressive `fillna` Defaults for Missing Risk Data

**Problem:** In `build_quad_model_alignment()` (lines 2915, 2937, 2959), when credit/dividend/anomaly data is missing
for a stock (left join produces NaN), the code fills with worst-case values:

- `distress_probability.fillna(1.0)` → assumes maximum distress
- `dividend_cut_probability.fillna(1.0)` → assumes certain dividend cut
- `anomaly_severity_score.fillna(1.0)` → though this should be higher (score is 0–100+)

**Refactoring task:**

- Replace worst-case fillna with **median or neutral values** from the available data, or use a dedicated
  `"coverage_flag"` column to distinguish "no data" from "failed the filter."
- Consider using `fillna(np.nan)` and excluding stocks without coverage from the risk score rather than penalizing them.

```python
# Instead of:
quad["distress_probability"] = quad["distress_probability"].fillna(1.0)

# Use:
median_distress = credit_slim["distress_probability"].median()
quad["distress_probability"] = quad["distress_probability"].fillna(median_distress)
quad["credit_coverage"] = quad["distress_probability"].notna().astype(int)
```

---

### Issue 3: Anomaly Severity Threshold is Too Restrictive

**Problem:** The default `anomaly_severity_threshold=50` (line 2846) rejects ~80% of stocks because the log shows **mean
severity = 75.32, median = 74.69**. Only stocks with severity < 50 pass, which is well below the median.

**Evidence from logs:**

```
Severity score — mean: 75.32, median: 74.69, max: 173.00
Anomaly clean: 1118/5523 stocks flagged anomaly-clean
```

**Refactoring task:**

- Raise the default `anomaly_severity_threshold` to a **percentile-based** value (e.g., 25th percentile of the
  distribution, ~60–65), or use the median as the threshold.
- Better yet, make it **data-adaptive**: compute the threshold from the actual distribution at runtime.

```python
# In build_quad_model_alignment() or PipelineConfig:
if anomaly_severity_threshold is None:
    anomaly_severity_threshold = anomaly["anomaly_severity_score"].quantile(0.50)
```

---

### Issue 4: Inconsistent Consensus Models Between Step 6 and Step 7

**Problem:** Step 6 (`build_quad_model_alignment`) computes a 7-model `quad_agreement` score, but Step 7 (
`build_expected_returns_summary`, line 3312–3318) independently recomputes a **4-model** `agreement_score` (MC +
Kalman + PT + Earnings only). The main function (line 5069–5073) then tries to pick between them, leading to confusing
log output:

- Step 6 logs: `full consensus (7/7): 34`
- Step 7 logs: `Full consensus (4/4): 30`

**Refactoring task:**

- Remove the redundant 4-model `agreement_score` computation in `build_expected_returns_summary()`.
- Instead, merge the `quad_agreement` (or the proposed two-tier scores) from the quad alignment result directly into the
  summary DataFrame.
- This ensures a **single source of truth** for consensus scoring across the pipeline.

```python
# In build_expected_returns_summary(), replace lines 3306-3319 with:
if "quad_agreement" in quad.columns:
    quad_scores = quad[["ticker", "quad_agreement", "directional_agreement", 
                         "risk_quality_score", "signal"]].drop_duplicates(subset="ticker")
    summary = summary.merge(quad_scores, on="ticker", how="left")
```

---

### Issue 5: Quality Scoring Ignores Probabilistic Model Outputs

**Problem:** `filter_quality_stocks()` (line 2702–2731) and `rank_stocks_by_composite_score()` (screening.py line
882–942) use only **static fundamental metrics** (`piotroski_f_score`, `combined_distress_score`,
`earnings_quality_composite`, `cash_flow_quality_score`) with equal 0.25 weights. None of the probabilistic model
outputs (beat probability, achievement probability, MC confidence, Kalman variance) factor into quality scoring.

**Result:** Only **6 stocks** are "high-quality full consensus" — the quality tier and model consensus are almost
orthogonal.

**Refactoring task:**

- Create a **model-aware composite score** that blends fundamental quality with probabilistic model confidence:

```python
weights = {
    "piotroski_f_score": 0.15,           # Fundamental quality
    "combined_distress_score": 0.15,      # Credit quality
    "earnings_quality_composite": 0.10,   # Earnings quality
    "cash_flow_quality_score": 0.10,      # Cash flow quality
    "prob_positive_upside": 0.15,         # MC model confidence
    "achievement_probability": 0.15,      # PT model confidence
    "prob_beat_given_momentum": 0.10,     # Earnings beat probability
    "confidence_score": 0.10,             # Earnings model confidence
}
```

- Also consider adjusting `quality_tier` bins: the current bins `[0, 30, 50, 70, 100]` produce 1674 "High" and 1936 "
  Above Avg" (65% of universe), making the tier too lenient for fundamentals but too strict when intersected with
  consensus.

---

### Issue 6: Cross-Model Diagnostics Mix Percentage Returns with Dollar Price Targets

**Problem:** `compute_cross_model_diagnostics()` (line 3659–3665) computes dispersion and Kendall τ across:

- `implied_return_mc` (percentage, e.g., 20%)
- `price_target_mc` (dollar, e.g., $150)
- `implied_return_kalman` (percentage)
- `price_target_kalman` (dollar)
- `implied_return_pt` (percentage)

Mixing these scales causes **mean dispersion = 3614.70** (dominated by dollar-valued columns) and **spurious negative
Kendall τ** between return% and price$ columns (e.g., `implied_return_mc ↔ price_target_mc: -0.109`).

**Refactoring task:**

- Split diagnostics into two groups: **return-based** (`implied_return_*`) and **price-based** (`price_target_*`).
- Compute dispersion and concordance within each group separately.
- Alternatively, normalize all columns to z-scores before computing dispersion.

```python
return_cols = ["implied_return_mc", "implied_return_kalman", "implied_return_pt"]
price_cols = ["price_target_mc", "price_target_kalman", "price_target_prob_weighted"]
```

---

### Issue 7: Kalman Weight is Hardcoded in Confidence-Weighted Agreement

**Problem:** In `build_expected_returns_summary()` (line 3323), the Kalman model weight is hardcoded to `0.5`:

```python
kal_weight = 0.5
```

Meanwhile MC and PT weights are derived from their own confidence metrics. The Kalman filter produces a
`kalman_variance` column that could serve as a natural confidence measure.

**Refactoring task:**

- Derive `kal_weight` from `kalman_variance`: lower variance → higher confidence → higher weight.

```python
if "kalman_variance" in summary.columns:
    # Inverse variance weighting, clipped to [0.2, 0.9]
    max_var = summary["kalman_variance"].quantile(0.95)
    kal_weight = (1 - summary["kalman_variance"].clip(0, max_var) / max_var).clip(0.2, 0.9)
else:
    kal_weight = 0.5
```

---

### Issue 8: Strong Bullish Skew in Tri-Model (86% Strong Bullish)

**Problem:** 4,764 out of 5,523 stocks (86%) are "Strong Bullish (3/3)". This suggests the binary `> 0` threshold for
bullish classification (lines 2818–2820) is too permissive — any stock with even +0.01% implied return in all three
models gets "Strong Bullish."

**Refactoring task:**

- Replace the zero threshold with a **materiality threshold** (e.g., exceeds risk-free rate or a minimum return hurdle):

```python
min_return_threshold = 2.0  # Minimum 2% implied return to be considered bullish
summary["mc_bullish"] = summary["implied_return_mc"] > min_return_threshold
summary["kal_bullish"] = summary["implied_return_kalman"] > min_return_threshold
summary["pt_bullish"] = summary["implied_return_pt"] > min_return_threshold
```

- Add this as a configurable parameter in `PipelineConfig`:

```python
bullish_return_threshold: float = 2.0  # Minimum % return for bullish classification
```

---

### Summary of Refactoring Priority

| Priority | Task                                        | Impact                                  | File(s)                                                                |
|:---------|:--------------------------------------------|:----------------------------------------|:-----------------------------------------------------------------------|
| **P0**   | Two-tier ensemble (directional + risk)      | Fixes 7/7 consensus being 34 stocks     | `expected_returns_v3.py` lines 2972–3006                               |
| **P0**   | Separate return vs. price diagnostics       | Fixes 3614.70 dispersion & negative τ   | `expected_returns_v3.py` lines 3659–3665                               |
| **P1**   | Data-adaptive anomaly threshold             | Raises anomaly-clean from 20% to ~50%   | `expected_returns_v3.py` line 2846                                     |
| **P1**   | Remove redundant 4-model scoring in summary | Single source of truth for consensus    | `expected_returns_v3.py` lines 3306–3319                               |
| **P1**   | Materiality threshold for bullish           | Fixes 86% Strong Bullish skew           | `expected_returns_v3.py` lines 2818–2820, 3307–3310                    |
| **P2**   | Model-aware quality scoring                 | Increases high-quality consensus from 6 | `screening.py` lines 909–915, `expected_returns_v3.py` lines 2702–2731 |
| **P2**   | Neutral fillna for missing risk data        | Stops penalizing uncovered stocks       | `expected_returns_v3.py` lines 2915, 2937, 2959                        |
| **P2**   | Kalman variance-based weighting             | Better confidence-weighted agreement    | `expected_returns_v3.py` line 3323                                     |

