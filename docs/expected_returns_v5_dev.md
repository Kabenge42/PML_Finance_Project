# **Expected Returns Model Analysis**

## Expected Returns Notebook for all calculated stock features

### Covers all pymc models from :

- `probabilistic_ml_model.pymc_models.EarningsBeatModel.EarningsBeatBayesian` — hierarchical Beta-Binomial
  beat-probability model with DB-aligned `pm.Data` containers (`n_total`, `n_beats`, `sector_idx`, `earnings_features`).
- `probabilistic_ml_model.pymc_models.PriceTargetModel.PriceTargetAchievement` — hierarchical Beta-Binomial
  beat-probability model with DB-aligned `pm.Data` containers (`n_total`, `n_beats`, `sector_idx`, `pt_features`).
- `probabilistic_ml_model.pymc_models.KalmanFilterModel.KalmanFilterPriceTarget` — hierarchical Beta-Binomial
  beat-probability model with DB-aligned `pm.Data` containers (`n_total`, `n_beats`, `sector_idx`,
  `kalman_features_df`).
- `public.mv_all_stock_features` — main dataframe for model analyis.
- `public.calculated_features_registry` (category = `Earnings`) — drives the `earnings_feature` coord labels.

**Core PML Models:**

- **Monte Carlo Simulation** — Probabilistic upside/downside distributions with historical target drift
- **Price Target Achievement** — Probability-weighted expected returns with analyst sentiment & risk adjustment
- **Kalman Filtered Targets** — Noise-reduced price target signals with momentum-informed priors
- **Earnings Beat Analysis** — Three-layer Bayesian earnings beat probability with quality filters
- **Credit Risk Analysis** — Bayesian distress estimation with debt trajectory & balance sheet strength
- **Dividend Safety Analysis** — Dividend cut probability with FCF coverage & leverage signals
- **Accounting Anomaly Detection** — Multi-layered statistical anomaly detection with Mahalanobis distance

**Probabilistic Linear Market Model (MM) Regression**

- **Monte Carlo Simulation** — Probabilistic upside/downside distributions with historical target drift
- **Price Target Achievement** — Probability-weighted expected returns with analyst sentiment & risk adjustment
- **Kalman Filtered Targets** — Noise-reduced price target signals with momentum-informed priors
- **DCF Price Target Model** — discounted cash flow regression

**Probabilistic ML Ensembles**,

- **Prior Probability Distributions P(a, b, e)**
- **Likelihood Function P(Y| a, b, e, X)**
- **Marginal Likelihood Function P(Y|X)**
- **Posterior Probability Distributions P(a, b, e| X, Y)**
- **Multi-Level Hierarchical MCMC** — Cross-category shrinkage (region, country, sector, industry, style, size)
- **Feature View Posterior Panels** — Per-view InferenceData with ArviZ diagnostics

**Statistical Functions:**

- **Bayesian Category Analysis** — Per-feature-category posterior estimation
- **Gaussian Copula Dependency** — Tail dependence & joint distribution modeling
- **Parallel MCMC Chains** — Gelman-Rubin convergence diagnostics
- **Resampled Posterior Returns** — Bayesian technical resampling from historical snapshots
- **Student-t MCMC** — Heavy-tail robust posterior inference
- **Distribution Fitting** — AIC-based best-fit selection (Normal, Student-t, Skew-normal, Laplace)
- **Category-Level Distributions** — Per-category credible intervals & posterior means
- **Conditional Probability Analysis** — Feature-level P(anomaly | conditions)
- **Risk Metrics** — VaR, CVaR, downside deviation, gain/loss ratio

**Stock Screening:**

- **Undervalued Stocks Screening** — Investment opportunities with low P/E and high ROE
- **Earnings Quality Screening** — EPS consistency, GAAP divergence, revision momentum
- **Accounting Anomaly Screening** — Financial statement quality & consistency
- **Value Opportunities Screening** — Valuation reversion candidates
- **Growth Momentum Screening** — Revenue/EPS acceleration with profitability filters
- **GARP** — Growth at a reasonable price
- **Dividend Quality Screening** — Yield safety with coverage & streak metrics
- **Financial Health Screening** — Altman Z-score, Piotroski F-score, distress risk
- **Integrity-Filtered Growth Screening** — Accounting quality & growth alignment
- **High-Yield Safe Dividends** — Sustainable yield with leverage constraints
- **Low-Volatility Quality** — Beta stability with profitability
- **FCF Compounders** — Free cash flow growth consistency
- **Total Return Leaders** — Price appreciation + dividend yield

## 12. Summary — Findings, Insights & `feature_catalogue`-Aligned Recommendations

This section consolidates the results of the seven PyMC models fitted in Sections 4–11 and maps each model's posterior
signal back to the canonical `public.feature_catalogue` schema (`category`, `feature_alias`, `source_function`,
`calculation_type`, `data_type`) that drives both `MODEL_FEATURE_CONTAINERS` and the per-model
`_resolve_*_feature_aliases()` helpers.

### 12.1 Cross-Model Findings

| Model                                               | Key Posterior                                               | Behaviour Observed                                                                                                                                              | NUTS Geometry                                                                                            |
|:----------------------------------------------------|:------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|
| **EarningsBeat** (`EarningsBeatBayesian`)           | `beat_prob[isin]`, `sector_rate[sector]`                    | Sector-hierarchical Beta-Binomial; non-centred logit-Normal hierarchy mixes well; `BetaBinomial` marginalisation gives ~3–4× speed-up with no per-stock latent. | Excellent (`r_hat ≈ 1.00`, ESS bulk > 2k).                                                               |
| **PriceTarget** (`PriceTargetAchievement`)          | `achieve_prob[isin]`, `risk_adj_return[isin]`               | Logit-Normal hierarchy on achievement probability; risk penalty `exp(-λ·conviction)` shrinks `expected_return` toward 0 for low-conviction names.               | Good; `expected_return` σ proportional to `analyst_conviction` removes the prior=observation degeneracy. |
| **Kalman** (`KalmanFilterPriceTarget`)              | `log_state[time]`, `state[time]`                            | Non-centred GRW on log-prices; clamped scale (`σ ∈ [1e-3, 1.0]`) eliminates the float64 overflow seen on raw price levels.                                      | Stable; `marginalized` mode is fastest when no smoothed posterior is needed.                             |
| **DCF** (`DCFPriceTarget`)                          | `intrinsic_value`, `fcf_growth`, `wacc`                     | Non-centred Normal on `fcf_growth`; truncated WACC (`> terminal_growth + 0.005`) keeps Gordon-growth finite.                                                    | Good; sigma=500 likelihood is the dominant uncertainty source.                                           |
| **DividendSafety** (`DividendSafetyBayesian`)       | `cut_prob[isin]`, `risk_adj`, `expected_coverage`           | Conditional 1.3× cut-prob bump above `payout_ratio > 0.9`; `expected_coverage = clip(1/(risk_adj+0.01), 0, 20)` gives interpretable FCF-coverage scale.         | Good; logit-Normal hierarchy preferred over centred Beta.                                                |
| **CreditRisk** (`CreditRiskBayesian`)               | `distress_prob[isin]`, `expected_distress`, `sector_rate`   | Altman-zone adjustment precomputed as `pm.Data` (avoids nested `pt.switch` fusion errors); shared `debt_slope` replaces per-ISIN `debt_trend` latent.           | Excellent; kernel-fusion fix removed the previous compile failures.                                      |
| **AccountingAnomaly** (`AccountingAnomalyBayesian`) | `anomaly_prob[isin]`, `feature_scale[feature]`, `threshold` | Empirical z-score matrix consumed as data; non-centred log-Normal `feature_scale = exp(μ+σ·z)`.                                                                 | Excellent; ~60k spurious latents removed → seconds vs 15 min previously.                                 |

### 12.2 Insights

1. **Registry-driven feature alignment works end-to-end.** Every
   `pm.Data("<model>_features", …, dims=("isin", "<dim>_feature"))` container is populated from
   `_resolve_*_feature_aliases()`, which itself reads `public.calculated_features_registry`. The notebook helper
   `attach_features(idata, df, model_name)` reuses the same `MODEL_FEATURE_CONTAINERS` registry, so labelled coordinates
   in `idata.constant_data` match `feature_catalogue.feature_alias` exactly.
2. **Non-centred parameterisations dominate.** Across all hierarchical models (`EarningsBeat`, `PriceTarget`,
   `DividendSafety`, `CreditRisk`, `AccountingAnomaly`), the `non_centered` default delivered the best ESS/divergence
   trade-off. `marginalized` is the right choice for batch scoring where per-ISIN posterior spread is not needed.
3. **Data-space modelling matters.** Log-space Kalman, precomputed Altman zones, and pre-standardised anomaly z-scores
   each removed numerical / fusion failures that previously blocked sampling at scale.
4. **`constant_data` is the integration contract.** All seven models persist their inputs (and feature matrices, post-
   `attach_features`) into `idata.constant_data` keyed by `isin` × `<model>_feature`. This makes downstream cross-model
   joins (tri-/quad-model alignment in the v4 ensemble) trivial and registry-consistent.
5. **Sampler-dependent quirks are handled.** `nutpie` ignores `idata_kwargs` and is stripped automatically; missing
   `constant_data` (e.g. from `nutpie`) is reattached manually in `AccountingAnomalyBayesian.fit`.

### 12.3 `feature_catalogue`-Aligned Improvement Recommendations

The `public.feature_catalogue` table exposes five columns — `category`, `feature_alias`, `source_function`,
`calculation_type`, `data_type` — which are currently used only to resolve `feature_alias` lists. Each recommendation
below leverages an additional column to tighten the model layer:

| # | Recommendation                                                                                                                                                                                                                                                                                                                                                                                                              | Catalogue Column(s) Used        | Affected Model(s)                                         |
|:--|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------|:----------------------------------------------------------|
| 1 | **Type-aware coercion in `_align_*_features`.** Use `data_type` (`numeric`, `pct`, `flag`, `score`, …) to drive per-column dtype + bounded clipping (e.g. `pct ∈ [-1, 1]` clip, `flag` cast to `int8`) instead of the current uniform `astype('float64').fillna(0.0)`. Prevents flag-features (0/1) from being treated as continuous z-score inputs to `AccountingAnomalyBayesian`.                                         | `data_type`                     | All seven models                                          |
| 2 | **Calculation-type-driven priors.** Surface `calculation_type` (`level`, `growth`, `ratio`, `zscore`, `composite`, …) in `MODEL_FEATURE_CONTAINERS` and switch prior scales accordingly: ratios/zscores → `Normal(0, 1)`, growth → `Normal(0, 0.05)`, levels → log-transform. Removes the manual `0.05` `fcf_growth` σ in `DCFPriceTarget` and the hard-coded `5e-2` Normal in PriceTarget's `expected_return`.             | `calculation_type`              | DCF, PriceTarget, EarningsBeat                            |
| 3 | **Source-function provenance in `idata.attrs`.** Stamp `idata.constant_data[<var>].attrs["source_function"]` from `feature_catalogue.source_function` so downstream lineage tooling can map every posterior coordinate back to the SQL/Python function that materialised it.                                                                                                                                                | `source_function`               | All seven models                                          |
| 4 | **Category-conflict guard.** Several models share categories (`Cash Flow`, `Quality & Risk`, `Profitability`) — currently an alias appearing in two `_<MODEL>_CATEGORY_KEYS` tuples lands in two `pm.Data` containers under the same name. Add a `attach_features(..., strict=True)` mode that asserts the materialised `feature_alias` set is disjoint from any other model's previously-attached set on the same `idata`. | `category`                      | DCF ↔ DividendSafety ↔ CreditRisk ↔ AccountingAnomaly     |
| 5 | **Catalogue-driven test coverage.** Add a registry-parametrised `pytest` that, for every `(category, feature_alias)` row in `feature_catalogue`, asserts the alias is reachable through at least one `_resolve_*_feature_aliases()` helper. Catches drift the moment a new alias is registered without being wired into a model.                                                                                            | `category`, `feature_alias`     | Test layer (no runtime change)                            |
| 6 | **Hierarchical priors keyed by `category`.** Replace the per-feature `feature_scale` in `AccountingAnomalyBayesian` with a *per-category* hyperprior (`feature_scale[category]` shared across `feature_alias` in the same `category`). Reduces effective parameters from ≈30 to ≈8 and exposes interpretable category-level importance posteriors.                                                                          | `category`                      | AccountingAnomaly (extends to CreditRisk, DividendSafety) |
| 7 | **Out-of-sample contract.** Document a `pm.set_data({"<model>_features": new_arr})` recipe per model in `data_utils/inference_schema.py`, validated against `feature_catalogue` so `new_arr.shape[1] == len(feature_aliases)` is asserted before swap.                                                                                                                                                                      | `feature_alias` (count + order) | All seven models                                          |

### 12.4 Recommended Next Steps

1. Promote `_MODEL_FEATURE_DIM` (notebook-local) to `probabilistic_ml_model.pymc_models.__init__` so it becomes the
   single canonical mapping consumed by both the notebook and any future batch scorer.
2. Extend `feature_catalogue` with a `model_name` array column (or a join table `feature_catalogue_models`) so registry
   resolution becomes a direct SQL filter rather than the Python-side `_<MODEL>_CATEGORY_KEYS` tuples.
3. Persist the `attach_features`-augmented `idata` back to `outputs/` as `.nc` (NetCDF) per model — keeps the labelled
   feature axis intact for ArviZ 1.0 visualizations and enables hash-gated reuse in `expected_returns_v4`.
