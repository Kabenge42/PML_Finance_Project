# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
