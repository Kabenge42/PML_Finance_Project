# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Kalman v2 (branch `worktree-kalman-v2-design`)

Branch work, not a release. The three preceding v2 commits document themselves in
their commit messages; this entry records the correction to the last of them,
because it reverses a published diagnosis.

### Fixed

- **`ppc_decay` was an over-shrunk regression mean, not a covariance failure.**
  Run `fa532b925732` cleared every convergence and calibration gate and still
  blocked the analytics write on `rho_inf` 0.406 observed against a replicated
  `[0.319, 0.389]`. Reconstructing the posterior against the exported panel
  frame localises it precisely:

  | quantity | value |
  |---|---|
  | slope of `y_now` on `mu_reg` | **1.2301** (1.0 if calibrated) |
  | `Var(mu_reg)` | 0.2916, against 0.4706 for unweighted OLS on the identical design |
  | `2·Cov(mu_reg, resid_now)` | **0.1342 — 15.1% of `Var(y_now)`** |

  `mu_reg` is constant in `t`, so that 15.1% is *permanent* variance; and it is
  exactly **zero** under the generative model, which redraws the residual
  independently of the mean. Replicates therefore cannot carry it and land low on
  `rho_inf`. Propagating `Var(mu)` through `r(t,s) = [V_mu + E·A_ts] / sqrt(...)`
  with the exported `within_name_cov` reproduces the failing run (0.337 predicted
  against the actual `[0.319, 0.389]`) and predicts the fix (0.406 at
  `Var(mu) = 0.40`).

  **The covariance was exonerated and left alone.** Its fitted within-name
  correlation already tracked the empirical residual — 0.9235 / 0.3632 / 0.0220
  against 0.9405 / 0.3696 / 0.0203 at gaps of 7 / 91 / 365 days — so
  `rho_inf = 0.0044` was the right answer for that residual, as
  `KalmanModelConfig.rho_scale_buckets` already recorded. Rejected on
  measurement, not taste: re-enabling `rho_scale_buckets`, widening the
  `rho_inf` prior, and a second OU component (a two-exponential kernel moves the
  residual fit only from RMSE 0.0442 to 0.0386 and still misses the 274-day pair).

  **Cause.** The likelihood weights each name by `1/sigma_i^2` while `sigma_i`
  spans 0.26-0.92, so `beta` is fitted to the low-scale names — and the *signal*
  scales with `sigma_i` too, which the weighting cannot see. Unweighted OLS gives
  slope 1.000 / `Var` 0.4706; WLS at `1/sigma_i` gives 1.160 / 0.3383; the
  posterior gives 1.230 / 0.2916. Widening priors alone therefore closes only
  ~40% of the gap.

  **Fix.** A `signal_exponent` (`lambda`) on the mean, plus freeing the two fixed
  constants that held the rest:

  ```
  mean[i, t] = (sigma_i / geomean(sigma)) ** lambda * mu_reg[i] + alpha_time[t]
  ```

  `sigma_isin` is normalised by its geometric mean first, or `lambda` and
  `log_sigma_total` share a direction and both mix badly. `beta_prior_scale`
  also goes 0.5 -> 1.0.

  A learned crossed-group scale was re-tested at the same time and **rejected
  again** — v1's finding held. `trading_region_effect_scale` came back at ESS 8 /
  R-hat 1.21, the worst-mixing parameter in the model, and took the arm from 0
  divergences to 2. It was also unnecessary: the learned scales landed *below*
  the pinned 0.25 (0.199 / 0.072 / 0.032 / 0.055), so freeing them shrinks the
  group effects rather than releasing them, and the crossed effects carry only
  `Var` 0.010 of the mean's 0.292. `signal_exponent` alone gets the calibration
  slope to 1.075 with `ppc_decay` passing. Kept as
  `learn_group_effect_scale`, default `False`.

  Calibrated by `scripts/profile_signal_exponent.py` (new) — an alternating GLS
  that refits `beta`, `sigma` and `A` at each `lambda` on the production panel
  and scores it with the *same* `fit_trail_correlation_kernel` the gate uses.
  Four independent criteria land in one band:

  | criterion | optimal `lambda` |
  |---|---|
  | profile log-likelihood | 0.30 (+173 over `lambda = 0`) |
  | calibration slope = 1 | 0.45 (0.9993) |
  | `2·Cov(mu, resid)` = 0 | 0.45 (-0.0007) |
  | predicted `rho_inf` = 0.4058 | 0.50 (0.4032) |

  Hence `signal_exponent_prior = Beta(4.5, 5.5)` — mean 0.45, sd 0.15, 90%
  interval ~`[0.21, 0.70]`. `enable_signal_scaling = False` restores the additive
  mean as the comparison arm.

  `state_now_mean` now reads `mu_scaled + gain_full`, not `mu_reg + gain_full`:
  the decision latent is the level on the response scale, and reading the
  unscaled predictor would have dropped the scaling from the screen, the risk
  book and the analytics export while every gate still passed. The `mean_spread`
  and `ppc_decay_residual` consumers follow the same variable.

- **`mean_spread` could not have caught it, so `mean_calibration` was added.**
  `mean_spread` is one-sided (`<= 1.0`) and read a healthy-looking 0.33 while the
  mean was shrunk by 19%. The new gate regresses the response on the fitted mean,
  requires slope in `[0.9, 1.1]`, and reports `2·Cov(mu, resid)/Var(y)` beside
  it. `ppc_decay`'s detail now carries the mean/covariance split
  (`f = Var(mu)/(Var(mu)+Var(resid))`, a floor on the replicated `rho_inf`), so
  the next occurrence is legible from the gate report rather than from a
  posterior reconstruction.

- **`coverage_gradient` (1.57x against a 2x target): the per-lookback analyst
  counts were loaded and then discarded.** `prepare_panel` already built a
  per-`(name, time)` `coverage_profile` from the MV's `n_analysts_{1w..1y}`, then
  fed only the scalar `precision_weight` and an off-by-default bucketing — so a
  4-analyst consensus from a year ago was charged the same measurement precision
  as today's 30-analyst one, the exact defect the v2 module docstring claims to
  have fixed. `coverage_scale_per_cell` now applies the same `sigma_n_exponent`
  along time. It costs nothing: a per-cell scale is `diag(d) L`, so the group's
  shared Cholesky is reused and only its rows are rescaled, and no parameter is
  added. The smoother follows —
  `E[latent|y] = sigma_now · c' A^-1 D^-1 (y - mean)` — and reduces identically
  to the previous scale-free expression when the cell scale is constant in `t`.

  **It did not move the gate**, and that is reported rather than papered over:
  `coverage_gradient` is still 1.57x, and `sigma_n_exponent` refit to 0.267
  against 0.263. The reason is that the gate measures `er_sd`, whose gradient
  runs through the *snapshot* scale `sigma_isin` — and `coverage_profile` is
  anchored at 1.0 on the snapshot by construction, so the per-cell term is
  identically zero there. The change is a correctness improvement to the
  likelihood (older, thinner-coverage observations are now weighted as such);
  closing `coverage_gradient` needs the snapshot-scale coverage law itself, which
  the data currently puts at `n^-0.13` against the `n^-0.5` a 2x spread would
  require. The gate stays a WARN.

- **Three `arviz_stats` "invalid value encountered in scalar divide" warnings and
  one numpy All-NaN slice.** `alpha_time[t3]` and `sigma_time[t3]` are the pinned
  anchors of `pt.concatenate([free, zeros(1)])`; R-hat and ESS divide by a
  within-chain variance of exactly zero. The rows stay in the exported table
  (they are real parameters, just pinned) and the convergence gates now read the
  free ones only — no value changes today, but a genuinely stuck parameter can no
  longer read as converged. The All-NaN slice came from reducing over
  `(isin, time)` cells with no replicate at all; those are excluded and counted
  instead, with the count in the gate detail. The pytensor "Loop fusion failed"
  notice is benign (the `log_sigma` graph has six additive terms plus a sector
  gather) and is now recorded as expected.

- **`prob_pos_degenerate` (87.5% pinned at 1.0): the guidance and the artifact
  disagreed.** The gate already named `p_upside_pos_cond` as the column that can
  actually rank, but `p_upside_pos_cond` was in neither `_RANKING_COLS` nor
  `_RANKING_RANGE_COLS` — so a clip-pinned row kept a ranking probability while
  its other ranking metrics were NULLed. It is now a first-class ranking column
  in both, and `prob_pos` carries a `COMMENT ON COLUMN` saying it is reported,
  not ranked, and why.

- **The self-test could not run.** `_selftest` asked `az.summary` for
  `rho_scale_slope` unconditionally, which stopped existing the moment
  `rho_scale_buckets` was defaulted to 1 — so the acceptance test for this
  module's central claim raised `KeyError` instead of executing. Optional
  variables are now resolved against the posterior actually produced.
  `_simulate_panel` additionally accepts `vol_delta_true` /
  `signal_exponent_true` to generate a heteroscedastic panel with a scaled mean,
  because `lambda` is unidentified without one (`signal_scale = exp(lambda·0) = 1`
  for every `lambda`). Verified: at truth 0.450 on 400 names the posterior returns
  0.546 `[0.337, 0.757]` with the level/state split still recovered.

### Changed

- **`trail_days_*` retired from `mv_pymc_kalman_pt_v2` to `pml.vw_pymc_trail_days`.**
  The MV emitted six SQL literals (0/7/30/91/182/365) identical on every one of
  ~6,500 rows — zero information, stored 6,500 times — while the model built the
  same grid from `DEFAULT_LOOKBACK_DAYS` in Python and never read the columns.
  Two sources of truth for the OU kernel's x-axis, and the one the model used was
  the one the database could not see.

  The new view is a standalone `VALUES` lookup, deliberately not a `SELECT` over
  the MV: it has to survive the `DROP MATERIALIZED VIEW` that changing that MV
  requires (it is `CREATE ... IF NOT EXISTS`), and the offsets are metadata about
  the grid rather than data about any name. Each row maps
  `lookback_key -> response_column -> trail_days`, and because a view cannot
  declare a foreign key, `pml.assert_pymc_trail_days_map()` enforces the
  equivalent in both directions — every mapped `response_column` exists on the MV
  and every `feat_log_uplift_*` the MV emits is mapped. It is called from
  `pml.assert_pymc_catalogue_coverage()`, so it runs wherever every other
  MV↔catalogue contract already does. Materialized-view columns are resolved
  through `pg_attribute`; `information_schema.columns` does not list them.

  Python reads it through `load_trail_days_map()` (`@lru_cache`, the
  `_resolve_feature_aliases` idiom), which `KalmanModelConfig.lookback_days` now
  takes as its `default_factory`. `DEFAULT_LOOKBACK_DAYS` remains as the offline
  fallback so `--selftest` and the unit tests run without a database, with
  `PML_STRICT_TRAIL_DAYS=1` turning the fallback into a hard failure for CI
  (mirroring `PML_STRICT_STREAK_MERGE`).

  A *tightening*, deliberately: the view lists only the six lookbacks the MV
  actually trails, where the literal carried thirteen including `3y` / `5y` that
  have no `feat_log_uplift` column. `KalmanModelConfig.__post_init__` now rejects
  such a lookback up front instead of failing in `prepare_panel` several stages
  later.

  Three coordinated edits, the 0.9.9.15 §7j shape: the MV definition, an
  idempotent `array_remove` in `pml_df_metadata_populate.sql` §7l, and the view
  plus assertion in `pml_feature_catalogue.sql`. The de-registration is not
  optional — `vw_pymc_feature_catalogue` is
  `metadata CROSS JOIN LATERAL UNNEST(model_targets)`, so a still-tagged column
  the MV no longer emits raises `PHANTOM_CATALOGUE_ALIAS`. Applied and verified:
  198 columns (was 204), both indexes rebuilt, `kalman_pt_v2` coverage violations
  0, and the one remaining DB-wide violation
  (`price_target.feat_pt_achievement_1y`) is the pre-existing one.

- **Reference notebooks: nothing adopted, and the reason is recorded.**
  `Forecasting_with_structural_timeseries.ipynb` is the pre-`pymc_extras` AR
  example — no `statespace` / `LevelTrendComponent` API, one scalar `sigma`
  shared between innovation and observation, no variance split, no decay
  statistic. `MvGaussianRandomWalk_demo.ipynb` puts a single `LKJCholeskyCov`
  over three *cross-sectional* series with sampled latents, strictly less
  advanced than the shared-Cholesky-per-missingness-group `MvStudentT` here.
  `multinomial_ppcs.ipynb` is PyMC3-era with entirely visual calibration and no
  pass/fail rule. The one transferable idiom is its cell-4 pattern — draw from
  `.dist()` and push through the link with no `pm.sample` — which is the shape a
  prior-predictive check on the decay statistic should take.

- **`drift_contrast_leakage` fired on sampling noise.** The gate blocked run
  `6a0f957972b1` on `feat_eps_signal_beat`: correlation **-0.101** with the
  response level against **-0.115** with the (now, 3m) contrast. The rule was
  `|corr_contrast| > max(|corr_level|, CONTRAST_CORR_FLOOR)` — a bare
  inequality between two correlations, so a feature whose two correlations are
  equal up to noise fails it about half the time. On 6,499 names a correlation
  carries `se ~ 1/sqrt(n) = 0.013`, and the excess here is **0.014**: one
  standard error. `CONTRAST_CORR_FLOOR` guards the absolute size of the two
  correlations and does nothing about their ratio, which is what the rule
  actually tests.

  The verdict was also wrong on the merits. `feat_eps_signal_beat` is the
  consolidated EPS beat-rate block; it contains no price and no price-target
  term, so it cannot be a leg of `Δ log PT − Δ log P` by construction, and
  excluding it would have cost the drift matrix a fundamental signal to satisfy
  a coin flip.

  Fixed by requiring the contrast correlation to *dominate* the level
  correlation by `CONTRAST_DOMINANCE_MARGIN = 1.5`, and by carrying the
  measured `dominance` as a column on the screen frame so the artifact shows
  the margin rather than only the verdict. Calibrated by re-admitting the six
  known identity legs to the design matrix un-rotated and measuring both
  correlations on the same universe the gate runs on:

  | feature | dominance | identity |
  |---|---|---|
  | `feat_pt_drift` | 7.17 | yes |
  | `feat_pt_noise_drift` | 5.09 | yes |
  | `feat_pt_accuracy_1y` | 3.26 | yes |
  | `feat_price_drift` | 1.97 | yes |
  | `feat_one_day_return` | 1.88 | yes |
  | `feat_price_chg_pct_3m` | 1.65 | yes |
  | — margin — | 1.50 | |
  | `feat_eps_signal_beat` | 1.14 | no |
  | `feat_median_piotroski_f_score` | 0.62 | no |
  | `feat_coverage_drift` | 0.46 | no |

  The two populations are separated by a factor of 1.45 with nothing between
  them, so the margin is not fitted to the boundary case — it can sit anywhere
  in `(1.15, 1.65)` without changing a verdict on this universe. Verified: all
  six known legs are still flagged, no false positives, and the T=4 panel audit
  now clears all three gates (`--dry-run`, exit 0, max dominance 1.14).

  Not changed: `DRIFT_EXCLUSIONS` keeps all six legs. The screen measures; the
  exclusions are the SSOT, so a run's design matrix stays reproducible from the
  source rather than from whichever universe it loaded.

- **The three failing posterior-predictive gates were one runaway regression
  mean, not the Student-t.** Run `4f713551bb7a` failed `ppc_t_spread` (IQR 1.017
  observed vs 1.134-1.168 replicated), `ppc_coverage` (0.958 / 0.960 at the two
  oldest steps against a 0.92 target) and `ppc_decay` (rho_inf 0.429 observed vs
  0.678-0.740 replicated), and `fb8f86d` attributed all three to the marginal
  Student-t being a per-NAME rather than per-cell scale mixture, proposing `n x T`
  conjugate scale variables as the fix.

  Reconstructing `mu_reg` from the exported posterior against the exported panel
  frame (validated: the reconstructed `expected_upside` matches the exported
  column at r = 0.9999) gives **Var(mu_reg) = 2.503 against Var(y_now) = 0.872** —
  a mean with 2.9x the variance of the response it predicts, with
  `beta[pt_hist_pc3] = -1.414` against a next-largest coefficient of 0.077.
  Propagating that forward reproduces the gates: predicted replicate sd ratio
  **1.76** against the 1.75 measured, and replicate correlations of
  0.985 / 0.861 / 0.793 at 7 / 91 / 365 days fitting rho_inf ~ 0.78 against the
  0.68-0.74 measured. A multivariate t's correlation matrix is its scale
  matrix's, so the likelihood family contributes nothing to any of it. The
  per-cell augmentation would have restored ~26k latents for no gate movement,
  and is now recorded as rejected in the builder docstring.

  The mean is not a sampler artifact: at the posterior betas the log-likelihood
  is -11,212 against -17,951 at pooled OLS. Two structural defects let it be the
  optimum, and both are fixed.

- **The design matrix contained the response's own increments.**
  `feat_log_uplift = log PT - log P`, so the trail's differences are
  `Δ log PT - Δ log P`, and six drift features were the two legs of that
  identity: `feat_one_day_return` (corr 0.03 with the level, -0.46 with the
  now-1w contrast), `feat_price_chg_pct_3m` (0.40 / -0.67), `feat_price_drift`
  (-0.20 / -0.33), `feat_pt_drift` (-0.00 / +0.35), `feat_pt_accuracy_1y`
  (0.02 / -0.23) and `feat_pt_noise_drift` (0.02 / +0.24).

  They were harmless in v1 (|beta| <= 0.09) because a factorised likelihood never
  weights a contrast. The v2 correlated likelihood weights the (1w, now) contrast
  — sd 0.248 against a level sd of 0.934 — by `1/(1-rho) ~ 12x`, while `mu_reg`
  is constant in `t` and so cannot express contrast content at all. **The v2
  likelihood change is what made v1's inherited feature set unusable.** Removing
  them takes Var(mu_reg) from 1.006 to 0.321 under weighted GLS and max |beta|
  from 0.90 to 0.45; the design matrix goes from 14 columns to 8.

  Rejected instead of dropping them: a per-lookback slope block, which legalises
  the leverage rather than removing it, spends `p x (T-1)` parameters on a panel
  worth `T_eff = 1.42` observations per name, and fits an accounting identity.

- **`sigma_time` scaled only the observation leg, and that covariance family
  cannot represent this panel.** With `A = w_L*J + w_S*K + w_O*diag(tau^2)`,
  raising a far column's variance divides every one of its correlations by
  `sqrt(A_ss * A_tt)`. Solved against the measured residual structure — variance
  ratios [2.13, 1.49, 1.11, 1.0], correlations 0.921 / 0.315 / -0.065 at
  7 / 91 / 365 days — the family needs `w_S > 1`, **infeasible by 74%**. The run
  took the only escape available and drove `tau` to [12.26, 9.24, 2.54, 1.0], ten
  prior sd out, buying the variance ratio by destroying the correlation.

  `tau` now scales the whole matrix, `A = diag(tau) C diag(tau)`, so correlations
  are `tau`-free and variance ratios are `tau^2`. The same measurement then
  solves at `tau ~ [1.46, 1.22, 1.05, 1.0]`, `ell ~ 82d`, `w_L ~ 0.02`,
  `w_O ~ 0` — feasible to within 1.4%, with `tau` inside its prior. The prior
  widens 0.25 -> 0.40 on the log scale to match. `time_scale_applies_to`
  keeps the old form available as a comparison arm.

  With sane betas and this covariance the implied **raw** panel correlations are
  0.965 / 0.674 / 0.449 against observed 0.966 / 0.678 / 0.427, which is what
  `ppc_decay` reads.

### Added

- **`kalman_pt_v2` is now an ordinary model target, folded into the four SQL
  SSOT scripts.** `sql_scripts/pml/mv_pymc_kalman_pt_v2.sql` and its companion
  `..._metadata.sql` were working DDL sitting in a directory CLAUDE.md defines as
  a pg_dump extract — 48 of its 64 files are `-- missing source code` stubs,
  including every other `mv_pymc_*.sql`. Both files said so in their own headers
  and asked to be folded. Until now the MV definition, the `model_targets`
  allow-list and the coverage-check `mv_map` each existed in two places, and the
  four-step ordering the metadata file documents (widen CHECK → register columns
  → teach the coverage check → verify) was enforced only by remembering to run
  two files together.

  What moved where:

  | Landed in | Content |
  |---|---|
  | `pml_feature_catalogue.sql` | the `mv_pymc_kalman_pt_v2` DDL (placed after its parent, since it `SELECT`s from it), `pml.kalman_pt_v2_asof()`, `pml.refresh_kalman_pt_v2()`, the `mv_map` row, the `mvs` array entry |
  | `pml_df_metadata.sql` | `'kalman_pt_v2'` in both CHECK constraints; a v2 role-notes block |
  | `pml_df_metadata_populate.sql` | STEP 0 idempotent CHECK widening; new §7k (inheritance, new columns, role overrides, retirement of the superseded rows) |
  | `sql_scripts/pml/{pml_df_metadata,pml_df_feature_alias}.sql` | the widened CHECK arrays, so the extracts stay faithful |

  Three things the fold had to get right that the standalone files did not:

  - **`pml_df_metadata.sql` cannot widen a live vocabulary.** It opens with
    `DROP TABLE ... CASCADE`, so it only ever runs against a disposable
    database. The idempotent `ALTER TABLE ... DROP CONSTRAINT / ADD CONSTRAINT`
    pair now at the top of `pml_df_metadata_populate.sql` is the only path to a
    live one. Without it every `kalman_pt_v2` INSERT fails the CHECK, and it
    surfaces as a constraint violation on an unrelated-looking statement
    hundreds of lines later.
  - **Refresh order is now structural, not documentary.** `mv_pymc_kalman_pt_v2`
    is `SELECT b.*` over `mv_pymc_kalman_pt`, so it sits immediately after its
    parent in the `refresh_pymc_materialized_views` array (`FOREACH` walks it in
    sequence), and `refresh_kalman_pt_v2` defaults `refresh_parent => TRUE`.
  - **`built_at` is `derived_input`, not `constant_data`.** `constant_data`
    places it in `vw_pymc_feature_aliases.constant_data_aliases`, and
    `coerce_by_data_type()` casts every alias handed to it to `float64` — a
    `timestamptz` there is a failure waiting on the first consumer.
    `derived_input` keeps the column documented in the catalogue and out of all
    four alias arrays.

  The `data_type` values were also corrected from PostgreSQL type names
  (`'double precision'`, `'integer'`) to the semantic vocabulary `_DTYPE_RULES`
  in `_feature_alignment.py` actually keys on (`ratio`, `count`, `level`,
  `pct`). The old values matched no rule and fell through to the untyped
  default — inert, but they read as if they were configuring something.

  Verified against the live database in a rolled-back transaction:
  `vw_pymc_catalogue_coverage_check` returns **zero** `kalman_pt_v2` rows, and
  the DB-wide count is unchanged at the known 1 (`price_target`,
  `MISSING_FROM_CATALOGUE`). `kalman_pt_v2` alias arrays resolve to 58
  predictors / 86 observed / 21 constant_data / 19 coords against `kalman_pt`'s
  55 / 80 / 9 / 19, and `built_at` is absent from the constant_data array.

### Changed

- **The consolidated EPS block is split by quantity — this changes fitted
  values.** `feat_eps_signal` averaged five legs sitting on three incompatible
  scales: `feat_last_{q,y}_surprise` are `eps_neg0f{q,y}surprise_pct`, i.e.
  PERCENT (`5.2` means +5.2%); `feat_eps_beat_rate{,_annual}` are `n_beats /
  n_total`, shares in [0,1]; `feat_net_eps_drift` is a `pml.target_drift` ratio,
  a raw decimal. The percent legs are ~100× the others, so the "consolidated
  signal" was the surprise legs wearing a different name — and it violated the
  0.9.9.7 raw-decimal convention the MV header itself claims.

  `mv_pymc_kalman_pt_v2` now emits one column per quantity:

  | column | legs | scale |
  |---|---|---|
  | `feat_eps_signal_surprise` | the two `_pct` legs, each `/100` | signed raw decimal |
  | `feat_eps_signal_beat` | the two beat rates | share in [0, 1] |
  | `feat_eps_signal_coverage` | count of all five non-NULL `/5` | share in [0, 1] |

  The trend leg needs no new column: `feat_net_eps_drift` is already a raw
  decimal ratio, so it is re-admitted to the drift matrix in
  `pymc_kalman_filter_pt_v2.py` rather than folded into an average. Measured on
  the rebuilt MV (n = 6,529): `feat_eps_signal_surprise` mean 0.0012 / sd 0.0264
  — hundredths, as a decimal surprise should be — `feat_eps_signal_beat` mean
  0.5627 within [0, 1], `feat_eps_signal_coverage` mean 0.7275.

  Net effect on the v2 drift design: `feat_eps_signal` and `feat_eps_coverage`
  out, the three `feat_eps_signal_*` columns and `feat_net_eps_drift` in — **+2
  columns**. **Any recorded `feat_eps_signal` beta is stale; the v2 fit must be
  re-run.**

- **`pml_df_metadata.sql` de-duplicated.** It defined both tables twice, each as
  `CREATE TABLE IF NOT EXISTS` with the constraints repeated verbatim so
  "whichever runs first wins". The `DROP TABLE` at the top guarantees the first
  always wins, so the second copy was unreachable — and it meant a vocabulary
  change had to be made at four constraint sites instead of two. One definition
  per table now, with an explicit note at the foot listing the three files a new
  model target has to touch.

### Notes

- **Editing `mv_pymc_kalman_pt_v2` requires an explicit `DROP MATERIALIZED VIEW`
  first.** It uses `CREATE MATERIALIZED VIEW IF NOT EXISTS` to match its seven
  siblings in `pml_feature_catalogue.sql`, so re-running the script does not
  pick up a definition change. This bit during verification: the deployed MV
  still had the old EPS columns and the `CREATE` silently no-opped. True of
  every MV in that file; now stated once, on the newest one.
- `pml.kalman_pt_v2_asof(DATE)` covers six of the parent's seven `days_*`
  horizons. `days_to_next_fiscal_quarter` is omitted deliberately: the parent
  computes it as `(next_fiscal_quarter - CURRENT_DATE)` where
  `next_fiscal_quarter` is a 1–4 quarter ordinal, not a date. Its other sign
  conventions — including `days_since_fy_end = fy_end_date - p_asof`, negative
  for a past fiscal-year end — deliberately reproduce the parent's, so the
  function and the column it shadows cannot disagree.

### Added

- **`drift_contrast_leakage`** (blocking, §4b) — no drift feature may correlate
  more strongly with a trail contrast than with the response level. Measured by
  `screen_contrast_identities` on the design matrix **before** the PT-history
  rotation, because an orthonormal rotation mixes an identity column into its
  neighbours: on the rotated basis the flag lands on `pt_hist_pc1` and clears
  `feat_pt_drift`, which is the wrong answer to act on. Costs milliseconds and
  runs before the fit, so `--dry-run` sees it.
- **`mean_spread`** (blocking, §9) — `Var(mu_reg) / Var(y_snapshot) <= 1`. An
  additive mean with more variance than its response implies a negative residual
  variance, so the threshold is arithmetic rather than a tuning choice. This one
  line would have caught the whole failure before the posterior predictive ran.
- **`ppc_decay_residual`** (reported, not gated, §8) — `ppc_decay` with the
  posterior-mean mean removed from both sides. A mean that is constant in time
  contributes the same constant at every gap and so reads as a permanent level;
  the pair separates a mean failure from a covariance one in one read. Both are
  computed through one shared helper so they cannot drift apart in method.
- The self-test now simulates and recovers the per-lookback time scale. Measured
  at 400 draws / 400 tune, n=400: every truth inside its 94% interval —
  `sigma_time` [1.412, 1.245, 1.067] against [1.45, 1.20, 1.05] — at 0
  divergences, min bulk ESS 492, max R-hat 1.0029.

### Fixed (found by the verification run)

- **`ppc_coverage` graded a 94 % interval against a 92 % target.** The statistic
  has always built its interval from `np.nanpercentile(rep, [3, 97])` — 94 %
  nominal — while `gate_coverage_target` was a hard-coded 0.92 carried over from
  a v1 statistic that used a different interval, so the gate asked a *correctly
  calibrated* model to under-cover its own interval by two points. The target is
  now derived from `gate_coverage_percentiles`, which makes the two
  unrepresentable as different values. Measured on the full panel after the
  fixes above: per-step coverage [0.9407, 0.9484, 0.9384, 0.9364] against a 0.94
  nominal — a worst deviation of 0.008, and flat across steps where the
  2026-08-18 run was [0.958, 0.960, 0.924, 0.928].

- **`apply_out_of_support` tested one distribution and protected metrics built
  from two.** `er_*` comes from the forward-return Monte Carlo; `cvar05` and
  `exp_vol` — hence `reward_to_cvar` and `expected_sharpe_ratio` — come from the
  Kalman upside posterior. One name reached the export with
  `expected_return_kalman` 5.0, `cvar_5pct_kalman` 5.0 and
  `expected_vol_kalman` 0.0 — a degenerate distribution sitting on the +500 %
  cap, `tail_risk` on its floor and a STARR of exactly **500** — while its
  `er_p05` of 0.64 cleared the test. The Kalman distribution is now tested on its
  own tails: `cvar_5pct_kalman` plays the role `er_p05` plays for the Monte
  Carlo. Re-applied to both exports, it suppresses that one row and no others.

### Notes

- `w_obs` heading to ~0 is a finding, not a defect: at a 7-day gap a consensus
  price target is sticky, so the panel identifies no measurement noise and the
  smoother leaves `state_now ~ y_now`. The two WARN gates — `coverage_gradient`
  and `prob_pos_degenerate` — are consequences of that and are expected to
  persist. Moving them needs a decision about what `state_now` *is*, not a
  tuning change.
- **`ppc_decay` is the one gate still failing, and it is now fully
  characterised.** Raw: observed rho_inf 0.429 against a replicated
  [0.320, 0.393]. Residual: observed 0.070 against [0.000, 0.052]. The model fits
  one global `w_level` at 0.0066 where the unweighted residual panel carries
  ~0.070.

  **Tried and rejected — the `('1y','6m','3m','1w')` grid.** The hypothesis was
  that only one column (1y) informs the permanent component, so a second
  long-gap column would identify it. Measured on the full panel at 2000/4000:
  T_eff 1.50 vs 1.42, min bulk ESS 1442 vs 1305, gradient 3.52 vs 2.48 ms — and
  `w_level` went **0.0067 -> 0.0058**, i.e. slightly *down*. `ppc_decay` reads
  0.432 vs [0.328, 0.399], the same gap. Identification was never the problem;
  the default T=4 grid stands.

  What it is instead: **the variance split is global while the observation scale
  is per-name, and the panel says the split depends on the scale.** Residual
  kernel by `sigma_isin` quintile —

  | quintile | mean sigma | residual rho_inf | corr(1y, now) |
  |---|---|---|---|
  | Q1 | 0.258 | 0.0000 | -0.037 |
  | Q2 | 0.353 | 0.0000 | -0.058 |
  | Q3 | 0.440 | 0.0000 | -0.088 |
  | Q4 | 0.568 | 0.0000 | -0.106 |
  | Q5 | 0.910 | **0.1163** | +0.060 |

  The permanent level exists only in the noisiest fifth of the universe; the
  other four fifths are purely mean-reverting. `rho_inf` is one number for every
  name, fitted in a metric that weights by `1/sigma_i^2` — under which Q5 holds
  ~1.5 % of the weight. The same residual kernel reads rho_inf **0.0000**
  weighted by `1/sigma_i^2`, **0.0130** by `1/sigma_i` and **0.0703** unweighted,
  against a fitted 0.0067: the model is estimating the panel correctly *in its
  own metric*, and the gate measures it in the unweighted one, which the same Q5
  names dominate because they carry the largest residuals.

  **Built and rejected on measurement (2026-08-19).** `rho_scale_buckets` /
  `rho_scale_slope` implement exactly that expansion — one logit-scale tilt of
  `rho_inf` across quantile buckets of a per-name scale index, evaluated per
  bucket so the Cholesky count tracks groups rather than names. The mechanism
  works: on simulated data where the structured variance is held fixed and only
  its split varies, the tilt is recovered (truth 0.800, posterior 0.553
  `[0.237, 0.925]`). On the production panel it is **not identified** —
  `rho_scale_slope` 0.578 +/- 0.828, an 89 % interval of `[-0.836, 1.761]`
  spanning zero at ESS 1986, so the width is posterior uncertainty rather than
  anything a longer run would sharpen. Buckets came out monotone as predicted,
  `[0.0041, 0.0043, 0.0052, 0.0076, 0.0196]`, and `ppc_decay` did not move:
  0.429 observed against [0.334, 0.399] replicated versus [0.320, 0.393] with
  one global split. Cost is +23 % per gradient and 6 -> 21 covariance groups.
  **Defaulted to `rho_scale_buckets = 1`**; the machinery, the scale index and
  the self-test all remain, and `replace(cfg, rho_scale_buckets=5)` re-enables
  it.

  **And the motivating measurement was half artifact.** A per-name scale
  multiplies a name's whole trail together, which in a correlation estimate is a
  rank-1 common component indistinguishable from a permanent level. Standardising
  the residual by the model's own `sigma_i * tau_t` before re-measuring:

  | quintile | mean sigma | raw rho_inf | standardised rho_inf |
  |---|---|---|---|
  | Q1-Q4 | 0.258-0.568 | 0.0000 | 0.0000 |
  | Q5 | 0.910 | 0.1170 | **0.0723** |
  | pooled | 0.506 | 0.0706 | **0.0000** |

  The pooled row is the result: there is no pooled permanent level at all, and
  `rho_inf ~ 0.006` was right rather than under-powered. Q5 keeps a genuine level
  after standardisation, but at 0.072 rather than 0.117, carried by a fifth of
  the universe — real, and about one gate-width of the decay statistic.

  What that leaves: `ppc_decay` compares an estimator that cannot separate
  per-name scale heterogeneity from a permanent level, applied to a model whose
  scale heterogeneity is fitted rather than exact. Closing it means either
  measuring the decay on standardised residuals — which the model *does*
  reproduce, pooled — or accepting that the raw statistic carries a component the
  model is not trying to match. That is a decision about the gate, and it should
  be made deliberately rather than by tuning the model into it.
- After the screen the largest surviving predictor is `feat_pt_achievement_1y`
  at corr -0.542 with the response — mechanically anti-correlated through the
  shared `last_price` leg, the same identity (v1 measured -0.545) for which
  `DRIFT_EXCLUDED_PREFIXES` already bans the whole `feat_total_return_*` /
  `feat_tr_cagr_*` family. Not blocking under the new gates, but inherited rather
  than chosen.

## [0.9.9.17] - 2026-08-17

### Fixed

- **The Kalman notebook was 233 MB, and 232.1 MB of it was nine Plotly outputs.**
  Profiling `pymc_kalman_filter_pt_v4.ipynb` by MIME type put
  `application/vnd.plotly.v1+json` at 232.1 MB against 0.5 MB for everything else
  — and the notebook was not even fully run (only §2 EDA and §6 prior had output).

  **207.7 MB came from one figure**: the §6 prior-predictive panel, whose three
  `go.Histogram` traces each shipped `prior_draws × n_isin` ≈ 6.5 M float64 to
  the browser to be binned client-side. The largest single line in the file was
  **69.2 MB of base64**. The same figure exports as a **122 KB PNG** — nothing in
  it ever needed the raw sample.

  Traces are now pre-binned through `_binned_density_trace` (an SSOT lift of the
  `np.histogram` → `go.Scatter(shape='hvh')` pattern that already sat ten lines
  below the first offender, for the empirical overlay). Measured on a faithful
  6,487,000-value array: **87.6 MB → 9.0 KB per trace, a 9,700× reduction**, with
  hover preserved and the exported PNG unchanged. A `density=False` mode keeps
  the one count-axis histogram (`plot_screen_overview`) honest.

- **`matplotlib.use("TkAgg")` at import made the matplotlib backend unusable in a
  notebook.** It overrides the inline backend, so an mpl figure opens in an
  off-screen Tk window and the cell renders nothing. Now conditional on
  `_in_ipython_kernel()`. This was invisible while every panel was Plotly and is
  a hard blocker for the change below.

### Changed

- **The dense arviz-plots diagnostics render through matplotlib.** Backend choice
  is now one decision in one place (`_azp_backend(heavy=...)`, overridable with
  `PML_AZP_HEAVY_BACKEND`) rather than fourteen `backend='plotly'` literals.
  Heavy = the panels that fan one facet per vector element and draw every chain's
  full draw sequence: trace, rank-dist, prior-posterior, ESS-evolution, the §9.4
  scale forest and the PPC t-stat. Measured on a 4 × 2000 × 17 `beta` trace grid:
  **2.25 MB of Plotly JSON → 0.38 MB of PNG**, and the raster is flat in draw
  count where the JSON is linear. Bounded panels (PIT ECDF, energy, the capped
  forests, the ridges that carry `_add_ref_line` geometry) stay interactive.

  Three supporting fixes, since nothing but `_export_figure` had ever seen a
  matplotlib figure: `_safe_show` now routes them to `IPython.display` and closes
  them (a bare `.show()` neither displays nor blocks correctly); `_apply_dark_template`
  no longer silently no-ops on them; and `_next_stem` reads the mpl suptitle, so
  filenames keep their descriptive slugs instead of degrading to bare counters.

- **Full-universe scatters are decimated uniformly, and say so.** `_decimate_frame`
  caps the §2.4e driver grid at 1,200 markers per facet (it was 17 facets ×
  ~6.5 k names, each shipping its ticker and company name as `hover_data` — 21.5 MB)
  and the §14 screen scatter at 2,500. The per-driver Spearman ρ and the OLS
  trendline are still computed on the **full** frame, so the quantitative content
  is unchanged, and the sampled count is stated in the title.

  The sampling is **uniform, not top-N**. `plot_risk_return_scatter` previously
  cut with `nlargest(max_points, 'expected_upside')` at a cap of 7000 — above the
  universe size, so it never bound. Lowering that cap without changing the rule
  would have deleted the entire lower tail of the y-axis and left a cloud that
  reads as a uniformly positive screen.

- **§8 posterior-predictive: thinned, whitelisted, and no longer welded onto the
  production `idata`.** `pm.sample_posterior_predictive` replicates once per
  posterior sample and takes no draw count, so it replayed the whole
  `chains × draws` grid. It now runs against a **thinned copy**
  (`thin_posterior`, `KalmanRunConfig.ppc_draws = 1000`) with
  `var_names=['target_pct_obs']` — the pattern `validate_kalman_state.py:159` and
  `run_prior_predictive` already used — and `extend_inferencedata=True` targets
  that copy, so the multi-GB predictive group no longer rides into
  `07_posterior_idata.nc` or gets swept again by §9.

  **The binding constraint is memory, not arithmetic**, and measuring said so:
  forward sampling costs only ~1.4 s per 1,000 draws even under the pure-Python
  VM, but the posterior alone is 8.16 GB on disk (it carries `state_path` plus
  sixteen `dims="isin"` deterministics) against ~17 GB of free RAM, so welding a
  predictive group on top pushed the working set into swap.

  Two candidates were tried and **rejected**, recorded so they are not retried:
  **PyTensor's numba mode** for the predictive call is a *pessimisation* at these
  draw counts (~1.5 s of graph compilation against ~1.4 s of sampling per 1,000
  draws; 2.80 s vs 1.35 s on a faithful n=6487, T=4, 17-variable reproduction) —
  nutpie already bypasses the mode for NUTS, so the predictive path is the only
  place the VM runs, and it is not the bottleneck. And **`var_names` alone** is
  worth only ~7 % (1.44 s → 1.35 s); it is kept because it is correct, not because
  it is the win.

  Also: one `quantile` call for both interval edges instead of two, with
  `skipna=False` — xarray's default dispatches to `np.nanquantile`, which copies
  the array and builds a NaN mask that the `nan_to_num`'d response tensor does not
  need. Measured 11.60 s → 3.78 s on a 1.66 GB group.

- **The §8 ECDF overlay was 1.6 M points.** `pick` thinned the sample axis to 60
  draws but never the observation axis, so each of 61 curves carried all 25,948
  cells. Curves are now evaluated on a fixed 512-point probability grid
  (`_ecdf_xy`) — **42.2 MB → 0.86 MB of JSON**, max deviation 0.0000, and the
  kaleido rasterisation halves.

- **§9 stopped computing tail-ESS for numbers nothing reads.** `ess_tail_ds` was
  swept over the whole posterior — ~130k independent per-element FFT
  autocorrelations — and then read only for the handful of `sigma_<coord>`
  scalars in the group-effect report. The sweep is restricted to exactly those.
  **Every printed value is identical.** `_degenerate_posterior_vars` also gained
  an all-finite fast path, avoiding a full float64 `np.where` copy per variable
  (~1.7 GB for `state_path` alone) in the common case.

- **`draws` / `tune` split asymmetrically to 2000 / 4000.** The 0.9.9.16
  measurement is explicit that the two knobs are not interchangeable — tune bought
  the R-hat, draws bought the ESS — so the cut goes where the headroom is. Bulk
  ESS came in at **884 against a `MIN_ESS_GATE` of 400** (~2× margin, and ESS
  scales about linearly in draws) while R-hat had none to give.

  This also resolves an **uncommitted `tune: int = 1500`** that was sitting in the
  working tree, contradicting HEAD, the comment directly above it, CLAUDE.md and
  the 0.9.9.16 entry. It attacked precisely the half that carried the gain and had
  never been through the gate.

- **`cores=1` was a notebook constraint applied everywhere.** nutpie's parallel
  native workers crash an IDE-managed Jupyter kernel on Windows, which is why the
  config default is 1 — but `main()` and both scripts inherited it, so four chains
  ran **sequentially even headless**. `main(cores=...)` plus a `--cores` flag
  (default 4) on both `pymc_kalman_filter_pt.py` and
  `scripts/export_kalman_analytics.py`. Wall-clock only; the chains, seeds and
  posterior are identical.

### Validation (2026-08-17, full 6,489-ISIN panel, both arms)

**The budget change is certified.** Gate 1 at 2000/4000 against a control run at
the previously certified 4000/4000, same day, same panel:

|                     | 2000/4000  | 4000/4000 (control) | gate   |
|---------------------|------------|---------------------|--------|
| divergences         | 0          | 0                   | 0      |
| global max R-hat    | **1.0037** | 1.0016              | < 1.01 |
| global min bulk ESS | **698.5**  | 1393.8              | > 400  |

ESS scaled almost exactly linearly in draws (698.5 × 2 = 1397 against a measured
1393.8), which is the premise the asymmetric split rested on, and R-hat stayed far
inside the gate because *tune* is untouched. Gates 2, 4 and 6 pass at both budgets;
per-time PPC coverage is 93.0 / 93.8 / 93.8 / 94.1 % against a 92 % target
(spread 0.011 against a 0.10 fail threshold), with no off-target step.

**Gate 3 fails, and it is PRE-EXISTING — not caused by the budget change.** The two
runs are identical to four significant figures:

|                                         | 2000/4000           | 4000/4000           |
|-----------------------------------------|---------------------|---------------------|
| production `sigma_base` / `nu`          | 0.1777 / 5.528      | 0.1777 / 5.530      |
| baseline `sigma_base` / `nu`            | 0.1676 / 4.822      | 0.1676 / 4.821      |
| predictive scale, baseline → production | 0.2191 → **0.2224** | 0.2191 → **0.2224** |
| mean per-name posterior sd              | 0.0141 → 0.0861     | 0.0141 → 0.0861     |

Doubling the draws moves the gate-3 statistic by ~0.0002 (`nu`'s fourth decimal)
and does not move the verdict at all. This is the **same signature 0.9.9.16
recorded at `isin_level_scale = 0.40`** — the per-name level is added *on top of*
the observation noise rather than displacing it, so `sigma_base` rises with it
(0.1676 → 0.1777) instead of falling. The revert to 0.10 reduced its magnitude by
an order of magnitude (a +1.5 % rise in total predictive scale here, against
+16 % at 0.40) but did **not** eliminate it. The per-name sd widens 6× as
predicted, so the latent is doing its job; what fails is the displacement half of
the signature.

**Consequence: `export_analytics(write=True)` was already blocked before this
release, and still is.** Nothing here made it worse. Resolving gate 3 is a
calibration question about the observation-scale model, deliberately out of scope
for this entry — it belongs with the two open items 0.9.9.16 left under *Measured
but not fixed* (T = std fails; `sigma_time` over-fitted).

Logs: `validation_0.9.9.17.log`, `validation_control_4000.log`.

## [0.9.9.16] - 2026-08-16

### Fixed

- **The out-of-support detector was one-sided.** `export_analytics` tested only
  `UPLIFT_CLIP_HI` (+500 %) and never the −95 % floor, so names whose entire
  forward-return draw set pinned at the floor shipped
  `expected_sharpe_ratio = −4.28e15` **unflagged** — the cap test matched **0 of
  6,487** rows on the 2026-08-15 export. The test is now mirrored: `er_p05`
  against the cap, `er_p95` against the floor, each direction reported
  separately. It flags 4 names, all at the floor (Kioxia, Yuanjie
  Semiconductor, AXTI, SNDK).

  Test the *percentile*, not `er_sd`: Yuanjie has `er_sd = 0.0020`, not zero, so
  a guard keyed on `er_sd == 0` would have missed it.

- **No finite guard on the reward/risk ratios.** `RiskBookModel` guarded its
  denominators with `> 0`, which a denormal `er_sd` of ~4e−16 passes. All three
  ratios now floor at `MIN_RATIO_DENOMINATOR = 1e-4` and re-check `isfinite`.
  Exported `expected_sharpe_ratio` spans −33.73 … 7.998 instead of reaching
  −4.28e15; real values are unchanged (Tencent 7.4928 before and after).

- **The analytics schema served two vintages, and the cause was structural.**
  `export_all_artifacts` writes five of the seven curated frames, and
  `scripts/export_kalman_analytics.py` **never called it** — the production path
  wrote 2 of 7 tables by construction, so the divergence reappeared on *every*
  refresh, not just once. Cross-table `er_mean` disagreement was 6,425 of 6,427;
  it is now **0 of 6,487**. Every frame carries `run_id` / `exported_at` via
  `stamp_export_provenance`, and `check_export_vintage()` reports them.

- **The GEIB launcher could not reach the database.** Running
  `dashboards/global_equity_investment_dashboard.py` puts `dashboards/` on
  `sys.path[0]`, not the repo root, so `probabilistic_ml_model` was unimportable
  and `geib/data.py`'s guarded import left `get_analytics_engine = None` — every
  card rendered against an **empty frame** behind one WARNING line. The launcher
  now inserts the repo root and that path logs at ERROR with the remedy.

- **The Kelly card sized on the wrong denominator.** `charts/kelly.py` used
  `abs(cvar_5pct_kalman)` as the loss leg of the odds ratio, but that column is a
  positive *return level*, so `b` had a median of 1.28 with an sd of 23.7. It now
  reuses the risk book's own `tail_risk` definition (median 4.27, sd 2.77).
  **This changes allocation values, not just labels.**

- **`_subsample_panel` dropped `response_mean` / `response_std`**, silently
  pushing every subsampled run onto the legacy snapshot de-standardisation that
  0.9.9.14 replaced. Affected `validate_kalman_state.py --isins N`.

### Changed

- **`sigma_isin` is now a log-linear observation-scale model.** It was
  `sigma_base * (1 + cv) / precision_weight`, i.e. two drivers, one of them
  mis-specified. Correlation of each candidate with `log |residual|` after an OLS
  fit of the 17 drift features (n = 6,533):

  | driver                 | corr            | previously in the scale?                 |
  |------------------------|-----------------|------------------------------------------|
  | `cv = pt_stddev/price` | +0.2245         | yes                                      |
  | `feat_log_mcap`        | −0.2100         | no — not even catalogued for `kalman_pt` |
  | `volatility_1m` / `1y` | +0.192 / +0.190 | no                                       |
  | `log n_analysts`       | −0.1696         | yes                                      |
  | `feat_pt_range_norm`   | +0.1150         | **documented as yes; never wired**       |
  | `feat_vol_drift`       | **−0.0349**     | documented as yes; never wired           |

  0.9.9.6 replaced the absolute realized-vol **levels** with `feat_vol_drift`,
  trading a +0.19 driver for a −0.03 one. The levels return as
  `feat_vol_level` (winsorised median of `volatility_{1m,3m,6m,1y}`, which are
  0.53–0.94 correlated so one composite replaces four columns), joined by
  `feat_log_mcap`. Both are new `mv_pymc_kalman_pt` columns at 100 % coverage,
  catalogued for `kalman_pt`, and barred from the drift design — they belong in
  the variance.

  The scale also gains a **sector-level offset** (`ZeroSumNormal` at
  `GROUP_EFFECT_SCALE`): residual sd moves **2.14×** across sectors, 1.82× across
  `size_class` and 1.68× across `trading_region`, and the model previously
  carried group effects on the mean and none on the variance.

  The old model is the exact special case `delta_* = 0, sigma_n_exponent = 1`.

- **`sigma_n_exponent` replaces the pinned `1/√n` rate.** `sd ∝ n^-0.5` assumes
  analysts are independent draws about a common truth; they anchor on each other,
  so dispersion falls far more slowly. Measured over 24 coverage levels and 6,270
  names the rate is **n^-0.306** (RMSE on log-sd 0.427 → 0.082). Learned, with the
  prior centred on the *old* value so a posterior away from it is evidence rather
  than a prior echo.

- **`isin_level_scale`: raising it 0.10 → 0.40 was tried and REVERTED.** The
  marginal case is real and is recorded so it is not re-derived: a pooled OLS with
  shared slopes and free per-time intercepts — the structure the model imposes —
  puts the between-name level sd at **0.4718**, carrying **46.6 %** of residual
  variance, and re-measuring with the crossed group effects in the design barely
  moves it (0.3854), so it is not double-counting sector/region/style/size.

  At 0.40 the level does real work — realised effect sd 0.2383 (60 % of scale)
  against 0.0468 at 0.10 — and `nu` rises further, 4.77 → 8.58. But `sigma_base`
  rises *with* it, 0.1653 → 0.2205, so the level is added **on top of** the
  observation noise rather than displacing it. Total predictive scale
  `σ·√(ν/(ν−2))` goes **0.2168 → 0.2517**, per-time coverage over-shoots to
  **98.4 %** against a 92 % target at the decision slice, and the validation
  gate's "`sigma_base` falls versus the no-latent baseline" signature inverts.

  Reverted to 0.10. **The trap:** the arithmetic measures how much per-name level
  *exists* in the residual, not how much this model can absorb without the rest of
  the scale expanding to match.

- **`draws` / `tune` raised 2000/1000 → 4000/4000.** The log-linear scale mixes
  harder than the two-term form it replaced: at 2000/1000 the full fit gave max
  global R-hat **1.0134** / min bulk ESS **460** at **zero divergences** — slow
  mixing, not bad geometry. At 4000/4000, **R-hat 1.0063 / min ESS 884**. The
  *tune* increase carries most of that. Both the validation gate and the export
  read this config, so clearing the gate via a `--draws` CLI override alone would
  certify a model the export never fits.

- **`build_fused_kalman_pt_model` gains `likelihood=`** (`'student_t'` /
  `'mixture'` / `'normal'`). `None` resolves from `robust`, so every existing
  caller is unchanged. The mixture is **opt-in and not recommended** — see below.

- **Column semantics corrected without renaming.** `cvar_5pct_kalman` is a
  conditional tail **mean**, positive for 5,475 of 6,487 rows, not a loss;
  `expected_vol_kalman` is posterior dispersion (~2.5 %), not return volatility;
  `expected_sharpe_ratio` is a t-statistic on log price-target uplift. The
  `COMMENT ON COLUMN` text and GEIB labels now say so. No column renamed, so the
  `dashboards/geib/data.py` contract is unchanged.

### Measured but not fixed

- **Calibration is improved, not resolved.** Cumulative effect on the probe panel
  (1,500 ISINs, 500 draws): `nu` **2.73 → 7.50** (off its 2.5 floor and matching
  the ν ≈ 5 the marginal kurtosis independently implies), PIT max deviation
  0.0265 → 0.0225, pooled replicated std 1.55 → 1.22 against an observed 0.99.
  **T = std still fails at every time step**, including the decision slice.

  Three hypotheses were tested and refuted along the way, recorded so they are not
  retried: the response clip (it winsorises **7 of 25,962** observations, 0.03 %);
  a two-component mixture likelihood (it improved dispersion but *regressed* PIT
  0.0265 → 0.0402); and response skew (marginal +0.766 but conditional **−0.464**,
  it flips sign). A fourth — that the three large PT-history betas are collinear —
  is also refuted: drift condition number **4.4**, max VIF **4.25**, and
  `feat_pt_accuracy_1y ⟂ feat_pt_achievement_1y` at r = 0.0028.

- **`sigma_time` remains over-fitted.** The model infers [2.30, 1.69, 1.14, 1.00]
  across the lookbacks; the same pooled-OLS decomposition says [1.32, 1.16, 1.16,
  1.00]. Part is genuine feature staleness — the snapshot drift design scores
  R² 0.718 against "now" and 0.555 against 6m-ago — but not all of it. Open.

## [0.9.9.15] - 2026-08-13

### Changed

- **`kalman_pt` swaps the price-derived market-cap / EV drift family for an EPS family.**
  `mv_pymc_kalman_pt` no longer emits `feat_mv_ev_drift`, `feat_mcap_trend_1y`,
  `feat_mcap_vs_3yavg`, `feat_ev_vs_3yavg`, or the eleven raw `market_cap_ev*`
  columns `feat_mv_ev_drift` was built from. All four were price-derived —
  `market_cap` is `last_price × shrs_out` — so they restated price history the
  drift design already carried through `feat_price_drift`,
  `feat_price_chg_pct_3m`, `feat_one_day_return` and the `feat_total_return_*`
  family, without informing *why* analysts revise a target.

  Replaced by five earnings-derived predictors plus one support counter:
  `feat_net_eps_drift` (+ `feat_net_eps_drift_n`), `feat_last_q_surprise`,
  `feat_last_y_surprise`, `feat_eps_beat_rate`, `feat_eps_beat_rate_annual`.
  Drift design: **15 → 16 columns**.

  Measured on the live 6,538-name MV under one consistent response construction:

  | set                       | k  | cond | max VIF | R²     |
  |---------------------------|----|------|---------|--------|
  | mcap/EV family (previous) | 15 | 24.7 | 4.36    | 0.6695 |
  | EPS family (this build)   | 16 | 19.7 | 4.25    | 0.6672 |

  **This is not an R² improvement** — explanatory power is flat (−0.3 % relative).
  The mcap/EV columns scored as well as they did *because* they restate price, and
  the response is analyst-implied upside off that same price, so their
  contribution was closer to duplication than independent signal. What the swap
  buys is modestly better conditioning and five predictors near-orthogonal to the
  rest of the design: every new column lands at VIF ≤ 1.19, and the strongest
  correlation among them is `r(eps_beat_rate, eps_beat_rate_annual) = +0.369`.

  Coverage on the same MV: `feat_net_eps_drift` 100 %,
  `feat_eps_beat_rate_annual` 85.2 %, `feat_last_y_surprise` 76.4 %,
  `feat_eps_beat_rate` 54.3 %, `feat_last_q_surprise` 46.6 %. The two quarterly
  columns are thin; a sparse *predictor* zero-fills to the column mean after
  standardisation, shrinking its `beta` toward 0 without dropping a name — unlike
  a sparse *response* series, which is what produced the rank-1 ICM identification
  failure (max R-hat 4.45, min ESS 4.3). Watch those betas on the next full-scale
  fit.

  `feat_mcap_country_r` is **retained** — it is the size-tilt `pm.Data` container,
  not a drift predictor. The cross-cutting trio survives unchanged in the other six
  `mv_pymc_*` views; only the `kalman_pt` `model_target` was removed.

### Added

- **`pml.signed_drift(arr[, min_points])`** (NUMERIC + DOUBLE PRECISION overloads).
  Identical to `pml.target_drift` except the denominator is `ABS(prev)`. Every
  pre-existing drift feature runs over a strictly positive series (prices, targets,
  coverage counts, realized vol), where the raw denominator is correct. EPS is not:
  a loss narrowing from `-2.00` to `-1.00` scores `(-1 - -2) / -2 = -0.5` under
  `target_drift`, recording an improvement as negative drift — and winsorising caps
  the magnitude while preserving the wrong sign. Verified against the live database:
  sign-flipped on negative series, bit-identical on positive ones, `min_points`
  guard intact. There is deliberately no `signed_drift_n`; the validity rule is
  unchanged, so `pml.target_drift_n` is the counter for both families.

### Migration

Schema/catalogue only — **the 82-column `analytics.kalman_filtered_price_targets`
layout is unchanged**, so no DDL migration and no GEIB redeploy is required for the
schema. `export_analytics` reads `market_cap` / `enterprise_value` /
`mcap_country_r`, all retained. The exported **values** will change on the next fit
(different drift design → different posterior), so the usual
`scripts/validate_kalman_state.py` → `scripts/export_kalman_analytics.py` → deploy
pair still applies when refreshing.

Applied to the live database in this order: `pml.signed_drift` helpers →
`DROP MATERIALIZED VIEW pml.mv_pymc_kalman_pt CASCADE` + recreate + unique index →
metadata reconciliation (`pml_df_metadata_populate.sql` §7c.3, TASK 4/4b, §7j) →
coverage check. `kalman_pt` reports **0 violations**; the one pre-existing DB-wide
violation (`price_target` / `feat_pt_achievement_1y`) is unchanged.

Note that TASK 4's `ON CONFLICT` only *unions* `model_targets`, so shrinking a tag
set requires the explicit `array_remove` / `DELETE` in the new §7j block. Both
statements are no-ops on a from-scratch run, so the script stays idempotent.

## [0.9.9.14] - 2026-08-10

### Fixed

- **`expm1` blow-up in the exported Monte-Carlo and price-target columns.**
  The response is winsorised to a decimal `[-0.95, +5.0]` band before `log1p`, so
  the model never observes an uplift outside it — but the two places that map
  *back out* of log space (`panel_posterior_upside`'s `expm1(latent)` and
  `summarize_panel_screen`'s `expm1(mc)`) were unbounded. Exponentiating an
  unclipped tail produced, in the 2026-08-10 export, `er_mean` up to **7.4e12**,
  `er_sd` up to **1.32e15** (82 names) and `kalman_variance` up to **5.09e9**
  (1,980 names) — while medians stayed healthy at 0.17 / 0.13 / 1.31.

  The asymmetry was latent: the pre-change table already showed `er_sd` mean
  1,443 against a median of 0.258. Widening the per-name posterior ~5.4× (the
  correct pseudo-replication fix) widened the log-space tail that `expm1` then
  exponentiated, amplifying it by ~8 orders of magnitude.

  Both directions now share one band — `UPLIFT_CLIP_{LO,HI}` /
  `LOG_UPLIFT_CLIP_{LO,HI}` — and clipping happens in **log** space before
  `expm1`, which is sign-preserving and therefore leaves `prob_pos` untouched.
  Read the result honestly: this truncates the posterior to the support the model
  was fit on, rather than extrapolating past it.

- **Drift design matrix pruned 21 → 15 columns — the last failing convergence gate.**
  `beta` was the only global parameter missing the gates on the full 6,540-name
  T=4 validation: R-hat 1.026 / bulk-ESS 140 against 1.01 / 400, at **zero
  divergences**. Zero divergences with slow mixing is the signature of poor
  conditioning, not bad geometry — and the design carried condition number
  **1,580** with a smallest eigenvalue of 0.004. Two families each restated one
  signal:

  - `feat_pt_{median,high,low}_drift` are the same `pml.target_drift()` run over
    the median / high / low target trails as `feat_pt_drift` runs over the mean.
    A consensus revision moves the whole band together: r = 0.81–0.89,
    VIF = 162 / 77 / 25 / 6.6.
  - `feat_analyst_{bullish_pct,bearish_pct,neutral_pct,conviction,rating}` are
    all functions of the same six `num_*_ratings` buckets, with `conviction`
    literally `|bullish − bearish|` and the three pct legs summing to ~1.
    r(bullish, conviction) = 0.926, r(bullish, rating) = 0.909.

  One representative survives per family — `feat_pt_drift` and
  `feat_analyst_rating`. Measured on the live MV:

  | set    | k      | cond   | max VIF | R²         |
  |--------|--------|--------|---------|------------|
  | before | 21     | 1,580  | 162.5   | 0.6538     |
  | after  | **15** | **23** | **3.8** | **0.6499** |

  69× better conditioning for a 0.6 % relative loss in explanatory power.
  `feat_analyst_rating` was kept over the `bullish_pct` + `bearish_pct` pair
  because it scored *higher* (R² 0.6499 vs 0.6463) using one fewer column, at
  99.76 % coverage. An orthogonal replacement for the dropped drift siblings
  (`high_drift − low_drift`, band-widening dynamics) was tried and rejected: it
  correlates −0.003 with the response and moved R² by 0.0001 — the siblings are
  duplication, not a distinct dispersion signal, which `feat_pt_noise_drift`
  already carries.

  This also retires the 2026-07-31 note in `KALMAN_DRIFT_EXCLUDED_FEATURES` that
  flagged `feat_analyst_conviction` as a null beta "to revisit if it stays null
  on the genuine panel". It did, and the reason is now clear.

  Implemented in `KALMAN_DRIFT_EXCLUDED_FEATURES` (the SSOT CLAUDE.md names), NOT
  in SQL: the columns stay in `mv_pymc_kalman_pt` and the catalogue, because they
  remain valid for EDA / the analytics export and the analyst family is shared
  with the `price_target` model. Flipping `pymc_role` to `'excluded'` would drop
  them from `vw_pymc_feature_catalogue` while the MV still emits them, making
  `assert_pymc_catalogue_coverage()` raise MISSING_FROM_CATALOGUE.
  Verified after the change: `kalman_pt` reports 64 OK / 0 violations.
- **`region` dropped from the crossed group effects.** It and `trading_region`
  agree for **96.12 %** of the universe (Cramér's V **0.938**; only cross-listings
  differ) against V ≤ 0.24 for every other coord pair — the same near-duplicate
  pattern as the pruned drift families. Carrying both left the effects trading off
  along a ridge, and once the per-ISIN intercept landed they became the two
  worst-mixing globals (`region_effect` R-hat 1.0130 / ESS 243,
  `trading_region_effect` 1.0121 / 236), blocking the convergence gate. They mixed
  acceptably *before* the per-ISIN latent existed (R-hat 1.002, ESS ~1.4–1.6k), so
  the redundancy was latent and only bit when another per-name absorber competed
  for the same variance. `trading_region` is kept: listing venue determines analyst
  coverage and currency, which is the mechanism the drift features measure.
- **`KalmanRunConfig.draws` settled at 2000**, after separating the two effects
  that were confounded in the `beta` failure. All measured at draws=1000, zero
  divergences throughout:

  | drift cols | group coords            | `beta` R-hat | global min ESS |
  |------------|-------------------------|--------------|----------------|
  | 21         | region + trading_region | 1.0261       | 139.8          |
  | 15         | region + trading_region | <1.0121      | 235.7          |
  | 15         | trading_region only     | 1.0262       | 296.0          |

  Removing the collinearity was necessary and it worked — min ESS rose **2.1×**
  at an unchanged budget — but it was not sufficient. Nothing collinear remains:
  drift design condition number 23, group coords orthogonal (max Cramér's V 0.24),
  no drift feature more than R² 0.25 explained by the group dummies. ESS still
  lands near 300 against a 400 gate, with R-hat bouncing 1.013–1.026 between runs
  because R-hat is itself noisy at that ESS. That is an under-sampled posterior,
  not a structural defect. An earlier build used 3000 to out-sample the *un-pruned*
  ridge (treating the symptom); reverting to 1000 then overshot the other way.

- **Exported price targets were ~1.5–2.3 pp too high on every T>1 run.**
  `prepare_kalman_panel_inputs` standardises the response tensor on the **pooled
  (isin × time)** moments of the genuine `price_target_{lb}_ago` /
  `price_{lb}_ago` trails, but `_panel_response_stats` recomputed the inverse by
  **tiling the snapshot column** across `T` — correct only for the tile-based
  panel removed in 0.9.9.10, and asserted as such in its own docstring ("tiling
  across `T` leaves the moments unchanged"). `panel_posterior_upside` therefore
  de-standardised the posterior with the wrong `(mean, std)`. Measured on the
  2026-08 6 401-name T=4 run: pooled `(0.207540, 0.249392)` vs snapshot
  `(0.224745, 0.241625)`, inflating `expected_upside` by **+2.32 pp** at a
  −0.5 latent through **+1.50 pp** at +1.0, with a 1.032 slope distortion. The
  bias flowed into `10_screen_results`, the CVaR risk book and
  `analytics.kalman_filtered_price_targets`. `KalmanPanelInputs` now carries the
  fit-time `response_mean` / `response_std`, making the inverse exact by
  construction; a panel lacking them falls back to the legacy computation with a
  **warning**, never silently. **Re-export required** — see the unit/schema pair
  rule in `CLAUDE.md`.
- **`beta_t` was a structurally-zero posterior variable.**
  `prepare_kalman_panel_inputs` builds the T>1 time axis with `np.tile`, so
  `t_scaled` is identical for every ISIN and the `t_isin_varying` guard in
  `build_fused_kalman_pt_model` always failed — a per-series slope is exactly
  spanned by the `T` free `alpha_level[t, d]` intercepts there. It was still
  published as a Deterministic pinned at 0, which read as a fitted-and-dead
  parameter in the §9 summary, gave arviz a constant to divide 0/0 on (the
  `RuntimeWarning: invalid value encountered in scalar divide` in every run log),
  and drew a flat zero line in the §13b slope panel. It is no longer materialised
  on an isin-constant axis; it returns unchanged when `t_scaled` genuinely varies.
- **Constant posterior variables no longer pollute the R-hat/ESS sweep** —
  `run_diagnostics` filters `keep_vars` through the existing
  `_degenerate_posterior_vars` before `azs.rhat` / `azs.ess`, removing the
  divide warning at source. The nan-aware reductions stay as a backstop.

### Added

- **Per-ISIN latent restored on genuine panels — the fused Kalman panel now uses
  its time axis.**
  With `beta_t` zeroed, `D == 1`, and `expected_return` a pure Deterministic (no
  per-ISIN random effect), the entire time dimension contributed nothing but `T`
  cross-sectionally-shared intercepts: each name's `T` strongly serially-correlated
  log-uplift observations were treated as `T` iid draws around **one** constant
  latent. That pseudo-replication shrank the per-name posterior sd by ~√T and
  produced over-confident `expected_upside` HDIs downstream.

  The per-ISIN random intercept (`sigma_isin_level` × a `ZeroSumNormal`,
  non-centred) is restored **for `T > 1` only**. It was dropped in an earlier
  release as non-identified, which was correct *for the T=1 cross-section*; T
  repeated observations per name identify it in the ordinary way.
  `isin_level_scale=0.0` pins it off, recovering the previous build as a
  comparison baseline.

  Measured on the live 5 605-name T=4 panel: `sigma_base` 0.3373 → 0.2871 (per-name
  signal moves out of the residual) and mean per-name posterior sd 0.0545 → 0.2694
  — the previous build was ~5× over-confident. On synthetic data with a known
  per-name level, correlation between the recovered and true latent rises from
  **0.047 to 0.951**.
- **`state_now` / `state_path` decision latent + an opt-in AR(1) state layer.**
  `state_now` is the per-ISIN quantity every decision consumer reads (screen,
  price-target Monte-Carlo, risk book, analytics export, §13b plots, prior
  predictive), resolved through `KALMAN_SCREEN_LATENT`. `achieve_prob` is now
  `sigmoid(state_now)`.

  `state_innovation_scale > 0` adds a stationary AR(1) deviation on top of the
  intercept, letting a name's level move across the lookback window.
  **It defaults to 0.0 (off)** — the expansion was tried, measured and rejected at
  T=4, per the Bayesian workflow's expansion/simplification stage:

  - A literal cumulative random walk mis-calibrates the panel. Its marginal
    variance grows as √t, so the full-scale run produced per-time predictive
    coverage of 89.9 % → 95.3 % → 97.2 % → 98.2 % against a 94 % target (too
    narrow at the oldest lookback, too wide at the snapshot), max R-hat 1.054,
    min ESS 74.
  - An unstructured doubly-centred field fixes calibration and destroys
    identification: with one observation per `(isin, time)` cell it has the same
    diagonal covariance as the measurement noise. `sigma_state` collapsed to 0.027
    against a true 0.35 and recovery correlation fell to 0.04.
  - A stationary AR(1) supplies both the temporal correlation that identifies the
    state and the constant marginal variance that calibrates it — but at T=4 it
    still cannot be cleanly separated from the per-name intercept. It bought
    +0.013 recovery correlation (0.964 vs 0.951) for min ESS 14 vs 69 and max
    R-hat 1.13 vs 1.03, with `sigma_state`/`rho` drifting between draw budgets
    (0.265→0.216, 0.63→0.49) — the "two fits disagree" signature of a
    non-identified variance component.

  The machinery is retained, tested and documented for a longer panel; `rho` is
  capped at `_STATE_RHO_MAX = 0.95` because `rho → 1` degenerates it into a second
  per-name intercept.
- **`resolve_screen_latent` / `KALMAN_SCREEN_LATENT`** — one name for the decision
  latent, resolved at every consumer (`panel_posterior_upside`,
  `summarize_panel_screen`, `export_analytics`, the §13b plots, the prior
  predictive), with a documented fallback to `risk_adj_return` so pre-0.9.9.14
  NetCDF artifacts and the notebook twin stay readable.
- **§9b model comparison — the last `❌` in this module's workflow row.**
  `run_model_comparison` refits the local-level and static arms on a common
  subsample, attaches the pointwise likelihood with `attach_log_likelihood`
  (`pm.compute_log_likelihood`; the `idata_kwargs` route is silently stripped
  under nutpie) and reports `azs.compare` / `azs.loo`, reading **`.elpd`** —
  `.elpd_loo` was removed in ArviZ 1.x and yields a silent `nan`. Opt-in via
  `enable_model_comparison`, since the log-likelihood group is ~820 MB per arm at
  full panel size; `comparison_max_isins` (default 800) bounds it and the retained
  fraction is logged so a truncated comparison never reads as a full one.
  New `09b_comparison` export section.
- **Per-time posterior-predictive coverage** (§8) — the calibration statistic that
  tests the state layer specifically. Pooled coverage can look correct while the
  model is over-confident at the oldest lookbacks and over-dispersed at the
  snapshot; a monotone drift across `t` indicates a mis-set innovation scale.
- **Opt-in second response series** — `KalmanRunConfig.panel_response_extra` plus
  the `KALMAN_PANEL_RESPONSE_EXTRA` registry activates the rank-1 ICM
  (`mu_isin_loading` / `sigma_series`), dormant while `D == 1`. The one supplied
  series, `pt_dispersion` = `log1p(price_target_stddev_{lb} / price_{lb})`, is a
  distinct signal (disagreement, not direction) with a genuine `*_ago` trail, so
  it populates the time panel rather than standardising to a near-constant column.
  Promoting it drops the collinear drift predictor `feat_pt_noise_drift`
  (response↔predictor disjointness) with a printed note. **Default off**: the
  `D > 1` path is what produced the historic R-hat 4.45 / min-ESS 4.3 freeze.
- **`scripts/validate_kalman_state.py`** — gates the re-export: convergence
  (0 divergences, R-hat < 1.01, ESS > 400), `sigma_state` bounded away from 0,
  the predicted `sigma_base`-falls / per-name-sd-widens signature versus the
  static twin, per-time PPC coverage, and the de-standardisation delta.

### Changed

- §13b panel (d) now plots the `state_path` median and 10–90 % cross-sectional
  band instead of the `beta_t` slope, falling back to `beta_t` on an isin-varying
  time axis. A visibly widening band from t=0 is the state layer working; a flat
  one means `sigma_state` collapsed.
- The `risk_adj_return` **column** in the screen and analytics export keeps its
  name and units but now reports the filtered level; `risk_adj_return` the
  **posterior variable** remains the t=0 structural anchor. The two coincide when
  `T == 1`.

## [0.9.9.13] - 2026-08-05

### Added

- **Curated bulk frames export to the analytics schema instead of CSV** —
  `_export_dataframe` (`pymc_kalman_filter_pt.py` §1c) now routes the stems in
  the new `_SQL_EXPORT_ARTIFACTS` frozenset — `04_panel_frame`,
  `09_diagnostics_01_table`, `10_screen_results`, `10_screen_mc_summary`,
  `10b_risk_analytics`, `10b_risk_book`, `10c_kalman_results` — through the new
  `_export_table`, which writes `analytics."<stem>"` via the existing
  `export_to_analytics_db(..., if_exists='replace')` **and** emits a generated
  `<stem>.sql` DDL file beside it. Everything else (the auto-numbered
  `display()` snapshots, whose stems shift between runs) still writes CSV.
  New `_sql_table_ddl` / `_sql_column_type` render PostgreSQL DDL from frame
  dtypes with no database connection required, reproducing the layout of the
  hand-written files in `sql_scripts/analytics/` (tab-indented, name-aligned,
  `ALTER TABLE … OWNER TO`). New `KALMAN_PT_SQL_EXPORT=0` emits DDL + CSV
  without touching the database; an unreachable database triggers the same CSV
  fallback automatically (first failure memoised on `_ExportState.sql_ok`, so
  later frames skip the doomed reconnect). `DB_ANALYTICS_OWNER` overrides the
  `postgres` owner. `10c_kalman_results` skips its table write when
  `export_analytics(write=True)` already wrote the identical frame to
  `analytics.kalman_filtered_price_targets` (tracked via the new
  `note_analytics_written`).
- **Per-section results tree** — artifacts now land in a subdirectory named
  for their workflow step (`01_data/`, `02_eda/`, … `14b_recommendations/`,
  `00_misc/` fallback) instead of one flat directory of 167+ files. The new
  `_EXPORT_SECTION_DIRS` tuple plus `_export_dir_for` (longest-prefix match on
  the artifact **stem**, with `_EXPORT_DIR_ALIASES` covering the one genuine
  mismatch, `10c_kalman_results` → `10c_analytics/`) is the single rule;
  `_export_path` applies it, so every writer — PNG/HTML, CSV, SQL, JSON,
  NetCDF, and the two raw `to_netcdf` calls — inherits the tree unchanged.
- **`migrate_results_layout` + `--migrate-layout` CLI** — one-off, idempotent
  re-filing of a pre-existing flat results directory into the tree. Dry-run by
  default; `--apply` performs the moves, `--results-dir` overrides the target.
- **`set_export_section`** — non-context-manager section setter for notebooks,
  which have no enclosing block per cell. `pymc_kalman_filter_pt_v3.ipynb` now
  calls `enable_artifact_export()` in §0 and scopes each cell with
  `kf.set_export_section(...)`; previously a notebook run produced **no**
  artifacts at all while the script run produced the full set.
- **`KALMAN_PT_CLEAN_RESULTS=1`** — purge a section's subdirectory on first
  entry so a re-run does not interleave with the previous run's artifacts
  (per-section counters restart each run while title slugs drift). Off by
  default so an interrupted run never destroys the previous output.
- **Reference-geometry SSOT** — new `_add_ref_line` / `_add_ref_band` helpers
  and the `_REF_LINE_KINDS` spec (`zero` / `anchor` / `emphasis`) replace ~40
  ad-hoc `add_hline` / `add_vline` / `add_vrect` call sites that had grown
  seven widths, three dash treatments and two different Plotly APIs for what is
  semantically one piece of geometry. All guides now draw with `layer='below'`
  (none did before, so every zero-line rendered **over** its own data),
  uniform opacity, and uniform annotation font.

### Changed

- **`analytics.kalman_filtered_price_targets` DDL is generated** —
  `export_analytics` calls the new `write_analytics_ddl` on every run,
  rewriting `sql_scripts/analytics/kalman_filtered_price_targets.sql` from the
  frame it actually exports. Because `if_exists='replace'` drops and recreates
  the table each run, the unit-convention header and `COMMENT ON COLUMN`
  statements only survive in the checked-in file — restored here from the new
  `_ANALYTICS_COLUMN_COMMENTS` map (see Fixed).
- **Colour and background constants** — `C_PANEL_BG` / `C_AXES_BG` replace
  eight hard-coded `#1e1e1e` literals; `CS_SEQ` / `CS_DIV` (+ `*_MPL`) replace
  the ad-hoc `Viridis` / `Magma` / `flare` / `vlag` ramps so two views of the
  same quantity no longer read as different measurements. The local
  `_C_POST, _C_PRIOR, _C_CONS, _C_REF = …` aliases that shadowed the module
  palette inside `run_granular_further_views` are gone.
- **`export_section('10b_risk_book')` → `export_section('10b_risk')`** so the
  section label and its directory agree with the `10b_risk_*` bulk stems.
- **`.gitignore`** — the two literal NetCDF paths are replaced by
  `pymc_kalman_filter_pt_results/**/*.nc` (+ the top-level glob). The literals
  would no longer match after the move, and the glob additionally covers
  `10k_universe_idata.nc`, `10k_universe_predictions.nc` and
  `13_forest_ctx_ppc_tree.nc`, which were never excluded before.
- **`KalmanRunConfig.results_dir` is now read.** The field was populated by
  `from_env()` and consumed nowhere; `get_export_state()` now prefers it over
  `KALMAN_PT_RESULTS_DIR`, so `main(config=…)` genuinely redirects artifacts.

### Fixed

- **Dark template applied in one place, at the funnel.** `_render_plotly` and
  `_write_plotly_figure` each stamped `template='arviz-tumma'`, but
  `_safe_show` — the funnel every `arviz_plots.PlotCollection` passes through —
  did not, so a collection composed against a non-default template could
  display and export light. The new `_apply_dark_template` is called from
  `_safe_show` before both display and export, and the other two call it rather
  than inlining the update, so displayed and exported output cannot diverge.
- **Silent arviz-style fallback.** `setup_plotting` swallowed a failing
  `azp.style.use('arviz-tumma')` with a bare `except … pass`, which is what
  would have made a light-theme fallback invisible. It now tries
  `_ARVIZ_STYLE_CANDIDATES` (`arviz-tumma`, then the ArviZ 1.x `arviz-variat`
  rename) and warns when neither resolves.
- **Matplotlib PNG background.** `_export_figure` relied on the seaborn
  `savefig.facecolor` rc surviving alongside `bbox_inches='tight'`; it now pins
  `facecolor=obj.get_facecolor(), edgecolor='none'` explicitly.
- **`probabilistic_ml_model/visualizations/_shared.py` did not import.** Three
  `except A, B, C:` clauses used Python-2 tuple syntax — a hard `SyntaxError`
  on Python 3, which made the entire `visualizations` package un-importable.
  Rewritten as a candidate loop over `ARVIZ_TEMPLATE_CANDIDATES` with a
  matplotlib fallback and a warning.
- **`analytics.kalman_filtered_price_targets` DDL drift.** CHANGELOG 0.9.9.7
  and `CLAUDE.md` both state the file carries a unit-convention header and
  `COMMENT ON COLUMN` statements for every return/risk/probability column;
  `grep 'COMMENT ON' sql_scripts/analytics/` returned nothing. Both are
  restored and are now regenerated on every export.
- **append/replace documentation drift.** `export_analytics`'s docstring, its
  `write=False` print and `main`'s docstring all described the analytics write
  as an *append*; the code has always used `if_exists='replace'` (drop and
  recreate). All three now say so, and note that the hand-maintained types and
  comments do not survive the recreate.
- **`09_diagnostics_01_table` stem stability.** The §9 summary reached its
  filename by being the first `display()` in the section; it now passes an
  explicit `label='table'`, since the stem is a curated SQL table name.

## [0.9.9.12] - 2026-08-02

### Added

- **Market-cap pre-selection gate for the CVaR-aware long book** —
  `compute_cvar_aware_book` (`pymc_kalman_filter_pt.py`) now restricts
  long-book candidates to top-of-country names: eligibility requires
  `mcap_country_r < mcap_r_max`, where `mcap_country_r` is the MV-derived
  `feat_mcap_country_r = (100 − market_cap_country_r) / 100` ratio (smaller
  = larger cap). New `KalmanRunConfig.mcap_country_r_max` knob (default
  `0.02` ⇔ raw rank `> 98` — the ratio-scale mirror of the §11–§13
  `min_mcap_country_rank` candidate filter) and matching keyword-only
  `mcap_r_max` override. Missing ranks fail the gate (strict, matching the
  SQL `> 98` NULL semantics); frames predating the column skip the gate
  with a warning. The §10 `results` frame (`summarize_panel_screen`) now
  carries `mcap_country_r`, `RiskBook.summary` gains `mcap_r_max` /
  `n_mcap_eligible`, and the §14b §10 sizing header + empty-book fallback
  surface the gate. Unit tests added in `tests/test_kalman_filter_pt.py`
  (first coverage of `compute_cvar_aware_book`).

### Fixed

- **`feat_mcap_country_r` doc drift** — `KalmanFilterModel.py`
  (`size_ratio` docstring + model-builder comment) claimed
  `market_cap / market_cap_3yavg` and `pml_df_metadata_populate.sql`
  claimed "1 = largest in country"; both now match the SQL SSOT
  (`mv_pymc_kalman_pt.sql`): `(100 − market_cap_country_r) / 100`,
  **~0 = largest in country**.
- **`export_analytics` hard column read** — `feat_mcap_country_r` is now
  read via the NaN-tolerant `_numcol` helper instead of an unguarded
  `model_df[...]` lookup that raised `KeyError` when the MV column was
  absent.

*(Follow-up unchanged: `pyproject.toml` and the README badge still lag at
0.9.9.5 pending the next packaging bump.)*

## [0.9.9.11] - 2026-08-01

### Added

- **Return-space structural forecast** — new `plot_kalman_forecast_returns`
  (`pymc_kalman_filter_pt.py`): observed implied returns, the smoothed
  implied-upside path + 94 % band, and **nested per-fiscal-event forecast
  bands** — predictive (`forecast_pt / last_price − 1`, incl. observation
  noise) around latent (the previously **unused**
  `implied_upside_future` draws from `KalmanFilterModel.forecast()`) — with
  a 0 % break-even line, per-horizon `+X.X%` annotations and the
  fiscal-event markers. §10K/§11/§11b/§12 now render this panel **instead
  of** the price-space `plot_kalman_forecast` (which stays exported for
  price views); their forecast tables always carry `implied_upside_pct`
  (the §10K/§12 call sites previously dropped it).
- **Portfolio & recommendations visuals** (previously console-only):
  `plot_group_signal_forest` (stacked shrunk-excess forest over the
  group-effect coords with per-coord OW/UW bands and verdict colouring),
  `plot_book_composition` (book weights + cap line + per-name CVaR5 with
  the portfolio aggregates in the title), a portfolio **star marker** +
  held-name **efficient hull** on the §10c overview risk-return map (and
  the same marker on the tail-asymmetry panel), `run_summary` decision
  panels (cohort/baseline/universe grouped metric bars; sector-tilt
  diverging bar), a §8 coverage bar against the 0.94 target, and a §9
  variance-partition stacked bar.
- **Promoted notebook builders** — `plot_screen_overview`,
  `plot_risk_return_scatter`, `plot_top_candidate_forest` move from
  notebook-inline cells into the module (SSOT; standardised hover/axes).
- **`pymc_kalman_filter_pt_v3.ipynb`** — new 42-cell notebook twin:
  `KalmanRunConfig`-driven §0 (with a commented T=4 opt-in toggle and the
  validation history), rewritten §5 generative form (per-time direct
  intercepts, `beta_t` only when `t_scaled` varies across ISINs,
  Deterministic `achieve_prob`), new §4 lookback-panel narrative,
  return-space §10K/§11/§12 narrative, §14 visuals. Supersedes
  `pymc_kalman_filter_pt_v2.ipynb`.

### Changed

- **Visualization hygiene standardised across the suite** — semantic palette
  constants (`C_POSTERIOR/C_OBSERVED/C_FORECAST/C_REF/C_HIGHLIGHT/C_VOL/
  C_DRAWS/C_ACCENT/C_MUTED`; ~70 inline hexes replaced, §11b/§12b/§13b
  one-off palettes retired) and height ladder (`H_SHORT…H_GRID`); shared
  `_hover_pct` / `_hover_price` / `_hover_prob` / `_fmt_axis` helpers +
  conventions (every trace named or hover-skipped, `<extra></extra>`
  templates, `ticksuffix='%'` on percent axes, `tickformat='.0%'` on
  decimal-probability axes, `legendgroup` on trace families, one legend
  font size); `hovermode` support in `_render_plotly` (`'x unified'` on
  time-series panels); `_render_plotly`/`_safe_show` hoisted into §1.
  Fixed defects: the unnamed "trace N" hover in `plot_fused_model_effects`
  (b), the blank hover key in the §2.4e driver facets, NaN markers hovering
  as `null` (§10b sector reference points, §10c MC fan — now masked), the
  §6 prior panel's decimal-scale axis (now %), and bare arviz figures
  (§9.3 / §11b / §12b traces titled + polished; §10 industry forest and
  §2.2 ridge gain 0 %-lines, axis titles and median sorting).
- **EDA decision context** — §2.4e driver facets annotate per-facet
  Spearman ρ (+ OLS trend when `statsmodels` is present); the §2.4f
  per-coord forests (up to 7 figures) consolidate into **one faceted
  panel** gated to the model's group-effect coords, levels sorted by
  median with universe-median reference.
- **Figure consolidation** (~6 figures fewer per run; artifact filenames
  shift accordingly): price-space shrinkage scatter removed (the
  percent-space view remains; §10c reuses the shared signed-log
  `create_kalman_vs_raw_scatter` from
  `probabilistic_ml_model.visualizations`, guarded import), duplicate
  average-upside KDE removed from `_plot_comparative_returns`, duplicate
  arviz PPC ECDF removed (the pooled hand-built overlay remains).

## [0.9.9.10] - 2026-07-31

### Added

- **Genuine (isin, time) fused-panel time dimension (T=4, opt-in)** —
  `prepare_kalman_panel_inputs` gains `history_lookbacks`
  (`KalmanRunConfig.panel_lookbacks`, e.g. `('6m', '3m', '1m')`; default `()`
  keeps the collapsed T=1 cross-section — see the validation note below):
  each lookback's implied uplift
  `price_target_{lb}_ago / price_{lb}_ago − 1` is winsorised to
  [−95 %, +500 %] and `log1p`-transformed onto the response scale, giving a
  real oldest→newest `(isin, T=4, 1)` log-uplift panel with the current
  snapshot as the final step. This activates `build_fused_kalman_pt_model`'s
  zero-anchored GRW deviations on `alpha`/`beta_t` (their innovation scales
  are now identified by the multiple time steps); `t_scaled` carries the
  standardised calendar offsets (≈ −182/−91/−30/0 days). Missing history
  cells (~a few % per trail) are filled with the name's **own** snapshot
  uplift — never a fake cross-sectional-mean observation. The legacy
  `collapse_time=False` branch, which merely `np.tile`-d the snapshot across
  the fiscal anchors (T-fold observation double-counting, the historic
  ill-conditioned posterior), is removed; `()` lookbacks reproduce the
  collapsed T=1 cross-section. `sample_posterior` bumps `tune` to ≥ 2000
  automatically when T > 1. **Validation note (2026-07-31): the first full
  T=4 run (6,398 ISINs, tune=2000, target_accept=0.9; 1.4 % of history
  cells snapshot-filled) was inconclusive-to-negative — nutpie sampled the
  full budget but its result was destroyed by a cp1252 console-encoding
  crash (headless stdout redirect; U+2009 in nutpie's output) before any
  diagnostics, and the numpyro fallback then logged 315 post-tuning
  divergences (~7.9 % of kept draws); a nutpie re-run at
  `target_accept=0.95` still logged 190, with the pathology isolated to the
  per-series level/slope + GRW-innovation block (`alpha_level` r_hat 1.06 /
  bulk-ESS 73, `sigma_alpha_innov` 1.04 / 122). The block was then
  **reparameterised (2026-08-01)**: at T > 1 the aliased scalar level +
  slope + two zero-anchored GRW deviation walks + two innovation scales are
  replaced by **T direct per-time intercepts** (`alpha_level` dims
  `(time, y_series)`, exactly identified — the same direct-Normal medicine
  as the historic T=1 ridge fix), and `beta_t` is materialised only when
  `t_scaled` genuinely varies across ISINs (fixed at 0 on an isin-constant
  lookback axis; also fixes the prior-only `beta_slope` sampled against an
  all-zero T=1 fallback covariate). `sigma_alpha_innov` / `sigma_beta_innov`
  / `z_alpha` / `z_beta` are removed, and the T>1 tune bump (→ 2000) is
  dropped with them. **Re-validation passes**: 0 divergences, worst r_hat
  1.00, worst bulk-ESS ≈ 1.6k across the per-time intercepts and all 21
  drift betas, 15.7 min end-to-end (vs ~36 min sampling alone before). The
  per-time intercepts recover a clean monotone level path (−0.164 at 6m ago
  → −0.036 now) and the T=4 panel identifies two betas that were null on
  the T=1 snapshot (`feat_analyst_conviction` +0.022, `feat_one_day_return`
  −0.024). The default stays `panel_lookbacks=()` (T=1) per the 2026-07-31
  decision; flipping it to `('6m', '3m', '1m')` is now validated-safe.**
- **`KalmanRunConfig` frozen dataclass + `from_env()`**
  (`pymc_kalman_filter_pt.py`, per the `PipelineConfig` "Configuration as
  Dataclass" pattern): consolidates the NUTS budget
  (`draws/tune/chains/cores/target_accept/random_seed/prior_draws`), screen
  Monte-Carlo (`mc_horizon`, `mc_rho`), risk book (`cvar_alpha`,
  `weight_cap`, `k_book`, `p_long`), panel lookbacks, universe-query dates /
  thresholds (`min_next_earnings`, `min_report_date`,
  `min_mcap_country_rank`, `candidate_limit`, `earnings_window_days`) and
  env plumbing (`results_dir`, `export_draws`, `fig_width_px`, `log_level`).
  `main(config=…)` threads it end-to-end; `get_run_config()` /
  `set_run_config()` expose the lazy module singleton. The previously
  **hardcoded** `next_earnings >= '2026-01-01'` /
  `income_statement_report_date >= '2025-01-01'` universe-date literals (a
  silent time-bomb as dates roll forward) now live on the config
  (ISO-validated, injection-safe) and feed `kalman_df_query()` plus the
  §11–§13 candidate queries (`market_cap_country_r` floor, `LIMIT`, earnings
  window all bind-parameterised).

### Changed

- **SSOT dedup between `KalmanFilterModel.py` and the workflow script** —
  `FISCAL_CALENDAR_COLS_ALL` / `DAY_COUNT_COLS_ALL` are now derived from
  `FISCAL_HORIZONS`; `HIST_COL_PATTERN` derives its suffix alternation from
  the new public `AGO_SUFFIX_PATTERN` (kept free of Python named groups so it
  stays a valid PostgreSQL POSIX regex); the noise-widener / tilt column
  literals resolve through the new `KALMAN_RANGE_WIDENER_FEATURE` /
  `KALMAN_CONSENSUS_SIGMA_FEATURE` / `KALMAN_VOL_DRIFT_FEATURE` /
  `KALMAN_TILT_FEATURE_ORDER` constants. New lazy exports on
  `pymc_models/__init__.py` for all of the above plus `FISCAL_HORIZONS` /
  `FiscalHorizon` / `AGO_HISTORY_RE`.
- **Helper extraction** — shared `_forecast_table` / `_plot_sigma_obs_path` /
  `_fmt_or_na` / `_display_label` / `_resolve_risk_book` replace the
  byte-identical §10K/§11/§12(±b)/§14(b) blocks; `_HDI_LO/_HDI_HI = 0.03/0.97`
  replace 13 repeated quantile literals; `_KF_SUMMARY_VARS` /
  `_SV_SUMMARY_VARS` / `_SV_TRACE_VARS` replace the per-section var-list
  literals. `KalmanFilterModel`: triplicated nan-safe z-score block →
  `_nan_zscore()`; `select_drift_features` dedup is O(n) (`seen` set);
  `_build_ago_offset_map` is `lru_cache`d and the `*_ago` regex compiled once;
  `_DEFAULT_SAMPLES/_DEFAULT_TUNE/_DEFAULT_CHAINS/_DEFAULT_TARGET_ACCEPT/
  _DEFAULT_RANDOM_SEED` + `_MIN_ESS_GATE` name the magic numbers.
- **Lazy-import contract restored** — the unguarded module-level
  `from pymc.backends.base import MultiTrace` in `KalmanFilterModel.py` and
  `AccountingAnomalyModel.py` (which made both modules hard-require pymc at
  import time) moved under `TYPE_CHECKING`; both modules now import cleanly
  with pymc absent.
- **Dataclass hygiene** — `ScreenContext` / `RiskBook` are `frozen=True,
  eq=False`; `KalmanPanelInputs` gains `eq=False` (ndarray `__eq__` hazard on
  a frozen dataclass).
- Type/docs polish: `build_price_target_history` third return annotated
  `Optional[str]` (stale `# type: ignore` dropped), `forecast` /
  `fit` / `fit_from_snapshot` return types concretised
  (`"pm_typing.Model"`, `xr` Dataset union), `main() -> dict[str, Any]`;
  stale `compute_cvar_aware_book` docstring defaults (25 / 0.80 → the actual
  50 / 0.67) and the inverted `main(robust=…)` docstring fixed; root-logger
  `logging.info` calls in `export_to_analytics_db` now use the module logger.
- **Model-results note**: `beta[feat_one_day_return]` (0.002 ± 0.006) and
  `beta[feat_analyst_conviction]` (−0.016 ± 0.017) straddle zero on the T=1
  snapshot — deliberately kept in the drift matrix (documented next to
  `KALMAN_DRIFT_EXCLUDED_FEATURES`); revisit on the genuine time panel.

### Removed

- **Legacy §4/§5 pre-fusion cross-sectional model** (`ModelData`,
  `build_model_data`, `build_kalman_pt_model`, `_CANDIDATE_GROUPS`) from
  `pymc_kalman_filter_pt.py` (~185 lines; superseded by
  `prepare_kalman_panel_inputs` + `build_panel_model`; deprecation note added
  to `docs/pymc_kalman_filter_pt.md` §5). Dead `KalmanFilterModel` APIs:
  `resolve_drift_features` (no caller), `_align_kalman_features` (+ its
  now-unused `_feature_alignment` imports), and the unused
  `FISCAL_HORIZON_LABELS` / `DAY_COUNT_HORIZON_LABELS` dicts.

> Packaging note: `pyproject.toml` and the README badge still lag at 0.9.9.5
> pending the next packaging bump (recurring follow-up).

## [0.9.9.9] - 2026-07-31

### Added

- **Piotroski F-score composite (SQL)** — new `pml.piotroski_f_score(roa, roa_prev,
  cfo, ni, ltde, ltde_prev, cr, cr_prev, shrs, shrs_prev, gpm, gpm_prev, at,
  at_prev) → INTEGER` scalar function (`sql_scripts/pml/piotroski_f_score.sql`;
  `NUMERIC` + `DOUBLE PRECISION` overloads, `IMMUTABLE PARALLEL SAFE`): the
  classic 9-signal 0–9 fundamental-quality composite (positive ROA, positive
  CFO, rising ROA, accruals quality CFO > NI, falling LT-debt/equity, rising
  current ratio, no share dilution, rising gross margin, rising asset
  turnover). NULL-tolerant: a NULL comparison scores 0 for that signal, never
  NULL overall. Plus `pml.calc_piotroski_f_score(p_isin DEFAULT NULL) →
  TABLE(isin, piotroski_f_score)`, a thin set-returning LTM screener wrapper
  (`sql_scripts/pml/calc_piotroski_f_score.sql`).
- **New raw fundamentals column families on `pml.pml_df` / `pml.staging`** —
  the Piotroski inputs, each as level (`_ltm`/`_fy`/`_fq`) plus
  `neg1..neg4` fy/fqfq lag variants: `return_on_assets_roa_pct_*`,
  `asset_turnover_*`, `quick_ratio_*`, `current_ratio_*`,
  `long_term_debt_equity_*`, `net_income_*`. Mapped from the vendor headers in
  `import_pml_data.sql` and registered (roles, `data_type`, descriptions) in
  `pml_df_metadata_populate.sql`; regional `data/pml/` and `data/screening_*`
  CSV snapshots refreshed to carry the new vendor columns.
- **Piotroski features on `pml.mv_pymc_kalman_pt`** — four per-fiscal-year
  composites `feat_piotroski_f_score_{fy,neg1fy,neg2fy,neg3fy}`
  (`pml.piotroski_f_score` over consecutive lag pairs, fy vs neg1fy …
  neg3fy vs neg4fy) plus their median `feat_median_piotroski_f_score`,
  registered as `mutable_predictor` catalogue rows for `kalman_pt`. Only the
  median enters the fused drift design matrix as the fundamental-quality
  level; the per-year components are collinear with it and are barred via the
  new `KALMAN_PIOTROSKI_COMPONENT_FEATURES` frozenset (folded into
  `KALMAN_DRIFT_EXCLUDED_FEATURES` /
  `KalmanFilterPriceTarget.select_drift_features()` in
  `probabilistic_ml_model/pymc_models/KalmanFilterModel.py`) — EDA / analytics
  export only.
- **Analytics export columns** — the `kalman_results` export frame and
  `analytics.kalman_filtered_price_targets` gain `n_analysts` and
  `piotroski_f_score_{median,fy,neg1fy,neg2fy,neg3fy}`
  (`sql_scripts/analytics/kalman_filtered_price_targets.sql`). As usual,
  schema changes ship as a pair: re-run `export_analytics(write=True)` and
  redeploy the GEIB dashboard.
- **New analytics / screening DDL** — `analytics."10b_risk_book"`
  (`sql_scripts/analytics/10b_risk_book.sql`, persisted CVaR-aware risk book)
  and `pml.screening_global_yields`
  (`sql_scripts/pml/screening_global_yields.sql`, backed by the new
  `data/playground/screening_global_yields.csv` snapshot).
- **Import / registry hardening** — `import_pml_data.sql` fails fast if the
  vendor CSV header has no `ISIN` column, filters ISIN-less rows at `\copy`
  time (`WHERE NULLIF(BTRIM("ISIN"), '') IS NOT NULL`) and asserts none reach
  `pml_df` before the `TRUNCATE`; `feature_registry.sql` pre-flights that
  `equities_schema_metadata` is seeded (its FK target) with an actionable
  error; `equities_schema_metadata_setup.sql` documents why it must not
  `DROP TABLE … CASCADE` (would silently drop the registry FK).

### Changed

- **Kalman drift design matrix slimmed** — the high/low/median analyst-target
  trail drifts (`feat_pt_high_drift`, `feat_pt_low_drift`,
  `feat_pt_median_drift`) were removed from the expected drift-feature set and
  the EDA drift panels in `pymc_kalman_filter_pt.py` (they remain available on
  the MV / in the catalogue); `feat_median_piotroski_f_score` joins as the
  fundamental-quality drift predictor.

> Follow-up (recurring): `pyproject.toml` / README badge still lag the
> CHANGELOG version pending the next packaging bump.

## [0.9.9.8] - 2026-07-26

### Changed

- **Dependency refresh (Bayesian stack)** — bumped the locked versions and the
  matching floors in `Pipfile`, `requirements.txt` and `pyproject.toml`:
  - `pymc` 6.1.0 → **6.2.0** and `pytensor` 3.1.2 → **3.2.3** (coupled pair:
    pymc 6.1 caps pytensor `<3.2`; pymc 6.2 requires pytensor `>=3.2.2,<3.3`).
  - `jax` / `jaxlib` 0.10.2 → **0.11.0** (mutually pinned pair; numpyro 0.21,
    blackjax and optax all accept it).
  - `bambi` 0.18.0 → **0.19.0**, `blackjax` 1.5 → **1.6.2**,
    `pandas` 3.0.3 → **3.0.5**, `tqdm` 4.68.4 → **4.69.1**.
  - Full `pipenv lock` re-resolve also refreshed in-range transitive pins
    (certifi, dash 4.4.1, lightgbm 4.7.0, matplotlib 3.11.1, pyarrow 25.0.0,
    filelock, tzdata, xarray-einstats 0.11.0, …) and dropped the miniKanren
    deps (`cons`, `etuples`, `logical-unification`, `minikanren`, `toolz`)
    that pytensor 3.2.3 moved out of its core requirements.
- **Deferred (blocked) updates** — documented in-line in the three dependency
  files:
  - `numpy` stays **2.4.6** (not 2.5.1): numba 0.65.1 *and* 0.66.0 both
    require `numpy<2.5`, and numba is the project's default PyTensor backend.
  - `numba` stays **0.65.1** (not 0.66.0): pytensor 3.2.x requires
    `numba<=0.65.1` (0.66 also needs llvmlite 0.48; lock has 0.47).

> Follow-up note (recurring): `pyproject.toml` `version` and the README badge
> still lag at 0.9.9.5 pending the next packaging bump.

## [0.9.9.7] - 2026-07-25

### Changed

- **Decimal-unit consistency across the Kalman price-target pipeline** — all
  persistent frames (`screen.results`, `RiskBook.analytics` / `.book`, the
  `kalman_results` export row-set) and `analytics.kalman_filtered_price_targets`
  now store raw decimal returns (0.25 = +25%); percent scaling is applied only
  at visualization / print boundaries.
  - `summarize_panel_screen` columns renamed: `expected_upside_pct` →
    `expected_upside`, `implied_upside_pct` → `implied_upside`,
    `total_return_{ytd,5y,10y}_pct` → `total_return_{ytd,5y,10y}`,
    `tr_cagr_3y_pct` → `tr_cagr_3y` (values now decimal).
  - `compute_cvar_aware_book` columns renamed: `band_width_pct` → `band_width`,
    `cvar05_pct` → `cvar05`, `exp_vol_pct` → `exp_vol`, `tail_risk_pct` →
    `tail_risk` (values now decimal; the tail-risk floor is 0.01 = 1pp).
    `RiskBook.summary` return metrics (`port_up`, `port_cvar`, `wavg_cvar`,
    `port_vol`) are decimal. `reward_to_cvar` (STARR) is numerically unchanged
    (ratio of same-unit terms).
  - **BREAKING (DB consumers):** `cvar_5pct_kalman` and `expected_vol_kalman`
    in `analytics.kalman_filtered_price_targets` changed from percent to
    decimal units. Re-run the export (`export_analytics(write=True)`) together
    with deploying the updated GEIB dashboard.
- **`expected_sharpe_ratio` redefined** — previously
  `expected_upside / posterior-draw dispersion` (a parameter-uncertainty ratio
  that produced unrealistically large values); now `er_mean / er_sd` of the
  structural-TS Monte-Carlo forward-return distribution. The old
  posterior-dispersion ratio survives internally as
  `RiskBook.analytics['ret_vol_ratio']` (§14b risk-adjusted screen) but is no
  longer exported. Dashboard sorts on `expected_sharpe_ratio` will re-rank.
- **GEIB dashboard aligned with decimal storage** — `kelly.py` no longer
  divides `cvar_5pct_kalman` by 100 (and scales it ×100 for display);
  `monte_carlo_forecast.py` formats CVaR with the shared percent renderer.

### Added

- **`er_sd`** — new column in `summarize_mc_returns` output, the §10 screen and
  the analytics export: pooled std of the MC forward-return draws (denominator
  of `expected_sharpe_ratio`).
- **Unit documentation in the analytics DDL** —
  `sql_scripts/analytics/kalman_filtered_price_targets.sql` now carries a
  unit-convention header and `COMMENT ON COLUMN` statements for every
  return/risk/probability column.

### Fixed

- **Double-scaling in `plot_kalman_results_overview`** — `expected_vol_kalman`
  and `cvar_5pct_kalman` (already percent under the old convention) were
  multiplied by 100 again, rendering panels (a)/(d) 100× off, and the
  dimensionless `reward_to_cvar` STARR colourbar was scaled ×100. All four
  series now scale correctly from decimal storage.
- **GEIB `data.py` column inventory** — added the missing
  `expected_vol_kalman` / `expected_sharpe_ratio` (and new `er_sd`) to
  `NUMERIC_COLUMNS` and fixed the `exchange_name_name` typo, so the
  empty-frame fallback no longer drops columns `high_conviction.py` requests.

> Follow-up (recurring): `pyproject.toml` / README badge still lag the
> CHANGELOG version pending the next packaging bump.

## [0.9.9.6] - 2026-07-24

### Added

- **`feat_vol_drift` / `feat_vol_drift_n`** — `pml.mv_pymc_kalman_pt` now emits
  the winsorised drift across the realized-vol term structure
  (`volatility_{1m,3m,6m,1y}`, mirroring `feat_pt_noise_drift`) plus its
  `pml.target_drift_n` valid-pair counter, registered as engineered self-rows
  (`mutable_predictor`, `kalman_pt`) with descriptions in
  `pml_df_metadata_populate.sql`.
- **Analyst rating-mix / achievement features on `mv_pymc_kalman_pt`** — copied
  from `mv_pymc_price_target`: `feat_holds`, `feat_buys`, `feat_sells`,
  `feat_no_opinion`, `feat_analyst_{bullish,bearish,neutral}_pct`,
  `feat_analyst_conviction`, `feat_analyst_rating`, `feat_pt_achievement_1y`,
  `feat_pt_accuracy_1y`, `feat_pt_range_hit_rate`; carrier aliases and
  self-rows wired for `kalman_pt` (all `mutable_predictor`). Registering
  `feat_buys` / `feat_sells` self-rows also fixes the pre-existing
  `assert_pymc_catalogue_coverage()` failure for `price_target`.
- **Raw observed trails on `mv_pymc_kalman_pt`** — `price_{1d,mtd,ytd}_ago`,
  the eight-horizon `price_target_stddev_*_ago` dispersion trail, and
  `price_target_num_6m_ago` are now emitted so their catalogue roles are live.
- **TASK 4b metadata backfill** — `data_type` + `description` for all
  previously-NULL engineered self-rows (`feat_pt_drift`, `feat_avg_beta`,
  `feat_mv_ev_drift`, `feat_mcap_*`, drift `*_n` counters, …).

### Changed

- **`feat_vol_{1m,3m,6m,1y}` removed from `mv_pymc_kalman_pt`** — replaced by
  `feat_vol_drift(_n)`; the raw `volatility_*` columns no longer target
  `kalman_pt` (their `price_target` / `credit_risk` aliases are unchanged).
- **kalman_pt catalogue role flips** (per-model overrides in
  `pml.pml_df_feature_alias`): `last_price` and `price_target_stddev`
  (`feat_pt_noise_sigma`) → `observed`; all 22 `total_return_*` /
  `tot_return_pct_cagr_*` aliases → `observed`; `price_target_stddev_*_ago`
  and `price_{1d,mtd,ytd}_ago` trails → `observed`; `price_target_num_6m_ago`
  → `constant_data`; `feat_pt_drift_n` / `feat_price_drift_n` →
  `mutable_predictor`. (The requested `SET column_name = 'observed'` clauses
  were not implementable — `column_name` is the catalogue join key.)
- **`pymc_kalman_filter_pt.py`** — σ_obs widener is now
  `sigma_obs_base * (1 + range + cv + 0.5 * max(feat_vol_drift, 0)) /
  sqrt(n_analysts)`; the stochastic-volatility re-fits drop the
  `feat_vol_*`-derived `realized_vol` anchor (constant `log(scale)` anchor;
  the `fit(realized_vol=…)` API is unchanged); the CVaR book's
  `expected_vol_kalman` export is now the posterior upside-draw dispersion.
- **`KalmanFilterModel.py`** — `KalmanPanelInputs.expected_vol` renamed to
  `vol_drift`; provenance containers `feat_expected_vol(_z)` renamed to
  `feat_vol_drift(_z)`.

## [0.9.9.5] - 2026-06-08

### Added

- **Fused price-target panel model** — new `build_fused_price_target_model`
  builder in `probabilistic_ml_model/pymc_models/PriceTargetModel.py`
  (exported lazily from `pymc_models/__init__.py`), backed by a
  `prepare_price_target_panel_inputs` helper that assembles the 3-D
  `(isin, time, y_series)` response tensor, fiscal-anchor time matrix,
  `sqrt(n_analysts)` weights and standardised predictor matrix column-aligned
  with the materialized-view DDL
  ([`6897b91`](https://github.com/Kabenge42/PML_Finance_Project/commit/6897b91)).
- **Kalman `feat_implied_upside` surface** — `KalmanFilterPriceTarget` gained
  the pure-NumPy `implied_upside_from_state` helper and a `last_price`-anchored
  `implied_upside` Deterministic on `fit(...)`, mirroring the SQL
  `calc_change_ratio(price_target, last_price)` feature
  ([`6897b91`](https://github.com/Kabenge42/PML_Finance_Project/commit/6897b91)).
- **Test coverage** — added `tests/test_kalman_filter_pt.py` and
  `tests/test_price_target_panel.py` exercising the deterministic helpers and
  PyMC graph construction for the refactored Kalman / price-target stack
  ([`6897b91`](https://github.com/Kabenge42/PML_Finance_Project/commit/6897b91)).
- **Notebooks & rendered docs** — added `pymc_kalman_filter_pt.ipynb`,
  `pymc_price_target_v3.ipynb`, the rendered `pymc_kalman_filter_pt.md` and
  reference MyST docs (`Forecasting_with_structural_timeseries.myst.md`,
  `MvGaussianRandomWalk_demo.myst.md`, `bayesian_var_model.myst.md`,
  `stochastic_volatility.myst.md`)
  ([`6897b91`](https://github.com/Kabenge42/PML_Finance_Project/commit/6897b91)).

### Changed

- **`KalmanFilterModel.py` refactor** — reworked the PyMC helpers with
  schema-aligned data-preparation functions for the
  `pymc_kalman_filter_pt.ipynb` notebook, plus touch-ups to
  `_price_target_mc.py` and `_feature_alignment.py`
  ([`6897b91`](https://github.com/Kabenge42/PML_Finance_Project/commit/6897b91)).
- **`sql_scripts/pml/mv_pymc_kalman_pt.sql`** and
  `mv_pymc_price_target.sql` extended with fiscal-anchor date columns,
  `feat_implied_upside` and additional drift features
  (`feat_pt_high_drift`, `feat_pt_low_drift`, `feat_pt_median_drift`,
  `feat_coverage_drift`); refreshed companion `sql_scripts/pml/*` helpers,
  `pml_df.sql`, `pml_feature_catalogue.sql`, `pml_df_metadata_populate.sql`
  and `feature_registry.sql`
  ([`6897b91`](https://github.com/Kabenge42/PML_Finance_Project/commit/6897b91)).
- **Notebook / data / config refresh** — re-ran the PyMC notebooks
  (`pymc_dcf.ipynb`, `pymc_earnings_beat.ipynb`, `pymc_price_target.ipynb`),
  regenerated the regional PML CSV snapshots (`data/pml/pml_{us,eu,apac,rotw}.csv`),
  and refreshed `CLAUDE.md`, `environment_variables.txt` and `.idea/*` settings
  ([`6897b91`](https://github.com/Kabenge42/PML_Finance_Project/commit/6897b91)).

### Notes

- Patch-level bump (`0.9.9.4 → 0.9.9.5`) — the release packages a
  Kalman / price-target PyMC refactor, new fused-panel model builder, SQL
  feature-surface additions, notebooks and tests; no breaking public-API
  changes (existing `fit(...)` signatures remain backwards compatible).
- Follow-up: bump `pyproject.toml` `version` and the README badge to
  `0.9.9.5` in the next packaging pass (left untouched here to keep this
  CHANGELOG refresh purely a documentation change).

[0.9.9.5]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.9.4...v0.9.9.5

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