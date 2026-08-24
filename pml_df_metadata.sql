-- =============================================================================
-- PML SCHEMA METADATA TABLE
-- =============================================================================
-- Two-dimensional metadata catalogue for pml.pml_df:
--   * (category, feature_role) -> legacy data-centric vocabulary used for
--     domain/SQL filtering during feature engineering (see
--     pml_df_metadata_populate.sql for the full vocabulary).
--   * (pymc_role, model_targets) -> PyMC alignment layer used to drive
--     `pm.Data` container assignment and per-model feature selection.
--       - pymc_role vocabulary:
--           coord | index | observed | mutable_predictor |
--           constant_data | derived_input | excluded
--       - model_targets: TEXT[] keyed by MODEL_FEATURE_CONTAINERS
--         (earnings_beat, price_target, kalman_pt, kalman_pt_v2, dcf_pt,
--          dividend_safety, credit_risk, accounting_anomaly).
--
-- ADDING A MODEL TARGET IS A THREE-FILE CHANGE. All of:
--   1. the two CHECK constraints below (each repeated once per duplicate
--      CREATE TABLE block, so FOUR edits in this file);
--   2. the idempotent ALTER at the top of pml_df_metadata_populate.sql, which
--      re-applies both CHECKs to a live database that already has the table;
--   3. the mv_map VALUES list inside vw_pymc_catalogue_coverage_check in
--      pml_feature_catalogue.sql.
-- Miss (3) and the new MV's columns are never checked, which reads as passing
-- rather than failing. Miss (1) -- as happened to 'kalman_pt_v2' between
-- 2026-08-19 and 2026-08-21 -- and a rebuild from this file alone REJECTS every
-- row for that target until the populate script's ALTER runs, because this DDL
-- and the populate script disagreed about the vocabulary.
-- =============================================================================
-- =============================================================================
-- PML SCHEMA METADATA TABLE FIXED
-- =============================================================================
DROP TABLE IF EXISTS pml.pml_df_metadata CASCADE;

-- Re-create the metadata table with appropriate options
CREATE TABLE IF NOT EXISTS pml.pml_df_metadata
(
	column_name      TEXT PRIMARY KEY,
	category         TEXT   NOT NULL DEFAULT 'n/a',
	feature_role     TEXT   NOT NULL,
	feature_alias    TEXT, -- canonical alias used inside mv_pymc_* materialized views
	ordinal_position INTEGER,
	description      TEXT,
	data_type        TEXT,
	pymc_role        TEXT,
	model_targets    TEXT[] NOT NULL DEFAULT ARRAY []::TEXT[],
	updated_at       TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
	-- ------------------------------------------------------------------
	-- Controlled vocabularies (see CK_VOCABULARIES note at the foot of
	-- this file). These three columns drive Python feature selection
	-- end-to-end, and an unrecognised value fails SILENTLY rather than
	-- loudly -- hence the CHECKs.
	-- ------------------------------------------------------------------
	CONSTRAINT ck_pml_df_metadata_pymc_role
		CHECK (pymc_role IS NULL OR pymc_role IN
		       ('coord', 'index', 'observed', 'mutable_predictor',
		        'constant_data', 'derived_input', 'excluded')),
	CONSTRAINT ck_pml_df_metadata_feature_role
		CHECK (feature_role IN
		       ('id', 'categorical', 'date', 'target', 'predictor', 'count',
		        'score', 'historical', 'surprise', 'revision', 'metadata')),
	-- `<@` = "is contained by": every element must be a known model target.
	-- The empty-array default trivially satisfies this.
	CONSTRAINT ck_pml_df_metadata_model_targets
		CHECK (model_targets <@ ARRAY ['earnings_beat', 'price_target', 'kalman_pt',
			'kalman_pt_v2', 'dcf_pt', 'dividend_safety', 'credit_risk',
			'accounting_anomaly']::TEXT[])
);

-- References fix to avoid foreign key violation
CREATE TABLE IF NOT EXISTS pml.pml_df_feature_alias
(
	column_name   TEXT NOT NULL,
	model_target  TEXT NOT NULL,
	feature_alias TEXT NOT NULL,
	-- Per-model pymc_role override. The same source column can play a
	-- different PyMC role in different models (e.g. price_target_stddev is
	-- globally 'observed' but is a 'mutable_predictor' for kalman/price_target).
	-- vw_pymc_feature_catalogue resolves the effective role via
	-- COALESCE(fa.pymc_role, md.pymc_role). NULL = inherit the global role.
	pymc_role TEXT,
	PRIMARY KEY (column_name, model_target),
	CONSTRAINT fk_column_name FOREIGN KEY (column_name) REFERENCES pml.pml_df_metadata (column_name) ON DELETE CASCADE,
	-- Same vocabularies as pml_df_metadata: an override with an unknown role
	-- would win the COALESCE in vw_pymc_feature_catalogue and silently drop
	-- the column from that model's feature list.
	CONSTRAINT ck_pml_df_feature_alias_pymc_role
		CHECK (pymc_role IS NULL OR pymc_role IN
		       ('coord', 'index', 'observed', 'mutable_predictor',
		        'constant_data', 'derived_input', 'excluded')),
	CONSTRAINT ck_pml_df_feature_alias_model_target
		CHECK (model_target IN ('earnings_beat', 'price_target', 'kalman_pt',
		                        'kalman_pt_v2', 'dcf_pt', 'dividend_safety',
		                        'credit_risk', 'accounting_anomaly'))
);

CREATE INDEX IF NOT EXISTS idx_pml_df_feature_alias_model ON pml.pml_df_feature_alias (model_target);

-- Recreate GIN index with metadata defaults
CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_feature_role ON pml.pml_df_metadata (feature_role);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_category ON pml.pml_df_metadata (category);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_pymc_role ON pml.pml_df_metadata (pymc_role);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_feature_alias ON pml.pml_df_metadata (feature_alias);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_model_targets ON pml.pml_df_metadata USING gin (model_targets);

COMMENT ON TABLE pml.pml_df_metadata IS 'Metadata for pml.pml_df. (category, feature_role) drive domain/data-centric SQL filters; (pymc_role, model_targets) drive PyMC pm.Data container assignment and per-model feature selection. pymc_role vocabulary: coord | index | observed | mutable_predictor | constant_data | derived_input | excluded. model_targets is a TEXT[] keyed by MODEL_FEATURE_CONTAINERS (earnings_beat, price_target, kalman_pt, kalman_pt_v2, dcf_pt, dividend_safety, credit_risk, accounting_anomaly).';
-- Create a metadata table documenting available pml_df schema columns.
-- NOTE: this is the same definition as the block above (both are
-- CREATE TABLE IF NOT EXISTS, so whichever runs first wins). The constraints
-- are repeated verbatim so the table is correctly constrained either way.
CREATE TABLE IF NOT EXISTS pml.pml_df_metadata
(
	column_name      TEXT PRIMARY KEY,
	category         TEXT   NOT NULL DEFAULT 'n/a',
	feature_role     TEXT   NOT NULL,
	feature_alias    TEXT, -- canonical alias used inside mv_pymc_* materialized views
	ordinal_position INTEGER,
	description      TEXT,
	data_type        TEXT,
	pymc_role        TEXT,
	model_targets    TEXT[] NOT NULL DEFAULT ARRAY []::TEXT[],
	updated_at       TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT ck_pml_df_metadata_pymc_role
		CHECK (pymc_role IS NULL OR pymc_role IN
		       ('coord', 'index', 'observed', 'mutable_predictor',
		        'constant_data', 'derived_input', 'excluded')),
	CONSTRAINT ck_pml_df_metadata_feature_role
		CHECK (feature_role IN
		       ('id', 'categorical', 'date', 'target', 'predictor', 'count',
		        'score', 'historical', 'surprise', 'revision', 'metadata')),
	CONSTRAINT ck_pml_df_metadata_model_targets
		CHECK (model_targets <@ ARRAY ['earnings_beat', 'price_target', 'kalman_pt',
			'kalman_pt_v2', 'dcf_pt', 'dividend_safety', 'credit_risk',
			'accounting_anomaly']::TEXT[])
);

-- ---------------------------------------------------------------------------
-- Per-model alias side table: the same source column may map to different
-- aliases across the per-model materialized views (e.g. price_target ->
-- observed_pt in mv_pymc_kalman_pt/mv_pymc_dcf_pt and feat_implied_upside in
-- mv_pymc_price_target). Keep pml_df_metadata.feature_alias as the default
-- (model-agnostic) alias and override per model here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pml.pml_df_feature_alias
(
	column_name   TEXT NOT NULL REFERENCES pml.pml_df_metadata (column_name) ON DELETE CASCADE,
	model_target  TEXT NOT NULL,
	feature_alias TEXT NOT NULL,
	-- Per-model pymc_role override (NULL = inherit pml_df_metadata.pymc_role).
	-- Resolved in vw_pymc_feature_catalogue via COALESCE(fa.pymc_role, md.pymc_role).
	pymc_role TEXT,
	PRIMARY KEY (column_name, model_target),
	CONSTRAINT ck_pml_df_feature_alias_pymc_role
		CHECK (pymc_role IS NULL OR pymc_role IN
		       ('coord', 'index', 'observed', 'mutable_predictor',
		        'constant_data', 'derived_input', 'excluded')),
	CONSTRAINT ck_pml_df_feature_alias_model_target
		CHECK (model_target IN ('earnings_beat', 'price_target', 'kalman_pt',
		                        'kalman_pt_v2', 'dcf_pt', 'dividend_safety',
		                        'credit_risk', 'accounting_anomaly'))
);

CREATE INDEX IF NOT EXISTS idx_pml_df_feature_alias_model ON pml.pml_df_feature_alias (model_target);

-- Indices mirror those created in pml_df_metadata_populate.sql so the DDL is
-- self-sufficient when the table is (re)created from scratch.
CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_feature_role ON pml.pml_df_metadata (feature_role);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_category ON pml.pml_df_metadata (category);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_pymc_role ON pml.pml_df_metadata (pymc_role);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_feature_alias ON pml.pml_df_metadata (feature_alias);

-- GIN index over model_targets so `WHERE 'earnings_beat' = ANY(model_targets)`
-- (and `model_targets @> ARRAY['dcf_pt']`) become index-backed.
CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_model_targets ON pml.pml_df_metadata USING gin (model_targets);

COMMENT ON TABLE pml.pml_df_metadata IS 'Metadata for pml.pml_df. (category, feature_role) drive domain/data-centric SQL filters; (pymc_role, model_targets) drive PyMC pm.Data container assignment and per-model feature selection. pymc_role vocabulary: coord | index | observed | mutable_predictor | constant_data | derived_input | excluded. model_targets is a TEXT[] keyed by MODEL_FEATURE_CONTAINERS (earnings_beat, price_target, kalman_pt, kalman_pt_v2, dcf_pt, dividend_safety, credit_risk, accounting_anomaly).';

COMMENT ON COLUMN pml.pml_df_metadata.pymc_role IS 'PyMC pm.Data container kind for this column. Aligns with arviz.InferenceData groups: coord/index -> idata.constant_data + posterior.coords; observed -> idata.observed_data; mutable_predictor/constant_data -> idata.constant_data (mutable_predictor supports pm.set_data for OOS); derived_input -> must be transformed before pm.Data; excluded -> never wrapped.';

COMMENT ON COLUMN pml.pml_df_metadata.model_targets IS 'Array of PyMC model names from probabilistic_ml_model.pymc_models that consume this column. Mirrors MODEL_FEATURE_CONTAINERS keys in pymc_expected_returns_model.ipynb.';

COMMENT ON COLUMN pml.pml_df_metadata.feature_alias IS 'Default (model-agnostic) alias used inside pml.mv_pymc_* materialized views. For model-specific overrides see pml.pml_df_feature_alias.';

COMMENT ON COLUMN pml.pml_df_feature_alias.pymc_role IS 'Per-model pymc_role override. NULL inherits pml_df_metadata.pymc_role. vw_pymc_feature_catalogue resolves the effective role via COALESCE(fa.pymc_role, md.pymc_role), so a column can be globally observed/derived_input yet act as a mutable_predictor for a specific model.';

COMMENT ON TABLE pml.pml_df_feature_alias IS 'Per-model alias overrides for source columns in pml.pml_df_metadata. Surfaces through pml.vw_pymc_feature_catalogue.feature_alias and the notebook''s MODEL_FEATURE_CONTAINERS registry. Multi-source engineered features (e.g. the normalized analyst-sentiment feat_analyst_bullish_pct / feat_analyst_bearish_pct / feat_analyst_neutral_pct / feat_analyst_conviction columns in pml.mv_pymc_price_target, each derived from all six num_*_ratings buckets) record provenance against a single representative source column per (column_name, model_target) key.';

COMMENT ON COLUMN pml.pml_df_metadata.feature_role IS 'Coarse ML role used for SQL filtering during feature engineering. Vocabulary (CHECK-enforced): id | categorical | date | target | predictor | count | score | historical | surprise | revision | metadata. Also the deterministic seed for pymc_role -- see the CASE in pml_df_metadata_populate.sql step 2.';

-- ---------------------------------------------------------------------------
-- CK_VOCABULARIES — why the three CHECK constraints above exist.
--
-- `feature_role`, `pymc_role` and `model_targets` drive Python feature
-- selection end-to-end:
--
--     pml_df_metadata (global pymc_role, model_targets[])
--       -> pml_df_feature_alias (per-model feature_alias AND pymc_role override)
--       -> vw_pymc_feature_catalogue  (COALESCE(fa.pymc_role, md.pymc_role),
--                                      rows with 'excluded' filtered out)
--       -> vw_pymc_feature_aliases    (the arrays consumed by each model's
--                                      _resolve_<model>_feature_aliases())
--
-- An unrecognised value in any of the three fails **silently**, not loudly:
-- vw_pymc_feature_catalogue only filters `<> 'excluded'`, so a typo'd role
-- neither raises nor matches any consumer -- the column simply vanishes from
-- the model's feature list and is later reindexed to 0.0 by the alignment
-- layer. That is precisely the failure class pml.assert_pymc_catalogue_coverage()
-- was written to catch after the fact; these CHECKs stop it at write time.
--
-- The vocabularies were previously documented only in prose (this file's
-- header and pml_df_metadata_populate.sql lines 9-34). Keep them in sync with
-- CLAUDE.md ("pymc_role Enum" table) when adding a model or a role.
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Fused Kalman MvGRW panel (kalman_pt) — role notes.
--   The fused cross-sectional Kalman model (build_fused_kalman_pt_model) reuses
--   the existing kalman_pt catalogue roles in fused state-space positions:
--     * feat_vol_drift / feat_vol_drift_n (mutable_predictor, engineered
--       self-rows over volatility_{1m,3m,6m,1y}) -> realized-vol DRIFT: how
--       price volatility itself is evolving; widens sigma_obs via
--       (1 + range + cv + 0.5 * max(vol_drift, 0)) / sqrt(n_analysts). The
--       absolute feat_vol_* levels are no longer emitted; feat_avg_beta is the
--       systematic-risk driver of risk_adj_return.
--     * feat_pt_noise_sigma (observed) -> the current consensus-dispersion
--       snapshot; still forms cv = feat_pt_noise_sigma / last_price in the
--       sigma_isin widener. Its price_target_stddev_*_ago trail (observed)
--       exposes the evolving measurement-noise level.
--     * last_price (observed) -> the realised spot anchor; the price_*_ago
--       spot trail (observed) pools with the target trail.
--     * n_analysts (constant_data) -> sqrt(n) precision weight.
--     * the price_target_*_ago response trails (observed) -> the (isin, time,
--       y_series) MvGRW response tensor.
--     * feat_total_return_* / feat_tr_cagr_* (observed) -> realised-return
--       outcome series available to the ICM panel as observed evidence.
--   No metadata schema change is required; this note documents the consumption.
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Fused Kalman drift matrix — catalogue-registered but DELIBERATELY UNUSED.
--
-- These kalman_pt mutable_predictors keep their catalogue rows and stay in
-- mv_pymc_kalman_pt, but are barred from the model's drift DESIGN MATRIX by
-- KALMAN_DRIFT_EXCLUDED_FEATURES in
-- probabilistic_ml_model/pymc_models/KalmanFilterModel.py:
--
--     feat_pt_median_drift / feat_pt_high_drift / feat_pt_low_drift
--         Siblings of feat_pt_drift: the same pml.target_drift() over the
--         median / high / low target trails. A consensus revision moves the
--         whole target band together, so r = 0.81-0.89 among them and
--         VIF = 162 / 77 / 25 / 6.6. feat_pt_drift (the consensus mean) is
--         retained as the representative.
--
--     feat_analyst_bullish_pct / feat_analyst_bearish_pct /
--     feat_analyst_neutral_pct / feat_analyst_conviction
--         All functions of the same six num_*_ratings buckets, with conviction
--         literally |bullish - bearish| and the three pct legs summing to ~1
--         (hence exactly collinear after z-scoring). r(bullish, conviction) =
--         0.926, r(bullish, rating) = 0.909. feat_analyst_rating is retained:
--         it scored higher against the response than the bullish+bearish pair
--         (R^2 0.6499 vs 0.6463) using one fewer column, at 99.76 % coverage.
--
-- Effect: the drift design went from 21 columns / condition number 1 580 /
-- max VIF 162 to 15 / 23 / 3.8, for a 0.6 % relative loss in R^2. That was the
-- last failing gate on the 2026-08-10 full-scale validation, where beta sat at
-- R-hat 1.026 / bulk-ESS 140 with zero divergences.
--
-- DO NOT implement this by flipping pymc_role to 'excluded' here.
-- vw_pymc_feature_catalogue filters 'excluded' rows out, while
-- mv_pymc_kalman_pt still emits the columns — so
-- vw_pymc_catalogue_coverage_check would report MISSING_FROM_CATALOGUE and
-- pml.assert_pymc_catalogue_coverage() would RAISE. The analyst family is also
-- shared with the price_target model (see the pml_df_feature_alias inserts in
-- pml_df_metadata_populate.sql), which must keep using it. Membership of one
-- model's design matrix is a Python-side concern, not a property of the column.
-- ---------------------------------------------------------------------------