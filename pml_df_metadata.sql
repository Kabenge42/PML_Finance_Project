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
--         (earnings_beat, price_target, kalman_pt, dcf_pt,
--          dividend_safety, credit_risk, accounting_anomaly).
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
	updated_at       TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
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
	CONSTRAINT fk_column_name FOREIGN KEY (column_name) REFERENCES pml.pml_df_metadata (column_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pml_df_feature_alias_model ON pml.pml_df_feature_alias (model_target);

-- Recreate GIN index with metadata defaults
CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_feature_role ON pml.pml_df_metadata (feature_role);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_category ON pml.pml_df_metadata (category);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_pymc_role ON pml.pml_df_metadata (pymc_role);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_feature_alias ON pml.pml_df_metadata (feature_alias);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_model_targets ON pml.pml_df_metadata USING gin (model_targets);

COMMENT ON TABLE pml.pml_df_metadata IS 'Metadata for pml.pml_df. (category, feature_role) drive domain/data-centric SQL filters; (pymc_role, model_targets) drive PyMC pm.Data container assignment and per-model feature selection. pymc_role vocabulary: coord | index | observed | mutable_predictor | constant_data | derived_input | excluded. model_targets is a TEXT[] keyed by MODEL_FEATURE_CONTAINERS (earnings_beat, price_target, kalman_pt, dcf_pt, dividend_safety, credit_risk, accounting_anomaly).';;
-- Create a metadata table documenting available pml_df schema columns
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
	updated_at       TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
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
	PRIMARY KEY (column_name, model_target)
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

COMMENT ON TABLE pml.pml_df_metadata IS 'Metadata for pml.pml_df. (category, feature_role) drive domain/data-centric SQL filters; (pymc_role, model_targets) drive PyMC pm.Data container assignment and per-model feature selection. pymc_role vocabulary: coord | index | observed | mutable_predictor | constant_data | derived_input | excluded. model_targets is a TEXT[] keyed by MODEL_FEATURE_CONTAINERS (earnings_beat, price_target, kalman_pt, dcf_pt, dividend_safety, credit_risk, accounting_anomaly).';

COMMENT ON COLUMN pml.pml_df_metadata.pymc_role IS 'PyMC pm.Data container kind for this column. Aligns with arviz.InferenceData groups: coord/index -> idata.constant_data + posterior.coords; observed -> idata.observed_data; mutable_predictor/constant_data -> idata.constant_data (mutable_predictor supports pm.set_data for OOS); derived_input -> must be transformed before pm.Data; excluded -> never wrapped.';

COMMENT ON COLUMN pml.pml_df_metadata.model_targets IS 'Array of PyMC model names from probabilistic_ml_model.pymc_models that consume this column. Mirrors MODEL_FEATURE_CONTAINERS keys in pymc_expected_returns_model.ipynb.';

COMMENT ON COLUMN pml.pml_df_metadata.feature_alias IS 'Default (model-agnostic) alias used inside pml.mv_pymc_* materialized views. For model-specific overrides see pml.pml_df_feature_alias.';

COMMENT ON COLUMN pml.pml_df_feature_alias.pymc_role IS 'Per-model pymc_role override. NULL inherits pml_df_metadata.pymc_role. vw_pymc_feature_catalogue resolves the effective role via COALESCE(fa.pymc_role, md.pymc_role), so a column can be globally observed/derived_input yet act as a mutable_predictor for a specific model.';

COMMENT ON TABLE pml.pml_df_feature_alias IS 'Per-model alias overrides for source columns in pml.pml_df_metadata. Surfaces through pml.vw_pymc_feature_catalogue.feature_alias and the notebook''s MODEL_FEATURE_CONTAINERS registry. Multi-source engineered features (e.g. the normalized analyst-sentiment feat_analyst_bullish_pct / feat_analyst_bearish_pct / feat_analyst_neutral_pct / feat_analyst_conviction columns in pml.mv_pymc_price_target, each derived from all six num_*_ratings buckets) record provenance against a single representative source column per (column_name, model_target) key.';