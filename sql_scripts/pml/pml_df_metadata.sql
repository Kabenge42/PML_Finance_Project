CREATE TABLE pml_df_metadata
(
	column_name      text                               NOT NULL PRIMARY KEY,
	category         text      DEFAULT 'n/a'::text      NOT NULL,
	feature_role     text                               NOT NULL
		CONSTRAINT ck_pml_df_metadata_feature_role CHECK (feature_role = ANY
		                                                  (ARRAY ['id'::text, 'categorical'::text, 'date'::text, 'target'::text, 'predictor'::text, 'count'::text, 'score'::text, 'historical'::text, 'surprise'::text, 'revision'::text, 'metadata'::text])),
	feature_alias    text,
	ordinal_position integer,
	description      text,
	data_type        text,
	pymc_role        text
		CONSTRAINT ck_pml_df_metadata_pymc_role CHECK ((pymc_role IS NULL) OR (pymc_role = ANY
		                                                                       (ARRAY ['coord'::text, 'index'::text, 'observed'::text, 'mutable_predictor'::text, 'constant_data'::text, 'derived_input'::text, 'excluded'::text]))),
	model_targets    text[]    DEFAULT ARRAY []::text[] NOT NULL
		CONSTRAINT ck_pml_df_metadata_model_targets CHECK (model_targets <@
		                                                   ARRAY ['earnings_beat'::text, 'price_target'::text, 'kalman_pt'::text, 'dcf_pt'::text, 'dividend_safety'::text, 'credit_risk'::text, 'accounting_anomaly'::text]),
	updated_at       timestamp DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE pml_df_metadata IS 'Metadata for pml.pml_df. (category, feature_role) drive domain/data-centric SQL filters; (pymc_role, model_targets) drive PyMC pm.Data container assignment and per-model feature selection. pymc_role vocabulary: coord | index | observed | mutable_predictor | constant_data | derived_input | excluded. model_targets is a TEXT[] keyed by MODEL_FEATURE_CONTAINERS (earnings_beat, price_target, kalman_pt, dcf_pt, dividend_safety, credit_risk, accounting_anomaly).';

COMMENT ON COLUMN pml_df_metadata.feature_role IS 'Coarse ML role used for SQL filtering during feature engineering. Vocabulary (CHECK-enforced): id | categorical | date | target | predictor | count | score | historical | surprise | revision | metadata. Also the deterministic seed for pymc_role -- see the CASE in pml_df_metadata_populate.sql step 2.';

COMMENT ON COLUMN pml_df_metadata.feature_alias IS 'Default (model-agnostic) alias used inside pml.mv_pymc_* materialized views. For model-specific overrides see pml.pml_df_feature_alias.';

COMMENT ON COLUMN pml_df_metadata.pymc_role IS 'PyMC pm.Data container kind for this column. Aligns with arviz.InferenceData groups: coord/index -> idata.constant_data + posterior.coords; observed -> idata.observed_data; mutable_predictor/constant_data -> idata.constant_data (mutable_predictor supports pm.set_data for OOS); derived_input -> must be transformed before pm.Data; excluded -> never wrapped.';

COMMENT ON COLUMN pml_df_metadata.model_targets IS 'Array of PyMC model names from probabilistic_ml_model.pymc_models that consume this column. Mirrors MODEL_FEATURE_CONTAINERS keys in pymc_expected_returns_model.ipynb.';

ALTER TABLE pml_df_metadata
	OWNER TO postgres;

CREATE INDEX idx_pml_df_metadata_feature_role ON pml_df_metadata (feature_role);

CREATE INDEX idx_pml_df_metadata_category ON pml_df_metadata (category);

CREATE INDEX idx_pml_df_metadata_pymc_role ON pml_df_metadata (pymc_role);

CREATE INDEX idx_pml_df_metadata_feature_alias ON pml_df_metadata (feature_alias);

CREATE INDEX idx_pml_df_metadata_model_targets ON pml_df_metadata USING gin (model_targets);