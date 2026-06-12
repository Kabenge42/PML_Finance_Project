CREATE TABLE pml_df_feature_alias
(
	column_name text NOT NULL
		CONSTRAINT fk_column_name REFERENCES pml_df_metadata ON DELETE CASCADE,
	model_target  text NOT NULL,
	feature_alias text NOT NULL,
	pymc_role   text,
	PRIMARY KEY (column_name, model_target)
);

COMMENT ON TABLE pml_df_feature_alias IS 'Per-model alias overrides for source columns in pml.pml_df_metadata. Surfaces through pml.vw_pymc_feature_catalogue.feature_alias and the notebook''s MODEL_FEATURE_CONTAINERS registry. Multi-source engineered features (e.g. the normalized analyst-sentiment feat_analyst_bullish_pct / feat_analyst_bearish_pct / feat_analyst_neutral_pct / feat_analyst_conviction columns in pml.mv_pymc_price_target, each derived from all six num_*_ratings buckets) record provenance against a single representative source column per (column_name, model_target) key.';

COMMENT ON COLUMN pml_df_feature_alias.pymc_role IS 'Per-model pymc_role override. NULL inherits pml_df_metadata.pymc_role. vw_pymc_feature_catalogue resolves the effective role via COALESCE(fa.pymc_role, md.pymc_role), so a column can be globally observed/derived_input yet act as a mutable_predictor for a specific model.';

ALTER TABLE pml_df_feature_alias
	OWNER TO postgres;

CREATE INDEX idx_pml_df_feature_alias_model ON pml_df_feature_alias (model_target);