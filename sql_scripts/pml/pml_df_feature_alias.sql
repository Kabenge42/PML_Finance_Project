CREATE TABLE pml_df_feature_alias
(
	column_name text NOT NULL
		CONSTRAINT fk_column_name REFERENCES pml_df_metadata ON DELETE CASCADE,
	model_target  text NOT NULL,
	feature_alias text NOT NULL,
	PRIMARY KEY (column_name, model_target)
);

COMMENT ON TABLE pml_df_feature_alias IS 'Per-model alias overrides for source columns in pml.pml_df_metadata. Surfaces through pml.vw_pymc_feature_catalogue.feature_alias and the notebook''s MODEL_FEATURE_CONTAINERS registry.';

ALTER TABLE pml_df_feature_alias
	OWNER TO postgres;

CREATE INDEX idx_pml_df_feature_alias_model ON pml_df_feature_alias (model_target);