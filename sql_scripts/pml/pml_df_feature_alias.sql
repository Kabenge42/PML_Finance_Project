create table pml.pml_df_feature_alias
(
	column_name   text not null,
	model_target  text not null,
	feature_alias text not null,
	pymc_role     text,
	primary key (column_name, model_target)
)
;

comment on table pml.pml_df_feature_alias is 'Per-model alias overrides for source columns in pml.pml_df_metadata. Surfaces through pml.vw_pymc_feature_catalogue.feature_alias and the notebook''s MODEL_FEATURE_CONTAINERS registry. Multi-source engineered features (e.g. the normalized analyst-sentiment feat_analyst_bullish_pct / feat_analyst_bearish_pct / feat_analyst_neutral_pct / feat_analyst_conviction columns in pml.mv_pymc_price_target, each derived from all six num_*_ratings buckets) record provenance against a single representative source column per (column_name, model_target) key.'
;

comment on column pml.pml_df_feature_alias.pymc_role is 'Per-model pymc_role override. NULL inherits pml_df_metadata.pymc_role. vw_pymc_feature_catalogue resolves the effective role via COALESCE(fa.pymc_role, md.pymc_role), so a column can be globally observed/derived_input yet act as a mutable_predictor for a specific model.'
;

alter table pml.pml_df_feature_alias
	owner to postgres
;

create index idx_pml_df_feature_alias_model
	on pml.pml_df_feature_alias (model_target)
;