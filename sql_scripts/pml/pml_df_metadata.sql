create table pml.pml_df_metadata
(
	column_name      text                               not null
		primary key,
	category         text      default 'n/a'::text      not null,
	feature_role     text                               not null
		constraint ck_pml_df_metadata_feature_role
			check (feature_role = ANY
			       (ARRAY ['id'::text, 'categorical'::text, 'date'::text, 'target'::text, 'predictor'::text, 'count'::text, 'score'::text, 'historical'::text, 'surprise'::text, 'revision'::text, 'metadata'::text])),
	feature_alias    text,
	ordinal_position integer,
	description      text,
	data_type        text,
	pymc_role        text
		constraint ck_pml_df_metadata_pymc_role
			check ((pymc_role IS NULL) OR (pymc_role = ANY
			                               (ARRAY ['coord'::text, 'index'::text, 'observed'::text, 'mutable_predictor'::text, 'constant_data'::text, 'derived_input'::text, 'excluded'::text]))),
	model_targets    text[]    default ARRAY []::text[] not null
		constraint ck_pml_df_metadata_model_targets
			check (model_targets <@
			       ARRAY ['earnings_beat'::text, 'price_target'::text, 'kalman_pt'::text, 'kalman_pt_v2'::text, 'dcf_pt'::text, 'dividend_safety'::text, 'credit_risk'::text, 'accounting_anomaly'::text]),
	updated_at       timestamp default CURRENT_TIMESTAMP
)
;

comment on table pml.pml_df_metadata is 'Metadata for pml.pml_df. (category, feature_role) drive domain/data-centric SQL filters; (pymc_role, model_targets) drive PyMC pm.Data container assignment and per-model feature selection. pymc_role vocabulary: coord | index | observed | mutable_predictor | constant_data | derived_input | excluded. model_targets is a TEXT[] keyed by MODEL_FEATURE_CONTAINERS (earnings_beat, price_target, kalman_pt, kalman_pt_v2, dcf_pt, dividend_safety, credit_risk, accounting_anomaly).'
;

comment on column pml.pml_df_metadata.feature_role is 'Coarse ML role used for SQL filtering during feature engineering. Vocabulary (CHECK-enforced): id | categorical | date | target | predictor | count | score | historical | surprise | revision | metadata. Also the deterministic seed for pymc_role -- see the CASE in pml_df_metadata_populate.sql step 2.'
;

comment on column pml.pml_df_metadata.feature_alias is 'Default (model-agnostic) alias used inside pml.mv_pymc_* materialized views. For model-specific overrides see pml.pml_df_feature_alias.'
;

comment on column pml.pml_df_metadata.pymc_role is 'PyMC pm.Data container kind for this column. Aligns with arviz.InferenceData groups: coord/index -> idata.constant_data + posterior.coords; observed -> idata.observed_data; mutable_predictor/constant_data -> idata.constant_data (mutable_predictor supports pm.set_data for OOS); derived_input -> must be transformed before pm.Data; excluded -> never wrapped.'
;

comment on column pml.pml_df_metadata.model_targets is 'Array of PyMC model names from probabilistic_ml_model.pymc_models that consume this column. Mirrors MODEL_FEATURE_CONTAINERS keys in pymc_expected_returns_model.ipynb.'
;

alter table pml.pml_df_metadata
	owner to postgres
;

create index idx_pml_df_metadata_feature_role
	on pml.pml_df_metadata (feature_role)
;

create index idx_pml_df_metadata_category
	on pml.pml_df_metadata (category)
;

create index idx_pml_df_metadata_pymc_role
	on pml.pml_df_metadata (pymc_role)
;

create index idx_pml_df_metadata_feature_alias
	on pml.pml_df_metadata (feature_alias)
;

create index idx_pml_df_metadata_model_targets
	on pml.pml_df_metadata using gin (model_targets)
;