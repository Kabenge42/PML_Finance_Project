create view vw_pml_df_pymc_features
			(model_name, column_name, category, feature_role, pymc_role, data_type, ordinal_position, description) as
SELECT m.model_name,
       md.column_name,
       md.category,
       md.feature_role,
       md.pymc_role,
       md.data_type,
       md.ordinal_position,
       md.description
FROM pml_df_metadata                  md,
     LATERAL unnest(md.model_targets) m(model_name)
WHERE md.pymc_role = ANY (ARRAY ['coord'::text, 'observed'::text, 'mutable_predictor'::text, 'constant_data'::text])
;

alter table vw_pml_df_pymc_features
	owner to postgres
;