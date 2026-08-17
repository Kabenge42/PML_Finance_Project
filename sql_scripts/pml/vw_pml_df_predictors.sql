create view vw_pml_df_predictors(column_name, category, data_type, ordinal_position, description)
as
SELECT column_name,
       category,
       data_type,
       ordinal_position,
       description
FROM pml_df_metadata
WHERE pymc_role = 'mutable_predictor'::text
;

alter table vw_pml_df_predictors
	owner to postgres
;