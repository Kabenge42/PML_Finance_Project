create view vw_pml_df_observed(column_name, category, data_type, ordinal_position, description) as
SELECT column_name,
       category,
       data_type,
       ordinal_position,
       description
FROM pml_df_metadata
WHERE pymc_role = 'observed'::text
;

alter table vw_pml_df_observed
	owner to postgres
;