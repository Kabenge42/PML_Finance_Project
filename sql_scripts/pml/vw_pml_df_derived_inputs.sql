create view pml.vw_pml_df_derived_inputs(column_name, category, data_type, ordinal_position, description)
as
SELECT column_name,
       category,
       data_type,
       ordinal_position,
       description
FROM pml_df_metadata
WHERE pymc_role = 'derived_input'::text
;

alter table pml.vw_pml_df_derived_inputs
	owner to postgres
;