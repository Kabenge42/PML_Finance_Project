create view pml.vw_pml_df_coords(column_name, category, data_type, ordinal_position, description)
as
SELECT column_name,
       category,
       data_type,
       ordinal_position,
       description
FROM pml_df_metadata
WHERE pymc_role = 'coord'::text
;

alter table pml.vw_pml_df_coords
	owner to postgres
;