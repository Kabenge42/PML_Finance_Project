CREATE VIEW pml.vw_pml_df_observed(column_name, category, data_type, ordinal_position, description) AS
SELECT column_name, category, data_type, ordinal_position, description
FROM pml_df_metadata
WHERE pymc_role = 'observed'::text;

ALTER TABLE pml.vw_pml_df_observed
	OWNER TO postgres;