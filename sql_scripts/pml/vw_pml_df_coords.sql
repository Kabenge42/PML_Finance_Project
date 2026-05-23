CREATE VIEW vw_pml_df_coords(column_name, category, data_type, ordinal_position, description) AS
SELECT column_name, category, data_type, ordinal_position, description
FROM pml_df_metadata
WHERE pymc_role = 'coord'::text;

ALTER TABLE vw_pml_df_coords
	OWNER TO postgres;