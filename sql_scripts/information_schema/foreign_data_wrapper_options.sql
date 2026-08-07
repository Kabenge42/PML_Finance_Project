CREATE VIEW information_schema.foreign_data_wrapper_options
			(foreign_data_wrapper_catalog, foreign_data_wrapper_name, option_name, option_value) AS
SELECT foreign_data_wrapper_catalog,
       foreign_data_wrapper_name,
       (pg_options_to_table(fdwoptions)).option_name::information_schema.sql_identifier  AS option_name,
       (pg_options_to_table(fdwoptions)).option_value::information_schema.character_data AS option_value
FROM information_schema._pg_foreign_data_wrappers w;

ALTER TABLE information_schema.foreign_data_wrapper_options
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_data_wrapper_options TO PUBLIC;