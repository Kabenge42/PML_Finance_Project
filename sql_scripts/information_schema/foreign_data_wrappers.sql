CREATE VIEW information_schema.foreign_data_wrappers
			(foreign_data_wrapper_catalog, foreign_data_wrapper_name, authorization_identifier, library_name,
			 foreign_data_wrapper_language)
AS
SELECT foreign_data_wrapper_catalog,
       foreign_data_wrapper_name,
       authorization_identifier,
       NULL::character varying::information_schema.character_data AS library_name,
       foreign_data_wrapper_language
FROM information_schema._pg_foreign_data_wrappers w;

ALTER TABLE information_schema.foreign_data_wrappers
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_data_wrappers TO PUBLIC;