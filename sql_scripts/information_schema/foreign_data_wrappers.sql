CREATE VIEW information_schema.foreign_data_wrappers
			(foreign_data_wrapper_catalog, foreign_data_wrapper_name, authorization_identifier, library_name,
			 foreign_data_wrapper_language)
AS
-- missing source code
;

ALTER TABLE information_schema.foreign_data_wrappers
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_data_wrappers TO PUBLIC;