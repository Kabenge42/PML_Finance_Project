CREATE VIEW information_schema.foreign_data_wrapper_options
			(foreign_data_wrapper_catalog, foreign_data_wrapper_name, option_name, option_value) AS
-- missing source code
;

ALTER TABLE information_schema.foreign_data_wrapper_options
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_data_wrapper_options TO PUBLIC;