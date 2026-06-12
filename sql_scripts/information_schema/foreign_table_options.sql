CREATE VIEW information_schema.foreign_table_options
			(foreign_table_catalog, foreign_table_schema, foreign_table_name, option_name, option_value) AS
-- missing source code
;

ALTER TABLE information_schema.foreign_table_options
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_table_options TO PUBLIC;