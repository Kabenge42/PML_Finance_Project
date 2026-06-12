CREATE VIEW information_schema.column_options
			(table_catalog, table_schema, table_name, column_name, option_name, option_value) AS
-- missing source code
;

ALTER TABLE information_schema.column_options
	OWNER TO postgres;

GRANT SELECT ON information_schema.column_options TO PUBLIC;