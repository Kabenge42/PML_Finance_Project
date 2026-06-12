CREATE VIEW information_schema.column_column_usage
			(table_catalog, table_schema, table_name, column_name, dependent_column) AS
-- missing source code
;

ALTER TABLE information_schema.column_column_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.column_column_usage TO PUBLIC;