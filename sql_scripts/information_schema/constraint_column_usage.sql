CREATE VIEW information_schema.constraint_column_usage
			(table_catalog, table_schema, table_name, column_name, constraint_catalog, constraint_schema,
			 constraint_name) AS
-- missing source code
;

ALTER TABLE information_schema.constraint_column_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.constraint_column_usage TO PUBLIC;