CREATE VIEW information_schema.view_column_usage
			(view_catalog, view_schema, view_name, table_catalog, table_schema, table_name, column_name) AS
-- missing source code
;

ALTER TABLE information_schema.view_column_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.view_column_usage TO PUBLIC;