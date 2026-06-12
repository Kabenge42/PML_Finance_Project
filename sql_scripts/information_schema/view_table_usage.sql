CREATE VIEW information_schema.view_table_usage
			(view_catalog, view_schema, view_name, table_catalog, table_schema, table_name) AS
-- missing source code
;

ALTER TABLE information_schema.view_table_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.view_table_usage TO PUBLIC;