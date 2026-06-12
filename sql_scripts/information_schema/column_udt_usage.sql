CREATE VIEW information_schema.column_udt_usage
			(udt_catalog, udt_schema, udt_name, table_catalog, table_schema, table_name, column_name) AS
-- missing source code
;

ALTER TABLE information_schema.column_udt_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.column_udt_usage TO PUBLIC;