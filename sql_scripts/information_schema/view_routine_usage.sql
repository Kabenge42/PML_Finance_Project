CREATE VIEW information_schema.view_routine_usage
			(table_catalog, table_schema, table_name, specific_catalog, specific_schema, specific_name) AS
-- missing source code
;

ALTER TABLE information_schema.view_routine_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.view_routine_usage TO PUBLIC;