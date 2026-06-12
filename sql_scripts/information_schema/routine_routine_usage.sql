CREATE VIEW information_schema.routine_routine_usage
			(specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name) AS
-- missing source code
;

ALTER TABLE information_schema.routine_routine_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.routine_routine_usage TO PUBLIC;