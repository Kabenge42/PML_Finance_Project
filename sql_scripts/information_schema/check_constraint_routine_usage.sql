CREATE VIEW information_schema.check_constraint_routine_usage
			(constraint_catalog, constraint_schema, constraint_name, specific_catalog, specific_schema,
			 specific_name) AS
-- missing source code
;

ALTER TABLE information_schema.check_constraint_routine_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.check_constraint_routine_usage TO PUBLIC;