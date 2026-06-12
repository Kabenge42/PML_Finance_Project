CREATE VIEW information_schema.check_constraints(constraint_catalog, constraint_schema, constraint_name, check_clause) AS
-- missing source code
;

ALTER TABLE information_schema.check_constraints
	OWNER TO postgres;

GRANT SELECT ON information_schema.check_constraints TO PUBLIC;