CREATE VIEW information_schema.referential_constraints
			(constraint_catalog, constraint_schema, constraint_name, unique_constraint_catalog,
			 unique_constraint_schema, unique_constraint_name, match_option, update_rule, delete_rule)
AS
-- missing source code
;

ALTER TABLE information_schema.referential_constraints
	OWNER TO postgres;

GRANT SELECT ON information_schema.referential_constraints TO PUBLIC;