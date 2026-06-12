CREATE VIEW information_schema.domain_constraints
			(constraint_catalog, constraint_schema, constraint_name, domain_catalog, domain_schema, domain_name,
			 is_deferrable, initially_deferred)
AS
-- missing source code
;

ALTER TABLE information_schema.domain_constraints
	OWNER TO postgres;

GRANT SELECT ON information_schema.domain_constraints TO PUBLIC;