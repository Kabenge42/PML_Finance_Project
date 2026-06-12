CREATE VIEW information_schema.table_constraints
			(constraint_catalog, constraint_schema, constraint_name, table_catalog, table_schema, table_name,
			 constraint_type, is_deferrable, initially_deferred, enforced, nulls_distinct)
AS
-- missing source code
;

ALTER TABLE information_schema.table_constraints
	OWNER TO postgres;

GRANT SELECT ON information_schema.table_constraints TO PUBLIC;