CREATE VIEW information_schema.domain_constraints
			(constraint_catalog, constraint_schema, constraint_name, domain_catalog, domain_schema, domain_name,
			 is_deferrable, initially_deferred)
AS
SELECT current_database()::information_schema.sql_identifier                                          AS constraint_catalog,
       rs.nspname::information_schema.sql_identifier                                                  AS constraint_schema,
       con.conname::information_schema.sql_identifier                                                 AS constraint_name,
       current_database()::information_schema.sql_identifier                                          AS domain_catalog,
       n.nspname::information_schema.sql_identifier                                                   AS domain_schema,
       t.typname::information_schema.sql_identifier                                                   AS domain_name,
       CASE WHEN con.condeferrable THEN 'YES'::text ELSE 'NO'::text END::information_schema.yes_or_no AS is_deferrable,
       CASE WHEN con.condeferred THEN 'YES'::text ELSE 'NO'::text END::information_schema.yes_or_no   AS initially_deferred
FROM pg_namespace  rs,
     pg_namespace  n,
     pg_constraint con,
     pg_type       t
WHERE rs.oid = con.connamespace
  AND n.oid = t.typnamespace
  AND t.oid = con.contypid
  AND (pg_has_role(t.typowner, 'USAGE'::text) OR has_type_privilege(t.oid, 'USAGE'::text));

ALTER TABLE information_schema.domain_constraints
	OWNER TO postgres;

GRANT SELECT ON information_schema.domain_constraints TO PUBLIC;