CREATE VIEW information_schema.column_domain_usage
			(domain_catalog, domain_schema, domain_name, table_catalog, table_schema, table_name, column_name) AS
SELECT current_database()::information_schema.sql_identifier AS domain_catalog,
       nt.nspname::information_schema.sql_identifier         AS domain_schema,
       t.typname::information_schema.sql_identifier          AS domain_name,
       current_database()::information_schema.sql_identifier AS table_catalog,
       nc.nspname::information_schema.sql_identifier         AS table_schema,
       c.relname::information_schema.sql_identifier          AS table_name,
       a.attname::information_schema.sql_identifier          AS column_name
FROM pg_type      t,
     pg_namespace nt,
     pg_class     c,
     pg_namespace nc,
     pg_attribute a
WHERE t.typnamespace = nt.oid
  AND c.relnamespace = nc.oid
  AND a.attrelid = c.oid
  AND a.atttypid = t.oid
  AND t.typtype = 'd'::"char"
  AND (c.relkind = ANY (ARRAY ['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND pg_has_role(t.typowner, 'USAGE'::text);

ALTER TABLE information_schema.column_domain_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.column_domain_usage TO PUBLIC;