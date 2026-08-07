CREATE VIEW information_schema.constraint_table_usage
			(table_catalog, table_schema, table_name, constraint_catalog, constraint_schema, constraint_name) AS
SELECT current_database()::information_schema.sql_identifier AS table_catalog,
       nr.nspname::information_schema.sql_identifier         AS table_schema,
       r.relname::information_schema.sql_identifier          AS table_name,
       current_database()::information_schema.sql_identifier AS constraint_catalog,
       nc.nspname::information_schema.sql_identifier         AS constraint_schema,
       c.conname::information_schema.sql_identifier          AS constraint_name
FROM pg_constraint c,
     pg_namespace  nc,
     pg_class      r,
     pg_namespace  nr
WHERE c.connamespace = nc.oid
  AND r.relnamespace = nr.oid
  AND (c.contype = 'f'::"char" AND c.confrelid = r.oid OR
       (c.contype = ANY (ARRAY ['p'::"char", 'u'::"char"])) AND c.conrelid = r.oid)
  AND (r.relkind = ANY (ARRAY ['r'::"char", 'p'::"char"]))
  AND pg_has_role(r.relowner, 'USAGE'::text);

ALTER TABLE information_schema.constraint_table_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.constraint_table_usage TO PUBLIC;