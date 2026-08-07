CREATE VIEW information_schema.column_udt_usage
			(udt_catalog, udt_schema, udt_name, table_catalog, table_schema, table_name, column_name) AS
SELECT current_database()::information_schema.sql_identifier                AS udt_catalog,
       COALESCE(nbt.nspname, nt.nspname)::information_schema.sql_identifier AS udt_schema,
       COALESCE(bt.typname, t.typname)::information_schema.sql_identifier   AS udt_name,
       current_database()::information_schema.sql_identifier                AS table_catalog,
       nc.nspname::information_schema.sql_identifier                        AS table_schema,
       c.relname::information_schema.sql_identifier                         AS table_name,
       a.attname::information_schema.sql_identifier                         AS column_name
FROM pg_attribute               a,
     pg_class                   c,
     pg_namespace               nc,
     pg_type                    t
	     JOIN      pg_namespace nt ON t.typnamespace = nt.oid
	     LEFT JOIN (pg_type bt JOIN pg_namespace nbt ON bt.typnamespace = nbt.oid)
	               ON t.typtype = 'd'::"char" AND t.typbasetype = bt.oid
WHERE a.attrelid = c.oid
  AND a.atttypid = t.oid
  AND nc.oid = c.relnamespace
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND (c.relkind = ANY (ARRAY ['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
  AND pg_has_role(COALESCE(bt.typowner, t.typowner), 'USAGE'::text);

ALTER TABLE information_schema.column_udt_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.column_udt_usage TO PUBLIC;