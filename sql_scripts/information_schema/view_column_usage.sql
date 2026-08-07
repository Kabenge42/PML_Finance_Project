CREATE VIEW information_schema.view_column_usage
			(view_catalog, view_schema, view_name, table_catalog, table_schema, table_name, column_name) AS
SELECT DISTINCT
       current_database()::information_schema.sql_identifier AS view_catalog,
       nv.nspname::information_schema.sql_identifier         AS view_schema,
       v.relname::information_schema.sql_identifier          AS view_name,
       current_database()::information_schema.sql_identifier AS table_catalog,
       nt.nspname::information_schema.sql_identifier         AS table_schema,
       t.relname::information_schema.sql_identifier          AS table_name,
       a.attname::information_schema.sql_identifier          AS column_name
FROM pg_namespace nv,
     pg_class     v,
     pg_depend    dv,
     pg_depend    dt,
     pg_class     t,
     pg_namespace nt,
     pg_attribute a
WHERE nv.oid = v.relnamespace
  AND v.relkind = 'v'::"char"
  AND v.oid = dv.refobjid
  AND dv.refclassid = 'pg_class'::regclass::oid
  AND dv.classid = 'pg_rewrite'::regclass::oid
  AND dv.deptype = 'i'::"char"
  AND dv.objid = dt.objid
  AND dv.refobjid <> dt.refobjid
  AND dt.classid = 'pg_rewrite'::regclass::oid
  AND dt.refclassid = 'pg_class'::regclass::oid
  AND dt.refobjid = t.oid
  AND t.relnamespace = nt.oid
  AND (t.relkind = ANY (ARRAY ['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
  AND t.oid = a.attrelid
  AND dt.refobjsubid = a.attnum
  AND pg_has_role(t.relowner, 'USAGE'::text);

ALTER TABLE information_schema.view_column_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.view_column_usage TO PUBLIC;