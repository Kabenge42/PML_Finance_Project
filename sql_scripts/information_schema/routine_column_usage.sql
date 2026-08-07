CREATE VIEW information_schema.routine_column_usage
			(specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name,
			 table_catalog, table_schema, table_name, column_name)
AS
SELECT DISTINCT
       current_database()::information_schema.sql_identifier              AS specific_catalog,
       np.nspname::information_schema.sql_identifier                      AS specific_schema,
       nameconcatoid(p.proname, p.oid)::information_schema.sql_identifier AS specific_name,
       current_database()::information_schema.sql_identifier              AS routine_catalog,
       np.nspname::information_schema.sql_identifier                      AS routine_schema,
       p.proname::information_schema.sql_identifier                       AS routine_name,
       current_database()::information_schema.sql_identifier              AS table_catalog,
       nt.nspname::information_schema.sql_identifier                      AS table_schema,
       t.relname::information_schema.sql_identifier                       AS table_name,
       a.attname::information_schema.sql_identifier                       AS column_name
FROM pg_namespace np,
     pg_proc      p,
     pg_depend    d,
     pg_class     t,
     pg_namespace nt,
     pg_attribute a
WHERE np.oid = p.pronamespace
  AND p.oid = d.objid
  AND d.classid = 'pg_proc'::regclass::oid
  AND d.refobjid = t.oid
  AND d.refclassid = 'pg_class'::regclass::oid
  AND t.relnamespace = nt.oid
  AND (t.relkind = ANY (ARRAY ['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
  AND t.oid = a.attrelid
  AND d.refobjsubid = a.attnum
  AND pg_has_role(t.relowner, 'USAGE'::text);

ALTER TABLE information_schema.routine_column_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.routine_column_usage TO PUBLIC;