create view information_schema.routine_table_usage
            (specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name,
             table_catalog, table_schema, table_name)
as
SELECT DISTINCT current_database()::sql_identifier              AS specific_catalog,
                np.nspname::sql_identifier                      AS specific_schema,
                nameconcatoid(p.proname, p.oid)::sql_identifier AS specific_name,
                current_database()::sql_identifier              AS routine_catalog,
                np.nspname::sql_identifier                      AS routine_schema,
                p.proname::sql_identifier                       AS routine_name,
                current_database()::sql_identifier              AS table_catalog,
                nt.nspname::sql_identifier                      AS table_schema,
                t.relname::sql_identifier                       AS table_name
FROM pg_namespace np,
     pg_proc p,
     pg_depend d,
     pg_class t,
     pg_namespace nt
WHERE np.oid = p.pronamespace
  AND p.oid = d.objid
  AND d.classid = 'pg_proc'::regclass::oid
  AND d.refobjid = t.oid
  AND d.refclassid = 'pg_class'::regclass::oid
  AND t.relnamespace = nt.oid
  AND (t.relkind = ANY (ARRAY ['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
  AND pg_has_role(t.relowner, 'USAGE'::text);

alter table information_schema.routine_table_usage
    owner to postgres;

grant select on information_schema.routine_table_usage to public;

