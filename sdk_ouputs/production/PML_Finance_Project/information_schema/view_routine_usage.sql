create view information_schema.view_routine_usage
            (table_catalog, table_schema, table_name, specific_catalog, specific_schema, specific_name) as
SELECT DISTINCT current_database()::sql_identifier              AS table_catalog,
                nv.nspname::sql_identifier                      AS table_schema,
                v.relname::sql_identifier                       AS table_name,
                current_database()::sql_identifier              AS specific_catalog,
                np.nspname::sql_identifier                      AS specific_schema,
                nameconcatoid(p.proname, p.oid)::sql_identifier AS specific_name
FROM pg_namespace nv,
     pg_class v,
     pg_depend dv,
     pg_depend dp,
     pg_proc p,
     pg_namespace np
WHERE nv.oid = v.relnamespace
  AND v.relkind = 'v'::"char"
  AND v.oid = dv.refobjid
  AND dv.refclassid = 'pg_class'::regclass::oid
  AND dv.classid = 'pg_rewrite'::regclass::oid
  AND dv.deptype = 'i'::"char"
  AND dv.objid = dp.objid
  AND dp.classid = 'pg_rewrite'::regclass::oid
  AND dp.refclassid = 'pg_proc'::regclass::oid
  AND dp.refobjid = p.oid
  AND p.pronamespace = np.oid
  AND pg_has_role(p.proowner, 'USAGE'::text);

alter table information_schema.view_routine_usage
    owner to postgres;

grant select on information_schema.view_routine_usage to public;

