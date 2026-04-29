create view information_schema.view_table_usage
            (view_catalog, view_schema, view_name, table_catalog, table_schema, table_name) as
SELECT DISTINCT current_database()::sql_identifier AS view_catalog,
                nv.nspname::sql_identifier         AS view_schema,
                v.relname::sql_identifier          AS view_name,
                current_database()::sql_identifier AS table_catalog,
                nt.nspname::sql_identifier         AS table_schema,
                t.relname::sql_identifier          AS table_name
FROM pg_namespace nv,
     pg_class v,
     pg_depend dv,
     pg_depend dt,
     pg_class t,
     pg_namespace nt
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
  AND pg_has_role(t.relowner, 'USAGE'::text);

alter table information_schema.view_table_usage
    owner to postgres;

grant select on information_schema.view_table_usage to public;

