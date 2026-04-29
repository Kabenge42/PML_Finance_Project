create view information_schema.routine_routine_usage
            (specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name) as
SELECT DISTINCT current_database()::sql_identifier                AS specific_catalog,
                np.nspname::sql_identifier                        AS specific_schema,
                nameconcatoid(p.proname, p.oid)::sql_identifier   AS specific_name,
                current_database()::sql_identifier                AS routine_catalog,
                np1.nspname::sql_identifier                       AS routine_schema,
                nameconcatoid(p1.proname, p1.oid)::sql_identifier AS routine_name
FROM pg_namespace np,
     pg_proc p,
     pg_depend d,
     pg_proc p1,
     pg_namespace np1
WHERE np.oid = p.pronamespace
  AND p.oid = d.objid
  AND d.classid = 'pg_proc'::regclass::oid
  AND d.refobjid = p1.oid
  AND d.refclassid = 'pg_proc'::regclass::oid
  AND p1.pronamespace = np1.oid
  AND (p.prokind = ANY (ARRAY ['f'::"char", 'p'::"char"]))
  AND (p1.prokind = ANY (ARRAY ['f'::"char", 'p'::"char"]))
  AND pg_has_role(p1.proowner, 'USAGE'::text);

alter table information_schema.routine_routine_usage
    owner to postgres;

grant select on information_schema.routine_routine_usage to public;

