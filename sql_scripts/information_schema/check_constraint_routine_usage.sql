create view information_schema.check_constraint_routine_usage
            (constraint_catalog, constraint_schema, constraint_name, specific_catalog, specific_schema,
             specific_name)
as
SELECT DISTINCT current_database()::sql_identifier              AS constraint_catalog,
                nc.nspname::sql_identifier                      AS constraint_schema,
                c.conname::sql_identifier                       AS constraint_name,
                current_database()::sql_identifier              AS specific_catalog,
                np.nspname::sql_identifier                      AS specific_schema,
                nameconcatoid(p.proname, p.oid)::sql_identifier AS specific_name
FROM pg_namespace nc,
     pg_constraint c,
     pg_depend d,
     pg_proc p,
     pg_namespace np
WHERE nc.oid = c.connamespace
  AND c.contype = 'c'::"char"
  AND c.oid = d.objid
  AND d.classid = 'pg_constraint'::regclass::oid
  AND d.refobjid = p.oid
  AND d.refclassid = 'pg_proc'::regclass::oid
  AND p.pronamespace = np.oid
  AND pg_has_role(p.proowner, 'USAGE'::text);

alter table information_schema.check_constraint_routine_usage
    owner to postgres;

grant select on information_schema.check_constraint_routine_usage to public;

