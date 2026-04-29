create view information_schema.routine_sequence_usage
            (specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name,
             sequence_catalog, sequence_schema, sequence_name)
as
SELECT DISTINCT current_database()::sql_identifier              AS specific_catalog,
                np.nspname::sql_identifier                      AS specific_schema,
                nameconcatoid(p.proname, p.oid)::sql_identifier AS specific_name,
                current_database()::sql_identifier              AS routine_catalog,
                np.nspname::sql_identifier                      AS routine_schema,
                p.proname::sql_identifier                       AS routine_name,
                current_database()::sql_identifier              AS sequence_catalog,
                ns.nspname::sql_identifier                      AS sequence_schema,
                s.relname::sql_identifier                       AS sequence_name
FROM pg_namespace np,
     pg_proc p,
     pg_depend d,
     pg_class s,
     pg_namespace ns
WHERE np.oid = p.pronamespace
  AND p.oid = d.objid
  AND d.classid = 'pg_proc'::regclass::oid
  AND d.refobjid = s.oid
  AND d.refclassid = 'pg_class'::regclass::oid
  AND s.relnamespace = ns.oid
  AND s.relkind = 'S'::"char"
  AND pg_has_role(s.relowner, 'USAGE'::text);

alter table information_schema.routine_sequence_usage
    owner to postgres;

grant select on information_schema.routine_sequence_usage to public;

