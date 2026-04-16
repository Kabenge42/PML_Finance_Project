create view information_schema.routine_privileges
            (grantor, grantee, specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema,
             routine_name, privilege_type, is_grantable)
as
SELECT u_grantor.rolname::sql_identifier               AS grantor,
       grantee.rolname::sql_identifier                 AS grantee,
       current_database()::sql_identifier              AS specific_catalog,
       n.nspname::sql_identifier                       AS specific_schema,
       nameconcatoid(p.proname, p.oid)::sql_identifier AS specific_name,
       current_database()::sql_identifier              AS routine_catalog,
       n.nspname::sql_identifier                       AS routine_schema,
       p.proname::sql_identifier                       AS routine_name,
       'EXECUTE'::character varying::character_data    AS privilege_type,
       CASE
           WHEN pg_has_role(grantee.oid, p.proowner, 'USAGE'::text) OR p.grantable THEN 'YES'::text
           ELSE 'NO'::text
           END::yes_or_no                              AS is_grantable
FROM (SELECT pg_proc.oid,
             pg_proc.proname,
             pg_proc.proowner,
             pg_proc.pronamespace,
             (aclexplode(COALESCE(pg_proc.proacl, acldefault('f'::"char", pg_proc.proowner)))).grantor AS grantor,
             (aclexplode(COALESCE(pg_proc.proacl,
                                  acldefault('f'::"char", pg_proc.proowner)))).grantee                 AS grantee,
             (aclexplode(COALESCE(pg_proc.proacl,
                                  acldefault('f'::"char", pg_proc.proowner)))).privilege_type          AS privilege_type,
             (aclexplode(COALESCE(pg_proc.proacl,
                                  acldefault('f'::"char", pg_proc.proowner)))).is_grantable            AS is_grantable
      FROM pg_proc) p(oid, proname, proowner, pronamespace, grantor, grantee, prtype, grantable),
     pg_namespace n,
     pg_authid u_grantor,
     (SELECT pg_authid.oid,
             pg_authid.rolname
      FROM pg_authid
      UNION ALL
      SELECT 0::oid AS oid,
             'PUBLIC'::name) grantee(oid, rolname)
WHERE p.pronamespace = n.oid
  AND grantee.oid = p.grantee
  AND u_grantor.oid = p.grantor
  AND p.prtype = 'EXECUTE'::text
  AND (pg_has_role(u_grantor.oid, 'USAGE'::text) OR pg_has_role(grantee.oid, 'USAGE'::text) OR
       grantee.rolname = 'PUBLIC'::name);

alter table information_schema.routine_privileges
    owner to postgres;

grant select on information_schema.routine_privileges to public;

