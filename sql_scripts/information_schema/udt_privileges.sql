create view information_schema.udt_privileges
            (grantor, grantee, udt_catalog, udt_schema, udt_name, privilege_type, is_grantable) as
SELECT u_grantor.rolname::sql_identifier               AS grantor,
       grantee.rolname::sql_identifier                 AS grantee,
       current_database()::sql_identifier              AS udt_catalog,
       n.nspname::sql_identifier                       AS udt_schema,
       t.typname::sql_identifier                       AS udt_name,
       'TYPE USAGE'::character varying::character_data AS privilege_type,
       CASE
           WHEN pg_has_role(grantee.oid, t.typowner, 'USAGE'::text) OR t.grantable THEN 'YES'::text
           ELSE 'NO'::text
           END::yes_or_no                              AS is_grantable
FROM (SELECT pg_type.oid,
             pg_type.typname,
             pg_type.typnamespace,
             pg_type.typtype,
             pg_type.typowner,
             (aclexplode(COALESCE(pg_type.typacl, acldefault('T'::"char", pg_type.typowner)))).grantor AS grantor,
             (aclexplode(COALESCE(pg_type.typacl,
                                  acldefault('T'::"char", pg_type.typowner)))).grantee                 AS grantee,
             (aclexplode(COALESCE(pg_type.typacl,
                                  acldefault('T'::"char", pg_type.typowner)))).privilege_type          AS privilege_type,
             (aclexplode(COALESCE(pg_type.typacl,
                                  acldefault('T'::"char", pg_type.typowner)))).is_grantable            AS is_grantable
      FROM pg_type) t(oid, typname, typnamespace, typtype, typowner, grantor, grantee, prtype, grantable),
     pg_namespace n,
     pg_authid u_grantor,
     (SELECT pg_authid.oid,
             pg_authid.rolname
      FROM pg_authid
      UNION ALL
      SELECT 0::oid AS oid,
             'PUBLIC'::name) grantee(oid, rolname)
WHERE t.typnamespace = n.oid
  AND t.typtype = 'c'::"char"
  AND t.grantee = grantee.oid
  AND t.grantor = u_grantor.oid
  AND t.prtype = 'USAGE'::text
  AND (pg_has_role(u_grantor.oid, 'USAGE'::text) OR pg_has_role(grantee.oid, 'USAGE'::text) OR
       grantee.rolname = 'PUBLIC'::name);

alter table information_schema.udt_privileges
    owner to postgres;

grant select on information_schema.udt_privileges to public;

