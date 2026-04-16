create view information_schema.schemata
            (catalog_name, schema_name, schema_owner, default_character_set_catalog, default_character_set_schema,
             default_character_set_name, sql_path)
as
SELECT current_database()::sql_identifier      AS catalog_name,
       n.nspname::sql_identifier               AS schema_name,
       u.rolname::sql_identifier               AS schema_owner,
       NULL::name::sql_identifier              AS default_character_set_catalog,
       NULL::name::sql_identifier              AS default_character_set_schema,
       NULL::name::sql_identifier              AS default_character_set_name,
       NULL::character varying::character_data AS sql_path
FROM pg_namespace n,
     pg_authid u
WHERE n.nspowner = u.oid
  AND (pg_has_role(n.nspowner, 'USAGE'::text) OR has_schema_privilege(n.oid, 'CREATE, USAGE'::text));

alter table information_schema.schemata
    owner to postgres;

grant select on information_schema.schemata to public;

