create view information_schema.role_column_grants
            (grantor, grantee, table_catalog, table_schema, table_name, column_name, privilege_type, is_grantable) as
SELECT grantor,
       grantee,
       table_catalog,
       table_schema,
       table_name,
       column_name,
       privilege_type,
       is_grantable
FROM column_privileges
WHERE (grantor::name IN (SELECT enabled_roles.role_name
                         FROM enabled_roles))
   OR (grantee::name IN (SELECT enabled_roles.role_name
                         FROM enabled_roles));

alter table information_schema.role_column_grants
    owner to postgres;

grant select on information_schema.role_column_grants to public;

