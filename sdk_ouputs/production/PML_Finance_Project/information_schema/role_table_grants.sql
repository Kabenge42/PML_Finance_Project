create view information_schema.role_table_grants
            (grantor, grantee, table_catalog, table_schema, table_name, privilege_type, is_grantable, with_hierarchy) as
SELECT grantor,
       grantee,
       table_catalog,
       table_schema,
       table_name,
       privilege_type,
       is_grantable,
       with_hierarchy
FROM table_privileges
WHERE (grantor::name IN (SELECT enabled_roles.role_name
                         FROM enabled_roles))
   OR (grantee::name IN (SELECT enabled_roles.role_name
                         FROM enabled_roles));

alter table information_schema.role_table_grants
    owner to postgres;

grant select on information_schema.role_table_grants to public;

