create view information_schema.role_routine_grants
            (grantor, grantee, specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema,
             routine_name, privilege_type, is_grantable)
as
SELECT grantor,
       grantee,
       specific_catalog,
       specific_schema,
       specific_name,
       routine_catalog,
       routine_schema,
       routine_name,
       privilege_type,
       is_grantable
FROM routine_privileges
WHERE (grantor::name IN (SELECT enabled_roles.role_name
                         FROM enabled_roles))
   OR (grantee::name IN (SELECT enabled_roles.role_name
                         FROM enabled_roles));

alter table information_schema.role_routine_grants
    owner to postgres;

grant select on information_schema.role_routine_grants to public;

