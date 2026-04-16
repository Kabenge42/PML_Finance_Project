create view information_schema.role_usage_grants
            (grantor, grantee, object_catalog, object_schema, object_name, object_type, privilege_type, is_grantable) as
SELECT grantor,
       grantee,
       object_catalog,
       object_schema,
       object_name,
       object_type,
       privilege_type,
       is_grantable
FROM usage_privileges
WHERE (grantor::name IN (SELECT enabled_roles.role_name
                         FROM enabled_roles))
   OR (grantee::name IN (SELECT enabled_roles.role_name
                         FROM enabled_roles));

alter table information_schema.role_usage_grants
    owner to postgres;

grant select on information_schema.role_usage_grants to public;

