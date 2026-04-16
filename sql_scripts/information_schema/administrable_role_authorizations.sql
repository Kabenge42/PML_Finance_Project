create view information_schema.administrable_role_authorizations(grantee, role_name, is_grantable) as
SELECT grantee,
       role_name,
       is_grantable
FROM applicable_roles
WHERE is_grantable::text = 'YES'::text;

alter table information_schema.administrable_role_authorizations
    owner to postgres;

grant select on information_schema.administrable_role_authorizations to public;

