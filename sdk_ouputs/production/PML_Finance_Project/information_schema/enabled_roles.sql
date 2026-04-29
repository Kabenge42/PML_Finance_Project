create view information_schema.enabled_roles(role_name) as
SELECT rolname::sql_identifier AS role_name
FROM pg_authid a
WHERE pg_has_role(oid, 'USAGE'::text);

alter table information_schema.enabled_roles
    owner to postgres;

grant select on information_schema.enabled_roles to public;

