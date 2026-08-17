create view information_schema.administrable_role_authorizations(grantee, role_name, is_grantable)
as
-- missing source code
;

alter table information_schema.administrable_role_authorizations
	owner to postgres
;

grant select on information_schema.administrable_role_authorizations to public
;