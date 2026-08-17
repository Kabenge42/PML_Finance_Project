create view information_schema.applicable_roles(grantee, role_name, is_grantable)
as
-- missing source code
;

alter table information_schema.applicable_roles
	owner to postgres
;

grant select on information_schema.applicable_roles to public
;