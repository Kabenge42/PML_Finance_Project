create view information_schema.enabled_roles(role_name)
as
-- missing source code
;

alter table information_schema.enabled_roles
	owner to postgres
;

grant select on information_schema.enabled_roles to public
;