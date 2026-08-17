create view information_schema.usage_privileges
			(grantor, grantee, object_catalog, object_schema, object_name, object_type, privilege_type, is_grantable)
as
-- missing source code
;

alter table information_schema.usage_privileges
	owner to postgres
;

grant select on information_schema.usage_privileges to public
;