create view information_schema.role_usage_grants
			(grantor, grantee, object_catalog, object_schema, object_name, object_type, privilege_type, is_grantable)
as
-- missing source code
;

alter table information_schema.role_usage_grants
	owner to postgres
;

grant select on information_schema.role_usage_grants to public
;