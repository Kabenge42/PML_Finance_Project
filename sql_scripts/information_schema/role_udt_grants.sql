create view information_schema.role_udt_grants
			(grantor, grantee, udt_catalog, udt_schema, udt_name, privilege_type, is_grantable)
as
-- missing source code
;

alter table information_schema.role_udt_grants
	owner to postgres
;

grant select on information_schema.role_udt_grants to public
;