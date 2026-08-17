create view information_schema.udt_privileges
			(grantor, grantee, udt_catalog, udt_schema, udt_name, privilege_type, is_grantable)
as
-- missing source code
;

alter table information_schema.udt_privileges
	owner to postgres
;

grant select on information_schema.udt_privileges to public
;