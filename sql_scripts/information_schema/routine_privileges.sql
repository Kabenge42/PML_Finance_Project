create view information_schema.routine_privileges
			(grantor, grantee, specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema,
			 routine_name, privilege_type, is_grantable)
as
-- missing source code
;

alter table information_schema.routine_privileges
	owner to postgres
;

grant select on information_schema.routine_privileges to public
;