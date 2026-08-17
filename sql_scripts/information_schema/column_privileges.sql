create view information_schema.column_privileges
			(grantor, grantee, table_catalog, table_schema, table_name, column_name, privilege_type, is_grantable)
as
-- missing source code
;

alter table information_schema.column_privileges
	owner to postgres
;

grant select on information_schema.column_privileges to public
;