create view information_schema.table_privileges
			(grantor, grantee, table_catalog, table_schema, table_name, privilege_type, is_grantable, with_hierarchy)
as
-- missing source code
;

alter table information_schema.table_privileges
	owner to postgres
;

grant select on information_schema.table_privileges to public
;