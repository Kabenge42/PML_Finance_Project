create view information_schema.role_table_grants
			(grantor, grantee, table_catalog, table_schema, table_name, privilege_type, is_grantable, with_hierarchy)
as
-- missing source code
;

alter table information_schema.role_table_grants
	owner to postgres
;

grant select on information_schema.role_table_grants to public
;