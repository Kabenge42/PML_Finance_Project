create view information_schema.role_column_grants
			(grantor, grantee, table_catalog, table_schema, table_name, column_name, privilege_type, is_grantable)
as
-- missing source code
;

alter table information_schema.role_column_grants
	owner to postgres
;

grant select on information_schema.role_column_grants to public
;