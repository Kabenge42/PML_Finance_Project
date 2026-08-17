create view information_schema.column_udt_usage
			(udt_catalog, udt_schema, udt_name, table_catalog, table_schema, table_name, column_name)
as
-- missing source code
;

alter table information_schema.column_udt_usage
	owner to postgres
;

grant select on information_schema.column_udt_usage to public
;