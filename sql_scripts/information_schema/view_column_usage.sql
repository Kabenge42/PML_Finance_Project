create view information_schema.view_column_usage
			(view_catalog, view_schema, view_name, table_catalog, table_schema, table_name, column_name)
as
-- missing source code
;

alter table information_schema.view_column_usage
	owner to postgres
;

grant select on information_schema.view_column_usage to public
;