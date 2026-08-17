create view information_schema.column_column_usage
			(table_catalog, table_schema, table_name, column_name, dependent_column)
as
-- missing source code
;

alter table information_schema.column_column_usage
	owner to postgres
;

grant select on information_schema.column_column_usage to public
;