create view information_schema.column_options
			(table_catalog, table_schema, table_name, column_name, option_name, option_value)
as
-- missing source code
;

alter table information_schema.column_options
	owner to postgres
;

grant select on information_schema.column_options to public
;