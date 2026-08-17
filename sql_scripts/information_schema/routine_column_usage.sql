create view information_schema.routine_column_usage
			(specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name,
			 table_catalog, table_schema, table_name, column_name)
as
-- missing source code
;

alter table information_schema.routine_column_usage
	owner to postgres
;

grant select on information_schema.routine_column_usage to public
;