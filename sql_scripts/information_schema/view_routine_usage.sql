create view information_schema.view_routine_usage
			(table_catalog, table_schema, table_name, specific_catalog, specific_schema, specific_name)
as
-- missing source code
;

alter table information_schema.view_routine_usage
	owner to postgres
;

grant select on information_schema.view_routine_usage to public
;