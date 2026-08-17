create view information_schema.constraint_table_usage
			(table_catalog, table_schema, table_name, constraint_catalog, constraint_schema, constraint_name)
as
-- missing source code
;

alter table information_schema.constraint_table_usage
	owner to postgres
;

grant select on information_schema.constraint_table_usage to public
;