create view information_schema.foreign_tables
			(foreign_table_catalog, foreign_table_schema, foreign_table_name, foreign_server_catalog,
			 foreign_server_name)
as
-- missing source code
;

alter table information_schema.foreign_tables
	owner to postgres
;

grant select on information_schema.foreign_tables to public
;