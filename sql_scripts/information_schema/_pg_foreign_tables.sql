create view information_schema._pg_foreign_tables
			(foreign_table_catalog, foreign_table_schema, foreign_table_name, ftoptions, foreign_server_catalog,
			 foreign_server_name, authorization_identifier)
as
-- missing source code
;

alter table information_schema._pg_foreign_tables
	owner to postgres
;