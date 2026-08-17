create view information_schema._pg_foreign_servers
			(oid, srvoptions, foreign_server_catalog, foreign_server_name, foreign_data_wrapper_catalog,
			 foreign_data_wrapper_name, foreign_server_type, foreign_server_version, authorization_identifier)
as
-- missing source code
;

alter table information_schema._pg_foreign_servers
	owner to postgres
;