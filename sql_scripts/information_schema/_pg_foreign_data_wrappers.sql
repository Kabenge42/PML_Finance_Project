create view information_schema._pg_foreign_data_wrappers
			(oid, fdwowner, fdwoptions, foreign_data_wrapper_catalog, foreign_data_wrapper_name,
			 authorization_identifier, foreign_data_wrapper_language)
as
-- missing source code
;

alter table information_schema._pg_foreign_data_wrappers
	owner to postgres
;