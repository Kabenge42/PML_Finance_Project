create view information_schema._pg_user_mappings
			(oid, umoptions, umuser, authorization_identifier, foreign_server_catalog, foreign_server_name, srvowner)
as
-- missing source code
;

alter table information_schema._pg_user_mappings
	owner to postgres
;