CREATE VIEW information_schema._pg_foreign_servers
			(oid, srvoptions, foreign_server_catalog, foreign_server_name, foreign_data_wrapper_catalog,
			 foreign_data_wrapper_name, foreign_server_type, foreign_server_version, authorization_identifier)
AS
SELECT s.oid,
       s.srvoptions,
       current_database()::information_schema.sql_identifier AS foreign_server_catalog,
       s.srvname::information_schema.sql_identifier          AS foreign_server_name,
       current_database()::information_schema.sql_identifier AS foreign_data_wrapper_catalog,
       w.fdwname::information_schema.sql_identifier          AS foreign_data_wrapper_name,
       s.srvtype::information_schema.character_data          AS foreign_server_type,
       s.srvversion::information_schema.character_data       AS foreign_server_version,
       u.rolname::information_schema.sql_identifier          AS authorization_identifier
FROM pg_foreign_server       s,
     pg_foreign_data_wrapper w,
     pg_authid               u
WHERE w.oid = s.srvfdw
  AND u.oid = s.srvowner
  AND (pg_has_role(s.srvowner, 'USAGE'::text) OR has_server_privilege(s.oid, 'USAGE'::text));

ALTER TABLE information_schema._pg_foreign_servers
	OWNER TO postgres;