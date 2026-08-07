CREATE VIEW information_schema._pg_user_mappings
			(oid, umoptions, umuser, authorization_identifier, foreign_server_catalog, foreign_server_name, srvowner) AS
SELECT um.oid,
       um.umoptions,
       um.umuser,
       COALESCE(u.rolname, 'PUBLIC'::name)::information_schema.sql_identifier AS authorization_identifier,
       s.foreign_server_catalog,
       s.foreign_server_name,
       s.authorization_identifier                                             AS srvowner
FROM pg_user_mapping                        um
	     LEFT JOIN pg_authid                u ON u.oid = um.umuser,
     information_schema._pg_foreign_servers s
WHERE s.oid = um.umserver;

ALTER TABLE information_schema._pg_user_mappings
	OWNER TO postgres;