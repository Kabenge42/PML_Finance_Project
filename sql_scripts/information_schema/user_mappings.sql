CREATE VIEW information_schema.user_mappings(authorization_identifier, foreign_server_catalog, foreign_server_name) AS
SELECT authorization_identifier, foreign_server_catalog, foreign_server_name
FROM information_schema._pg_user_mappings;

ALTER TABLE information_schema.user_mappings
	OWNER TO postgres;

GRANT SELECT ON information_schema.user_mappings TO PUBLIC;