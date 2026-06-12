CREATE VIEW information_schema.user_mappings(authorization_identifier, foreign_server_catalog, foreign_server_name) AS
-- missing source code
;

ALTER TABLE information_schema.user_mappings
	OWNER TO postgres;

GRANT SELECT ON information_schema.user_mappings TO PUBLIC;