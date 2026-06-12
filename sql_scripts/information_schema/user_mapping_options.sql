CREATE VIEW information_schema.user_mapping_options
			(authorization_identifier, foreign_server_catalog, foreign_server_name, option_name, option_value) AS
-- missing source code
;

ALTER TABLE information_schema.user_mapping_options
	OWNER TO postgres;

GRANT SELECT ON information_schema.user_mapping_options TO PUBLIC;