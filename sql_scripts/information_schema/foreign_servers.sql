CREATE VIEW information_schema.foreign_servers
			(foreign_server_catalog, foreign_server_name, foreign_data_wrapper_catalog, foreign_data_wrapper_name,
			 foreign_server_type, foreign_server_version, authorization_identifier)
AS
-- missing source code
;

ALTER TABLE information_schema.foreign_servers
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_servers TO PUBLIC;