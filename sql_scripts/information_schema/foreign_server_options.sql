CREATE VIEW information_schema.foreign_server_options
			(foreign_server_catalog, foreign_server_name, option_name, option_value) AS
-- missing source code
;

ALTER TABLE information_schema.foreign_server_options
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_server_options TO PUBLIC;