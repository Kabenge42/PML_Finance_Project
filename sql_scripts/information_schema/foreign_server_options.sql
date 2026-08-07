CREATE VIEW information_schema.foreign_server_options
			(foreign_server_catalog, foreign_server_name, option_name, option_value) AS
SELECT foreign_server_catalog,
       foreign_server_name,
       (pg_options_to_table(srvoptions)).option_name::information_schema.sql_identifier  AS option_name,
       (pg_options_to_table(srvoptions)).option_value::information_schema.character_data AS option_value
FROM information_schema._pg_foreign_servers s;

ALTER TABLE information_schema.foreign_server_options
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_server_options TO PUBLIC;