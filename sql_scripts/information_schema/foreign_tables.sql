CREATE VIEW information_schema.foreign_tables
			(foreign_table_catalog, foreign_table_schema, foreign_table_name, foreign_server_catalog,
			 foreign_server_name) AS
-- missing source code
;

ALTER TABLE information_schema.foreign_tables
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_tables TO PUBLIC;