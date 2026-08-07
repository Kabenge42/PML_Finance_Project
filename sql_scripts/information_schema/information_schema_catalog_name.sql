CREATE VIEW information_schema.information_schema_catalog_name(catalog_name) AS
SELECT current_database()::information_schema.sql_identifier AS catalog_name;

ALTER TABLE information_schema.information_schema_catalog_name
	OWNER TO postgres;

GRANT SELECT ON information_schema.information_schema_catalog_name TO PUBLIC;