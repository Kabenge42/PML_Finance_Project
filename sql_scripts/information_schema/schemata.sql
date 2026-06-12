CREATE VIEW information_schema.schemata
			(catalog_name, schema_name, schema_owner, default_character_set_catalog, default_character_set_schema,
			 default_character_set_name, sql_path)
AS
-- missing source code
;

ALTER TABLE information_schema.schemata
	OWNER TO postgres;

GRANT SELECT ON information_schema.schemata TO PUBLIC;