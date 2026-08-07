CREATE VIEW information_schema.schemata
			(catalog_name, schema_name, schema_owner, default_character_set_catalog, default_character_set_schema,
			 default_character_set_name, sql_path)
AS
SELECT current_database()::information_schema.sql_identifier      AS catalog_name,
       n.nspname::information_schema.sql_identifier               AS schema_name,
       u.rolname::information_schema.sql_identifier               AS schema_owner,
       NULL::name::information_schema.sql_identifier              AS default_character_set_catalog,
       NULL::name::information_schema.sql_identifier              AS default_character_set_schema,
       NULL::name::information_schema.sql_identifier              AS default_character_set_name,
       NULL::character varying::information_schema.character_data AS sql_path
FROM pg_namespace n,
     pg_authid    u
WHERE n.nspowner = u.oid
  AND (pg_has_role(n.nspowner, 'USAGE'::text) OR has_schema_privilege(n.oid, 'CREATE, USAGE'::text));

ALTER TABLE information_schema.schemata
	OWNER TO postgres;

GRANT SELECT ON information_schema.schemata TO PUBLIC;