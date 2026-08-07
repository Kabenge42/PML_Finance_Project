CREATE VIEW information_schema.character_sets
			(character_set_catalog, character_set_schema, character_set_name, character_repertoire, form_of_use,
			 default_collate_catalog, default_collate_schema, default_collate_name)
AS
SELECT NULL::name::information_schema.sql_identifier                         AS character_set_catalog,
       NULL::name::information_schema.sql_identifier                         AS character_set_schema,
       getdatabaseencoding()::information_schema.sql_identifier              AS character_set_name,
       CASE
	       WHEN getdatabaseencoding() = 'UTF8'::name THEN 'UCS'::name
	       ELSE getdatabaseencoding() END::information_schema.sql_identifier AS character_repertoire,
       getdatabaseencoding()::information_schema.sql_identifier              AS form_of_use,
       current_database()::information_schema.sql_identifier                 AS default_collate_catalog,
       nc.nspname::information_schema.sql_identifier                         AS default_collate_schema,
       c.collname::information_schema.sql_identifier                         AS default_collate_name
FROM pg_database d
	     LEFT JOIN (pg_collation c JOIN pg_namespace nc ON c.collnamespace = nc.oid)
	               ON d.datcollate = c.collcollate AND d.datctype = c.collctype
WHERE d.datname = current_database()
ORDER BY (char_length(c.collname::text)) DESC, c.collname
LIMIT 1;

ALTER TABLE information_schema.character_sets
	OWNER TO postgres;

GRANT SELECT ON information_schema.character_sets TO PUBLIC;