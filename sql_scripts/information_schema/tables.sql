CREATE VIEW information_schema.tables
			(table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation,
			 user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed,
			 commit_action)
AS
SELECT current_database()::information_schema.sql_identifier                                                          AS table_catalog,
       nc.nspname::information_schema.sql_identifier                                                                  AS table_schema,
       c.relname::information_schema.sql_identifier                                                                   AS table_name,
       CASE
	       WHEN nc.oid = pg_my_temp_schema() THEN 'LOCAL TEMPORARY'::text
	       WHEN c.relkind = ANY (ARRAY ['r'::"char", 'p'::"char"]) THEN 'BASE TABLE'::text
	       WHEN c.relkind = 'v'::"char" THEN 'VIEW'::text
	       WHEN c.relkind = 'f'::"char" THEN 'FOREIGN'::text
	       ELSE NULL::text END::information_schema.character_data                                                     AS table_type,
       NULL::name::information_schema.sql_identifier                                                                  AS self_referencing_column_name,
       NULL::character varying::information_schema.character_data                                                     AS reference_generation,
       CASE
	       WHEN t.typname IS NOT NULL THEN current_database()
	       ELSE NULL::name END::information_schema.sql_identifier                                                     AS user_defined_type_catalog,
       nt.nspname::information_schema.sql_identifier                                                                  AS user_defined_type_schema,
       t.typname::information_schema.sql_identifier                                                                   AS user_defined_type_name,
       CASE
	       WHEN (c.relkind = ANY (ARRAY ['r'::"char", 'p'::"char"])) OR
	            (c.relkind = ANY (ARRAY ['v'::"char", 'f'::"char"])) AND
	            (pg_relation_is_updatable(c.oid::regclass, FALSE) & 8) = 8 THEN 'YES'::text
	       ELSE 'NO'::text END::information_schema.yes_or_no                                                          AS is_insertable_into,
       CASE WHEN t.typname IS NOT NULL THEN 'YES'::text ELSE 'NO'::text END::information_schema.yes_or_no             AS is_typed,
       NULL::character varying::information_schema.character_data                                                     AS commit_action
FROM pg_namespace           nc
	     JOIN      pg_class c ON nc.oid = c.relnamespace
	     LEFT JOIN (pg_type t JOIN pg_namespace nt ON t.typnamespace = nt.oid) ON c.reloftype = t.oid
WHERE (c.relkind = ANY (ARRAY ['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
  AND NOT pg_is_other_temp_schema(nc.oid)
  AND (pg_has_role(c.relowner, 'USAGE'::text) OR
       has_table_privilege(c.oid, 'SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER'::text) OR
       has_any_column_privilege(c.oid, 'SELECT, INSERT, UPDATE, REFERENCES'::text));

ALTER TABLE information_schema.tables
	OWNER TO postgres;

GRANT SELECT ON information_schema.tables TO PUBLIC;