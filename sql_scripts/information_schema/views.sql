CREATE VIEW information_schema.views
			(table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into,
			 is_trigger_updatable, is_trigger_deletable, is_trigger_insertable_into)
AS
SELECT current_database()::information_schema.sql_identifier        AS table_catalog,
       nc.nspname::information_schema.sql_identifier                AS table_schema,
       c.relname::information_schema.sql_identifier                 AS table_name,
       CASE
	       WHEN pg_has_role(c.relowner, 'USAGE'::text) THEN pg_get_viewdef(c.oid)
	       ELSE NULL::text END::information_schema.character_data   AS view_definition,
       CASE
	       WHEN 'check_option=cascaded'::text = ANY (c.reloptions) THEN 'CASCADED'::text
	       WHEN 'check_option=local'::text = ANY (c.reloptions) THEN 'LOCAL'::text
	       ELSE 'NONE'::text END::information_schema.character_data AS check_option,
       CASE
	       WHEN (pg_relation_is_updatable(c.oid::regclass, FALSE) & 20) = 20 THEN 'YES'::text
	       ELSE 'NO'::text END::information_schema.yes_or_no        AS is_updatable,
       CASE
	       WHEN (pg_relation_is_updatable(c.oid::regclass, FALSE) & 8) = 8 THEN 'YES'::text
	       ELSE 'NO'::text END::information_schema.yes_or_no        AS is_insertable_into,
       CASE
	       WHEN (EXISTS (SELECT 1
	                     FROM pg_trigger
	                     WHERE pg_trigger.tgrelid = c.oid AND (pg_trigger.tgtype::integer & 81) = 81)) THEN 'YES'::text
	       ELSE 'NO'::text END::information_schema.yes_or_no        AS is_trigger_updatable,
       CASE
	       WHEN (EXISTS (SELECT 1
	                     FROM pg_trigger
	                     WHERE pg_trigger.tgrelid = c.oid AND (pg_trigger.tgtype::integer & 73) = 73)) THEN 'YES'::text
	       ELSE 'NO'::text END::information_schema.yes_or_no        AS is_trigger_deletable,
       CASE
	       WHEN (EXISTS (SELECT 1
	                     FROM pg_trigger
	                     WHERE pg_trigger.tgrelid = c.oid AND (pg_trigger.tgtype::integer & 69) = 69)) THEN 'YES'::text
	       ELSE 'NO'::text END::information_schema.yes_or_no        AS is_trigger_insertable_into
FROM pg_namespace nc,
     pg_class     c
WHERE c.relnamespace = nc.oid
  AND c.relkind = 'v'::"char"
  AND NOT pg_is_other_temp_schema(nc.oid)
  AND (pg_has_role(c.relowner, 'USAGE'::text) OR
       has_table_privilege(c.oid, 'SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER'::text) OR
       has_any_column_privilege(c.oid, 'SELECT, INSERT, UPDATE, REFERENCES'::text));

ALTER TABLE information_schema.views
	OWNER TO postgres;

GRANT SELECT ON information_schema.views TO PUBLIC;