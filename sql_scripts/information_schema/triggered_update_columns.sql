CREATE VIEW information_schema.triggered_update_columns
			(trigger_catalog, trigger_schema, trigger_name, event_object_catalog, event_object_schema,
			 event_object_table, event_object_column)
AS
SELECT current_database()::information_schema.sql_identifier AS trigger_catalog,
       n.nspname::information_schema.sql_identifier          AS trigger_schema,
       t.tgname::information_schema.sql_identifier           AS trigger_name,
       current_database()::information_schema.sql_identifier AS event_object_catalog,
       n.nspname::information_schema.sql_identifier          AS event_object_schema,
       c.relname::information_schema.sql_identifier          AS event_object_table,
       a.attname::information_schema.sql_identifier          AS event_object_column
FROM pg_namespace n,
     pg_class     c,
     pg_trigger   t,
     (SELECT ta0.tgoid, (ta0.tgat).x AS tgattnum, (ta0.tgat).n AS tgattpos
      FROM (SELECT pg_trigger.oid AS tgoid, information_schema._pg_expandarray(pg_trigger.tgattr) AS tgat
            FROM pg_trigger) ta0
     )            ta,
     pg_attribute a
WHERE n.oid = c.relnamespace
  AND c.oid = t.tgrelid
  AND t.oid = ta.tgoid
  AND a.attrelid = t.tgrelid
  AND a.attnum = ta.tgattnum
  AND NOT t.tgisinternal
  AND NOT pg_is_other_temp_schema(n.oid)
  AND (pg_has_role(c.relowner, 'USAGE'::text) OR
       has_column_privilege(c.oid, a.attnum, 'INSERT, UPDATE, REFERENCES'::text));

ALTER TABLE information_schema.triggered_update_columns
	OWNER TO postgres;

GRANT SELECT ON information_schema.triggered_update_columns TO PUBLIC;