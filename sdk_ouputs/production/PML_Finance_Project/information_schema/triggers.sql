create view information_schema.triggers
            (trigger_catalog, trigger_schema, trigger_name, event_manipulation, event_object_catalog,
             event_object_schema, event_object_table, action_order, action_condition, action_statement,
             action_orientation, action_timing, action_reference_old_table, action_reference_new_table,
             action_reference_old_row, action_reference_new_row, created)
as
SELECT current_database()::sql_identifier                                                                                                                                         AS trigger_catalog,
       n.nspname::sql_identifier                                                                                                                                                  AS trigger_schema,
       t.tgname::sql_identifier                                                                                                                                                   AS trigger_name,
       em.text::character_data                                                                                                                                                    AS event_manipulation,
       current_database()::sql_identifier                                                                                                                                         AS event_object_catalog,
       n.nspname::sql_identifier                                                                                                                                                  AS event_object_schema,
       c.relname::sql_identifier                                                                                                                                                  AS event_object_table,
       rank()
       OVER (PARTITION BY (n.nspname::sql_identifier), (c.relname::sql_identifier), em.num, (t.tgtype::integer & 1), (t.tgtype::integer & 66) ORDER BY t.tgname)::cardinal_number AS action_order,
       CASE
           WHEN pg_has_role(c.relowner, 'USAGE'::text) THEN (regexp_match(pg_get_triggerdef(t.oid),
                                                                          '.{35,} WHEN \((.+)\) EXECUTE FUNCTION'::text))[1]
           ELSE NULL::text
           END::character_data                                                                                                                                                    AS action_condition,
       SUBSTRING(pg_get_triggerdef(t.oid) FROM
                 POSITION(('EXECUTE FUNCTION'::text) IN (SUBSTRING(pg_get_triggerdef(t.oid) FROM 48))) +
                 47)::character_data                                                                                                                                              AS action_statement,
       CASE t.tgtype::integer & 1
           WHEN 1 THEN 'ROW'::text
           ELSE 'STATEMENT'::text
           END::character_data                                                                                                                                                    AS action_orientation,
       CASE t.tgtype::integer & 66
           WHEN 2 THEN 'BEFORE'::text
           WHEN 64 THEN 'INSTEAD OF'::text
           ELSE 'AFTER'::text
           END::character_data                                                                                                                                                    AS action_timing,
       t.tgoldtable::sql_identifier                                                                                                                                               AS action_reference_old_table,
       t.tgnewtable::sql_identifier                                                                                                                                               AS action_reference_new_table,
       NULL::name::sql_identifier                                                                                                                                                 AS action_reference_old_row,
       NULL::name::sql_identifier                                                                                                                                                 AS action_reference_new_row,
       NULL::timestamp with time zone::time_stamp                                                                                                                                 AS created
FROM pg_namespace n,
     pg_class c,
     pg_trigger t,
     (VALUES (4, 'INSERT'::text), (8, 'DELETE'::text), (16, 'UPDATE'::text)) em(num, text)
WHERE n.oid = c.relnamespace
  AND c.oid = t.tgrelid
  AND (t.tgtype::integer & em.num) <> 0
  AND NOT t.tgisinternal
  AND NOT pg_is_other_temp_schema(n.oid)
  AND (pg_has_role(c.relowner, 'USAGE'::text) OR
       has_table_privilege(c.oid, 'INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER'::text) OR
       has_any_column_privilege(c.oid, 'INSERT, UPDATE, REFERENCES'::text));

alter table information_schema.triggers
    owner to postgres;

grant select on information_schema.triggers to public;

