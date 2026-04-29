create view information_schema.sequences
            (sequence_catalog, sequence_schema, sequence_name, data_type, numeric_precision, numeric_precision_radix,
             numeric_scale, start_value, minimum_value, maximum_value, increment, cycle_option)
as
SELECT current_database()::sql_identifier                                AS sequence_catalog,
       nc.nspname::sql_identifier                                        AS sequence_schema,
       c.relname::sql_identifier                                         AS sequence_name,
       format_type(s.seqtypid, NULL::integer)::character_data            AS data_type,
       _pg_numeric_precision(s.seqtypid, '-1'::integer)::cardinal_number AS numeric_precision,
       2::cardinal_number                                                AS numeric_precision_radix,
       0::cardinal_number                                                AS numeric_scale,
       s.seqstart::character_data                                        AS start_value,
       s.seqmin::character_data                                          AS minimum_value,
       s.seqmax::character_data                                          AS maximum_value,
       s.seqincrement::character_data                                    AS increment,
       CASE
           WHEN s.seqcycle THEN 'YES'::text
           ELSE 'NO'::text
           END::yes_or_no                                                AS cycle_option
FROM pg_namespace nc,
     pg_class c,
     pg_sequence s
WHERE c.relnamespace = nc.oid
  AND c.relkind = 'S'::"char"
  AND NOT (EXISTS (SELECT 1
                   FROM pg_depend
                   WHERE pg_depend.classid = 'pg_class'::regclass::oid
                     AND pg_depend.objid = c.oid
                     AND pg_depend.deptype = 'i'::"char"))
  AND NOT pg_is_other_temp_schema(nc.oid)
  AND c.oid = s.seqrelid
  AND (pg_has_role(c.relowner, 'USAGE'::text) OR has_sequence_privilege(c.oid, 'SELECT, UPDATE, USAGE'::text));

alter table information_schema.sequences
    owner to postgres;

grant select on information_schema.sequences to public;

