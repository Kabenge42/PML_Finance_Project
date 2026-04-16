create view information_schema.attributes
            (udt_catalog, udt_schema, udt_name, attribute_name, ordinal_position, attribute_default, is_nullable,
             data_type, character_maximum_length, character_octet_length, character_set_catalog, character_set_schema,
             character_set_name, collation_catalog, collation_schema, collation_name, numeric_precision,
             numeric_precision_radix, numeric_scale, datetime_precision, interval_type, interval_precision,
             attribute_udt_catalog, attribute_udt_schema, attribute_udt_name, scope_catalog, scope_schema, scope_name,
             maximum_cardinality, dtd_identifier, is_derived_reference_attribute)
as
SELECT current_database()::sql_identifier                                                    AS udt_catalog,
       nc.nspname::sql_identifier                                                            AS udt_schema,
       c.relname::sql_identifier                                                             AS udt_name,
       a.attname::sql_identifier                                                             AS attribute_name,
       a.attnum::cardinal_number                                                             AS ordinal_position,
       pg_get_expr(ad.adbin, ad.adrelid)::character_data                                     AS attribute_default,
       CASE
           WHEN a.attnotnull OR t.typtype = 'd'::"char" AND t.typnotnull THEN 'NO'::text
           ELSE 'YES'::text
           END::yes_or_no                                                                    AS is_nullable,
       CASE
           WHEN t.typelem <> 0::oid AND t.typlen = '-1'::integer THEN 'ARRAY'::text
           WHEN nt.nspname = 'pg_catalog'::name THEN format_type(a.atttypid, NULL::integer)
           ELSE 'USER-DEFINED'::text
           END::character_data                                                               AS data_type,
       _pg_char_max_length(_pg_truetypid(a.*, t.*),
                           _pg_truetypmod(a.*, t.*))::cardinal_number                        AS character_maximum_length,
       _pg_char_octet_length(_pg_truetypid(a.*, t.*),
                             _pg_truetypmod(a.*, t.*))::cardinal_number                      AS character_octet_length,
       NULL::name::sql_identifier                                                            AS character_set_catalog,
       NULL::name::sql_identifier                                                            AS character_set_schema,
       NULL::name::sql_identifier                                                            AS character_set_name,
       CASE
           WHEN nco.nspname IS NOT NULL THEN current_database()
           ELSE NULL::name
           END::sql_identifier                                                               AS collation_catalog,
       nco.nspname::sql_identifier                                                           AS collation_schema,
       co.collname::sql_identifier                                                           AS collation_name,
       _pg_numeric_precision(_pg_truetypid(a.*, t.*),
                             _pg_truetypmod(a.*, t.*))::cardinal_number                      AS numeric_precision,
       _pg_numeric_precision_radix(_pg_truetypid(a.*, t.*),
                                   _pg_truetypmod(a.*, t.*))::cardinal_number                AS numeric_precision_radix,
       _pg_numeric_scale(_pg_truetypid(a.*, t.*), _pg_truetypmod(a.*, t.*))::cardinal_number AS numeric_scale,
       _pg_datetime_precision(_pg_truetypid(a.*, t.*),
                              _pg_truetypmod(a.*, t.*))::cardinal_number                     AS datetime_precision,
       _pg_interval_type(_pg_truetypid(a.*, t.*), _pg_truetypmod(a.*, t.*))::character_data  AS interval_type,
       NULL::integer::cardinal_number                                                        AS interval_precision,
       current_database()::sql_identifier                                                    AS attribute_udt_catalog,
       nt.nspname::sql_identifier                                                            AS attribute_udt_schema,
       t.typname::sql_identifier                                                             AS attribute_udt_name,
       NULL::name::sql_identifier                                                            AS scope_catalog,
       NULL::name::sql_identifier                                                            AS scope_schema,
       NULL::name::sql_identifier                                                            AS scope_name,
       NULL::integer::cardinal_number                                                        AS maximum_cardinality,
       a.attnum::sql_identifier                                                              AS dtd_identifier,
       'NO'::character varying::yes_or_no                                                    AS is_derived_reference_attribute
FROM pg_attribute a
         LEFT JOIN pg_attrdef ad ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
         JOIN (pg_class c
    JOIN pg_namespace nc ON c.relnamespace = nc.oid) ON a.attrelid = c.oid
         JOIN (pg_type t
    JOIN pg_namespace nt ON t.typnamespace = nt.oid) ON a.atttypid = t.oid
         LEFT JOIN (pg_collation co
    JOIN pg_namespace nco ON co.collnamespace = nco.oid)
                   ON a.attcollation = co.oid AND (nco.nspname <> 'pg_catalog'::name OR co.collname <> 'default'::name)
WHERE a.attnum > 0
  AND NOT a.attisdropped
  AND c.relkind = 'c'::"char"
  AND (pg_has_role(c.relowner, 'USAGE'::text) OR has_type_privilege(c.reltype, 'USAGE'::text));

alter table information_schema.attributes
    owner to postgres;

grant select on information_schema.attributes to public;

