create view information_schema.columns
            (table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable,
             data_type, character_maximum_length, character_octet_length, numeric_precision, numeric_precision_radix,
             numeric_scale, datetime_precision, interval_type, interval_precision, character_set_catalog,
             character_set_schema, character_set_name, collation_catalog, collation_schema, collation_name,
             domain_catalog, domain_schema, domain_name, udt_catalog, udt_schema, udt_name, scope_catalog, scope_schema,
             scope_name, maximum_cardinality, dtd_identifier, is_self_referencing, is_identity, identity_generation,
             identity_start, identity_increment, identity_maximum, identity_minimum, identity_cycle, is_generated,
             generation_expression, is_updatable)
as
SELECT current_database()::sql_identifier                                                    AS table_catalog,
       nc.nspname::sql_identifier                                                            AS table_schema,
       c.relname::sql_identifier                                                             AS table_name,
       a.attname::sql_identifier                                                             AS column_name,
       a.attnum::cardinal_number                                                             AS ordinal_position,
       CASE
           WHEN a.attgenerated = ''::"char" THEN pg_get_expr(ad.adbin, ad.adrelid)
           ELSE NULL::text
           END::character_data                                                               AS column_default,
       CASE
           WHEN a.attnotnull OR t.typtype = 'd'::"char" AND t.typnotnull THEN 'NO'::text
           ELSE 'YES'::text
           END::yes_or_no                                                                    AS is_nullable,
       CASE
           WHEN t.typtype = 'd'::"char" THEN
               CASE
                   WHEN bt.typelem <> 0::oid AND bt.typlen = '-1'::integer THEN 'ARRAY'::text
                   WHEN nbt.nspname = 'pg_catalog'::name THEN format_type(t.typbasetype, NULL::integer)
                   ELSE 'USER-DEFINED'::text
                   END
           ELSE
               CASE
                   WHEN t.typelem <> 0::oid AND t.typlen = '-1'::integer THEN 'ARRAY'::text
                   WHEN nt.nspname = 'pg_catalog'::name THEN format_type(a.atttypid, NULL::integer)
                   ELSE 'USER-DEFINED'::text
                   END
           END::character_data                                                               AS data_type,
       _pg_char_max_length(_pg_truetypid(a.*, t.*),
                           _pg_truetypmod(a.*, t.*))::cardinal_number                        AS character_maximum_length,
       _pg_char_octet_length(_pg_truetypid(a.*, t.*),
                             _pg_truetypmod(a.*, t.*))::cardinal_number                      AS character_octet_length,
       _pg_numeric_precision(_pg_truetypid(a.*, t.*),
                             _pg_truetypmod(a.*, t.*))::cardinal_number                      AS numeric_precision,
       _pg_numeric_precision_radix(_pg_truetypid(a.*, t.*),
                                   _pg_truetypmod(a.*, t.*))::cardinal_number                AS numeric_precision_radix,
       _pg_numeric_scale(_pg_truetypid(a.*, t.*), _pg_truetypmod(a.*, t.*))::cardinal_number AS numeric_scale,
       _pg_datetime_precision(_pg_truetypid(a.*, t.*),
                              _pg_truetypmod(a.*, t.*))::cardinal_number                     AS datetime_precision,
       _pg_interval_type(_pg_truetypid(a.*, t.*), _pg_truetypmod(a.*, t.*))::character_data  AS interval_type,
       NULL::integer::cardinal_number                                                        AS interval_precision,
       NULL::name::sql_identifier                                                            AS character_set_catalog,
       NULL::name::sql_identifier                                                            AS character_set_schema,
       NULL::name::sql_identifier                                                            AS character_set_name,
       CASE
           WHEN nco.nspname IS NOT NULL THEN current_database()
           ELSE NULL::name
           END::sql_identifier                                                               AS collation_catalog,
       nco.nspname::sql_identifier                                                           AS collation_schema,
       co.collname::sql_identifier                                                           AS collation_name,
       CASE
           WHEN t.typtype = 'd'::"char" THEN current_database()
           ELSE NULL::name
           END::sql_identifier                                                               AS domain_catalog,
       CASE
           WHEN t.typtype = 'd'::"char" THEN nt.nspname
           ELSE NULL::name
           END::sql_identifier                                                               AS domain_schema,
       CASE
           WHEN t.typtype = 'd'::"char" THEN t.typname
           ELSE NULL::name
           END::sql_identifier                                                               AS domain_name,
       current_database()::sql_identifier                                                    AS udt_catalog,
       COALESCE(nbt.nspname, nt.nspname)::sql_identifier                                     AS udt_schema,
       COALESCE(bt.typname, t.typname)::sql_identifier                                       AS udt_name,
       NULL::name::sql_identifier                                                            AS scope_catalog,
       NULL::name::sql_identifier                                                            AS scope_schema,
       NULL::name::sql_identifier                                                            AS scope_name,
       NULL::integer::cardinal_number                                                        AS maximum_cardinality,
       a.attnum::sql_identifier                                                              AS dtd_identifier,
       'NO'::character varying::yes_or_no                                                    AS is_self_referencing,
       CASE
           WHEN a.attidentity = ANY (ARRAY ['a'::"char", 'd'::"char"]) THEN 'YES'::text
           ELSE 'NO'::text
           END::yes_or_no                                                                    AS is_identity,
       CASE a.attidentity
           WHEN 'a'::"char" THEN 'ALWAYS'::text
           WHEN 'd'::"char" THEN 'BY DEFAULT'::text
           ELSE NULL::text
           END::character_data                                                               AS identity_generation,
       seq.seqstart::character_data                                                          AS identity_start,
       seq.seqincrement::character_data                                                      AS identity_increment,
       seq.seqmax::character_data                                                            AS identity_maximum,
       seq.seqmin::character_data                                                            AS identity_minimum,
       CASE
           WHEN seq.seqcycle THEN 'YES'::text
           ELSE 'NO'::text
           END::yes_or_no                                                                    AS identity_cycle,
       CASE
           WHEN a.attgenerated <> ''::"char" THEN 'ALWAYS'::text
           ELSE 'NEVER'::text
           END::character_data                                                               AS is_generated,
       CASE
           WHEN a.attgenerated <> ''::"char" THEN pg_get_expr(ad.adbin, ad.adrelid)
           ELSE NULL::text
           END::character_data                                                               AS generation_expression,
       CASE
           WHEN (c.relkind = ANY (ARRAY ['r'::"char", 'p'::"char"])) OR
                (c.relkind = ANY (ARRAY ['v'::"char", 'f'::"char"])) AND
                pg_column_is_updatable(c.oid::regclass, a.attnum, false) THEN 'YES'::text
           ELSE 'NO'::text
           END::yes_or_no                                                                    AS is_updatable
FROM pg_attribute a
         LEFT JOIN pg_attrdef ad ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
         JOIN (pg_class c
    JOIN pg_namespace nc ON c.relnamespace = nc.oid) ON a.attrelid = c.oid
         JOIN (pg_type t
    JOIN pg_namespace nt ON t.typnamespace = nt.oid) ON a.atttypid = t.oid
         LEFT JOIN (pg_type bt
    JOIN pg_namespace nbt ON bt.typnamespace = nbt.oid) ON t.typtype = 'd'::"char" AND t.typbasetype = bt.oid
         LEFT JOIN (pg_collation co
    JOIN pg_namespace nco ON co.collnamespace = nco.oid)
                   ON a.attcollation = co.oid AND (nco.nspname <> 'pg_catalog'::name OR co.collname <> 'default'::name)
         LEFT JOIN (pg_depend dep
    JOIN pg_sequence seq ON dep.classid = 'pg_class'::regclass::oid AND dep.objid = seq.seqrelid AND
                            dep.deptype = 'i'::"char")
                   ON dep.refclassid = 'pg_class'::regclass::oid AND dep.refobjid = c.oid AND dep.refobjsubid = a.attnum
WHERE NOT pg_is_other_temp_schema(nc.oid)
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND (c.relkind = ANY (ARRAY ['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
  AND (pg_has_role(c.relowner, 'USAGE'::text) OR
       has_column_privilege(c.oid, a.attnum, 'SELECT, INSERT, UPDATE, REFERENCES'::text));

alter table information_schema.columns
    owner to postgres;

grant select on information_schema.columns to public;

