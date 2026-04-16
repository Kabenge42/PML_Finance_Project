create view information_schema.domains
            (domain_catalog, domain_schema, domain_name, data_type, character_maximum_length, character_octet_length,
             character_set_catalog, character_set_schema, character_set_name, collation_catalog, collation_schema,
             collation_name, numeric_precision, numeric_precision_radix, numeric_scale, datetime_precision,
             interval_type, interval_precision, domain_default, udt_catalog, udt_schema, udt_name, scope_catalog,
             scope_schema, scope_name, maximum_cardinality, dtd_identifier)
as
SELECT current_database()::sql_identifier                                       AS domain_catalog,
       nt.nspname::sql_identifier                                               AS domain_schema,
       t.typname::sql_identifier                                                AS domain_name,
       CASE
           WHEN t.typelem <> 0::oid AND t.typlen = '-1'::integer THEN 'ARRAY'::text
           WHEN nbt.nspname = 'pg_catalog'::name THEN format_type(t.typbasetype, NULL::integer)
           ELSE 'USER-DEFINED'::text
           END::character_data                                                  AS data_type,
       _pg_char_max_length(t.typbasetype, t.typtypmod)::cardinal_number         AS character_maximum_length,
       _pg_char_octet_length(t.typbasetype, t.typtypmod)::cardinal_number       AS character_octet_length,
       NULL::name::sql_identifier                                               AS character_set_catalog,
       NULL::name::sql_identifier                                               AS character_set_schema,
       NULL::name::sql_identifier                                               AS character_set_name,
       CASE
           WHEN nco.nspname IS NOT NULL THEN current_database()
           ELSE NULL::name
           END::sql_identifier                                                  AS collation_catalog,
       nco.nspname::sql_identifier                                              AS collation_schema,
       co.collname::sql_identifier                                              AS collation_name,
       _pg_numeric_precision(t.typbasetype, t.typtypmod)::cardinal_number       AS numeric_precision,
       _pg_numeric_precision_radix(t.typbasetype, t.typtypmod)::cardinal_number AS numeric_precision_radix,
       _pg_numeric_scale(t.typbasetype, t.typtypmod)::cardinal_number           AS numeric_scale,
       _pg_datetime_precision(t.typbasetype, t.typtypmod)::cardinal_number      AS datetime_precision,
       _pg_interval_type(t.typbasetype, t.typtypmod)::character_data            AS interval_type,
       NULL::integer::cardinal_number                                           AS interval_precision,
       t.typdefault::character_data                                             AS domain_default,
       current_database()::sql_identifier                                       AS udt_catalog,
       nbt.nspname::sql_identifier                                              AS udt_schema,
       bt.typname::sql_identifier                                               AS udt_name,
       NULL::name::sql_identifier                                               AS scope_catalog,
       NULL::name::sql_identifier                                               AS scope_schema,
       NULL::name::sql_identifier                                               AS scope_name,
       NULL::integer::cardinal_number                                           AS maximum_cardinality,
       1::sql_identifier                                                        AS dtd_identifier
FROM pg_type t
         JOIN pg_namespace nt ON t.typnamespace = nt.oid
         JOIN (pg_type bt
    JOIN pg_namespace nbt ON bt.typnamespace = nbt.oid) ON t.typbasetype = bt.oid AND t.typtype = 'd'::"char"
         LEFT JOIN (pg_collation co
    JOIN pg_namespace nco ON co.collnamespace = nco.oid)
                   ON t.typcollation = co.oid AND (nco.nspname <> 'pg_catalog'::name OR co.collname <> 'default'::name)
WHERE pg_has_role(t.typowner, 'USAGE'::text)
   OR has_type_privilege(t.oid, 'USAGE'::text);

alter table information_schema.domains
    owner to postgres;

grant select on information_schema.domains to public;

