create view information_schema.collations(collation_catalog, collation_schema, collation_name, pad_attribute) as
SELECT current_database()::sql_identifier          AS collation_catalog,
       nc.nspname::sql_identifier                  AS collation_schema,
       c.collname::sql_identifier                  AS collation_name,
       'NO PAD'::character varying::character_data AS pad_attribute
FROM pg_collation c,
     pg_namespace nc
WHERE c.collnamespace = nc.oid
  AND (c.collencoding = ANY (ARRAY ['-1'::integer, (SELECT pg_database.encoding
                                                    FROM pg_database
                                                    WHERE pg_database.datname = current_database())]));

alter table information_schema.collations
    owner to postgres;

grant select on information_schema.collations to public;

