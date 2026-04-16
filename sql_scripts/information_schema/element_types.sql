create view information_schema.element_types
            (object_catalog, object_schema, object_name, object_type, collection_type_identifier, data_type,
             character_maximum_length, character_octet_length, character_set_catalog, character_set_schema,
             character_set_name, collation_catalog, collation_schema, collation_name, numeric_precision,
             numeric_precision_radix, numeric_scale, datetime_precision, interval_type, interval_precision, udt_catalog,
             udt_schema, udt_name, scope_catalog, scope_schema, scope_name, maximum_cardinality, dtd_identifier)
as
SELECT current_database()::sql_identifier              AS object_catalog,
       n.nspname::sql_identifier                       AS object_schema,
       x.objname                                       AS object_name,
       x.objtype::character_data                       AS object_type,
       x.objdtdid::sql_identifier                      AS collection_type_identifier,
       CASE
           WHEN nbt.nspname = 'pg_catalog'::name THEN format_type(bt.oid, NULL::integer)
           ELSE 'USER-DEFINED'::text
           END::character_data                         AS data_type,
       NULL::integer::cardinal_number                  AS character_maximum_length,
       NULL::integer::cardinal_number                  AS character_octet_length,
       NULL::name::sql_identifier                      AS character_set_catalog,
       NULL::name::sql_identifier                      AS character_set_schema,
       NULL::name::sql_identifier                      AS character_set_name,
       CASE
           WHEN nco.nspname IS NOT NULL THEN current_database()
           ELSE NULL::name
           END::sql_identifier                         AS collation_catalog,
       nco.nspname::sql_identifier                     AS collation_schema,
       co.collname::sql_identifier                     AS collation_name,
       NULL::integer::cardinal_number                  AS numeric_precision,
       NULL::integer::cardinal_number                  AS numeric_precision_radix,
       NULL::integer::cardinal_number                  AS numeric_scale,
       NULL::integer::cardinal_number                  AS datetime_precision,
       NULL::character varying::character_data         AS interval_type,
       NULL::integer::cardinal_number                  AS interval_precision,
       current_database()::sql_identifier              AS udt_catalog,
       nbt.nspname::sql_identifier                     AS udt_schema,
       bt.typname::sql_identifier                      AS udt_name,
       NULL::name::sql_identifier                      AS scope_catalog,
       NULL::name::sql_identifier                      AS scope_schema,
       NULL::name::sql_identifier                      AS scope_name,
       NULL::integer::cardinal_number                  AS maximum_cardinality,
       ('a'::text || x.objdtdid::text)::sql_identifier AS dtd_identifier
FROM pg_namespace n,
     pg_type at,
     pg_namespace nbt,
     pg_type bt,
     (SELECT c.relnamespace,
             c.relname::sql_identifier AS relname,
             CASE
                 WHEN c.relkind = 'c'::"char" THEN 'USER-DEFINED TYPE'::text
                 ELSE 'TABLE'::text
                 END                   AS "case",
             a.attnum,
             a.atttypid,
             a.attcollation
      FROM pg_class c,
           pg_attribute a
      WHERE c.oid = a.attrelid
        AND (c.relkind = ANY (ARRAY ['r'::"char", 'v'::"char", 'f'::"char", 'c'::"char", 'p'::"char"]))
        AND a.attnum > 0
        AND NOT a.attisdropped
      UNION ALL
      SELECT t.typnamespace,
             t.typname::sql_identifier AS typname,
             'DOMAIN'::text            AS text,
             1,
             t.typbasetype,
             t.typcollation
      FROM pg_type t
      WHERE t.typtype = 'd'::"char"
      UNION ALL
      SELECT ss.pronamespace,
             nameconcatoid(ss.proname, ss.oid)::sql_identifier AS nameconcatoid,
             'ROUTINE'::text                                   AS text,
             (ss.x).n                                          AS n,
             (ss.x).x                                          AS x,
             0
      FROM (SELECT p.pronamespace,
                   p.proname,
                   p.oid,
                   _pg_expandarray(COALESCE(p.proallargtypes, p.proargtypes::oid[])) AS x
            FROM pg_proc p) ss
      UNION ALL
      SELECT p.pronamespace,
             nameconcatoid(p.proname, p.oid)::sql_identifier AS nameconcatoid,
             'ROUTINE'::text                                 AS text,
             0,
             p.prorettype,
             0
      FROM pg_proc p) x(objschema, objname, objtype, objdtdid, objtypeid, objcollation)
         LEFT JOIN (pg_collation co
         JOIN pg_namespace nco ON co.collnamespace = nco.oid)
                   ON x.objcollation = co.oid AND (nco.nspname <> 'pg_catalog'::name OR co.collname <> 'default'::name)
WHERE n.oid = x.objschema
  AND at.oid = x.objtypeid
  AND at.typelem <> 0::oid
  AND at.typlen = '-1'::integer
  AND at.typelem = bt.oid
  AND nbt.oid = bt.typnamespace
  AND ((n.nspname, x.objname::name, x.objtype, x.objdtdid::sql_identifier::name) IN
       (SELECT data_type_privileges.object_schema,
               data_type_privileges.object_name,
               data_type_privileges.object_type,
               data_type_privileges.dtd_identifier
        FROM data_type_privileges));

alter table information_schema.element_types
    owner to postgres;

grant select on information_schema.element_types to public;

