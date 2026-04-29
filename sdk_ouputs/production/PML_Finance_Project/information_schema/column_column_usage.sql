create view information_schema.column_column_usage
            (table_catalog, table_schema, table_name, column_name, dependent_column) as
SELECT DISTINCT current_database()::sql_identifier AS table_catalog,
                n.nspname::sql_identifier          AS table_schema,
                c.relname::sql_identifier          AS table_name,
                ac.attname::sql_identifier         AS column_name,
                ad.attname::sql_identifier         AS dependent_column
FROM pg_namespace n,
     pg_class c,
     pg_depend d,
     pg_attribute ac,
     pg_attribute ad,
     pg_attrdef atd
WHERE n.oid = c.relnamespace
  AND c.oid = ac.attrelid
  AND c.oid = ad.attrelid
  AND ac.attnum <> ad.attnum
  AND ad.attrelid = atd.adrelid
  AND ad.attnum = atd.adnum
  AND d.classid = 'pg_attrdef'::regclass::oid
  AND d.refclassid = 'pg_class'::regclass::oid
  AND d.objid = atd.oid
  AND d.refobjid = ac.attrelid
  AND d.refobjsubid = ac.attnum
  AND ad.attgenerated <> ''::"char"
  AND pg_has_role(c.relowner, 'USAGE'::text);

alter table information_schema.column_column_usage
    owner to postgres;

grant select on information_schema.column_column_usage to public;

