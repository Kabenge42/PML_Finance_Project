CREATE VIEW information_schema.column_column_usage
			(table_catalog, table_schema, table_name, column_name, dependent_column) AS
SELECT DISTINCT
       current_database()::information_schema.sql_identifier AS table_catalog,
       n.nspname::information_schema.sql_identifier          AS table_schema,
       c.relname::information_schema.sql_identifier          AS table_name,
       ac.attname::information_schema.sql_identifier         AS column_name,
       ad.attname::information_schema.sql_identifier         AS dependent_column
FROM pg_namespace n,
     pg_class     c,
     pg_depend    d,
     pg_attribute ac,
     pg_attribute ad,
     pg_attrdef   atd
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

ALTER TABLE information_schema.column_column_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.column_column_usage TO PUBLIC;