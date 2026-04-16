create view information_schema.constraint_column_usage
            (table_catalog, table_schema, table_name, column_name, constraint_catalog, constraint_schema,
             constraint_name)
as
SELECT current_database()::sql_identifier AS table_catalog,
       tblschema::sql_identifier          AS table_schema,
       tblname::sql_identifier            AS table_name,
       colname::sql_identifier            AS column_name,
       current_database()::sql_identifier AS constraint_catalog,
       cstrschema::sql_identifier         AS constraint_schema,
       cstrname::sql_identifier           AS constraint_name
FROM (SELECT DISTINCT nr.nspname,
                      r.relname,
                      r.relowner,
                      a.attname,
                      nc.nspname,
                      c.conname
      FROM pg_namespace nr,
           pg_class r,
           pg_attribute a,
           pg_depend d,
           pg_namespace nc,
           pg_constraint c
      WHERE nr.oid = r.relnamespace
        AND r.oid = a.attrelid
        AND d.refclassid = 'pg_class'::regclass::oid
        AND d.refobjid = r.oid
        AND d.refobjsubid = a.attnum
        AND d.classid = 'pg_constraint'::regclass::oid
        AND d.objid = c.oid
        AND c.connamespace = nc.oid
        AND c.contype = 'c'::"char"
        AND (r.relkind = ANY (ARRAY ['r'::"char", 'p'::"char"]))
        AND NOT a.attisdropped
      UNION ALL
      SELECT DISTINCT nr.nspname,
                      r.relname,
                      r.relowner,
                      a.attname,
                      nc.nspname,
                      c.conname
      FROM pg_namespace nr,
           pg_class r,
           pg_attribute a,
           pg_namespace nc,
           pg_constraint c
      WHERE nr.oid = r.relnamespace
        AND r.oid = a.attrelid
        AND r.oid = c.conrelid
        AND a.attnum = c.conkey[1]
        AND c.connamespace = nc.oid
        AND c.contype = 'n'::"char"
        AND (r.relkind = ANY (ARRAY ['r'::"char", 'p'::"char"]))
        AND NOT a.attisdropped
      UNION ALL
      SELECT nr.nspname,
             r.relname,
             r.relowner,
             a.attname,
             nc.nspname,
             c.conname
      FROM pg_namespace nr,
           pg_class r,
           pg_attribute a,
           pg_namespace nc,
           pg_constraint c
      WHERE nr.oid = r.relnamespace
        AND r.oid = a.attrelid
        AND nc.oid = c.connamespace
        AND r.oid =
            CASE c.contype
                WHEN 'f'::"char" THEN c.confrelid
                ELSE c.conrelid
                END
        AND (a.attnum = ANY (
          CASE c.contype
              WHEN 'f'::"char" THEN c.confkey
              ELSE c.conkey
              END))
        AND NOT a.attisdropped
        AND (c.contype = ANY (ARRAY ['p'::"char", 'u'::"char", 'f'::"char"]))
        AND (r.relkind = ANY (ARRAY ['r'::"char", 'p'::"char"]))) x(tblschema, tblname, tblowner, colname, cstrschema, cstrname)
WHERE pg_has_role(tblowner, 'USAGE'::text);

alter table information_schema.constraint_column_usage
    owner to postgres;

grant select on information_schema.constraint_column_usage to public;

