create view information_schema.referential_constraints
            (constraint_catalog, constraint_schema, constraint_name, unique_constraint_catalog,
             unique_constraint_schema, unique_constraint_name, match_option, update_rule, delete_rule)
as
SELECT current_database()::sql_identifier AS constraint_catalog,
       ncon.nspname::sql_identifier       AS constraint_schema,
       con.conname::sql_identifier        AS constraint_name,
       CASE
           WHEN npkc.nspname IS NULL THEN NULL::name
           ELSE current_database()
           END::sql_identifier            AS unique_constraint_catalog,
       npkc.nspname::sql_identifier       AS unique_constraint_schema,
       pkc.conname::sql_identifier        AS unique_constraint_name,
       CASE con.confmatchtype
           WHEN 'f'::"char" THEN 'FULL'::text
           WHEN 'p'::"char" THEN 'PARTIAL'::text
           WHEN 's'::"char" THEN 'NONE'::text
           ELSE NULL::text
           END::character_data            AS match_option,
       CASE con.confupdtype
           WHEN 'c'::"char" THEN 'CASCADE'::text
           WHEN 'n'::"char" THEN 'SET NULL'::text
           WHEN 'd'::"char" THEN 'SET DEFAULT'::text
           WHEN 'r'::"char" THEN 'RESTRICT'::text
           WHEN 'a'::"char" THEN 'NO ACTION'::text
           ELSE NULL::text
           END::character_data            AS update_rule,
       CASE con.confdeltype
           WHEN 'c'::"char" THEN 'CASCADE'::text
           WHEN 'n'::"char" THEN 'SET NULL'::text
           WHEN 'd'::"char" THEN 'SET DEFAULT'::text
           WHEN 'r'::"char" THEN 'RESTRICT'::text
           WHEN 'a'::"char" THEN 'NO ACTION'::text
           ELSE NULL::text
           END::character_data            AS delete_rule
FROM pg_namespace ncon
         JOIN pg_constraint con ON ncon.oid = con.connamespace
         JOIN pg_class c ON con.conrelid = c.oid AND con.contype = 'f'::"char"
         LEFT JOIN pg_depend d1 ON d1.objid = con.oid AND d1.classid = 'pg_constraint'::regclass::oid AND
                                   d1.refclassid = 'pg_class'::regclass::oid AND d1.refobjsubid = 0
         LEFT JOIN pg_depend d2
                   ON d2.refclassid = 'pg_constraint'::regclass::oid AND d2.classid = 'pg_class'::regclass::oid AND
                      d2.objid = d1.refobjid AND d2.objsubid = 0 AND d2.deptype = 'i'::"char"
         LEFT JOIN pg_constraint pkc
                   ON pkc.oid = d2.refobjid AND (pkc.contype = ANY (ARRAY ['p'::"char", 'u'::"char"])) AND
                      pkc.conrelid = con.confrelid
         LEFT JOIN pg_namespace npkc ON pkc.connamespace = npkc.oid
WHERE pg_has_role(c.relowner, 'USAGE'::text)
   OR has_table_privilege(c.oid, 'INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER'::text)
   OR has_any_column_privilege(c.oid, 'INSERT, UPDATE, REFERENCES'::text);

alter table information_schema.referential_constraints
    owner to postgres;

grant select on information_schema.referential_constraints to public;

