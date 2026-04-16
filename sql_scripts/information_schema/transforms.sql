create view information_schema.transforms
            (udt_catalog, udt_schema, udt_name, specific_catalog, specific_schema, specific_name, group_name,
             transform_type)
as
SELECT current_database()::sql_identifier              AS udt_catalog,
       nt.nspname::sql_identifier                      AS udt_schema,
       t.typname::sql_identifier                       AS udt_name,
       current_database()::sql_identifier              AS specific_catalog,
       np.nspname::sql_identifier                      AS specific_schema,
       nameconcatoid(p.proname, p.oid)::sql_identifier AS specific_name,
       l.lanname::sql_identifier                       AS group_name,
       'FROM SQL'::character varying::character_data   AS transform_type
FROM pg_type t
         JOIN pg_transform x ON t.oid = x.trftype
         JOIN pg_language l ON x.trflang = l.oid
         JOIN pg_proc p ON x.trffromsql::oid = p.oid
         JOIN pg_namespace nt ON t.typnamespace = nt.oid
         JOIN pg_namespace np ON p.pronamespace = np.oid
UNION
SELECT current_database()::sql_identifier              AS udt_catalog,
       nt.nspname::sql_identifier                      AS udt_schema,
       t.typname::sql_identifier                       AS udt_name,
       current_database()::sql_identifier              AS specific_catalog,
       np.nspname::sql_identifier                      AS specific_schema,
       nameconcatoid(p.proname, p.oid)::sql_identifier AS specific_name,
       l.lanname::sql_identifier                       AS group_name,
       'TO SQL'::character varying::character_data     AS transform_type
FROM pg_type t
         JOIN pg_transform x ON t.oid = x.trftype
         JOIN pg_language l ON x.trflang = l.oid
         JOIN pg_proc p ON x.trftosql::oid = p.oid
         JOIN pg_namespace nt ON t.typnamespace = nt.oid
         JOIN pg_namespace np ON p.pronamespace = np.oid
ORDER BY 1, 2, 3, 7, 8;

alter table information_schema.transforms
    owner to postgres;

