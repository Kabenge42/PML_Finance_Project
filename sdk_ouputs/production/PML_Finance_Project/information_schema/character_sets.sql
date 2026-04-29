create view information_schema.character_sets
            (character_set_catalog, character_set_schema, character_set_name, character_repertoire, form_of_use,
             default_collate_catalog, default_collate_schema, default_collate_name)
as
SELECT NULL::name::sql_identifier            AS character_set_catalog,
       NULL::name::sql_identifier            AS character_set_schema,
       getdatabaseencoding()::sql_identifier AS character_set_name,
       CASE
           WHEN getdatabaseencoding() = 'UTF8'::name THEN 'UCS'::name
           ELSE getdatabaseencoding()
           END::sql_identifier               AS character_repertoire,
       getdatabaseencoding()::sql_identifier AS form_of_use,
       current_database()::sql_identifier    AS default_collate_catalog,
       nc.nspname::sql_identifier            AS default_collate_schema,
       c.collname::sql_identifier            AS default_collate_name
FROM pg_database d
         LEFT JOIN (pg_collation c
    JOIN pg_namespace nc ON c.collnamespace = nc.oid) ON d.datcollate = c.collcollate AND d.datctype = c.collctype
WHERE d.datname = current_database()
ORDER BY (char_length(c.collname::text)) DESC, c.collname
LIMIT 1;

alter table information_schema.character_sets
    owner to postgres;

grant select on information_schema.character_sets to public;

