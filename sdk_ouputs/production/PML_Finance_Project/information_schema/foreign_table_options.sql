create view information_schema.foreign_table_options
            (foreign_table_catalog, foreign_table_schema, foreign_table_name, option_name, option_value) as
SELECT foreign_table_catalog,
       foreign_table_schema,
       foreign_table_name,
       (pg_options_to_table(ftoptions)).option_name::sql_identifier  AS option_name,
       (pg_options_to_table(ftoptions)).option_value::character_data AS option_value
FROM _pg_foreign_tables t;

alter table information_schema.foreign_table_options
    owner to postgres;

grant select on information_schema.foreign_table_options to public;

