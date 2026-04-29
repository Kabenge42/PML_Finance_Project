create view information_schema.column_options
            (table_catalog, table_schema, table_name, column_name, option_name, option_value) as
SELECT current_database()::sql_identifier                                AS table_catalog,
       nspname::sql_identifier                                           AS table_schema,
       relname::sql_identifier                                           AS table_name,
       attname::sql_identifier                                           AS column_name,
       (pg_options_to_table(attfdwoptions)).option_name::sql_identifier  AS option_name,
       (pg_options_to_table(attfdwoptions)).option_value::character_data AS option_value
FROM _pg_foreign_table_columns c;

alter table information_schema.column_options
    owner to postgres;

grant select on information_schema.column_options to public;

