create view information_schema.foreign_server_options
            (foreign_server_catalog, foreign_server_name, option_name, option_value) as
SELECT foreign_server_catalog,
       foreign_server_name,
       (pg_options_to_table(srvoptions)).option_name::sql_identifier  AS option_name,
       (pg_options_to_table(srvoptions)).option_value::character_data AS option_value
FROM _pg_foreign_servers s;

alter table information_schema.foreign_server_options
    owner to postgres;

grant select on information_schema.foreign_server_options to public;

