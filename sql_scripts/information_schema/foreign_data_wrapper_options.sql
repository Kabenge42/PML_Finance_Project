create view information_schema.foreign_data_wrapper_options
            (foreign_data_wrapper_catalog, foreign_data_wrapper_name, option_name, option_value) as
SELECT foreign_data_wrapper_catalog,
       foreign_data_wrapper_name,
       (pg_options_to_table(fdwoptions)).option_name::sql_identifier  AS option_name,
       (pg_options_to_table(fdwoptions)).option_value::character_data AS option_value
FROM _pg_foreign_data_wrappers w;

alter table information_schema.foreign_data_wrapper_options
    owner to postgres;

grant select on information_schema.foreign_data_wrapper_options to public;

