create view information_schema.foreign_data_wrappers
            (foreign_data_wrapper_catalog, foreign_data_wrapper_name, authorization_identifier, library_name,
             foreign_data_wrapper_language)
as
SELECT foreign_data_wrapper_catalog,
       foreign_data_wrapper_name,
       authorization_identifier,
       NULL::character varying::character_data AS library_name,
       foreign_data_wrapper_language
FROM _pg_foreign_data_wrappers w;

alter table information_schema.foreign_data_wrappers
    owner to postgres;

grant select on information_schema.foreign_data_wrappers to public;

