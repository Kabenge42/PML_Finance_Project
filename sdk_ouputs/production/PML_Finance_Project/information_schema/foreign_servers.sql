create view information_schema.foreign_servers
            (foreign_server_catalog, foreign_server_name, foreign_data_wrapper_catalog, foreign_data_wrapper_name,
             foreign_server_type, foreign_server_version, authorization_identifier)
as
SELECT foreign_server_catalog,
       foreign_server_name,
       foreign_data_wrapper_catalog,
       foreign_data_wrapper_name,
       foreign_server_type,
       foreign_server_version,
       authorization_identifier
FROM _pg_foreign_servers;

alter table information_schema.foreign_servers
    owner to postgres;

grant select on information_schema.foreign_servers to public;

