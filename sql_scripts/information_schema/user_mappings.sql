create view information_schema.user_mappings(authorization_identifier, foreign_server_catalog, foreign_server_name) as
SELECT authorization_identifier,
       foreign_server_catalog,
       foreign_server_name
FROM _pg_user_mappings;

alter table information_schema.user_mappings
    owner to postgres;

grant select on information_schema.user_mappings to public;

