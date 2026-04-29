create view information_schema._pg_user_mappings
            (oid, umoptions, umuser, authorization_identifier, foreign_server_catalog, foreign_server_name, srvowner) as
SELECT um.oid,
       um.umoptions,
       um.umuser,
       COALESCE(u.rolname, 'PUBLIC'::name)::sql_identifier AS authorization_identifier,
       s.foreign_server_catalog,
       s.foreign_server_name,
       s.authorization_identifier                          AS srvowner
FROM pg_user_mapping um
         LEFT JOIN pg_authid u ON u.oid = um.umuser,
     _pg_foreign_servers s
WHERE s.oid = um.umserver;

alter table information_schema._pg_user_mappings
    owner to postgres;

