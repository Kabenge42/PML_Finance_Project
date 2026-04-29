create view information_schema.user_mapping_options
            (authorization_identifier, foreign_server_catalog, foreign_server_name, option_name, option_value) as
SELECT um.authorization_identifier,
       um.foreign_server_catalog,
       um.foreign_server_name,
       opts.option_name::sql_identifier AS option_name,
       CASE
           WHEN um.umuser <> 0::oid AND um.authorization_identifier::name = CURRENT_USER OR
                um.umuser = 0::oid AND pg_has_role(um.srvowner::name, 'USAGE'::text) OR (SELECT pg_authid.rolsuper
                                                                                         FROM pg_authid
                                                                                         WHERE pg_authid.rolname = CURRENT_USER)
               THEN opts.option_value
           ELSE NULL::text
           END::character_data          AS option_value
FROM _pg_user_mappings um,
     LATERAL pg_options_to_table(um.umoptions) opts(option_name, option_value);

alter table information_schema.user_mapping_options
    owner to postgres;

grant select on information_schema.user_mapping_options to public;

