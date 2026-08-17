create view information_schema.user_mappings(authorization_identifier, foreign_server_catalog, foreign_server_name)
as
-- missing source code
;

alter table information_schema.user_mappings
	owner to postgres
;

grant select on information_schema.user_mappings to public
;