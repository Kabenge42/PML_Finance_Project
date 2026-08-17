create view information_schema.user_mapping_options
			(authorization_identifier, foreign_server_catalog, foreign_server_name, option_name, option_value)
as
-- missing source code
;

alter table information_schema.user_mapping_options
	owner to postgres
;

grant select on information_schema.user_mapping_options to public
;