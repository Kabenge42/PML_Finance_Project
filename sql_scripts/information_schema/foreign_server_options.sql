create view information_schema.foreign_server_options
			(foreign_server_catalog, foreign_server_name, option_name, option_value)
as
-- missing source code
;

alter table information_schema.foreign_server_options
	owner to postgres
;

grant select on information_schema.foreign_server_options to public
;