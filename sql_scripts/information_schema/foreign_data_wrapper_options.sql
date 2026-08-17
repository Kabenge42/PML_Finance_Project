create view information_schema.foreign_data_wrapper_options
			(foreign_data_wrapper_catalog, foreign_data_wrapper_name, option_name, option_value)
as
-- missing source code
;

alter table information_schema.foreign_data_wrapper_options
	owner to postgres
;

grant select on information_schema.foreign_data_wrapper_options to public
;