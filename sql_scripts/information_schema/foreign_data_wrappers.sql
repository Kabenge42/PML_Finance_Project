create view information_schema.foreign_data_wrappers
			(foreign_data_wrapper_catalog, foreign_data_wrapper_name, authorization_identifier, library_name,
			 foreign_data_wrapper_language)
as
-- missing source code
;

alter table information_schema.foreign_data_wrappers
	owner to postgres
;

grant select on information_schema.foreign_data_wrappers to public
;