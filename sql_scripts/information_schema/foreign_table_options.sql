create view information_schema.foreign_table_options
			(foreign_table_catalog, foreign_table_schema, foreign_table_name, option_name, option_value)
as
-- missing source code
;

alter table information_schema.foreign_table_options
	owner to postgres
;

grant select on information_schema.foreign_table_options to public
;