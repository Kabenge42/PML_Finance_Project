create view information_schema.views
			(table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into,
			 is_trigger_updatable, is_trigger_deletable, is_trigger_insertable_into)
as
-- missing source code
;

alter table information_schema.views
	owner to postgres
;

grant select on information_schema.views to public
;