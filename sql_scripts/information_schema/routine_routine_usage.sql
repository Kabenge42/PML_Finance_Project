create view information_schema.routine_routine_usage
			(specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name)
as
-- missing source code
;

alter table information_schema.routine_routine_usage
	owner to postgres
;

grant select on information_schema.routine_routine_usage to public
;