create view information_schema.check_constraint_routine_usage
			(constraint_catalog, constraint_schema, constraint_name, specific_catalog, specific_schema, specific_name)
as
-- missing source code
;

alter table information_schema.check_constraint_routine_usage
	owner to postgres
;

grant select on information_schema.check_constraint_routine_usage to public
;