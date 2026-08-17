create view information_schema.check_constraints(constraint_catalog, constraint_schema, constraint_name, check_clause)
as
-- missing source code
;

alter table information_schema.check_constraints
	owner to postgres
;

grant select on information_schema.check_constraints to public
;