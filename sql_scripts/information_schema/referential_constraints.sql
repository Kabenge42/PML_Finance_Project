create view information_schema.referential_constraints
			(constraint_catalog, constraint_schema, constraint_name, unique_constraint_catalog,
			 unique_constraint_schema, unique_constraint_name, match_option, update_rule, delete_rule)
as
-- missing source code
;

alter table information_schema.referential_constraints
	owner to postgres
;

grant select on information_schema.referential_constraints to public
;