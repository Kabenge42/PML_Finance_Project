create view information_schema.domain_constraints
			(constraint_catalog, constraint_schema, constraint_name, domain_catalog, domain_schema, domain_name,
			 is_deferrable, initially_deferred)
as
-- missing source code
;

alter table information_schema.domain_constraints
	owner to postgres
;

grant select on information_schema.domain_constraints to public
;