create view information_schema.table_constraints
			(constraint_catalog, constraint_schema, constraint_name, table_catalog, table_schema, table_name,
			 constraint_type, is_deferrable, initially_deferred, enforced, nulls_distinct)
as
-- missing source code
;

alter table information_schema.table_constraints
	owner to postgres
;

grant select on information_schema.table_constraints to public
;