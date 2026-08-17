create view information_schema.tables
			(table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation,
			 user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed,
			 commit_action)
as
-- missing source code
;

alter table information_schema.tables
	owner to postgres
;

grant select on information_schema.tables to public
;