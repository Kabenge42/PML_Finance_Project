create view information_schema.key_column_usage
			(constraint_catalog, constraint_schema, constraint_name, table_catalog, table_schema, table_name,
			 column_name, ordinal_position, position_in_unique_constraint)
as
-- missing source code
;

alter table information_schema.key_column_usage
	owner to postgres
;

grant select on information_schema.key_column_usage to public
;