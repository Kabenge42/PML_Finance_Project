create view information_schema.transforms
			(udt_catalog, udt_schema, udt_name, specific_catalog, specific_schema, specific_name, group_name,
			 transform_type)
as
-- missing source code
;

alter table information_schema.transforms
	owner to postgres
;