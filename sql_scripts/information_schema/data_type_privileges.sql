create view information_schema.data_type_privileges
			(object_catalog, object_schema, object_name, object_type, dtd_identifier)
as
-- missing source code
;

alter table information_schema.data_type_privileges
	owner to postgres
;

grant select on information_schema.data_type_privileges to public
;