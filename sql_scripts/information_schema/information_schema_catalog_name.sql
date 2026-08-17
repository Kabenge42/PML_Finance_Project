create view information_schema.information_schema_catalog_name(catalog_name)
as
-- missing source code
;

alter table information_schema.information_schema_catalog_name
	owner to postgres
;

grant select on information_schema.information_schema_catalog_name to public
;