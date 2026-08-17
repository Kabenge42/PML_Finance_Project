create view information_schema.collations(collation_catalog, collation_schema, collation_name, pad_attribute)
as
-- missing source code
;

alter table information_schema.collations
	owner to postgres
;

grant select on information_schema.collations to public
;