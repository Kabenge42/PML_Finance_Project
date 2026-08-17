create view information_schema.domain_udt_usage
			(udt_catalog, udt_schema, udt_name, domain_catalog, domain_schema, domain_name)
as
-- missing source code
;

alter table information_schema.domain_udt_usage
	owner to postgres
;

grant select on information_schema.domain_udt_usage to public
;