create view information_schema.column_domain_usage
			(domain_catalog, domain_schema, domain_name, table_catalog, table_schema, table_name, column_name)
as
-- missing source code
;

alter table information_schema.column_domain_usage
	owner to postgres
;

grant select on information_schema.column_domain_usage to public
;