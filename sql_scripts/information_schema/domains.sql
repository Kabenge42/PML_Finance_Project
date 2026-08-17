create view information_schema.domains
			(domain_catalog, domain_schema, domain_name, data_type, character_maximum_length, character_octet_length,
			 character_set_catalog, character_set_schema, character_set_name, collation_catalog, collation_schema,
			 collation_name, numeric_precision, numeric_precision_radix, numeric_scale, datetime_precision,
			 interval_type, interval_precision, domain_default, udt_catalog, udt_schema, udt_name, scope_catalog,
			 scope_schema, scope_name, maximum_cardinality, dtd_identifier)
as
-- missing source code
;

alter table information_schema.domains
	owner to postgres
;

grant select on information_schema.domains to public
;