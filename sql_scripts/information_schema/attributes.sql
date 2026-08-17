create view information_schema.attributes
			(udt_catalog, udt_schema, udt_name, attribute_name, ordinal_position, attribute_default, is_nullable,
			 data_type, character_maximum_length, character_octet_length, character_set_catalog, character_set_schema,
			 character_set_name, collation_catalog, collation_schema, collation_name, numeric_precision,
			 numeric_precision_radix, numeric_scale, datetime_precision, interval_type, interval_precision,
			 attribute_udt_catalog, attribute_udt_schema, attribute_udt_name, scope_catalog, scope_schema, scope_name,
			 maximum_cardinality, dtd_identifier, is_derived_reference_attribute)
as
-- missing source code
;

alter table information_schema.attributes
	owner to postgres
;

grant select on information_schema.attributes to public
;