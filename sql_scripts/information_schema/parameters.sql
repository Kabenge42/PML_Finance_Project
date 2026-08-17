create view information_schema.parameters
			(specific_catalog, specific_schema, specific_name, ordinal_position, parameter_mode, is_result, as_locator,
			 parameter_name, data_type, character_maximum_length, character_octet_length, character_set_catalog,
			 character_set_schema, character_set_name, collation_catalog, collation_schema, collation_name,
			 numeric_precision, numeric_precision_radix, numeric_scale, datetime_precision, interval_type,
			 interval_precision, udt_catalog, udt_schema, udt_name, scope_catalog, scope_schema, scope_name,
			 maximum_cardinality, dtd_identifier, parameter_default)
as
-- missing source code
;

alter table information_schema.parameters
	owner to postgres
;

grant select on information_schema.parameters to public
;