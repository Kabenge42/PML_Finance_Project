CREATE VIEW information_schema.element_types
			(object_catalog, object_schema, object_name, object_type, collection_type_identifier, data_type,
			 character_maximum_length, character_octet_length, character_set_catalog, character_set_schema,
			 character_set_name, collation_catalog, collation_schema, collation_name, numeric_precision,
			 numeric_precision_radix, numeric_scale, datetime_precision, interval_type, interval_precision, udt_catalog,
			 udt_schema, udt_name, scope_catalog, scope_schema, scope_name, maximum_cardinality, dtd_identifier)
AS
-- missing source code
;

ALTER TABLE information_schema.element_types
	OWNER TO postgres;

GRANT SELECT ON information_schema.element_types TO PUBLIC;