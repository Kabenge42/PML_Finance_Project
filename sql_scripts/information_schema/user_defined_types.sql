CREATE VIEW information_schema.user_defined_types
			(user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, user_defined_type_category,
			 is_instantiable, is_final, ordering_form, ordering_category, ordering_routine_catalog,
			 ordering_routine_schema, ordering_routine_name, reference_type, data_type, character_maximum_length,
			 character_octet_length, character_set_catalog, character_set_schema, character_set_name, collation_catalog,
			 collation_schema, collation_name, numeric_precision, numeric_precision_radix, numeric_scale,
			 datetime_precision, interval_type, interval_precision, source_dtd_identifier, ref_dtd_identifier)
AS
SELECT current_database()::information_schema.sql_identifier              AS user_defined_type_catalog,
       n.nspname::information_schema.sql_identifier                       AS user_defined_type_schema,
       c.relname::information_schema.sql_identifier                       AS user_defined_type_name,
       'STRUCTURED'::character varying::information_schema.character_data AS user_defined_type_category,
       'YES'::character varying::information_schema.yes_or_no             AS is_instantiable,
       NULL::character varying::information_schema.yes_or_no              AS is_final,
       NULL::character varying::information_schema.character_data         AS ordering_form,
       NULL::character varying::information_schema.character_data         AS ordering_category,
       NULL::name::information_schema.sql_identifier                      AS ordering_routine_catalog,
       NULL::name::information_schema.sql_identifier                      AS ordering_routine_schema,
       NULL::name::information_schema.sql_identifier                      AS ordering_routine_name,
       NULL::character varying::information_schema.character_data         AS reference_type,
       NULL::character varying::information_schema.character_data         AS data_type,
       NULL::integer::information_schema.cardinal_number                  AS character_maximum_length,
       NULL::integer::information_schema.cardinal_number                  AS character_octet_length,
       NULL::name::information_schema.sql_identifier                      AS character_set_catalog,
       NULL::name::information_schema.sql_identifier                      AS character_set_schema,
       NULL::name::information_schema.sql_identifier                      AS character_set_name,
       NULL::name::information_schema.sql_identifier                      AS collation_catalog,
       NULL::name::information_schema.sql_identifier                      AS collation_schema,
       NULL::name::information_schema.sql_identifier                      AS collation_name,
       NULL::integer::information_schema.cardinal_number                  AS numeric_precision,
       NULL::integer::information_schema.cardinal_number                  AS numeric_precision_radix,
       NULL::integer::information_schema.cardinal_number                  AS numeric_scale,
       NULL::integer::information_schema.cardinal_number                  AS datetime_precision,
       NULL::character varying::information_schema.character_data         AS interval_type,
       NULL::integer::information_schema.cardinal_number                  AS interval_precision,
       NULL::name::information_schema.sql_identifier                      AS source_dtd_identifier,
       NULL::name::information_schema.sql_identifier                      AS ref_dtd_identifier
FROM pg_namespace n,
     pg_class     c,
     pg_type      t
WHERE n.oid = c.relnamespace
  AND t.typrelid = c.oid
  AND c.relkind = 'c'::"char"
  AND (pg_has_role(t.typowner, 'USAGE'::text) OR has_type_privilege(t.oid, 'USAGE'::text));

ALTER TABLE information_schema.user_defined_types
	OWNER TO postgres;

GRANT SELECT ON information_schema.user_defined_types TO PUBLIC;