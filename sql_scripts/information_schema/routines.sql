CREATE VIEW information_schema.routines
			(specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name,
			 routine_type, module_catalog, module_schema, module_name, udt_catalog, udt_schema, udt_name, data_type,
			 character_maximum_length, character_octet_length, character_set_catalog, character_set_schema,
			 character_set_name, collation_catalog, collation_schema, collation_name, numeric_precision,
			 numeric_precision_radix, numeric_scale, datetime_precision, interval_type, interval_precision,
			 type_udt_catalog, type_udt_schema, type_udt_name, scope_catalog, scope_schema, scope_name,
			 maximum_cardinality, dtd_identifier, routine_body, routine_definition, external_name, external_language,
			 parameter_style, is_deterministic, sql_data_access, is_null_call, sql_path, schema_level_routine,
			 max_dynamic_result_sets, is_user_defined_cast, is_implicitly_invocable, security_type,
			 to_sql_specific_catalog, to_sql_specific_schema, to_sql_specific_name, as_locator, created, last_altered,
			 new_savepoint_level, is_udt_dependent, result_cast_from_data_type, result_cast_as_locator,
			 result_cast_char_max_length, result_cast_char_octet_length, result_cast_char_set_catalog,
			 result_cast_char_set_schema, result_cast_char_set_name, result_cast_collation_catalog,
			 result_cast_collation_schema, result_cast_collation_name, result_cast_numeric_precision,
			 result_cast_numeric_precision_radix, result_cast_numeric_scale, result_cast_datetime_precision,
			 result_cast_interval_type, result_cast_interval_precision, result_cast_type_udt_catalog,
			 result_cast_type_udt_schema, result_cast_type_udt_name, result_cast_scope_catalog,
			 result_cast_scope_schema, result_cast_scope_name, result_cast_maximum_cardinality,
			 result_cast_dtd_identifier)
AS
SELECT current_database()::information_schema.sql_identifier                                                    AS specific_catalog,
       n.nspname::information_schema.sql_identifier                                                             AS specific_schema,
       nameconcatoid(p.proname, p.oid)::information_schema.sql_identifier                                       AS specific_name,
       current_database()::information_schema.sql_identifier                                                    AS routine_catalog,
       n.nspname::information_schema.sql_identifier                                                             AS routine_schema,
       p.proname::information_schema.sql_identifier                                                             AS routine_name,
       CASE p.prokind
	       WHEN 'f'::"char" THEN 'FUNCTION'::text
	       WHEN 'p'::"char" THEN 'PROCEDURE'::text
	       ELSE NULL::text END::information_schema.character_data                                               AS routine_type,
       NULL::name::information_schema.sql_identifier                                                            AS module_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS module_schema,
       NULL::name::information_schema.sql_identifier                                                            AS module_name,
       NULL::name::information_schema.sql_identifier                                                            AS udt_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS udt_schema,
       NULL::name::information_schema.sql_identifier                                                            AS udt_name,
       CASE
	       WHEN p.prokind = 'p'::"char" THEN NULL::text
	       WHEN t.typelem <> 0::oid AND t.typlen = '-1'::integer THEN 'ARRAY'::text
	       WHEN nt.nspname = 'pg_catalog'::name THEN format_type(t.oid, NULL::integer)
	       ELSE 'USER-DEFINED'::text END::information_schema.character_data                                     AS data_type,
       NULL::integer::information_schema.cardinal_number                                                        AS character_maximum_length,
       NULL::integer::information_schema.cardinal_number                                                        AS character_octet_length,
       NULL::name::information_schema.sql_identifier                                                            AS character_set_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS character_set_schema,
       NULL::name::information_schema.sql_identifier                                                            AS character_set_name,
       NULL::name::information_schema.sql_identifier                                                            AS collation_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS collation_schema,
       NULL::name::information_schema.sql_identifier                                                            AS collation_name,
       NULL::integer::information_schema.cardinal_number                                                        AS numeric_precision,
       NULL::integer::information_schema.cardinal_number                                                        AS numeric_precision_radix,
       NULL::integer::information_schema.cardinal_number                                                        AS numeric_scale,
       NULL::integer::information_schema.cardinal_number                                                        AS datetime_precision,
       NULL::character varying::information_schema.character_data                                               AS interval_type,
       NULL::integer::information_schema.cardinal_number                                                        AS interval_precision,
       CASE
	       WHEN nt.nspname IS NOT NULL THEN current_database()
	       ELSE NULL::name END::information_schema.sql_identifier                                               AS type_udt_catalog,
       nt.nspname::information_schema.sql_identifier                                                            AS type_udt_schema,
       t.typname::information_schema.sql_identifier                                                             AS type_udt_name,
       NULL::name::information_schema.sql_identifier                                                            AS scope_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS scope_schema,
       NULL::name::information_schema.sql_identifier                                                            AS scope_name,
       NULL::integer::information_schema.cardinal_number                                                        AS maximum_cardinality,
       CASE WHEN p.prokind <> 'p'::"char" THEN 0 ELSE NULL::integer END::information_schema.sql_identifier      AS dtd_identifier,
       CASE
	       WHEN l.lanname = 'sql'::name THEN 'SQL'::text
	       ELSE 'EXTERNAL'::text END::information_schema.character_data                                         AS routine_body,
       CASE
	       WHEN pg_has_role(p.proowner, 'USAGE'::text) THEN p.prosrc
	       ELSE NULL::text END::information_schema.character_data                                               AS routine_definition,
       CASE WHEN l.lanname = 'c'::name THEN p.prosrc ELSE NULL::text END::information_schema.character_data     AS external_name,
       upper(l.lanname::text)::information_schema.character_data                                                AS external_language,
       'GENERAL'::character varying::information_schema.character_data                                          AS parameter_style,
       CASE
	       WHEN p.provolatile = 'i'::"char" THEN 'YES'::text
	       ELSE 'NO'::text END::information_schema.yes_or_no                                                    AS is_deterministic,
       'MODIFIES'::character varying::information_schema.character_data                                         AS sql_data_access,
       CASE
	       WHEN p.prokind <> 'p'::"char" THEN CASE WHEN p.proisstrict THEN 'YES'::text ELSE 'NO'::text END
	       ELSE NULL::text END::information_schema.yes_or_no                                                    AS is_null_call,
       NULL::character varying::information_schema.character_data                                               AS sql_path,
       'YES'::character varying::information_schema.yes_or_no                                                   AS schema_level_routine,
       0::information_schema.cardinal_number                                                                    AS max_dynamic_result_sets,
       NULL::character varying::information_schema.yes_or_no                                                    AS is_user_defined_cast,
       NULL::character varying::information_schema.yes_or_no                                                    AS is_implicitly_invocable,
       CASE WHEN p.prosecdef THEN 'DEFINER'::text ELSE 'INVOKER'::text END::information_schema.character_data   AS security_type,
       NULL::name::information_schema.sql_identifier                                                            AS to_sql_specific_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS to_sql_specific_schema,
       NULL::name::information_schema.sql_identifier                                                            AS to_sql_specific_name,
       'NO'::character varying::information_schema.yes_or_no                                                    AS as_locator,
       NULL::timestamp with time zone::information_schema.time_stamp                                            AS created,
       NULL::timestamp with time zone::information_schema.time_stamp                                            AS last_altered,
       NULL::character varying::information_schema.yes_or_no                                                    AS new_savepoint_level,
       'NO'::character varying::information_schema.yes_or_no                                                    AS is_udt_dependent,
       NULL::character varying::information_schema.character_data                                               AS result_cast_from_data_type,
       NULL::character varying::information_schema.yes_or_no                                                    AS result_cast_as_locator,
       NULL::integer::information_schema.cardinal_number                                                        AS result_cast_char_max_length,
       NULL::integer::information_schema.cardinal_number                                                        AS result_cast_char_octet_length,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_char_set_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_char_set_schema,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_char_set_name,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_collation_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_collation_schema,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_collation_name,
       NULL::integer::information_schema.cardinal_number                                                        AS result_cast_numeric_precision,
       NULL::integer::information_schema.cardinal_number                                                        AS result_cast_numeric_precision_radix,
       NULL::integer::information_schema.cardinal_number                                                        AS result_cast_numeric_scale,
       NULL::integer::information_schema.cardinal_number                                                        AS result_cast_datetime_precision,
       NULL::character varying::information_schema.character_data                                               AS result_cast_interval_type,
       NULL::integer::information_schema.cardinal_number                                                        AS result_cast_interval_precision,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_type_udt_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_type_udt_schema,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_type_udt_name,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_scope_catalog,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_scope_schema,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_scope_name,
       NULL::integer::information_schema.cardinal_number                                                        AS result_cast_maximum_cardinality,
       NULL::name::information_schema.sql_identifier                                                            AS result_cast_dtd_identifier
FROM pg_namespace              n
	     JOIN      pg_proc     p ON n.oid = p.pronamespace
	     JOIN      pg_language l ON p.prolang = l.oid
	     LEFT JOIN (pg_type t JOIN pg_namespace nt ON t.typnamespace = nt.oid)
	               ON p.prorettype = t.oid AND p.prokind <> 'p'::"char"
WHERE pg_has_role(p.proowner, 'USAGE'::text)
   OR has_function_privilege(p.oid, 'EXECUTE'::text);

ALTER TABLE information_schema.routines
	OWNER TO postgres;

GRANT SELECT ON information_schema.routines TO PUBLIC;