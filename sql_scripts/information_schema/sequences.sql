CREATE VIEW information_schema.sequences
			(sequence_catalog, sequence_schema, sequence_name, data_type, numeric_precision, numeric_precision_radix,
			 numeric_scale, start_value, minimum_value, maximum_value, increment, cycle_option)
AS
SELECT current_database()::information_schema.sql_identifier                                                   AS sequence_catalog,
       nc.nspname::information_schema.sql_identifier                                                           AS sequence_schema,
       c.relname::information_schema.sql_identifier                                                            AS sequence_name,
       format_type(s.seqtypid, NULL::integer)::information_schema.character_data                               AS data_type,
       information_schema._pg_numeric_precision(s.seqtypid, '-1'::integer)::information_schema.cardinal_number AS numeric_precision,
       2::information_schema.cardinal_number                                                                   AS numeric_precision_radix,
       0::information_schema.cardinal_number                                                                   AS numeric_scale,
       s.seqstart::information_schema.character_data                                                           AS start_value,
       s.seqmin::information_schema.character_data                                                             AS minimum_value,
       s.seqmax::information_schema.character_data                                                             AS maximum_value,
       s.seqincrement::information_schema.character_data                                                       AS increment,
       CASE WHEN s.seqcycle THEN 'YES'::text ELSE 'NO'::text END::information_schema.yes_or_no                 AS cycle_option
FROM pg_namespace nc,
     pg_class     c,
     pg_sequence  s
WHERE c.relnamespace = nc.oid
  AND c.relkind = 'S'::"char"
  AND NOT (EXISTS (SELECT 1
                   FROM pg_depend
                   WHERE pg_depend.classid = 'pg_class'::regclass::oid
	                 AND pg_depend.objid = c.oid
	                 AND pg_depend.deptype = 'i'::"char"
                  ))
  AND NOT pg_is_other_temp_schema(nc.oid)
  AND c.oid = s.seqrelid
  AND (pg_has_role(c.relowner, 'USAGE'::text) OR has_sequence_privilege(c.oid, 'SELECT, UPDATE, USAGE'::text));

ALTER TABLE information_schema.sequences
	OWNER TO postgres;

GRANT SELECT ON information_schema.sequences TO PUBLIC;