CREATE VIEW information_schema.check_constraints(constraint_catalog, constraint_schema, constraint_name, check_clause) AS
SELECT current_database()::information_schema.sql_identifier                               AS constraint_catalog,
       rs.nspname::information_schema.sql_identifier                                       AS constraint_schema,
       con.conname::information_schema.sql_identifier                                      AS constraint_name,
       pg_get_expr(con.conbin, COALESCE(c.oid, 0::oid))::information_schema.character_data AS check_clause
FROM pg_constraint              con
	     LEFT JOIN pg_namespace rs ON rs.oid = con.connamespace
	     LEFT JOIN pg_class     c ON c.oid = con.conrelid
	     LEFT JOIN pg_type      t ON t.oid = con.contypid
WHERE pg_has_role(COALESCE(c.relowner, t.typowner), 'USAGE'::text)
  AND con.contype = 'c'::"char"
UNION ALL
SELECT current_database()::information_schema.sql_identifier                                                  AS constraint_catalog,
       rs.nspname::information_schema.sql_identifier                                                          AS constraint_schema,
       con.conname::information_schema.sql_identifier                                                         AS constraint_name,
       format('%s IS NOT NULL'::text,
              COALESCE(at.attname, 'VALUE'::name))::information_schema.character_data                         AS check_clause
FROM pg_constraint              con
	     LEFT JOIN pg_namespace rs ON rs.oid = con.connamespace
	     LEFT JOIN pg_class     c ON c.oid = con.conrelid
	     LEFT JOIN pg_type      t ON t.oid = con.contypid
	     LEFT JOIN pg_attribute at ON con.conrelid = at.attrelid AND con.conkey[1] = at.attnum
WHERE pg_has_role(COALESCE(c.relowner, t.typowner), 'USAGE'::text)
  AND con.contype = 'n'::"char";

ALTER TABLE information_schema.check_constraints
	OWNER TO postgres;

GRANT SELECT ON information_schema.check_constraints TO PUBLIC;