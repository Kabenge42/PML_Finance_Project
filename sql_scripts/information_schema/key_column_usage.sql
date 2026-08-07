CREATE VIEW information_schema.key_column_usage
			(constraint_catalog, constraint_schema, constraint_name, table_catalog, table_schema, table_name,
			 column_name, ordinal_position, position_in_unique_constraint)
AS
SELECT current_database()::information_schema.sql_identifier          AS constraint_catalog,
       ss.nc_nspname::information_schema.sql_identifier               AS constraint_schema,
       ss.conname::information_schema.sql_identifier                  AS constraint_name,
       current_database()::information_schema.sql_identifier          AS table_catalog,
       ss.nr_nspname::information_schema.sql_identifier               AS table_schema,
       ss.relname::information_schema.sql_identifier                  AS table_name,
       a.attname::information_schema.sql_identifier                   AS column_name,
       (ss.x).n::information_schema.cardinal_number                   AS ordinal_position,
       CASE
	       WHEN ss.contype = 'f'::"char" THEN information_schema._pg_index_position(ss.conindid, ss.confkey[(ss.x).n])
	       ELSE NULL::integer END::information_schema.cardinal_number AS position_in_unique_constraint
FROM pg_attribute a,
     (SELECT r.oid                                        AS roid,
             r.relname,
             r.relowner,
             nc.nspname                                   AS nc_nspname,
             nr.nspname                                   AS nr_nspname,
             c.oid                                        AS coid,
             c.conname,
             c.contype,
             c.conindid,
             c.confkey,
             c.confrelid,
             information_schema._pg_expandarray(c.conkey) AS x
      FROM pg_namespace  nr,
           pg_class      r,
           pg_namespace  nc,
           pg_constraint c
      WHERE nr.oid = r.relnamespace
	    AND r.oid = c.conrelid
	    AND nc.oid = c.connamespace
	    AND (c.contype = ANY (ARRAY ['p'::"char", 'u'::"char", 'f'::"char"]))
	    AND (r.relkind = ANY (ARRAY ['r'::"char", 'p'::"char"]))
	    AND NOT pg_is_other_temp_schema(nr.oid)
     )            ss
WHERE ss.roid = a.attrelid
  AND a.attnum = (ss.x).x
  AND NOT a.attisdropped
  AND (pg_has_role(ss.relowner, 'USAGE'::text) OR
       has_column_privilege(ss.roid, a.attnum, 'SELECT, INSERT, UPDATE, REFERENCES'::text));

ALTER TABLE information_schema.key_column_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.key_column_usage TO PUBLIC;