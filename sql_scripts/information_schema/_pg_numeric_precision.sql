CREATE FUNCTION information_schema._pg_numeric_precision(typid oid, typmod integer) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql
RETURN CASE typid WHEN 21 THEN 16 WHEN 23 THEN 32 WHEN 20 THEN 64 WHEN 1700 THEN CASE WHEN (typmod = '-1'::integer) THEN NULL::integer ELSE (((typmod - 4) >> 16) & 65535) END WHEN 700 THEN 24 WHEN 701 THEN 53 ELSE NULL::integer END;

ALTER FUNCTION information_schema._pg_numeric_precision(unknown, unknown) OWNER TO postgres;