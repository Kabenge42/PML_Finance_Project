CREATE FUNCTION information_schema._pg_truetypid(pg_attribute, pg_type) RETURNS oid
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql
RETURN CASE WHEN (($2).typtype = 'd'::"char") THEN ($2).typbasetype ELSE ($1).atttypid END;

ALTER FUNCTION information_schema._pg_truetypid(unknown, unknown) OWNER TO postgres;