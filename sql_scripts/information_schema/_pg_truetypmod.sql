CREATE FUNCTION information_schema._pg_truetypmod(pg_attribute, pg_type) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql
RETURN CASE WHEN (($2).typtype = 'd'::"char") THEN ($2).typtypmod ELSE ($1).atttypmod END;

ALTER FUNCTION information_schema._pg_truetypmod(unknown, unknown) OWNER TO postgres;