CREATE FUNCTION information_schema._pg_char_max_length(typid oid, typmod integer) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql
RETURN CASE WHEN (typmod = '-1'::integer) THEN NULL::integer WHEN (typid = ANY (ARRAY[(1042)::oid, (1043)::oid])) THEN (typmod - 4) WHEN (typid = ANY (ARRAY[(1560)::oid, (1562)::oid])) THEN typmod ELSE NULL::integer END;

ALTER FUNCTION information_schema._pg_char_max_length(unknown, unknown) OWNER TO postgres;