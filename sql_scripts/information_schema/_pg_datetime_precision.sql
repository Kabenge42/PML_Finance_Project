CREATE FUNCTION information_schema._pg_datetime_precision(typid oid, typmod integer) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql
RETURN CASE WHEN (typid = (1082)::oid) THEN 0 WHEN (typid = ANY (ARRAY[(1083)::oid, (1114)::oid, (1184)::oid, (1266)::oid])) THEN CASE WHEN (typmod < 0) THEN 6 ELSE typmod END WHEN (typid = (1186)::oid) THEN CASE WHEN ((typmod < 0) OR ((typmod & 65535) = 65535)) THEN 6 ELSE (typmod & 65535) END ELSE NULL::integer END;

ALTER FUNCTION information_schema._pg_datetime_precision(unknown, unknown) OWNER TO postgres;