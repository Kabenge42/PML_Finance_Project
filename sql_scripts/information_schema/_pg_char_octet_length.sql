CREATE FUNCTION information_schema._pg_char_octet_length(typid oid, typmod integer) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql
RETURN CASE WHEN (typid = ANY (ARRAY[(25)::oid, (1042)::oid, (1043)::oid])) THEN CASE WHEN (typmod = '-1'::integer) THEN (((2)::double precision ^ (30)::double precision))::integer ELSE (information_schema._pg_char_max_length(typid, typmod) * pg_encoding_max_length((SELECT pg_database.encoding FROM pg_database WHERE (pg_database.datname = current_database())))) END ELSE NULL::integer END;

ALTER FUNCTION information_schema._pg_char_octet_length(unknown, unknown) OWNER TO postgres;