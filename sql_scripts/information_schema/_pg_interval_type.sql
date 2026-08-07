CREATE FUNCTION information_schema._pg_interval_type(typid oid, mod integer) RETURNS text
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql
RETURN CASE WHEN (typid = (1186)::oid) THEN upper(SUBSTRING(format_type(typid, mod) SIMILAR 'interval[()0-9]* #"%#"'::text ESCAPE '#'::text)) ELSE NULL::text END;

ALTER FUNCTION information_schema._pg_interval_type(unknown, unknown) OWNER TO postgres;