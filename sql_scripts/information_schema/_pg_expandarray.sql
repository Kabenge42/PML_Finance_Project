CREATE FUNCTION information_schema._pg_expandarray(anyarray, OUT x anyelement, OUT n integer) RETURNS SETOF record
	IMMUTABLE STRICT PARALLEL SAFE ROWS 100
	LANGUAGE sql AS
$$SELECT * FROM pg_catalog.unnest($1) WITH ORDINALITY$$;

ALTER FUNCTION information_schema._pg_expandarray(unknown, OUT unknown, OUT unknown) OWNER TO postgres;