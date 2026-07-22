CREATE FUNCTION information_schema._pg_expandarray(anyarray, out x anyelement, out n integer) RETURNS setof record
	IMMUTABLE STRICT PARALLEL SAFE ROWS 100
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION information_schema._pg_expandarray(anyarray, OUT anyelement, OUT integer) OWNER TO postgres;