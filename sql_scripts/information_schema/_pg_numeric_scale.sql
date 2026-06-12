CREATE FUNCTION information_schema._pg_numeric_scale(typid oid, typmod integer) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION information_schema._pg_numeric_scale(oid, integer) OWNER TO postgres;