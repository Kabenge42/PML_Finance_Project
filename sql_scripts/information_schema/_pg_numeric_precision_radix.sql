CREATE FUNCTION information_schema._pg_numeric_precision_radix(typid oid, typmod integer) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION information_schema._pg_numeric_precision_radix(oid, integer) OWNER TO postgres;