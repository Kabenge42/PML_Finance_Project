CREATE FUNCTION information_schema._pg_interval_type(typid oid, mod integer) RETURNS text
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION information_schema._pg_interval_type(oid, integer) OWNER TO postgres;