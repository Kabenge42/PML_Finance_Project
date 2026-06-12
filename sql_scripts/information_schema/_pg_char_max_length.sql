CREATE FUNCTION information_schema._pg_char_max_length(typid oid, typmod integer) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION information_schema._pg_char_max_length(oid, integer) OWNER TO postgres;