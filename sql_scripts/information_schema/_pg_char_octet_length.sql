CREATE FUNCTION information_schema._pg_char_octet_length(typid oid, typmod integer) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION information_schema._pg_char_octet_length(oid, integer) OWNER TO postgres;