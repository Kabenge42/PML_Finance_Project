CREATE FUNCTION information_schema._pg_truetypid(pg_attribute, pg_type) RETURNS oid
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION information_schema._pg_truetypid(pg_attribute, pg_type) OWNER TO postgres;