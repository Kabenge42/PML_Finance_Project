CREATE FUNCTION information_schema._pg_truetypmod(pg_attribute, pg_type) RETURNS integer
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION information_schema._pg_truetypmod(pg_attribute, pg_type) OWNER TO postgres;