CREATE FUNCTION months_to_frequency(interval_months unknown) RETURNS text
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION months_to_frequency(unknown) OWNER TO postgres;