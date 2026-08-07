CREATE FUNCTION month_abbrev_to_number(month_abbrev unknown) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION month_abbrev_to_number(unknown) OWNER TO postgres;