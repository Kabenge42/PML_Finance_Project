CREATE FUNCTION altman_zone(z numeric) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION altman_zone(numeric) OWNER TO postgres;

CREATE FUNCTION altman_zone(z double precision) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION altman_zone(double precision) OWNER TO postgres;