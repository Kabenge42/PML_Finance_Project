CREATE FUNCTION clamp_score(val numeric, min_val numeric default 0, max_val numeric default 100) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION clamp_score(numeric, numeric, numeric) OWNER TO postgres;

CREATE FUNCTION clamp_score(val double precision, min_val double precision default 0, max_val double precision default 100) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION clamp_score(double precision, double precision, double precision) OWNER TO postgres;