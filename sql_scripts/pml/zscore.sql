CREATE FUNCTION zscore(val numeric, mu numeric, sigma numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION zscore(numeric, numeric, numeric) OWNER TO postgres;

CREATE FUNCTION zscore(val double precision, mu double precision, sigma double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION zscore(double precision, double precision, double precision) OWNER TO postgres;