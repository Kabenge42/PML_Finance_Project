CREATE FUNCTION accruals_ratio(ni numeric, cfo numeric, scale numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION accruals_ratio(numeric, numeric, numeric) OWNER TO postgres;

CREATE FUNCTION accruals_ratio(ni double precision, cfo double precision, scale double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION accruals_ratio(double precision, double precision, double precision) OWNER TO postgres;