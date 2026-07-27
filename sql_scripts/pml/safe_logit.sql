CREATE FUNCTION safe_logit(p numeric, eps numeric default 0.000001) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION safe_logit(numeric, numeric) OWNER TO postgres;

CREATE FUNCTION safe_logit(p double precision, eps double precision default 0.000001) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION safe_logit(double precision, double precision) OWNER TO postgres;