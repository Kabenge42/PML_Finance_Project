CREATE FUNCTION coef_var(mu numeric, sigma numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT sigma / NULLIF(ABS(mu), 0);
$$;

ALTER FUNCTION coef_var(numeric, numeric) OWNER TO postgres;

CREATE FUNCTION coef_var(mu double precision, sigma double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT sigma / NULLIF(ABS(mu), 0);
$$;

ALTER FUNCTION coef_var(double precision, double precision) OWNER TO postgres;