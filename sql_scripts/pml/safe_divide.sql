-- Cyclic dependencies found

CREATE FUNCTION safe_divide(numerator double precision, denominator double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT numerator / NULLIF(denominator, 0);
$$;

ALTER FUNCTION safe_divide(unknown, unknown) OWNER TO postgres;

CREATE FUNCTION safe_divide(numerator numeric, denominator numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT numerator / NULLIF(denominator, 0);
$$;

ALTER FUNCTION safe_divide(unknown, unknown) OWNER TO postgres;