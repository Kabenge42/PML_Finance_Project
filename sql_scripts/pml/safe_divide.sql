-- Cyclic dependencies found

CREATE FUNCTION pml.safe_divide(numerator double precision, denominator double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT numerator / NULLIF(denominator, 0);
$$;

ALTER FUNCTION pml.safe_divide(unknown, unknown) OWNER TO postgres;

CREATE FUNCTION pml.safe_divide(numerator numeric, denominator numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT numerator / NULLIF(denominator, 0);
$$;

ALTER FUNCTION pml.safe_divide(unknown, unknown) OWNER TO postgres;