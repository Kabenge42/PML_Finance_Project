CREATE FUNCTION public.safe_divide(numerator numeric, denominator numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT numerator / NULLIF(denominator, 0) AS result;
$$;

ALTER FUNCTION public.safe_divide(unknown, unknown) OWNER TO postgres;