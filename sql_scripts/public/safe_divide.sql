CREATE FUNCTION public.safe_divide(numerator numeric, denominator numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.safe_divide(numeric, numeric) OWNER TO postgres;

CREATE FUNCTION public.safe_divide(numerator double precision, denominator double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.safe_divide(double precision, double precision) OWNER TO postgres;