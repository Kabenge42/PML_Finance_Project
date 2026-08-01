CREATE FUNCTION public.safe_divide(numerator numeric, denominator numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.safe_divide(numeric, numeric) OWNER TO postgres;