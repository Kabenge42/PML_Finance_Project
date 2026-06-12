CREATE FUNCTION public.pct_change(current_val numeric, previous_val numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.pct_change(numeric, numeric) OWNER TO postgres;

CREATE FUNCTION public.pct_change(current_val double precision, previous_val double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.pct_change(double precision, double precision) OWNER TO postgres;