CREATE FUNCTION public.clamp_score(val numeric, min_val numeric DEFAULT 0, max_val numeric DEFAULT 100) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.clamp_score(numeric, numeric, numeric) OWNER TO postgres;

CREATE FUNCTION public.clamp_score(val     double precision, min_val double precision DEFAULT 0,
                                   max_val double precision DEFAULT 100) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.clamp_score(double precision, double precision, double precision) OWNER TO postgres;