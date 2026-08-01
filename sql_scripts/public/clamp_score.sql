CREATE FUNCTION public.clamp_score(val numeric, min_val numeric default 0, max_val numeric default 100) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.clamp_score(numeric, numeric, numeric) OWNER TO postgres;