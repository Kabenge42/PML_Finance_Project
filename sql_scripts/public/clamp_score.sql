CREATE FUNCTION public.clamp_score(val numeric, min_val numeric DEFAULT 0, max_val numeric DEFAULT 100) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(min_val, LEAST(max_val, val)) AS result;
$$;

ALTER FUNCTION public.clamp_score(unknown, unknown, unknown) OWNER TO postgres;