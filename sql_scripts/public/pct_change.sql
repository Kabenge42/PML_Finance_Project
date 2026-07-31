CREATE FUNCTION public.pct_change(current_val numeric, previous_val numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100 AS result;
$$;

ALTER FUNCTION public.pct_change(unknown, unknown) OWNER TO postgres;