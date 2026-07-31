CREATE FUNCTION public.calc_change_ratio(current_val numeric, previous_val numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) AS result;
$$;

ALTER FUNCTION public.calc_change_ratio(unknown, unknown) OWNER TO postgres;