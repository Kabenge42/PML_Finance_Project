CREATE FUNCTION public.calc_change_ratio(current_val numeric, previous_val numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_change_ratio(numeric, numeric) OWNER TO postgres;