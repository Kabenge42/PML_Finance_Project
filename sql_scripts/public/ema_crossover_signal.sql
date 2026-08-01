CREATE FUNCTION public.ema_crossover_signal(fast_ema numeric, slow_ema numeric) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.ema_crossover_signal(numeric, numeric) OWNER TO postgres;