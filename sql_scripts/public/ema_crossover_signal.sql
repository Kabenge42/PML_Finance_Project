CREATE FUNCTION public.ema_crossover_signal(fast_ema numeric, slow_ema numeric) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN fast_ema > slow_ema THEN 1 WHEN fast_ema < slow_ema THEN -1 ELSE 0 END AS result;
$$;

ALTER FUNCTION public.ema_crossover_signal(unknown, unknown) OWNER TO postgres;