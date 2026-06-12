CREATE FUNCTION public.ema_crossover_signal(fast_ema numeric, slow_ema numeric) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.ema_crossover_signal(numeric, numeric) OWNER TO postgres;

CREATE FUNCTION public.ema_crossover_signal(fast_ema double precision, slow_ema double precision) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.ema_crossover_signal(double precision, double precision) OWNER TO postgres;