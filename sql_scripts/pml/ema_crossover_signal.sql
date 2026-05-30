CREATE FUNCTION ema_crossover_signal(fast_ema numeric, slow_ema numeric) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN fast_ema > slow_ema THEN 1 WHEN fast_ema < slow_ema THEN -1 ELSE 0 END AS result;
$$;

ALTER FUNCTION ema_crossover_signal(numeric, numeric) OWNER TO postgres;

CREATE FUNCTION ema_crossover_signal(fast_ema double precision, slow_ema double precision) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN fast_ema > slow_ema THEN 1 WHEN fast_ema < slow_ema THEN -1 ELSE 0 END;
$$;

ALTER FUNCTION ema_crossover_signal(double precision, double precision) OWNER TO postgres;