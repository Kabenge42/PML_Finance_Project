-- Cyclic dependencies found

CREATE FUNCTION pml.ema_crossover_signal(fast_ema double precision, slow_ema double precision) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN fast_ema > slow_ema THEN 1 WHEN fast_ema < slow_ema THEN -1 ELSE 0 END;
$$;

ALTER FUNCTION pml.ema_crossover_signal(unknown, unknown) OWNER TO postgres;

CREATE FUNCTION pml.ema_crossover_signal(fast_ema numeric, slow_ema numeric) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN fast_ema > slow_ema THEN 1 WHEN fast_ema < slow_ema THEN -1 ELSE 0 END;
$$;

ALTER FUNCTION pml.ema_crossover_signal(unknown, unknown) OWNER TO postgres;