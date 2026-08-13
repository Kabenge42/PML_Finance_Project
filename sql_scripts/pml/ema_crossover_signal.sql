create function pml.ema_crossover_signal(fast_ema numeric, slow_ema numeric) returns integer
	immutable
	parallel safe
	language sql
as
$$
SELECT CASE WHEN fast_ema > slow_ema THEN 1 WHEN fast_ema < slow_ema THEN -1 ELSE 0 END AS result;
$$
;

alter function pml.ema_crossover_signal(numeric, numeric) owner to postgres
;

create function pml.ema_crossover_signal(fast_ema double precision, slow_ema double precision) returns integer
	immutable
	parallel safe
	language sql
as
$$
SELECT CASE WHEN fast_ema > slow_ema THEN 1 WHEN fast_ema < slow_ema THEN -1 ELSE 0 END;
$$
;

alter function pml.ema_crossover_signal(double precision, double precision) owner to postgres
;