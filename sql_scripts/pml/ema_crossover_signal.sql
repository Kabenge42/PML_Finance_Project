create function ema_crossover_signal(fast_ema numeric, slow_ema numeric) returns integer
	immutable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function ema_crossover_signal(numeric, numeric) owner to postgres
;

create function ema_crossover_signal(fast_ema double precision, slow_ema double precision) returns integer
	immutable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function ema_crossover_signal(double precision, double precision) owner to postgres
;