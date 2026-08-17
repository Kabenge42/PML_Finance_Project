create function public.ema_crossover_signal(fast_ema numeric, slow_ema numeric) returns integer
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

alter function public.ema_crossover_signal(numeric, numeric) owner to postgres
;