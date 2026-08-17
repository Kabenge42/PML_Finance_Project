create function public.clamp_score(val numeric, min_val numeric default 0, max_val numeric default 100) returns numeric
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

alter function public.clamp_score(numeric, numeric, numeric) owner to postgres
;