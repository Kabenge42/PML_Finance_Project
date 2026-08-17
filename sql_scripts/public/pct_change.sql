create function public.pct_change(current_val numeric, previous_val numeric) returns numeric
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

alter function public.pct_change(numeric, numeric) owner to postgres
;