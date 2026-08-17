create function public.calc_change_ratio(current_val numeric, previous_val numeric) returns numeric
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

alter function public.calc_change_ratio(numeric, numeric) owner to postgres
;