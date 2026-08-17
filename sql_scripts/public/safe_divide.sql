create function public.safe_divide(numerator numeric, denominator numeric) returns numeric
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

alter function public.safe_divide(numeric, numeric) owner to postgres
;