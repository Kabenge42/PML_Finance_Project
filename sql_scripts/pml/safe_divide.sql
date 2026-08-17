create function safe_divide(numerator numeric, denominator numeric) returns numeric
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

alter function safe_divide(numeric, numeric) owner to postgres
;

create function safe_divide(numerator double precision, denominator double precision) returns double precision
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

alter function safe_divide(double precision, double precision) owner to postgres
;