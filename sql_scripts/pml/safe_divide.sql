-- Cyclic dependencies found

create function safe_divide(numerator double precision, denominator double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT numerator / NULLIF(denominator, 0);
$$
;

alter function safe_divide(unknown, unknown) owner to postgres
;

create function safe_divide(numerator numeric, denominator numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT numerator / NULLIF(denominator, 0);
$$
;

alter function safe_divide(unknown, unknown) owner to postgres
;