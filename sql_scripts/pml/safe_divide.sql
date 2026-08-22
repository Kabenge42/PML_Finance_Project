create function pml.safe_divide(numerator numeric, denominator numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT numerator / NULLIF(denominator, 0) AS result;
$$
;

alter function pml.safe_divide(numeric, numeric) owner to postgres
;

create function pml.safe_divide(numerator double precision, denominator double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT numerator / NULLIF(denominator, 0);
$$
;

alter function pml.safe_divide(double precision, double precision) owner to postgres
;