create function pml.winsorise(val numeric, lo numeric, hi numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$
;

alter function pml.winsorise(numeric, numeric, numeric) owner to postgres
;

create function pml.winsorise(val double precision, lo double precision, hi double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$
;

alter function pml.winsorise(double precision, double precision, double precision) owner to postgres
;