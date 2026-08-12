-- Cyclic dependencies found

create function winsorise(val double precision, lo double precision, hi double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$
;

alter function winsorise(unknown, unknown, unknown) owner to postgres
;

create function winsorise(val numeric, lo numeric, hi numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$
;

alter function winsorise(unknown, unknown, unknown) owner to postgres
;