create function pml.clamp_score(val numeric, min_val numeric DEFAULT 0, max_val numeric DEFAULT 100) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT GREATEST(min_val, LEAST(max_val, val)) AS result;
$$
;

alter function pml.clamp_score(numeric, numeric, numeric) owner to postgres
;

create function pml.clamp_score(val double precision, min_val double precision DEFAULT 0, max_val double precision DEFAULT 100) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT GREATEST(min_val, LEAST(max_val, val));
$$
;

alter function pml.clamp_score(double precision, double precision, double precision) owner to postgres
;