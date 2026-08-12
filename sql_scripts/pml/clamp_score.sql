-- Cyclic dependencies found

create function clamp_score(val double precision, min_val double precision DEFAULT 0, max_val double precision DEFAULT 100) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT GREATEST(min_val, LEAST(max_val, val));
$$
;

alter function clamp_score(unknown, unknown, unknown) owner to postgres
;

create function clamp_score(val numeric, min_val numeric DEFAULT 0, max_val numeric DEFAULT 100) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT GREATEST(min_val, LEAST(max_val, val));
$$
;

alter function clamp_score(unknown, unknown, unknown) owner to postgres
;