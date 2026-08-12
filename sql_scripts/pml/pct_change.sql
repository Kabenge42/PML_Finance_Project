-- Cyclic dependencies found

create function pct_change(current_val double precision, previous_val double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100;
$$
;

alter function pct_change(unknown, unknown) owner to postgres
;

create function pct_change(current_val numeric, previous_val numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100;
$$
;

alter function pct_change(unknown, unknown) owner to postgres
;