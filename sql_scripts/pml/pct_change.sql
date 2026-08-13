create function pml.pct_change(current_val numeric, previous_val numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100 AS result;
$$
;

alter function pml.pct_change(numeric, numeric) owner to postgres
;

create function pml.pct_change(current_val double precision, previous_val double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100;
$$
;

alter function pml.pct_change(double precision, double precision) owner to postgres
;