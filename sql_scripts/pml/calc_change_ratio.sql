create function pml.calc_change_ratio(current_val numeric, previous_val numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) AS result;
$$
;

alter function pml.calc_change_ratio(numeric, numeric) owner to postgres
;

create function pml.calc_change_ratio(current_val double precision, previous_val double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0);
$$
;

alter function pml.calc_change_ratio(double precision, double precision) owner to postgres
;