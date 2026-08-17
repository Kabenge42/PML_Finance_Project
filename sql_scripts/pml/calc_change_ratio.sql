create function calc_change_ratio(current_val numeric, previous_val numeric) returns numeric
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

alter function calc_change_ratio(numeric, numeric) owner to postgres
;

create function calc_change_ratio(current_val double precision, previous_val double precision) returns double precision
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

alter function calc_change_ratio(double precision, double precision) owner to postgres
;