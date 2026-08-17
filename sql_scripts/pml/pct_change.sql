create function pct_change(current_val numeric, previous_val numeric) returns numeric
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

alter function pct_change(numeric, numeric) owner to postgres
;

create function pct_change(current_val double precision, previous_val double precision) returns double precision
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

alter function pct_change(double precision, double precision) owner to postgres
;