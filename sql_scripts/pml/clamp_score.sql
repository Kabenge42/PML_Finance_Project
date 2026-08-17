create function clamp_score(val numeric, min_val numeric default 0, max_val numeric default 100) returns numeric
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

alter function clamp_score(numeric, numeric, numeric) owner to postgres
;

create function clamp_score(val double precision, min_val double precision default 0, max_val double precision default 100) returns double precision
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

alter function clamp_score(double precision, double precision, double precision) owner to postgres
;