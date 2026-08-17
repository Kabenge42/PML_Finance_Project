create function accruals_ratio(ni numeric, cfo numeric, scale numeric) returns numeric
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

alter function accruals_ratio(numeric, numeric, numeric) owner to postgres
;

create function accruals_ratio(ni double precision, cfo double precision, scale double precision) returns double precision
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

alter function accruals_ratio(double precision, double precision, double precision) owner to postgres
;