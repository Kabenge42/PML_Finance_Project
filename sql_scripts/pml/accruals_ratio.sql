-- Cyclic dependencies found

create function accruals_ratio(ni double precision, cfo double precision, scale double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT pml.safe_divide(ni - cfo, NULLIF(scale, 0));
$$
;

alter function accruals_ratio(unknown, unknown, unknown) owner to postgres
;

create function accruals_ratio(ni numeric, cfo numeric, scale numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT pml.safe_divide(ni - cfo, NULLIF(scale, 0));
$$
;

alter function accruals_ratio(unknown, unknown, unknown) owner to postgres
;