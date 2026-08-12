-- Cyclic dependencies found

create function coef_var(mu double precision, sigma double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT sigma / NULLIF(ABS(mu), 0);
$$
;

alter function coef_var(unknown, unknown) owner to postgres
;

create function coef_var(mu numeric, sigma numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT sigma / NULLIF(ABS(mu), 0);
$$
;

alter function coef_var(unknown, unknown) owner to postgres
;