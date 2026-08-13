create function pml.coef_var(mu numeric, sigma numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT sigma / NULLIF(ABS(mu), 0);
$$
;

alter function pml.coef_var(numeric, numeric) owner to postgres
;

create function pml.coef_var(mu double precision, sigma double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT sigma / NULLIF(ABS(mu), 0);
$$
;

alter function pml.coef_var(double precision, double precision) owner to postgres
;