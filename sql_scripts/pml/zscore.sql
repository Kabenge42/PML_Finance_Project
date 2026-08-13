create function pml.zscore(val numeric, mu numeric, sigma numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT (val - mu) / NULLIF(sigma, 0);
$$
;

alter function pml.zscore(numeric, numeric, numeric) owner to postgres
;

create function pml.zscore(val double precision, mu double precision, sigma double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT (val - mu) / NULLIF(sigma, 0);
$$
;

alter function pml.zscore(double precision, double precision, double precision) owner to postgres
;