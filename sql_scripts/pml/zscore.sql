-- Cyclic dependencies found

create function zscore(val double precision, mu double precision, sigma double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT (val - mu) / NULLIF(sigma, 0);
$$
;

alter function zscore(unknown, unknown, unknown) owner to postgres
;

create function zscore(val numeric, mu numeric, sigma numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT (val - mu) / NULLIF(sigma, 0);
$$
;

alter function zscore(unknown, unknown, unknown) owner to postgres
;