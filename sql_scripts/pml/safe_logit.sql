-- Cyclic dependencies found

create function safe_logit(p double precision, eps double precision DEFAULT 0.000001) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT LN(GREATEST(eps, LEAST(1 - eps, p)) / (1 - GREATEST(eps, LEAST(1 - eps, p))));
$$
;

alter function safe_logit(unknown, unknown) owner to postgres
;

create function safe_logit(p numeric, eps numeric DEFAULT 0.000001) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT LN(GREATEST(eps, LEAST(1 - eps, p)) / (1 - GREATEST(eps, LEAST(1 - eps, p))));
$$
;

alter function safe_logit(unknown, unknown) owner to postgres
;