create function pml.safe_logit(p numeric, eps numeric DEFAULT 0.000001) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT LN(GREATEST(eps, LEAST(1 - eps, p)) / (1 - GREATEST(eps, LEAST(1 - eps, p))));
$$
;

alter function pml.safe_logit(numeric, numeric) owner to postgres
;

create function pml.safe_logit(p double precision, eps double precision DEFAULT 0.000001) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT LN(GREATEST(eps, LEAST(1 - eps, p)) / (1 - GREATEST(eps, LEAST(1 - eps, p))));
$$
;

alter function pml.safe_logit(double precision, double precision) owner to postgres
;