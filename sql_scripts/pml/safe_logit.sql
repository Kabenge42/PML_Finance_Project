create function safe_logit(p numeric, eps numeric default 0.000001) returns numeric
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

alter function safe_logit(numeric, numeric) owner to postgres
;

create function safe_logit(p double precision, eps double precision default 0.000001) returns double precision
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

alter function safe_logit(double precision, double precision) owner to postgres
;