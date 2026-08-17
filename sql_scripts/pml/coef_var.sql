create function coef_var(mu numeric, sigma numeric) returns numeric
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

alter function coef_var(numeric, numeric) owner to postgres
;

create function coef_var(mu double precision, sigma double precision) returns double precision
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

alter function coef_var(double precision, double precision) owner to postgres
;