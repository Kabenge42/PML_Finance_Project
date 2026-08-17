create function zscore(val numeric, mu numeric, sigma numeric) returns numeric
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

alter function zscore(numeric, numeric, numeric) owner to postgres
;

create function zscore(val double precision, mu double precision, sigma double precision) returns double precision
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

alter function zscore(double precision, double precision, double precision) owner to postgres
;