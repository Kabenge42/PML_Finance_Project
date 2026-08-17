create function winsorise(val numeric, lo numeric, hi numeric) returns numeric
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

alter function winsorise(numeric, numeric, numeric) owner to postgres
;

create function winsorise(val double precision, lo double precision, hi double precision) returns double precision
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

alter function winsorise(double precision, double precision, double precision) owner to postgres
;