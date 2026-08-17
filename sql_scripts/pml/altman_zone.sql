create function altman_zone(z numeric) returns integer
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

alter function altman_zone(numeric) owner to postgres
;

create function altman_zone(z double precision) returns integer
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

alter function altman_zone(double precision) owner to postgres
;