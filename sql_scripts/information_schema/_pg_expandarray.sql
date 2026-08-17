create function information_schema._pg_expandarray(anyarray, out x anyelement, out n integer) returns setof record
	immutable
	strict
	parallel safe
	rows 100
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function information_schema._pg_expandarray(anyarray, out anyelement, out integer) owner to postgres
;