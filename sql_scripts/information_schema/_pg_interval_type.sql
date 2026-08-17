create function information_schema._pg_interval_type(typid oid, mod integer) returns text
	immutable
	strict
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function information_schema._pg_interval_type(oid, integer) owner to postgres
;