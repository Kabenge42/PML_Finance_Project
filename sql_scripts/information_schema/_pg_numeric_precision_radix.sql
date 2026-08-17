create function information_schema._pg_numeric_precision_radix(typid oid, typmod integer) returns integer
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

alter function information_schema._pg_numeric_precision_radix(oid, integer) owner to postgres
;