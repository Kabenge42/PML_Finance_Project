create function information_schema._pg_truetypid(pg_attribute, pg_type) returns oid
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

alter function information_schema._pg_truetypid(pg_attribute, pg_type) owner to postgres
;