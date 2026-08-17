create function information_schema._pg_truetypmod(pg_attribute, pg_type) returns integer
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

alter function information_schema._pg_truetypmod(pg_attribute, pg_type) owner to postgres
;