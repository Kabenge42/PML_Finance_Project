create function information_schema._pg_char_max_length(typid oid, typmod integer) returns integer
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

alter function information_schema._pg_char_max_length(oid, integer) owner to postgres
;