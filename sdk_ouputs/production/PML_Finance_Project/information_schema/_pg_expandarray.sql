create function information_schema._pg_expandarray(anyarray, OUT x anyelement, OUT n integer) returns SETOF record
    immutable
    strict
    parallel safe
    rows 100
    language sql
as
$$
SELECT *
FROM pg_catalog.unnest($1) WITH ORDINALITY
$$;

alter function information_schema._pg_expandarray(unknown, out unknown, out unknown) owner to postgres;

