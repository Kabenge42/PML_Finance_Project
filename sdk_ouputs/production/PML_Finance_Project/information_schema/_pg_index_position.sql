create function information_schema._pg_index_position(oid, smallint) returns integer
    stable
    strict
    language sql
BEGIN
    ATOMIC
    SELECT (ss.a).n AS n
    FROM (SELECT _pg_expandarray(pg_index.indkey) AS a
          FROM pg_index
          WHERE (pg_index.indexrelid = $1)) ss
    WHERE ((ss.a).x = $2);
END;

alter function information_schema._pg_index_position(unknown, unknown) owner to postgres;

