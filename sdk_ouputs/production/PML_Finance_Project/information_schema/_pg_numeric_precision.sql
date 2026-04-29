create function information_schema._pg_numeric_precision(typid oid, typmod integer) returns integer
    immutable
    strict
    parallel safe
    language sql
RETURN CASE typid
           WHEN 21 THEN 16
           WHEN 23 THEN 32
           WHEN 20 THEN 64
           WHEN 1700 THEN CASE WHEN (typmod = '-1'::integer) THEN NULL::integer ELSE (((typmod - 4) >> 16) & 65535) END
           WHEN 700 THEN 24
           WHEN 701 THEN 53
           ELSE NULL::integer END;

alter function information_schema._pg_numeric_precision(unknown, unknown) owner to postgres;

