create function information_schema._pg_numeric_scale(typid oid, typmod integer) returns integer
    immutable
    strict
    parallel safe
    language sql
RETURN CASE
           WHEN (typid = ANY (ARRAY [(21)::oid, (23)::oid, (20)::oid])) THEN 0
           WHEN (typid = (1700)::oid) THEN CASE
                                               WHEN (typmod = '-1'::integer) THEN NULL::integer
                                               ELSE ((typmod - 4) & 65535) END
           ELSE NULL::integer END;

alter function information_schema._pg_numeric_scale(unknown, unknown) owner to postgres;

