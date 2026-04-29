create function information_schema._pg_numeric_precision_radix(typid oid, typmod integer) returns integer
    immutable
    strict
    parallel safe
    language sql
RETURN CASE
           WHEN (typid = ANY (ARRAY [(21)::oid, (23)::oid, (20)::oid, (700)::oid, (701)::oid])) THEN 2
           WHEN (typid = (1700)::oid) THEN 10
           ELSE NULL::integer END;

alter function information_schema._pg_numeric_precision_radix(unknown, unknown) owner to postgres;

