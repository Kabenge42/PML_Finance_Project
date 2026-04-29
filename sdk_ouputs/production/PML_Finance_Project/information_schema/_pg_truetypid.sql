create function information_schema._pg_truetypid(pg_attribute, pg_type) returns oid
    immutable
    strict
    parallel safe
    language sql
RETURN CASE WHEN (($2).typtype = 'd'::"char") THEN ($2).typbasetype ELSE ($1).atttypid END;

alter function information_schema._pg_truetypid(unknown, unknown) owner to postgres;

