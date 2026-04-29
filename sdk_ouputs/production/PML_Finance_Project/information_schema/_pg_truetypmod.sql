create function information_schema._pg_truetypmod(pg_attribute, pg_type) returns integer
    immutable
    strict
    parallel safe
    language sql
RETURN CASE WHEN (($2).typtype = 'd'::"char") THEN ($2).typtypmod ELSE ($1).atttypmod END;

alter function information_schema._pg_truetypmod(unknown, unknown) owner to postgres;

