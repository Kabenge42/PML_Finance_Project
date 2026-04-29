create function information_schema._pg_interval_type(typid oid, mod integer) returns text
    immutable
    strict
    parallel safe
    language sql
RETURN CASE
           WHEN (typid = (1186)::oid) THEN upper(SUBSTRING(format_type(typid, mod) SIMILAR 'interval[()0-9]* #"%#"'::text
                                                           ESCAPE '#'::text))
           ELSE NULL::text END;

alter function information_schema._pg_interval_type(unknown, unknown) owner to postgres;

