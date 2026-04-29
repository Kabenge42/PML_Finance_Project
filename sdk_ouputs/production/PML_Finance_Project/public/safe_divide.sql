create function safe_divide(numerator numeric, denominator numeric) returns numeric
    immutable
    parallel safe
    language sql
as
$$
SELECT numerator / NULLIF(denominator, 0) AS result;
$$;

alter function safe_divide(numeric, numeric) owner to postgres;

