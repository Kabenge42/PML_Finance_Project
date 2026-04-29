create function clamp_score(val numeric, min_val numeric DEFAULT 0, max_val numeric DEFAULT 100) returns numeric
    immutable
    parallel safe
    language sql
as
$$
SELECT GREATEST(min_val, LEAST(max_val, val)) AS result;
$$;

alter function clamp_score(numeric, numeric, numeric) owner to postgres;

