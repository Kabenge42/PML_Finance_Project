create function pct_change(current_val numeric, previous_val numeric) returns numeric
    immutable
    parallel safe
    language sql
as
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100 AS result;
$$;

alter function pct_change(numeric, numeric) owner to postgres;

