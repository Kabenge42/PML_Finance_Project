create function calculate_next_fy_end_date(fy_end_date date) returns date
    immutable
    strict
    parallel safe
    language sql
as
$$
SELECT (fy_end_date + make_interval(years => 1))::DATE
$$;

alter function calculate_next_fy_end_date(date) owner to postgres;

