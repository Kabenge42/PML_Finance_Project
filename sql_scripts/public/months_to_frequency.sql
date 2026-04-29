create function months_to_frequency(interval_months integer) returns text
    immutable
    parallel safe
    language sql
as
$$
SELECT CASE
           WHEN interval_months IS NULL THEN 'Quarterly'
           WHEN interval_months <= 3 THEN 'Quarterly'
           WHEN interval_months <= 6 THEN 'Semi-Annually'
           ELSE 'Annually'
           END
$$;

alter function months_to_frequency(integer) owner to postgres;

