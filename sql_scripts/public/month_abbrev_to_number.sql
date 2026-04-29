create function month_abbrev_to_number(month_abbrev text) returns integer
    immutable
    parallel safe
    language sql
as
$$
SELECT CASE UPPER(LEFT(TRIM(COALESCE(month_abbrev, '')), 3))
           WHEN 'JAN' THEN 1
           WHEN 'FEB' THEN 2
           WHEN 'MAR' THEN 3
           WHEN 'APR' THEN 4
           WHEN 'MAY' THEN 5
           WHEN 'JUN' THEN 6
           WHEN 'JUL' THEN 7
           WHEN 'AUG' THEN 8
           WHEN 'SEP' THEN 9
           WHEN 'OCT' THEN 10
           WHEN 'NOV' THEN 11
           WHEN 'DEC' THEN 12
           END
$$;

alter function month_abbrev_to_number(text) owner to postgres;

