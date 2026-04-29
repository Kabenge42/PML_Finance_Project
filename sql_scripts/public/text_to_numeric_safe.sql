create function text_to_numeric_safe(input_text text) returns numeric
    immutable
    parallel safe
    language sql
as
$$
SELECT CASE
           WHEN input_text IS NULL
               OR TRIM(input_text) IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE', 'n/a', 'na', 'null', 'none')
               THEN NULL
           WHEN TRIM(input_text) ~ '^-?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
               THEN TRIM(input_text)::NUMERIC
           END AS result
$$;

alter function text_to_numeric_safe(text) owner to postgres;

