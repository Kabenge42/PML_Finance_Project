create function text_to_date_safe(input_text text, date_format text DEFAULT 'YYYY-MM-DD'::text) returns date
    immutable
    strict
    language plpgsql
as
$$
BEGIN
    IF input_text IS NULL OR TRIM(input_text) IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE') THEN
        RETURN NULL;
    END IF;
    RETURN TO_DATE(TRIM(input_text), date_format);
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$;

alter function text_to_date_safe(text, text) owner to postgres;

