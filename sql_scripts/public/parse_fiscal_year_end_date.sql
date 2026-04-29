create function parse_fiscal_year_end_date(fy_end_text text) returns date
    immutable
    strict
    language plpgsql
as
$$
DECLARE
    parts      TEXT[];
    month_num  INT;
    year_value INT;
BEGIN
    IF fy_end_text IS NULL OR TRIM(fy_end_text) = '' THEN
        RETURN NULL;
    END IF;

    parts := regexp_split_to_array(TRIM(fy_end_text), '\s+');
    IF array_length(parts, 1) < 2 OR parts[2] !~ '^\d{4}$' THEN
        RETURN NULL;
    END IF;

    year_value := parts[2]::INT;
    month_num := month_abbrev_to_number(parts[1]);

    IF month_num IS NULL OR year_value NOT BETWEEN 1900 AND 2100 THEN
        RETURN NULL;
    END IF;

    -- Last day of month via single interval literal date-math idiom
    RETURN (MAKE_DATE(year_value, month_num, 1)
        + INTERVAL '1 month - 1 day')::DATE;
END;
$$;

alter function parse_fiscal_year_end_date(text) owner to postgres;

