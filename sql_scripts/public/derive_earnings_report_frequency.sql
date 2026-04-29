create function derive_earnings_report_frequency(income_statement_report_date date, fy_end_date date) returns text
    immutable
    language plpgsql
as
$$
DECLARE
    months_diff INT;
BEGIN
    IF income_statement_report_date IS NULL OR fy_end_date IS NULL THEN
        RETURN 'Quarterly';
    END IF;

    -- AGE() handles direction & year wrap automatically; it also
    -- respects the DAY component, unlike raw EXTRACT() subtraction.
    months_diff := ABS(
            (DATE_PART('year', AGE(income_statement_report_date, fy_end_date)) * 12
                + DATE_PART('month', AGE(income_statement_report_date, fy_end_date)))::INT
                   );

    -- Normalize within a 12-month window, but treat exact FY-end (0) as Annually
    -- rather than conflating it with Semi-Annually.
    IF months_diff = 0 THEN
        RETURN 'Annually';
    END IF;

    months_diff := months_diff % 12;
    IF months_diff = 0 THEN
        months_diff := 12;
    END IF;

    RETURN CASE
               WHEN months_diff = 12 THEN 'Annually'
               WHEN months_diff = 6 THEN 'Semi-Annually'
               ELSE 'Quarterly'
        END;
END;
$$;

alter function derive_earnings_report_frequency(date, date) owner to postgres;

