create function validate_fiscal_dates(fy_end_date date, report_date date, reference_date date DEFAULT CURRENT_DATE)
    returns TABLE(issue text, severity text)
    immutable
    language plpgsql
as
$$
BEGIN
    IF fy_end_date > reference_date THEN
        RETURN QUERY SELECT 'FY End Date is in the future'::TEXT, 'WARNING'::TEXT;
    END IF;

    -- A report that is MORE THAN a full fiscal year before the FY end
    -- cannot belong to the current fiscal period. Use strict inequality
    -- with make_interval for clarity.
    IF report_date IS NOT NULL AND report_date < (fy_end_date - make_interval(years => 1))::DATE THEN
        RETURN QUERY SELECT 'Report date predates fiscal year'::TEXT, 'ERROR'::TEXT;
    END IF;

    -- Allow at most 1 day of clock skew; anything beyond is a future-date error
    IF report_date IS NOT NULL AND report_date > reference_date + INTERVAL '1 day' THEN
        RETURN QUERY SELECT 'Report date is in the future'::TEXT, 'WARNING'::TEXT;
    END IF;

    -- Idiomatic end-of-month test via DATE_TRUNC
    IF fy_end_date IS NOT NULL
        AND fy_end_date <> (DATE_TRUNC('month', fy_end_date)
            + INTERVAL '1 month - 1 day')::DATE THEN
        RETURN QUERY SELECT 'FY End is not last day of month'::TEXT, 'INFO'::TEXT;
    END IF;
END;
$$;

alter function validate_fiscal_dates(date, date, date) owner to postgres;

