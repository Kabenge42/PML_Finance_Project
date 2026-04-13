create function frequency_to_months(earnings_report_frequency text, fy_end_date date DEFAULT NULL::date,
                                    next_fy_end_date date DEFAULT NULL::date) returns integer
    immutable
    language plpgsql
as
$$
DECLARE
    fy_range_months INTEGER;
BEGIN
    -- Calculate the fiscal year range in months (should always be 12)
    IF fy_end_date IS NOT NULL AND next_fy_end_date IS NOT NULL THEN
        fy_range_months := ((EXTRACT(YEAR FROM next_fy_end_date) - EXTRACT(YEAR FROM fy_end_date)) * 12
            + (EXTRACT(MONTH FROM next_fy_end_date) - EXTRACT(MONTH FROM fy_end_date)))::INTEGER;
    ELSE
        -- Default to standard 12-month fiscal year
        fy_range_months := 12;
    END IF;

    -- Derive reporting interval as a divisor of the fiscal year range
    RETURN CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
        -- Quarterly: FY range / 4 reporting periods
               WHEN 'QUARTERLY' THEN fy_range_months / 4
        -- Semi-Annual: FY range / 2 reporting periods
               WHEN 'SEMI-ANNUALLY' THEN fy_range_months / 2
               WHEN 'SEMI-ANNUAL' THEN fy_range_months / 2
        -- Annual: Full FY range (1 reporting period)
               WHEN 'ANNUALLY' THEN fy_range_months
               WHEN 'ANNUAL' THEN fy_range_months
        -- Default to quarterly (FY range / 4)
               ELSE fy_range_months / 4
        END;
END;
$$;

alter function frequency_to_months(text, date, date) owner to postgres;

