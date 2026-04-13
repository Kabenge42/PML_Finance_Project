create function calculate_next_fiscal_quarter(next_earnings_date date, income_statement_report_date date,
                                              fy_end_date date,
                                              earnings_report_frequency text DEFAULT 'Quarterly'::text) returns integer
    immutable
    language plpgsql
as
$$
DECLARE
    next_fy_end_date      DATE;
    reference_date        DATE;
    fy_range_months       INTEGER;
    interval_months       INTEGER;
    months_into_fy        INTEGER;
    next_period_end_month INTEGER;
    fiscal_quarter        INTEGER;
BEGIN
    -- Return NULL if essential dates are missing
    IF fy_end_date IS NULL THEN
        RETURN NULL;
    END IF;

    -- Determine the reference date: prefer Next Earnings, fallback to Income Statement + interval
    IF next_earnings_date IS NOT NULL THEN
        reference_date := next_earnings_date;
    ELSIF income_statement_report_date IS NOT NULL THEN
        -- Estimate next report date by adding the reporting interval
        interval_months := CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
                               WHEN 'QUARTERLY' THEN 3
                               WHEN 'SEMI-ANNUALLY' THEN 6
                               WHEN 'SEMI-ANNUAL' THEN 6
                               WHEN 'ANNUALLY' THEN 12
                               WHEN 'ANNUAL' THEN 12
                               ELSE 3
            END;
        reference_date := (income_statement_report_date + (interval_months || ' months')::INTERVAL)::DATE;
    ELSE
        RETURN NULL;
    END IF;

    -- Calculate fiscal year boundaries
    -- Determine which fiscal year the reference_date falls into
    next_fy_end_date := fy_end_date;
    WHILE next_fy_end_date < reference_date
        LOOP
            next_fy_end_date := (next_fy_end_date + INTERVAL '1 year')::DATE;
        END LOOP;

    -- The current FY end for this period is one year before next_fy_end_date
    -- unless reference_date is exactly on or before the original fy_end_date
    IF next_fy_end_date = fy_end_date THEN
        -- Reference date is before/on the first FY end, use it directly
        NULL; -- next_fy_end_date is already correct
    END IF;

    -- Fiscal year range is always 12 months
    fy_range_months := 12;

    -- Calculate months from the START of the fiscal year to the reference date
    -- FY starts the day after the previous FY end
    months_into_fy := (
        (EXTRACT(YEAR FROM reference_date) - EXTRACT(YEAR FROM (next_fy_end_date - INTERVAL '1 year'))) * 12
            + (EXTRACT(MONTH FROM reference_date) - EXTRACT(MONTH FROM (next_fy_end_date - INTERVAL '1 year')))
        )::INTEGER;

    -- Normalize to 1-12 range (months within the fiscal year)
    months_into_fy := ((months_into_fy - 1) % 12) + 1;
    IF months_into_fy <= 0 THEN
        months_into_fy := months_into_fy + 12;
    END IF;

    -- Derive fiscal quarter from the fiscal month
    -- Q1: months 1-3, Q2: months 4-6, Q3: months 7-9, Q4: months 10-12
    fiscal_quarter := CEIL(months_into_fy / 3.0)::INTEGER;

    -- Ensure quarter is within valid range
    IF fiscal_quarter < 1 THEN
        fiscal_quarter := 1;
    ELSIF fiscal_quarter > 4 THEN
        fiscal_quarter := 4;
    END IF;

    RETURN fiscal_quarter;
END;
$$;

alter function calculate_next_fiscal_quarter(date, date, date, text) owner to postgres;

