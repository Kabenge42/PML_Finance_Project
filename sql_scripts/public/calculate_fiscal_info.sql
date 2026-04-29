create function calculate_fiscal_info(reference_date date, fy_end_date date, input_earnings_frequency text DEFAULT NULL::text, OUT fiscal_month integer, OUT fiscal_quarter integer, OUT fiscal_year integer, OUT next_quarter integer, OUT next_quarter_year integer, OUT reporting_interval integer, OUT earnings_report_frequency text, OUT next_earnings_report_type text) returns record
    immutable
    language plpgsql
as
$$
DECLARE
    next_fy_end_date    DATE;
    fy_range_months     INTEGER;
    months_since_fy_end INTEGER;
    interval_months     INTEGER;
    periods_per_year    INTEGER;
    current_period      INTEGER;
    next_period         INTEGER;
BEGIN
    IF reference_date IS NULL OR fy_end_date IS NULL THEN
        RETURN;
    END IF;

    -- Use make_interval + AGE() for date-math-correct boundaries
    next_fy_end_date := (fy_end_date + make_interval(years => 1))::DATE;

    -- Fiscal year range in months via AGE() (respects day component)
    fy_range_months := (DATE_PART('year', AGE(next_fy_end_date, fy_end_date)) * 12
        + DATE_PART('month', AGE(next_fy_end_date, fy_end_date)))::INTEGER;

    -- Determine earnings frequency
    earnings_report_frequency := COALESCE(NULLIF(TRIM(input_earnings_frequency), ''),
                                          derive_earnings_report_frequency(reference_date, fy_end_date));

    -- Derive interval months based on FY range
    interval_months := CASE UPPER(TRIM(earnings_report_frequency))
                           WHEN 'QUARTERLY' THEN fy_range_months / 4
                           WHEN 'SEMI-ANNUALLY' THEN fy_range_months / 2
                           WHEN 'SEMI-ANNUAL' THEN fy_range_months / 2
                           WHEN 'ANNUALLY' THEN fy_range_months
                           WHEN 'ANNUAL' THEN fy_range_months
                           ELSE fy_range_months / 4
        END;

    reporting_interval := interval_months;

    -- Calculate periods per fiscal year based on the FY range
    periods_per_year := fy_range_months / interval_months;

    -- Months since fiscal year end via AGE() (correct across year boundaries
    -- AND accounts for the day of month, unlike raw EXTRACT subtraction).
    months_since_fy_end := (DATE_PART('year', AGE(reference_date, fy_end_date)) * 12
        + DATE_PART('month', AGE(reference_date, fy_end_date)))::INTEGER;

    -- Fiscal month (1..fy_range_months); safe for negative months_since_fy_end too
    fiscal_month := ((months_since_fy_end - 1) % fy_range_months + fy_range_months) % fy_range_months + 1;

    -- Fiscal quarter derived from fiscal month relative to FY range
    -- Each quarter represents (fy_range_months / 4) months
    fiscal_quarter := CEIL(fiscal_month / (fy_range_months / 4.0))::INTEGER;

    -- Ensure fiscal_quarter stays within 1-4 range
    IF fiscal_quarter > 4 THEN
        fiscal_quarter := 4;
    END IF;

    -- Calculate current reporting period within the fiscal year
    current_period := CEIL(fiscal_month / interval_months::NUMERIC)::INTEGER;
    IF current_period > periods_per_year THEN
        current_period := periods_per_year;
    END IF;

    -- Calculate next reporting period
    next_period := current_period + 1;
    IF next_period > periods_per_year THEN
        next_period := 1;
    END IF;

    -- Convert next_period back to quarter for output
    -- Next quarter is derived from which reporting period we're moving to
    next_quarter := CASE
                        WHEN periods_per_year = 4 THEN next_period -- Quarterly
                        WHEN periods_per_year = 2 THEN next_period * 2 -- Semi-annual (Q2 or Q4)
                        WHEN periods_per_year = 1 THEN 4 -- Annual (always Q4/full year)
                        ELSE ((fiscal_quarter + (interval_months / (fy_range_months / 4)) - 1) % 4) + 1
        END;

    -- Fiscal year calculations based on FY range
    fiscal_year := EXTRACT(YEAR FROM fy_end_date)::INTEGER + 1 + ((months_since_fy_end - 1) / fy_range_months);

    -- Next quarter year
    next_quarter_year := CASE
                             WHEN next_period = 1 AND current_period = periods_per_year THEN fiscal_year + 1
                             ELSE fiscal_year
        END;

    -- Report type derived from reporting periods and FY range
    next_earnings_report_type := CASE
        -- Full year if annual reporting OR if next period completes the FY
                                     WHEN interval_months = fy_range_months THEN 'Full Year'
                                     WHEN next_period = periods_per_year AND periods_per_year > 1 THEN 'Full Year'
        -- Half year for semi-annual mid-year report
                                     WHEN interval_months = fy_range_months / 2 AND next_period = 1 THEN 'Half Year'
                                     ELSE 'Interim'
        END;
END;
$$;

alter function calculate_fiscal_info(date, date, text, out integer, out integer, out integer, out integer, out integer, out integer, out text, out text) owner to postgres;

