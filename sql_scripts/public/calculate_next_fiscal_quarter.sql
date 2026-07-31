CREATE FUNCTION public.calculate_next_fiscal_quarter(next_earnings_date date, income_statement_report_date date, fy_end_date date, earnings_report_frequency text DEFAULT 'Quarterly'::text) RETURNS integer
	IMMUTABLE
	LANGUAGE plpgsql AS
$$
DECLARE
    reference_date   DATE;
    interval_months  INT;
    years_ahead      INT;
    current_fy_start DATE;
    months_into_fy   INT;
BEGIN
    IF fy_end_date IS NULL THEN
        RETURN NULL;
    END IF;

    interval_months := frequency_to_months(earnings_report_frequency);

    -- Choose reference date
    IF income_statement_report_date IS NOT NULL THEN
        reference_date := income_statement_report_date;
    ELSIF next_earnings_date IS NOT NULL THEN
        reference_date := next_earnings_date;
    ELSE
        RETURN NULL;
    END IF;

    -- How many whole fiscal years between fy_end and reference, using AGE()
    -- so that day-of-month is respected. FLOOR + 1 keeps us inside the CURRENT
    -- fiscal year even when reference_date falls exactly on an FY boundary.
    IF reference_date <= fy_end_date THEN
        years_ahead := 0;
    ELSE
        years_ahead := FLOOR(
                               (DATE_PART('year', AGE(reference_date, fy_end_date)) * 12
                                   + DATE_PART('month', AGE(reference_date, fy_end_date)))::NUMERIC / 12
                       )::INT + 1;
    END IF;

    -- Start of the current fiscal year = (fy_end + (years_ahead - 1) years) + 1 day
    current_fy_start := (fy_end_date + make_interval(years => years_ahead - 1)
        + INTERVAL '1 day')::DATE;

    -- Months into FY using AGE (handles month-length variations correctly)
    months_into_fy := (DATE_PART('year', AGE(reference_date, current_fy_start)) * 12
        + DATE_PART('month', AGE(reference_date, current_fy_start)))::INT + 1;

    -- Safe 1â€“12 normalization even for negative values
    months_into_fy := ((months_into_fy - 1) % 12 + 12) % 12 + 1;

    RETURN LEAST(4, GREATEST(1, CEIL(months_into_fy / 3.0)::INT));
END;
$$;

ALTER FUNCTION public.calculate_next_fiscal_quarter(unknown, unknown, unknown, unknown) OWNER TO postgres;