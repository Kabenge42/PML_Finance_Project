create function calculate_next_fiscal_quarter(next_earnings_date date, income_statement_report_date date,
                                              fy_end_date date,
                                              earnings_report_frequency text DEFAULT 'Quarterly'::text) returns integer
    immutable
    language plpgsql
as
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

    -- Closed-form: how many whole fiscal years between fy_end and reference?
    -- GREATEST(0, ceil(months_diff / 12)) â€” avoids the WHILE loop entirely
    years_ahead := GREATEST(0,
                            CEIL(
                                    (DATE_PART('year', AGE(reference_date, fy_end_date)) * 12
                                        + DATE_PART('month', AGE(reference_date, fy_end_date)))::NUMERIC / 12
                            )::INT
                   );

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

alter function calculate_next_fiscal_quarter(date, date, date, text) owner to postgres;

