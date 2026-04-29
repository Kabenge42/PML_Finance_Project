create function calculate_reporting_lag(next_earnings date, income_statement_report_date date, earnings_report_frequency text DEFAULT 'Quarterly'::text) returns integer
    immutable
    parallel safe
    language sql
as
$$
SELECT CASE
           WHEN next_earnings IS NULL OR income_statement_report_date IS NULL THEN NULL
           -- Date - Date returns an INTEGER number of days in PostgreSQL.
           -- We compare against the expected reporting lag for the given frequency
           -- to produce the deviation (positive = late, negative = early).
           ELSE (next_earnings - income_statement_report_date)
               - get_expected_reporting_lag_days(earnings_report_frequency)
           END
$$;

alter function calculate_reporting_lag(date, date, text) owner to postgres;

