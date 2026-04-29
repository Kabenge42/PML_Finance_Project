create function calculate_expected_report_date(period_end_date date, earnings_report_frequency text) returns date
    immutable
    parallel safe
    language sql
as
$$
SELECT CASE
           WHEN period_end_date IS NULL THEN NULL
           ELSE (period_end_date
               + make_interval(days => get_expected_reporting_lag_days(earnings_report_frequency)))::DATE
           END
$$;

alter function calculate_expected_report_date(date, text) owner to postgres;

