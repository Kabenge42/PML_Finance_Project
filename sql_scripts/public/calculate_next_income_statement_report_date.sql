create function calculate_next_income_statement_report_date(income_statement_report_date date, earnings_report_frequency text) returns date
    immutable
    parallel safe
    language sql
as
$$
SELECT CASE
           WHEN income_statement_report_date IS NULL THEN NULL
           ELSE (income_statement_report_date
               + make_interval(months => frequency_to_months(earnings_report_frequency)))::DATE
           END
$$;

alter function calculate_next_income_statement_report_date(date, text) owner to postgres;

