create function calculate_next_fiscal_quarter_date(income_statement_report_date date) returns date
	immutable
	strict
	parallel safe
	language sql
as
$$
SELECT (income_statement_report_date + make_interval(months => 3))::DATE
$$
;

alter function calculate_next_fiscal_quarter_date(unknown) owner to postgres
;