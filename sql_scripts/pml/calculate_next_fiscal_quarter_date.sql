create function calculate_next_fiscal_quarter_date(income_statement_report_date date) returns date
	immutable
	strict
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function calculate_next_fiscal_quarter_date(date) owner to postgres
;