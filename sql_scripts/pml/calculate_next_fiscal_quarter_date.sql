create function pml.calculate_next_fiscal_quarter_date(income_statement_report_date date) returns date
	immutable
	strict
	parallel safe
	language plpgsql
as
$$
begin
	-- Calculate the first day of the fiscal quarter after the given date
	return date_trunc('quarter', income_statement_report_date)::date + interval '3 months';
end;
$$
;

alter function pml.calculate_next_fiscal_quarter_date(date) owner to postgres
;