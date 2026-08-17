create function public.calculate_next_fiscal_quarter(next_earnings_date date, income_statement_report_date date, fy_end_date date, earnings_report_frequency text default 'Quarterly'::text) returns integer
	immutable
	language plpgsql
as
$$
begin
	-- missing source code
end;
$$
;

alter function public.calculate_next_fiscal_quarter(date, date, date, text) owner to postgres
;