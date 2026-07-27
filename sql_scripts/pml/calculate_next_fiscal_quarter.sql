CREATE FUNCTION calculate_next_fiscal_quarter(next_earnings date, income_statement_report_date date, fy_end_date date, earnings_report_frequency text default 'Quarterly'::text) RETURNS integer
	IMMUTABLE
	LANGUAGE plpgsql AS
$$
BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION calculate_next_fiscal_quarter(date, date, date, text) OWNER TO postgres;