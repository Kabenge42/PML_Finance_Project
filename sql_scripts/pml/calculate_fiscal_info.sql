CREATE FUNCTION calculate_fiscal_info(reference_date unknown, fy_end_date unknown, input_earnings_frequency unknown default NULL::text, out fiscal_month unknown, out fiscal_quarter unknown, out fiscal_year unknown, out next_quarter unknown, out next_quarter_year unknown, out reporting_interval unknown, out earnings_report_frequency unknown, out next_earnings_report_type unknown) RETURNS record
	IMMUTABLE
	LANGUAGE plpgsql AS
$$
BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION calculate_fiscal_info(unknown, unknown, unknown, OUT unknown, OUT unknown, OUT unknown, OUT unknown, OUT unknown, OUT unknown, OUT unknown, OUT unknown) OWNER TO postgres;