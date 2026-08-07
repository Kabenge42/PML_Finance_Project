CREATE FUNCTION derive_earnings_report_frequency(income_statement_report_date unknown, fy_end_date unknown) RETURNS text
	IMMUTABLE
	LANGUAGE plpgsql AS
$$
BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION derive_earnings_report_frequency(unknown, unknown) OWNER TO postgres;