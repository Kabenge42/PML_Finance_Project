CREATE FUNCTION calculate_next_income_statement_report_date(income_statement_report_date unknown, earnings_report_frequency unknown) RETURNS date
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION calculate_next_income_statement_report_date(unknown, unknown) OWNER TO postgres;