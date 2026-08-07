CREATE FUNCTION calculate_reporting_lag(next_earnings unknown, income_statement_report_date unknown, earnings_report_frequency unknown default 'Quarterly'::text) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION calculate_reporting_lag(unknown, unknown, unknown) OWNER TO postgres;