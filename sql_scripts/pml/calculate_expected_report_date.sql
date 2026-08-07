CREATE FUNCTION calculate_expected_report_date(period_end_date unknown, earnings_report_frequency unknown) RETURNS date
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION calculate_expected_report_date(unknown, unknown) OWNER TO postgres;