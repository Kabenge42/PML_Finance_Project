CREATE FUNCTION get_expected_reporting_lag_days(earnings_report_frequency unknown) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION get_expected_reporting_lag_days(unknown) OWNER TO postgres;