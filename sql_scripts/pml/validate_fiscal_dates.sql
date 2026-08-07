CREATE FUNCTION validate_fiscal_dates(fy_end_date unknown, report_date unknown, reference_date unknown default CURRENT_DATE)
	RETURNS table("issue" text, "severity" text)
	IMMUTABLE
	LANGUAGE plpgsql
AS
$$
BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION validate_fiscal_dates(unknown, unknown, unknown) OWNER TO postgres;