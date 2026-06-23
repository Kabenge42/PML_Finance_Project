CREATE FUNCTION frequency_to_months(earnings_report_frequency text, fy_end_date date DEFAULT NULL::date, next_fy_end_date date DEFAULT NULL::date) RETURNS integer
	IMMUTABLE
	LANGUAGE plpgsql AS
$$
DECLARE
	fy_range_months INT := 12;
BEGIN
	-- Use AGE() for month arithmetic â€” correct across year boundaries
	IF fy_end_date IS NOT NULL AND next_fy_end_date IS NOT NULL THEN
		fy_range_months := (DATE_PART('year', AGE(next_fy_end_date, fy_end_date)) * 12 +
		                    DATE_PART('month', AGE(next_fy_end_date, fy_end_date)))::INT;
	END IF;

	RETURN CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
		       WHEN 'QUARTERLY' THEN fy_range_months / 4
		       WHEN 'SEMI-ANNUAL' THEN fy_range_months / 2
		       WHEN 'SEMI-ANNUALLY' THEN fy_range_months / 2
		       WHEN 'ANNUAL' THEN fy_range_months
		       WHEN 'ANNUALLY' THEN fy_range_months
		       ELSE fy_range_months / 4 END;
END;
$$;

ALTER FUNCTION frequency_to_months(unknown, unknown, unknown) OWNER TO postgres;