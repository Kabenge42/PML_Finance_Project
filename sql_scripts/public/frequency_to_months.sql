CREATE FUNCTION public.frequency_to_months(earnings_report_frequency text, fy_end_date date DEFAULT NULL::date,
                                           next_fy_end_date          date DEFAULT NULL::date) RETURNS integer
	IMMUTABLE
	LANGUAGE plpgsql AS
$$
BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.frequency_to_months(text, date, date) OWNER TO postgres;