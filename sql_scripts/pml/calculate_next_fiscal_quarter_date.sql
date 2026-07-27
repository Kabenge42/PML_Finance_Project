CREATE FUNCTION calculate_next_fiscal_quarter_date(income_statement_report_date date) RETURNS date
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION calculate_next_fiscal_quarter_date(date) OWNER TO postgres;