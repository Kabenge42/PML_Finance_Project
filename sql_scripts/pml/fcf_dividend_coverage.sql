CREATE FUNCTION fcf_dividend_coverage(fcf numeric, dividends_paid numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION fcf_dividend_coverage(numeric, numeric) OWNER TO postgres;

CREATE FUNCTION fcf_dividend_coverage(fcf double precision, dividends_paid double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION fcf_dividend_coverage(double precision, double precision) OWNER TO postgres;