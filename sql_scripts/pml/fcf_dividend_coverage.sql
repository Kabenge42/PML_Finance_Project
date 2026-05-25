-- Cyclic dependencies found

CREATE FUNCTION pml.fcf_dividend_coverage(fcf double precision, dividends_paid double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT pml.safe_divide(fcf, ABS(dividends_paid));
$$;

ALTER FUNCTION pml.fcf_dividend_coverage(unknown, unknown) OWNER TO postgres;

CREATE FUNCTION pml.fcf_dividend_coverage(fcf numeric, dividends_paid numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT pml.safe_divide(fcf, ABS(dividends_paid));
$$;

ALTER FUNCTION pml.fcf_dividend_coverage(unknown, unknown) OWNER TO postgres;