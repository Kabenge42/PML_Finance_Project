CREATE FUNCTION public.calc_dividend_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                        text,
		        "dividend_streak"             integer,
		        "dividend_yield_ltm"          numeric,
		        "dividend_yield_ntm"          numeric,
		        "dividend_payout_ratio"       numeric,
		        "fcf_dividend_coverage"       numeric,
		        "buyback_yield"               numeric,
		        "total_shareholder_yield"     numeric,
		        "dividend_growth_expectation" numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_dividend_features(text) OWNER TO postgres;