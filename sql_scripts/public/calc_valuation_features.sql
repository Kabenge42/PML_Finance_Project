CREATE FUNCTION public.calc_valuation_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"            text,
		        "p_e_ratio"       numeric,
		        "p_b_ratio"       numeric,
		        "ev_ebitda_ratio" numeric,
		        "ev_sales_ratio"  numeric,
		        "dividend_yield"  numeric,
		        "peg_ratio"       numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_valuation_features(text) OWNER TO postgres;