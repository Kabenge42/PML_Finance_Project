CREATE FUNCTION public.calc_dividend_history_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                     text,
		        "div_yield_2fy"            numeric,
		        "div_yield_3fy"            numeric,
		        "div_yield_4fy"            numeric,
		        "div_yield_5fy"            numeric,
		        "div_yield_trend_3y"       numeric,
		        "div_yield_volatility"     numeric,
		        "div_yield_declining_flag" integer,
		        "div_yield_mean_5y"        numeric,
		        "div_yield_vs_5y_mean"     numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_dividend_history_features(text) OWNER TO postgres;