CREATE FUNCTION public.calc_long_term_momentum_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "price_momentum_1y" numeric, "price_momentum_3y" numeric, "price_momentum_5y" numeric, "long_term_trend_score" numeric, "price_vs_ema_250d" numeric, "multi_year_high_flag" integer, "secular_trend_flag" integer, "total_return_ytd" numeric, "total_return_5y" numeric, "total_return_10y" numeric, "return_cagr_3y" numeric, "return_cagr_10y" numeric, "return_vs_price_momentum" numeric, "return_consistency_score" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_long_term_momentum_features(text) OWNER TO postgres;