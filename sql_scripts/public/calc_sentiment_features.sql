CREATE FUNCTION public.calc_sentiment_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                      text,
		        "analyst_bullish_pct"       numeric,
		        "analyst_bearish_pct"       numeric,
		        "analyst_neutral_pct"       numeric,
		        "analyst_conviction"        numeric,
		        "upside_potential"          numeric,
		        "price_target_spread_pct"   numeric,
		        "price_target_revision_1m"  numeric,
		        "price_target_revision_3m"  numeric,
		        "eps_revision_momentum"     numeric,
		        "analyst_rating_normalized" numeric,
		        "analyst_coverage_quality"  numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_sentiment_features(text) OWNER TO postgres;