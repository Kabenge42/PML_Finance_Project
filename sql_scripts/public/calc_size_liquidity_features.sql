CREATE FUNCTION public.calc_size_liquidity_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "market_cap" numeric, "market_cap_country_r" numeric, "log_market_cap" numeric, "volume_shrs" numeric, "relative_volume" numeric, "shares_outstanding" numeric, "daily_turnover_ratio" numeric, "size_class" text, "style_class" text, "liquidity_score" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_size_liquidity_features(text) OWNER TO postgres;