CREATE FUNCTION public.calc_earnings_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "eps_surprise_pct" numeric, "revenue_surprise_pct" numeric, "eps_adjustment_ratio" numeric, "gaap_adj_eps_gap_pct" numeric, "ebitda_adjustment_ratio" numeric, "eps_quarterly_trend" numeric, "eps_yoy_growth" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_earnings_features(text) OWNER TO postgres;