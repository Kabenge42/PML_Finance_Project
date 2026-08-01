CREATE FUNCTION public.calc_opex_temporal_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "opex_fq" numeric, "opex_ltm" numeric, "opex_fy" numeric, "opex_qoq_growth" numeric, "opex_yoy_growth" numeric, "opex_vs_revenue_trend" numeric, "sga_qoq_growth" numeric, "sga_yoy_growth" numeric, "operating_leverage_score" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_opex_temporal_features(text) OWNER TO postgres;