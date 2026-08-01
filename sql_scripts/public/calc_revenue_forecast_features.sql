CREATE FUNCTION public.calc_revenue_forecast_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "revenue_est_spread" numeric, "revenue_beat_potential" numeric, "revenue_est_revision_trend" numeric, "ebitda_est_vs_actual" numeric, "forward_revenue_multiple" numeric, "revenue_estimate_count" numeric, "revenue_guidance_gap" numeric, "consensus_revenue_growth" numeric, "ebit_estimate_spread" numeric, "forward_ebitda_margin" numeric, "revenue_acceleration" numeric, "estimate_confidence_score" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_revenue_forecast_features(text) OWNER TO postgres;