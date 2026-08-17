create function public.calc_revenue_forecast_features(p_isin text default NULL::text)
	returns table("isin" text, "revenue_est_spread" numeric, "revenue_beat_potential" numeric, "revenue_est_revision_trend" numeric, "ebitda_est_vs_actual" numeric, "forward_revenue_multiple" numeric, "revenue_estimate_count" numeric, "revenue_guidance_gap" numeric, "consensus_revenue_growth" numeric, "ebit_estimate_spread" numeric, "forward_ebitda_margin" numeric, "revenue_acceleration" numeric, "estimate_confidence_score" numeric)
	stable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function public.calc_revenue_forecast_features(text) owner to postgres
;