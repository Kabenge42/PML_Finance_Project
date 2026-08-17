create function public.calc_price_target_achievement_features(p_isin text default NULL::text)
	returns table("isin" text, "pt_achievement_1y" numeric, "pt_accuracy_1y" numeric, "pt_optimism_bias" numeric, "pt_range_hit_rate" numeric, "pt_median_vs_mean_spread" numeric, "pt_high_low_convergence_1y" numeric, "analyst_count_stability" numeric)
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

alter function public.calc_price_target_achievement_features(text) owner to postgres
;