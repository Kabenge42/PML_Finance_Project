CREATE FUNCTION public.calc_price_target_achievement_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                       text,
		        "pt_achievement_1y"          numeric,
		        "pt_accuracy_1y"             numeric,
		        "pt_optimism_bias"           numeric,
		        "pt_range_hit_rate"          numeric,
		        "pt_median_vs_mean_spread"   numeric,
		        "pt_high_low_convergence_1y" numeric,
		        "analyst_count_stability"    numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_price_target_achievement_features(text) OWNER TO postgres;