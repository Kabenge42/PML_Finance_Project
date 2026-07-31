CREATE FUNCTION public.calc_price_target_achievement_features(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, pt_achievement_1y numeric, pt_accuracy_1y numeric, pt_optimism_bias numeric, pt_range_hit_rate numeric, pt_median_vs_mean_spread numeric, pt_high_low_convergence_1y numeric, analyst_count_stability numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT "ISIN",
       CASE
	       WHEN "Price Target (1Y Ago)" > 0 AND "Last Price" >= "Price Target (1Y Ago)" THEN 1.0
	       WHEN "Price Target (1Y Ago)" > 0
		       THEN public.safe_divide("Last Price", "Price Target (1Y Ago)") END            AS pt_achievement_1y,
       ABS("Last Price" - "Price Target (1Y Ago)") / NULLIF(ABS("Price Target (1Y Ago)"), 0) AS pt_accuracy_1y,
       ("Price Target (1Y Ago)" - "Last Price") / NULLIF(ABS("Price Target (1Y Ago)"), 0)    AS pt_optimism_bias,
       CASE
	       WHEN "Last Price" BETWEEN "Price Target - Low (1Y Ago)" AND "Price Target - High (1Y Ago)" THEN 1.0
	       ELSE 0.0 END                                                                      AS pt_range_hit_rate,
       ("Price Target" - "Price Target - Median") /
       NULLIF("Price Target - Median", 0)                                                    AS pt_median_vs_mean_spread,
       (("Price Target - High" - "Price Target - Low") / NULLIF("Price Target - Median", 0)) -
       (("Price Target - High (1Y Ago)" - "Price Target - Low (1Y Ago)") /
        NULLIF("Price Target - Median (1Y Ago)", 0))                                         AS pt_high_low_convergence_1y,
       public.safe_divide("Price Target - #",
                          ("Price Target - # (1Y Ago)" + "Price Target - # (6M Ago)" + "Price Target - # (3M Ago)") /
                          3.0)                                                               AS analyst_count_stability
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

ALTER FUNCTION public.calc_price_target_achievement_features(unknown) OWNER TO postgres;