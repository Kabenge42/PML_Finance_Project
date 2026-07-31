CREATE FUNCTION public.calc_dividend_history_features(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, div_yield_2fy numeric, div_yield_3fy numeric, div_yield_4fy numeric, div_yield_5fy numeric, div_yield_trend_3y numeric, div_yield_volatility numeric, div_yield_declining_flag integer, div_yield_mean_5y numeric, div_yield_vs_5y_mean numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT "ISIN",
       "Div Yield (-2FYInd)",
       "Div Yield (-3FYInd)",
       "Div Yield (-4FYInd)",
       "Div Yield (-5FYInd)",
       ("Div Yield (Ind)" - "Div Yield (-3FYInd)") / 3.0                                                              AS div_yield_trend_3y,
       GREATEST("Div Yield (Ind)", "Div Yield (-1FYInd)", "Div Yield (-2FYInd)", "Div Yield (-3FYInd)",
                "Div Yield (-4FYInd)") -
       LEAST("Div Yield (Ind)", "Div Yield (-1FYInd)", "Div Yield (-2FYInd)", "Div Yield (-3FYInd)",
             "Div Yield (-4FYInd)")                                                                                   AS div_yield_volatility,
       CASE
	       WHEN "Div Yield (Ind)" < "Div Yield (-1FYInd)" AND "Div Yield (-1FYInd)" < "Div Yield (-2FYInd)" AND
	            "Div Yield (-2FYInd)" < "Div Yield (-3FYInd)" THEN 1
	       ELSE 0 END                                                                                                 AS div_yield_declining_flag,
       (COALESCE("Div Yield (Ind)", 0) + COALESCE("Div Yield (-1FYInd)", 0) + COALESCE("Div Yield (-2FYInd)", 0) +
        COALESCE("Div Yield (-3FYInd)", 0) + COALESCE("Div Yield (-4FYInd)", 0)) / NULLIF(
		       (CASE WHEN "Div Yield (Ind)" IS NOT NULL THEN 1 ELSE 0 END +
		        CASE WHEN "Div Yield (-1FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
		        CASE WHEN "Div Yield (-2FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
		        CASE WHEN "Div Yield (-3FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
		        CASE WHEN "Div Yield (-4FYInd)" IS NOT NULL THEN 1 ELSE 0 END)::NUMERIC,
		       0)                                                                                                     AS div_yield_mean_5y,
       public.calc_change_ratio("Div Yield (Ind)",
                                (COALESCE("Div Yield (Ind)", 0) + COALESCE("Div Yield (-1FYInd)", 0) +
                                 COALESCE("Div Yield (-2FYInd)", 0) + COALESCE("Div Yield (-3FYInd)", 0) +
                                 COALESCE("Div Yield (-4FYInd)", 0)) / NULLIF(
		                                (CASE WHEN "Div Yield (Ind)" IS NOT NULL THEN 1 ELSE 0 END +
		                                 CASE WHEN "Div Yield (-1FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
		                                 CASE WHEN "Div Yield (-2FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
		                                 CASE WHEN "Div Yield (-3FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
		                                 CASE WHEN "Div Yield (-4FYInd)" IS NOT NULL THEN 1 ELSE 0 END)::NUMERIC,
		                                0))                                                                           AS div_yield_vs_5y_mean
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

ALTER FUNCTION public.calc_dividend_history_features(unknown) OWNER TO postgres;