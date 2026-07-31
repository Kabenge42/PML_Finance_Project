CREATE FUNCTION public.calc_dividend_yield_comprehensive(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, div_yield_ltm numeric, div_yield_ntm numeric, div_yield_ind numeric, div_yield_1fy_ind numeric, div_yield_5y_avg numeric, div_yield_vs_5y_avg numeric, div_yield_growth_expected numeric, dividend_streak integer, high_yield_flag integer, sustainable_dividend_flag integer)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT "ISIN"                                                                       AS isin,
       "Div Yield (LTM)"                                                            AS div_yield_ltm,
       "Div Yield (NTM)"                                                            AS div_yield_ntm,
       "Div Yield (Ind)"                                                            AS div_yield_ind,
       "Div Yield (-1FYInd)"                                                        AS div_yield_1fy_ind,
       "Div Yield (5YAVGLTM)"                                                       AS div_yield_5y_avg,
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0)                        AS div_yield_vs_5y_avg,
       ("Div Yield (NTM)" - "Div Yield (LTM)") / NULLIF("Div Yield (LTM)", 0) * 100 AS div_yield_growth_expected,
       "Dividend Streak"::INTEGER                                                   AS dividend_streak,
       CASE WHEN "Div Yield (LTM)" > 0.05 THEN 1 ELSE 0 END                         AS high_yield_flag,
       CASE
	       WHEN "Div Yield (LTM)" > 0 AND "FCF (LTM)" > ABS(COALESCE("Common Dividends Paid (LTM)", 0)) AND
	            "Dividend Streak" >= 5 THEN 1
	       ELSE 0 END                                                               AS sustainable_dividend_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

ALTER FUNCTION public.calc_dividend_yield_comprehensive(unknown) OWNER TO postgres;