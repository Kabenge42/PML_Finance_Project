CREATE FUNCTION public.calc_fcf_growth_estimates(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, fcf_est_fy1 numeric, fcf_est_fy2 numeric, fcf_est_fy3 numeric, fcf_est_fy4 numeric, fcf_est_fy5 numeric, fcf_est_growth_fy1_vs_ltm numeric, fcf_est_growth_fy2_vs_fy1 numeric, fcf_est_growth_fy3_vs_fy2 numeric, fcf_est_growth_fy4_vs_fy3 numeric, fcf_est_growth_fy5_vs_fy4 numeric, fcf_est_cagr_3y numeric, fcf_est_cagr_5y numeric, fcf_est_margin_fy1 numeric, fcf_est_yield_fy1 numeric, fcf_est_growth_acceleration numeric, fcf_est_growth_deceleration integer, fcf_est_trajectory_score numeric, fcf_est_always_positive integer, fcf_est_vs_historical numeric, fcf_est_capex_implied_ratio numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT "ISIN"                                                                                               AS isin,

       -- Raw forward estimates
       "FCF - Est Avg (FY1E)"                                                                               AS fcf_est_fy1,
       "FCF - Est Avg (FY2E)"                                                                               AS fcf_est_fy2,
       "FCF - Est Avg (FY3E)"                                                                               AS fcf_est_fy3,
       "FCF - Est Avg (FY4E)"                                                                               AS fcf_est_fy4,
       "FCF - Est Avg (FY5E)"                                                                               AS fcf_est_fy5,

       -- YoY estimated growth rates (as percentages)
       ("FCF - Est Avg (FY1E)" - "FCF (LTM)") / NULLIF(ABS("FCF (LTM)"), 0) *
       100                                                                                                  AS fcf_est_growth_fy1_vs_ltm,

       ("FCF - Est Avg (FY2E)" - "FCF - Est Avg (FY1E)") / NULLIF(ABS("FCF - Est Avg (FY1E)"), 0) *
       100                                                                                                  AS fcf_est_growth_fy2_vs_fy1,

       ("FCF - Est Avg (FY3E)" - "FCF - Est Avg (FY2E)") / NULLIF(ABS("FCF - Est Avg (FY2E)"), 0) *
       100                                                                                                  AS fcf_est_growth_fy3_vs_fy2,

       ("FCF - Est Avg (FY4E)" - "FCF - Est Avg (FY3E)") / NULLIF(ABS("FCF - Est Avg (FY3E)"), 0) *
       100                                                                                                  AS fcf_est_growth_fy4_vs_fy3,

       ("FCF - Est Avg (FY5E)" - "FCF - Est Avg (FY4E)") / NULLIF(ABS("FCF - Est Avg (FY4E)"), 0) *
       100                                                                                                  AS fcf_est_growth_fy5_vs_fy4,

       -- 3-year estimated CAGR: (FY3E / LTM)^(1/3) - 1
       CASE
	       WHEN "FCF (LTM)" > 0 AND "FCF - Est Avg (FY3E)" > 0 THEN
		       (POWER("FCF - Est Avg (FY3E)" / NULLIF("FCF (LTM)", 0), 1.0 / 3.0) - 1) *
		       100 END                                                                                      AS fcf_est_cagr_3y,

       -- 5-year estimated CAGR: (FY5E / LTM)^(1/5) - 1
       CASE
	       WHEN "FCF (LTM)" > 0 AND "FCF - Est Avg (FY5E)" > 0 THEN
		       (POWER("FCF - Est Avg (FY5E)" / NULLIF("FCF (LTM)", 0), 1.0 / 5.0) - 1) *
		       100 END                                                                                      AS fcf_est_cagr_5y,

       -- Forward FCF margin (FY1E FCF as % of current revenue)
       "FCF - Est Avg (FY1E)" / NULLIF("Total Revenues (LTM)", 0) * 100                                     AS fcf_est_margin_fy1,

       -- Forward FCF yield (FY1E FCF as % of market cap)
       "FCF - Est Avg (FY1E)" / NULLIF("Market Cap", 0) * 100                                               AS fcf_est_yield_fy1,

       -- Growth acceleration: is FY2â†’FY1 growth faster than FY1â†’LTM growth?
       (("FCF - Est Avg (FY2E)" - "FCF - Est Avg (FY1E)") / NULLIF(ABS("FCF - Est Avg (FY1E)"), 0) * 100) -
       (("FCF - Est Avg (FY1E)" - "FCF (LTM)") / NULLIF(ABS("FCF (LTM)"), 0) *
        100)                                                                                                AS fcf_est_growth_acceleration,

       -- Growth deceleration flag: each subsequent growth rate is lower
       CASE
	       WHEN ("FCF - Est Avg (FY2E)" - "FCF - Est Avg (FY1E)") / NULLIF(ABS("FCF - Est Avg (FY1E)"), 0) <
	            ("FCF - Est Avg (FY1E)" - "FCF (LTM)") / NULLIF(ABS("FCF (LTM)"), 0) AND
	            ("FCF - Est Avg (FY3E)" - "FCF - Est Avg (FY2E)") / NULLIF(ABS("FCF - Est Avg (FY2E)"), 0) <
	            ("FCF - Est Avg (FY2E)" - "FCF - Est Avg (FY1E)") / NULLIF(ABS("FCF - Est Avg (FY1E)"), 0) THEN 1
	       ELSE 0 END                                                                                       AS fcf_est_growth_deceleration,

       -- Forward trajectory score: how many of 5 forward years have positive FCF
       (CASE WHEN "FCF - Est Avg (FY1E)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF - Est Avg (FY2E)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF - Est Avg (FY3E)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF - Est Avg (FY4E)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF - Est Avg (FY5E)" > 0 THEN 1 ELSE 0 END) / 5.0 *
       100                                                                                                  AS fcf_est_trajectory_score,

       -- All 5 forward estimates positive
       CASE
	       WHEN "FCF - Est Avg (FY1E)" > 0 AND "FCF - Est Avg (FY2E)" > 0 AND "FCF - Est Avg (FY3E)" > 0 AND
	            "FCF - Est Avg (FY4E)" > 0 AND "FCF - Est Avg (FY5E)" > 0 THEN 1
	       ELSE 0 END                                                                                       AS fcf_est_always_positive,

       -- Estimated vs historical: compare forward FY1 growth to last actual FY growth
       (("FCF - Est Avg (FY1E)" - "FCF (LTM)") / NULLIF(ABS("FCF (LTM)"), 0) * 100) -
       (("FCF (FY)" - "FCF (-1FY)") / NULLIF(ABS("FCF (-1FY)"), 0) * 100)                                   AS fcf_est_vs_historical,

       -- Implied CapEx conversion: FY1E FCF relative to current LTM operating CF
       "FCF - Est Avg (FY1E)" / NULLIF(ABS("CFO (LTM)") - ABS(COALESCE("Capital Expenditure (LTM)", 0)),
                                       0)                                                                   AS fcf_est_capex_implied_ratio

FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

COMMENT ON FUNCTION public.calc_fcf_growth_estimates(unknown) IS 'Estimated free cash flow growth rates from consensus FCF forecasts (FY1E-FY5E).
     Calculates YoY growth rates, 3Y/5Y CAGRs, growth acceleration, forward margins/yields,
     and trajectory quality scores. Source: FCF - Est Avg (FY1E through FY5E).';

ALTER FUNCTION public.calc_fcf_growth_estimates(unknown) OWNER TO postgres;