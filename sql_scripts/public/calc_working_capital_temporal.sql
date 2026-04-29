create function calc_working_capital_temporal(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, wc_fq numeric, wc_fy numeric, wc_ltm numeric, wc_5yavgfy numeric, wc_1fq numeric, wc_2fq numeric, wc_3fq numeric, wc_4fq numeric, wc_1fy numeric, wc_2fy numeric, wc_3fy numeric, wc_4fy numeric, wc_qoq_change numeric, wc_yoy_change numeric, wc_4q_trend numeric, wc_vs_5y_avg numeric, wc_positive_quarters integer, wc_improving_flag integer, wc_volatility numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                                    AS isin,
       -- Current values
       "Working Capital (FQ)"                                                                    AS wc_fq,
       "Working Capital (FY)"                                                                    AS wc_fy,
       "Working Capital (LTM)"                                                                   AS wc_ltm,
       "Working Capital (5YAVGFY)"                                                               AS wc_5yavgfy,
       -- Quarterly historical
       "Working Capital (-1FQ)"                                                                  AS wc_1fq,
       "Working Capital (-2FQ)"                                                                  AS wc_2fq,
       "Working Capital (-3FQ)"                                                                  AS wc_3fq,
       "Working Capital (-4FQ)"                                                                  AS wc_4fq,
       -- Yearly historical
       "Working Capital (-1FY)"                                                                  AS wc_1fy,
       "Working Capital (-2FY)"                                                                  AS wc_2fy,
       "Working Capital (-3FY)"                                                                  AS wc_3fy,
       "Working Capital (-4FY)"                                                                  AS wc_4fy,
       -- Trend metrics
       public.pct_change("Working Capital (FQ)"::NUMERIC, "Working Capital (-1FQ)"::NUMERIC)     AS wc_qoq_change,
       public.pct_change("Working Capital (FY)"::NUMERIC, "Working Capital (-1FY)"::NUMERIC)     AS wc_yoy_change,
       public.pct_change("Working Capital (FQ)"::NUMERIC, "Working Capital (-4FQ)"::NUMERIC)     AS wc_4q_trend,
       public.safe_divide("Working Capital (FQ)"::NUMERIC, "Working Capital (5YAVGFY)"::NUMERIC) AS wc_vs_5y_avg,
       (CASE WHEN "Working Capital (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-1FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-2FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-3FQ)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN "Working Capital (-4FQ)" > 0 THEN 1
            ELSE 0 END)::INTEGER                                                                 AS wc_positive_quarters,
       CASE
           WHEN "Working Capital (FQ)" > "Working Capital (-1FQ)"
               AND "Working Capital (-1FQ)" > "Working Capital (-2FQ)"
               THEN 1
           ELSE 0 END                                                                            AS wc_improving_flag,
       -- Volatility: coefficient of variation across quarters
       (ABS("Working Capital (FQ)" - "Working Capital (-1FQ)") +
        ABS("Working Capital (-1FQ)" - "Working Capital (-2FQ)") +
        ABS("Working Capital (-2FQ)" - "Working Capital (-3FQ)") +
        ABS("Working Capital (-3FQ)" - "Working Capital (-4FQ)")) /
       NULLIF(ABS(("Working Capital (FQ)" + "Working Capital (-1FQ)" +
                   "Working Capital (-2FQ)" + "Working Capital (-3FQ)" +
                   "Working Capital (-4FQ)") / 5.0), 0)                                          AS wc_volatility
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_working_capital_temporal(text) owner to postgres;

