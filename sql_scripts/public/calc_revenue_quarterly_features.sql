create function calc_revenue_quarterly_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, revenue_fq numeric, revenue_fy numeric, revenue_ltm numeric, revenue_5y_avg numeric, revenue_1fqfq numeric, revenue_2fqfq numeric, revenue_3fqfq numeric, revenue_4fqfq numeric, revenue_1fy numeric, revenue_2fy numeric, revenue_3fy numeric, revenue_4fy numeric, revenue_yoy_growth numeric, revenue_vs_5y_avg numeric, revenue_ltm_vs_fy numeric, revenue_fq_vs_5y_avg_fq numeric, revenue_qoq_growth numeric, revenue_qoq_2q numeric, revenue_qoq_3q numeric, revenue_qoq_4q numeric, revenue_yoy_quarterly numeric, revenue_2y_growth numeric, revenue_3y_growth numeric, revenue_4y_growth numeric, revenue_cagr_3y numeric, revenue_cagr_4y numeric, revenue_4q_trend numeric, revenue_4q_avg numeric, revenue_fq_vs_4q_avg numeric, revenue_growth_flag integer, revenue_stability_score numeric, revenue_accelerating_flag integer, revenue_positive_qoq_streak integer)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                                    AS isin,
       -- Base revenue values
       "Total Revenues (FQ)"                                                                     AS revenue_fq,
       "Total Revenues (FY)"                                                                     AS revenue_fy,
       "Total Revenues (LTM)"                                                                    AS revenue_ltm,
       "Total Revenues (5YAVGLTM)"                                                               AS revenue_5y_avg,
       -- Quarterly historical values
       "Total Revenues (-1FQFQ)"                                                                 AS revenue_1fqfq,
       "Total Revenues (-2FQFQ)"                                                                 AS revenue_2fqfq,
       "Total Revenues (-3FQFQ)"                                                                 AS revenue_3fqfq,
       "Total Revenues (-4FQFQ)"                                                                 AS revenue_4fqfq,
       -- Extended yearly historical
       "Total Revenues (-1FY)"                                                                   AS revenue_1fy,
       "Total Revenues (-2FY)"                                                                   AS revenue_2fy,
       "Total Revenues (-3FY)"                                                                   AS revenue_3fy,
       "Total Revenues (-4FY)"                                                                   AS revenue_4fy,
       -- Year-over-year growth using FY data
       public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC)       AS revenue_yoy_growth,
       -- Current vs 5-year average
       public.safe_divide("Total Revenues (LTM)"::NUMERIC, "Total Revenues (5YAVGLTM)"::NUMERIC) AS revenue_vs_5y_avg,
       -- LTM vs FY comparison
       public.safe_divide("Total Revenues (LTM)"::NUMERIC, "Total Revenues (FY)"::NUMERIC)       AS revenue_ltm_vs_fy,
       -- FQ vs 5-year average FQ
       public.safe_divide("Total Revenues (FQ)"::NUMERIC,
                          "Total Revenues (5YAVGFQ)"::NUMERIC)                                   AS revenue_fq_vs_5y_avg_fq,
       -- Quarterly momentum: QoQ growth rates
       public.pct_change("Total Revenues (FQ)"::NUMERIC, "Total Revenues (-1FQFQ)"::NUMERIC)     AS revenue_qoq_growth,
       public.pct_change("Total Revenues (-1FQFQ)"::NUMERIC, "Total Revenues (-2FQFQ)"::NUMERIC) AS revenue_qoq_2q,
       public.pct_change("Total Revenues (-2FQFQ)"::NUMERIC, "Total Revenues (-3FQFQ)"::NUMERIC) AS revenue_qoq_3q,
       public.pct_change("Total Revenues (-3FQFQ)"::NUMERIC, "Total Revenues (-4FQFQ)"::NUMERIC) AS revenue_qoq_4q,
       -- YoY quarterly comparison (current FQ vs same quarter last year)
       public.pct_change("Total Revenues (FQ)"::NUMERIC,
                         "Total Revenues (-4FQFQ)"::NUMERIC)                                     AS revenue_yoy_quarterly,
       -- Multi-year growth rates
       public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-2FY)"::NUMERIC)       AS revenue_2y_growth,
       public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-3FY)"::NUMERIC)       AS revenue_3y_growth,
       public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-4FY)"::NUMERIC)       AS revenue_4y_growth,
       -- CAGR calculations
       CASE
           WHEN "Total Revenues (-3FY)" > 0 AND "Total Revenues (FY)" > 0
               THEN
               (POWER(public.safe_divide("Total Revenues (FY)"::NUMERIC, "Total Revenues (-3FY)"::NUMERIC), 1.0 / 3.0) -
                1) *
               100
           END                                                                                   AS revenue_cagr_3y,
       CASE
           WHEN "Total Revenues (-4FY)" > 0 AND "Total Revenues (FY)" > 0
               THEN
               (POWER(public.safe_divide("Total Revenues (FY)"::NUMERIC, "Total Revenues (-4FY)"::NUMERIC), 1.0 / 4.0) -
                1) *
               100
           END                                                                                   AS revenue_cagr_4y,
       -- Quarterly trend: FQ vs 4 quarters ago
       public.pct_change("Total Revenues (FQ)"::NUMERIC, "Total Revenues (-4FQFQ)"::NUMERIC)     AS revenue_4q_trend,
       -- Trailing 4-quarter average
       ("Total Revenues (FQ)" + "Total Revenues (-1FQFQ)" +
        "Total Revenues (-2FQFQ)" + "Total Revenues (-3FQFQ)") / 4.0                             AS revenue_4q_avg,
       -- FQ vs trailing 4Q average
       public.safe_divide("Total Revenues (FQ)"::NUMERIC,
                          ("Total Revenues (FQ)" + "Total Revenues (-1FQFQ)" +
                           "Total Revenues (-2FQFQ)" + "Total Revenues (-3FQFQ)") /
                          4.0)                                                                   AS revenue_fq_vs_4q_avg,
       -- Growth flag: 1 if growing YoY
       CASE
           WHEN "Total Revenues (FY)" > "Total Revenues (-1FY)" THEN 1
           ELSE 0
           END                                                                                   AS revenue_growth_flag,
       -- Revenue stability: how close LTM is to 5Y average
       public.clamp_score(
               100 - ABS(public.safe_divide("Total Revenues (LTM)"::NUMERIC - "Total Revenues (5YAVGLTM)",
                                            "Total Revenues (5YAVGLTM)"::NUMERIC)) * 100
       )                                                                                         AS revenue_stability_score,
       -- Accelerating growth flag: recent growth > historical growth
       CASE
           WHEN public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) >
                public.pct_change("Total Revenues (-1FY)"::NUMERIC, "Total Revenues (-2FY)"::NUMERIC)
               THEN 1
           ELSE 0
           END                                                                                   AS revenue_accelerating_flag,
       -- Positive QoQ streak count
       (CASE WHEN "Total Revenues (FQ)" > "Total Revenues (-1FQFQ)" THEN 1 ELSE 0 END +
        CASE WHEN "Total Revenues (-1FQFQ)" > "Total Revenues (-2FQFQ)" THEN 1 ELSE 0 END +
        CASE WHEN "Total Revenues (-2FQFQ)" > "Total Revenues (-3FQFQ)" THEN 1 ELSE 0 END +
        CASE WHEN "Total Revenues (-3FQFQ)" > "Total Revenues (-4FQFQ)" THEN 1 ELSE 0 END)::INTEGER
                                                                                                 AS revenue_positive_qoq_streak
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_revenue_quarterly_features(text) owner to postgres;

