create function calc_net_income_comprehensive(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, net_income_is_fq numeric, net_income_is_ltm numeric, net_income_is_fy numeric, net_income_adj_ltm numeric, normalized_ni_ltm numeric, net_income_is_1fqfq numeric, net_income_is_2fqfq numeric, net_income_is_3fqfq numeric, net_income_is_4fqfq numeric, net_income_is_1fy numeric, net_income_is_2fy numeric, net_income_is_3fy numeric, net_income_is_4fy numeric, net_income_is_5yavgfq numeric, net_income_is_5yavgltm numeric, normalized_ni_5yavgfq numeric, normalized_ni_5yavgltm numeric, net_income_growth_yoy numeric, net_income_margin_ltm numeric, ni_adjustment_ratio numeric, net_income_positive_years integer, earnings_quality_composite numeric, net_income_qoq_growth numeric, net_income_yoy_quarterly numeric, net_income_vs_5y_avg numeric, normalized_ni_vs_5y_avg numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                      AS isin,
       -- Base values
       "Net Income - (IS) (FQ)"                                    AS net_income_is_fq,
       "Net Income - (IS) (LTM)"                                   AS net_income_is_ltm,
       "Net Income - (IS) (FY)"                                    AS net_income_is_fy,
       "Net Income/Adj. (LTM)"                                     AS net_income_adj_ltm,
       "Normalized Net Income (LTM)"                               AS normalized_ni_ltm,
       -- Extended quarterly historical
       "Net Income - (IS) (-1FQFQ)"                                AS net_income_is_1fqfq,
       "Net Income - (IS) (-2FQFQ)"                                AS net_income_is_2fqfq,
       "Net Income - (IS) (-3FQFQ)"                                AS net_income_is_3fqfq,
       "Net Income - (IS) (-4FQFQ)"                                AS net_income_is_4fqfq,
       -- Extended yearly historical
       "Net Income - (IS) (-1FY)"                                  AS net_income_is_1fy,
       "Net Income - (IS) (-2FY)"                                  AS net_income_is_2fy,
       "Net Income - (IS) (-3FY)"                                  AS net_income_is_3fy,
       "Net Income - (IS) (-4FY)"                                  AS net_income_is_4fy,
       -- 5-year averages
       "Net Income - (IS) (5YAVGFQ)"                               AS net_income_is_5yavgfq,
       "Net Income - (IS) (5YAVGLTM)"                              AS net_income_is_5yavgltm,
       "Normalized Net Income (5YAVGFQ)"                           AS normalized_ni_5yavgfq,
       "Normalized Net Income (5YAVGLTM)"                          AS normalized_ni_5yavgltm,
       -- Derived metrics
       public.pct_change("Net Income - (IS) (FY)"::NUMERIC,
                         "Net Income - (IS) (-1FY)"::NUMERIC)      AS net_income_growth_yoy,
       "Net Income Margin % (LTM)"::NUMERIC                        AS net_income_margin_ltm,
       public.safe_divide("Net Income/Adj. (LTM)"::NUMERIC,
                          "Net Income - (IS) (LTM)"::NUMERIC)      AS ni_adjustment_ratio,
       (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN "Net Income - (IS) (-4FY)" > 0 THEN 1
            ELSE 0 END)::INTEGER                                   AS net_income_positive_years,
       public.clamp_score(
               50 +
               (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 10 ELSE -10 END) +
               (CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 5 ELSE -5 END) +
               (CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 5 ELSE -5 END) +
               (CASE
                    WHEN ABS(public.safe_divide(("Net Income/Adj. (LTM)"::NUMERIC - "Net Income - (IS) (LTM)"::NUMERIC),
                                                "Net Income - (IS) (LTM)"::NUMERIC)) < 0.10 THEN 15
                    ELSE -15 END) +
               (CASE WHEN "Net Income - (IS) (FY)" > "Net Income - (IS) (-1FY)" THEN 10 ELSE -5 END) +
               (CASE WHEN "Net Income - (IS) (-1FY)" > "Net Income - (IS) (-2FY)" THEN 5 ELSE -5 END)
       )                                                           AS earnings_quality_composite,
       -- Quarterly trends
       public.pct_change("Net Income - (IS) (FQ)"::NUMERIC,
                         "Net Income - (IS) (-1FQFQ)"::NUMERIC)    AS net_income_qoq_growth,
       public.pct_change("Net Income - (IS) (FQ)"::NUMERIC,
                         "Net Income - (IS) (-4FQFQ)"::NUMERIC)    AS net_income_yoy_quarterly,
       -- vs 5Y averages
       public.safe_divide("Net Income - (IS) (LTM)"::NUMERIC,
                          "Net Income - (IS) (5YAVGLTM)"::NUMERIC) AS net_income_vs_5y_avg,
       public.safe_divide("Normalized Net Income (LTM)"::NUMERIC, "Normalized Net Income (5YAVGLTM)"::NUMERIC)
                                                                   AS normalized_ni_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_net_income_comprehensive(text) owner to postgres;

