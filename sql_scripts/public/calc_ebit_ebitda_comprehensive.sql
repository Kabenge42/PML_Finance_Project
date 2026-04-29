create function calc_ebit_ebitda_comprehensive(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, ebit_fq numeric, ebit_ltm numeric, ebit_fy numeric, ebit_1fy numeric, ebitda_fq numeric, ebitda_ltm numeric, ebitda_fy numeric, ebitda_1fy numeric, ebit_2fy numeric, ebit_3fy numeric, ebit_4fy numeric, ebitda_2fy numeric, ebitda_3fy numeric, ebitda_4fy numeric, ebit_1fqfq numeric, ebit_2fqfq numeric, ebit_3fqfq numeric, ebit_4fqfq numeric, ebitda_1fqfq numeric, ebitda_2fqfq numeric, ebitda_3fqfq numeric, ebitda_4fqfq numeric, ebit_5yavgfq numeric, ebit_5yavgltm numeric, ebitda_5yavgfq numeric, ebitda_5yavgltm numeric, ebit_adj_fq numeric, ebit_adj_ltm numeric, ebit_adj_fy numeric, ebitda_adj_fq numeric, ebitda_adj_ltm numeric, ebitda_adj_fy numeric, ebit_growth_yoy numeric, ebitda_growth_yoy numeric, ebit_margin_ltm numeric, ebitda_margin_ltm numeric, ebit_positive_years integer, ebitda_positive_years integer, ebit_qoq_growth numeric, ebitda_qoq_growth numeric, ebit_cagr_3y numeric, ebitda_cagr_3y numeric, ebit_vs_5y_avg numeric, ebitda_vs_5y_avg numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                    AS isin,
       -- Current values
       "EBIT (FQ)"                                                               AS ebit_fq,
       "EBIT (LTM)"                                                              AS ebit_ltm,
       "EBIT (FY)"                                                               AS ebit_fy,
       "EBIT (-1FY)"                                                             AS ebit_1fy,
       "EBITDA (FQ)"                                                             AS ebitda_fq,
       "EBITDA (LTM)"                                                            AS ebitda_ltm,
       "EBITDA (FY)"                                                             AS ebitda_fy,
       "EBITDA (-1FY)"                                                           AS ebitda_1fy,
       -- Extended historical FY
       "EBIT (-2FY)"                                                             AS ebit_2fy,
       "EBIT (-3FY)"                                                             AS ebit_3fy,
       "EBIT (-4FY)"                                                             AS ebit_4fy,
       "EBITDA (-2FY)"                                                           AS ebitda_2fy,
       "EBITDA (-3FY)"                                                           AS ebitda_3fy,
       "EBITDA (-4FY)"                                                           AS ebitda_4fy,
       -- Quarterly historical
       "EBIT (-1FQFQ)"                                                           AS ebit_1fqfq,
       "EBIT (-2FQFQ)"                                                           AS ebit_2fqfq,
       "EBIT (-3FQFQ)"                                                           AS ebit_3fqfq,
       "EBIT (-4FQFQ)"                                                           AS ebit_4fqfq,
       "EBITDA (-1FQFQ)"                                                         AS ebitda_1fqfq,
       "EBITDA (-2FQFQ)"                                                         AS ebitda_2fqfq,
       "EBITDA (-3FQFQ)"                                                         AS ebitda_3fqfq,
       "EBITDA (-4FQFQ)"                                                         AS ebitda_4fqfq,
       -- 5-year averages
       "EBIT (5YAVGFQ)"                                                          AS ebit_5yavgfq,
       "EBIT (5YAVGLTM)"                                                         AS ebit_5yavgltm,
       "EBITDA (5YAVGFQ)"                                                        AS ebitda_5yavgfq,
       "EBITDA (5YAVGLTM)"                                                       AS ebitda_5yavgltm,
       -- Adjusted variants
       "EBIT/Adj. (FQ)"                                                          AS ebit_adj_fq,
       "EBIT/Adj. (LTM)"                                                         AS ebit_adj_ltm,
       "EBIT/Adj. (FY)"                                                          AS ebit_adj_fy,
       "EBITDA/Adj. (FQ)"                                                        AS ebitda_adj_fq,
       "EBITDA/Adj. (LTM)"                                                       AS ebitda_adj_ltm,
       "EBITDA/Adj. (FY)"                                                        AS ebitda_adj_fy,
       -- Derived metrics
       ("EBIT (FY)" - "EBIT (-1FY)") / NULLIF(ABS("EBIT (-1FY)"), 0) * 100       AS ebit_growth_yoy,
       ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100 AS ebitda_growth_yoy,
       "EBIT (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                    AS ebit_margin_ltm,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                  AS ebitda_margin_ltm,
       (CASE WHEN "EBIT (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER                  AS ebit_positive_years,
       (CASE WHEN "EBITDA (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER                AS ebitda_positive_years,
       -- NEW: Quarterly momentum
       ("EBIT (FQ)" - "EBIT (-1FQFQ)") / NULLIF(ABS("EBIT (-1FQFQ)"), 0) * 100   AS ebit_qoq_growth,
       ("EBITDA (FQ)" - "EBITDA (-1FQFQ)") / NULLIF(ABS("EBITDA (-1FQFQ)"), 0) * 100
                                                                                 AS ebitda_qoq_growth,
       -- NEW: Multi-year CAGR (3-year)
       CASE
           WHEN "EBIT (-3FY)" > 0 AND "EBIT (FY)" > 0
               THEN (POWER("EBIT (FY)" / NULLIF("EBIT (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebit_cagr_3y,
       CASE
           WHEN "EBITDA (-3FY)" > 0 AND "EBITDA (FY)" > 0
               THEN (POWER("EBITDA (FY)" / NULLIF("EBITDA (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebitda_cagr_3y,
       -- NEW: vs 5Y average
       "EBIT (LTM)" / NULLIF("EBIT (5YAVGLTM)", 0)                               AS ebit_vs_5y_avg,
       "EBITDA (LTM)" / NULLIF("EBITDA (5YAVGLTM)", 0)                           AS ebitda_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_ebit_ebitda_comprehensive(text) owner to postgres;

