create function calc_gaap_adjusted_analytics(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, eps_adjustment_spread_ltm numeric, eps_adjustment_spread_fy numeric, eps_adjustment_spread_1fy numeric, eps_adjustment_spread_fq numeric, eps_adjustment_spread_1fqfq numeric, eps_adjustment_spread_2fqfq numeric, eps_adjustment_spread_3fqfq numeric, eps_adjustment_spread_4fqfq numeric, eps_adjustment_spread_2fy numeric, eps_adjustment_spread_3fy numeric, eps_adjustment_spread_4fy numeric, eps_adjustment_pct numeric, net_income_adjustment_ratio_ltm numeric, net_income_adjustment_ratio_fy numeric, net_income_adjustment_ratio_1fy numeric, net_income_adjustment_ratio_fq numeric, net_income_adjustment_ratio_5yavgfq numeric, net_income_adjustment_ratio_1fqfq numeric, net_income_adjustment_ratio_2fqfq numeric, net_income_adjustment_ratio_3fqfq numeric, net_income_adjustment_ratio_4fqfq numeric, net_income_adjustment_ratio_2fy numeric, net_income_adjustment_ratio_3fy numeric, net_income_adjustment_ratio_4fy numeric, net_income_adjustment_pct numeric, ebitda_adjustment_pct_ltm numeric, ebitda_adjustment_pct_fy numeric, ebitda_adjustment_pct_1fy numeric, ebitda_adjustment_pct_fq numeric, ebitda_adjustment_pct_1fqfq numeric, ebitda_adjustment_pct_2fqfq numeric, ebitda_adjustment_pct_3fqfq numeric, ebitda_adjustment_pct_4fqfq numeric, ebitda_adjustment_pct_2fy numeric, ebitda_adjustment_pct_3fy numeric, ebitda_adjustment_pct_4fy numeric, ebit_adjustment_pct_ltm numeric, ebit_adjustment_pct_fy numeric, ebit_adjustment_pct_1fy numeric, ebit_adjustment_pct_fq numeric, ebit_adjustment_pct_1fqfq numeric, ebit_adjustment_pct_2fqfq numeric, ebit_adjustment_pct_3fqfq numeric, ebit_adjustment_pct_4fqfq numeric, ebit_adjustment_pct_2fy numeric, ebit_adjustment_pct_3fy numeric, ebit_adjustment_pct_4fy numeric, earnings_quality_score numeric, earnings_quality_warning integer, forward_eps_gaap_adj_spread numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                       AS isin,
       -- EPS Adjustment Spreads (EPS/Adj. - Net EPS - Basic)
       "EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)"                                   AS eps_adjustment_spread_ltm,
       "EPS/Adj. (FY)" - "Net EPS - Basic (FY)"                                     AS eps_adjustment_spread_fy,
       "EPS/Adj. (-1FY)" - "Net EPS - Basic (-1FY)"                                 AS eps_adjustment_spread_1fy,
       "EPS/Adj. (FQ)" - "Net EPS - Basic (FQ)"                                     AS eps_adjustment_spread_fq,
       "EPS/Adj. (-1FQFQ)" - "Net EPS - Basic (-1FQFQ)"                             AS eps_adjustment_spread_1fqfq,
       "EPS/Adj. (-2FQFQ)" - "Net EPS - Basic (-2FQFQ)"                             AS eps_adjustment_spread_2fqfq,
       "EPS/Adj. (-3FQFQ)" - "Net EPS - Basic (-3FQFQ)"                             AS eps_adjustment_spread_3fqfq,
       "EPS/Adj. (-4FQFQ)" - "Net EPS - Basic (-4FQFQ)"                             AS eps_adjustment_spread_4fqfq,
       "EPS/Adj. (-2FY)" - "Net EPS - Basic (-2FY)"                                 AS eps_adjustment_spread_2fy,
       "EPS/Adj. (-3FY)" - "Net EPS - Basic (-3FY)"                                 AS eps_adjustment_spread_3fy,
       "EPS/Adj. (-4FY)" - "Net EPS - Basic (-4FY)"                                 AS eps_adjustment_spread_4fy,
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
       NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100                                AS eps_adjustment_pct,

       -- Net Income Adjustment Ratios (Net Income/Adj. / Net Income - (IS))
       "Net Income/Adj. (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)               AS net_income_adjustment_ratio_ltm,
       "Net Income/Adj. (FY)" / NULLIF("Net Income - (IS) (FY)", 0)                 AS net_income_adjustment_ratio_fy,
       "Net Income/Adj. (-1FY)" / NULLIF("Net Income - (IS) (-1FY)", 0)             AS net_income_adjustment_ratio_1fy,
       "Net Income/Adj. (FQ)" / NULLIF("Net Income - (IS) (FQ)", 0)                 AS net_income_adjustment_ratio_fq,
       "Net Income/Adj. (5YAVGFQ)" / NULLIF("Net Income - (IS) (5YAVGFQ)", 0)       AS net_income_adjustment_ratio_5yavgfq,
       "Net Income/Adj. (-1FQFQ)" / NULLIF("Net Income - (IS) (-1FQFQ)", 0)         AS net_income_adjustment_ratio_1fqfq,
       "Net Income/Adj. (-2FQFQ)" / NULLIF("Net Income - (IS) (-2FQFQ)", 0)         AS net_income_adjustment_ratio_2fqfq,
       "Net Income/Adj. (-3FQFQ)" / NULLIF("Net Income - (IS) (-3FQFQ)", 0)         AS net_income_adjustment_ratio_3fqfq,
       "Net Income/Adj. (-4FQFQ)" / NULLIF("Net Income - (IS) (-4FQFQ)", 0)         AS net_income_adjustment_ratio_4fqfq,
       "Net Income/Adj. (-2FY)" / NULLIF("Net Income - (IS) (-2FY)", 0)             AS net_income_adjustment_ratio_2fy,
       "Net Income/Adj. (-3FY)" / NULLIF("Net Income - (IS) (-3FY)", 0)             AS net_income_adjustment_ratio_3fy,
       "Net Income/Adj. (-4FY)" / NULLIF("Net Income - (IS) (-4FY)", 0)             AS net_income_adjustment_ratio_4fy,
       ("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0) *
       100                                                                          AS net_income_adjustment_pct,

       -- EBITDA Adjustment Percentages (EBITDA/Adj. - EBITDA) / |EBITDA| * 100
       ("EBITDA/Adj. (LTM)" - "EBITDA (LTM)") / NULLIF(ABS("EBITDA (LTM)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_ltm,
       ("EBITDA/Adj. (FY)" - "EBITDA (FY)") / NULLIF(ABS("EBITDA (FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_fy,
       ("EBITDA/Adj. (-1FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_1fy,
       ("EBITDA/Adj. (FQ)" - "EBITDA (FQ)") / NULLIF(ABS("EBITDA (FQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_fq,
       ("EBITDA/Adj. (-1FQFQ)" - "EBITDA (-1FQFQ)") / NULLIF(ABS("EBITDA (-1FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_1fqfq,
       ("EBITDA/Adj. (-2FQFQ)" - "EBITDA (-2FQFQ)") / NULLIF(ABS("EBITDA (-2FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_2fqfq,
       ("EBITDA/Adj. (-3FQFQ)" - "EBITDA (-3FQFQ)") / NULLIF(ABS("EBITDA (-3FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_3fqfq,
       ("EBITDA/Adj. (-4FQFQ)" - "EBITDA (-4FQFQ)") / NULLIF(ABS("EBITDA (-4FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_4fqfq,
       ("EBITDA/Adj. (-2FY)" - "EBITDA (-2FY)") / NULLIF(ABS("EBITDA (-2FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_2fy,
       ("EBITDA/Adj. (-3FY)" - "EBITDA (-3FY)") / NULLIF(ABS("EBITDA (-3FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_3fy,
       ("EBITDA/Adj. (-4FY)" - "EBITDA (-4FY)") / NULLIF(ABS("EBITDA (-4FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_4fy,

       -- EBIT Adjustment Percentages (EBIT/Adj. - EBIT) / |EBIT| * 100
       ("EBIT/Adj. (LTM)" - "EBIT (LTM)") / NULLIF(ABS("EBIT (LTM)"), 0) *
       100                                                                          AS ebit_adjustment_pct_ltm,
       ("EBIT/Adj. (FY)" - "EBIT (FY)") / NULLIF(ABS("EBIT (FY)"), 0) * 100         AS ebit_adjustment_pct_fy,
       ("EBIT/Adj. (-1FY)" - "EBIT (-1FY)") / NULLIF(ABS("EBIT (-1FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_1fy,
       ("EBIT/Adj. (FQ)" - "EBIT (FQ)") / NULLIF(ABS("EBIT (FQ)"), 0) * 100         AS ebit_adjustment_pct_fq,
       ("EBIT/Adj. (-1FQFQ)" - "EBIT (-1FQFQ)") / NULLIF(ABS("EBIT (-1FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_1fqfq,
       ("EBIT/Adj. (-2FQFQ)" - "EBIT (-2FQFQ)") / NULLIF(ABS("EBIT (-2FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_2fqfq,
       ("EBIT/Adj. (-3FQFQ)" - "EBIT (-3FQFQ)") / NULLIF(ABS("EBIT (-3FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_3fqfq,
       ("EBIT/Adj. (-4FQFQ)" - "EBIT (-4FQFQ)") / NULLIF(ABS("EBIT (-4FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_4fqfq,
       ("EBIT/Adj. (-2FY)" - "EBIT (-2FY)") / NULLIF(ABS("EBIT (-2FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_2fy,
       ("EBIT/Adj. (-3FY)" - "EBIT (-3FY)") / NULLIF(ABS("EBIT (-3FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_3fy,
       ("EBIT/Adj. (-4FY)" - "EBIT (-4FY)") / NULLIF(ABS("EBIT (-4FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_4fy,

       -- Quality Scores (based on LTM EPS adjustment)
       GREATEST(0, LEAST(100,
                         100 - ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                                   NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100))) AS earnings_quality_score,
       CASE
           WHEN ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                    NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100) > 15
               THEN 1
           ELSE 0
           END                                                                      AS earnings_quality_warning,
       "EPS Norm - Est Avg (FY1E)" - "EPS GAAP - Est Avg (FY1E)"                    AS forward_eps_gaap_adj_spread
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_gaap_adjusted_analytics(text) owner to postgres;

