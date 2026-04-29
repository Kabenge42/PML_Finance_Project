create function calc_cashflow_comprehensive(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                    text,
                cfo_fq                  numeric,
                cfo_ltm                 numeric,
                cfo_fy                  numeric,
                fcf_fq                  numeric,
                fcf_ltm                 numeric,
                fcf_fy                  numeric,
                cfo_growth_yoy          numeric,
                fcf_growth_yoy          numeric,
                cfo_to_net_income       numeric,
                fcf_margin              numeric,
                fcf_yield               numeric,
                cfo_positive_years      integer,
                fcf_positive_years      integer,
                cash_flow_quality_score numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                           AS isin,
       "CFO (FQ)"                                                       AS cfo_fq,
       "CFO (LTM)"                                                      AS cfo_ltm,
       "CFO (FY)"                                                       AS cfo_fy,
       "FCF (FQ)"                                                       AS fcf_fq,
       "FCF (LTM)"                                                      AS fcf_ltm,
       "FCF (FY)"                                                       AS fcf_fy,
       ("CFO (FY)" - "CFO (-1FY)") / NULLIF(ABS("CFO (-1FY)"), 0) * 100 AS cfo_growth_yoy,
       ("FCF (FY)" - "FCF (-1FY)") / NULLIF(ABS("FCF (-1FY)"), 0) * 100 AS fcf_growth_yoy,
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)               AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100            AS fcf_margin,
       "FCF (LTM)" / NULLIF("Market Cap", 0) * 100                      AS fcf_yield,
       (CASE WHEN "CFO (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER          AS cfo_positive_years,
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER          AS fcf_positive_years,
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
                AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0 THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC          AS cash_flow_quality_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_cashflow_comprehensive(text) owner to postgres;

