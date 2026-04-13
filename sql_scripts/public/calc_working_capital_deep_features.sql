create function calc_working_capital_deep_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                 text,
                working_capital_ltm  numeric,
                working_capital_fq   numeric,
                working_capital_fy   numeric,
                wc_to_revenue        numeric,
                wc_to_assets         numeric,
                wc_change_qoq        numeric,
                wc_change_yoy        numeric,
                days_working_capital numeric,
                wc_efficiency_score  numeric,
                negative_wc_flag     integer,
                wc_improvement_flag  integer
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                              AS isin,
       "Working Capital (LTM)"                                             AS working_capital_ltm,
       "Working Capital (FQ)"                                              AS working_capital_fq,
       "Working Capital (FY)"                                              AS working_capital_fy,
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100   AS wc_to_revenue,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100     AS wc_to_assets,
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(ABS("Working Capital (FY)"), 0) * 100                        AS wc_change_qoq,
       ("Working Capital (FY)" - "Working Capital (-1FY)") /
       NULLIF(ABS("Working Capital (-1FY)"), 0) * 100                      AS wc_change_yoy,
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)" / 365.0, 0) AS days_working_capital,
       GREATEST(0, LEAST(100,
                         50 + (CASE WHEN "Working Capital (LTM)" > 0 THEN 25 ELSE -25 END) +
                         (CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 15 ELSE 0 END) +
                         (CASE WHEN ("Working Capital (FQ)" - "Working Capital (FY)") > 0 THEN 10 ELSE -10 END)
                   ))                                                      AS wc_efficiency_score,
       CASE WHEN "Working Capital (LTM)" < 0 THEN 1 ELSE 0 END             AS negative_wc_flag,
       CASE
           WHEN "Working Capital (FQ)" > "Working Capital (FY)"
               AND "Working Capital (FY)" > "Working Capital (-1FY)"
               THEN 1
           ELSE 0
           END                                                             AS wc_improvement_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_working_capital_deep_features(text) owner to postgres;

