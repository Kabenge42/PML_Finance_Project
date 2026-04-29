create function calc_gross_profit_temporal(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, gp_fq numeric, gp_fy numeric, gp_ltm numeric, gp_1fqfq numeric, gp_2fqfq numeric, gp_3fqfq numeric, gp_4fqfq numeric, gp_1fy numeric, gp_2fy numeric, gp_3fy numeric, gp_4fy numeric, gp_qoq_growth numeric, gp_yoy_growth numeric, gp_margin_fq numeric, gp_margin_trend numeric, gp_positive_quarters integer, gp_margin_expansion integer)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                              AS isin,
       "Gross Profit (FQ)"                                 AS gp_fq,
       "Gross Profit (FY)"                                 AS gp_fy,
       "Gross Profit (LTM)"                                AS gp_ltm,
       "Gross Profit (-1FQFQ)"                             AS gp_1fqfq,
       "Gross Profit (-2FQFQ)"                             AS gp_2fqfq,
       "Gross Profit (-3FQFQ)"                             AS gp_3fqfq,
       "Gross Profit (-4FQFQ)"                             AS gp_4fqfq,
       "Gross Profit (-1FY)"                               AS gp_1fy,
       "Gross Profit (-2FY)"                               AS gp_2fy,
       "Gross Profit (-3FY)"                               AS gp_3fy,
       "Gross Profit (-4FY)"                               AS gp_4fy,
       public.pct_change("Gross Profit (FQ)"::NUMERIC,
                         "Gross Profit (-1FQFQ)"::NUMERIC) AS gp_qoq_growth,
       public.pct_change("Gross Profit (FY)"::NUMERIC,
                         "Gross Profit (-1FY)"::NUMERIC)   AS gp_yoy_growth,
       public.safe_divide("Gross Profit (FQ)"::NUMERIC, "Total Revenues (FQ)"::NUMERIC) *
       100                                                 AS gp_margin_fq,
       (public.safe_divide("Gross Profit (FQ)"::NUMERIC, "Total Revenues (FQ)"::NUMERIC) -
        public.safe_divide("Gross Profit (-4FQFQ)"::NUMERIC, "Total Revenues (5YAVGFQ)"::NUMERIC)) *
       100                                                 AS gp_margin_trend,
       (CASE WHEN "Gross Profit (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN "Gross Profit (-4FQFQ)" > 0 THEN 1
            ELSE 0 END)::INTEGER                           AS gp_positive_quarters,
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)"
               THEN 1
           ELSE 0 END                                      AS gp_margin_expansion
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_gross_profit_temporal(text) owner to postgres;

