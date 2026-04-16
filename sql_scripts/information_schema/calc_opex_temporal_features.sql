create function information_schema.calc_opex_temporal_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                     text,
                opex_fq                  numeric,
                opex_ltm                 numeric,
                opex_fy                  numeric,
                opex_qoq_growth          numeric,
                opex_yoy_growth          numeric,
                opex_vs_revenue_trend    numeric,
                sga_qoq_growth           numeric,
                sga_yoy_growth           numeric,
                operating_leverage_score numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN",
       "Total Operating Expenses (FQ)",
       "Total Operating Expenses (LTM)",
       "Total Operating Expenses (FY)",
       public.calc_change_ratio("Total Operating Expenses (FQ)",
                                "Total Operating Expenses (-1FQFQ)")             AS opex_qoq_growth,
       public.calc_change_ratio("Total Operating Expenses (FY)",
                                "Total Operating Expenses (-1FY)")               AS opex_yoy_growth,
       -- Change in opex/revenue ratio (FY vs -1FY)
       (public.safe_divide("Total Operating Expenses (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) -
        public.safe_divide("Total Operating Expenses (-1FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC)) *
       100                                                                       AS opex_vs_revenue_trend,
       public.calc_change_ratio("Selling General & Admin Expenses/Total (FQ)",
                                "Selling General & Admin Expenses/Total (-1FY)") AS sga_qoq_growth,
       public.calc_change_ratio("Selling General & Admin Expenses/Total (FY)",
                                "Selling General & Admin Expenses/Total (-1FY)") AS sga_yoy_growth,
       -- Operating leverage: revenue growth minus opex growth
       public.calc_change_ratio("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) -
       public.calc_change_ratio("Total Operating Expenses (FY)"::NUMERIC,
                                "Total Operating Expenses (-1FY)"::NUMERIC)      AS operating_leverage_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function information_schema.calc_opex_temporal_features(unknown) owner to postgres;

