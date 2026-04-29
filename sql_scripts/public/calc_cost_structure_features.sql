create function calc_cost_structure_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, cogs_to_revenue numeric, opex_to_revenue numeric, sga_to_revenue numeric, rnd_to_revenue numeric, interest_to_revenue numeric, sga_trend_yoy numeric, operating_leverage_proxy numeric, cost_efficiency_score numeric, marketing_to_revenue numeric, marketing_trend_yoy numeric, marketing_vs_5y_avg numeric, sga_vs_5y_avg numeric, sga_efficiency_trend numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                          AS isin,
       public.safe_divide("Cost Of Revenues (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                             AS cogs_to_revenue,
       public.safe_divide("Total Operating Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                             AS opex_to_revenue,
       public.safe_divide("Selling General & Admin Expenses/Total (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) * 100
                                                                                       AS sga_to_revenue,
       public.safe_divide("R&D Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                             AS rnd_to_revenue,
       public.safe_divide("Interest Expense/Total (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                             AS interest_to_revenue,
       -- SG&A trend using available FY columns
       (public.safe_divide("Selling General & Admin Expenses/Total (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) -
        public.safe_divide("Selling General & Admin Expenses/Total (-1FY)"::NUMERIC,
                           "Total Revenues (-1FY)"::NUMERIC)) * 100
                                                                                       AS sga_trend_yoy,
       CASE
           WHEN public.calc_change_ratio("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) > 0
               THEN public.safe_divide(
                   public.calc_change_ratio("Operating Income (FY)"::NUMERIC, "Operating Income (-1FY)"::NUMERIC),
                   public.calc_change_ratio("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC)
                    )
           END                                                                         AS operating_leverage_proxy,
       public.clamp_score(
               100 -
               public.safe_divide("Cost Of Revenues (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) * 100 * 0.5 -
               public.safe_divide("Total Operating Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) * 100 *
               0.3
       )                                                                               AS cost_efficiency_score,
       -- NEW: Marketing efficiency metrics using schema columns
       public.safe_divide("Marketing Expenses (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) *
       100                                                                             AS marketing_to_revenue,
       public.pct_change("Marketing Expenses (FY)"::NUMERIC,
                         "Marketing Expenses (-1FY)"::NUMERIC)                         AS marketing_trend_yoy,
       public.safe_divide("Marketing Expenses (FY)"::NUMERIC,
                          "Marketing Expenses (5YAVGLTM)"::NUMERIC)                    AS marketing_vs_5y_avg,
       -- NEW: SG&A vs 5Y average
       public.safe_divide("Selling General & Admin Expenses/Total (FQ)"::NUMERIC,
                          "Selling General & Admin Expenses/Total (5YAVGFQ)"::NUMERIC) AS sga_vs_5y_avg,
       -- NEW: SG&A efficiency trend (lower ratio = better efficiency)
       (public.safe_divide("Selling General & Admin Expenses/Total (-1FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) -
        public.safe_divide("Selling General & Admin Expenses/Total (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC)) *
       100
                                                                                       AS sga_efficiency_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_cost_structure_features(text) owner to postgres;

