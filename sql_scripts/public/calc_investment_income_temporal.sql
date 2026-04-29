create function calc_investment_income_temporal(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, inv_income_ltm numeric, inv_income_fq numeric, inv_income_fy numeric, inv_income_qoq_growth numeric, inv_income_yoy_growth numeric, inv_income_to_revenue numeric, inv_income_trend_3y numeric, inv_income_positive_quarters integer, financial_company_proxy integer)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN",
       "Interest And Investment Income (LTM)",
       "Interest And Investment Income (FQ)",
       "Interest And Investment Income (FY)",
       public.calc_change_ratio("Interest And Investment Income (FQ)",
                                "Interest And Investment Income (-1FQFQ)")                AS inv_income_qoq_growth,
       public.calc_change_ratio("Interest And Investment Income (FY)",
                                "Interest And Investment Income (-1FY)")                  AS inv_income_yoy_growth,
       public.safe_divide("Interest And Investment Income (LTM)", "Total Revenues (LTM)") AS inv_income_to_revenue,
       CASE
           WHEN "Interest And Investment Income (-3FY)" > 0 AND "Interest And Investment Income (FY)" > 0
               THEN (POWER(public.safe_divide("Interest And Investment Income (FY)",
                                              "Interest And Investment Income (-3FY)"), 1.0 / 3.0) - 1) * 100
           END                                                                            AS inv_income_trend_3y,
       (CASE WHEN "Interest And Investment Income (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Interest And Investment Income (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Interest And Investment Income (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Interest And Investment Income (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN "Interest And Investment Income (-4FQFQ)" > 0 THEN 1
            ELSE 0 END)::INTEGER                                                          AS inv_income_positive_quarters,
       CASE
           WHEN public.safe_divide("Interest And Investment Income (LTM)", "Total Revenues (LTM)") > 0.2
               THEN 1
           ELSE 0 END                                                                     AS financial_company_proxy
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_investment_income_temporal(text) owner to postgres;

