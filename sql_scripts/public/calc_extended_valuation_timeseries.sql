create function calc_extended_valuation_timeseries(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, ev_sales_qoq_1q numeric, ev_sales_qoq_2q numeric, ev_sales_qoq_3q numeric, ev_sales_qoq_4q numeric, p_e_vs_5y_avg numeric, p_e_percentile_proxy numeric, valuation_mean_reversion numeric, ev_ebitda_qoq_trend numeric, p_b_momentum_yoy numeric, valuation_compression numeric, forward_pe_premium numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                      AS isin,
       public.calc_change_ratio("EV/Sales (LTM)", "EV/Sales (-1FQLTM)")            AS ev_sales_qoq_1q,
       public.calc_change_ratio("EV/Sales (-1FQLTM)", "EV/Sales (-2FQLTM)")        AS ev_sales_qoq_2q,
       public.calc_change_ratio("EV/Sales (-2FQLTM)", "EV/Sales (-3FQLTM)")        AS ev_sales_qoq_3q,
       public.calc_change_ratio("EV/Sales (-3FQLTM)", "EV/Sales (-4FQLTM)")        AS ev_sales_qoq_4q,
       public.calc_change_ratio("P/E (LTM)", "P/E (5YAVGLTM)")                     AS p_e_vs_5y_avg,
       CASE
           WHEN "P/E (LTM)" IS NOT NULL AND "P/E (3YAVGLTM)" IS NOT NULL
               THEN ("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF(ABS("P/E (3YAVGLTM)") * 0.5, 0)
           END                                                                     AS p_e_percentile_proxy,
       (public.calc_change_ratio("P/E (LTM)", "P/E (3YAVGLTM)") +
        public.calc_change_ratio("EV/Sales (LTM)", "EV/Sales (3YAVGLTM)") +
        public.calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (3YAVGLTM)")) / 3.0
                                                                                   AS valuation_mean_reversion,
       public.calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (-1FQLTM)")          AS ev_ebitda_qoq_trend,
       public.calc_change_ratio("P/B (LTM)", "P/B (-1FY)")                         AS p_b_momentum_yoy,
       (public.safe_divide("P/E (LTM)", "P/E (3YAVGLTM)") +
        public.safe_divide("EV/EBITDA (LTM)", "EV/EBITDA (3YAVGLTM)")) / 2.0 - 1.0 AS valuation_compression,
       public.calc_change_ratio("P/E (EST FY1)", "P/E (LTM)") * 100                AS forward_pe_premium
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_extended_valuation_timeseries(text) owner to postgres;

