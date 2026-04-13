create function calc_valuation_timeseries_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                       text,
                ev_sales_trend_1y          numeric,
                ev_ebitda_momentum         numeric,
                p_e_momentum_yoy           numeric,
                p_e_momentum_qoq           numeric,
                ev_sales_vs_3y_avg         numeric,
                ev_ebitda_vs_3y_avg        numeric,
                p_e_vs_3y_avg              numeric,
                ev_sales_forward_discount  numeric,
                ev_ebitda_forward_discount numeric,
                p_e_forward_discount       numeric,
                p_b_vs_5y_avg              numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                                AS isin,
       public.calc_change_ratio("EV/Sales (LTM)"::NUMERIC, "EV/Sales (-1FYLTM)"::NUMERIC)    AS ev_sales_trend_1y,
       public.calc_change_ratio("EV/EBITDA (LTM)"::NUMERIC, "EV/EBITDA (-1FYLTM)"::NUMERIC)  AS ev_ebitda_momentum,
       public.calc_change_ratio("P/E (LTM)"::NUMERIC, "P/E (-1FYLTM)"::NUMERIC)              AS p_e_momentum_yoy,
       public.calc_change_ratio("P/E (LTM)"::NUMERIC, "P/E (-1FQLTM)"::NUMERIC)              AS p_e_momentum_qoq,
       public.calc_change_ratio("EV/Sales (LTM)"::NUMERIC, "EV/Sales (3YAVGLTM)"::NUMERIC)   AS ev_sales_vs_3y_avg,
       public.calc_change_ratio("EV/EBITDA (LTM)"::NUMERIC, "EV/EBITDA (3YAVGLTM)"::NUMERIC) AS ev_ebitda_vs_3y_avg,
       public.calc_change_ratio("P/E (LTM)"::NUMERIC, "P/E (3YAVGLTM)"::NUMERIC)             AS p_e_vs_3y_avg,
       public.calc_change_ratio("EV/Sales (NTM)"::NUMERIC,
                                "EV/Sales (LTM)"::NUMERIC)                                   AS ev_sales_forward_discount,
       public.calc_change_ratio("EV/EBITDA (NTM)"::NUMERIC,
                                "EV/EBITDA (LTM)"::NUMERIC)                                  AS ev_ebitda_forward_discount,
       public.calc_change_ratio("P/E (EST FY1)"::NUMERIC, "P/E (LTM)"::NUMERIC)              AS p_e_forward_discount,
       public.safe_divide("P/B (LTM)"::NUMERIC, "P/B (5YAVG)"::NUMERIC)                      AS p_b_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_valuation_timeseries_features(text) owner to postgres;

