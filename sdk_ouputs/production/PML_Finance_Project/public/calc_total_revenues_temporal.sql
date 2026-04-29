create function calc_total_revenues_temporal(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                  text,
                revenue_fq            numeric,
                revenue_ltm           numeric,
                revenue_fy            numeric,
                revenue_1fy           numeric,
                revenue_5yavgfq       numeric,
                revenue_5yavgltm      numeric,
                revenue_growth_yoy    numeric,
                revenue_vs_5y_avg_fq  numeric,
                revenue_vs_5y_avg_ltm numeric,
                revenue_fq_vs_avg     numeric,
                revenue_momentum      numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                   AS isin,
       "Total Revenues (FQ)"                                    AS revenue_fq,
       "Total Revenues (LTM)"                                   AS revenue_ltm,
       "Total Revenues (FY)"                                    AS revenue_fy,
       "Total Revenues (-1FY)"                                  AS revenue_1fy,
       "Total Revenues (5YAVGFQ)"                               AS revenue_5yavgfq,
       "Total Revenues (5YAVGLTM)"                              AS revenue_5yavgltm,
       public.pct_change("Total Revenues (FY)"::NUMERIC,
                         "Total Revenues (-1FY)"::NUMERIC)      AS revenue_growth_yoy,
       public.safe_divide("Total Revenues (FQ)"::NUMERIC,
                          "Total Revenues (5YAVGFQ)"::NUMERIC)  AS revenue_vs_5y_avg_fq,
       public.safe_divide("Total Revenues (LTM)"::NUMERIC,
                          "Total Revenues (5YAVGLTM)"::NUMERIC) AS revenue_vs_5y_avg_ltm,
       public.safe_divide(("Total Revenues (FQ)"::NUMERIC - "Total Revenues (5YAVGFQ)"::NUMERIC),
                          "Total Revenues (5YAVGFQ)"::NUMERIC) * 100
                                                                AS revenue_fq_vs_avg,
       public.calc_change_ratio("Total Revenues (LTM)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) *
       100                                                      AS revenue_momentum
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_total_revenues_temporal(text) owner to postgres;

