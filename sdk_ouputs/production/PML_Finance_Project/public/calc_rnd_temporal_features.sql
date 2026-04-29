create function calc_rnd_temporal_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                    text,
                rnd_ltm                 numeric,
                rnd_fq                  numeric,
                rnd_fy                  numeric,
                rnd_1fqfq               numeric,
                rnd_2fqfq               numeric,
                rnd_3fqfq               numeric,
                rnd_4fqfq               numeric,
                rnd_1fy                 numeric,
                rnd_2fy                 numeric,
                rnd_3fy                 numeric,
                rnd_4fy                 numeric,
                rnd_intensity_ltm       numeric,
                rnd_intensity_fy        numeric,
                rnd_intensity_trend     numeric,
                rnd_qoq_growth          numeric,
                rnd_yoy_growth          numeric,
                rnd_cagr_3y             numeric,
                rnd_per_employee        numeric,
                rnd_to_gross_profit     numeric,
                rnd_roi_proxy           numeric,
                rnd_increasing_flag     integer,
                rnd_cut_flag            integer,
                high_rnd_intensity_flag integer
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                                 AS isin,
       -- Current values
       "R&D Expenses (LTM)"                                                                   AS rnd_ltm,
       "R&D Expenses (FQ)"                                                                    AS rnd_fq,
       "R&D Expenses (FY)"                                                                    AS rnd_fy,
       -- Quarterly historical
       "R&D Expenses (-1FQFQ)"                                                                AS rnd_1fqfq,
       "R&D Expenses (-2FQFQ)"                                                                AS rnd_2fqfq,
       "R&D Expenses (-3FQFQ)"                                                                AS rnd_3fqfq,
       "R&D Expenses (-4FQFQ)"                                                                AS rnd_4fqfq,
       -- Yearly historical
       "R&D Expenses (-1FY)"                                                                  AS rnd_1fy,
       "R&D Expenses (-2FY)"                                                                  AS rnd_2fy,
       "R&D Expenses (-3FY)"                                                                  AS rnd_3fy,
       "R&D Expenses (-4FY)"                                                                  AS rnd_4fy,
       -- Intensity metrics (R&D / Revenue)
       public.safe_divide("R&D Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                                    AS rnd_intensity_ltm,
       public.safe_divide("R&D Expenses (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) * 100 AS rnd_intensity_fy,
       -- Intensity trend (increasing R&D commitment)
       (public.safe_divide("R&D Expenses (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) -
        public.safe_divide("R&D Expenses (-1FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC)) *
       100                                                                                    AS rnd_intensity_trend,
       -- Growth metrics
       public.pct_change("R&D Expenses (FQ)"::NUMERIC, "R&D Expenses (-1FQFQ)"::NUMERIC)      AS rnd_qoq_growth,
       public.pct_change("R&D Expenses (FY)"::NUMERIC, "R&D Expenses (-1FY)"::NUMERIC)        AS rnd_yoy_growth,
       CASE
           WHEN "R&D Expenses (-3FY)" > 0 AND "R&D Expenses (FY)" > 0
               THEN
               (POWER(public.safe_divide("R&D Expenses (FY)"::NUMERIC, "R&D Expenses (-3FY)"::NUMERIC), 1.0 / 3.0) -
                1) *
               100
           END                                                                                AS rnd_cagr_3y,
       -- Efficiency metrics
       public.safe_divide("R&D Expenses (FY)"::NUMERIC, "Full Time Employees (FY)"::NUMERIC)  AS rnd_per_employee,
       public.safe_divide("R&D Expenses (LTM)"::NUMERIC, "Gross Profit (LTM)"::NUMERIC) *
       100                                                                                    AS rnd_to_gross_profit,
       -- R&D ROI proxy: revenue growth relative to R&D spend
       CASE
           WHEN "R&D Expenses (-1FY)" > 0
               THEN public.safe_divide(
                   public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC),
                   public.safe_divide("R&D Expenses (-1FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) * 100
                    )
           END                                                                                AS rnd_roi_proxy,
       -- R&D increasing flag (4 consecutive quarterly increases)
       CASE
           WHEN "R&D Expenses (FQ)" > "R&D Expenses (-1FQFQ)"
               AND "R&D Expenses (-1FQFQ)" > "R&D Expenses (-2FQFQ)"
               AND "R&D Expenses (-2FQFQ)" > "R&D Expenses (-3FQFQ)"
               THEN 1
           ELSE 0 END                                                                         AS rnd_increasing_flag,
       -- R&D cut flag (significant decline may signal distress)
       CASE
           WHEN public.pct_change("R&D Expenses (FY)"::NUMERIC, "R&D Expenses (-1FY)"::NUMERIC) < -15
               THEN 1
           ELSE 0 END                                                                         AS rnd_cut_flag,
       -- High R&D intensity flag (tech/pharma typical >10%)
       CASE
           WHEN public.safe_divide("R&D Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) > 0.10
               THEN 1
           ELSE 0 END                                                                         AS high_rnd_intensity_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_rnd_temporal_features(text) owner to postgres;

