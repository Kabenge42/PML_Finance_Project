create function calc_total_assets_temporal(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin               text,
                assets_fq          numeric,
                assets_fy          numeric,
                assets_ltm         numeric,
                assets_1fq         numeric,
                assets_2fq         numeric,
                assets_3fq         numeric,
                assets_4fq         numeric,
                assets_1fy         numeric,
                assets_2fy         numeric,
                assets_3fy         numeric,
                assets_4fy         numeric,
                assets_qoq_growth  numeric,
                assets_yoy_growth  numeric,
                assets_3y_cagr     numeric,
                asset_growth_accel numeric,
                asset_base_stable  integer
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                            AS isin,
       "Total Assets (FQ)"                                                               AS assets_fq,
       "Total Assets (FY)"                                                               AS assets_fy,
       "Total Assets (LTM)"                                                              AS assets_ltm,
       "Total Assets (-1FQ)"                                                             AS assets_1fq,
       "Total Assets (-2FQ)"                                                             AS assets_2fq,
       "Total Assets (-3FQ)"                                                             AS assets_3fq,
       "Total Assets (-4FQ)"                                                             AS assets_4fq,
       "Total Assets (-1FY)"                                                             AS assets_1fy,
       "Total Assets (-2FY)"                                                             AS assets_2fy,
       "Total Assets (-3FY)"                                                             AS assets_3fy,
       "Total Assets (-4FY)"                                                             AS assets_4fy,
       public.pct_change("Total Assets (FQ)"::NUMERIC, "Total Assets (-1FQ)"::NUMERIC)   AS assets_qoq_growth,
       public.pct_change("Total Assets (FY)"::NUMERIC, "Total Assets (-1FY)"::NUMERIC)   AS assets_yoy_growth,
       CASE
           WHEN "Total Assets (-3FY)" > 0
               THEN
               (POWER(public.safe_divide("Total Assets (FY)"::NUMERIC, "Total Assets (-3FY)"::NUMERIC), 1.0 / 3.0) -
                1) *
               100
           END                                                                           AS assets_3y_cagr,
       -- Growth acceleration: recent growth vs historical
       public.pct_change("Total Assets (FY)"::NUMERIC, "Total Assets (-1FY)"::NUMERIC) -
       public.pct_change("Total Assets (-1FY)"::NUMERIC, "Total Assets (-2FY)"::NUMERIC) AS asset_growth_accel,
       -- Stability flag: growing consistently
       CASE
           WHEN "Total Assets (FY)" >= "Total Assets (-1FY)"
               AND "Total Assets (-1FY)" >= "Total Assets (-2FY)"
               AND "Total Assets (-2FY)" >= "Total Assets (-3FY)"
               THEN 1
           ELSE 0 END                                                                    AS asset_base_stable
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_total_assets_temporal(text) owner to postgres;

