create function calc_goodwill_temporal_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                       text,
                goodwill_fq                numeric,
                goodwill_ltm               numeric,
                goodwill_fy                numeric,
                goodwill_1fq               numeric,
                goodwill_2fq               numeric,
                goodwill_3fq               numeric,
                goodwill_4fq               numeric,
                goodwill_1fy               numeric,
                goodwill_2fy               numeric,
                goodwill_3fy               numeric,
                goodwill_4fy               numeric,
                goodwill_qoq_change        numeric,
                goodwill_yoy_change        numeric,
                goodwill_3y_growth         numeric,
                goodwill_vs_5y_avg         numeric,
                recent_acquisition_flag    integer,
                goodwill_accumulation_rate numeric,
                goodwill_to_assets_trend   numeric,
                impairment_risk_score      numeric,
                goodwill_concentration     numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                             AS isin,
       -- Current values
       "Goodwill (FQ)"                                                                    AS goodwill_fq,
       "Goodwill (LTM)"                                                                   AS goodwill_ltm,
       "Goodwill (FY)"                                                                    AS goodwill_fy,
       -- Quarterly historical
       "Goodwill (-1FQ)"                                                                  AS goodwill_1fq,
       "Goodwill (-2FQ)"                                                                  AS goodwill_2fq,
       "Goodwill (-3FQ)"                                                                  AS goodwill_3fq,
       "Goodwill (-4FQ)"                                                                  AS goodwill_4fq,
       -- Yearly historical
       "Goodwill (-1FY)"                                                                  AS goodwill_1fy,
       "Goodwill (-2FY)"                                                                  AS goodwill_2fy,
       "Goodwill (-3FY)"                                                                  AS goodwill_3fy,
       "Goodwill (-4FY)"                                                                  AS goodwill_4fy,
       -- Trend metrics
       public.pct_change("Goodwill (FQ)"::NUMERIC, "Goodwill (-1FQ)"::NUMERIC)            AS goodwill_qoq_change,
       public.pct_change("Goodwill (FY)"::NUMERIC, "Goodwill (-1FY)"::NUMERIC)            AS goodwill_yoy_change,
       public.pct_change("Goodwill (FY)"::NUMERIC, "Goodwill (-3FY)"::NUMERIC)            AS goodwill_3y_growth,
       public.safe_divide("Goodwill (FQ)"::NUMERIC, "Goodwill (5YAVGFQ)"::NUMERIC)        AS goodwill_vs_5y_avg,
       -- Recent acquisition flag (goodwill increased significantly)
       CASE
           WHEN public.pct_change("Goodwill (FQ)"::NUMERIC, "Goodwill (-1FQ)"::NUMERIC) > 20
               THEN 1
           ELSE 0 END                                                                     AS recent_acquisition_flag,
       -- Goodwill accumulation rate (avg annual increase)
       CASE
           WHEN "Goodwill (-3FY)" > 0
               THEN (POWER(public.safe_divide("Goodwill (FY)"::NUMERIC, "Goodwill (-3FY)"::NUMERIC), 1.0 / 3.0) - 1) *
                    100
           END                                                                            AS goodwill_accumulation_rate,
       -- Goodwill to assets trend (increasing concentration risk)
       (public.safe_divide("Goodwill (FY)"::NUMERIC, "Total Assets (FY)"::NUMERIC) -
        public.safe_divide("Goodwill (-1FY)"::NUMERIC, "Total Assets (-1FY)"::NUMERIC)) *
       100                                                                                AS goodwill_to_assets_trend,
       -- Impairment risk score (high goodwill + declining earnings = risk)
       CASE
           WHEN "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.25
               AND "Net Income - (IS) (FY)" < "Net Income - (IS) (-1FY)"
               THEN public.clamp_score(
                   ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0)) * 200 +
                   ABS(public.pct_change("Net Income - (IS) (FY)"::NUMERIC, "Net Income - (IS) (-1FY)"::NUMERIC)) * 0.5
                    )
           ELSE public.clamp_score(
                   ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0)) * 100
                )
           END                                                                            AS impairment_risk_score,
       -- Goodwill concentration (relative to equity)
       public.safe_divide("Goodwill (LTM)"::NUMERIC, "Total Equity (LTM)"::NUMERIC) * 100 AS goodwill_concentration
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_goodwill_temporal_features(text) owner to postgres;

