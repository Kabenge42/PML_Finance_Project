create function calc_revenue_forecast_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                       text,
                revenue_est_spread         numeric,
                revenue_beat_potential     numeric,
                revenue_est_revision_trend numeric,
                ebitda_est_vs_actual       numeric,
                forward_revenue_multiple   numeric,
                revenue_estimate_count     numeric,
                revenue_guidance_gap       numeric,
                consensus_revenue_growth   numeric,
                ebit_estimate_spread       numeric,
                forward_ebitda_margin      numeric,
                revenue_acceleration       numeric,
                estimate_confidence_score  numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                   AS isin,
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
       NULLIF("Revenues - Est Med (FY1E)", 0) * 100                             AS revenue_est_spread,
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_beat_potential,
       "Revenues - Est YoY % (FY1E)"                                            AS revenue_est_revision_trend,
       ("EBITDA (LTM)" - "EBITDA - Est Avg (FY1E)") /
       NULLIF(ABS("EBITDA - Est Avg (FY1E)"), 0) * 100                          AS ebitda_est_vs_actual,
       "Enterprise Value" / NULLIF("Revenues - Est Avg (FY1E)", 0)              AS forward_revenue_multiple,
       "EPS Norm - Est # (FY1E)"                                                AS revenue_estimate_count,
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_guidance_gap,
       ("Revenues - Est Avg (FY1E)" - "Total Revenues (FY)") /
       NULLIF(ABS("Total Revenues (FY)"), 0) * 100                              AS consensus_revenue_growth,
       ("EBIT - Est Med (FY1E)" - "EBIT - Est Med (NTM)") /
       NULLIF(ABS("EBIT - Est Med (NTM)"), 0) * 100                             AS ebit_estimate_spread,
       "EBITDA - Est Avg (FY1E)" / NULLIF("Revenues - Est Avg (FY1E)", 0) * 100 AS forward_ebitda_margin,
       "Revenues - Est YoY % (FY1E)" - "Total Revenues/CAGR (5Y FY)"            AS revenue_acceleration,
       GREATEST(0, LEAST(100,
                         100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                   NULLIF("Revenues - Est Med (FY1E)", 0) * 100)
                   ))                                                           AS estimate_confidence_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_revenue_forecast_features(text) owner to postgres;

