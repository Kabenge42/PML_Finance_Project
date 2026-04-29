create function calc_revenue_estimate_consensus(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, revenue_est_avg_fy1e numeric, revenue_est_med_fy1e numeric, revenue_est_avg_ntm numeric, revenue_est_med_ntm numeric, revenue_avg_med_diff_pct numeric, revenue_consensus_strength numeric, revenue_revision_trend numeric, revenue_vs_current numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                                    AS isin,
       "Revenues - Est Avg (FY1E)"                                                               AS revenue_est_avg_fy1e,
       "Revenues - Est Med (FY1E)"                                                               AS revenue_est_med_fy1e,
       "Revenues - Est Avg (NTM)"                                                                AS revenue_est_avg_ntm,
       "Revenues - Est Med (NTM)"                                                                AS revenue_est_med_ntm,
       -- Difference between avg and median as proxy for estimate dispersion
       public.safe_divide("Revenues - Est Avg (FY1E)"::NUMERIC - "Revenues - Est Med (FY1E)",
                          "Revenues - Est Med (FY1E)"::NUMERIC) *
       100                                                                                       AS revenue_avg_med_diff_pct,
       -- Consensus strength: closer avg to median = stronger consensus
       public.clamp_score(
               100 - ABS(public.safe_divide("Revenues - Est Avg (FY1E)"::NUMERIC - "Revenues - Est Med (FY1E)",
                                            "Revenues - Est Med (FY1E)"::NUMERIC) * 100) * 2
       )                                                                                         AS revenue_consensus_strength,
       "Revenues - Est YoY % (FY1E)"                                                             AS revenue_revision_trend,
       -- Compare estimate to current revenue
       public.safe_divide("Revenues - Est Avg (FY1E)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) AS revenue_vs_current
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_revenue_estimate_consensus(text) owner to postgres;

