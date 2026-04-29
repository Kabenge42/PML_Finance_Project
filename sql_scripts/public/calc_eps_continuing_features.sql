create function calc_eps_continuing_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, eps_cont_ltm numeric, eps_cont_fq numeric, eps_cont_fy numeric, eps_cont_1fqfq numeric, eps_cont_2fqfq numeric, eps_cont_3fqfq numeric, eps_cont_4fqfq numeric, eps_cont_1fy numeric, eps_cont_2fy numeric, eps_cont_3fy numeric, eps_cont_4fy numeric, eps_cont_qoq_growth numeric, eps_cont_yoy_growth numeric, eps_cont_cagr_3y numeric, eps_cont_vs_total_eps numeric, eps_cont_positive_streak integer, eps_cont_trajectory_score numeric, discontinued_ops_impact numeric, core_earnings_stability numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                                    AS isin,
       -- Current period values
       "Basic EPS - Cont (LTM)"                                                                  AS eps_cont_ltm,
       "Basic EPS - Cont (FQ)"                                                                   AS eps_cont_fq,
       "Basic EPS - Cont (FY)"                                                                   AS eps_cont_fy,
       -- Historical FQ
       "Basic EPS - Cont (-1FQFQ)"                                                               AS eps_cont_1fqfq,
       "Basic EPS - Cont (-2FQFQ)"                                                               AS eps_cont_2fqfq,
       "Basic EPS - Cont (-3FQFQ)"                                                               AS eps_cont_3fqfq,
       "Basic EPS - Cont (-4FQFQ)"                                                               AS eps_cont_4fqfq,
       -- Historical FY
       "Basic EPS - Cont (-1FY)"                                                                 AS eps_cont_1fy,
       "Basic EPS - Cont (-2FY)"                                                                 AS eps_cont_2fy,
       "Basic EPS - Cont (-3FY)"                                                                 AS eps_cont_3fy,
       "Basic EPS - Cont (-4FY)"                                                                 AS eps_cont_4fy,
       -- QoQ growth
       public.pct_change("Basic EPS - Cont (FQ)"::NUMERIC, "Basic EPS - Cont (-1FQFQ)"::NUMERIC) AS eps_cont_qoq_growth,
       -- YoY growth
       public.pct_change("Basic EPS - Cont (FY)"::NUMERIC, "Basic EPS - Cont (-1FY)"::NUMERIC)   AS eps_cont_yoy_growth,
       -- 3-year CAGR
       CASE
           WHEN "Basic EPS - Cont (-3FY)" > 0 AND "Basic EPS - Cont (FY)" > 0
               THEN
               (POWER("Basic EPS - Cont (FY)"::NUMERIC / NULLIF("Basic EPS - Cont (-3FY)"::NUMERIC, 0), 1.0 / 3.0) -
                1) * 100
           END                                                                                   AS eps_cont_cagr_3y,
       -- Continuing vs Total EPS ratio (how much comes from continuing ops)
       public.safe_divide("Basic EPS - Cont (LTM)"::NUMERIC,
                          "Net EPS - Basic (LTM)"::NUMERIC)                                      AS eps_cont_vs_total_eps,
       -- Positive streak count
       (CASE WHEN "Basic EPS - Cont (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER
                                                                                                 AS eps_cont_positive_streak,
       -- Trajectory score (improving trend = higher score)
       (CASE WHEN "Basic EPS - Cont (FY)" > "Basic EPS - Cont (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-1FY)" > "Basic EPS - Cont (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-2FY)" > "Basic EPS - Cont (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-3FY)" > "Basic EPS - Cont (-4FY)" THEN 1 ELSE 0 END
           ) / 4.0 *
       100                                                                                       AS eps_cont_trajectory_score,
       -- Discontinued operations impact (difference between total and continuing)
       (("Net EPS - Basic (LTM)" - "Basic EPS - Cont (LTM)") /
        NULLIF(ABS("Net EPS - Basic (LTM)"), 0)) *
       100                                                                                       AS discontinued_ops_impact,
       -- Core earnings stability score
       public.clamp_score(
               100 - ABS(public.pct_change("Basic EPS - Cont (FQ)"::NUMERIC, "Basic EPS - Cont (-4FQFQ)"::NUMERIC) -
                         public.pct_change("Basic EPS - Cont (-1FQFQ)"::NUMERIC,
                                           "Basic EPS - Cont (-4FQFQ)"::NUMERIC)) * 0.5
       )                                                                                         AS core_earnings_stability
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_eps_continuing_features(text) owner to postgres;

