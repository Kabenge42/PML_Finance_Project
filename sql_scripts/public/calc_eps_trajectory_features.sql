create function calc_eps_trajectory_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, eps_qoq_growth numeric, eps_yoy_quarterly numeric, eps_positive_streak integer, eps_cagr_3y numeric, eps_cagr_5y numeric, eps_growth_accel numeric, eps_vs_5y_avg numeric, eps_improvement_count integer, eps_trajectory_score numeric, eps_stability numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                AS isin,
       CASE
           WHEN ABS("Net EPS - Basic (-1FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-1FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-1FQFQ)"), 0) * 100
           END                                                               AS eps_qoq_growth,
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0) * 100
           END                                                               AS eps_yoy_quarterly,
       (CASE WHEN "Net EPS - Basic (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_streak,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                               AS eps_cagr_3y,
       CASE
           WHEN "Net EPS - Basic (-5FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1) * 100
           END                                                               AS eps_cagr_5y,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (-5FY)" > 0
               AND "Net EPS - Basic (FY)" > 0
               THEN ((POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) -
                     (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1)) * 100
           END                                                               AS eps_growth_accel,
       CASE
           WHEN ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                     "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0) > 0
               THEN ("Net EPS - Basic (FY)" -
                     (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                       "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0)) /
                    NULLIF(ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                                "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0), 0) * 100
           END                                                               AS eps_vs_5y_avg,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END)::INTEGER
                                                                             AS eps_improvement_count,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                     AS eps_trajectory_score,
       CASE
           WHEN ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                     "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0) > 0
               THEN 1.0 - LEAST(1.0,
                                SQRT(
                                        (POWER("Net EPS - Basic (FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2) +
                                         POWER("Net EPS - Basic (-1FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2) +
                                         POWER("Net EPS - Basic (-2FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2) +
                                         POWER("Net EPS - Basic (-3FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2) +
                                         POWER("Net EPS - Basic (-4FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2)
                                            ) / 5.0
                                ) / NULLIF(ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                "Net EPS - Basic (-2FY)" +
                                                "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0), 0)
                          )
           END                                                               AS eps_stability -- 0 = chaotic, 1 = perfectly stable
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_eps_trajectory_features(text) owner to postgres;

