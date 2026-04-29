create function calc_eps_comprehensive(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, eps_basic_fq numeric, eps_basic_ltm numeric, eps_basic_fy numeric, eps_adj_ltm numeric, eps_norm_est_fy1e numeric, eps_growth_yoy numeric, eps_cagr_3y numeric, eps_adjustment_ratio numeric, eps_positive_years integer, eps_trajectory_score numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                              AS isin,
       "Net EPS - Basic (FQ)"                                              AS eps_basic_fq,
       "Net EPS - Basic (LTM)"                                             AS eps_basic_ltm,
       "Net EPS - Basic (FY)"                                              AS eps_basic_fy,
       "EPS/Adj. (LTM)"                                                    AS eps_adj_ltm,
       "EPS Norm - Est Avg (FY1E)"                                         AS eps_norm_est_fy1e,
       ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
       NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100                      AS eps_growth_yoy,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                             AS eps_cagr_3y,
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0)               AS eps_adjustment_ratio,
       (CASE WHEN "Net EPS - Basic (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_years,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                   AS eps_trajectory_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_eps_comprehensive(text) owner to postgres;

