create function calc_fcf_estimate_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin             text,
                fcf_est_avg_fy1e numeric,
                fcf_est_avg_fy2e numeric,
                fcf_est_avg_fy3e numeric,
                fcf_est_avg_fy4e numeric,
                fcf_est_avg_fy5e numeric,
                fcf_est_cagr_5y  numeric,
                fcf_est_trend    numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN",
       "FCF - Est Avg (FY1E)",
       "FCF - Est Avg (FY2E)",
       "FCF - Est Avg (FY3E)",
       "FCF - Est Avg (FY4E)",
       "FCF - Est Avg (FY5E)",
       -- Implied 5Y CAGR from FY1E to FY5E
       CASE
           WHEN "FCF - Est Avg (FY1E)" > 0 AND "FCF - Est Avg (FY5E)" > 0
               THEN (POWER(public.safe_divide("FCF - Est Avg (FY5E)",
                                              "FCF - Est Avg (FY1E)"), 0.25) - 1) * 100
           END                                                                  AS fcf_est_cagr_5y,
       -- Linear trend: (FY5E - FY1E) / FY1E
       public.calc_change_ratio("FCF - Est Avg (FY5E)", "FCF - Est Avg (FY1E)") AS fcf_est_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_fcf_estimate_features(text) owner to postgres;

