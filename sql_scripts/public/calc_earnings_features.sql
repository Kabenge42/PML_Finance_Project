create function calc_earnings_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, eps_surprise_pct numeric, revenue_surprise_pct numeric, eps_adjustment_ratio numeric, gaap_adj_eps_gap_pct numeric, ebitda_adjustment_ratio numeric, eps_quarterly_trend numeric, eps_yoy_growth numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                AS isin,
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS/Adj. (LTM)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS eps_surprise_pct,
       CASE
           WHEN ABS("Revenues - Est Avg (FY1E)") > 0
               THEN ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
                    NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100
           END                                               AS revenue_surprise_pct,
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0) AS eps_adjustment_ratio,
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS gaap_adj_eps_gap_pct,
       "EBITDA/Adj. (LTM)" / NULLIF("EBITDA (LTM)", 0)       AS ebitda_adjustment_ratio,
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0)
           END                                               AS eps_quarterly_trend,
       CASE
           WHEN ABS("Net EPS - Basic (-1FY)") > 0
               THEN ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
                    NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100
           END                                               AS eps_yoy_growth
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_earnings_features(text) owner to postgres;

