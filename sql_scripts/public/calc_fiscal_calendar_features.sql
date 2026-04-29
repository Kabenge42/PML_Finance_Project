create function calc_fiscal_calendar_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, days_since_last_report integer, days_to_fy_end integer, is_quarter_end_month integer, is_fy_end_month integer, earnings_season_flag integer, pre_earnings_window integer, post_earnings_window integer, reporting_freshness_score numeric, fiscal_quarter_progress numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                   AS isin,
       (CURRENT_DATE - "Income Statement Report Date")::INTEGER AS days_since_last_report,
       ("FY End Date" - CURRENT_DATE)::INTEGER                  AS days_to_fy_end,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (3, 6, 9, 12)
               THEN 1
           ELSE 0
           END                                                  AS is_quarter_end_month,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) = EXTRACT(MONTH FROM "FY End Date")
               THEN 1
           ELSE 0
           END                                                  AS is_fy_end_month,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (1, 2, 4, 5, 7, 8, 10, 11)
               THEN 1
           ELSE 0
           END                                                  AS earnings_season_flag,
       CASE
           WHEN ("Next Earnings" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                  AS pre_earnings_window,
       CASE
           WHEN (CURRENT_DATE - "Income Statement Report Date") BETWEEN 0 AND 7
               THEN 1
           ELSE 0
           END                                                  AS post_earnings_window,
       GREATEST(0, LEAST(100,
                         100 - ((CURRENT_DATE - "Income Statement Report Date")::NUMERIC / 90.0 * 100)
                   ))                                           AS reporting_freshness_score,
       CASE
           WHEN "Fiscal Month" IS NOT NULL
               THEN (("Fiscal Month" - 1) % 3 + 1) / 3.0
           END                                                  AS fiscal_quarter_progress
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_fiscal_calendar_features(text) owner to postgres;

