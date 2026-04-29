create function calc_dividend_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, dividend_streak integer, dividend_yield_ltm numeric, dividend_yield_ntm numeric, dividend_payout_ratio numeric, fcf_dividend_coverage numeric, buyback_yield numeric, total_shareholder_yield numeric, dividend_growth_expectation numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                  AS isin,
       "Dividend Streak"::INTEGER                                              AS dividend_streak,
       "Div Yield (LTM)"                                                       AS dividend_yield_ltm,
       "Div Yield (NTM)"                                                       AS dividend_yield_ntm,
       ABS("Common Dividends Paid (LTM)") / NULLIF("Net Income/Adj. (LTM)", 0) AS dividend_payout_ratio,
       CASE
           WHEN ABS("Common Dividends Paid (LTM)") > 0
               THEN "FCF (LTM)" / NULLIF(ABS("Common Dividends Paid (LTM)"), 0)
           END                                                                 AS fcf_dividend_coverage,
       "Buyback Yield (LTM)"                                                   AS buyback_yield,
       COALESCE("Buyback Yield (LTM)", 0) + COALESCE("Div Yield (LTM)", 0)     AS total_shareholder_yield,
       "Div Yield (NTM)" - "Div Yield (LTM)"                                   AS dividend_growth_expectation
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_dividend_features(text) owner to postgres;

