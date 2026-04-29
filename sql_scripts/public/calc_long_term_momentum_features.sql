create function calc_long_term_momentum_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, price_momentum_1y numeric, price_momentum_3y numeric, price_momentum_5y numeric, long_term_trend_score numeric, price_vs_ema_250d numeric, multi_year_high_flag integer, secular_trend_flag integer, total_return_ytd numeric, total_return_5y numeric, total_return_10y numeric, return_cagr_3y numeric, return_cagr_10y numeric, return_vs_price_momentum numeric, return_consistency_score numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                              AS isin,
       public.pct_change("Last Price"::NUMERIC, "Price (1Y Ago)"::NUMERIC) AS price_momentum_1y,
       public.pct_change("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC) AS price_momentum_3y,
       public.pct_change("Last Price"::NUMERIC, "Price (5Y Ago)"::NUMERIC) AS price_momentum_5y,
       -- Weighted trend score using available periods (1Y: 50%, 3Y: 30%, 5Y: 20%)
       (COALESCE(public.pct_change("Last Price"::NUMERIC, "Price (1Y Ago)"::NUMERIC), 0) * 0.50 +
        COALESCE(public.pct_change("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC), 0) * 0.30 +
        COALESCE(public.pct_change("Last Price"::NUMERIC, "Price (5Y Ago)"::NUMERIC), 0) * 0.20) / 100
                                                                           AS long_term_trend_score,
       public.pct_change("Last Price"::NUMERIC, "EMA (250D)"::NUMERIC)     AS price_vs_ema_250d,
       CASE
           WHEN public.calc_change_ratio("52W High/Adj"::NUMERIC - "Last Price", "52W High/Adj"::NUMERIC) <= 0.10
               AND public.calc_change_ratio("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC) > 0.5
               THEN 1
           ELSE 0
           END                                                             AS multi_year_high_flag,
       CASE
           WHEN public.calc_change_ratio("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC) > 0.20
               AND public.calc_change_ratio("Last Price"::NUMERIC, "Price (1Y Ago)"::NUMERIC) > 0
               AND "EMA (50D)" > "EMA (250D)"
               THEN 1
           ELSE 0
           END                                                             AS secular_trend_flag,
       "Total Return (YTD)"                                                AS total_return_ytd,
       "Total Return (5Y)"                                                 AS total_return_5y,
       "Total Return (10Y)"                                                AS total_return_10y,
       "Tot. Return %/CAGR (3Y)"                                           AS return_cagr_3y,
       "Tot. Return %/CAGR (10Y)"                                          AS return_cagr_10y,
       "Tot. Return %/CAGR (3Y)" - public.pct_change("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC)
                                                                           AS return_vs_price_momentum,
       public.safe_divide("Tot. Return %/CAGR (3Y)", "Volatility (1Y)")    AS return_consistency_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_long_term_momentum_features(text) owner to postgres;

