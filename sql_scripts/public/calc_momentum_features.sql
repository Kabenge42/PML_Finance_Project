create function calc_momentum_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, price_momentum_1m numeric, price_momentum_3m numeric, price_momentum_6m numeric, price_momentum_1y numeric, price_momentum_5d numeric, ema_crossover_20_50 integer, ema_crossover_50_250 integer, price_vs_ema_20d numeric, price_vs_ema_250d numeric, pct_off_52w_high numeric, pct_above_52w_low numeric, range_52w_position numeric, beta_momentum numeric, volatility_regime numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                     AS isin,
       public.pct_change("Last Price"::NUMERIC, "Price (1M Ago)"::NUMERIC)        AS price_momentum_1m,
       public.pct_change("Last Price"::NUMERIC, "Price (3M Ago)"::NUMERIC)        AS price_momentum_3m,
       public.pct_change("Last Price"::NUMERIC, "Price (6M Ago)"::NUMERIC)        AS price_momentum_6m,
       public.pct_change("Last Price"::NUMERIC, "Price (1Y Ago)"::NUMERIC)        AS price_momentum_1y,
       public.pct_change("Last Price"::NUMERIC, "Price (5D Ago)"::NUMERIC)        AS price_momentum_5d,
       public.ema_crossover_signal("EMA (20D)"::NUMERIC, "EMA (50D)"::NUMERIC)    AS ema_crossover_20_50,
       public.ema_crossover_signal("EMA (50D)"::NUMERIC, "EMA (250D)"::NUMERIC)   AS ema_crossover_50_250,
       public.calc_change_ratio("Last Price"::NUMERIC, "EMA (20D)"::NUMERIC)      AS price_vs_ema_20d,
       public.calc_change_ratio("Last Price"::NUMERIC, "EMA (250D)"::NUMERIC)     AS price_vs_ema_250d,
       public.calc_change_ratio(("52W High/Adj"::NUMERIC - "Last Price"::NUMERIC),
                                "52W High/Adj"::NUMERIC)                          AS pct_off_52w_high,
       public.calc_change_ratio(("Last Price"::NUMERIC - "52W Low/Adj"::NUMERIC),
                                "52W Low/Adj"::NUMERIC)                           AS pct_above_52w_low,
       public.clamp_score(public.safe_divide(("Last Price"::NUMERIC - "52W Low/Adj"::NUMERIC),
                                             ("52W High/Adj"::NUMERIC - "52W Low/Adj"::NUMERIC)), 0,
                          1)                                                      AS range_52w_position,
       "Beta (1Y)"::NUMERIC - "Beta (5Y)"::NUMERIC                                AS beta_momentum,
       public.safe_divide("Volatility (1M)"::NUMERIC, "Volatility (1Y)"::NUMERIC) AS volatility_regime
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_momentum_features(text) owner to postgres;

