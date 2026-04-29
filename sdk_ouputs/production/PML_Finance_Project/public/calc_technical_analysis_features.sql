create function calc_technical_analysis_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                      text,
                ema_slope_20d             numeric,
                ema_trend_consistency     integer,
                price_vs_ema_100d         numeric,
                near_52w_high_flag        integer,
                near_52w_low_flag         integer,
                volume_momentum_score     numeric,
                breakout_signal           integer,
                high_volume_flag          integer,
                low_volume_flag           integer,
                volatility_compression    numeric,
                volatility_term_structure numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                        AS isin,
       ("EMA (20D)" - "EMA (50D)") / NULLIF("EMA (50D)", 0)          AS ema_slope_20d,
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" AND "EMA (50D)" > "EMA (100D)"
               AND "EMA (100D)" > "EMA (250D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" AND "EMA (50D)" < "EMA (100D)"
               AND "EMA (100D)" < "EMA (250D)" THEN -1
           ELSE 0
           END                                                       AS ema_trend_consistency,
       ("Last Price" - "EMA (100D)") / NULLIF("EMA (100D)", 0) * 100 AS price_vs_ema_100d,
       CASE
           WHEN ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_high_flag,
       CASE
           WHEN ("Last Price" - "52W Low/Adj") / NULLIF("52W Low/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_low_flag,
       "Rel. Volume" * "Price Chg. % (1M)"                           AS volume_momentum_score,
       CASE
           WHEN "EMA (20D)" > "EMA (50D)"
               AND ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS breakout_signal,
       CASE WHEN "Rel. Volume" > 1.5 THEN 1 ELSE 0 END               AS high_volume_flag,
       CASE WHEN "Rel. Volume" < 0.5 THEN 1 ELSE 0 END               AS low_volume_flag,
       "Volatility (1Y)" - "Volatility (1M)"                         AS volatility_compression,
       "Volatility (3M)" - "Volatility (6M)"                         AS volatility_term_structure
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_technical_analysis_features(text) owner to postgres;

