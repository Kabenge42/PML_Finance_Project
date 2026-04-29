create function information_schema.calc_size_liquidity_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                 text,
                market_cap           numeric,
                market_cap_country_r numeric,
                log_market_cap       numeric,
                volume_shrs          numeric,
                relative_volume      numeric,
                shares_outstanding   numeric,
                daily_turnover_ratio numeric,
                size_class           text,
                style_class          text,
                liquidity_score      numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN",
       "Market Cap",
       "Market Cap (Country R)",
       LN(GREATEST("Market Cap", 1))                                                           AS log_market_cap,
       "Volume (Shrs)",
       "Rel. Volume",
       "Shrs Out",
       public.safe_divide("Volume (Shrs)", "Shrs Out")                                         AS daily_turnover_ratio,
       "Size Class",
       "Style Class",
       "Volume (Shrs)" * COALESCE("Rel. Volume", 1) / NULLIF(LN(GREATEST("Market Cap", 1)), 0) AS liquidity_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function information_schema.calc_size_liquidity_features(unknown) owner to postgres;

