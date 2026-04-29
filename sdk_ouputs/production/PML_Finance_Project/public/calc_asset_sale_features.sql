create function calc_asset_sale_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                            text,
                gain_loss_on_sale_of_assets_ltm numeric,
                asset_sale_frequency            integer,
                asset_sale_trend                numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN",
       "Gain (Loss) On Sale Of Assets (LTM)",
       -- Count of non-zero periods across available quarters and years
       (CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (FQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-1FQFQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-2FQFQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-3FQFQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-4FQFQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (FY)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-1FY)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-2FY)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-3FY)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-4FY)", 0)) > 0 THEN 1
            ELSE 0 END)::INTEGER                                     AS asset_sale_frequency,
       -- Trend: FQ vs average of prior quarters
       "Gain (Loss) On Sale Of Assets (FQ)" -
       (COALESCE("Gain (Loss) On Sale Of Assets (-1FQFQ)", 0) +
        COALESCE("Gain (Loss) On Sale Of Assets (-2FQFQ)", 0) +
        COALESCE("Gain (Loss) On Sale Of Assets (-3FQFQ)", 0) +
        COALESCE("Gain (Loss) On Sale Of Assets (-4FQFQ)", 0)) / 4.0 AS asset_sale_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_asset_sale_features(text) owner to postgres;

