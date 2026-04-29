create function calc_unusual_items_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, other_unusual_items_ltm numeric, impairment_goodwill_ltm numeric, asset_writedown_ltm numeric, restructuring_charges_ltm numeric, total_unusual_items numeric, unusual_items_to_revenue numeric, unusual_items_to_ebitda numeric, has_unusual_items_flag integer, earnings_quality_impact numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                     AS isin,
       "Other Unusual Items/Total (LTM)"          AS other_unusual_items_ltm,
       "Impairment of Goodwill (LTM)"             AS impairment_goodwill_ltm,
       "Asset Writedown (LTM)"                    AS asset_writedown_ltm,
       "Restructuring Charges (LTM)"              AS restructuring_charges_ltm,
       -- Total unusual/non-recurring items
       COALESCE("Other Unusual Items/Total (LTM)", 0) +
       COALESCE("Impairment of Goodwill (LTM)", 0) +
       COALESCE("Asset Writedown (LTM)", 0) +
       COALESCE("Restructuring Charges (LTM)", 0) AS total_unusual_items,
       -- Unusual items as % of revenue
       public.safe_divide(
               ABS(COALESCE("Other Unusual Items/Total (LTM)", 0) +
                   COALESCE("Impairment of Goodwill (LTM)", 0) +
                   COALESCE("Asset Writedown (LTM)", 0) +
                   COALESCE("Restructuring Charges (LTM)", 0)),
               "Total Revenues (LTM)"::NUMERIC
       ) * 100                                    AS unusual_items_to_revenue,
       -- Unusual items as % of EBITDA
       public.safe_divide(
               ABS(COALESCE("Other Unusual Items/Total (LTM)", 0) +
                   COALESCE("Impairment of Goodwill (LTM)", 0) +
                   COALESCE("Asset Writedown (LTM)", 0) +
                   COALESCE("Restructuring Charges (LTM)", 0)),
               ABS("EBITDA (LTM)")::NUMERIC
       ) * 100                                    AS unusual_items_to_ebitda,
       -- Flag if any unusual items present
       CASE
           WHEN ABS(COALESCE("Other Unusual Items/Total (LTM)", 0)) +
                ABS(COALESCE("Impairment of Goodwill (LTM)", 0)) +
                ABS(COALESCE("Asset Writedown (LTM)", 0)) +
                ABS(COALESCE("Restructuring Charges (LTM)", 0)) > 0
               THEN 1
           ELSE 0 END                             AS has_unusual_items_flag,
       -- Earnings quality impact (higher = better quality, less impacted by unusual items)
       public.clamp_score(
               100 - public.safe_divide(
                             ABS(COALESCE("Other Unusual Items/Total (LTM)", 0) +
                                 COALESCE("Impairment of Goodwill (LTM)", 0) +
                                 COALESCE("Asset Writedown (LTM)", 0) +
                                 COALESCE("Restructuring Charges (LTM)", 0)),
                             ABS("Net Income - (IS) (LTM)")::NUMERIC
                     ) * 100
       )                                          AS earnings_quality_impact
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_unusual_items_features(text) owner to postgres;

