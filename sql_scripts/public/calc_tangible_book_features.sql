create function calc_tangible_book_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, tangible_book_value_fy numeric, tangible_book_value_ltm numeric, tangible_book_per_share numeric, price_to_tangible_book numeric, tangible_equity_ratio numeric, intangibles_to_equity numeric, goodwill_to_equity numeric, tangible_asset_quality numeric, tbv_yoy_growth numeric, tbv_vs_calculated numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                                 AS isin,
       -- Use native TBV columns from schema (more accurate than calculation)
       "TBV (FY)"                                                                             AS tangible_book_value_fy,
       "TBV (LTM)"                                                                            AS tangible_book_value_ltm,
       -- Per share using native TBV
       "TBV (LTM)" / NULLIF("Shrs Out", 0)                                                    AS tangible_book_per_share,
       -- P/TBV using native column (already in schema as P/TBV (LTM))
       "P/TBV (LTM)"                                                                          AS price_to_tangible_book,
       -- Tangible equity ratio using native TBV
       "TBV (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100                                    AS tangible_equity_ratio,
       COALESCE("Gross Intangible Assets (LTM)", 0) / NULLIF("Total Equity (LTM)", 0) * 100
                                                                                              AS intangibles_to_equity,
       COALESCE("Goodwill (LTM)", 0) / NULLIF("Total Equity (LTM)", 0) * 100                  AS goodwill_to_equity,
       GREATEST(0, LEAST(100,
                         100 - (COALESCE("Goodwill (LTM)", 0) + COALESCE("Gross Intangible Assets (LTM)", 0)) /
                               NULLIF("Total Assets (LTM)", 0) * 100
                   ))                                                                         AS tangible_asset_quality,
       -- NEW: TBV growth (FY to LTM)
       public.pct_change("TBV (LTM)"::NUMERIC, "TBV (FY)"::NUMERIC)                           AS tbv_yoy_growth,
       -- Validation: compare native TBV to calculated (should be ~1.0)
       public.safe_divide("TBV (LTM)"::NUMERIC, "Total Equity (LTM)"::NUMERIC - COALESCE("Goodwill (LTM)", 0) -
                                                COALESCE("Gross Intangible Assets (LTM)", 0)) AS tbv_vs_calculated
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_tangible_book_features(text) owner to postgres;

