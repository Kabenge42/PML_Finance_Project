create function calc_accounting_quality_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, goodwill_change_rate numeric, restructuring_intensity numeric, exceptional_items_frequency integer, merger_impact_ratio numeric, non_operating_income_share numeric, asset_sale_boost integer, accounting_quality_score numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                AS isin,
       ("Goodwill (LTM)" - "Goodwill (-1FY)") / NULLIF("Goodwill (-1FY)", 0) AS goodwill_change_rate,
       "Restructuring Charges (LTM)" / NULLIF("Total Assets (LTM)", 0)       AS restructuring_intensity,
       (CASE WHEN ABS("Impairment of Goodwill (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Asset Writedown (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Restructuring Charges (FQ)") > 0 THEN 1 ELSE 0 END)   AS exceptional_items_frequency,
       "Merger & Restructuring Charges (LTM)" / NULLIF("Market Cap", 0)      AS merger_impact_ratio,
       "Interest And Investment Income (LTM)" / NULLIF(ABS("Net Income - (IS) (LTM)"), 0)
                                                                             AS non_operating_income_share,
       CASE WHEN "Gain (Loss) On Sale Of Assets (LTM)" > 0 THEN 1 ELSE 0 END AS asset_sale_boost,
       GREATEST(0, LEAST(100,
                         100 -
                         (CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 25 ELSE 0 END) -
                         (CASE WHEN "Asset Writedown (LTM)" <> 0 THEN 10 ELSE 0 END) -
                         (CASE WHEN "Restructuring Charges (LTM)" <> 0 THEN 15 ELSE 0 END) -
                         (CASE WHEN "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.30 THEN 15 ELSE 0 END) -
                         (CASE
                              WHEN (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
                                    ABS("Restructuring Charges (LTM)")) /
                                   NULLIF(ABS("Net Income - (IS) (LTM)"), 0) > 0.10 THEN 15
                              ELSE 0 END)
                   ))                                                        AS accounting_quality_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_accounting_quality_features(text) owner to postgres;

