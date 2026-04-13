create function calc_quality_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                        text,
                has_goodwill_impairment     integer,
                has_asset_writedown         integer,
                has_restructuring           integer,
                goodwill_to_assets_pct      numeric,
                intangible_intensity        numeric,
                exceptional_items_to_ebitda numeric,
                altman_z_score              numeric,
                altman_z_trend              numeric,
                current_ratio               numeric,
                quick_ratio                 numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                                            AS isin,
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END                                   AS has_goodwill_impairment,
       CASE WHEN "Asset Writedown (LTM)" <> 0 THEN 1 ELSE 0 END                                          AS has_asset_writedown,
       CASE WHEN "Restructuring Charges (LTM)" <> 0 THEN 1 ELSE 0 END                                    AS has_restructuring,
       "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100                                          AS goodwill_to_assets_pct,
       "Gross Intangible Assets (LTM)" / NULLIF("Total Assets (LTM)", 0)                                 AS intangible_intensity,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) /
       NULLIF(ABS("EBITDA (LTM)"), 0)                                                                    AS exceptional_items_to_ebitda,
       "Altman Z-Score (LTM)"                                                                            AS altman_z_score,
       "Altman Z-Score (FY)" - "Altman Z-Score (LTM)"                                                    AS altman_z_trend,
       "Current Ratio (LTM)"                                                                             AS current_ratio,
       ("Total Current Assets (LTM)" - "Inventory (LTM)") / NULLIF("Total Current Liabilities (LTM)", 0) AS quick_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_quality_features(text) owner to postgres;

