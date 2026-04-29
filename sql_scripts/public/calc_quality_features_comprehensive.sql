create function calc_quality_features_comprehensive(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, goodwill_impairment_ltm numeric, asset_writedown_ltm numeric, restructuring_ltm numeric, has_goodwill_impairment_ltm integer, goodwill_impairment_frequency integer, asset_writedown_frequency integer, restructuring_frequency integer, exceptional_items_total_ltm numeric, exceptional_items_to_ebitda numeric, quality_issues_count_5y integer, accounting_quality_score numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                       AS isin,
       "Impairment of Goodwill (LTM)"                                               AS goodwill_impairment_ltm,
       "Asset Writedown (LTM)"                                                      AS asset_writedown_ltm,
       "Restructuring Charges (LTM)"                                                AS restructuring_ltm,
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END              AS has_goodwill_impairment_ltm,
       (CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER  AS goodwill_impairment_frequency,
       (CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER         AS asset_writedown_frequency,
       (CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER   AS restructuring_frequency,
       ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
       ABS("Restructuring Charges (LTM)")                                           AS exceptional_items_total_ltm,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) / NULLIF(ABS("EBITDA (LTM)"), 0)        AS exceptional_items_to_ebitda,
       ((CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END) +
        (CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END) +
        (CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END))::INTEGER AS quality_issues_count_5y,
       GREATEST(0, LEAST(100,
                         100 -
                         ((CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END) * 8) -
                         ((CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END) * 4) -
                         ((CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END) * 4)
                   ))                                                               AS accounting_quality_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_quality_features_comprehensive(text) owner to postgres;

