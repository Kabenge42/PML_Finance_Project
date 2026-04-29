create function calc_cashflow_temporal_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, cfo_quarterly_trend numeric, cfo_yoy_quarterly numeric, cfi_quarterly_trend numeric, cff_quarterly_trend numeric, fcf_quarterly_trend numeric, cfo_positive_quarters integer, cfi_negative_quarters integer, cff_pattern_score numeric, cash_burn_rate numeric, cf_volatility_score numeric, operating_cf_momentum numeric, financing_dependency numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                               AS isin,
       ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100 AS cfo_quarterly_trend,
       CASE
           WHEN ABS("CFO (-4FQFQ)") > 0
               THEN ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100
           END                                                              AS cfo_yoy_quarterly,
       ("CFI (FQ)" - "CFI (-4FQFQ)") / NULLIF(ABS("CFI (-4FQFQ)"), 0) * 100 AS cfi_quarterly_trend,
       ("CFF (FQ)" - "CFF (-4FQFQ)") / NULLIF(ABS("CFF (-4FQFQ)"), 0) * 100 AS cff_quarterly_trend,
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0) * 100 AS fcf_quarterly_trend,
       (CASE WHEN "CFO (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER            AS cfo_positive_quarters,
       (CASE WHEN "CFI (FQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-1FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-2FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-3FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-4FQFQ)" < 0 THEN 1 ELSE 0 END)::INTEGER            AS cfi_negative_quarters,
       CASE
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") > 0
               THEN -1
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") < 0
               THEN 1
           ELSE 0
           END::NUMERIC                                                     AS cff_pattern_score,
       CASE
           WHEN "FCF (LTM)" < 0
               THEN ABS("FCF (LTM)") / NULLIF("Cash And Equivalents (FQ)", 0) / 12.0
           ELSE 0
           END                                                              AS cash_burn_rate,
       (ABS("CFO (FQ)" - "CFO (-1FQFQ)") + ABS("CFO (-1FQFQ)" - "CFO (-2FQFQ)") +
        ABS("CFO (-2FQFQ)" - "CFO (-3FQFQ)") + ABS("CFO (-3FQFQ)" - "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (FQ)" + "CFO (-1FQFQ)" + "CFO (-2FQFQ)" +
                  "CFO (-3FQFQ)" + "CFO (-4FQFQ)") / 5.0, 0)                AS cf_volatility_score,
       (("CFO (FQ)" + "CFO (-1FQFQ)") - ("CFO (-3FQFQ)" + "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (-3FQFQ)" + "CFO (-4FQFQ)"), 0) * 100                AS operating_cf_momentum,
       ABS("CFF (LTM)") / NULLIF(ABS("CFO (LTM)"), 0)                       AS financing_dependency
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_cashflow_temporal_features(text) owner to postgres;

