create function calc_cashflow_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, cfo_to_net_income numeric, fcf_to_net_income numeric, fcf_margin numeric, cfo_growth_yoy numeric, fcf_positive_ratio numeric, acquisition_intensity numeric, self_funding_ratio numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                 AS isin,
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)     AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)     AS fcf_to_net_income,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0)        AS fcf_margin,
       ("CFO (LTM)" - "CFO (-1FY)") / NULLIF("CFO (-1FY)", 0) AS cfo_growth_yoy,
       (CASE WHEN "FCF (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FQFQ)" > 0 THEN 1 ELSE 0 END) / 5.0 AS fcf_positive_ratio,
       ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))         AS acquisition_intensity,
       CASE
           WHEN ABS("CFI (LTM)") > 0
               THEN "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0)
           END                                                AS self_funding_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_cashflow_features(text) owner to postgres;

