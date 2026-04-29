create function information_schema.calc_tax_rate_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                   text,
                effective_tax_rate_ltm numeric,
                effective_tax_rate_fy  numeric,
                tax_rate_yoy_change    numeric,
                tax_rate_qoq_change    numeric,
                tax_rate_stability     numeric,
                low_tax_flag           integer,
                tax_rate_trend_4q      numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN",
       "Effective Tax Rate - (Ratio) (LTM)",
       "Effective Tax Rate - (Ratio) (FY)",
       "Effective Tax Rate - (Ratio) (FY)" - "Effective Tax Rate - (Ratio) (-1FY)" AS tax_rate_yoy_change,
       "Effective Tax Rate - (Ratio) (FQ)" -
       "Effective Tax Rate - (Ratio) (-1FQFQ)"                                     AS tax_rate_qoq_change,
       -- Stability: range across available quarterly periods (lower = more stable)
       GREATEST(
               COALESCE("Effective Tax Rate - (Ratio) (FQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-1FQFQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-2FQFQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-3FQFQ)", 0)
       ) - LEAST(
               COALESCE("Effective Tax Rate - (Ratio) (FQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-1FQFQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-2FQFQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-3FQFQ)", 0)
           )                                                                       AS tax_rate_stability,
       CASE WHEN "Effective Tax Rate - (Ratio) (LTM)" < 0.10 THEN 1 ELSE 0 END     AS low_tax_flag,
       -- Trend across 4 quarters (FQ vs avg of prior 3)
       "Effective Tax Rate - (Ratio) (FQ)" -
       (COALESCE("Effective Tax Rate - (Ratio) (-1FQFQ)", 0) +
        COALESCE("Effective Tax Rate - (Ratio) (-2FQFQ)", 0) +
        COALESCE("Effective Tax Rate - (Ratio) (-3FQFQ)", 0)) /
       NULLIF((CASE WHEN "Effective Tax Rate - (Ratio) (-1FQFQ)" IS NOT NULL THEN 1 ELSE 0 END +
               CASE WHEN "Effective Tax Rate - (Ratio) (-2FQFQ)" IS NOT NULL THEN 1 ELSE 0 END +
               CASE WHEN "Effective Tax Rate - (Ratio) (-3FQFQ)" IS NOT NULL THEN 1 ELSE 0 END)::NUMERIC,
              0)                                                                   AS tax_rate_trend_4q
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function information_schema.calc_tax_rate_features(unknown) owner to postgres;

