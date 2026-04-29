create function calc_quality_momentum_composite(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, quality_momentum_score numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN" AS isin,
       (((CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
          CASE WHEN "Return On Equity % (LTM)" > 15 THEN 25 ELSE 0 END +
          CASE WHEN "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0) < 1 THEN 25 ELSE 0 END +
          CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END) * 0.40) +
        (LEAST(100, GREATEST(0,
                             (("Last Price" - "Price (3M Ago)") / NULLIF("Price (3M Ago)", 0) * 100 + 50))) * 0.30) +
        (CASE
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 20 THEN 100
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 10 THEN 75
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 0 THEN 50
             ELSE 25
             END * 0.30)
           )  AS quality_momentum_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_quality_momentum_composite(text) owner to postgres;

