create function calc_growth_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                    text,
                revenue_growth_yoy      numeric,
                ebitda_growth_yoy       numeric,
                operating_income_growth numeric,
                fcf_growth              numeric,
                revenue_cagr_5y         numeric,
                forward_revenue_growth  numeric,
                revenue_vs_5y_avg       numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                          AS isin,
       CASE
           WHEN ABS("Total Revenues (-1FY)") > 0
               THEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                    NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100
           END                                                         AS revenue_growth_yoy,
       CASE
           WHEN ABS("EBITDA (-1FY)") > 0
               THEN ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100
           END                                                         AS ebitda_growth_yoy,
       CASE
           WHEN ABS("Operating Income (FY)") > 0
               THEN ("Operating Income (LTM)" - "Operating Income (FY)") /
                    NULLIF(ABS("Operating Income (FY)"), 0) * 100
           END                                                         AS operating_income_growth,
       CASE
           WHEN ABS("FCF (FY)") > 0
               THEN ("FCF (LTM)" - "FCF (FY)") / NULLIF(ABS("FCF (FY)"), 0) * 100
           END                                                         AS fcf_growth,
       "Total Revenues/CAGR (5Y FY)"                                   AS revenue_cagr_5y,
       "Revenues - Est YoY % (FY1E)"                                   AS forward_revenue_growth,
       "Total Revenues (LTM)" / NULLIF("Total Revenues (5YAVGLTM)", 0) AS revenue_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_growth_features(text) owner to postgres;

