create function calc_margin_trends(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, gross_margin_trend_yoy numeric, operating_margin_trend numeric, net_margin_trend_yoy numeric, ebitda_margin_trend numeric, margin_expansion_flag integer, margin_stability_score numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                               AS isin,
       ("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)")       AS gross_margin_trend_yoy,
       (("Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("Operating Income (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100 AS operating_margin_trend,
       ("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")           AS net_margin_trend_yoy,
       (("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100           AS ebitda_margin_trend,
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)"
               AND "Net Income Margin % (LTM)" > "Net Income Margin % (FY)"
               AND ("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) >
                   ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))
               THEN 1
           ELSE 0
           END                                                              AS margin_expansion_flag,
       GREATEST(0, LEAST(100,
                         100 - (ABS("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)") +
                                ABS("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")) / 2
                   ))                                                       AS margin_stability_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_margin_trends(text) owner to postgres;

