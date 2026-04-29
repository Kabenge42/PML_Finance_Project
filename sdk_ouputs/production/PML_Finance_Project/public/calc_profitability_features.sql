create function calc_profitability_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                 text,
                roe                  numeric,
                roa                  numeric,
                gross_margin_pct     numeric,
                operating_margin_pct numeric,
                net_margin_pct       numeric,
                ebitda_margin_pct    numeric,
                roic                 numeric,
                rnd_intensity        numeric,
                equity_multiplier    numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                             AS isin,
       "Return On Equity % (LTM)"                                         AS roe,
       "Return on Assets (ROA) % (LTM)"                                   AS roa,
       "Gross Profit Margin % (LTM)"                                      AS gross_margin_pct,
       "Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100 AS operating_margin_pct,
       "Net Income Margin % (LTM)"                                        AS net_margin_pct,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100           AS ebitda_margin_pct,
       "EBIT (LTM)" * (1 - 0.25) / NULLIF("Total Equity (LTM)" + "Total Debt (LTM)" - "Cash And Equivalents (LTM)", 0) *
       100                                                                AS roic,
       "R&D Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0)           AS rnd_intensity,
       "Total Assets (LTM)" / NULLIF("Total Equity (LTM)", 0)             AS equity_multiplier
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_profitability_features(text) owner to postgres;

