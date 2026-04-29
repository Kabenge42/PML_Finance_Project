create function calc_balance_sheet_dynamics(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                      text,
                cash_to_assets_pct        numeric,
                cash_change_qoq           numeric,
                cash_vs_5y_avg            numeric,
                inventory_change_yoy      numeric,
                inventory_vs_5y_avg       numeric,
                receivables_change_yoy    numeric,
                receivables_vs_5y_avg     numeric,
                working_capital_vs_5y_avg numeric,
                retained_earnings_vs_5y   numeric,
                intangibles_growth_flag   integer,
                asset_quality_score       numeric,
                balance_sheet_strength    numeric,
                debt_maturity_risk        numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                    AS isin,
       "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100      AS cash_to_assets_pct,
       ("Cash And Equivalents (FQ)" - "Cash And Equivalents (FY)") /
       NULLIF(ABS("Cash And Equivalents (FY)"), 0)                               AS cash_change_qoq,
       "Cash And Equivalents (FQ)" / NULLIF("Cash And Equivalents (5YAVGFQ)", 0) AS cash_vs_5y_avg,
       ("Inventory (FY)" - "Inventory (FQ)") / NULLIF(ABS("Inventory (FQ)"), 0)  AS inventory_change_yoy,
       "Inventory (FQ)" / NULLIF("Inventory (5YAVGFQ)", 0)                       AS inventory_vs_5y_avg,
       ("Accounts Receivable/Total (FY)" - "Accounts Receivable/Total (-1FY)") /
       NULLIF(ABS("Accounts Receivable/Total (-1FY)"), 0)                        AS receivables_change_yoy,
       "Accounts Receivable/Total (FY)" / NULLIF("Accounts Receivable/Total (5YAVGFQ)", 0)
                                                                                 AS receivables_vs_5y_avg,
       "Working Capital (FQ)" / NULLIF("Working Capital (5YAVGFY)", 0)           AS working_capital_vs_5y_avg,
       "Retained Earnings (FQ)" / NULLIF("Retained Earnings (5YAVGFQ)", 0)       AS retained_earnings_vs_5y,
       CASE
           WHEN "Gross Intangible Assets (FY)" / NULLIF("Gross Intangible Assets (5YAVGFQ)", 0) > 1.5
               THEN 1
           ELSE 0
           END                                                                   AS intangibles_growth_flag,
       GREATEST(0, LEAST(100,
                         50 + ("Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100) -
                         ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100)
                   ))                                                            AS asset_quality_score,
       GREATEST(0, LEAST(100,
                         (CASE
                              WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.10 THEN 25
                              ELSE 0 END) +
                         (CASE WHEN "Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.40 THEN 25 ELSE 0 END) +
                         (CASE WHEN "Working Capital (LTM)" > 0 THEN 25 ELSE 0 END) +
                         (CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END)
                   ))                                                            AS balance_sheet_strength,
       "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0)                            AS debt_maturity_risk
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_balance_sheet_dynamics(text) owner to postgres;

