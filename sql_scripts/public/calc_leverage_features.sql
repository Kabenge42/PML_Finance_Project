create function calc_leverage_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, debt_to_equity numeric, debt_to_assets numeric, equity_ratio numeric, interest_coverage numeric, current_ratio numeric, cash_ratio numeric, working_capital_ratio numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                      AS isin,
       "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0)                        AS debt_to_equity,
       "Total Debt (LTM)" / NULLIF("Total Assets (LTM)", 0)                        AS debt_to_assets,
       "Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS equity_ratio,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                    AS interest_coverage,
       "Current Ratio (LTM)"                                                       AS current_ratio,
       "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) AS cash_ratio,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0)                   AS working_capital_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_leverage_features(text) owner to postgres;

