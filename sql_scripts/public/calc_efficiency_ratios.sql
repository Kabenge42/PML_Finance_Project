create function calc_efficiency_ratios(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, asset_turnover numeric, inventory_turnover numeric, receivables_days numeric, working_capital_turns numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                        AS isin,
       "Total Revenues (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS asset_turnover,
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0)                       AS inventory_turnover,
       ("Accounts Receivable/Total (FY)" / NULLIF("Total Revenues (FY)" / 365.0, 0)) AS receivables_days,
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                   AS working_capital_turns
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_efficiency_ratios(text) owner to postgres;

