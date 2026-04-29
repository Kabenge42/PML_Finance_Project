create function information_schema.calc_interest_income_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                        text,
                interest_income_ltm         numeric,
                interest_expense_ltm        numeric,
                net_interest_income         numeric,
                interest_coverage_ratio     numeric,
                interest_income_to_revenue  numeric,
                interest_expense_to_revenue numeric,
                net_interest_margin_proxy   numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                   AS isin,
       "Interest And Investment Income (LTM)"                                   AS interest_income_ltm,
       "Interest Expense/Total (LTM)"                                           AS interest_expense_ltm,
       COALESCE("Interest And Investment Income (LTM)", 0) -
       COALESCE("Interest Expense/Total (LTM)", 0)                              AS net_interest_income,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                 AS interest_coverage_ratio,
       "Interest And Investment Income (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100
                                                                                AS interest_income_to_revenue,
       "Interest Expense/Total (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100 AS interest_expense_to_revenue,
       (COALESCE("Interest And Investment Income (LTM)", 0) -
        COALESCE("Interest Expense/Total (LTM)", 0)) /
       NULLIF("Total Assets (LTM)", 0) * 100                                    AS net_interest_margin_proxy
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function information_schema.calc_interest_income_features(unknown) owner to postgres;

