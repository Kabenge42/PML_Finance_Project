create function calc_valuation_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, p_e_ratio numeric, p_b_ratio numeric, ev_ebitda_ratio numeric, ev_sales_ratio numeric, dividend_yield numeric, peg_ratio numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                     AS isin,
       "P/E (LTM)"::NUMERIC       AS p_e_ratio,
       "P/B (LTM)"::NUMERIC       AS p_b_ratio,
       "EV/EBITDA (LTM)"::NUMERIC AS ev_ebitda_ratio,
       "EV/Sales (LTM)"::NUMERIC  AS ev_sales_ratio,
       "Div Yield (LTM)"::NUMERIC AS dividend_yield,
       CASE
           WHEN "Net EPS - Basic (FY)" > 0 AND "Net EPS - Basic (-3FY)" > 0
               THEN public.safe_divide(
                   "P/E (LTM)"::NUMERIC,
                   ((POWER(
                             public.safe_divide("Net EPS - Basic (FY)"::NUMERIC, "Net EPS - Basic (-3FY)"::NUMERIC),
                             (1.0 / 3.0)::NUMERIC
                     ) - 1) * 100)::NUMERIC
                    )
           END                    AS peg_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_valuation_features(text) owner to postgres;

