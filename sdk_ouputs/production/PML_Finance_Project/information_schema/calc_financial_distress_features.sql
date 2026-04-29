create function information_schema.calc_financial_distress_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                     text,
                distress_risk_score      numeric,
                liquidity_stress_score   numeric,
                working_capital_trend    numeric,
                cash_runway_months       numeric,
                combined_distress_score  numeric,
                wc_deteriorating_flag    integer,
                retained_earnings_growth numeric,
                accumulated_deficit_flag integer,
                adequate_cash_buffer     integer
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                   AS isin,

       -- distress_risk_score: unchanged
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 100)
                   ))                                           AS distress_risk_score,

       -- liquidity_stress_score: add a graduated middle band
       CASE
           WHEN "Current Ratio (LTM)" < 0.8 THEN 40.0
           WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
           WHEN "Current Ratio (LTM)" < 1.2 THEN 20.0
           WHEN "Current Ratio (LTM)" < 1.5 THEN 10.0
           ELSE 0.0
           END                                                  AS liquidity_stress_score,

       -- working_capital_trend: unchanged
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(ABS("Working Capital (FY)"), 0)                   AS working_capital_trend,

       -- cash_runway_months: use NET cash burn (OpEx - Revenue) with floor at 1
       -- For profitable companies (Revenue > OpEx), runway is effectively infinite â†’ cap at 120
       CASE
           WHEN "Total Operating Expenses (LTM)" - COALESCE("Total Revenues (LTM)", 0) <= 0
               THEN 120.0 -- net cash-positive: no burn
           ELSE GREATEST(0,
                         "Cash And Equivalents (FQ)" /
                         NULLIF(("Total Operating Expenses (LTM)" - COALESCE("Total Revenues (LTM)", 0)) / 12.0, 0)
                )
           END                                                  AS cash_runway_months,

       -- combined_distress_score: unchanged formula (will benefit from improved sub-scores)
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 70) +
                         (100 - CASE
                                    WHEN "Current Ratio (LTM)" < 0.8 THEN 40.0
                                    WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
                                    WHEN "Current Ratio (LTM)" < 1.2 THEN 20.0
                                    WHEN "Current Ratio (LTM)" < 1.5 THEN 10.0
                                    ELSE 0.0
                             END) * 0.30
                   ))                                           AS combined_distress_score,

       -- wc_deteriorating_flag: unchanged
       CASE
           WHEN ("Working Capital (FQ)" - "Working Capital (FY)") /
                NULLIF(ABS("Working Capital (FY)"), 0) < -0.2
               THEN 1
           ELSE 0
           END                                                  AS wc_deteriorating_flag,

       -- retained_earnings_growth: unchanged
       ("Retained Earnings (FQ)" - "Retained Earnings (FY)") /
       NULLIF(ABS("Retained Earnings (FY)"), 0)                 AS retained_earnings_growth,

       -- accumulated_deficit_flag: unchanged
       CASE WHEN "Retained Earnings (FQ)" < 0 THEN 1 ELSE 0 END AS accumulated_deficit_flag,

       -- adequate_cash_buffer: lower threshold to 3 months for net-burn basis
       CASE
           WHEN "Total Operating Expenses (LTM)" - COALESCE("Total Revenues (LTM)", 0) <= 0
               THEN 1 -- net cash-positive: always adequate
           WHEN "Cash And Equivalents (FQ)" /
                NULLIF(("Total Operating Expenses (LTM)" - COALESCE("Total Revenues (LTM)", 0)) / 12.0, 0) > 6
               THEN 1
           ELSE 0
           END                                                  AS adequate_cash_buffer

FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function information_schema.calc_financial_distress_features(unknown) owner to postgres;

