create function calc_enhanced_cashflow_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                    text,
                fcf_positive_years      integer,
                fcf_always_positive     integer,
                capex_vs_5y_avg         numeric,
                underinvestment_flag    integer,
                cfo_share_of_cf         numeric,
                cfi_share_of_cf         numeric,
                cff_share_of_cf         numeric,
                self_funding_flag       integer,
                acquisition_to_fcf      numeric,
                sustainable_ma_flag     integer,
                fcf_4q_improvement      numeric,
                cash_flow_quality_score numeric,
                capex_yoy_growth        numeric,
                capex_qoq_growth        numeric,
                capex_3y_trend          numeric,
                capex_volatility        numeric,
                capex_acceleration      integer,
                capex_cut_flag          integer,
                overinvestment_flag     integer,
                acquisitions_yoy_growth numeric,
                acquisitions_vs_5y_avg  numeric,
                acquisitions_ltm_total  numeric,
                ma_intensity_score      numeric,
                serial_acquirer_flag    integer,
                acquisition_pause_flag  integer,
                total_investment_to_cfo numeric,
                organic_vs_inorganic    numeric,
                investment_efficiency   numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                            AS isin,
       -- Existing features (unchanged)
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER           AS fcf_positive_years,
       CASE
           WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
               AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0
               THEN 1
           ELSE 0
           END                                                           AS fcf_always_positive,
       ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0)
                                                                         AS capex_vs_5y_avg,
       CASE
           WHEN ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0) < 0.7
               THEN 1
           ELSE 0
           END                                                           AS underinvestment_flag,
       ABS("CFO (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfo_share_of_cf,
       ABS("CFI (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfi_share_of_cf,
       ABS("CFF (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cff_share_of_cf,
       CASE
           WHEN "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0) > 1
               THEN 1
           ELSE 0
           END                                                           AS self_funding_flag,
       (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
       NULLIF(ABS("FCF (LTM)"), 0)                                       AS acquisition_to_fcf,
       CASE
           WHEN (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
                NULLIF(ABS("FCF (LTM)"), 0) < 0.5
               THEN 1
           ELSE 0
           END                                                           AS sustainable_ma_flag,
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0)    AS fcf_4q_improvement,
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
                AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0 THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC           AS cash_flow_quality_score,

       -- NEW: CapEx YoY growth (FY vs -1FY)
       (ABS("Capital Expenditure (FY)") - ABS("Capital Expenditure (-1FY)")) /
       NULLIF(ABS("Capital Expenditure (-1FY)"), 0) * 100                AS capex_yoy_growth,

       -- NEW: CapEx QoQ growth (FQ vs -1FQFQ)
       (ABS("Capital Expenditure (FQ)") - ABS("Capital Expenditure (-1FQFQ)")) /
       NULLIF(ABS("Capital Expenditure (-1FQFQ)"), 0) * 100              AS capex_qoq_growth,

       -- NEW: CapEx 3-year trend (FY vs -3FY)
       (ABS("Capital Expenditure (FY)") - ABS("Capital Expenditure (-3FY)")) /
       NULLIF(ABS("Capital Expenditure (-3FY)"), 0) * 100                AS capex_3y_trend,

       -- NEW: CapEx volatility (variation across quarters)
       (ABS(ABS("Capital Expenditure (FQ)") - ABS("Capital Expenditure (-1FQFQ)")) +
        ABS(ABS("Capital Expenditure (-1FQFQ)") - ABS("Capital Expenditure (-2FQFQ)")) +
        ABS(ABS("Capital Expenditure (-2FQFQ)") - ABS("Capital Expenditure (-3FQFQ)")) +
        ABS(ABS("Capital Expenditure (-3FQFQ)") - ABS("Capital Expenditure (-4FQFQ)"))) /
       NULLIF((ABS("Capital Expenditure (FQ)") + ABS("Capital Expenditure (-1FQFQ)") +
               ABS("Capital Expenditure (-2FQFQ)") + ABS("Capital Expenditure (-3FQFQ)") +
               ABS("Capital Expenditure (-4FQFQ)")) / 5.0, 0)            AS capex_volatility,

       -- NEW: CapEx acceleration flag (increasing investment rate)
       CASE
           WHEN ABS("Capital Expenditure (FY)") > ABS("Capital Expenditure (-1FY)")
               AND ABS("Capital Expenditure (-1FY)") > ABS("Capital Expenditure (-2FY)")
               THEN 1
           ELSE 0
           END                                                           AS capex_acceleration,

       -- NEW: CapEx cut flag (significant decline may signal distress or maturity)
       CASE
           WHEN (ABS("Capital Expenditure (FY)") - ABS("Capital Expenditure (-1FY)")) /
                NULLIF(ABS("Capital Expenditure (-1FY)"), 0) < -0.25
               THEN 1
           ELSE 0
           END                                                           AS capex_cut_flag,

       -- NEW: Overinvestment flag (CapEx significantly above historical average)
       CASE
           WHEN ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0) > 1.5
               THEN 1
           ELSE 0
           END                                                           AS overinvestment_flag,

       -- NEW: Cash Acquisitions YoY growth
       (ABS(COALESCE("Cash Acquisitions (FY)", 0)) - ABS(COALESCE("Cash Acquisitions (-1FY)", 0))) /
       NULLIF(ABS(COALESCE("Cash Acquisitions (-1FY)", 0)), 0) * 100     AS acquisitions_yoy_growth,

       -- NEW: Cash Acquisitions vs 5Y average
       ABS(COALESCE("Cash Acquisitions (FQ)", 0)) /
       NULLIF(ABS(COALESCE("Cash Acquisitions (5YAVGFQ)", 0)), 0)        AS acquisitions_vs_5y_avg,

       -- NEW: LTM total acquisitions
       ABS(COALESCE("Cash Acquisitions (LTM)", 0))                       AS acquisitions_ltm_total,

       -- NEW: M&A intensity score (acquisitions relative to market cap proxy via total assets)
       ABS(COALESCE("Cash Acquisitions (LTM)", 0)) /
       NULLIF("Total Assets (LTM)", 0) * 100                             AS ma_intensity_score,

       -- NEW: Serial acquirer flag (significant acquisitions in 3+ of last 4 years)
       CASE
           WHEN (CASE WHEN ABS(COALESCE("Cash Acquisitions (FY)", 0)) > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS(COALESCE("Cash Acquisitions (-1FY)", 0)) > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS(COALESCE("Cash Acquisitions (-2FY)", 0)) > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS(COALESCE("Cash Acquisitions (-3FY)", 0)) > 0 THEN 1 ELSE 0 END) >= 3
               THEN 1
           ELSE 0
           END                                                           AS serial_acquirer_flag,

       -- NEW: Acquisition pause flag (no recent acquisitions after historical activity)
       CASE
           WHEN ABS(COALESCE("Cash Acquisitions (FY)", 0)) = 0
               AND (ABS(COALESCE("Cash Acquisitions (-1FY)", 0)) > 0
                   OR ABS(COALESCE("Cash Acquisitions (-2FY)", 0)) > 0)
               THEN 1
           ELSE 0
           END                                                           AS acquisition_pause_flag,

       -- NEW: Total investment (CapEx + Acquisitions) to CFO ratio
       (ABS(COALESCE("Capital Expenditure (LTM)", 0)) + ABS(COALESCE("Cash Acquisitions (LTM)", 0))) /
       NULLIF(ABS("CFO (LTM)"), 0)                                       AS total_investment_to_cfo,

       -- NEW: Organic vs Inorganic growth ratio (CapEx / Acquisitions)
       ABS(COALESCE("Capital Expenditure (LTM)", 0)) /
       NULLIF(ABS(COALESCE("Cash Acquisitions (LTM)", 0)), 0)            AS organic_vs_inorganic,

       -- NEW: Investment efficiency (revenue growth per unit of total investment)
       CASE
           WHEN (ABS(COALESCE("Capital Expenditure (-1FY)", 0)) + ABS(COALESCE("Cash Acquisitions (-1FY)", 0))) > 0
               THEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                    NULLIF(ABS(COALESCE("Capital Expenditure (-1FY)", 0)) +
                           ABS(COALESCE("Cash Acquisitions (-1FY)", 0)), 0)
           END                                                           AS investment_efficiency

FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_enhanced_cashflow_features(text) owner to postgres;

