CREATE MATERIALIZED VIEW public.mv_dcf AS
-- missing source code
;

COMMENT ON MATERIALIZED VIEW public.mv_dcf IS 'Materialized view for probabilistic Discounted Cash Flow (DCF) analysis.
    Combines features from 5 core views + supplementary inputs:

    Core Feature Views (per specification):
      1. Growth           (5 functions) - Revenue forecasts, growth rates, CAGR
      2. Cash Flow        (5 functions) - CFO, FCF, CapEx, FCF estimates
      3. Balance Sheet    (3 functions) - Assets, inventory, goodwill trends
      4. Earnings         (6 functions) - EPS, GAAP adjustments, revisions
      5. Cost Structure   (3 functions) - COGS, SG&A, R&D, interest analysis

    Supplementary Inputs:
      6. Profitability    (4 functions) - Margins for FCF margin forecasting
      7. Unusual Items    (1 function)  - Earnings quality normalization
      8. Quality & Risk   (2 functions) - Beta/distress for discount rate
      9. Leverage         (2 functions) - Debt structure for WACC
     10. Employment       (1 function)  - Productivity for revenue/employee

    Total: 32 calc_* functions joined via ISIN.
    Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dcf;';

ALTER MATERIALIZED VIEW public.mv_dcf OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_dcf_isin ON public.mv_dcf (isin);

CREATE INDEX idx_mv_dcf_sector ON public.mv_dcf (sector);

CREATE INDEX idx_mv_dcf_industry ON public.mv_dcf (industry);

CREATE INDEX idx_mv_dcf_country ON public.mv_dcf (country);