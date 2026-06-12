CREATE MATERIALIZED VIEW public.mv_expected_returns AS
-- missing source code
;

COMMENT ON MATERIALIZED VIEW public.mv_expected_returns IS 'Materialized view for Expected Returns Analytics (v2.5).
Data source for Monte Carlo simulation, Kalman filter, Price Target Achievement, and Earnings Beat models.

Feature categories included:
1. Identifier columns (9 cols from vw_identifier_columns)
2. Temporal/Date columns (7 cols)
3. Market data columns (16 cols - price, volume, beta)
4. Valuation ratios (6 cols)
5. Analyst sentiment (11 cols)
6. Price target dynamics (15 cols)
7. Momentum features (10 cols)
8. Earnings features (17 cols)
9. Profitability (7 cols)
10. Growth metrics (13 cols)
11. Quality & Risk (14 cols)
12. Composite scores (3 cols)
13. Leverage & Liquidity (6 cols)
14. Cash flow (12 cols)
15. Dividends (8 cols)
16. Technical analysis (5 cols)
17. Temporal features (8 cols)

Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_expected_returns;';

ALTER MATERIALIZED VIEW public.mv_expected_returns OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_expected_returns_isin ON public.mv_expected_returns (isin);

CREATE INDEX idx_mv_expected_returns_ticker ON public.mv_expected_returns (ticker);

CREATE INDEX idx_mv_expected_returns_sector ON public.mv_expected_returns (sector);

CREATE INDEX idx_mv_expected_returns_industry ON public.mv_expected_returns (industry);

CREATE INDEX idx_mv_expected_returns_sector_upside ON public.mv_expected_returns (sector ASC, upside_potential DESC);

CREATE INDEX idx_mv_expected_returns_conviction ON public.mv_expected_returns (analyst_conviction DESC) WHERE (analyst_conviction IS NOT NULL);