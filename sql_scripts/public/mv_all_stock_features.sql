create materialized view public.mv_all_stock_features
as
-- missing source code
;

comment on materialized view public.mv_all_stock_features is 'Unified materialized view containing all calculated stock features.
    Covers 26 feature categories from 63 calc_* functions:
    1. Valuation Ratios (4 functions)
    2. Momentum (2 functions)
    3. Technical Analysis (1 function)
    4. Profitability (4 functions)
    5. Earnings (6 functions)
    6. Growth (5 functions)
    7. Quality & Risk (5 functions)
    8. Leverage & Liquidity (6 functions)
    9. Analyst Sentiment (2 functions)
    10. Dividends (3 functions)
    11. Employment (2 functions)
    12. Cash Flow (4 functions)
    13. Temporal (2 functions)
    14. Balance Sheet (3 functions)
    15. Cost Structure (3 functions)
    16. Composite Scores (2 functions)
    17. Unusual Items (1 function)
    18. Volatility Surface (1 function) - Enhancement 2+3
    19. Tax Rate Features (1 function) - Enhancement 4
    20. OpEx Temporal (1 function) - Enhancement 5
    21. Asset Sale Features (1 function) - Enhancement 8
    22. FCF Estimate Curve (1 function) - Enhancement 9
    23. Dividend History (1 function) - Enhancement 10
    24. Investment Income Temporal (1 function) - Enhancement 11
    25. Share Dilution Tracking (1 function) - Enhancement 12
    26. Forward Consensus (1 function) - Enhancement 7

    Direct reference columns include: Enhancement 1 (17 cols), Enhancement 6 (6 cols), Enhancement 7 (4 cols)

    Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;'
;

alter materialized view public.mv_all_stock_features owner to postgres
;

create unique index idx_mv_all_stock_features_isin
	on public.mv_all_stock_features (isin)
;

create index idx_mv_all_stock_features_ticker
	on public.mv_all_stock_features (ticker)
;

create index idx_mv_all_stock_features_sector_industry
	on public.mv_all_stock_features (sector, industry)
;

create index idx_mv_all_stock_features_region_country
	on public.mv_all_stock_features (region, country, trading_country)
;

create index idx_mv_all_stock_features_exchange
	on public.mv_all_stock_features (exchange)
;

create index idx_mv_all_stock_features_market_cap
	on public.mv_all_stock_features (market_cap desc)
;