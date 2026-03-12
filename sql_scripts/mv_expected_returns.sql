-- expected_returns.sql
-- Materialized view for Expected Returns Analytics (v2.5)
-- Data source for: Monte Carlo, Kalman Filter, Price Target Achievement, Earnings Beat models
--
-- Prerequisites: Run the following scripts BEFORE this one:
--   1. create_helper_functions.sql  (safe_divide, pct_change, calc_change_ratio, etc.)
--   2. feature_registry.sql         (calc_valuation_features, calc_sentiment_features, etc.)

-- Pre-flight check: ensure prerequisite functions exist
DO
$$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'safe_divide') THEN
            RAISE EXCEPTION 'Required function safe_divide() does not exist. Run create_helper_functions.sql first.';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'calc_valuation_features') THEN
            RAISE EXCEPTION 'Required function calc_valuation_features() does not exist. Run feature_registry.sql first.';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'calc_sentiment_features') THEN
            RAISE EXCEPTION 'Required function calc_sentiment_features() does not exist. Run feature_registry.sql first.';
        END IF;
    END
$$;

DROP MATERIALIZED VIEW IF EXISTS mv_expected_returns CASCADE;

CREATE MATERIALIZED VIEW mv_expected_returns AS
SELECT
    -- ═══════════════════════════════════════════════════════════════════════════════
    -- IDENTIFIER COLUMNS (from vw_identifier_columns)
    -- ═══════════════════════════════════════════════════════════════════════════════
    id.isin,
    id.ticker,
    id.name,
    id.industry,
    id.sector,
    id.trading_country,
    id.region,
    id.country,
    id.exchange,
    id.size_class,
    id.style_class,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- TEMPORAL / DATE COLUMNS
    -- ═══════════════════════════════════════════════════════════════════════════════
    e."Last Updated"                      AS last_updated,
    e."Reference Date"                    AS reference_date,
    e."FY End Date"                       AS fy_end_date,
    e."Next FY End Date"                  AS next_fy_end_date,
    e."Next Earnings"                     AS next_earnings,
    e."Income Statement Report Date"      AS income_statement_report_date,
    e."Next Income Statement Report Date" AS next_income_statement_report_date,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- MARKET DATA COLUMNS (role = 'market_data' from equities_schema_metadata)
    -- ═══════════════════════════════════════════════════════════════════════════════
    e."Market Cap"                        AS market_cap,
    e."Enterprise Value"                  AS enterprise_value,
    e."Last Price"                        AS last_price,
    e."Price Target"                      AS price_target,
    e."Price Target - Median"             AS price_target_median,
    e."Price Target (YTD Ago)"            AS price_target_ytd_ago,
    e."Price Target - Low"                AS price_target_low,
    e."Price Target - High"               AS price_target_high,
    e."Shrs Out"                          AS shares_outstanding,
    e."Volume (Shrs)"                     AS volume_shrs,
    e."52W High/Adj"                      AS "52w_high_adj",
    e."52W Low/Adj"                       AS "52w_low_adj",
    e."Beta (1Y)"                         AS beta_1y,
    e."Beta (2Y)"                         AS beta_2y,
    e."Beta (5Y)"                         AS beta_5y,


    -- ═══════════════════════════════════════════════════════════════════════════════
    -- PRICE MOMENTUM COLUMNS (for Monte Carlo / Kalman models)
    -- ═══════════════════════════════════════════════════════════════════════════════

    e."Price (5D Ago)"                    AS price_5d_ago,
    e."Price (1W Ago)"                    AS price_1w_ago,
    e."Price (1M Ago)"                    AS price_1m_ago,
    e."Price (3M Ago)"                    AS price_3m_ago,
    e."Price (6M Ago)"                    AS price_6m_ago,
    e."Price (1Y Ago)"                    AS price_1y_ago,
    e."Price (3Y Ago)"                    AS price_3y_ago,
    e."Price (5Y Ago)"                    AS price_5y_ago,
    e."Price (QTD Ago)"                   AS price_qtd_ago,
    e."Price Target (1W Ago)"             AS price_target_1w_ago,
    e."Price Target (1M Ago)"             AS price_target_1m_ago,
    e."Price Target (3M Ago)"             AS price_target_3m_ago,
    e."Price Target (6M Ago)"             AS price_target_6m_ago,
    e."Price Target (MTD Ago)"            AS price_target_mtd_ago,
    e."Price Target (QTD Ago)"            AS price_target_qtd_ago,
    e."Price Target (1Y Ago)"             AS price_target_1y_ago,
    e."Price Target - High (1W Ago)"      AS price_target_high_1w_ago,
    e."Price Target - High (1M Ago)"      AS price_target_high_1m_ago,
    e."Price Target - High (6M Ago)"      AS price_target_high_6m_ago,
    e."Price Target - High (MTD Ago)"     AS price_target_high_mtd_ago,
    e."Price Target - High (3M Ago)"      AS price_target_high_3m_ago,
    e."Price Target - High (QTD Ago)"     AS price_target_high_qtd_ago,
    e."Price Target - High (1Y Ago)"      AS price_target_high_1y_ago,
    e."Price Target - High (YTD Ago)"     AS price_target_high_ytd_ago,
    e."Price Target - Low (1W Ago)"       AS price_target_low_1w_ago,
    e."Price Target - Low (1M Ago)"       AS price_target_low_1m_ago,
    e."Price Target - Low (3M Ago)"       AS price_target_low_3m_ago,
    e."Price Target - Low (6M Ago)"       AS price_target_low_6m_ago,
    e."Price Target - Low (MTD Ago)"      AS price_target_low_mtd_ago,
    e."Price Target - Low (QTD Ago)"      AS price_target_low_qtd_ago,
    e."Price Target - Low (YTD Ago)"      AS price_target_low_ytd_ago,
    e."Price Target - Low (1Y Ago)"       AS price_target_low_1y_ago,
    e."Price Target - Median (1W Ago)"    AS price_target_median_1w_ago,
    e."Price Target - Median (1M Ago)"    AS price_target_median_1m_ago,
    e."Price Target - Median (3M Ago)"    AS price_target_median_3m_ago,
    e."Price Target - Median (6M Ago)"    AS price_target_median_6m_ago,
    e."Price Target - Median (MTD Ago)"   AS price_target_median_mtd_ago,
    e."Price Target - Median (QTD Ago)"   AS price_target_median_qtd_ago,
    e."Price Target - Median (YTD Ago)"   AS price_target_median_ytd_ago,
    e."Price Target - Median (1Y Ago)"    AS price_target_median_1y_ago,


    -- ═══════════════════════════════════════════════════════════════════════════════
    -- VALUATION RATIOS (for expected return models)
    -- ═══════════════════════════════════════════════════════════════════════════════
    vf.p_e_ratio,
    vf.p_b_ratio,
    vf.ev_ebitda_ratio,
    vf.ev_sales_ratio,
    vf.dividend_yield                     AS valuation_dividend_yield,
    vf.peg_ratio,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- ANALYST SENTIMENT (core inputs for achievement model)
    -- ═══════════════════════════════════════════════════════════════════════════════
    sf.analyst_bullish_pct,
    sf.analyst_bearish_pct,
    sf.analyst_neutral_pct,
    sf.analyst_conviction,
    sf.upside_potential,
    sf.price_target_spread_pct,
    sf.price_target_revision_1m,
    sf.price_target_revision_3m,
    sf.eps_revision_momentum,
    sf.analyst_rating_normalized,
    sf.analyst_coverage_quality,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- PRICE TARGET DYNAMICS (Kalman filter inputs)
    -- ═══════════════════════════════════════════════════════════════════════════════
    ptd.pt_momentum_1w,
    ptd.pt_momentum_1m,
    ptd.pt_momentum_3m,
    ptd.pt_momentum_6m,
    ptd.pt_momentum_1y,
    ptd.pt_median_momentum_1m,
    ptd.pt_median_momentum_3m,
    ptd.pt_acceleration_short,
    ptd.pt_acceleration_long,
    ptd.pt_consensus_convergence,
    ptd.analyst_coverage_change_1m,
    ptd.analyst_coverage_change_3m,
    ptd.analyst_coverage_change_1y,
    ptd.pt_vs_price_momentum,
    ptd.analyst_coverage_trend,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- MOMENTUM FEATURES (Monte Carlo simulation inputs)
    -- ═══════════════════════════════════════════════════════════════════════════════
    mf.price_momentum_1m,
    mf.price_momentum_3m,
    mf.price_momentum_6m,
    mf.price_momentum_1y,
    mf.price_momentum_5d,
    mf.pct_off_52w_high,
    mf.pct_above_52w_low,
    mf.range_52w_position,
    mf.beta_momentum,
    mf.volatility_regime,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- EARNINGS FEATURES (Earnings Beat model inputs)
    -- ═══════════════════════════════════════════════════════════════════════════════
    ef.eps_surprise_pct,
    ef.revenue_surprise_pct,
    ef.eps_adjustment_ratio,
    ef.gaap_adj_eps_gap_pct,
    ef.ebitda_adjustment_ratio,
    ef.eps_quarterly_trend,
    ef.eps_yoy_growth,

    -- EPS trajectory features
    etf.eps_qoq_growth,
    etf.eps_yoy_quarterly,
    etf.eps_positive_streak,
    etf.eps_cagr_3y,
    etf.eps_cagr_5y,
    etf.eps_growth_accel,
    etf.eps_vs_5y_avg,
    etf.eps_improvement_count,
    etf.eps_trajectory_score,
    etf.eps_stability,

    -- EPS comprehensive
    ec.eps_basic_fq,
    ec.eps_basic_ltm,
    ec.eps_basic_fy,
    ec.eps_adj_ltm,
    ec.eps_norm_est_fy1e,
    ec.eps_positive_years,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- PROFITABILITY (Quality factor for expected returns)
    -- ═══════════════════════════════════════════════════════════════════════════════
    pf.roe,
    pf.roa,
    pf.gross_margin_pct,
    pf.operating_margin_pct,
    pf.net_margin_pct,
    pf.ebitda_margin_pct,
    pf.roic,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- GROWTH METRICS (Forward return expectations)
    -- ═══════════════════════════════════════════════════════════════════════════════
    gf.revenue_growth_yoy,
    gf.ebitda_growth_yoy                  AS growth_ebitda_growth_yoy,
    gf.operating_income_growth,
    gf.fcf_growth,
    gf.revenue_cagr_5y,
    gf.forward_revenue_growth,
    gf.revenue_vs_5y_avg,

    -- Revenue quarterly features
    rqf.revenue_fq,
    rqf.revenue_fy,
    rqf.revenue_ltm,
    rqf.revenue_qoq_growth,
    rqf.revenue_yoy_quarterly,
    rqf.revenue_cagr_3y,
    rqf.revenue_stability_score,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- QUALITY & RISK SCORES (Risk-adjusted return inputs)
    -- ═══════════════════════════════════════════════════════════════════════════════
    qf.altman_z_score,
    qf.altman_z_trend,
    qf.current_ratio,
    qf.quick_ratio,
    qf.goodwill_to_assets_pct,
    qf.intangible_intensity,
    qf.exceptional_items_to_ebitda,

    -- Financial distress features
    fdf.distress_risk_score,
    fdf.liquidity_stress_score,
    fdf.working_capital_trend,
    fdf.cash_runway_months,
    fdf.combined_distress_score,

    -- Beta risk features
    br.beta_spread,
    br.beta_trend,
    br.high_beta_flag,
    br.low_beta_flag,
    br.beta_stability_score,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- COMPOSITE SCORES (Piotroski F-Score, Quality Momentum)
    -- ═══════════════════════════════════════════════════════════════════════════════
    cs.piotroski_f_score,
    cs.dilution_score,
    cs.quality_momentum_score,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- LEVERAGE & LIQUIDITY (Credit risk inputs)
    -- ═══════════════════════════════════════════════════════════════════════════════
    lf.debt_to_equity,
    lf.debt_to_assets,
    lf.equity_ratio,
    lf.interest_coverage,
    lf.cash_ratio,
    lf.working_capital_ratio,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- CASH FLOW FEATURES (FCF yield, self-funding)
    -- ═══════════════════════════════════════════════════════════════════════════════
    cf.cfo_to_net_income,
    cf.fcf_to_net_income,
    cf.fcf_margin,
    cf.cfo_growth_yoy,
    cf.fcf_positive_ratio,
    cf.self_funding_ratio,

    cc.fcf_fq,
    cc.fcf_ltm,
    cc.fcf_fy,
    cc.fcf_growth_yoy,
    cc.fcf_yield,
    cc.fcf_positive_years,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- DIVIDEND FEATURES (Total shareholder yield)
    -- ═══════════════════════════════════════════════════════════════════════════════
    df.dividend_streak,
    df.dividend_yield_ltm,
    df.dividend_yield_ntm,
    df.dividend_payout_ratio,
    df.fcf_dividend_coverage,
    df.buyback_yield,
    df.total_shareholder_yield,
    df.dividend_growth_expectation,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- TECHNICAL ANALYSIS (Trend signals)
    -- ═══════════════════════════════════════════════════════════════════════════════
    ta.ema_slope_20d,
    ta.ema_trend_consistency,
    ta.breakout_signal,
    ta.volatility_compression,
    ta.volatility_term_structure,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- TEMPORAL FEATURES (Earnings calendar)
    -- ═══════════════════════════════════════════════════════════════════════════════
    tf.fiscal_quarter,
    tf.fiscal_year,
    tf.days_to_earnings,
    tf.earnings_report_recency,
    tf.fiscal_year_progress,

    fcf_cal.earnings_season_flag,
    fcf_cal.pre_earnings_window,
    fcf_cal.post_earnings_window,
    fcf_cal.reporting_freshness_score,

    -- ═══════════════════════════════════════════════════════════════════════════════
    -- METADATA
    -- ═══════════════════════════════════════════════════════════════════════════════
    CURRENT_TIMESTAMP                     AS feature_calculated_at

FROM public.vw_identifier_columns                            id
         JOIN      public.equities                           e ON id.isin = e."ISIN"
         LEFT JOIN public.calc_valuation_features()          vf(isin, p_e_ratio, p_b_ratio, ev_ebitda_ratio,
                                                                ev_sales_ratio,
                                                                dividend_yield, peg_ratio) ON id.isin = vf.isin
         LEFT JOIN public.calc_sentiment_features()          sf(isin, analyst_bullish_pct, analyst_bearish_pct,
                                                                analyst_neutral_pct, analyst_conviction,
                                                                upside_potential,
                                                                price_target_spread_pct, price_target_revision_1m,
                                                                price_target_revision_3m, eps_revision_momentum,
                                                                analyst_rating_normalized, analyst_coverage_quality)
                   ON id.isin = sf.isin
         LEFT JOIN public.calc_price_target_dynamics()       ptd(isin, pt_momentum_1w, pt_momentum_1m, pt_momentum_3m,
                                                                 pt_momentum_6m, pt_momentum_1y, pt_median_momentum_1m,
                                                                 pt_median_momentum_3m, pt_acceleration_short,
                                                                 pt_acceleration_long, pt_consensus_convergence,
                                                                 analyst_coverage_change_1m, analyst_coverage_change_3m,
                                                                 analyst_coverage_change_1y, pt_vs_price_momentum,
                                                                 analyst_coverage_trend) ON id.isin = ptd.isin
         LEFT JOIN public.calc_momentum_features()           mf(isin, price_momentum_1m, price_momentum_3m,
                                                                price_momentum_6m, price_momentum_1y, price_momentum_5d,
                                                                ema_crossover_20_50, ema_crossover_50_250,
                                                                price_vs_ema_20d,
                                                                price_vs_ema_250d, pct_off_52w_high, pct_above_52w_low,
                                                                range_52w_position, beta_momentum, volatility_regime)
                   ON id.isin = mf.isin
         LEFT JOIN public.calc_earnings_features()           ef(isin, eps_surprise_pct, revenue_surprise_pct,
                                                                eps_adjustment_ratio, gaap_adj_eps_gap_pct,
                                                                ebitda_adjustment_ratio, eps_quarterly_trend,
                                                                eps_yoy_growth) ON id.isin = ef.isin
         LEFT JOIN public.calc_eps_trajectory_features()     etf(isin, eps_qoq_growth, eps_yoy_quarterly,
                                                                 eps_positive_streak, eps_cagr_3y, eps_cagr_5y,
                                                                 eps_growth_accel, eps_vs_5y_avg, eps_improvement_count,
                                                                 eps_trajectory_score, eps_stability)
                   ON id.isin = etf.isin
         LEFT JOIN public.calc_eps_comprehensive()           ec(isin, eps_basic_fq, eps_basic_ltm, eps_basic_fy,
                                                                eps_adj_ltm, eps_norm_est_fy1e, eps_growth_yoy,
                                                                eps_cagr_3y,
                                                                eps_adjustment_ratio, eps_positive_years,
                                                                eps_trajectory_score) ON id.isin = ec.isin
         LEFT JOIN public.calc_profitability_features()      pf(isin, roe, roa, gross_margin_pct, operating_margin_pct,
                                                                net_margin_pct, ebitda_margin_pct, roic, rnd_intensity,
                                                                equity_multiplier) ON id.isin = pf.isin
         LEFT JOIN public.calc_growth_features()             gf(isin, revenue_growth_yoy, ebitda_growth_yoy,
                                                                operating_income_growth, fcf_growth, revenue_cagr_5y,
                                                                forward_revenue_growth, revenue_vs_5y_avg)
                   ON id.isin = gf.isin
         LEFT JOIN public.calc_revenue_quarterly_features()  rqf(isin, revenue_fq, revenue_fy, revenue_ltm,
                                                                 revenue_5y_avg,
                                                                 revenue_1fqfq, revenue_2fqfq, revenue_3fqfq,
                                                                 revenue_4fqfq,
                                                                 revenue_1fy, revenue_2fy, revenue_3fy, revenue_4fy,
                                                                 revenue_yoy_growth, revenue_vs_5y_avg,
                                                                 revenue_ltm_vs_fy,
                                                                 revenue_fq_vs_5y_avg_fq, revenue_qoq_growth,
                                                                 revenue_qoq_2q, revenue_qoq_3q, revenue_qoq_4q,
                                                                 revenue_yoy_quarterly, revenue_2y_growth,
                                                                 revenue_3y_growth, revenue_4y_growth, revenue_cagr_3y,
                                                                 revenue_cagr_4y, revenue_4q_trend, revenue_4q_avg,
                                                                 revenue_fq_vs_4q_avg, revenue_growth_flag,
                                                                 revenue_stability_score, revenue_accelerating_flag,
                                                                 revenue_positive_qoq_streak) ON id.isin = rqf.isin
         LEFT JOIN public.calc_quality_features()            qf(isin, has_goodwill_impairment, has_asset_writedown,
                                                                has_restructuring, goodwill_to_assets_pct,
                                                                intangible_intensity, exceptional_items_to_ebitda,
                                                                altman_z_score, altman_z_trend, current_ratio,
                                                                quick_ratio)
                   ON id.isin = qf.isin
         LEFT JOIN public.calc_financial_distress_features() fdf(isin, distress_risk_score, liquidity_stress_score,
                                                                 working_capital_trend, cash_runway_months,
                                                                 combined_distress_score, wc_deteriorating_flag,
                                                                 retained_earnings_growth, accumulated_deficit_flag,
                                                                 adequate_cash_buffer) ON id.isin = fdf.isin
         LEFT JOIN public.calc_beta_risk_features()          br(isin, beta_1y, beta_5y, beta_spread, beta_trend,
                                                                high_beta_flag, low_beta_flag, beta_stability_score)
                   ON id.isin = br.isin
         LEFT JOIN public.calc_composite_scores()            cs(isin, piotroski_f_score, dilution_score, quality_momentum_score)
                   ON id.isin = cs.isin
         LEFT JOIN public.calc_leverage_features()           lf(isin, debt_to_equity, debt_to_assets, equity_ratio,
                                                                interest_coverage, current_ratio, cash_ratio,
                                                                working_capital_ratio) ON id.isin = lf.isin
         LEFT JOIN public.calc_cashflow_features()           cf(isin, cfo_to_net_income, fcf_to_net_income, fcf_margin,
                                                                cfo_growth_yoy, fcf_positive_ratio,
                                                                acquisition_intensity,
                                                                self_funding_ratio) ON id.isin = cf.isin
         LEFT JOIN public.calc_cashflow_comprehensive()      cc(isin, cfo_fq, cfo_ltm, cfo_fy, fcf_fq, fcf_ltm, fcf_fy,
                                                                cfo_growth_yoy, fcf_growth_yoy, cfo_to_net_income,
                                                                fcf_margin, fcf_yield, cfo_positive_years,
                                                                fcf_positive_years, cash_flow_quality_score)
                   ON id.isin = cc.isin
         LEFT JOIN public.calc_dividend_features()           df(isin, dividend_streak, dividend_yield_ltm,
                                                                dividend_yield_ntm, dividend_payout_ratio,
                                                                fcf_dividend_coverage, buyback_yield,
                                                                total_shareholder_yield, dividend_growth_expectation)
                   ON id.isin = df.isin
         LEFT JOIN public.calc_technical_analysis_features() ta(isin, ema_slope_20d, ema_trend_consistency,
                                                                price_vs_ema_100d, near_52w_high_flag,
                                                                near_52w_low_flag,
                                                                volume_momentum_score, breakout_signal,
                                                                high_volume_flag,
                                                                low_volume_flag, volatility_compression,
                                                                volatility_term_structure) ON id.isin = ta.isin
         LEFT JOIN public.calc_temporal_features()           tf(isin, fiscal_quarter, fiscal_month, fiscal_year,
                                                                days_to_earnings, earnings_report_recency,
                                                                reporting_lag,
                                                                fiscal_year_progress) ON id.isin = tf.isin
         LEFT JOIN public.calc_fiscal_calendar_features()    fcf_cal(isin, days_since_last_report, days_to_fy_end,
                                                                     is_quarter_end_month, is_fy_end_month,
                                                                     earnings_season_flag, pre_earnings_window,
                                                                     post_earnings_window, reporting_freshness_score,
                                                                     fiscal_quarter_progress) ON id.isin = fcf_cal.isin;

-- ═══════════════════════════════════════════════════════════════════════════════
-- COMMENTS & INDEXES
-- ═══════════════════════════════════════════════════════════════════════════════

COMMENT ON MATERIALIZED VIEW mv_expected_returns IS 'Materialized view for Expected Returns Analytics (v2.5).
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

-- Primary index for fast lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_expected_returns_isin
    ON mv_expected_returns (isin);

-- Secondary indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_mv_expected_returns_ticker
    ON mv_expected_returns (ticker);

CREATE INDEX IF NOT EXISTS idx_mv_expected_returns_sector
    ON mv_expected_returns (sector);

CREATE INDEX IF NOT EXISTS idx_mv_expected_returns_industry
    ON mv_expected_returns (industry);

-- Composite index for sector-level aggregations
CREATE INDEX IF NOT EXISTS idx_mv_expected_returns_sector_upside
    ON mv_expected_returns (sector, upside_potential DESC NULLS LAST);

-- Index for filtering by analyst conviction
CREATE INDEX IF NOT EXISTS idx_mv_expected_returns_conviction
    ON mv_expected_returns (analyst_conviction DESC NULLS LAST)
    WHERE analyst_conviction IS NOT NULL;

ALTER MATERIALIZED VIEW mv_expected_returns OWNER TO postgres;
