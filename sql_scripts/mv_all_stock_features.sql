-- =============================================================================
-- UNIFIED MATERIALIZED VIEW - ALL STOCK FEATURES (RESTRUCTURED)
-- Aligns with 17 feature views covering all 54 calc_* functions
-- Standardized identifier column ordering per calculated_features_registry
-- =============================================================================

-- Pre-flight check: ensure prerequisite functions exist
DO
$$
    BEGIN
        -- Verify critical helper functions exist
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'safe_divide') THEN
            RAISE EXCEPTION 'Required function safe_divide() does not exist. Run feature_registry.sql first.';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'calc_piotroski_f_score') THEN
            RAISE EXCEPTION 'Required function calc_piotroski_f_score() does not exist. Run feature_registry.sql first.';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'pct_change') THEN
            RAISE EXCEPTION 'Required function pct_change() does not exist. Run feature_registry.sql first.';
        END IF;
    END
$$;

DROP MATERIALIZED VIEW IF EXISTS mv_all_stock_features CASCADE;

CREATE MATERIALIZED VIEW mv_all_stock_features AS
SELECT
    -- =========================================================================
    -- SECTION 0: IDENTIFIER COLUMNS (from calculated_features_registry)
    -- Source: vw_identifier_columns -> equities table
    -- =========================================================================
    id.isin,
    id.ticker,
    id.name,
    id.description,
    id.industry,
    id.sector,
    id.trading_country,
    id.region,
    id.country,
    id.exchange,

    -- Reference metadata from equities
    e."Last Updated"                      AS last_updated,
    e."Reference Date"                    AS reference_date,
    e."FY End Date"                       AS fy_end_date,
    e."Next FY End Date"                  AS next_fy_end_date,
    e."Next Earnings"                     AS next_earnings,
    e."Income Statement Report Date"      AS income_statement_report_date,
    e."Next Income Statement Report Date" AS next_income_statement_report_date,
    e."Market Cap"                        AS market_cap,
    e."Enterprise Value"                  AS enterprise_value,
    e."Last Price"                        AS last_price,
    e."Price Target"                      AS price_target,
    e."Price Target - Low"                AS price_target_low,
    e."Price Target - High"               AS price_target_high,
    e."Price Target - Median"             AS price_target_median,
    e."Price Target (YTD Ago)"            AS price_target_ytd_ago,
    e."Shrs Out"                          AS shares_outstanding,
    e."Volume (Shrs)"                     AS volume_shrs,

    -- =========================================================================
    -- SECTION 1: VALUATION RATIOS (vw_features_valuation_ratios)
    -- Source: calc_valuation_features, calc_valuation_timeseries_features,
    --         calc_extended_valuation_timeseries, calc_tangible_book_features
    -- =========================================================================
    -- calc_valuation_features
    vf.p_e_ratio,
    vf.p_b_ratio,
    vf.ev_ebitda_ratio,
    vf.ev_sales_ratio,
    vf.dividend_yield                     AS valuation_dividend_yield,
    vf.peg_ratio,

    -- calc_valuation_timeseries_features
    vts.ev_sales_trend_1y,
    vts.ev_ebitda_momentum,
    vts.p_e_momentum_yoy,
    vts.p_e_momentum_qoq,
    vts.ev_sales_vs_3y_avg,
    vts.ev_ebitda_vs_3y_avg,
    vts.p_e_vs_3y_avg,
    vts.ev_sales_forward_discount,
    vts.ev_ebitda_forward_discount,
    vts.p_e_forward_discount,
    vts.p_b_vs_5y_avg,

    -- calc_extended_valuation_timeseries
    evt.ev_sales_qoq_1q,
    evt.ev_sales_qoq_2q,
    evt.ev_sales_qoq_3q,
    evt.ev_sales_qoq_4q,
    evt.p_e_vs_5y_avg,
    evt.p_e_percentile_proxy,
    evt.valuation_mean_reversion,
    evt.ev_ebitda_qoq_trend,
    evt.p_b_momentum_yoy,
    evt.valuation_compression,
    evt.forward_pe_premium,

    -- calc_tangible_book_features
    tb.tangible_book_value_fy,
    tb.tangible_book_value_ltm,
    tb.tangible_book_per_share,
    tb.price_to_tangible_book,
    tb.tangible_equity_ratio,
    tb.intangibles_to_equity,
    tb.goodwill_to_equity,
    tb.tangible_asset_quality,
    tb.tbv_yoy_growth,
    tb.tbv_vs_calculated,

    -- =========================================================================
    -- SECTION 2: MOMENTUM (vw_features_momentum)
    -- Source: calc_momentum_features, calc_long_term_momentum_features
    -- =========================================================================
    -- calc_momentum_features
    mf.price_momentum_1m,
    mf.price_momentum_3m,
    mf.price_momentum_6m,
    mf.price_momentum_1y,
    mf.price_momentum_5d,
    mf.ema_crossover_20_50,
    mf.ema_crossover_50_250,
    mf.price_vs_ema_20d,
    mf.price_vs_ema_250d,
    mf.pct_off_52w_high,
    mf.pct_above_52w_low,
    mf.range_52w_position,
    mf.beta_momentum,
    mf.volatility_regime,

    -- calc_long_term_momentum_features
    ltm.price_momentum_3y,
    ltm.price_momentum_5y,
    ltm.long_term_trend_score,
    ltm.multi_year_high_flag,
    ltm.secular_trend_flag,

    -- =========================================================================
    -- SECTION 3: TECHNICAL ANALYSIS (vw_features_technical_analysis)
    -- Source: calc_technical_analysis_features
    -- =========================================================================
    ta.ema_slope_20d,
    ta.ema_trend_consistency,
    ta.price_vs_ema_100d,
    ta.near_52w_high_flag,
    ta.near_52w_low_flag,
    ta.volume_momentum_score,
    ta.breakout_signal,
    ta.high_volume_flag,
    ta.low_volume_flag,
    ta.volatility_compression,
    ta.volatility_term_structure,

    -- =========================================================================
    -- SECTION 4: PROFITABILITY (vw_features_profitability)
    -- Source: calc_profitability_features, calc_margin_trends,
    --         calc_ebit_ebitda_comprehensive, calc_gross_profit_temporal
    -- =========================================================================
    -- calc_profitability_features
    pf.roe,
    pf.roa,
    pf.gross_margin_pct,
    pf.operating_margin_pct,
    pf.net_margin_pct,
    pf.ebitda_margin_pct,
    pf.roic,
    pf.rnd_intensity,
    pf.equity_multiplier,

    -- calc_margin_trends
    mt.gross_margin_trend_yoy,
    mt.operating_margin_trend,
    mt.net_margin_trend_yoy,
    mt.ebitda_margin_trend,
    mt.margin_expansion_flag,
    mt.margin_stability_score,

    -- calc_ebit_ebitda_comprehensive
    eec.ebit_fq,
    eec.ebit_ltm,
    eec.ebit_fy,
    eec.ebit_1fy,
    eec.ebit_2fy,
    eec.ebit_3fy,
    eec.ebit_4fy,
    eec.ebit_1fqfq,
    eec.ebit_2fqfq,
    eec.ebit_3fqfq,
    eec.ebit_4fqfq,
    eec.ebit_5yavgfq,
    eec.ebit_5yavgltm,
    eec.ebit_adj_fq,
    eec.ebit_adj_ltm,
    eec.ebit_adj_fy,
    eec.ebitda_fq,
    eec.ebitda_ltm,
    eec.ebitda_fy,
    eec.ebitda_1fy,
    eec.ebitda_2fy,
    eec.ebitda_3fy,
    eec.ebitda_4fy,
    eec.ebitda_1fqfq,
    eec.ebitda_2fqfq,
    eec.ebitda_3fqfq,
    eec.ebitda_4fqfq,
    eec.ebitda_5yavgfq,
    eec.ebitda_5yavgltm,
    eec.ebitda_adj_fq,
    eec.ebitda_adj_ltm,
    eec.ebitda_adj_fy,
    eec.ebit_growth_yoy,
    eec.ebitda_growth_yoy,
    eec.ebit_margin_ltm,
    eec.ebitda_margin_ltm,
    eec.ebit_positive_years,
    eec.ebitda_positive_years,
    eec.ebit_qoq_growth,
    eec.ebitda_qoq_growth,
    eec.ebit_cagr_3y,
    eec.ebitda_cagr_3y,
    eec.ebit_vs_5y_avg,
    eec.ebitda_vs_5y_avg,

    -- calc_gross_profit_temporal
    gpt.gp_fq,
    gpt.gp_fy,
    gpt.gp_ltm,
    gpt.gp_1fqfq,
    gpt.gp_2fqfq,
    gpt.gp_3fqfq,
    gpt.gp_4fqfq,
    gpt.gp_1fy,
    gpt.gp_2fy,
    gpt.gp_3fy,
    gpt.gp_4fy,
    gpt.gp_qoq_growth,
    gpt.gp_yoy_growth,
    gpt.gp_margin_fq,
    gpt.gp_margin_trend,
    gpt.gp_positive_quarters,
    gpt.gp_margin_expansion,

    -- =========================================================================
    -- SECTION 5: EARNINGS (vw_features_earnings)
    -- Source: calc_earnings_features, calc_eps_trajectory_features,
    --         calc_eps_comprehensive, calc_eps_continuing_features,
    --         calc_gaap_adjusted_analytics, calc_gaap_revision_features
    -- =========================================================================
    -- calc_earnings_features
    ef.eps_surprise_pct,
    ef.revenue_surprise_pct,
    ef.eps_adjustment_ratio,
    ef.gaap_adj_eps_gap_pct,
    ef.ebitda_adjustment_ratio,
    ef.eps_quarterly_trend,
    ef.eps_yoy_growth,

    -- calc_eps_trajectory_features
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

    -- calc_eps_comprehensive
    ec.eps_basic_fq,
    ec.eps_basic_ltm,
    ec.eps_basic_fy,
    ec.eps_adj_ltm,
    ec.eps_norm_est_fy1e,
    ec.eps_positive_years,

    -- calc_eps_continuing_features
    ecf.eps_cont_ltm,
    ecf.eps_cont_fq,
    ecf.eps_cont_fy,
    ecf.eps_cont_1fqfq,
    ecf.eps_cont_2fqfq,
    ecf.eps_cont_3fqfq,
    ecf.eps_cont_4fqfq,
    ecf.eps_cont_1fy,
    ecf.eps_cont_2fy,
    ecf.eps_cont_3fy,
    ecf.eps_cont_4fy,
    ecf.eps_cont_qoq_growth,
    ecf.eps_cont_yoy_growth,
    ecf.eps_cont_cagr_3y,
    ecf.eps_cont_vs_total_eps,
    ecf.eps_cont_positive_streak,
    ecf.eps_cont_trajectory_score,
    ecf.discontinued_ops_impact,
    ecf.core_earnings_stability,

    -- calc_gaap_adjusted_analytics
    gaa.eps_adjustment_spread_ltm,
    gaa.eps_adjustment_spread_fy,
    gaa.eps_adjustment_spread_1fy,
    gaa.eps_adjustment_spread_fq,
    gaa.eps_adjustment_spread_1fqfq,
    gaa.eps_adjustment_spread_2fqfq,
    gaa.eps_adjustment_spread_3fqfq,
    gaa.eps_adjustment_spread_4fqfq,
    gaa.eps_adjustment_spread_2fy,
    gaa.eps_adjustment_spread_3fy,
    gaa.eps_adjustment_spread_4fy,
    gaa.eps_adjustment_pct,
    gaa.net_income_adjustment_ratio_ltm,
    gaa.net_income_adjustment_ratio_fy,
    gaa.net_income_adjustment_ratio_1fy,
    gaa.net_income_adjustment_ratio_fq,
    gaa.net_income_adjustment_ratio_5yavgfq,
    gaa.net_income_adjustment_ratio_1fqfq,
    gaa.net_income_adjustment_ratio_2fqfq,
    gaa.net_income_adjustment_ratio_3fqfq,
    gaa.net_income_adjustment_ratio_4fqfq,
    gaa.net_income_adjustment_ratio_2fy,
    gaa.net_income_adjustment_ratio_3fy,
    gaa.net_income_adjustment_ratio_4fy,
    gaa.net_income_adjustment_pct,
    gaa.ebitda_adjustment_pct_ltm,
    gaa.ebitda_adjustment_pct_fy,
    gaa.ebitda_adjustment_pct_1fy,
    gaa.ebitda_adjustment_pct_fq,
    gaa.ebitda_adjustment_pct_1fqfq,
    gaa.ebitda_adjustment_pct_2fqfq,
    gaa.ebitda_adjustment_pct_3fqfq,
    gaa.ebitda_adjustment_pct_4fqfq,
    gaa.ebitda_adjustment_pct_2fy,
    gaa.ebitda_adjustment_pct_3fy,
    gaa.ebitda_adjustment_pct_4fy,
    gaa.ebit_adjustment_pct_ltm,
    gaa.ebit_adjustment_pct_fy,
    gaa.ebit_adjustment_pct_1fy,
    gaa.ebit_adjustment_pct_fq,
    gaa.ebit_adjustment_pct_1fqfq,
    gaa.ebit_adjustment_pct_2fqfq,
    gaa.ebit_adjustment_pct_3fqfq,
    gaa.ebit_adjustment_pct_4fqfq,
    gaa.ebit_adjustment_pct_2fy,
    gaa.ebit_adjustment_pct_3fy,
    gaa.ebit_adjustment_pct_4fy,
    gaa.earnings_quality_score,
    gaa.earnings_quality_warning,
    gaa.forward_eps_gaap_adj_spread,

    -- calc_gaap_revision_features
    grf.gaap_revision_momentum,
    grf.gaap_revision_1m,
    grf.gaap_revision_3m,
    grf.gaap_revision_6m,
    grf.gaap_revision_1y,
    grf.gaap_vs_norm_revision_spread,
    grf.gaap_revision_acceleration,
    grf.gaap_positive_revision_flag,
    grf.revision_quality_divergence,

    -- =========================================================================
    -- SECTION 6: GROWTH (vw_features_growth)
    -- Source: calc_growth_features, calc_revenue_forecast_features,
    --         calc_revenue_quarterly_features, calc_total_revenues_temporal
    -- =========================================================================
    -- calc_growth_features
    gf.revenue_growth_yoy,
    gf.ebitda_growth_yoy                  AS growth_ebitda_growth_yoy,
    gf.operating_income_growth,
    gf.fcf_growth,
    gf.revenue_cagr_5y,
    gf.forward_revenue_growth,
    gf.revenue_vs_5y_avg,

    -- calc_revenue_forecast_features
    rff.revenue_est_spread,
    rff.revenue_beat_potential,
    rff.revenue_est_revision_trend,
    rff.ebitda_est_vs_actual,
    rff.forward_revenue_multiple,
    rff.revenue_estimate_count,
    rff.revenue_guidance_gap,
    rff.consensus_revenue_growth,
    rff.ebit_estimate_spread,
    rff.forward_ebitda_margin,
    rff.revenue_acceleration,
    rff.estimate_confidence_score,

    -- calc_revenue_quarterly_features
    rqf.revenue_fq,
    rqf.revenue_fy,
    rqf.revenue_ltm,
    rqf.revenue_5y_avg,
    rqf.revenue_1fqfq,
    rqf.revenue_2fqfq,
    rqf.revenue_3fqfq,
    rqf.revenue_4fqfq,
    rqf.revenue_1fy,
    rqf.revenue_2fy,
    rqf.revenue_3fy,
    rqf.revenue_4fy,
    rqf.revenue_qoq_growth,
    rqf.revenue_qoq_2q,
    rqf.revenue_qoq_3q,
    rqf.revenue_qoq_4q,
    rqf.revenue_yoy_quarterly,
    rqf.revenue_2y_growth,
    rqf.revenue_3y_growth,
    rqf.revenue_4y_growth,
    rqf.revenue_cagr_3y,
    rqf.revenue_cagr_4y,
    rqf.revenue_4q_trend,
    rqf.revenue_4q_avg,
    rqf.revenue_fq_vs_4q_avg,
    rqf.revenue_growth_flag,
    rqf.revenue_stability_score,
    rqf.revenue_accelerating_flag,
    rqf.revenue_positive_qoq_streak,

    -- calc_total_revenues_temporal
    trt.revenue_5yavgfq,
    trt.revenue_5yavgltm,
    trt.revenue_vs_5y_avg_fq,
    trt.revenue_vs_5y_avg_ltm,
    trt.revenue_fq_vs_avg,
    trt.revenue_momentum,

    -- =========================================================================
    -- SECTION 7: QUALITY & RISK (vw_features_quality_risk)
    -- Source: calc_quality_features, calc_beta_risk_features,
    --         calc_financial_distress_features, calc_accounting_quality_features,
    --         calc_quality_features_comprehensive
    -- =========================================================================
    -- calc_quality_features
    qf.has_goodwill_impairment,
    qf.has_asset_writedown,
    qf.has_restructuring,
    qf.goodwill_to_assets_pct,
    qf.intangible_intensity,
    qf.exceptional_items_to_ebitda,
    qf.altman_z_score,
    qf.altman_z_trend,
    qf.current_ratio,
    qf.quick_ratio,

    -- calc_beta_risk_features
    br.beta_1y,
    br.beta_5y,
    br.beta_spread,
    br.beta_trend,
    br.high_beta_flag,
    br.low_beta_flag,
    br.beta_stability_score,

    -- calc_financial_distress_features
    fdf.distress_risk_score,
    fdf.liquidity_stress_score,
    fdf.working_capital_trend,
    fdf.cash_runway_months,
    fdf.combined_distress_score,
    fdf.wc_deteriorating_flag,
    fdf.retained_earnings_growth,
    fdf.accumulated_deficit_flag,
    fdf.adequate_cash_buffer,

    -- calc_accounting_quality_features
    aqf.goodwill_change_rate,
    aqf.restructuring_intensity,
    aqf.exceptional_items_frequency,
    aqf.merger_impact_ratio,
    aqf.non_operating_income_share,
    aqf.asset_sale_boost,
    aqf.accounting_quality_score,

    -- calc_quality_features_comprehensive
    qfc.goodwill_impairment_ltm,
    qfc.asset_writedown_ltm,
    qfc.restructuring_ltm,
    qfc.has_goodwill_impairment_ltm,
    qfc.goodwill_impairment_frequency,
    qfc.asset_writedown_frequency,
    qfc.restructuring_frequency,
    qfc.exceptional_items_total_ltm,
    qfc.quality_issues_count_5y,

    -- =========================================================================
    -- SECTION 8: LEVERAGE & LIQUIDITY (vw_features_leverage_liquidity)
    -- Source: calc_leverage_features, calc_efficiency_ratios,
    --         calc_balance_sheet_dynamics, calc_working_capital_temporal,
    --         calc_total_debt_temporal, calc_working_capital_deep_features
    -- =========================================================================
    -- calc_leverage_features
    lf.debt_to_equity,
    lf.debt_to_assets,
    lf.equity_ratio,
    lf.interest_coverage,
    lf.cash_ratio,
    lf.working_capital_ratio,

    -- calc_efficiency_ratios
    er.asset_turnover,
    er.inventory_turnover,
    er.receivables_days,
    er.working_capital_turns,

    -- calc_balance_sheet_dynamics
    bsd.cash_to_assets_pct,
    bsd.cash_change_qoq,
    bsd.cash_vs_5y_avg,
    bsd.inventory_change_yoy,
    bsd.inventory_vs_5y_avg,
    bsd.receivables_change_yoy,
    bsd.receivables_vs_5y_avg,
    bsd.working_capital_vs_5y_avg,
    bsd.retained_earnings_vs_5y,
    bsd.intangibles_growth_flag,
    bsd.asset_quality_score,
    bsd.balance_sheet_strength,
    bsd.debt_maturity_risk,

    -- calc_working_capital_temporal
    wct.wc_fq,
    wct.wc_fy,
    wct.wc_ltm,
    wct.wc_5yavgfy,
    wct.wc_1fq,
    wct.wc_2fq,
    wct.wc_3fq,
    wct.wc_4fq,
    wct.wc_1fy,
    wct.wc_2fy,
    wct.wc_3fy,
    wct.wc_4fy,
    wct.wc_qoq_change,
    wct.wc_yoy_change,
    wct.wc_4q_trend,
    wct.wc_vs_5y_avg,
    wct.wc_positive_quarters,
    wct.wc_improving_flag,
    wct.wc_volatility,

    -- calc_total_debt_temporal
    tdt.debt_fq,
    tdt.debt_fy,
    tdt.debt_ltm,
    tdt.debt_1fq,
    tdt.debt_2fq,
    tdt.debt_3fq,
    tdt.debt_4fq,
    tdt.debt_1fy,
    tdt.debt_2fy,
    tdt.debt_3fy,
    tdt.debt_4fy,
    tdt.debt_qoq_change,
    tdt.debt_yoy_change,
    tdt.debt_4q_trend,
    tdt.debt_3y_cagr,
    tdt.debt_deleveraging,
    tdt.debt_to_equity_trend,

    -- calc_working_capital_deep_features
    wcd.wc_to_revenue,
    wcd.wc_to_assets,
    wcd.days_working_capital,
    wcd.wc_efficiency_score,
    wcd.negative_wc_flag,

    -- =========================================================================
    -- SECTION 9: ANALYST SENTIMENT (vw_features_analyst_sentiment)
    -- Source: calc_sentiment_features, calc_price_target_dynamics
    -- =========================================================================
    -- calc_sentiment_features
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

    -- calc_price_target_dynamics
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

    -- =========================================================================
    -- SECTION 10: DIVIDENDS (vw_features_dividends)
    -- Source: calc_dividend_features, calc_dividend_timing,
    --         calc_dividend_yield_comprehensive
    -- =========================================================================
    -- calc_dividend_features
    df.dividend_streak,
    df.dividend_yield_ltm,
    df.dividend_yield_ntm,
    df.dividend_payout_ratio,
    df.fcf_dividend_coverage,
    df.buyback_yield,
    df.total_shareholder_yield,
    df.dividend_growth_expectation,

    -- calc_dividend_timing
    dt.days_since_ex_date,
    dt.days_to_payment,
    dt.dividend_announced_flag,
    dt.ex_date_approaching_flag,
    dt.dividend_frequency_score,
    dt.dividend_consistency,
    dt.recent_dividend_change,
    dt.dividend_yield_vs_5y_avg,

    -- calc_dividend_yield_comprehensive
    dyc.div_yield_ltm,
    dyc.div_yield_ntm,
    dyc.div_yield_ind,
    dyc.div_yield_1fy_ind,
    dyc.div_yield_5y_avg,
    dyc.div_yield_vs_5y_avg,
    dyc.div_yield_growth_expected,
    dyc.high_yield_flag,
    dyc.sustainable_dividend_flag,

    -- =========================================================================
    -- SECTION 11: EMPLOYMENT (vw_features_employment)
    -- Source: calc_employment_features, calc_employment_dynamics
    -- =========================================================================
    -- calc_employment_features
    emf.revenue_per_employee,
    emf.profit_per_employee,
    emf.ebitda_per_employee,
    emf.assets_per_employee,
    emf.fte_growth_1y_pct,
    emf.fte_growth_3y_pct,
    emf.workforce_stability,

    -- calc_employment_dynamics
    ed.fte_growth_2y_pct,
    ed.fte_acceleration,
    ed.workforce_volatility,
    ed.hiring_intensity,
    ed.productivity_trend,
    ed.headcount_vs_revenue,
    ed.workforce_efficiency_gain,
    ed.layoff_risk_flag,
    ed.rapid_hiring_flag,
    ed.sustainable_growth_flag,

    -- =========================================================================
    -- SECTION 12: CASH FLOW (vw_features_cashflow)
    -- Source: calc_cashflow_features, calc_enhanced_cashflow_features,
    --         calc_cashflow_temporal_features, calc_cashflow_comprehensive
    -- =========================================================================
    -- calc_cashflow_features
    cf.cfo_to_net_income,
    cf.fcf_to_net_income,
    cf.fcf_margin,
    cf.cfo_growth_yoy,
    cf.fcf_positive_ratio,
    cf.acquisition_intensity,
    cf.self_funding_ratio,

    -- calc_enhanced_cashflow_features
    ecff.fcf_positive_years,
    ecff.fcf_always_positive,
    ecff.capex_vs_5y_avg,
    ecff.underinvestment_flag,
    ecff.cfo_share_of_cf,
    ecff.cfi_share_of_cf,
    ecff.cff_share_of_cf,
    ecff.self_funding_flag,
    ecff.acquisition_to_fcf,
    ecff.sustainable_ma_flag,
    ecff.fcf_4q_improvement,
    ecff.cash_flow_quality_score,
    ecff.capex_yoy_growth,
    ecff.capex_qoq_growth,
    ecff.capex_3y_trend,
    ecff.capex_volatility,
    ecff.capex_acceleration,
    ecff.capex_cut_flag,
    ecff.overinvestment_flag,
    ecff.acquisitions_yoy_growth,
    ecff.acquisitions_vs_5y_avg,
    ecff.acquisitions_ltm_total,
    ecff.ma_intensity_score,
    ecff.serial_acquirer_flag,
    ecff.acquisition_pause_flag,
    ecff.total_investment_to_cfo,
    ecff.organic_vs_inorganic,
    ecff.investment_efficiency,

    -- calc_cashflow_temporal_features
    ctf.cfo_quarterly_trend,
    ctf.cfo_yoy_quarterly,
    ctf.cfi_quarterly_trend,
    ctf.cff_quarterly_trend,
    ctf.fcf_quarterly_trend,
    ctf.cfo_positive_quarters,
    ctf.cfi_negative_quarters,
    ctf.cff_pattern_score,
    ctf.cash_burn_rate,
    ctf.cf_volatility_score,
    ctf.operating_cf_momentum,
    ctf.financing_dependency,

    -- calc_cashflow_comprehensive
    cc.cfo_fq,
    cc.cfo_ltm,
    cc.cfo_fy,
    cc.fcf_fq,
    cc.fcf_ltm,
    cc.fcf_fy,
    cc.fcf_growth_yoy,
    cc.fcf_yield,
    cc.cfo_positive_years,
    cc.fcf_positive_years     AS fcf_positive_years_comp,

    -- calc_fcf_growth_estimates (NEW)
    fge.fcf_est_fy1,
    fge.fcf_est_fy2,
    fge.fcf_est_fy3,
    fge.fcf_est_fy4,
    fge.fcf_est_fy5,
    fge.fcf_est_growth_fy1_vs_ltm,
    fge.fcf_est_growth_fy2_vs_fy1,
    fge.fcf_est_growth_fy3_vs_fy2,
    fge.fcf_est_growth_fy4_vs_fy3,
    fge.fcf_est_growth_fy5_vs_fy4,
    fge.fcf_est_cagr_3y,
    fge.fcf_est_cagr_5y,
    fge.fcf_est_margin_fy1,
    fge.fcf_est_yield_fy1,
    fge.fcf_est_growth_acceleration,
    fge.fcf_est_growth_deceleration,
    fge.fcf_est_trajectory_score,
    fge.fcf_est_always_positive,
    fge.fcf_est_vs_historical,
    fge.fcf_est_capex_implied_ratio,

    -- =========================================================================
    -- SECTION 13: TEMPORAL (vw_features_temporal)
    -- Source: calc_temporal_features, calc_fiscal_calendar_features
    -- =========================================================================
    -- calc_temporal_features
    tf.fiscal_quarter,
    tf.fiscal_month,
    tf.fiscal_year,
    tf.days_to_earnings,
    tf.earnings_report_recency,
    tf.reporting_lag,
    tf.fiscal_year_progress,

    -- calc_fiscal_calendar_features
    fcf.days_since_last_report,
    fcf.days_to_fy_end,
    fcf.is_quarter_end_month,
    fcf.is_fy_end_month,
    fcf.earnings_season_flag,
    fcf.pre_earnings_window,
    fcf.post_earnings_window,
    fcf.reporting_freshness_score,
    fcf.fiscal_quarter_progress,

    -- =========================================================================
    -- SECTION 14: BALANCE SHEET (vw_features_balance_sheet)
    -- Source: calc_total_assets_temporal, calc_inventory_temporal_features,
    --         calc_goodwill_temporal_features
    -- =========================================================================
    -- calc_total_assets_temporal
    tat.assets_fq,
    tat.assets_fy,
    tat.assets_ltm,
    tat.assets_1fq,
    tat.assets_2fq,
    tat.assets_3fq,
    tat.assets_4fq,
    tat.assets_1fy,
    tat.assets_2fy,
    tat.assets_3fy,
    tat.assets_4fy,
    tat.assets_qoq_growth,
    tat.assets_yoy_growth,
    tat.assets_3y_cagr,
    tat.asset_growth_accel,
    tat.asset_base_stable,

    -- calc_inventory_temporal_features
    itf.inventory_ltm,
    itf.inventory_fq,
    itf.inventory_fy,
    itf.inventory_1fq,
    itf.inventory_2fq,
    itf.inventory_3fq,
    itf.inventory_4fq,
    itf.inventory_1fy,
    itf.inventory_2fy,
    itf.inventory_3fy,
    itf.inventory_4fy,
    itf.inventory_qoq_change,
    itf.inventory_yoy_change,
    itf.inventory_4q_trend,
    itf.inventory_vs_5y_avg               AS inventory_vs_5y_avg_itf,
    itf.inventory_days,
    itf.inventory_turnover                AS inventory_turnover_itf,
    itf.inventory_to_revenue,
    itf.inventory_to_assets,
    itf.inventory_buildup_flag,
    itf.inventory_reduction_flag,
    itf.inventory_volatility,

    -- calc_goodwill_temporal_features
    gtf.goodwill_fq,
    gtf.goodwill_ltm,
    gtf.goodwill_fy,
    gtf.goodwill_1fq,
    gtf.goodwill_2fq,
    gtf.goodwill_3fq,
    gtf.goodwill_4fq,
    gtf.goodwill_1fy,
    gtf.goodwill_2fy,
    gtf.goodwill_3fy,
    gtf.goodwill_4fy,
    gtf.goodwill_qoq_change,
    gtf.goodwill_yoy_change,
    gtf.goodwill_3y_growth,
    gtf.goodwill_vs_5y_avg,
    gtf.recent_acquisition_flag,
    gtf.goodwill_accumulation_rate,
    gtf.goodwill_to_assets_trend,
    gtf.impairment_risk_score,
    gtf.goodwill_concentration,

    -- =========================================================================
    -- SECTION 15: COST STRUCTURE (vw_features_cost_structure)
    -- Source: calc_cost_structure_features, calc_rnd_temporal_features,
    --         calc_interest_income_features
    -- =========================================================================
    -- calc_cost_structure_features
    csf.cogs_to_revenue,
    csf.opex_to_revenue,
    csf.sga_to_revenue,
    csf.rnd_to_revenue,
    csf.interest_to_revenue,
    csf.sga_trend_yoy,
    csf.operating_leverage_proxy,
    csf.cost_efficiency_score,
    csf.marketing_to_revenue,
    csf.marketing_trend_yoy,
    csf.marketing_vs_5y_avg,
    csf.sga_vs_5y_avg,
    csf.sga_efficiency_trend,

    -- calc_rnd_temporal_features
    rtf.rnd_ltm,
    rtf.rnd_fq,
    rtf.rnd_fy,
    rtf.rnd_1fqfq,
    rtf.rnd_2fqfq,
    rtf.rnd_3fqfq,
    rtf.rnd_4fqfq,
    rtf.rnd_1fy,
    rtf.rnd_2fy,
    rtf.rnd_3fy,
    rtf.rnd_4fy,
    rtf.rnd_intensity_ltm,
    rtf.rnd_intensity_fy,
    rtf.rnd_intensity_trend,
    rtf.rnd_qoq_growth,
    rtf.rnd_yoy_growth,
    rtf.rnd_cagr_3y,
    rtf.rnd_per_employee,
    rtf.rnd_to_gross_profit,
    rtf.rnd_roi_proxy,
    rtf.rnd_increasing_flag,
    rtf.rnd_cut_flag,
    rtf.high_rnd_intensity_flag,

    -- calc_interest_income_features
    iif.interest_income_ltm,
    iif.interest_expense_ltm,
    iif.net_interest_income,
    iif.interest_coverage_ratio,
    iif.interest_income_to_revenue,
    iif.interest_expense_to_revenue,
    iif.net_interest_margin_proxy,

    -- =========================================================================
    -- SECTION 16: COMPOSITE SCORES (vw_features_composite_scores)
    -- Source: calc_composite_scores, calc_net_income_comprehensive
    -- =========================================================================
    -- calc_composite_scores
    cs.piotroski_f_score,
    cs.dilution_score,
    cs.quality_momentum_score,

    -- calc_net_income_comprehensive
    nic.net_income_is_fq,
    nic.net_income_is_ltm,
    nic.net_income_is_fy,
    nic.net_income_adj_ltm,
    nic.normalized_ni_ltm,
    nic.net_income_is_1fqfq,
    nic.net_income_is_2fqfq,
    nic.net_income_is_3fqfq,
    nic.net_income_is_4fqfq,
    nic.net_income_is_1fy,
    nic.net_income_is_2fy,
    nic.net_income_is_3fy,
    nic.net_income_is_4fy,
    nic.net_income_is_5yavgfq,
    nic.net_income_is_5yavgltm,
    nic.normalized_ni_5yavgfq,
    nic.normalized_ni_5yavgltm,
    nic.net_income_growth_yoy,
    nic.net_income_margin_ltm,
    nic.ni_adjustment_ratio,
    nic.net_income_positive_years,
    nic.earnings_quality_composite,
    nic.net_income_qoq_growth,
    nic.net_income_yoy_quarterly,
    nic.net_income_vs_5y_avg,
    nic.normalized_ni_vs_5y_avg,

    -- =========================================================================
    -- SECTION 17: UNUSUAL ITEMS (vw_features_unusual_items)
    -- Source: calc_unusual_items_features
    -- =========================================================================
    uif.other_unusual_items_ltm,
    uif.impairment_goodwill_ltm,
    uif.asset_writedown_ltm               AS unusual_asset_writedown_ltm,
    uif.restructuring_charges_ltm,
    uif.total_unusual_items,
    uif.unusual_items_to_revenue,
    uif.unusual_items_to_ebitda,
    uif.has_unusual_items_flag,
    uif.earnings_quality_impact,

    -- =========================================================================
    -- METADATA: Timestamp for refresh tracking
    -- =========================================================================
    CURRENT_TIMESTAMP                     AS feature_calculated_at

FROM vw_identifier_columns                               id
-- Base equities for reference columns
         JOIN      postgres.public.equities              e ON id.isin = e."ISIN"

-- Section 1: Valuation Ratios
         LEFT JOIN calc_valuation_features()             vf ON id.isin = vf.isin
         LEFT JOIN calc_valuation_timeseries_features()  vts ON id.isin = vts.isin
         LEFT JOIN calc_extended_valuation_timeseries()  evt ON id.isin = evt.isin
         LEFT JOIN calc_tangible_book_features()         tb ON id.isin = tb.isin

-- Section 2: Momentum
         LEFT JOIN calc_momentum_features()              mf ON id.isin = mf.isin
         LEFT JOIN calc_long_term_momentum_features()    ltm ON id.isin = ltm.isin

-- Section 3: Technical Analysis
         LEFT JOIN calc_technical_analysis_features()    ta ON id.isin = ta.isin

-- Section 4: Profitability
         LEFT JOIN calc_profitability_features()         pf ON id.isin = pf.isin
         LEFT JOIN calc_margin_trends()                  mt ON id.isin = mt.isin
         LEFT JOIN calc_ebit_ebitda_comprehensive()      eec ON id.isin = eec.isin
         LEFT JOIN calc_gross_profit_temporal()          gpt ON id.isin = gpt.isin

-- Section 5: Earnings
         LEFT JOIN calc_earnings_features()              ef ON id.isin = ef.isin
         LEFT JOIN calc_eps_trajectory_features()        etf ON id.isin = etf.isin
         LEFT JOIN calc_eps_comprehensive()              ec ON id.isin = ec.isin
         LEFT JOIN calc_eps_continuing_features()        ecf ON id.isin = ecf.isin
         LEFT JOIN calc_gaap_adjusted_analytics()        gaa ON id.isin = gaa.isin
         LEFT JOIN calc_gaap_revision_features()         grf ON id.isin = grf.isin

-- Section 6: Growth
         LEFT JOIN calc_growth_features()                gf ON id.isin = gf.isin
         LEFT JOIN calc_revenue_forecast_features()      rff ON id.isin = rff.isin
         LEFT JOIN calc_revenue_estimate_consensus() rec ON id.isin = rec.isin
         LEFT JOIN calc_revenue_quarterly_features()     rqf ON id.isin = rqf.isin
         LEFT JOIN calc_total_revenues_temporal()        trt ON id.isin = trt.isin

-- Section 7: Quality & Risk
         LEFT JOIN calc_quality_features()               qf ON id.isin = qf.isin
         LEFT JOIN calc_beta_risk_features()             br ON id.isin = br.isin
         LEFT JOIN calc_financial_distress_features()    fdf ON id.isin = fdf.isin
         LEFT JOIN calc_accounting_quality_features()    aqf ON id.isin = aqf.isin
         LEFT JOIN calc_quality_features_comprehensive() qfc ON id.isin = qfc.isin

-- Section 8: Leverage & Liquidity
         LEFT JOIN calc_leverage_features()              lf ON id.isin = lf.isin
         LEFT JOIN calc_efficiency_ratios()              er ON id.isin = er.isin
         LEFT JOIN calc_balance_sheet_dynamics()         bsd ON id.isin = bsd.isin
         LEFT JOIN calc_working_capital_temporal()       wct ON id.isin = wct.isin
         LEFT JOIN calc_total_debt_temporal()            tdt ON id.isin = tdt.isin
         LEFT JOIN calc_working_capital_deep_features()  wcd ON id.isin = wcd.isin

-- Section 9: Analyst Sentiment
         LEFT JOIN calc_sentiment_features()             sf ON id.isin = sf.isin
         LEFT JOIN calc_price_target_dynamics()          ptd ON id.isin = ptd.isin

-- Section 10: Dividends
         LEFT JOIN calc_dividend_features()              df ON id.isin = df.isin
         LEFT JOIN calc_dividend_timing()                dt ON id.isin = dt.isin
         LEFT JOIN calc_dividend_yield_comprehensive()   dyc ON id.isin = dyc.isin

-- Section 11: Employment
         LEFT JOIN calc_employment_features()            emf ON id.isin = emf.isin
         LEFT JOIN calc_employment_dynamics()            ed ON id.isin = ed.isin

-- Section 12: Cash Flow
         LEFT JOIN calc_cashflow_features()              cf ON id.isin = cf.isin
         LEFT JOIN calc_enhanced_cashflow_features()     ecff ON id.isin = ecff.isin
         LEFT JOIN calc_cashflow_temporal_features()     ctf ON id.isin = ctf.isin
         LEFT JOIN calc_cashflow_comprehensive()         cc ON id.isin = cc.isin
         LEFT JOIN calc_fcf_growth_estimates()           fge ON id.isin = fge.isin

-- Section 13: Temporal
         LEFT JOIN calc_temporal_features()              tf ON id.isin = tf.isin
         LEFT JOIN calc_fiscal_calendar_features()       fcf ON id.isin = fcf.isin

-- Section 14: Balance Sheet
         LEFT JOIN calc_total_assets_temporal()          tat ON id.isin = tat.isin
         LEFT JOIN calc_inventory_temporal_features()    itf ON id.isin = itf.isin
         LEFT JOIN calc_goodwill_temporal_features()     gtf ON id.isin = gtf.isin

-- Section 15: Cost Structure
         LEFT JOIN calc_cost_structure_features()        csf ON id.isin = csf.isin
         LEFT JOIN calc_rnd_temporal_features()          rtf ON id.isin = rtf.isin
         LEFT JOIN calc_interest_income_features()       iif ON id.isin = iif.isin

-- Section 16: Composite Scores
         LEFT JOIN calc_composite_scores()               cs ON id.isin = cs.isin
         LEFT JOIN calc_net_income_comprehensive()       nic ON id.isin = nic.isin

-- Section 17: Unusual Items
         LEFT JOIN calc_unusual_items_features()         uif ON id.isin = uif.isin;

-- =============================================================================
-- INDEXES FOR OPTIMIZED QUERYING
-- =============================================================================
CREATE UNIQUE INDEX idx_mv_all_stock_features_isin
    ON mv_all_stock_features (isin);

-- =============================================================================
-- COMMENT ON MATERIALIZED VIEW
-- =============================================================================
COMMENT ON MATERIALIZED VIEW mv_all_stock_features IS
    'Unified materialized view containing all calculated stock features.
    Covers 17 feature categories from 54 calc_* functions:
    1. Valuation Ratios (4 functions)
    2. Momentum (2 functions)
    3. Technical Analysis (1 function)
    4. Profitability (4 functions)
    5. Earnings (6 functions)
    6. Growth (4 functions)
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

    Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;';

-- =============================================================================
-- REFRESH FUNCTION
-- =============================================================================
CREATE OR REPLACE FUNCTION refresh_all_stock_features()
    RETURNS void
    LANGUAGE plpgsql
AS
$$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;
    RAISE NOTICE 'mv_all_stock_features refreshed at %', NOW();
END;
$$;

COMMENT ON FUNCTION refresh_all_stock_features() IS
    'Refreshes the mv_all_stock_features materialized view concurrently (non-blocking).
    Call periodically after equities table updates.';
