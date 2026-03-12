-- =============================================================================
-- MATERIALIZED VIEW - Discounted Cash Flow (mv_dcf)
-- Combines vw_features_cashflow, vw_features_balance_sheet, vw_features_earnings,
-- vw_features_growth, and vw_features_cost_structure for probabilistic DCF forecasting.
-- Standardized identifier column ordering per calculated_features_registry
-- =============================================================================

-- Pre-flight check: ensure prerequisite functions exist
DO
$$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'safe_divide') THEN
            RAISE EXCEPTION 'Required function safe_divide() does not exist. Run feature_registry.sql first.';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'pct_change') THEN
            RAISE EXCEPTION 'Required function pct_change() does not exist. Run feature_registry.sql first.';
        END IF;
    END
$$;

DROP MATERIALIZED VIEW IF EXISTS mv_dcf CASCADE;

CREATE MATERIALIZED VIEW mv_dcf AS
SELECT
    -- =========================================================================
    -- SECTION 0: IDENTIFIER COLUMNS (from calculated_features_registry)
    -- Source: vw_identifier_columns -> equities table
    -- =========================================================================
    id.isin,
    id.ticker,
    id.name,
    id.description,
    id.region,
    id.country,
    id.trading_country,
    id.exchange,
    id.sector,
    id.industry,
    id.dividend_record_frequency,
    id.earnings_report_frequency,
    id.fy_end,
    id.next_earnings_report,
    id.next_earnings_status,
    id.next_earnings_when,
    id.next_fiscal_quarter,
    id.reporting_interval,
    id.size_class,
    id.style_class,
    id.unit,
    id.dividend_record_announce_date,
    id.dividend_record_ex_date,
    id.dividend_record_payable_date,
    id.dividend_record_record_date,
    id.fy_end_date,
    id.income_statement_report_date,
    id.last_updated,
    id.next_earnings,
    id.next_fy_end_date,
    id.next_income_statement_report_date,
    id.reference_date,

    -- Market reference data from equities
    e."Market Cap"             AS market_cap,
    e."Enterprise Value"       AS enterprise_value,
    e."Last Price"             AS last_price,
    e."Shrs Out"               AS shares_outstanding,

    -- =========================================================================
    -- SECTION 1: GROWTH FEATURES
    -- Source: vw_features_growth
    -- Functions: calc_growth_features, calc_revenue_forecast_features,
    --            calc_revenue_estimate_consensus, calc_revenue_quarterly_features,
    --            calc_total_revenues_temporal
    -- =========================================================================

    -- calc_growth_features
    gf.revenue_growth_yoy,
    gf.ebitda_growth_yoy,
    gf.operating_income_growth,
    gf.fcf_growth,
    gf.revenue_cagr_5y,

    -- calc_revenue_forecast_features
    gf.forward_revenue_growth,
    gf.revenue_vs_5y_avg,
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

    -- calc_revenue_estimate_consensus
    rec.revenue_est_avg_fy1e,
    rec.revenue_est_med_fy1e,
    rec.revenue_est_avg_ntm,
    rec.revenue_est_med_ntm,
    rec.revenue_avg_med_diff_pct,
    rec.revenue_consensus_strength,
    rec.revenue_revision_trend AS revenue_revision_trend_rec,
    rec.revenue_vs_current,

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
    -- SECTION 2: CASH FLOW FEATURES
    -- Source: vw_features_cashflow
    -- Functions: calc_cashflow_features, calc_enhanced_cashflow_features,
    --            calc_cashflow_temporal_features, calc_cashflow_comprehensive,
    --            calc_fcf_growth_estimates
    -- =========================================================================

    -- calc_cashflow_features
    cf.cfo_to_net_income,
    cf.fcf_to_net_income,
    cf.fcf_margin,
    cf.cfo_growth_yoy,
    cf.fcf_positive_ratio,
    cf.acquisition_intensity,
    cf.self_funding_ratio,
    ecff.fcf_positive_years,
    ecff.fcf_always_positive,

    -- calc_enhanced_cashflow_features
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

    -- calc_enhanced_cashflow_features (capex & acquisition temporal)
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
    cc.cfo_growth_yoy          AS cfo_growth_yoy_comp,
    cc.fcf_growth_yoy,
    cc.cfo_to_net_income       AS cfo_to_net_income_comp,
    cc.fcf_margin              AS fcf_margin_comp,
    cc.fcf_yield,
    cc.cfo_positive_years,
    cc.fcf_positive_years      AS fcf_positive_years_comp,
    cc.cash_flow_quality_score AS cash_flow_quality_score_comp,

    -- calc_fcf_growth_estimates
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
    -- SECTION 3: BALANCE SHEET FEATURES
    -- Source: vw_features_balance_sheet
    -- Functions: calc_total_assets_temporal, calc_inventory_temporal_features,
    --            calc_goodwill_temporal_features
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
    itf.inventory_vs_5y_avg,
    itf.inventory_days,
    itf.inventory_turnover,
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
    -- SECTION 4: EARNINGS FEATURES
    -- Source: vw_features_earnings
    -- Functions: calc_earnings_features, calc_eps_trajectory_features,
    --            calc_eps_comprehensive, calc_eps_continuing_features,
    --            calc_gaap_adjusted_analytics, calc_gaap_revision_features
    -- =========================================================================

    -- calc_earnings_features
    ef.eps_surprise_pct,
    ef.revenue_surprise_pct,
    ef.eps_adjustment_ratio,
    ef.gaap_adj_eps_gap_pct,
    ef.ebitda_adjustment_ratio,

    -- calc_earnings_features (continued)
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
    ec.eps_growth_yoy          AS eps_growth_yoy_comp,
    ec.eps_cagr_3y             AS eps_cagr_3y_comp,
    ec.eps_adjustment_ratio    AS eps_adjustment_ratio_comp,
    ec.eps_positive_years,
    ec.eps_trajectory_score    AS eps_trajectory_score_comp,

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
    gaa.eps_adjustment_pct,
    gaa.net_income_adjustment_ratio_ltm,
    gaa.net_income_adjustment_ratio_fy,
    gaa.net_income_adjustment_pct,
    gaa.ebitda_adjustment_pct_ltm,
    gaa.ebitda_adjustment_pct_fy,
    gaa.ebit_adjustment_pct_ltm,
    gaa.ebit_adjustment_pct_fy,
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
    -- SECTION 5: COST STRUCTURE FEATURES
    -- Source: vw_features_cost_structure
    -- Functions: calc_cost_structure_features, calc_rnd_temporal_features,
    --            calc_interest_income_features
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
    -- SECTION 6: SUPPLEMENTARY - PROFITABILITY (margins needed for DCF)
    -- Source: vw_features_profitability
    -- Functions: calc_profitability_features, calc_margin_trends,
    --            calc_ebit_ebitda_comprehensive, calc_gross_profit_temporal
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
    eec.ebitda_fq,
    eec.ebitda_ltm,
    eec.ebitda_fy,
    eec.ebitda_1fy,
    eec.ebitda_2fy,
    eec.ebitda_3fy,
    eec.ebitda_4fy,
    eec.ebit_5yavgfq,
    eec.ebit_5yavgltm,
    eec.ebitda_5yavgfq,
    eec.ebitda_5yavgltm,
    eec.ebit_adj_fq,
    eec.ebit_adj_ltm,
    eec.ebit_adj_fy,
    eec.ebitda_adj_fq,
    eec.ebitda_adj_ltm,
    eec.ebitda_adj_fy,
    eec.ebit_growth_yoy,
    eec.ebitda_growth_yoy      AS ebitda_growth_yoy_comp,
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
    -- SECTION 7: SUPPLEMENTARY - UNUSUAL ITEMS (earnings quality for DCF)
    -- Source: vw_features_unusual_items
    -- Function: calc_unusual_items_features
    -- =========================================================================
    uif.other_unusual_items_ltm,
    uif.impairment_goodwill_ltm,
    uif.asset_writedown_ltm,
    uif.restructuring_charges_ltm,
    uif.total_unusual_items,
    uif.unusual_items_to_revenue,
    uif.unusual_items_to_ebitda,
    uif.has_unusual_items_flag,
    uif.earnings_quality_impact,

    -- =========================================================================
    -- SECTION 8: SUPPLEMENTARY - QUALITY & RISK (discount rate inputs)
    -- Source: vw_features_quality_risk (subset)
    -- =========================================================================
    br.beta_1y,
    br.beta_5y,
    br.beta_spread,
    br.beta_trend,
    qf.altman_z_score,
    qf.altman_z_trend,
    fdf.distress_risk_score,
    fdf.combined_distress_score,

    -- =========================================================================
    -- SECTION 9: SUPPLEMENTARY - LEVERAGE (WACC inputs)
    -- Source: vw_features_leverage_liquidity (subset)
    -- =========================================================================
    lf.debt_to_equity,
    lf.debt_to_assets,
    lf.equity_ratio,
    lf.interest_coverage,

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

    -- =========================================================================
    -- SECTION 10: SUPPLEMENTARY - EMPLOYMENT (revenue/employee for forecasts)
    -- Source: vw_features_employment (subset)
    -- =========================================================================
    emf.revenue_per_employee,
    emf.profit_per_employee,
    emf.ebitda_per_employee,

    -- =========================================================================
    -- METADATA
    -- =========================================================================
    CURRENT_TIMESTAMP          AS feature_calculated_at

FROM vw_identifier_columns                            id

-- Base equities for market data columns
         JOIN      equities                           e ON id.isin = e."ISIN"

-- Section 1: Growth (5 functions)
         LEFT JOIN calc_growth_features()             gf ON id.isin = gf.isin
         LEFT JOIN calc_revenue_forecast_features()   rff ON id.isin = rff.isin
         LEFT JOIN calc_revenue_estimate_consensus()  rec ON id.isin = rec.isin
         LEFT JOIN calc_revenue_quarterly_features()  rqf ON id.isin = rqf.isin
         LEFT JOIN calc_total_revenues_temporal()     trt ON id.isin = trt.isin

-- Section 2: Cash Flow (5 functions)
         LEFT JOIN calc_cashflow_features()           cf ON id.isin = cf.isin
         LEFT JOIN calc_enhanced_cashflow_features()  ecff ON id.isin = ecff.isin
         LEFT JOIN calc_cashflow_temporal_features()  ctf ON id.isin = ctf.isin
         LEFT JOIN calc_cashflow_comprehensive()      cc ON id.isin = cc.isin
         LEFT JOIN calc_fcf_growth_estimates()        fge ON id.isin = fge.isin

-- Section 3: Balance Sheet (3 functions)
         LEFT JOIN calc_total_assets_temporal()       tat ON id.isin = tat.isin
         LEFT JOIN calc_inventory_temporal_features() itf ON id.isin = itf.isin
         LEFT JOIN calc_goodwill_temporal_features()  gtf ON id.isin = gtf.isin

-- Section 4: Earnings (6 functions)
         LEFT JOIN calc_earnings_features()           ef ON id.isin = ef.isin
         LEFT JOIN calc_eps_trajectory_features()     etf ON id.isin = etf.isin
         LEFT JOIN calc_eps_comprehensive()           ec ON id.isin = ec.isin
         LEFT JOIN calc_eps_continuing_features()     ecf ON id.isin = ecf.isin
         LEFT JOIN calc_gaap_adjusted_analytics()     gaa ON id.isin = gaa.isin
         LEFT JOIN calc_gaap_revision_features()      grf ON id.isin = grf.isin

-- Section 5: Cost Structure (3 functions)
         LEFT JOIN calc_cost_structure_features()     csf ON id.isin = csf.isin
         LEFT JOIN calc_rnd_temporal_features()       rtf ON id.isin = rtf.isin
         LEFT JOIN calc_interest_income_features()    iif ON id.isin = iif.isin

-- Section 6: Profitability (4 functions - margins critical for DCF)
         LEFT JOIN calc_profitability_features()      pf ON id.isin = pf.isin
         LEFT JOIN calc_margin_trends()               mt ON id.isin = mt.isin
         LEFT JOIN calc_ebit_ebitda_comprehensive()   eec ON id.isin = eec.isin
         LEFT JOIN calc_gross_profit_temporal()       gpt ON id.isin = gpt.isin

-- Section 7: Unusual Items (1 function - earnings quality adjustments)
         LEFT JOIN calc_unusual_items_features()      uif ON id.isin = uif.isin

-- Section 8: Quality & Risk subset (discount rate / beta)
         LEFT JOIN calc_quality_features()            qf ON id.isin = qf.isin
         LEFT JOIN calc_beta_risk_features()          br ON id.isin = br.isin
         LEFT JOIN calc_financial_distress_features() fdf ON id.isin = fdf.isin

-- Section 9: Leverage subset (WACC components)
         LEFT JOIN calc_leverage_features()           lf ON id.isin = lf.isin
         LEFT JOIN calc_total_debt_temporal()         tdt ON id.isin = tdt.isin

-- Section 10: Employment subset (productivity for forecasts)
         LEFT JOIN calc_employment_features()         emf ON id.isin = emf.isin;

-- =============================================================================
-- INDEXES FOR OPTIMIZED QUERYING
-- =============================================================================
CREATE UNIQUE INDEX idx_mv_dcf_isin
    ON mv_dcf (isin);

CREATE INDEX idx_mv_dcf_sector
    ON mv_dcf (sector);

CREATE INDEX idx_mv_dcf_industry
    ON mv_dcf (industry);

CREATE INDEX idx_mv_dcf_country
    ON mv_dcf (country);

-- =============================================================================
-- COMMENT ON MATERIALIZED VIEW
-- =============================================================================
COMMENT ON MATERIALIZED VIEW mv_dcf IS
    'Materialized view for probabilistic Discounted Cash Flow (DCF) analysis.
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

-- =============================================================================
-- REFRESH FUNCTION
-- =============================================================================
CREATE OR REPLACE FUNCTION refresh_mv_dcf()
    RETURNS void
    LANGUAGE plpgsql
AS
$$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dcf;
    RAISE NOTICE 'mv_dcf refreshed at %', NOW();
END;
$$;

COMMENT ON FUNCTION refresh_mv_dcf() IS
    'Refreshes the mv_dcf materialized view concurrently (non-blocking).
    Call periodically after equities table updates.';
