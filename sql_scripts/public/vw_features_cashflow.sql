CREATE VIEW public.vw_features_cashflow
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, cfo_to_net_income, fcf_to_net_income,
			 fcf_margin, cfo_growth_yoy, fcf_positive_ratio, acquisition_intensity, self_funding_ratio,
			 fcf_positive_years, fcf_always_positive, capex_vs_5y_avg, underinvestment_flag, cfo_share_of_cf,
			 cfi_share_of_cf, cff_share_of_cf, self_funding_flag, acquisition_to_fcf, sustainable_ma_flag,
			 fcf_4q_improvement, cash_flow_quality_score, capex_yoy_growth, capex_qoq_growth, capex_3y_trend,
			 capex_volatility, capex_acceleration, capex_cut_flag, overinvestment_flag, acquisitions_yoy_growth,
			 acquisitions_vs_5y_avg, acquisitions_ltm_total, ma_intensity_score, serial_acquirer_flag,
			 acquisition_pause_flag, total_investment_to_cfo, organic_vs_inorganic, investment_efficiency,
			 cfo_quarterly_trend, cfo_yoy_quarterly, cfi_quarterly_trend, cff_quarterly_trend, fcf_quarterly_trend,
			 cfo_positive_quarters, cfi_negative_quarters, cff_pattern_score, cash_burn_rate, cf_volatility_score,
			 operating_cf_momentum, financing_dependency, cfo_fq, cfo_ltm, cfo_fy, fcf_fq, fcf_ltm, fcf_fy,
			 cfo_growth_yoy_comp, fcf_growth_yoy, cfo_to_net_income_comp, fcf_margin_comp, fcf_yield,
			 cfo_positive_years, fcf_positive_years_comp, cash_flow_quality_score_comp, fcf_est_fy1, fcf_est_fy2,
			 fcf_est_fy3, fcf_est_fy4, fcf_est_fy5, fcf_est_growth_fy1_vs_ltm, fcf_est_growth_fy2_vs_fy1,
			 fcf_est_growth_fy3_vs_fy2, fcf_est_growth_fy4_vs_fy3, fcf_est_growth_fy5_vs_fy4, fcf_est_cagr_3y,
			 fcf_est_cagr_5y, fcf_est_margin_fy1, fcf_est_yield_fy1, fcf_est_growth_acceleration,
			 fcf_est_growth_deceleration, fcf_est_trajectory_score, fcf_est_always_positive, fcf_est_vs_historical,
			 fcf_est_capex_implied_ratio)
AS
-- missing source code
;

COMMENT ON VIEW public.vw_features_cashflow IS 'Cash flow metrics including CFO, FCF, CapEx analysis, and cash flow quality.
    Source functions: calc_cashflow_features, calc_enhanced_cashflow_features,
    calc_cashflow_temporal_features, calc_cashflow_comprehensive';

ALTER TABLE public.vw_features_cashflow
	OWNER TO postgres;