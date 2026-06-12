CREATE VIEW public.vw_features_earnings
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, eps_surprise_pct,
			 revenue_surprise_pct, eps_adjustment_ratio, gaap_adj_eps_gap_pct, ebitda_adjustment_ratio,
			 eps_quarterly_trend, eps_yoy_growth, eps_qoq_growth, eps_yoy_quarterly, eps_positive_streak, eps_cagr_3y,
			 eps_cagr_5y, eps_growth_accel, eps_vs_5y_avg, eps_improvement_count, eps_trajectory_score, eps_stability,
			 eps_basic_fq, eps_basic_ltm, eps_basic_fy, eps_adj_ltm, eps_norm_est_fy1e, eps_growth_yoy_comp,
			 eps_cagr_3y_comp, eps_adjustment_ratio_comp, eps_positive_years, eps_trajectory_score_comp, eps_cont_ltm,
			 eps_cont_fq, eps_cont_fy, eps_cont_1fqfq, eps_cont_2fqfq, eps_cont_3fqfq, eps_cont_4fqfq, eps_cont_1fy,
			 eps_cont_2fy, eps_cont_3fy, eps_cont_4fy, eps_cont_qoq_growth, eps_cont_yoy_growth, eps_cont_cagr_3y,
			 eps_cont_vs_total_eps, eps_cont_positive_streak, eps_cont_trajectory_score, discontinued_ops_impact,
			 core_earnings_stability, eps_adjustment_spread_ltm, eps_adjustment_spread_fy, eps_adjustment_pct,
			 net_income_adjustment_ratio_ltm, net_income_adjustment_ratio_fy, net_income_adjustment_pct,
			 ebitda_adjustment_pct_ltm, ebitda_adjustment_pct_fy, ebit_adjustment_pct_ltm, ebit_adjustment_pct_fy,
			 earnings_quality_score, earnings_quality_warning, forward_eps_gaap_adj_spread, gaap_revision_momentum,
			 gaap_revision_1m, gaap_revision_3m, gaap_revision_6m, gaap_revision_1y, gaap_vs_norm_revision_spread,
			 gaap_revision_acceleration, gaap_positive_revision_flag, revision_quality_divergence)
AS
-- missing source code
;

COMMENT ON VIEW public.vw_features_earnings IS 'Earnings metrics including EPS analysis, GAAP adjustments, and revision trends.
    Source functions: calc_earnings_features, calc_eps_trajectory_features, calc_eps_comprehensive,
    calc_eps_continuing_features, calc_gaap_adjusted_analytics, calc_gaap_revision_features';

ALTER TABLE public.vw_features_earnings
	OWNER TO postgres;