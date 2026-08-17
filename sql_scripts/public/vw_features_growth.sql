create view public.vw_features_growth
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, revenue_growth_yoy, ebitda_growth_yoy,
			 operating_income_growth, fcf_growth, revenue_cagr_5y, forward_revenue_growth, revenue_vs_5y_avg,
			 revenue_est_spread, revenue_beat_potential, revenue_est_revision_trend, ebitda_est_vs_actual,
			 forward_revenue_multiple, revenue_estimate_count, revenue_guidance_gap, consensus_revenue_growth,
			 ebit_estimate_spread, forward_ebitda_margin, revenue_acceleration, estimate_confidence_score,
			 revenue_est_avg_fy1e, revenue_est_med_fy1e, revenue_est_avg_ntm, revenue_est_med_ntm,
			 revenue_avg_med_diff_pct, revenue_consensus_strength, revenue_revision_trend_rec, revenue_vs_current,
			 revenue_fq, revenue_fy, revenue_ltm, revenue_5y_avg, revenue_1fqfq, revenue_2fqfq, revenue_3fqfq,
			 revenue_4fqfq, revenue_1fy, revenue_2fy, revenue_3fy, revenue_4fy, revenue_qoq_growth, revenue_qoq_2q,
			 revenue_qoq_3q, revenue_qoq_4q, revenue_yoy_quarterly, revenue_2y_growth, revenue_3y_growth,
			 revenue_4y_growth, revenue_cagr_3y, revenue_cagr_4y, revenue_4q_trend, revenue_4q_avg,
			 revenue_fq_vs_4q_avg, revenue_growth_flag, revenue_stability_score, revenue_accelerating_flag,
			 revenue_positive_qoq_streak, revenue_5yavgfq, revenue_5yavgltm, revenue_vs_5y_avg_fq,
			 revenue_vs_5y_avg_ltm, revenue_fq_vs_avg, revenue_momentum)
as
-- missing source code
;

comment on view public.vw_features_growth is 'Growth metrics including revenue, EBITDA, FCF growth rates and forecasts.
    Source functions: calc_growth_features, calc_revenue_forecast_features,
    calc_revenue_estimate_consensus, calc_revenue_quarterly_features, calc_total_revenues_temporal'
;

alter table public.vw_features_growth
	owner to postgres
;