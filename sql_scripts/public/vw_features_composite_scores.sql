create view public.vw_features_composite_scores
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, piotroski_f_score,
			 eps_trajectory_score, dilution_score, quality_momentum_score, net_income_is_fq, net_income_is_ltm,
			 net_income_is_fy, net_income_adj_ltm, normalized_ni_ltm, net_income_is_1fqfq, net_income_is_2fqfq,
			 net_income_is_3fqfq, net_income_is_4fqfq, net_income_is_1fy, net_income_is_2fy, net_income_is_3fy,
			 net_income_is_4fy, net_income_is_5yavgfq, net_income_is_5yavgltm, normalized_ni_5yavgfq,
			 normalized_ni_5yavgltm, net_income_growth_yoy, net_income_margin_ltm, ni_adjustment_ratio,
			 net_income_positive_years, earnings_quality_composite, net_income_qoq_growth, net_income_yoy_quarterly,
			 net_income_vs_5y_avg, normalized_ni_vs_5y_avg)
as
-- missing source code
;

comment on view public.vw_features_composite_scores is 'Composite scoring metrics including Piotroski F-Score and earnings quality.
    Source functions: calc_composite_scores, calc_net_income_comprehensive'
;

alter table public.vw_features_composite_scores
	owner to postgres
;