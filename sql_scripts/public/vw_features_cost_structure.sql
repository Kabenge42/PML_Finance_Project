create view public.vw_features_cost_structure
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, cogs_to_revenue, opex_to_revenue,
			 sga_to_revenue, rnd_to_revenue, interest_to_revenue, sga_trend_yoy, operating_leverage_proxy,
			 cost_efficiency_score, marketing_to_revenue, marketing_trend_yoy, marketing_vs_5y_avg, sga_vs_5y_avg,
			 sga_efficiency_trend, rnd_ltm, rnd_fq, rnd_fy, rnd_1fqfq, rnd_2fqfq, rnd_3fqfq, rnd_4fqfq, rnd_1fy,
			 rnd_2fy, rnd_3fy, rnd_4fy, rnd_intensity_ltm, rnd_intensity_fy, rnd_intensity_trend, rnd_qoq_growth,
			 rnd_yoy_growth, rnd_cagr_3y, rnd_per_employee, rnd_to_gross_profit, rnd_roi_proxy, rnd_increasing_flag,
			 rnd_cut_flag, high_rnd_intensity_flag, interest_income_ltm, interest_expense_ltm, net_interest_income,
			 interest_coverage_ratio, interest_income_to_revenue, interest_expense_to_revenue,
			 net_interest_margin_proxy)
as
-- missing source code
;

comment on view public.vw_features_cost_structure is 'Cost structure metrics including SG&A, R&D intensity, and interest analysis.
    Source functions: calc_cost_structure_features, calc_rnd_temporal_features, calc_interest_income_features'
;

alter table public.vw_features_cost_structure
	owner to postgres
;