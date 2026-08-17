create view public.vw_features_leverage_liquidity
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, debt_to_equity, debt_to_assets,
			 equity_ratio, interest_coverage, current_ratio, cash_ratio, working_capital_ratio, asset_turnover,
			 inventory_turnover, receivables_days, working_capital_turns, cash_to_assets_pct, cash_change_qoq,
			 cash_vs_5y_avg, inventory_change_yoy, inventory_vs_5y_avg, receivables_change_yoy, receivables_vs_5y_avg,
			 working_capital_vs_5y_avg, retained_earnings_vs_5y, intangibles_growth_flag, asset_quality_score,
			 balance_sheet_strength, debt_maturity_risk, wc_fq, wc_fy, wc_ltm, wc_5yavgfy, wc_1fq, wc_2fq, wc_3fq,
			 wc_4fq, wc_1fy, wc_2fy, wc_3fy, wc_4fy, wc_qoq_change, wc_yoy_change, wc_4q_trend, wc_vs_5y_avg,
			 wc_positive_quarters, wc_improving_flag, wc_volatility, debt_fq, debt_fy, debt_ltm, debt_1fq, debt_2fq,
			 debt_3fq, debt_4fq, debt_1fy, debt_2fy, debt_3fy, debt_4fy, debt_qoq_change, debt_yoy_change,
			 debt_4q_trend, debt_3y_cagr, debt_deleveraging, debt_to_equity_trend, wc_ltm_deep, wc_fq_deep, wc_fy_deep,
			 wc_to_revenue, wc_to_assets, wc_change_qoq_deep, wc_change_yoy_deep, days_working_capital,
			 wc_efficiency_score, negative_wc_flag, wc_improvement_flag_deep)
as
-- missing source code
;

comment on view public.vw_features_leverage_liquidity is 'Leverage and liquidity metrics including debt ratios, working capital, and balance sheet dynamics.
    Source functions: calc_leverage_features, calc_efficiency_ratios, calc_balance_sheet_dynamics,
    calc_working_capital_temporal, calc_total_debt_temporal, calc_working_capital_deep_features'
;

alter table public.vw_features_leverage_liquidity
	owner to postgres
;