create view public.vw_features_employment
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, revenue_per_employee,
			 profit_per_employee, ebitda_per_employee, assets_per_employee, fte_growth_1y_pct, fte_growth_3y_pct,
			 workforce_stability, fte_growth_2y_pct, fte_acceleration, workforce_volatility, hiring_intensity,
			 productivity_trend, headcount_vs_revenue, workforce_efficiency_gain, layoff_risk_flag, rapid_hiring_flag,
			 sustainable_growth_flag)
as
-- missing source code
;

comment on view public.vw_features_employment is 'Employment metrics including productivity, workforce trends, and efficiency.
    Source functions: calc_employment_features, calc_employment_dynamics'
;

alter table public.vw_features_employment
	owner to postgres
;