create view public.vw_features_technical_analysis
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, ema_slope_20d, ema_trend_consistency,
			 price_vs_ema_100d, near_52w_high_flag, near_52w_low_flag, volume_momentum_score, breakout_signal,
			 high_volume_flag, low_volume_flag, volatility_compression, volatility_term_structure)
as
-- missing source code
;

comment on view public.vw_features_technical_analysis is 'Technical analysis indicators including EMA trends, volume signals, and volatility patterns.
    Source function: calc_technical_analysis_features'
;

alter table public.vw_features_technical_analysis
	owner to postgres
;