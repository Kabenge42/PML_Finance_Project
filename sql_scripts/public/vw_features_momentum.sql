CREATE VIEW public.vw_features_momentum
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, price_momentum_1m, price_momentum_3m,
			 price_momentum_6m, price_momentum_1y, price_momentum_5d, ema_crossover_20_50, ema_crossover_50_250,
			 price_vs_ema_20d, price_vs_ema_250d, pct_off_52w_high, pct_above_52w_low, range_52w_position,
			 beta_momentum, volatility_regime, price_momentum_1y_long, price_momentum_3y, price_momentum_5y,
			 long_term_trend_score, price_vs_ema_250d_long, multi_year_high_flag, secular_trend_flag)
AS
-- missing source code
;

COMMENT ON VIEW public.vw_features_momentum IS 'Price momentum and trend indicators across multiple timeframes.
    Source functions: calc_momentum_features, calc_long_term_momentum_features';

ALTER TABLE public.vw_features_momentum
	OWNER TO postgres;